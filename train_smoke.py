"""공식 baseline.py의 특징/태양 계산을 재사용하는 50-tree 실행기.

원본 데이터 변경, 네트워크 요청, 자동 제출을 하지 않습니다.
검증 분할은 없으며 train MAE는 일반화 성능이나 대회 점수가 아닙니다.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import resource
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import sklearn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TREES = 50
START = time.perf_counter()


def log(message):
    print(f"[{time.perf_counter() - START:7.1f}s] {message}", flush=True)


def peak_memory_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value / (1024**2 if sys.platform == "darwin" else 1024), 1)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_baseline():
    spec = importlib.util.spec_from_file_location("oibc_official_baseline", DATA / "baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_data():
    manifest = json.loads((ROOT / "data-manifest.json").read_text())
    for name, expected in manifest.items():
        path = DATA / name
        if not path.is_file() or sha256(path) != expected["sha256"]:
            raise ValueError(f"데이터 누락 또는 변경: {name}. 원본 패키지를 다시 복사하세요.")
        if path.suffix == ".parquet":
            if pq.ParquetFile(path).metadata.num_rows != expected["rows"]:
                raise ValueError(f"Parquet 행 수 불일치: {name}")
    sites = pd.read_csv(DATA / "sites.csv")
    if len(sites) != 253 or sites.site_id.nunique() != 253:
        raise ValueError("발전소 253개가 필요합니다.")
    if sites.isna().any().any() or not sites.current_group.isin(["A", "B"]).all():
        raise ValueError("발전소 메타데이터가 유효하지 않습니다.")
    capacity = sites.groupby("current_group").capacity_kw.sum()
    if set(capacity.index) != {"A", "B"} or (capacity < 20000).any():
        raise ValueError("참조 배정이 각 20MW 조건을 충족하지 않습니다.")
    axis = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(DATA / "smp.csv").ts))
    expected_axis = pd.date_range("2025-06-01 01:00", periods=8760, freq="h", tz="Asia/Seoul")
    if axis.tz is None or not axis.tz_convert("Asia/Seoul").equals(expected_axis):
        raise ValueError("평가 시간축이 8,760개 KST 시각과 다릅니다.")
    log(f"데이터 {len(manifest)}개 무결성, 발전소 253개, 평가 시간축 8,760개 확인")


def self_test():
    # 설치된 CPU/OpenMP 라이브러리와 모델 저장/재로딩을 작게 검사합니다.
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(256, 4)).astype("float32"),
                     columns=["x0", "x1", "x2", "x3"])
    y = x.x0 * 2 - x.x1
    model = lgb.LGBMRegressor(n_estimators=5, num_leaves=7,
                             min_child_samples=5, n_jobs=2, verbosity=-1).fit(x, y)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.txt"
        model.booster_.save_model(str(path))
        restored = lgb.Booster(model_file=str(path))
        np.testing.assert_allclose(model.predict(x), restored.predict(x), rtol=1e-6)
    log(f"CPU 학습·저장·재로딩 확인 / Python {platform.python_version()} / {platform.machine()}")


def validate_outputs(folder, sites, index):
    comp = pd.read_csv(folder / "composition.csv")
    pred = pd.read_csv(folder / "forecast.csv")
    if comp.columns.tolist() != ["site_id", "group"] or len(comp) != len(sites):
        raise ValueError("구성 파일 형식/행 수 오류")
    if comp.site_id.duplicated().any() or set(comp.site_id) != set(sites.site_id):
        raise ValueError("구성 파일 발전소 누락/중복")
    if set(comp.group) != {"G1", "G2"}:
        raise ValueError("집합 이름 오류")
    capacity = comp.merge(sites, on="site_id").groupby("group").capacity_kw.sum()
    if (capacity < 20000).any():
        raise ValueError("집합 용량 20MW 미달")
    if pred.columns.tolist() != ["group", "ts", "forecast_kwh"] or len(pred) != 2 * len(index):
        raise ValueError("예측 파일 형식/행 수 오류")
    pred.ts = pd.to_datetime(pred.ts)
    if pred.ts.dt.tz is None or set(pred.group) != {"G1", "G2"}:
        raise ValueError("예측 파일 시간대/집합 오류")
    if pred.duplicated(["group", "ts"]).any():
        raise ValueError("예측 시각 중복")
    for group in ("G1", "G2"):
        axis = pd.DatetimeIndex(pred.loc[pred.group == group, "ts"]).sort_values()
        if not axis.tz_convert("Asia/Seoul").equals(index.tz_convert("Asia/Seoul").sort_values()):
            raise ValueError(f"{group} 예측 시간축 불일치")
    if not np.isfinite(pred.forecast_kwh).all() or (pred.forecast_kwh < 0).any():
        raise ValueError("음수/비유한 예측값")
    return {"composition_rows": len(comp), "forecast_rows": len(pred),
            "capacity_kw": capacity.to_dict(), "file_checks_passed": True}


def run(folder, data_dir=DATA, baseline=None):
    b = baseline or load_baseline()
    started = time.perf_counter()
    sites = pd.read_csv(data_dir / "sites.csv")[["site_id", "current_group", "capacity_kw",
                                               "lat", "lon", "wa_station", "wb_station"]]
    gen = pd.read_parquet(data_dir / "train/generation.parquet")
    gen.ts = pd.to_datetime(gen.ts)
    gen = gen[gen.ts >= b.TRAIN_FROM]
    fa = pd.read_parquet(data_dir / "train/forecast_wa.parquet", columns=["wa_station", "round", "ts"] + b.WA_COLS)
    fb = pd.read_parquet(data_dir / "train/forecast_wb.parquet", columns=["wb_station", "round", "ts"] + b.WB_COLS)
    for frame in (fa, fb):
        frame.ts = pd.to_datetime(frame.ts)
    fa = fa[fa.ts >= b.TRAIN_FROM]
    fb = fb[fb.ts >= b.TRAIN_FROM]
    log(f"학습 발전량 {len(gen):,}행 / 입력 {len(b.FEATS)}개 / 라운드별 {TREES}개 트리")
    models = {}
    stats = {"status": "running", "trees_requested_per_model": TREES,
             "train_from": str(b.TRAIN_FROM), "features": b.FEATS,
             "validation_split": False, "note": "환경 시험용. train MAE는 용량으로 정규화한 학습 오차이며 대회 점수가 아님.",
             "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__,
                          "pyarrow": pyarrow.__version__, "scikit-learn": sklearn.__version__},
             "machine": platform.machine(), "rounds": {}}
    for r in b.ROUNDS:
        stage = time.perf_counter()
        log(f"라운드 {r}: 특징 연결 시작")
        d = b.features(gen, sites, fa, fb, None, r)
        d = d[d.elev > 0]
        y = (d.energy / d.cap).clip(0, 1.2).astype("float32")
        x = d[b.FEATS].astype("float32")
        if len(x) == 0:
            raise ValueError("낮 시간 학습 데이터가 없습니다.")
        log(f"라운드 {r}: 학습 행렬 {x.shape}, 학습 시작")
        fit_start = time.perf_counter()
        def progress(env):
            if (env.iteration + 1) % 10 == 0:
                log(f"라운드 {r}: 트리 {env.iteration + 1}/{TREES}")
        model = lgb.LGBMRegressor(n_estimators=TREES, learning_rate=0.06, num_leaves=127,
                                 min_child_samples=200, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, verbose=-1, n_jobs=-1)
        model.fit(x, y, callbacks=[progress])
        fit_seconds = time.perf_counter() - fit_start
        train_mae = float(np.abs(model.predict(x) - y).mean())
        model.booster_.save_model(str(folder / f"model_round_{r}.txt"))
        models[r] = model
        stats["rounds"][str(r)] = {"training_rows": len(x), "fit_seconds": fit_seconds,
                                   "stage_seconds": time.perf_counter() - stage,
                                   "trees_built": model.booster_.num_trees(),
                                   "train_mae_normalized": train_mae}
        log(f"라운드 {r}: 학습 {fit_seconds:.1f}초, 정규화 train MAE {train_mae:.6f}")
        del d, x, y
        gc.collect()
    del gen, fa, fb
    gc.collect()
    index = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(data_dir / "smp.csv").ts)).tz_convert("Asia/Seoul")
    fa = pd.read_parquet(data_dir / "test/forecast_wa.parquet", columns=["wa_station", "round", "ts"] + b.WA_COLS)
    fb = pd.read_parquet(data_dir / "test/forecast_wb.parquet", columns=["wb_station", "round", "ts"] + b.WB_COLS)
    for frame in (fa, fb):
        frame.ts = pd.to_datetime(frame.ts)
    group = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    predictions = []
    site_sum = None
    site_keys = None
    for r in b.ROUNDS:
        log(f"라운드 {r}: 평가 기간 예측 시작")
        d = b.features(None, sites, fa, fb, index, r)
        x = d[b.FEATS].astype("float32")
        p = np.where(d.elev.to_numpy() > 0, models[r].predict(x), 0.0)
        d["kwh"] = np.clip(p, 0, None) * d.cap.to_numpy()
        d["group"] = d.site_id.map(group)
        predictions.append(d.groupby(["group", "ts"], as_index=False).kwh.sum())
        keys = d[["site_id", "ts"]].reset_index(drop=True)
        if site_sum is None:
            site_keys = keys
            site_sum = d.kwh.to_numpy(copy=True)
        else:
            if not keys.equals(site_keys):
                raise ValueError("라운드 간 발전소/시간 키 불일치")
            site_sum += d.kwh.to_numpy()
        del d, x, p
        gc.collect()
    forecast = (pd.concat(predictions).groupby(["group", "ts"], as_index=False).kwh.mean()
                .rename(columns={"kwh": "forecast_kwh"}).sort_values(["group", "ts"]))
    forecast.to_csv(folder / "forecast.csv", index=False)
    group.rename("group").reset_index().to_csv(folder / "composition.csv", index=False)
    site_keys["forecast_kwh"] = site_sum / len(b.ROUNDS)
    site_keys.to_parquet(folder / "site_forecasts.parquet", index=False, compression="zstd")
    stats["output_checks"] = validate_outputs(folder, sites, index)
    stats.update(status="complete", total_seconds=time.perf_counter() - started,
                 peak_process_memory_mib=peak_memory_mib(),
                 finished_at=datetime.now(timezone.utc).isoformat())
    (folder / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"완료: {folder} / 소요 {stats['total_seconds']:.1f}초 / 프로세스 최대 메모리 {stats['peak_process_memory_mib']:.1f} MiB")


def main():
    parser = argparse.ArgumentParser(description="OIBC LightGBM 50-tree 시험 학습")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.check_only:
        check_data()
        return
    if args.output_dir is None:
        parser.error("run_smoke.sh 또는 --output-dir 새폴더 로 실행하세요.")
    folder = args.output_dir.resolve()
    if folder == DATA or DATA in folder.parents:
        raise ValueError("출력 폴더는 data 밖에 있어야 합니다.")
    folder.mkdir(parents=True, exist_ok=True)
    if any(p.name != "run.log" for p in folder.iterdir()):
        raise ValueError("결과 덮어쓰기를 방지하기 위해 새 출력 폴더를 사용하세요.")
    try:
        run(folder)
    except BaseException as error:
        (folder / "summary.json").write_text(json.dumps({"status": "failed", "error": str(error),
            "peak_process_memory_mib": peak_memory_mib()}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
