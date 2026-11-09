"""Detached, fixed S929 control/raw18/PCA10 experiment; no parameter search.

Uses the supplied S929 predictions for unchanged members and the peer's
retrospective F2:F3 / F3:F2 calibration. This is not an independent final test.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ARMS = ("control", "raw18", "pca10")
FOLDS = ("F2", "F3")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")
    tmp.replace(path)


def setup(root):
    env = json.loads((root / "runtime_paths.json").read_text())
    if os.environ.get("S929_RUNTIME_READY") != str(root):
        os.execve(sys.executable, [sys.executable, *sys.argv],
                  {**os.environ, **env, "S929_RUNTIME_READY": str(root)})
    sys.path[:0] = env["PYTHONPATH"].split(os.pathsep)
    import pandas as pd
    from oibc import config as C
    for f, d in list(C.FOLDS.items()):
        C.FOLDS[f] = C.Fold(f, d.train_end - pd.Timedelta(hours=7), d.val_start, d.val_end)


def feature_table(root, arm, fold):
    import pandas as pd
    filename = "um_core_hourly.parquet" if arm == "raw18" else f"um_{fold}_pca.parquet"
    path = root / "work/external/kma" / filename
    frame = pd.read_parquet(path)
    assert not frame.ts.duplicated().any()
    assert len(frame.columns) == (19 if arm == "raw18" else 11)
    assert str(frame.ts.dt.tz) == "Asia/Seoul"
    return path, frame


def train(root, member, arm, fold):
    import numpy as np
    from oibc import config as C, fcst, fcst_pipe as P, submit, gfs
    import pandas as pd
    label = f"{member}_{arm}_{fold}"
    folder = root / "runs" / label
    target = C.outputs_dir() / f"{member}_{arm}_oof_{fold}.parquet"
    if (folder / "complete.json").exists():
        old = json.loads((folder / "complete.json").read_text())
        assert old["runner_sha256"] == sha(__file__)
        assert old["prediction_sha256"] == sha(target)
        return
    params = json.loads((root / "peer_source/reproduce/params_S929.json").read_text())
    cfg = {**P.base_cfg(), **params[member]["params"], "folds": fold, "boot_base": ""}
    GP = submit.load_experiment("F153_grid_pca")
    GP.GP_END_FINAL = C.FOLDS[fold].train_end + pd.Timedelta(hours=1)
    GP.install()
    inputs = {}
    if arm != "control":
        path, table = feature_table(root, arm, fold)
        inputs["um"] = sha(path)
        cols = list(table.columns[1:])
        original_build = P.build

        def add_um(df, config, stats=None, st=None):
            existing = original_build(df, config, stats, st)
            assert not set(existing) & set(cols)
            return existing + gfs.attach(df, table, cols)

        P.build = add_um
    tr, va = fcst.load_fold(fold, P.need_cols(cfg), years=None,
                           extra_train=P.extra_train(cfg), ver=cfg["feat_ver"])
    tr = tr.loc[tr.ts <= C.FOLDS[fold].train_end].reset_index(drop=True)
    assert cfg["years"] == 0 and cfg["seed"] == 0 and cfg["n_seeds"] == 1
    assert tr.ts.max() <= C.FOLDS[fold].train_end < va.ts.min()
    for key, path in {
        "train": C.cache_dir() / f"feat_v1_{fold}_train.parquet",
        "val": C.cache_dir() / f"feat_v1_{fold}_val.parquet",
        "gfs": root / "work/external/gfs/gfs_hourly_v1.parquet",
        "grid": root / "work/external/gfs/gfs_grid_v1.parquet",
    }.items():
        inputs[key] = sha(path)
    started = time.monotonic()
    save(folder / "status.json", {"status": "training", "pid": os.getpid(), "member": label})
    result = P.fit_predict(tr, va, cfg)
    pred = np.asarray(result["pred"], dtype=float)
    assert pred.shape == (len(va),) and np.isfinite(pred).all() and (pred >= 0).all()
    out = va[["site_id", "ts", "capacity_kw", "energy", "gen_missing", "seen"]].rename(columns={"energy": "actual"}).copy()
    assert not out.duplicated(["site_id", "ts"]).any()
    out["pred"] = pred
    tmp = target.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(target)
    info = {"status": "complete", "member": label, "cfg": cfg,
            "input_sha256": inputs, "prediction_sha256": sha(target),
            "runner_sha256": sha(__file__), "iterations": result["info"],
            "train_start": str(tr.ts.min()), "train_end": str(tr.ts.max()),
            "train_rows": len(tr), "validation_start": str(va.ts.min()),
            "validation_end": str(va.ts.max()), "validation_rows": len(va),
            "elapsed_seconds": time.monotonic() - started, "gfs_pca": GP.summary()}
    save(folder / "complete.json", info)
    save(folder / "status.json", {"status": "complete"})
    print(json.dumps({"member_complete": label, "seconds": info["elapsed_seconds"]}), flush=True)


def check_um(root):
    import numpy as np
    import pandas as pd
    from oibc import config as C
    base = root / "work/external/kma"
    raw = pd.read_parquet(base / "um_core_hourly.parquet")
    assert not raw.ts.duplicated().any() and raw.shape[1] == 19
    core = raw.ts.dt.hour.between(9, 17).to_numpy()
    assert raw.loc[~core, raw.columns != "ts"].isna().all().all()
    meta = json.loads((base / "um_features.json").read_text())
    assert meta["status"] == "verified"
    results = {}
    for f in FOLDS:
        path, pca = feature_table(root, "pca10", f)
        assert sha(path) == meta["folds"][f]["feature_sha256"]
        assert raw.ts.equals(pca.ts)
        fit = np.load(base / f"um_{f}_pca_fit.npz")
        for var in ("tdsw", "tcam"):
            x = raw[[f"um_{var}_p{i}" for i in range(1, 10)]].to_numpy(float)
            valid = np.isfinite(x).all(1)
            mask = valid & core & (raw.ts <= C.FOLDS[f].train_end).to_numpy()
            info = meta["folds"][f]["fit"][var]
            assert pd.Timestamp(info["fit_end"]) <= C.FOLDS[f].train_end
            assert info["fit_rows"] == int(mask.sum())
            mean, sd, loading = (fit[f"{var}_{k}"] for k in ("mean", "sd", "loadings"))
            np.testing.assert_allclose(x[mask].mean(0), mean, rtol=1e-6, atol=1e-5)
            np.testing.assert_allclose(x[mask].std(0), sd, rtol=1e-6, atol=1e-5)
            y = pca[[f"um_pca_{var}_{i}" for i in range(1, 6)]].to_numpy(float)
            np.testing.assert_array_equal(np.isfinite(y).all(1), valid)
            np.testing.assert_allclose(((x[valid] - mean) / sd) @ loading.T, y[valid], rtol=1e-4, atol=1e-5)
        results[f] = {"pca_sha256": sha(path), "fit_sha256": sha(base / f"um_{f}_pca_fit.npz")}
    return {"raw_sha256": sha(base / "um_core_hourly.parquet"), "folds": results}


def controller(root):
    lock = (root / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "completion.json").exists():
        return
    started = time.time()
    try:
        save(root / "status.json", {"status": "preflight", "pid": os.getpid(), "started": started})
        um_check = check_um(root)
        save(root / "um_check.json", um_check)
        subprocess.run([sys.executable, str(root / "scripts/compare_s929_um.py"), "--run-root", str(root), "--reference-only"], check=True)
        for i, (arm, fold, member) in enumerate((a, f, m) for a in ARMS for f in FOLDS for m in ("F153", "F155")):
            save(root / "status.json", {"status": "training", "pid": os.getpid(), "started": started,
                 "completed_members": i, "total_members": 12, "current": f"{member}_{arm}_{fold}"})
            subprocess.run([sys.executable, __file__, "--run-root", str(root), "--member", member,
                            "--arm", arm, "--fold", fold], check=True)
        save(root / "status.json", {"status": "scoring", "pid": os.getpid(), "started": started})
        subprocess.run([sys.executable, str(root / "scripts/compare_s929_um.py"), "--run-root", str(root)], check=True)
        result = json.loads((root / "comparison/results.json").read_text())
        assert result["status"] == "verified" and len(result["scores"]) == 4
        complete = {"status": "complete", "started": started, "finished": time.time(),
                    "elapsed_seconds": time.time() - started, "results_sha256": sha(root / "comparison/results.json")}
        save(root / "completion.json", complete)
        save(root / "status.json", complete)
        print("S929_UM_COMPARISON_COMPLETE", flush=True)
    except BaseException:
        failure = {"status": "failed", "time": time.time(), "traceback": traceback.format_exc()}
        save(root / "failure.json", failure)
        save(root / "status.json", failure)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--member", choices=["F153", "F155"])
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--fold", choices=FOLDS)
    args = ap.parse_args()
    root = args.run_root.resolve()
    setup(root)
    if args.member:
        assert args.arm and args.fold
        train(root, args.member, args.arm, args.fold)
    else:
        controller(root)


if __name__ == "__main__":
    main()
