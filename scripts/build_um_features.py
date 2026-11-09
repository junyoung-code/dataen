"""Audit UM core-hour responses and fit fold-specific PCA using past data only.

This uses official forecast responses and fixed meteorological locations; no
competition labels, plant positions, or model scores are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def build(collection, points_path, reference, out, raw_only=False):
    status = json.loads((collection / "status.json").read_text())
    assert status["status"] == "collected", "Collection is not finished"
    config = json.loads((collection / "config.json").read_text())
    assert config["points_sha256"] == sha(points_path) and config["leads"] == [24, 32, 1]
    ref = json.loads(reference.read_text())
    assert ref["status"] == "verified_core_only" and ref["comparisons"] == 162
    points = json.loads(points_path.read_text())["points"]
    days = pd.date_range(config["start"], config["end"], freq="D", tz="Asia/Seoul")
    assert status["completed_days"] == status["expected_days"] == len(days)
    ts = pd.date_range(days[0] + pd.Timedelta(hours=1), periods=len(days) * 24, freq="h")
    values = {v: np.full((len(ts), 9), np.nan) for v in ["tdsw", "tcam"]}
    response_hashes = {}
    for day_index, day in enumerate(days):
        day_string = day.strftime("%Y-%m-%d")
        init = (day - pd.Timedelta(days=1)).strftime("%Y%m%d00")
        saved_day = json.loads((collection / "days" / f"{day_string}.json").read_text())
        assert saved_day["target_day"] == day_string and saved_day["points_sha256"] == sha(points_path)
        for j, point in enumerate(points):
            path = collection / "responses" / f"{day_string}_{point['label']}.txt"
            meta = json.loads(path.with_suffix(".json").read_text())
            assert meta["status"] == "verified" and sha(path) == meta["sha256"] == saved_day["response_sha256"][point["label"]]
            p = meta["params"]
            assert (p["group"], p["nwp"], p["data"], p["varn"], p["tmfc"], p["ef"], p["X"], p["Y"]) == (
                "UMKR", "N512", "U", "4200,6213", init, "24,32,1", str(point["X"]), str(point["Y"]))
            # Independent parsing with pandas, without the collector's parser.
            text = path.read_text()
            assert text.startswith("#START7777") and "#7777END" in text
            if meta["rows"]:
                frame = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                                    names=["tmfc", "tmef", "varn", "level", "value"],
                                    dtype={"tmfc": str, "tmef": str})
                assert len(frame) == meta["rows"] and (frame.tmfc == init).all()
                assert not frame.duplicated(["tmef", "varn"]).any()
                assert (frame.level == 16777215).all() and frame.varn.isin([4200, 6213]).all()
                valid = pd.DatetimeIndex(pd.to_datetime(frame.tmef, format="%Y%m%d%H", utc=True)).tz_convert("Asia/Seoul")
                assert (valid.normalize() == day).all() and np.isin(valid.hour, np.arange(9, 18)).all()
                for code, variable in [(4200, "tdsw"), (6213, "tcam")]:
                    keep = (frame.varn == code).to_numpy()
                    x = frame.value.to_numpy(float)[keep]
                    missing = ~np.isfinite(x) | (x <= -900) | (x >= 1e19)
                    x[missing] = np.nan
                    assert np.nanmin(x, initial=0) >= -1e-3
                    assert np.nanmax(x, initial=0) <= (1700 if code == 4200 else 1.001)
                    # Only roundoff is clipped. Absent forecasts stay NaN.
                    x = np.clip(x, 0, None if code == 4200 else 1)
                    values[variable][day_index * 24 + valid.hour.to_numpy()[keep] - 1, j] = x
            response_hashes[path.name] = meta["sha256"]
    core = np.isin(ts.hour, np.arange(9, 18))
    complete = np.isfinite(values["tdsw"]).all(1) & np.isfinite(values["tcam"]).all(1)
    day_complete = (complete.reshape(-1, 24).sum(1) == 9)
    available = days[day_complete].strftime("%Y-%m-%d").tolist()
    f3_available = [d for d in available if "2024-06-01" <= d <= "2025-05-31"]
    if not raw_only:
        assert len(f3_available) >= 180, "Not enough complete historical F3 support for paired comparison"
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame({"ts": ts})
    for variable, x in values.items():
        for j, point in enumerate(points):
            raw[f"um_{variable}_{point['label']}"] = x[:, j].astype(np.float32)
    raw.to_parquet(out / "um_core_hourly.parquet", index=False)
    fold_info = {}
    for fold, end in ([] if raw_only else [("F2", "2023-05-31 17:00"), ("F3", "2024-05-31 17:00")]):
        cutoff = pd.Timestamp(end, tz="Asia/Seoul")
        table = pd.DataFrame({"ts": ts})
        fits, arrays = {}, {}
        for variable, x in values.items():
            valid = np.isfinite(x).all(1)
            fit = valid & core & (ts <= cutoff)
            assert fit.sum() >= 900
            xf = x[fit]
            mean, sd = xf.mean(0), xf.std(0)
            sd = np.where(sd < 1e-6, 1, sd)
            _, s, vt = np.linalg.svd((xf - mean) / sd, full_matrices=False)
            loadings = vt[:5].copy()
            signs = np.sign(loadings[np.arange(5), np.abs(loadings).argmax(1)])
            loadings *= np.where(signs == 0, 1, signs)[:, None]
            transformed = np.full((len(ts), 5), np.nan)
            transformed[valid] = ((x[valid] - mean) / sd) @ loadings.T
            for k in range(5):
                table[f"um_pca_{variable}_{k + 1}"] = transformed[:, k].astype(np.float32)
            fits[variable] = {"fit_rows": int(fit.sum()), "fit_start": str(ts[fit].min()),
                              "fit_end": str(ts[fit].max()), "cutoff": str(cutoff),
                              "explained_fraction": (s[:5]**2 / np.sum(s**2)).tolist()}
            arrays.update({f"{variable}_mean": mean, f"{variable}_sd": sd, f"{variable}_loadings": loadings})
        assert table.loc[~core, table.columns != "ts"].isna().all().all()
        path = out / f"um_{fold}_pca.parquet"
        table.to_parquet(path, index=False)
        np.savez(out / f"um_{fold}_pca_fit.npz", **arrays)
        fold_info[fold] = {"fit": fits, "feature_sha256": sha(path)}
    response_manifest = out / "um_response_manifest.json"
    save(response_manifest, response_hashes)
    meta = {"status": "verified", "source": "um", "start": config["start"], "end": config["end"],
            "available_target_days": available, "F3_complete_days": len(f3_available),
            "core_hours_kst": [9, 17], "expected_days": len(days), "complete_days": int(day_complete.sum()),
            "missing_days": days[~day_complete].strftime("%Y-%m-%d").tolist(),
            "point_sha256": sha(points_path), "reference_sha256": sha(reference),
            "response_manifest_sha256": sha(response_manifest), "builder_sha256": sha(__file__),
            "folds": fold_info, "raw_only": raw_only,
            "finite_core_cells": int(sum(np.isfinite(x[core]).sum() for x in values.values())),
            "expected_core_cells": int(core.sum() * 18),
            "raw_sha256": sha(out / "um_core_hourly.parquet"),
            "historical_publication_certified": False,
            "caveat": "Core-hour historical research comparison; other UM hours NaN due native/API mismatch; no zero filling"}
    save(out / "um_features.json", meta)
    ready = {"status": "verified", "source": "um", "sha256": {
        f"work/external/kma/{name}": sha(out / name)
        for name in (["um_core_hourly.parquet", "um_features.json"] if raw_only else
                     ["um_F2_pca.parquet", "um_F3_pca.parquet", "um_features.json"])}}
    save(out / "um_ready.json", ready)
    print(json.dumps({"status": "verified", "complete_days": int(day_complete.sum()), "F3_complete_days": len(f3_available),
                      "folds": fold_info}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-only", action="store_true", help="Audit and export raw18 only; no PCA or training")
    args = parser.parse_args()
    build(args.collection, args.points, args.reference, args.output_dir, raw_only=args.raw_only)


if __name__ == "__main__":
    main()
