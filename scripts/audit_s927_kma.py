"""Independently verify partial-observation S927 scores from original labels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from run_s927_kma import setup, sha, save


def audit(root, source):
    import numpy as np
    import pandas as pd
    from oibc import config as C
    start = time.monotonic()
    out = root / "runs/comparison"
    info = json.loads((out / f"{source}_F3.json").read_text())
    pred_path = out / f"{source}_F3.parquet"
    assert sha(pred_path) == info["prediction_sha256"]
    comp = pd.read_csv(out / "composition_F3.csv", dtype={"site_id": str})
    assert sha(out / "composition_F3.csv") == info["composition_sha256"]
    sites = pd.read_csv(C.data_dir() / "sites.csv", dtype={"site_id": str})
    raw_path = C.data_dir() / "train/generation.parquet"
    raw = pd.read_parquet(raw_path, columns=["site_id", "ts", "energy"])
    raw["site_id"] = raw["site_id"].astype(str)
    raw["ts"] = pd.to_datetime(raw["ts"]).dt.tz_convert("Asia/Seoul")
    raw = raw.drop_duplicates(["site_id", "ts"], keep="last")
    fold = C.FOLDS["F3"]
    raw = raw[raw.ts.between(fold.val_start, fold.val_end)].copy()
    cap = raw.site_id.map(sites.set_index("site_id").capacity_kw.astype(np.float32)).to_numpy(float)
    energy = raw.energy.to_numpy(float)
    has = np.isfinite(energy) & (energy >= 0) & (energy <= 1.05 * cap)
    raw = raw[has].copy()
    raw["actual"] = energy[has].astype(np.float32).astype(float)
    # Compare masks and labels against a complete model output before aggregating.
    member = "F153" if source == "baseline" else f"F153_{source}"
    oof_path = C.outputs_dir() / f"{member}_oof_F3.parquet"
    oof = pd.read_parquet(oof_path, columns=["site_id", "ts", "actual", "gen_missing"])
    assert not oof.duplicated(["site_id", "ts"]).any()
    grid = pd.MultiIndex.from_frame(oof[["site_id", "ts"]])
    original = raw.set_index(["site_id", "ts"]).actual.reindex(grid)
    expected_has = original.notna().to_numpy()
    model_has = (oof.gen_missing.to_numpy() == 0) & oof.actual.notna().to_numpy()
    np.testing.assert_array_equal(expected_has, model_has)
    np.testing.assert_array_equal(original.to_numpy()[expected_has], oof.actual.to_numpy()[expected_has])
    raw = raw.merge(comp, on="site_id", how="left", validate="many_to_one")
    assert raw.group.notna().all()
    axis = pd.date_range(fold.val_start, fold.val_end, freq="h")
    prices_path = C.data_dir() / "train/smp.csv"
    prices = pd.read_csv(prices_path)
    prices["ts"] = pd.to_datetime(prices["ts"]).dt.tz_convert("Asia/Seoul")
    prices = prices.drop_duplicates("ts", keep="last").set_index("ts")
    weights = prices.smp.reindex(axis).to_numpy(float) * np.where(np.isin(axis.hour, [11, 12, 13, 14, 15]), 3., 1.)
    assert np.isfinite(weights).all() and (weights >= 0).all()
    pred = pd.read_parquet(pred_path)
    modes = {}
    for mode in ["seen", "all"]:
        rows = raw if mode == "all" else raw[raw.seen]
        gi = rows.group.map({"G1": 0, "G2": 1}).to_numpy(int)
        ti = axis.get_indexer(rows.ts)
        assert (ti >= 0).all()
        actual = np.zeros((2, len(axis)))
        np.add.at(actual, (gi, ti), rows.actual.to_numpy())
        scored = pred[pred["mode"] == mode].set_index(["group", "ts"])
        expected = pd.MultiIndex.from_product([["G1", "G2"], axis], names=["group", "ts"])
        assert scored.index.equals(expected)
        np.testing.assert_allclose(actual.ravel(), scored.actual.to_numpy(), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.tile(weights, 2), scored.weight.to_numpy())
        score = math.fsum((abs(scored.pred.to_numpy() - actual.ravel()) * np.tile(weights, 2)).tolist())
        assert abs(score - info["scores"][mode]) < 1e-3
        modes[mode] = {"W": score, "observed_site_hours": len(rows)}
    result = {"status": "verified", "source": source, "scores": modes,
              "prediction_sha256": sha(pred_path), "generation_sha256": sha(raw_path),
              "prices_sha256": sha(prices_path), "oof_sha256": sha(oof_path),
              "auditor_sha256": sha(__file__), "elapsed_seconds": time.monotonic() - start,
              "caveat": "Partial observed historical labels only, not official full-hour score"}
    save(out / f"{source}_independent_audit.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source", choices=["baseline", "um", "kim"], required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    setup(root)
    audit(root, args.source)


if __name__ == "__main__":
    main()
