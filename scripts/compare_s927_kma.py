"""Paired historical S927-configured comparison with causal F2-to-F3 calibration."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd

from run_s927_kma import save, setup, sha


def truncate(d, cutoff):
    n = len(d["ts"])
    keep = d["ts"] <= cutoff
    out = {}
    for k, value in d.items():
        if isinstance(value, pd.DatetimeIndex): out[k] = value[keep]
        elif isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == n: out[k] = value[:, keep]
        elif isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == n: out[k] = value[keep]
        else: out[k] = value
    return out


def mixed(source, fold):
    from oibc.comp_io import load_fold
    from oibc.group_calib import VAL_EXTRA
    members = ["F153", "F114_xgb", "F155", "N502c20", "N530x20"]
    if source != "baseline":
        members[0] += "_" + source
        members[2] += "_" + source
    ds = [load_fold(m, fold, extra_val_cols=VAL_EXTRA) for m in members]
    first = dict(ds[0])
    for d in ds[1:]:
        for key in ["sites", "ts", "has", "actual", "cap", "seen", "w"]:
            assert np.array_equal(np.asarray(first[key]), np.asarray(d[key])), (source, fold, key)
    first["pred"] = sum(weight * d["pred"] for weight, d in zip(np.array([2, 1, 2, 1, 1]) / 7, ds))
    return first


def weighted_error(frame):
    return float((np.abs(frame.pred - frame.actual) * frame.weight).sum())


def uncertainty(base, candidate):
    from oibc.config import FOLDS
    day = (base.ts - pd.Timedelta(hours=1)).dt.normalize()
    daily = pd.DataFrame({"day": day,
                         "base": np.abs(base.pred - base.actual) * base.weight,
                         "new": np.abs(candidate.pred - candidate.actual) * candidate.weight}).groupby("day").sum()
    # Missing source days contribute no scored error, but retain calendar gaps.
    calendar = pd.date_range((FOLDS["F3"].val_start - pd.Timedelta(hours=1)).normalize(),
                             (FOLDS["F3"].val_end - pd.Timedelta(hours=1)).normalize(), freq="D")
    values = daily.reindex(calendar, fill_value=0).to_numpy()
    n = len(values)
    rng = np.random.default_rng(20261108)
    starts = rng.integers(n, size=(2000, (n + 6) // 7))
    indices = ((starts[:, :, None] + np.arange(7)) % n).reshape(2000, -1)[:, :n]
    totals = values[indices].sum(axis=1)
    change = 100 * (totals[:, 1] / totals[:, 0] - 1)
    return {"block_days": 7, "replicates": 2000, "CI90": np.quantile(change, [.05, .95]).tolist(),
            "fraction_better": float((change < 0).mean()), "diagnostic_only": True}


def run(root, source):
    from oibc import config as C, features as F, submit
    from oibc.group_calib import group_rows, strat_comp
    G = submit.load_experiment("G303_group_direct")
    out = root / "runs/comparison"
    out.mkdir(exist_ok=True)
    dest = out / f"{source}_F3.json"
    from oibc.comp_io import oof_path
    names = ["F153", "F114_xgb", "F155", "N502c20", "N530x20"]
    if source != "baseline":
        names[0] += "_" + source
        names[2] += "_" + source
    inputs = {f"{name}.{fold}": sha(oof_path(name, fold))
              for name in names for fold in ["F2", "F3"]}
    inputs["contract"] = sha(root / "frozen_contract.json")
    inputs["runner"] = sha(__file__)
    inputs["peer_parameters"] = sha(root / "peer_source/reproduce/params_S927.json")
    if source != "baseline":
        inputs["baseline"] = sha(out / "baseline_F3.json")
        inputs["availability"] = sha(root / f"work/external/kma/{source}_features.json")
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if dest.exists():
        info = json.loads(dest.read_text())
        assert info["signature"] == signature, "Inputs changed; preserve this run and use a new output folder"
        assert sha(out / f"{source}_F3.parquet") == info["prediction_sha256"]
        print(json.dumps({"status": "complete_result_reused", "source": source}), flush=True)
        return
    start = time.monotonic()
    d = mixed(source, "F3")
    past = truncate(mixed(source, "F2"), C.FOLDS["F3"].train_end)
    assert past["ts"].max() <= C.FOLDS["F3"].train_end < d["ts"].min()
    g2 = strat_comp(d["cap"], d["wa_st"], seed=6)
    assert [int((g2 == g).sum()) for g in [0, 1]] == [129, 124]
    composition = out / "composition_F3.csv"
    pd.DataFrame({"site_id": d["sites"], "group": np.where(g2, "G2", "G1"),
                  "seen": d["seen"]}).to_csv(composition, index=False)
    te = group_rows(d, g2)
    factors = G.calib_factors(past, te, g2, .5)
    factor_matrix = np.ones((2, len(d["ts"])))
    factor_matrix[te.group.to_numpy(int), te.t_idx.to_numpy(int)] = factors
    direct_file = out / "direct_F3.npy"
    if source == "baseline":
        per, common = G.site_cols("F3", "full")
        deriv = [f"{x}_r17" for x in F.E000_VARS] + ["kc_cf_wa_r17", "kc_cf_wb_r17"]
        per += [c for c in deriv if c not in per]
        acc = []
        for seed in [0, 1, 2]:
            cfg = argparse.Namespace(lr=.03, n_estimators=800, num_leaves=15, min_child_samples=100,
                                     v2=1, deriv=deriv, gfs="kr+gefs+x+kt", seed=seed)
            model, nrows = G.fit_direct("F3", pd.Series(g2, index=d["sites"]), per, common, 2, cfg)
            acc.append(G.val_direct_cf("F3", model, d, g2, per, common))
            model.booster_.save_model(str(out / f"direct_seed{seed}.txt"))
        direct_cf = np.mean(acc, axis=0)
        np.save(direct_file, direct_cf)
    else:
        baseline = json.loads((out / "baseline_F3.json").read_text())
        assert sha(direct_file) == baseline["direct_sha256"]
        direct_cf = np.load(direct_file)
    frames = []
    for mode in ["seen", "all"]:
        mask = d["has"] if mode == "all" else d["has"] & d["seen"][:, None]
        actual = np.vstack([(d["actual"] * mask)[g2 == g].sum(0) for g in [0, 1]])
        cap = np.vstack([(d["cap"][:, None] * mask)[g2 == g].sum(0) for g in [0, 1]])
        plant = np.vstack([(d["pred"] * mask)[g2 == g].sum(0) for g in [0, 1]]) * factor_matrix
        pred = .85 * plant + .15 * np.where(plant > 0, direct_cf * cap, 0)
        index = pd.MultiIndex.from_product([["G1", "G2"], d["ts"]], names=["group", "ts"])
        frame = pd.DataFrame({"pred": pred.ravel(), "actual": actual.ravel(), "weight": np.tile(d["w"], 2),
                              "plant": plant.ravel(), "direct": (direct_cf * cap).ravel()}, index=index).reset_index()
        frame["mode"] = mode
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    assert np.isfinite(frame.select_dtypes("number")).all().all()
    prediction_path = out / f"{source}_F3.parquet"
    frame.to_parquet(prediction_path, index=False)
    info = {"status": "complete", "source": source, "fold": "F3", "mix_weights": [2, 1, 2, 1, 1],
            "signature": signature, "input_sha256": inputs,
            "direct_weight": .15, "calibration_shrink": .5, "calibration_fold": "F2",
            "calibration_end": str(past["ts"].max()), "train_cutoff": str(C.FOLDS["F3"].train_end),
            "prediction_sha256": sha(prediction_path), "direct_sha256": sha(direct_file),
            "composition_sha256": sha(composition),
            "elapsed_seconds": time.monotonic() - start,
            "scores": {m: weighted_error(frame[frame['mode'] == m]) for m in ["seen", "all"]},
            "caveat": "Historical F3, S927 configuration with causal correction; not official server score or independent final test"}
    if source != "baseline":
        base = pd.read_parquet(out / "baseline_F3.parquet")
        availability = json.loads((root / f"work/external/kma/{source}_features.json").read_text())["available_target_days"]
        day = (frame.ts - pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%d")
        results = {}
        for mode in ["seen", "all"]:
            for span, keep in [("full_F3", np.ones(len(frame), bool)), ("source_available", day.isin(availability).to_numpy())]:
                mask = (frame["mode"] == mode).to_numpy() & keep
                b, c = base[mask].reset_index(drop=True), frame[mask].reset_index(drop=True)
                assert len(b), (mode, span, "No shared evaluation rows")
                pd.testing.assert_frame_equal(b[["group", "ts", "actual", "weight", "direct", "mode"]],
                                              c[["group", "ts", "actual", "weight", "direct", "mode"]], check_exact=True)
                bw, cw = weighted_error(b), weighted_error(c)
                monthly = []
                target_month = (c.ts - pd.Timedelta(hours=1)).dt.strftime("%Y-%m")
                for month in sorted(target_month.unique()):
                    mm = target_month == month
                    mb, mc = weighted_error(b[mm]), weighted_error(c[mm])
                    monthly.append({"month": month, "baseline_W": mb, "candidate_W": mc, "change_percent": (mc / mb - 1) * 100})
                results[f"{mode}.{span}"] = {"baseline_W": bw, "candidate_W": cw, "change_percent": (cw / bw - 1) * 100,
                                                "target_days": int(((c.ts - pd.Timedelta(hours=1)).dt.normalize()).nunique()),
                                                "uncertainty": uncertainty(b, c), "monthly": monthly}
        info["comparisons"] = results
    save(dest, info)
    print(json.dumps(info, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source", choices=["baseline", "um", "kim"], required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    setup(root)
    run(root, args.source)


if __name__ == "__main__": main()
