"""G315 causal baseline and one-source-at-a-time fixed F3 ablations on Mac mini.

Requires the isolated lane's run_peer_reproduction.py and verified BL2 caches.
Does not regenerate plant models or write through their shared output symlink.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from weather_ablation_features import save_json, sha


def score(frame):
    error = np.abs(frame.pred.to_numpy() - frame.actual.to_numpy()) * frame.weight.to_numpy()
    assert np.isfinite(error).all() and (frame.weight >= 0).all()
    return float(error.sum())


def block_uncertainty(base, candidate, mask):
    """Paired seven-calendar-day circular bootstrap; missing dates have zero weight."""
    day = (base.ts - pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%d")
    daily = pd.DataFrame({"day": day,
             "baseline": np.where(mask, np.abs(base.pred - base.actual) * base.weight, 0),
             "candidate": np.where(mask, np.abs(candidate.pred - candidate.actual) * candidate.weight, 0)})
    a = daily.groupby("day", sort=True)[["baseline", "candidate"]].sum().to_numpy()
    n = len(a); rng = np.random.default_rng(20261107)
    starts = rng.integers(n, size=(2000, (n + 6) // 7))
    indices = ((starts[:, :, None] + np.arange(7)) % n).reshape(2000, -1)[:, :n]
    totals = a[indices].sum(axis=1)
    assert (totals[:, 0] > 0).all()
    delta = (totals[:, 1] / totals[:, 0] - 1) * 100
    return {"block_days": 7, "resamples": 2000, "seed": 20261107,
            "change_percent_CI90": np.quantile(delta, [.05, .95]).tolist(),
            "fraction_better": float((delta < 0).mean()),
            "interpretation": "Historical paired resampling uncertainty, not a probability of future superiority"}


def run(source, fold):
    os.environ["OIBC_WORK_DIR"] = str(ROOT / "outputs/runs/work")
    os.environ["OIBC_DATA_DIR"] = str(ROOT / "data")
    os.environ.setdefault("OIBC_N_JOBS", "4")
    spec = importlib.util.spec_from_file_location("peer_reproduction", ROOT / "run_peer_reproduction.py")
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    G = R.submit.load_experiment("G303_group_direct")
    from oibc import gfs
    out = ROOT / "outputs/runs/ablation"; out.mkdir(parents=True, exist_ok=True)
    label = f"{source}_{fold}"
    assert not (out / f"{label}.json").exists(), "Completed run exists; review before replacing"
    input_hashes = {name: sha(gfs.gfs_dir() / name) for name in
                    ["gfs_hourly_v1.parquet", "gfs_hourly_v2.parquet", "gfs_gm_feat.parquet"]}
    # Audit all 47 G315 weather columns at evaluation hours before learning.
    eval_times = pd.date_range(R.C.FOLDS[fold].val_start, R.C.FOLDS[fold].val_end, freq="h")
    weather_probe = pd.DataFrame({"ts": eval_times})
    gcols = G.gfs_attach_group(weather_probe, "kr+gefs+x+kt+gm")
    assert len(gcols) == 47 and len(set(gcols)) == 47
    sun = gfs.hourly_solar_features(eval_times, *gfs.CENTER)["cs_ghi_wm2"].to_numpy() >= gfs.KT_CS_MIN
    weather_audit = {"columns": len(gcols), "sunlit_hours": int(sun.sum()),
                     "sunlit_missing_cells": int(weather_probe.loc[sun, gcols].isna().sum().sum()),
                     "nonfinite_nonmissing_cells": int(np.isinf(weather_probe[gcols].to_numpy()).sum())}
    assert weather_audit["sunlit_missing_cells"] == 0, weather_audit
    assert weather_audit["nonfinite_nonmissing_cells"] == 0
    feature_info = None; attachment_audit = []
    if source != "baseline":
        fp = ROOT / "outputs/runs/features" / f"{source}.parquet"
        feature_info = json.loads(fp.with_suffix(".json").read_text())
        assert sha(fp) == feature_info["sha256"]
        assert feature_info["training_days_before_F3"] >= 30 and feature_info["F3_days"] >= 180
        table = pd.read_parquet(fp); columns = feature_info["features"]
        original = G.gfs_attach_group
        def attach(gf, mode):
            cols = original(gf, mode)
            added = gfs.attach(gf, table, columns)
            attachment_audit.append({"rows": len(gf), "min_ts": str(gf.ts.min()), "max_ts": str(gf.ts.max()),
                                     "complete_rows": int(gf[added].notna().all(axis=1).sum())})
            return cols + added
        G.gfs_attach_group = attach
        input_hashes[source] = feature_info["sha256"]
    print(json.dumps({"status": "training", "source": source, "fold": fold}), flush=True)
    d = R.mixed(fold)
    previous = {"F2": "F1", "F3": "F2"}[fold]
    g2 = R.strat_comp(d["cap"], d["wa_st"], seed=6)
    assert [int((g2 == g).sum()) for g in [0, 1]] == [129, 124]
    if source == "baseline":
        tr = R.truncate(R.mixed(previous), R.C.FOLDS[fold].train_end)
        calibration_end = str(tr["ts"].max())
        assert tr["ts"].max() <= R.C.FOLDS[fold].train_end < d["ts"].min()
        te = R.group_rows(d, g2); factors = G.calib_factors(tr, te, g2, .5)
        plant = np.zeros((2, len(d["ts"])))
        plant[te.group.to_numpy(int), te.t_idx.to_numpy(int)] = factors * te.P.to_numpy()
    else:
        # Freeze the calibrated plant component byte-for-byte across ablations.
        base_info = json.loads((out / f"baseline_{fold}.json").read_text())
        bp = out / f"baseline_{fold}.parquet"
        assert sha(bp) == base_info["prediction_sha256"]
        assert base_info["input_sha256"] == {k:v for k,v in input_hashes.items() if k != source}
        base = pd.read_parquet(bp).set_index(["group", "ts"])
        idx = pd.MultiIndex.from_product([["G1", "G2"], d["ts"]], names=["group", "ts"])
        plant = base.loc[idx, "plant"].to_numpy().reshape(2, -1)
        calibration_end = base_info["calibration_end"]
        assert pd.Timestamp(calibration_end) <= R.C.FOLDS[fold].train_end
    per, common = G.site_cols(fold, "full")
    deriv = [f"{x}_r17" for x in R.F.E000_VARS] + ["kc_cf_wa_r17", "kc_cf_wb_r17"]
    per += [c for c in deriv if c not in per]
    started = time.monotonic(); acc, counts, model_info = [], [], []
    for seed in [0, 1, 2]:
        cfg = argparse.Namespace(lr=.03, n_estimators=800, num_leaves=15, min_child_samples=100,
                                 v2=1, deriv=deriv, gfs="kr+gefs+x+kt+gm", seed=seed)
        model, n = G.fit_direct(fold, pd.Series(g2, index=d["sites"]), per, common, 2, cfg)
        assert len([c for c in model.feats_ if c.startswith("gfs_gm_")]) == 11
        if source != "baseline": assert set(columns).issubset(model.feats_)
        prediction = G.val_direct_cf(fold, model, d, g2, per, common)
        assert np.isfinite(prediction).all()
        acc.append(prediction); counts.append(n)
        mp = out / f"{label}_seed{seed}.txt"; model.booster_.save_model(str(mp))
        model_info.append({"seed": seed, "features": model.feats_, "sha256": sha(mp),
                           "feature_gain": dict(zip(model.feats_, map(float, model.booster_.feature_importance("gain"))))})
        print(json.dumps({"source": source, "seed_complete": seed, "train_rows": n,
                          "features": len(model.feats_), "seconds": time.monotonic() - started}), flush=True)
    capacities = np.vstack([(d["cap"][:, None] * d["has"])[g2 == g].sum(0) for g in [0, 1]])
    direct = np.mean(acc, axis=0) * capacities
    pred = .7 * plant + .3 * np.where(plant > 0, direct, 0)
    actual = np.vstack([(d["actual"] * d["has"])[g2 == g].sum(0) for g in [0, 1]])
    index = pd.MultiIndex.from_product([["G1", "G2"], d["ts"]], names=["group", "ts"])
    frame = pd.DataFrame({"pred": pred.ravel(), "actual": actual.ravel(), "weight": np.tile(d["w"], 2),
                          "plant": plant.ravel(), "direct": direct.ravel(),
                          "observed_capacity": capacities.ravel()}, index=index).reset_index()
    assert not frame[["group", "ts"]].duplicated().any()
    assert np.isfinite(frame.select_dtypes("number")).all().all()
    path = out / f"{label}.parquet"; frame.to_parquet(path, index=False)
    comp = pd.DataFrame({"site_id": d["sites"], "group": np.where(g2, "G2", "G1")})
    comp.to_csv(out / f"composition_{label}.csv", index=False)
    info = {"status": "complete", "source": source, "fold": fold, "W": score(frame),
            "seconds": time.monotonic() - started, "rows": len(frame), "train_rows": counts,
            "train_end": str(R.C.FOLDS[fold].train_end), "calibration_fold": previous,
            "calibration_end": calibration_end, "validation_start": str(d["ts"].min()),
            "validation_end": str(d["ts"].max()), "input_sha256": input_hashes,
            "prediction_sha256": sha(path), "runner_sha256": sha(__file__),
            "helper_sha256": sha(ROOT / "run_peer_reproduction.py"),
            "peer_ref": "e4c05507aa7f4392f0a8fa3bb45ba66b51b8a7bc",
            "models": model_info, "attachment_audit": attachment_audit, "weather_audit": weather_audit,
            "comparison": "historical F3; fixed G315 parameters and causal calibration; not unseen final test"}
    save_json(out / f"{label}.json", info)
    print(json.dumps({k: info[k] for k in ["status", "source", "fold", "W", "seconds"]}), flush=True)


def report(fold):
    out = ROOT / "outputs/runs/ablation"
    frames, infos, availability = {}, {}, {}
    sources = [s for s in ["baseline", "icon", "ens", "tigge"] if (out / f"{s}_{fold}.json").exists()]
    assert "baseline" in sources and len(sources) > 1
    complete = len(sources) == 4
    for source in sources:
        info = json.loads((out / f"{source}_{fold}.json").read_text())
        path = out / f"{source}_{fold}.parquet"
        assert sha(path) == info["prediction_sha256"]
        frame = pd.read_parquet(path).sort_values(["group", "ts"]).reset_index(drop=True)
        assert abs(score(frame) - info["W"]) < .001
        if source != "baseline":
            base = frames["baseline"]
            pd.testing.assert_frame_equal(frame[["group", "ts", "actual", "weight", "plant", "observed_capacity"]],
                                          base[["group", "ts", "actual", "weight", "plant", "observed_capacity"]], check_exact=True)
            assert {k:v for k,v in info["input_sha256"].items() if k != source} == infos["baseline"]["input_sha256"]
            feature = json.loads((ROOT / "outputs/runs/features" / f"{source}.json").read_text())
            assert feature["sha256"] == info["input_sha256"][source]
            availability[source] = set(feature["available_days"])
        frames[source], infos[source] = frame, info
    days = (frames["baseline"].ts - pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%d")
    common = set.intersection(*availability.values()) if complete else set()
    results = {}
    for source in [s for s in sources if s != "baseline"]:
        masks = {"full_F3": np.ones(len(days), dtype=bool), "source_available": days.isin(availability[source])}
        if complete: masks["all_sources_common"] = days.isin(common)
        comparisons = {}
        for label, mask in masks.items():
            assert np.asarray(mask).any()
            b, c = frames["baseline"][mask], frames[source][mask]
            bw, cw = score(b), score(c)
            monthly = []
            for month in sorted(set(days[mask].str[:7])):
                mm = mask & (days.str[:7] == month)
                bm, cm = score(frames["baseline"][mm]), score(frames[source][mm])
                monthly.append({"month": month, "days": int(days[mm].nunique()), "baseline_W": bm, "candidate_W": cm,
                                "change_percent": (cm / bm - 1) * 100 if bm else None})
            comparisons[label] = {"days": int(days[mask].nunique()), "baseline_W": bw, "candidate_W": cw,
                                  "change_percent": (cw / bw - 1) * 100, "monthly": monthly,
                                  "uncertainty": block_uncertainty(frames["baseline"], frames[source], mask)}
        results[source] = comparisons
    label = "comparison" if complete else "comparison_partial"
    save_json(out / f"{label}_{fold}.json", {"verified": True, "complete": complete,
              "pending_sources": [s for s in ["icon", "ens", "tigge"] if s not in sources],
              "baseline_W": infos["baseline"]["W"],
              "comparisons": results, "negative_change_is_improvement": True,
              "caveat": "Historical evaluation, missing source inputs remain NaN, provisional historical publication lags"})
    print(json.dumps({s: v["source_available"] | {"monthly": "see JSON"} for s,v in results.items()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", choices=["baseline", "icon", "ens", "tigge", "report"])
    ap.add_argument("--fold", choices=["F2", "F3"], default="F3")
    a = ap.parse_args()
    if a.source == "report": report(a.fold)
    else: run(a.source, a.fold)
