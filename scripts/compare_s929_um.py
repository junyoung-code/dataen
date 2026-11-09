"""S929 peer W, fixed composition/calibration and independent label/price audit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_s929_um import ARMS, FOLDS, save, setup, sha


def aligned_prediction(path, d):
    frame = pd.read_parquet(path, columns=["site_id", "ts", "actual", "gen_missing", "pred"])
    assert not frame.duplicated(["site_id", "ts"]).any(), f"duplicate keys: {path.name}"
    si = pd.Index(d["sites"]).get_indexer(frame.site_id)
    ti = d["ts"].get_indexer(frame.ts)
    assert (si >= 0).all() and (ti >= 0).all()
    actual, has, pred = (np.zeros_like(d["actual"]) for _ in range(3))
    has[si, ti] = frame.actual.notna() & (frame.gen_missing == 0)
    actual[si, ti] = frame.actual.fillna(0).to_numpy(float)
    values = frame.pred.to_numpy(float)
    assert np.isfinite(values).all() and (values >= 0).all()
    pred[si, ti] = values
    np.testing.assert_array_equal(has.astype(bool), d["has"])
    np.testing.assert_array_equal(actual, d["actual"])
    # New runners predict the full validation grid; peer OOF omits plant-hours
    # outside each plant's recorded span. Keep that exact support fixed for
    # both scoring AND calibration features (which also use unobserved rows).
    support = d.get("prediction_support", np.broadcast_to(d["in_span"][:, None], pred.shape))
    present = np.zeros(pred.shape, bool)
    present[si, ti] = True
    assert np.all(present[support]), f"missing peer prediction rows: {path.name}"
    assert not np.any(d["has"] & ~support)
    return np.where(support, pred, 0.)


def independent_labels(d, fold, groups):
    """Rebuild valid labels and SMP*k from originals, without comp_io/scoring."""
    from oibc import config as C
    sites = pd.read_csv(C.data_dir() / "sites.csv", dtype={"site_id": str})
    capmap = sites.set_index("site_id").capacity_kw.astype(np.float32)
    np.testing.assert_array_equal(capmap.reindex(d["sites"]).to_numpy(float), d["cap"])
    rawpath = C.data_dir() / "train/generation.parquet"
    raw = pd.read_parquet(rawpath, columns=["site_id", "ts", "energy"])
    raw["site_id"] = raw.site_id.astype(str)
    raw["ts"] = pd.to_datetime(raw.ts).dt.tz_convert("Asia/Seoul")
    raw = raw.drop_duplicates(["site_id", "ts"], keep="last")
    raw = raw[raw.ts.between(d["ts"][0], d["ts"][-1])].copy()
    energy = raw.energy.to_numpy(float)
    cap = raw.site_id.map(capmap).to_numpy(float)
    valid = np.isfinite(energy) & (energy >= 0) & (energy <= 1.05 * cap)
    raw = raw[valid].copy()
    si, ti = pd.Index(d["sites"]).get_indexer(raw.site_id), d["ts"].get_indexer(raw.ts)
    assert (si >= 0).all() and (ti >= 0).all()
    values = energy[valid].astype(np.float32).astype(float)
    has = np.zeros_like(d["has"])
    actual = np.zeros_like(d["actual"])
    has[si, ti] = True
    actual[si, ti] = values
    np.testing.assert_array_equal(has, d["has"])
    np.testing.assert_array_equal(actual[has], d["actual"][has])
    pricespath = C.data_dir() / "train/smp.csv"
    prices = pd.read_csv(pricespath)
    prices["ts"] = pd.to_datetime(prices.ts).dt.tz_convert("Asia/Seoul")
    prices = prices.drop_duplicates("ts", keep="last").set_index("ts")
    weights = prices.smp.reindex(d["ts"]).to_numpy(float) * np.where(np.isin(d["ts"].hour, [11, 12, 13, 14, 15]), 3., 1.)
    assert np.isfinite(weights).all() and (weights >= 0).all()
    np.testing.assert_array_equal(weights, d["w"])
    totals = {}
    for mode in ("seen", "all"):
        keep = d["seen"][si] if mode == "seen" else np.ones(len(si), bool)
        total = np.zeros((2, len(d["ts"])))
        np.add.at(total, (groups[si[keep]], ti[keep]), values[keep])
        totals[mode] = total
    return totals, {"status": "verified", "generation_sha256": sha(rawpath),
                    "prices_sha256": sha(pricespath), "observed_site_hours": int(has.sum())}


def direct_cf(path, d, groups):
    frame = pd.read_parquet(path)
    assert not frame.duplicated(["group", "ts"]).any()
    axis = pd.MultiIndex.from_product([["G1", "G2"], d["ts"]], names=["group", "ts"])
    indexed = frame.set_index(["group", "ts"])
    assert indexed.index.isin(axis).all()
    aligned = indexed.reindex(axis)
    # Missing time rows mean zero by the supplied direct-model README.
    cf = aligned.cf.fillna(0).to_numpy(float).reshape(2, -1)
    assert np.isfinite(cf).all() and (cf >= 0).all()
    caps = np.array([(d["cap"][:, None] * d["has"])[groups == g].sum(0) for g in (0, 1)])
    observed = aligned.cap_has.notna().to_numpy()
    np.testing.assert_allclose(aligned.cap_has.to_numpy()[observed], caps.ravel()[observed], rtol=1e-6, atol=1e-3)
    np.testing.assert_allclose(aligned.pred.to_numpy()[observed], (cf * caps).ravel()[observed], rtol=1e-6, atol=1e-3)
    return cf


def compare(root, reference_only=False):
    from oibc import config as C, submit
    from oibc.comp_io import load_fold
    from oibc.group_calib import VAL_EXTRA, strat_comp
    from oibc.scoring import W_groups
    from oibc.boot import paired_block_boot
    X9 = submit.load_experiment("X619_nn_mix")
    params = json.loads((root / "peer_source/reproduce/params_S929.json").read_text())["S929_build"]["params"]
    assert params["mix_w"] == "2,1,3,3" and params["alpha"] == .15 and params["shrink"] == .5
    comp = pd.read_csv(C.outputs_dir() / "C_strat_v1_composition.csv", dtype={"site_id": str})
    assert not comp.site_id.duplicated().any() and set(comp.group) == {"G1", "G2"}
    groupmap = comp.set_index("site_id").group.map({"G1": 0, "G2": 1})
    arms = ["supplied_S929"] if reference_only else ["supplied_S929", *ARMS]
    ds, groups, mixes, direct, actuals, audits, inputs = {}, {}, {}, {}, {}, {}, {}
    for fold in FOLDS:
        d = load_fold("F153", fold, extra_val_cols=VAL_EXTRA)
        ref_keys = pd.read_parquet(C.outputs_dir() / f"F153_oof_{fold}.parquet", columns=["site_id", "ts"])
        support = np.zeros(d["actual"].shape, bool)
        si = pd.Index(d["sites"]).get_indexer(ref_keys.site_id)
        ti = d["ts"].get_indexer(ref_keys.ts)
        assert (si >= 0).all() and (ti >= 0).all()
        support[si, ti] = True
        d["prediction_support"] = support
        assert groupmap.index.isin(d["sites"]).all() and len(groupmap) == len(d["sites"])
        g = groupmap.reindex(d["sites"]).to_numpy(int)
        np.testing.assert_array_equal(g, strat_comp(d["cap"], d["wa_st"], seed=6))
        ds[fold], groups[fold] = d, g
        actuals[fold], audits[fold] = independent_labels(d, fold, g)
        paths = {m: C.outputs_dir() / f"{m}_oof_{fold}.parquet" for m in ("F153", "F114_xgb", "F155")}
        paths["N530"] = C.outputs_dir() / f"N530x20_{fold.lower()}_oof_{fold}.parquet"
        memberpred = {m: aligned_prediction(p, d) for m, p in paths.items()}
        mixes[fold] = {}
        inputs[fold] = {m: sha(p) for m, p in paths.items()}
        for arm in arms:
            pred = dict(memberpred)
            if arm != "supplied_S929":
                for m in ("F153", "F155"):
                    path = C.outputs_dir() / f"{m}_{arm}_oof_{fold}.parquet"
                    pred[m] = aligned_prediction(path, d)
                    inputs[fold][f"{m}_{arm}"] = sha(path)
            mixed = (2 * pred["F153"] + pred["F114_xgb"] + 3 * pred["F155"] + 3 * pred["N530"]) / 9
            mixes[fold][arm] = {**d, "pred": mixed}
        directpath = root / "received/direct/outputs" / f"G314cams_direct_oof_{fold}.parquet"
        direct[fold] = direct_cf(directpath, d, g)
        inputs[fold]["direct"] = sha(directpath)
    out = root / ("reference" if reference_only else "comparison")
    out.mkdir(exist_ok=True)
    result = {"status": "verified", "scores": {}, "paired_comparisons": {}, "audits": audits,
              "input_sha256": inputs, "scorer_sha256": sha(__file__),
              "source": json.loads((root / "source.json").read_text()),
              "settings": {"weights": [2, 1, 3, 3], "direct_alpha": .15, "calibration_shrink": .5,
                           "calibration_pairs": {"F2": "F3", "F3": "F2"}, "bootstrap_days": 7,
                           "n_boot": 1000, "bootstrap_seed": 0, "primary": "seen"},
              "prediction_support": "Exact supplied F153 OOF site-time keys, held fixed across all arms",
              "limitations": ["F1 OOF absent; full peer adoption rule cannot be evaluated",
                  "F2 uses later F3 calibration, following peer retrospective diagnostic",
                  "F2/F3 already inspected; not an independent final test",
                  "Only F153/F155 retrained, other S929 components reused unchanged",
                  "Control/raw18/PCA10 trees stop at prior May31 17 KST; supplied baseline differs",
                  "UM core09-17KST only; missing values stay NaN; historical publication unverified",
                  "Raw18 can retain partial-point information whereas PCA requires all nine per variable"]}
    raw = pd.read_parquet(root / "work/external/kma/um_core_hourly.parquet")
    complete_um = raw.set_index("ts").notna().all(axis=1)
    rows = []
    for fold, cal in [("F2", "F3"), ("F3", "F2")]:
        d, g = ds[fold], groups[fold]
        for mode in ("seen", "all"):
            key = f"{fold}_{mode}"
            result["scores"][key] = {}
            mask = d["has"] & (d["seen"][:, None] if mode == "seen" else True)
            actual = np.array([(d["actual"] * mask)[g == k].sum(0) for k in (0, 1)])
            np.testing.assert_allclose(actual, actuals[fold][mode], rtol=0, atol=1e-6)
            caphas = np.array([(d["cap"][:, None] * mask)[g == k].sum(0) for k in (0, 1)])
            predictions = {}
            for arm in arms:
                sp = X9.site_part(mixes[fold][arm], g, mixes[cal][arm], groups[cal], .5,
                                  score_sites=d["seen"] if mode == "seen" else None)
                pred = .85 * sp + .15 * np.where(sp > 0, direct[fold] * caphas, 0)
                assert np.isfinite(pred).all() and (pred >= 0).all()
                predictions[arm] = pred
                peer_score = W_groups(pred - actual, d["w"])
                independent = math.fsum((np.abs(pred - actuals[fold][mode]) * d["w"][None, :]).ravel().tolist())
                assert abs(peer_score - independent) < 1e-3
                result["scores"][key][arm] = peer_score
                rows.append(pd.DataFrame({"arm": arm, "mode": mode, "fold": fold,
                    "group": np.repeat(["G1", "G2"], len(d["ts"])), "ts": list(d["ts"]) * 2,
                    "pred": pred.ravel(), "actual": actual.ravel(), "weight": np.tile(d["w"], 2)}))
            if not reference_only:
                pairs = [("control", "raw18"), ("control", "pca10"), ("raw18", "pca10"),
                         ("supplied_S929", "raw18"), ("supplied_S929", "pca10")]
                valid = complete_um.reindex(d["ts"]).fillna(False).to_numpy(bool)
                comparisons = {}
                for base, cand in pairs:
                    boot = paired_block_boot(predictions[base] - actual, predictions[cand] - actual,
                                             d["w"], d["ts"], block_days=7, n_boot=1000, seed=0)
                    boot["exact_improvement_pct"] = 100 * (1 - result["scores"][key][cand] / result["scores"][key][base])
                    boot["complete_UM_hours"] = int(valid.sum())
                    boot["complete_UM_hours_W"] = {a: W_groups((predictions[a] - actual)[:, valid], d["w"][valid]) for a in (base, cand)}
                    comparisons[f"{cand}_vs_{base}"] = boot
                result["paired_comparisons"][key] = comparisons
    table = pd.concat(rows, ignore_index=True)
    table.to_parquet(out / "group_predictions.parquet", index=False)
    result["group_predictions_sha256"] = sha(out / "group_predictions.parquet")
    save(out / "results.json", result)
    print(json.dumps({"status": "verified", "reference_only": reference_only, "scores": result["scores"]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--reference-only", action="store_true")
    args = ap.parse_args()
    root = args.run_root.resolve()
    setup(root)
    compare(root, args.reference_only)
