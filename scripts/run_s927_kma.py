"""S927-configured causal plant models for an isolated Mac mini experiment.

No original data mutation, official submission, or parameter search. The peer
source and caches are supplied separately under --run-root.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    tmp.replace(path)


def setup(root):
    paths = json.loads((root / "runtime_paths.json").read_text())
    # macOS dyld reads its search path at process startup, before dlopen imports.
    if os.environ.get("S927_RUNTIME_READY") != str(root):
        env = dict(os.environ)
        env.update(paths)
        env["S927_RUNTIME_READY"] = str(root)
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    os.environ.update(paths)
    sys.path[:0] = paths["PYTHONPATH"].split(os.pathsep)
    os.environ.setdefault("OIBC_N_JOBS", "4")
    from oibc import config as C
    import pandas as pd
    for name, fold in list(C.FOLDS.items()):
        C.FOLDS[name] = C.Fold(name, fold.train_end - pd.Timedelta(hours=7), fold.val_start, fold.val_end)


def member(root, name, fold, source):
    import numpy as np
    import pandas as pd
    from oibc import config as C, fcst, fcst_pipe as P, submit, gfs

    params = json.loads((root / "peer_source/reproduce/params_S927.json").read_text())
    key = name if name in ("F153", "F155") else f"{name}_f{fold[-1]}"
    cfg = {**P.base_cfg(), **params[key]["params"], "folds": fold, "boot_base": ""}
    seeds = list(range(20)) if name in ("N502c", "N530") else [0]
    label = f"{name}_{source}_{fold}"
    folder = root / "runs" / label
    folder.mkdir(exist_ok=True)
    target_name = (name + "20" if name == "N502c" else "N530x20" if name == "N530" else name)
    if source != "baseline":
        target_name += "_" + source
    target = C.outputs_dir() / f"{target_name}_oof_{fold}.parquet"
    if (folder / "complete.json").exists():
        old = json.loads((folder / "complete.json").read_text())
        assert sha(target) == old["prediction_sha256"]
        print(json.dumps({"status": "completed_member_reused", "member": label}), flush=True)
        return
    if name in ("F153", "F155"):
        GP = submit.load_experiment("F153_grid_pca")
        GP.GP_END_FINAL = C.FOLDS[fold].train_end + pd.Timedelta(hours=1)
        GP.install()
    elif name == "N530":
        W = submit.load_experiment("W405_gfs_kt")
        W._KT, W._KTCOLS = W.load_kt("all")
        P.build = W._build_with_kt
    if source != "baseline":
        if name not in ("F153", "F155"):
            raise ValueError("The frozen experiment only adds KMA inputs to the two tree members")
        feature_path = root / "work/external/kma" / f"{source}_{fold}_pca.parquet"
        table = pd.read_parquet(feature_path)
        cols = [c for c in table if c != "ts"]
        original_build = P.build
        def add_kma(df, config, stats=None, st=None):
            existing = original_build(df, config, stats, st)
            return existing + gfs.attach(df, table, cols)
        P.build = add_kma
    # Load full cached history then enforce issuance cutoff explicitly.
    tr, va = fcst.load_fold(fold, P.need_cols(cfg), years=None,
                           extra_train=P.extra_train(cfg), ver=cfg["feat_ver"])
    cut = C.FOLDS[fold].train_end
    tr = tr[tr.ts <= cut]
    if cfg["years"]:
        lower = C.FOLDS[fold].val_start - pd.DateOffset(months=int(12 * cfg["years"]))
        tr = tr[tr.ts >= lower]
    tr = tr.reset_index(drop=True)
    assert tr.ts.max() <= cut < va.ts.min()
    inputs = {"cache_train": sha(C.cache_dir() / f"feat_{cfg['feat_ver']}_{fold}_train.parquet"),
              "cache_val": sha(C.cache_dir() / f"feat_{cfg['feat_ver']}_{fold}_val.parquet"),
              "gfs_v1": sha(root / "work/external/gfs/gfs_hourly_v1.parquet")}
    if name == "N530":
        inputs["gfs_v2"] = sha(root / "work/external/gfs/gfs_hourly_v2.parquet")
    if name in ("F153", "F155"):
        inputs["gfs_grid"] = sha(root / "work/external/gfs/gfs_grid_v1.parquet")
    if source != "baseline":
        inputs[source] = sha(feature_path)
    signature = hashlib.sha256(json.dumps({"cfg": cfg, "seeds": seeds, "inputs": inputs}, sort_keys=True).encode()).hexdigest()
    start = time.monotonic()
    acc = np.zeros(len(va), dtype=np.float64)
    iteration_counts = []
    for seed in seeds:
        seed_path = folder / f"seed{seed:02d}.npy"
        meta = seed_path.with_suffix(".json")
        if seed_path.exists() and meta.exists():
            saved = json.loads(meta.read_text())
            assert saved["signature"] == signature and sha(seed_path) == saved["sha256"]
            prediction = np.load(seed_path)
            iters = saved["iterations"]
        else:
            save(folder / "status.json", {"status": "training", "seed": seed, "seeds_total": len(seeds),
                 "member": label, "elapsed_seconds": time.monotonic() - start, "pid": os.getpid()})
            then = time.monotonic()
            res = P.fit_predict(tr, va, {**cfg, "seed": seed, "n_seeds": 1})
            prediction = np.asarray(res["pred"], dtype=np.float64)
            iters = res["info"]
            assert len(prediction) == len(va) and np.isfinite(prediction).all() and (prediction >= 0).all()
            np.save(seed_path, prediction)
            saved = {"signature": signature, "sha256": sha(seed_path), "iterations": iters,
                     "seconds": time.monotonic() - then}
            save(meta, saved)
            print(json.dumps({"status": "seed_complete", "member": label, "seed": seed,
                              "seconds": saved["seconds"], "iterations": iters}), flush=True)
        acc += prediction
        iteration_counts.append(iters)
        gc.collect()
    out = va[["site_id", "ts", "capacity_kw", "energy", "gen_missing", "seen"]].rename(columns={"energy": "actual"}).copy()
    out["pred"] = acc / len(seeds)
    out.to_parquet(target, index=False)
    info = {"status": "complete", "member": label, "cfg": cfg, "seeds": seeds,
            "input_sha256": inputs, "prediction_sha256": sha(target), "iterations": iteration_counts,
            "train_rows": len(tr), "train_start": str(tr.ts.min()), "train_end": str(tr.ts.max()),
            "validation_start": str(va.ts.min()), "validation_end": str(va.ts.max()),
            "elapsed_seconds": time.monotonic() - start, "runner_sha256": sha(__file__),
            "device": os.environ.get("OIBC_TORCH_DEVICE", "peer_default"),
            "caveat": "S927 settings, causal cutoff/calibration and local device; not identical server weights"}
    save(folder / "complete.json", info)
    save(folder / "status.json", {"status": "complete", "member": label})
    print(json.dumps({"status": "member_complete", "member": label, "seconds": info["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--member", choices=["F153", "F155", "N502c", "N530"], required=True)
    parser.add_argument("--fold", choices=["F2", "F3"], required=True)
    parser.add_argument("--source", choices=["baseline", "um", "kim"], default="baseline")
    args = parser.parse_args()
    root = args.run_root.resolve()
    setup(root)
    member(root, args.member, args.fold, args.source)


if __name__ == "__main__":
    main()
