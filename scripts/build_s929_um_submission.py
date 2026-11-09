"""Validate a fixed UM/S929 fallback, train two final trees, write files only.

No upload, competition submission, parameter search or credential access.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
from run_s929_um import save, sha

UM_COLS = [f"um_{v}_p{i}" for v in ("tdsw", "tcam") for i in range(1, 10)]
KEYS = ["group", "ts"]
CUTOFF = pd.Timestamp("2025-05-31 17:00", tz="Asia/Seoul")


def available(raw, times):
    assert set(raw.columns) == {"ts", *UM_COLS}
    assert not raw.ts.duplicated().any(), "Duplicate UM timestamps"
    values = raw.set_index("ts")[UM_COLS].reindex(pd.DatetimeIndex(times))
    return np.isfinite(values.to_numpy(float)).all(axis=1)


def fallback(base, candidate, raw, value="pred"):
    """Exact keys, exact baseline values wherever any UM feature is absent."""
    for frame in (base, candidate):
        assert not frame.duplicated(KEYS).any(), "Duplicate forecast keys"
        assert np.isfinite(frame[value]).all() and (frame[value] >= 0).all()
    b = base.set_index(KEYS)
    c = candidate.set_index(KEYS)
    assert len(b) == len(c) and b.index.isin(c.index).all(), "Forecast keys differ"
    c = c.reindex(b.index)
    use = available(raw, b.index.get_level_values("ts"))
    out = base.copy()
    out[value] = np.where(use, c[value].to_numpy(), b[value].to_numpy())
    out["source"] = np.where(use, "UM_raw18", "S929")
    np.testing.assert_array_equal(out.loc[~use, value], base.loc[~use, value])
    return out


def validate(root):
    from oibc.boot import paired_block_boot
    previous = Path(json.loads((root / "paths.json").read_text())["previous_run"])
    source = previous / "comparison/group_predictions.parquet"
    old = json.loads((previous / "comparison/results.json").read_text())
    assert sha(source) == old["group_predictions_sha256"]
    table = pd.read_parquet(source)
    raw = pd.read_parquet(root / "inputs/um_history.parquet")
    scores, outputs = {}, []
    for fold in ("F2", "F3"):
        for mode in ("seen", "all"):
            part = table[(table.fold == fold) & (table['mode'] == mode)]
            b = part[part.arm == "supplied_S929"].sort_values(KEYS).reset_index(drop=True)
            u = part[part.arm == "raw18"].sort_values(KEYS).reset_index(drop=True)
            for col in ["actual", "weight"]:
                np.testing.assert_array_equal(b[col], u[col])
            out = fallback(b, u, raw)
            times = pd.DatetimeIndex(sorted(b.ts.unique()))
            act = b.actual.to_numpy().reshape(2, -1)
            w = b.weight.to_numpy().reshape(2, -1)
            np.testing.assert_array_equal(w[0], w[1])
            eb = b.pred.to_numpy().reshape(2, -1) - act
            en = out.pred.to_numpy().reshape(2, -1) - act
            wb = float((np.abs(eb)*w).sum()); wn = float((np.abs(en)*w).sum())
            assert abs(wb-old["scores"][f"{fold}_{mode}"]["supplied_S929"]) < 1e-3
            # Fixed secondary diagnostic: simulate source termination from Apr2
            # to May31 in each historical year. This never changes the rule.
            stress = out.copy()
            late = ((stress.ts.dt.month == 4) & (stress.ts.dt.day >= 2)) | (stress.ts.dt.month == 5)
            stress.loc[late, "pred"] = b.loc[late, "pred"]
            ws = float((np.abs(stress.pred.to_numpy().reshape(2,-1)-act)*w).sum())
            scores[f"{fold}_{mode}"] = {"S929_W":wb, "fallback_W":wn,
                "improvement_pct":100*(1-wn/wb), "um_group_hours":int((out.source=="UM_raw18").sum()),
                "baseline_group_hours":int((out.source=="S929").sum()),
                "bootstrap":paired_block_boot(eb,en,w[0],times,block_days=7,n_boot=1000,seed=0),
                "simulated_Apr2_May31_fallback_W":ws, "simulated_improvement_pct":100*(1-ws/wb)}
            out["arm"] = "raw18_with_S929_fallback"
            outputs.append(out)
    target = root / "validation/group_predictions.parquet"
    target.parent.mkdir(exist_ok=True)
    pd.concat(outputs, ignore_index=True).to_parquet(target,index=False)
    result = {"status":"verified", "rule":"All 18 UM features finite at timestamp, else final S929 prediction",
        "scores":scores, "source_sha256":sha(source),"history_sha256":sha(root/"inputs/um_history.parquet"),
        "prediction_sha256":sha(target), "limitations":old["limitations"],
        "independent_final_test":False,"training_performed":False}
    save(root/"validation/results.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def train(root, member):
    from oibc import config as C, fcst, fcst_pipe as P, submit, gfs
    from oibc.group_calib import VAL_EXTRA
    folder = root / "models" / member
    folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/"complete.json").exists(), "Model already complete"
    cfg = {**P.base_cfg(), **json.loads((root/"peer_source/reproduce/params_S929.json").read_text())[member+"_final"]["params"],"boot_base":""}
    assert cfg["model"]=="lgb" and cfg["n_seeds"]==1 and cfg["seed"]==0 and cfg["years"]==0
    GP = submit.load_experiment("F153_grid_pca")
    GP.GP_END_FINAL = CUTOFF+pd.Timedelta(hours=1)
    GP.install()
    table = pd.read_parquet(root/"inputs/um_combined.parquet")
    original_build = P.build
    def add_um(df, config, stats=None, st=None):
        cols = original_build(df, config, stats, st)
        return cols+gfs.attach(df,table,UM_COLS)
    P.build = add_um
    original_fit = fcst.fit_lgb
    saved = []
    def save_fit(tr,cols,*args,**kwargs):
        assert tr.ts.max() <= CUTOFF
        model,best = original_fit(tr,cols,*args,**kwargs)
        path = folder/f"tree_{len(saved)}.txt"
        model.booster_.save_model(str(path))
        saved.append({"file":path.name,"sha256":sha(path),"features":cols,"iterations":best})
        return model,best
    fcst.fit_lgb = save_fit
    tr,ev = submit.load_final(P.need_cols(cfg),years=None,extra_train=P.extra_train(cfg),extra_eval=VAL_EXTRA,ver="v1")
    original_end = str(tr.ts.max())
    tr=tr[tr.ts<=CUTOFF].reset_index(drop=True)
    assert tr.ts.max()<=CUTOFF<ev.ts.min()
    started=time.monotonic()
    save(folder/"status.json",{"status":"training","member":member,"pid":os.getpid()})
    res=P.fit_predict(tr,ev,cfg)
    pred=np.asarray(res["pred"],float)
    assert len(pred)==len(ev) and np.isfinite(pred).all() and (pred>=0).all()
    assert (pred<=ev.capacity_kw.to_numpy(float)+1e-5).all()
    out=ev[list(dict.fromkeys(["site_id","ts","capacity_kw"]+VAL_EXTRA))].copy()
    out["pred"]=pred
    dest=root/"work/outputs"/f"{member}_um_final_eval_sites.parquet"
    out.to_parquet(dest,index=False)
    GP._ST["scores"].to_parquet(folder/"gfs_pca_scores.parquet",index=False)
    save(folder/"complete.json",{"status":"complete","cfg":cfg,"train_rows":len(tr),
        "train_start":str(tr.ts.min()),"train_end":str(tr.ts.max()),"source_train_end":original_end,
        "eval_start":str(ev.ts.min()),"eval_end":str(ev.ts.max()),"eval_rows":len(ev),
        "iterations":res["info"],"models":saved,"gfs_pca":GP.summary(),
        "gfs_pca_scores_sha256":sha(folder/"gfs_pca_scores.parquet"),
        "prediction_sha256":sha(dest),"seconds":time.monotonic()-started,
        "runner_sha256":sha(__file__),"um_sha256":sha(root/"inputs/um_combined.parquet")})
    print(member+" final training complete",flush=True)


def attach_predictions(base, path):
    other=pd.read_parquet(path,columns=["site_id","ts","pred"])
    assert not other.duplicated(["site_id","ts"]).any()
    joined=base[["site_id","ts"]].merge(other,on=["site_id","ts"],how="left",validate="one_to_one")
    assert np.isfinite(joined.pred).all() and (joined.pred>=0).all(), str(path)
    return joined.pred.to_numpy(float)


def label_calibrated_groups(frame, composition):
    """Peer calibrator emits 0/1; verify each plant before naming G1/G2."""
    assert not composition.site_id.duplicated().any()
    labels=frame.group.map({0:"G1",1:"G2"})
    expected=frame.site_id.map(composition.set_index("site_id").group)
    assert labels.notna().all() and expected.notna().all()
    np.testing.assert_array_equal(labels.to_numpy(),expected.to_numpy())
    return frame.assign(group=labels)


def assemble(root):
    import joblib
    from oibc import config as C, submit, scoring
    from oibc.group_calib import VAL_EXTRA,load_oof_for_calib,fit_calibrator,apply_calibrator
    comp=pd.read_csv(C.outputs_dir()/"C_strat_v1_composition.csv",dtype={"site_id":str})
    summary=submit.composition_summary(comp)
    assert summary["feasible"]
    outputs=C.outputs_dir()
    direct=pd.read_parquet(root/"received/direct/outputs/G314cams_direct_final_eval.parquet")
    assert not direct.duplicated(KEYS).any()
    cap=comp.merge(pd.read_csv(C.data_dir()/"sites.csv")[["site_id","capacity_kw"]],on="site_id").groupby("group").capacity_kw.sum()
    np.testing.assert_allclose(direct.cap_all,direct.group.map(cap),rtol=1e-6,atol=1e-3)
    np.testing.assert_allclose(direct.pred,direct.cf*direct.cap_all,rtol=1e-6,atol=1e-3)
    assert np.isfinite(direct.pred).all() and (direct.pred>=0).all()
    extra=pd.read_parquet(C.cache_dir()/"feat_v1_eval.parquet",columns=["site_id","ts","capacity_kw"]+VAL_EXTRA)
    groups={}
    calibration={}
    for arm in ["S929","UM_raw18"]:
        oofs=[]
        for fold in ["F2","F3"]:
            o=load_oof_for_calib("F153",fold)
            assert not o.duplicated(["site_id","ts"]).any()
            suffix="" if arm=="S929" else "_raw18"
            paths=[outputs/f"F153{suffix}_oof_{fold}.parquet",outputs/f"F114_xgb_oof_{fold}.parquet",
                   outputs/f"F155{suffix}_oof_{fold}.parquet",outputs/f"N530x20_{fold.lower()}_oof_{fold}.parquet"]
            o["pred"]=sum(w*attach_predictions(o,p) for w,p in zip([2,1,3,3],paths))/9
            # Preserve the peer's supplied S929 recipe. New UM calibration is
            # restricted to the same explicit 17KST cutoff as its new trees.
            if arm=="UM_raw18": o=o[o.ts<=CUTOFF].copy()
            oofs.append(o)
        model=fit_calibrator(oofs,[scoring.weights_for_fold(C.FOLDS[f]) for f in ["F2","F3"]],comp,{"shrink":.5,"noround":True})
        folder=root/"models"/arm;folder.mkdir(exist_ok=True)
        joblib.dump(model,folder/"calibrator.joblib")
        calibration[arm]={"n_rows":model["n_rows"],"cfg":model["cfg"],"max_label_ts":str(max(o.ts.max() for o in oofs)),"sha256":sha(folder/"calibrator.joblib")}
        base=pd.read_parquet(outputs/"F153_final_eval_sites.parquet",columns=["site_id","ts","pred"])
        names=["F153_final","S909_xgb_build","F155_final","N530_final20"] if arm=="S929" else ["F153_um_final","S909_xgb_build","F155_um_final","N530_final20"]
        base["pred"]=sum(w*attach_predictions(base,outputs/(name+"_eval_sites.parquet")) for w,name in zip([2,1,3,3],names))/9
        pdf=base.merge(extra,on=["site_id","ts"],how="left",validate="one_to_one")
        assert pdf.capacity_kw.notna().all()
        calibrated=apply_calibrator(model,pdf,comp)
        calibrated=label_calibrated_groups(calibrated,comp)
        calibrated["pred_cal"]=calibrated.pred_cal.clip(lower=0)
        site=calibrated.groupby(KEYS,as_index=False).pred_cal.sum().rename(columns={"pred_cal":"site"})
        gb=site.merge(direct[KEYS+["pred"]].rename(columns={"pred":"direct"}),on=KEYS,how="left",validate="one_to_one")
        assert direct.set_index(KEYS).index.isin(site.set_index(KEYS).index).all()
        gb["direct"]=np.where(gb.site>0,gb.direct.fillna(0),0)
        gb["pred"]=.85*gb.site+.15*gb.direct
        assert len(gb)==17520 and np.isfinite(gb.pred).all() and (gb.pred>=0).all()
        groups[arm]=gb[KEYS+["pred"]].sort_values(KEYS).reset_index(drop=True)
        groups[arm].to_parquet(root/f"{arm}_final_group_predictions.parquet",index=False)
    raw=pd.read_parquet(root/"inputs/um_combined.parquet")
    result=fallback(groups["S929"],groups["UM_raw18"],raw)
    result.to_parquet(root/"final_prediction_sources.parquet",index=False)
    dest=root/"submission";dest.mkdir(exist_ok=True)
    comp[["site_id","group"]].sort_values("site_id").to_csv(dest/"composition.csv",index=False)
    forecast=result[KEYS+["pred"]].rename(columns={"pred":"forecast_kwh"})
    forecast["ts"]=submit.format_ts(forecast.ts).to_numpy()
    forecast.to_csv(dest/"forecast.csv",index=False)
    check=verify_submission(dest,C.data_dir(),result,groups["S929"],raw)
    check.update(calibration=calibration,composition_summary=summary,
        baseline_origin="Reconstructed from supplied final member predictions and fixed peer S929 recipe; original submitted forecast not supplied",
        historical_publication_certified=False,uploaded=False,submitted=False)
    save(root/"submission_verification.json",check)
    print(json.dumps(check,ensure_ascii=False),flush=True)


def verify_submission(dest,data_dir,result,base,raw):
    comp=pd.read_csv(dest/"composition.csv",dtype={"site_id":str})
    fc=pd.read_csv(dest/"forecast.csv")
    sites=pd.read_csv(data_dir/"sites.csv",dtype={"site_id":str})
    assert list(comp.columns)==["site_id","group"] and len(comp)==253
    assert not comp.site_id.duplicated().any() and set(comp.site_id)==set(sites.site_id)
    assert set(comp.group)=={"G1","G2"}
    cap=comp.merge(sites[["site_id","capacity_kw"]],on="site_id",validate="one_to_one").groupby("group").capacity_kw.sum()
    assert (cap>=20000).all()
    assert list(fc.columns)==["group","ts","forecast_kwh"] and len(fc)==17520 and not fc.duplicated(KEYS).any()
    assert fc.ts.str.endswith("+09:00").all()
    fc["ts"]=pd.to_datetime(fc.ts)
    axis=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(data_dir/"smp.csv").ts))
    assert len(axis)==8760 and axis.is_unique
    for _,g in fc.groupby("group"):
        assert pd.DatetimeIndex(g.ts).equals(axis)
    assert np.isfinite(fc.forecast_kwh).all() and (fc.forecast_kwh>=0).all()
    assert (fc.forecast_kwh.to_numpy()<=fc.group.map(cap).to_numpy()*1.2+1e-6).all()
    use=available(raw,result.ts)
    np.testing.assert_array_equal(result.loc[~use,"pred"],base.loc[~use,"pred"])
    np.testing.assert_allclose(fc.forecast_kwh,result.pred,rtol=1e-14,atol=1e-9)
    late=result.ts>=pd.Timestamp("2026-04-02 01:00",tz="Asia/Seoul")
    assert (result.loc[late,"source"]=="S929").all()
    files={n:{"bytes":(dest/n).stat().st_size,"sha256":sha(dest/n)} for n in ["composition.csv","forecast.csv"]}
    assert all(v["bytes"]<20_000_000 for v in files.values())
    return {"status":"verified","rows":len(fc),"sites":len(comp),"start":str(axis.min()),"end":str(axis.max()),
        "um_group_hours":int(use.sum()),"S929_group_hours":int((~use).sum()),"files":files,
        "missing_predictions":0,"duplicate_keys":0,"group_capacity_kw":cap.to_dict()}


def controller(root):
    with (root/"controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/"completion.json").exists()
        started=time.time()
        try:
            manifest=json.loads((root/"input_manifest.json").read_text())
            for name,digest in manifest.items(): assert sha(root/name)==digest,name
            for step in ["validate","F153","F155","assemble"]:
                save(root/"status.json",{"status":"running","step":step,"pid":os.getpid(),"started":started})
                subprocess.run([sys.executable,__file__,"--run-root",str(root),"--step",step],check=True)
            verified=json.loads((root/"submission_verification.json").read_text())
            assert verified["status"]=="verified"
            for name,digest in manifest.items(): assert sha(root/name)==digest,name
            done={"status":"complete","started":started,"finished":time.time(),"seconds":time.time()-started,
                "verification_sha256":sha(root/"submission_verification.json"),"uploaded":False,"submitted":False}
            save(root/"completion.json",done);save(root/"status.json",done)
        except BaseException:
            save(root/"status.json",{"status":"failed","traceback":traceback.format_exc(),"time":time.time()})
            raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--step",choices=["validate","F153","F155","assemble","controller"],default="controller")
    args=ap.parse_args();root=args.run_root.resolve()
    env=json.loads((root/"runtime_paths.json").read_text())
    if os.environ.get("S929_FINAL_READY")!=str(root):
        os.execve(sys.executable,[sys.executable,*sys.argv],{**os.environ,**env,"S929_FINAL_READY":str(root)})
    sys.path[:0]=env["PYTHONPATH"].split(os.pathsep)
    if args.step=="controller":controller(root)
    elif args.step=="validate":validate(root)
    elif args.step=="assemble":assemble(root)
    else:train(root,args.step)


if __name__=="__main__":main()
