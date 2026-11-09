"""Independent checks of final CSV selection, saved models and PCA masks."""
from pathlib import Path
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from run_s929_um import save,sha


def audit(root):
    import lightgbm as lgb
    from oibc import config as C, gfs_grid
    done=json.loads((root/"completion.json").read_text())
    assert done["status"]=="complete" and not done["submitted"] and not done["uploaded"]
    report=json.loads((root/"submission_verification.json").read_text())
    external=json.loads((root/"external_manifest.json").read_text())
    for name,digest in external.items():assert sha(root/name)==digest,name
    assert sha(root/"submission_verification.json")==done["verification_sha256"]
    for name,v in report["files"].items():assert sha(root/"submission"/name)==v["sha256"]
    raw=pd.read_parquet(root/"inputs/um_combined.parquet").set_index("ts")
    fc=pd.read_csv(root/"submission/forecast.csv")
    fc["ts"]=pd.DatetimeIndex(pd.to_datetime(fc.ts)).tz_convert("Asia/Seoul").as_unit("ns")
    sources=pd.read_parquet(root/"final_prediction_sources.parquet")
    base=pd.read_parquet(root/"S929_final_group_predictions.parquet").set_index(["group","ts"])
    um=pd.read_parquet(root/"UM_raw18_final_group_predictions.parquet").set_index(["group","ts"])
    idx=pd.MultiIndex.from_frame(fc[["group","ts"]])
    matrix=raw.reindex(pd.DatetimeIndex(fc.ts)).to_numpy(float)
    gate=(~np.isnan(matrix)&~np.isinf(matrix)).sum(axis=1)==18
    expected=base.reindex(idx).pred.to_numpy().copy()
    expected[gate]=um.reindex(idx).pred.to_numpy()[gate]
    np.testing.assert_allclose(fc.forecast_kwh,expected,rtol=1e-14,atol=1e-9)
    np.testing.assert_array_equal(sources.source=="UM_raw18",gate)
    assert not fc.duplicated(["group","ts"]).any()
    ts=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(C.data_dir()/"smp.csv").ts)).tz_convert("Asia/Seoul").as_unit("ns")
    expected_keys=pd.MultiIndex.from_product([["G1","G2"],ts],names=["group","ts"])
    assert idx.equals(expected_keys)
    pca_info={};model_info={}
    for member in ["F153","F155"]:
        folder=root/"models"/member
        meta=json.loads((folder/"complete.json").read_text())
        assert pd.Timestamp(meta["train_end"])<=pd.Timestamp("2025-05-31 17:00",tz="Asia/Seoul")
        assert meta["eval_rows"]==253*8760 and meta["iterations"]["n_feat"]==236
        path=root/"work/outputs"/f"{member}_um_final_eval_sites.parquet"
        assert sha(path)==meta["prediction_sha256"]
        p=pd.read_parquet(path,columns=["site_id","ts","capacity_kw","pred"])
        assert len(p)==253*8760 and not p.duplicated(["site_id","ts"]).any()
        assert np.isfinite(p.pred).all() and (p.pred>=0).all() and (p.pred<=p.capacity_kw+1e-5).all()
        for model in meta["models"]:
            assert sha(folder/model["file"])==model["sha256"]
            booster=lgb.Booster(model_file=str(folder/model["file"]))
            assert booster.num_feature()==236 and booster.feature_name()==model["features"]
            assert booster.current_iteration()==model["iterations"]
        model_info[member]={"saved_model_verified":True,"seconds":meta["seconds"],
            "iterations":meta["iterations"],"train_end":meta["train_end"]}
        scores=pd.read_parquet(folder/"gfs_pca_scores.parquet")
        assert sha(folder/"gfs_pca_scores.parquet")==meta["gfs_pca_scores_sha256"]
        pca_info[member]={}
        for var in ["dswrf","tcdc","lcdc","mcdc","hcdc"]:
            src=gfs_grid.load_hourly(gfs_grid.grid_cols([var]))
            assert pd.DatetimeIndex(src.ts).as_unit("ns").equals(pd.DatetimeIndex(scores.ts).as_unit("ns"))
            x=src.iloc[:,1:].to_numpy(float)
            assert not np.isinf(x).any()
            valid=np.isfinite(x).all(axis=1)
            score=scores[gfs_grid.pca_cols([var],5)].to_numpy(float)
            assert np.isfinite(score[valid]).all(), "PCA introduced nonfinite values"
            assert np.isnan(score[~valid]).all(), "Missing source mask changed"
            pca_info[member][var]={"valid_rows":int(valid.sum()),"source_missing_rows":int((~valid).sum()),
                "max_abs_score":float(np.abs(score[valid]).max()),"unexpected_nonfinite":0}
    result={"status":"verified","forecast_rows":len(fc),"um_rows":int(gate.sum()),
        "S929_rows":int((~gate).sum()),"forecast_source_selection_verified":True,
        "model_verification":model_info,"pca_verification":pca_info,
        "pca_runtime_warnings":"Saved scores finite on every complete source row; no additional missingness",
        "external_inputs_verified":len(external),
        "submitted":False,"uploaded":False,"auditor_sha256":sha(__file__)}
    save(root/"independent_final_audit.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--run-root",type=Path,required=True)
    args=ap.parse_args();root=args.run_root.resolve()
    env=json.loads((root/"runtime_paths.json").read_text())
    if os.environ.get("S929_AUDIT_READY")!=str(root):
        os.execve(sys.executable,[sys.executable,*sys.argv],{**os.environ,**env,"S929_AUDIT_READY":str(root)})
    sys.path[:0]=env["PYTHONPATH"].split(os.pathsep)
    audit(root)
