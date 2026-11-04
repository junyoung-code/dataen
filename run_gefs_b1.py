"""Frozen P4 + four public national-area GEFS features on the shared F2/F3 contract."""
from __future__ import annotations
import argparse,gc,hashlib,json,platform,subprocess,sys,time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
import train_smoke as original
from run_peer_w import bounds,peer_labels,score,parameters,write_json
from features_p3 import build
from training_weights import make_weights
from features_gefs_b1 import load_table,attach,COLUMNS
from collect_gefs_b1 import status

ROOT=Path(__file__).resolve().parent
CONTRACT=ROOT/'AB_EVALUATION_CONTRACT.json'
CONTRACT_SHA='d65965390a3027d6b3267c352d7222cdfd76a9dca128e7d319bab3c77465b1ec'
BASELINE=ROOT.parent.parent/'peer_w_20261103_163512_rescore'

def mask_sha(rows):
    keys=rows.sort_values(['site_id','ts']).reset_index(drop=True)[['site_id','ts','has_peer']]
    return hashlib.sha256(keys.to_csv(index=False,lineterminator='\n').encode()).hexdigest()

def contract_check():
    assert original.sha256(CONTRACT)==CONTRACT_SHA
    cfg=json.loads(CONTRACT.read_text())
    for name,expected in cfg['fixed_source_sha256'].items():
        if original.sha256(ROOT/name)!=expected:raise ValueError('Frozen source changed: '+name)
    assert original.sha256(original.DATA/'sites.csv')==cfg['sites_file_sha256']
    sites=pd.read_csv(original.DATA/'sites.csv')
    gen=pd.read_parquet(original.DATA/'train/generation.parquet')
    gen.ts=pd.to_datetime(gen.ts).dt.tz_convert('Asia/Seoul')
    checks={}
    for fold,c in cfg['folds'].items():
        a,b,c0,d=bounds(fold)
        valid=gen.loc[(gen.ts>=c0)&(gen.ts<d)].copy()
        cap=valid.site_id.map(sites.set_index('site_id').capacity_kw)
        _,valid['has_peer']=peer_labels(valid.energy,cap)
        digest=mask_sha(valid)
        assert digest==c['evaluation_key_mask_sha256'],(fold,digest)
        assert len(valid)==c['eval_raw_rows']
        assert ((gen.ts>=a)&(gen.ts<=b)).sum()==c['train_raw_rows']
        base=pd.read_parquet(BASELINE/fold/'site_predictions.parquet',columns=['site_id','ts','has_peer'])
        assert mask_sha(base)==digest
        summary=json.loads((BASELINE/fold/'summary.json').read_text())
        assert abs(summary['metrics']['ensemble']['W']-c['baseline_W'])<1e-4
        checks[fold]=dict(evaluation_key_mask_sha256=digest,evaluation_rows=len(valid),baseline_W=c['baseline_W'])
    return checks

def worker(folder,fold,threads,gefs_path):
    started=time.perf_counter();a,cutoff,c,d=bounds(fold)
    contract=json.loads(CONTRACT.read_text())['folds'][fold]
    b=original.load_baseline();sites=pd.read_csv(original.DATA/'sites.csv')
    gen=pd.read_parquet(original.DATA/'train/generation.parquet')
    gen.ts=pd.to_datetime(gen.ts).dt.tz_convert('Asia/Seoul')
    if gen.duplicated(['site_id','ts']).any() or not np.isfinite(gen.energy).all():raise ValueError('Generation keys')
    train=gen.loc[(gen.ts>=a)&(gen.ts<=cutoff)].copy()
    valid=gen.loc[(gen.ts>=c)&(gen.ts<d)].copy();del gen
    assert len(train)==contract['train_raw_rows'] and len(valid)==contract['eval_raw_rows']
    prices=pd.read_csv(original.DATA/'train/smp.csv');prices.ts=pd.to_datetime(prices.ts).dt.tz_convert('Asia/Seoul')
    fa=pd.read_parquet(original.DATA/'train/forecast_wa.parquet',columns=['wa_station','round','ts']+b.WA_COLS)
    fb=pd.read_parquet(original.DATA/'train/forecast_wb.parquet',columns=['wb_station','round','ts']+b.WB_COLS)
    for f in (fa,fb):f.ts=pd.to_datetime(f.ts).dt.tz_convert('Asia/Seoul')
    fa=fa.loc[(fa.ts>=a)&(fa.ts<d)];fb=fb.loc[(fb.ts>=a)&(fb.ts<d)]
    gefs=load_table(gefs_path)
    output=valid.rename(columns={'energy':'energy_raw'}).merge(sites[['site_id','capacity_kw']],on='site_id',validate='many_to_one')
    output['energy_peer'],output['has_peer']=peer_labels(output.energy_raw,output.capacity_kw)
    digest=mask_sha(output);assert digest==contract['evaluation_key_mask_sha256']
    model_info={};params=parameters(threads)
    for r in (10,17):
        status('training',f'{fold} round {r}: build/fit 30 features',str(folder.parent.parent/(folder.parent.name+'.log')),result_dir=str(folder.parent))
        print(f'{fold} r{r}: build training features',flush=True)
        f,base_cols=build(b,train,sites,fa,fb,r,'station_distance')
        assert base_cols==contract['feature_columns']
        f=attach(f,gefs);f=f.loc[f.elev>0]
        assert f.ts.max()<=cutoff
        columns=base_cols+COLUMNS
        x=f[columns].astype('float32');y=(f.energy/f.cap).clip(0,1.2).astype('float32')
        w,wi=make_weights(f,prices,'capacity',a,cutoff+pd.Timedelta(nanoseconds=1))
        t=time.perf_counter();model=lgb.LGBMRegressor(**params).fit(x,y,sample_weight=w)
        fit_seconds=time.perf_counter()-t
        model_path=folder/f'model_round_{r}.txt';model.booster_.save_model(str(model_path))
        assert model.booster_.num_trees()==200
        model_info[str(r)]=dict(train_rows=len(f),fit_seconds=fit_seconds,train_min=str(f.ts.min()),train_max=str(f.ts.max()),
                weights=wi,missing_fraction=x.isna().mean().to_dict())
        del f,x,y,w;gc.collect()
        print(f'{fold} r{r}: predict; fit={fit_seconds:.3f}s',flush=True)
        f,cols=build(b,valid,sites,fa,fb,r,'station_distance');assert cols==base_cols
        f=attach(f,gefs)
        assert f[['site_id','ts']].reset_index(drop=True).equals(output[['site_id','ts']].reset_index(drop=True))
        x=f[columns].astype('float32');raw=model.predict(x)
        restored=lgb.Booster(model_file=str(model_path))
        np.testing.assert_allclose(raw[:1024],restored.predict(x.iloc[:1024]),rtol=1e-12)
        output[f'pred_{r}']=np.maximum(np.where(f.elev>0,raw,0),0)*f.cap.to_numpy()
        output['daylight']=f.elev.to_numpy()>0
        del f,x,raw,model,restored;gc.collect()
    output['ensemble']=0.25*output.pred_10+0.75*output.pred_17
    scores={}
    for pred in ('pred_10','pred_17','ensemble'):
        scores[pred],agg=score(output,sites,prices,c,d,prediction=pred)
        if pred=='ensemble':agg.to_csv(folder/'group_errors.csv',index=False)
    output.to_parquet(folder/'site_predictions.parquet',index=False)
    delta=(scores['ensemble']['W']/contract['baseline_W']-1)*100
    report=dict(status='complete',fold=fold,train_start=str(a),train_end_inclusive=str(cutoff),eval_start=str(c),eval_end_exclusive=str(d),
        train_raw_rows=len(train),eval_raw_rows=len(valid),evaluation_key_mask_sha256=digest,parameters=params,
        feature_columns=columns,weight_17=0.75,models=model_info,metrics=scores,baseline_W=contract['baseline_W'],
        baseline_change_percent=delta,total_seconds=time.perf_counter()-started,peak_process_memory_mib=original.peak_memory_mib())
    write_json(folder/'summary.json',report)
    print(f'{fold} W={scores["ensemble"]["W"]:.6f} change={delta:+.4f}%',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/b1_annual')
    p.add_argument('--gefs',type=Path,default=ROOT/'outputs/b1_gefs_full/hourly.parquet')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--fold',choices=['F2','F3'])
    p.add_argument('--check-contract',action='store_true');args=p.parse_args()
    if args.check_contract:
        checks=contract_check();write_json(ROOT/'outputs/b1_research/contract_verification.json',checks)
        print(json.dumps(checks,indent=2));return
    folder=args.output_dir.resolve()
    if not folder.is_relative_to(ROOT/'outputs'):raise ValueError('B outputs only')
    if args.fold:
        worker(folder,args.fold,args.threads,args.gefs);return
    folder.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    checks=contract_check()
    source=folder/'source';source.mkdir()
    names=['run_gefs_b1.py','verify_gefs_b1.py','features_gefs_b1.py','collect_gefs_b1.py','test_gefs_b1.py','b1_gefs_config.json',
           'requirements-gefs.txt','run_peer_w.py','features_p3.py','training_weights.py','train_smoke.py','AB_EVALUATION_CONTRACT.json']
    for name in names:(source/name).write_bytes((ROOT/name).read_bytes())
    report=dict(status='running',protocol='peer-annual-w-causal-v1',experiment='B1-P4-GEFS-area-v1',
        historical_comparison_not_independent_final_test=True,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),source_sha256={n:original.sha256(ROOT/n) for n in names},
        machine=dict(platform=platform.platform(),architecture=platform.machine(),node=platform.node()),
        versions=dict(python=platform.python_version(),lightgbm=lgb.__version__,numpy=np.__version__,pandas=pd.__version__,pyarrow=pyarrow.__version__),
        contract_checks=checks,gefs_table_sha256=original.sha256(args.gefs),folds={})
    write_json(folder/'summary.json',report)
    for fold in ('F2','F3'):
        target=folder/fold;target.mkdir()
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--fold',fold,'--threads',str(args.threads),
            '--gefs',str(args.gefs.resolve()),'--output-dir',str(target)],cwd=ROOT,check=True)
        report['folds'][fold]=json.loads((target/'summary.json').read_text());write_json(folder/'summary.json',report)
    report.update(status='awaiting_independent_verification',total_seconds=time.perf_counter()-started)
    write_json(folder/'summary.json',report)
    status('verifying','B1 F2/F3 predictions ready; independent verification required',str(folder.parent/(folder.name+'.log')),result_dir=str(folder))

if __name__=='__main__':main()
