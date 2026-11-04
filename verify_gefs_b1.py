"""Independent raw-label, contract, NumPy W and saved-model checks for B1."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import train_smoke as original
from features_p3 import build
from features_gefs_b1 import attach,load_table,COLUMNS
from collect_gefs_b1 import status

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent/'peer_w_20261103_163512_rescore'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):
    if not np.isclose(a,b,rtol=1e-12,atol=1e-3):raise ValueError('Independent aggregate mismatch')
def numpy_score(rows,sites,prices,start,end,prediction):
    axis=pd.date_range(start,end,freq='h',inclusive='left');n=len(axis)
    t=axis.get_indexer(rows.ts)
    if (t<0).any():raise ValueError('Out of range evaluation key')
    group=rows.site_id.map(sites.set_index('site_id').current_group).map({'A':0,'B':1}).to_numpy(int)
    has=rows.has_peer.to_numpy(bool);idx=group[has]*n+t[has]
    sums=np.bincount(idx,weights=rows[prediction].to_numpy(float)[has]-rows.energy_peer.to_numpy(float)[has],minlength=2*n)
    price=prices.set_index('ts').reindex(axis)
    k=price.k.to_numpy(float) if 'k' in price else np.where(axis.hour.isin(range(11,16)),3.,1.)
    weights=price.smp.to_numpy(float)*k
    if not np.isfinite(weights).all():raise ValueError('Missing price')
    contributions=np.abs(sums.reshape(2,n))*weights[None,:]
    months=(axis-pd.Timedelta(hours=1)).strftime('%Y-%m')
    return dict(W=float(contributions.sum()),W_by_group={g:float(contributions[i].sum()) for i,g in enumerate(['G1','G2'])},
                W_by_month={m:float(contributions[:,months==m].sum()) for m in sorted(set(months))})

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/b1_annual');a=p.parse_args()
    folder=a.output_dir.resolve();started=time.perf_counter()
    status('verifying','Independent B1 source/labels/masks/W/model checks',str(folder.parent/(folder.name+'.verify.log')),result_dir=str(folder))
    summary=json.loads((folder/'summary.json').read_text());contract=json.loads((ROOT/'AB_EVALUATION_CONTRACT.json').read_text())
    for name,digest in summary['source_sha256'].items():
        assert sha(folder/'source'/name)==digest and sha(ROOT/name)==digest, 'Source changed after execution'
    for name,digest in contract['fixed_source_sha256'].items():assert sha(ROOT/name)==digest
    gefs_path=ROOT/'outputs/b1_gefs_full/hourly.parquet'
    assert sha(gefs_path)==summary['gefs_table_sha256']
    collection=json.loads((gefs_path.parent/'collection_summary.json').read_text())
    assert collection['status']=='complete' and collection['days']==1461 and collection['hours']==35064 and not collection['failures']
    gefs=load_table(gefs_path);assert len(gefs)==35064
    expected_axis=pd.date_range('2021-06-01 01:00',periods=35064,freq='h',tz='Asia/Seoul')
    assert np.array_equal(gefs.ts.to_numpy(),expected_axis.to_numpy())
    cfg=json.loads((ROOT/'b1_gefs_config.json').read_text())
    # Verify every cached day checksum and independently recover every hour from saved member-level summaries.
    config_sha=sha(ROOT/'b1_gefs_config.json');cache_days=0
    for day in pd.date_range('2021-06-01','2025-05-31',freq='D').strftime('%Y-%m-%d'):
        f=gefs_path.parent/'days'/f'{day}.parquet';audit=json.loads(f.with_suffix('.json').read_text())
        assert sha(f)==audit['parquet_sha256'] and audit['config_sha256']==config_sha
        daily=pd.read_parquet(f)
        block=gefs.iloc[24*cache_days:24*(cache_days+1)]
        assert np.array_equal(block.ts.to_numpy(),daily.ts.to_numpy())
        assert np.array_equal(block[COLUMNS].to_numpy(),daily[COLUMNS].to_numpy())
        member=np.load(f.with_suffix('.members.npz'))
        # The 24 hour-ending labels cover leads40..63, selecting the eight endpoints42..63.
        idx=np.repeat(np.arange(1,9),3)
        for prefix in ('gefs_sw','gefs_cloud'):
            assert member[prefix].shape==(31,9) and np.isfinite(member[prefix]).all()
            np.testing.assert_allclose(daily[prefix+'_mean'],member[prefix].mean(axis=0)[idx],rtol=1e-6,atol=1e-5)
            np.testing.assert_allclose(daily[prefix+'_std'],member[prefix].std(axis=0,ddof=0)[idx],rtol=1e-6,atol=1e-5)
        cache_days+=1
    sites=pd.read_csv(original.DATA/'sites.csv')
    assert sha(original.DATA/'sites.csv')==contract['sites_file_sha256']
    gen=pd.read_parquet(original.DATA/'train/generation.parquet');gen.ts=pd.to_datetime(gen.ts).dt.tz_convert('Asia/Seoul')
    prices=pd.read_csv(original.DATA/'train/smp.csv');prices.ts=pd.to_datetime(prices.ts).dt.tz_convert('Asia/Seoul')
    b=original.load_baseline()
    fa=pd.read_parquet(original.DATA/'train/forecast_wa.parquet',columns=['wa_station','round','ts']+b.WA_COLS)
    fb=pd.read_parquet(original.DATA/'train/forecast_wb.parquet',columns=['wb_station','round','ts']+b.WB_COLS)
    for f in (fa,fb):f.ts=pd.to_datetime(f.ts).dt.tz_convert('Asia/Seoul')
    verified={};comparison=[]
    for fold,c in contract['folds'].items():
        report=summary['folds'][fold]
        rows=pd.read_parquet(folder/fold/'site_predictions.parquet')
        baseline=pd.read_parquet(BASE/fold/'site_predictions.parquet')
        base_summary=json.loads((BASE/fold/'summary.json').read_text())
        start=pd.Timestamp(c['eval_start']);end=pd.Timestamp(c['eval_end_exclusive'])
        raw=gen.loc[(gen.ts>=start)&(gen.ts<end)]
        assert len(rows)==len(raw)==c['eval_raw_rows'] and not rows.duplicated(['site_id','ts']).any()
        assert rows[['site_id','ts']].reset_index(drop=True).equals(raw[['site_id','ts']].reset_index(drop=True))
        assert np.array_equal(rows.energy_raw.to_numpy(),raw.energy.to_numpy())
        cap=raw.site_id.map(sites.set_index('site_id').capacity_kw).to_numpy('float32').astype(float)
        energy=raw.energy.to_numpy(float)
        has=np.isfinite(energy)&(energy>=0)&(energy<=1.05*cap)
        label=np.where(has,energy,np.nan).astype('float32')
        assert np.array_equal(rows.has_peer.to_numpy(bool),has)
        np.testing.assert_allclose(rows.energy_peer,label,rtol=0,atol=0,equal_nan=True)
        for name in ['site_id','ts','energy_raw','energy_peer','has_peer','capacity_kw']:
            assert np.array_equal(rows[name].to_numpy(),baseline[name].to_numpy(),equal_nan=True) if name not in ['site_id','ts'] else np.array_equal(rows[name].to_numpy(),baseline[name].to_numpy())
        keys=rows.sort_values(['site_id','ts']).reset_index(drop=True)[['site_id','ts','has_peer']]
        digest=hashlib.sha256(keys.to_csv(index=False,lineterminator='\n').encode()).hexdigest()
        assert digest==c['evaluation_key_mask_sha256']
        assert np.isfinite(rows[['pred_10','pred_17','ensemble']]).all().all()
        assert (rows[['pred_10','pred_17','ensemble']]>=0).all().all()
        assert (rows.loc[~rows.daylight,['pred_10','pred_17','ensemble']]==0).all().all()
        np.testing.assert_allclose(rows.ensemble,0.25*rows.pred_10+0.75*rows.pred_17,rtol=1e-14)
        results={}
        for prediction in ['pred_10','pred_17','ensemble']:
            computed=numpy_score(rows,sites,prices,start,end,prediction)
            stored=report['metrics'][prediction]
            close(computed['W'],stored['W'])
            for metric in ['W_by_group','W_by_month']:
                for key,value in computed[metric].items():close(value,stored[metric][key])
            results[prediction]=computed
        # Baseline metrics recomputed from the saved, previously validated predictions; no baseline fitting.
        base_metrics=numpy_score(baseline,sites,prices,start,end,'ensemble');close(base_metrics['W'],c['baseline_W'])
        for kind in ['W_by_group','W_by_month']:
            for key,value in results['ensemble'][kind].items():
                ref=base_metrics[kind][key]
                comparison.append(dict(fold=fold,level=kind,key=key,P4_W=ref,B1_W=value,change_percent=(value/ref-1)*100))
        indices=np.unique(np.linspace(0,len(rows)-1,1024).astype(int))
        sample=raw.iloc[indices].copy()
        replay={}
        for r in [10,17]:
            f,columns=build(b,sample,sites,fa,fb,r,'station_distance');f=attach(f,gefs)
            model=lgb.Booster(model_file=str(folder/fold/f'model_round_{r}.txt'))
            assert model.num_trees()==200 and model.feature_name()==columns+COLUMNS
            assert model.params['objective']=='regression_l1'
            prediction=np.maximum(np.where(f.elev>0,model.predict(f[columns+COLUMNS].astype('float32')),0),0)*f.cap.to_numpy()
            np.testing.assert_allclose(prediction,rows.iloc[indices][f'pred_{r}'],rtol=1e-12,atol=1e-10)
            mi=report['models'][str(r)];bm=base_summary['models'][str(r)]
            assert mi['train_rows']==bm['train_rows'] and mi['weights']==bm['weights']
            assert mi['train_min']==bm['train_min'] and mi['train_max']==bm['train_max']
            replay[str(r)]=len(indices)
        verified[fold]=dict(status='passed',evaluation_key_mask_sha256=digest,independent_W=results['ensemble']['W'],
              baseline_W=base_metrics['W'],change_percent=(results['ensemble']['W']/base_metrics['W']-1)*100,model_replay_rows=replay,
              eval_rows=len(rows),observed_site_hours=int(has.sum()),evaluation_hours=len(pd.date_range(start,end,freq='h',inclusive='left')))
        print(f'{fold}: labels/masks/NumPy W/model replay passed',flush=True)
    pd.DataFrame(comparison).to_csv(folder/'monthly_group_comparison.csv',index=False)
    result=dict(status='passed',folds=verified,all_cached_days_verified=cache_days,gefs_hours=35064,
                seconds=time.perf_counter()-started,publication_evidence='3-day837-file sample plus conservative24h lag; not all historical publication times audited')
    (folder/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    summary['status']='complete';summary['verification']=result
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    status('complete','B1 annual comparison and independent verification passed',str(folder.parent/(folder.name+'.verify.log')),result_dir=str(folder),folds=verified)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
