"""Independent raw cohort reconstruction and bootstrap replay for fixed A5."""
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
import train_smoke as original
from annual_neural_data import check_contract,key_mask_digest
from run_peer_w import TZ,bounds,write_json

ROOT=Path(__file__).resolve().parent


def quantile(values,q):
    v=sorted(values);p=(len(v)-1)*q;i=math.floor(p)
    return v[i]+(p-i)*(v[min(i+1,len(v)-1)]-v[i])


def replay(base,candidate,entry):
    assert entry['replicates']==5000
    if base.sum()<=0:
        assert entry['valid_replicates']==0 and entry['ci95_percent'] is None;return
    n=len(base);block=entry['block_days'];reps=entry['replicates']
    starts=np.random.default_rng(entry['seed']).integers(0,n,size=(reps,math.ceil(n/block)))
    samples=[]
    for row in starts:
        ids=np.concatenate([(int(s)+np.arange(block))%n for s in row])[:n]
        # Count repeated days rather than gather the same two series as the runner.
        counts=np.bincount(ids,minlength=n)
        b=float(np.sum(base*counts));c=float(np.sum(candidate*counts))
        if b>0:samples.append(100*(1-c/b))
    assert len(samples)==entry['valid_replicates']
    if len(samples)<.99*reps:assert entry['ci95_percent'] is None;return
    np.testing.assert_allclose([quantile(samples,.025),quantile(samples,.975)],entry['ci95_percent'],rtol=0,atol=1e-9)
    np.testing.assert_allclose(sum(x>0 for x in samples)/len(samples),entry['fraction_improved'],rtol=0,atol=1e-12)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--a1-dir',type=Path,required=True)
    p.add_argument('--subset-dir',type=Path,required=True);args=p.parse_args()
    folder=args.output_dir;r=json.loads((folder/'summary.json').read_text());assert r['status']=='computed_pending_verification'
    check_contract()
    for name,h in r['source_sha256'].items():
        assert original.sha256(ROOT/name)==h
        assert original.sha256(folder/'source'/name)==h
    assert original.sha256(args.source_dir/'summary.json')==r['source_summary_sha256']
    assert original.sha256(args.source_dir/'selection.json')==r['selection_sha256']
    assert original.sha256(args.subset_dir/'manifest.json')==r['subset_manifest_sha256']
    assert r['fixed_weights']==[.5,.2,.3] and r['retrained_models']==0 and not r['reselected_weights']
    manifest=json.loads((args.subset_dir/'manifest.json').read_text())
    sites=pd.read_csv(original.DATA/'sites.csv')
    gen=pd.read_parquet(original.DATA/'train/generation.parquet');gen.ts=pd.to_datetime(gen.ts).dt.tz_convert(TZ)
    cap=gen.site_id.map(sites.set_index('site_id').capacity_kw).to_numpy(dtype='float32').astype(float)
    raw=gen.energy.to_numpy(dtype=float);valid=np.isfinite(raw)&(raw>=0)&(raw<=1.05*cap)
    result=dict(status='running',source_hashes_verified=True,all_bootstrap_replicates_replayed=True,folds={})
    for fi,fold in enumerate(['F2','F3']):
        _,_,c,d=bounds(fold);axis=pd.date_range(c,d,freq='h',inclusive='left');hours=len(axis);days=hours//24
        index=pd.MultiIndex.from_product([['G1','G2'],axis],names=['group','ts'])
        meta=manifest['folds'][fold];assert original.sha256(args.subset_dir/meta['file'])==meta['sha256']
        p4=pd.DataFrame(json.loads((args.subset_dir/meta['file']).read_text()));p4.ts=pd.to_datetime(p4.ts,utc=True).dt.tz_convert(TZ)
        assert len(p4)==len(index) and not p4.duplicated(['group','ts']).any()
        p4=p4.set_index(['group','ts']).reindex(index);assert p4.weight.notna().all()
        saved=np.load(folder/f'{fold}_arrays.npz');np.testing.assert_array_equal(saved['weight'],p4.weight)
        errors={s:[p4[f'error_{s}'].to_numpy()] for s in ['all','seen','new']}
        history=set(gen.loc[(gen.ts<c)&valid,'site_id'])
        for family in ['tabm','tft']:
            path=args.a1_dir/family/fold/'site_predictions.parquet'
            assert original.sha256(path)==r['folds'][fold]['input_prediction_sha256'][family]
            frame=pd.read_parquet(path)
            np.testing.assert_array_equal(frame.seen,frame.site_id.isin(history))
            assert key_mask_digest(frame)==meta['evaluation_key_mask_sha256']
            group=frame.site_id.map(sites.set_index('site_id').current_group).map({'A':0,'B':1}).to_numpy(dtype=int)
            ts=axis.get_indexer(frame.ts);assert (ts>=0).all()
            values=frame.ensemble.to_numpy()-frame.energy_peer.to_numpy(dtype=float)
            for subset in ['all','seen','new']:
                use=frame.has_peer.to_numpy(bool).copy()
                if subset!='all':use &= frame.site_id.isin(history).to_numpy() if subset=='seen' else ~frame.site_id.isin(history).to_numpy()
                arr=np.zeros((2,hours));count=np.zeros((2,hours),dtype=int)
                np.add.at(arr,(group[use],ts[use]),values[use]);np.add.at(count,(group[use],ts[use]),1)
                np.testing.assert_array_equal(count.ravel(),p4[f'observed_{subset}'])
                errors[subset].append(arr.ravel())
        verified={}
        for subset in ['all','seen','new']:
            e=np.column_stack(errors[subset]);np.testing.assert_allclose(e,saved[f'errors_{subset}'],rtol=1e-12,atol=1e-8)
            mixed=e[:,0]*.5+e[:,1]*.2+e[:,2]*.3
            daily=[]
            for value in [e[:,0],mixed]:
                arr=np.zeros((2,days));amount=np.abs(value)*saved['weight']
                np.add.at(arr,(np.repeat(np.arange(2),hours),np.tile(np.arange(hours)//24,2)),amount)
                daily.append(arr)
            for key,value in zip(['base','candidate'],daily):np.testing.assert_allclose(value,saved[f'{key}_{subset}'],rtol=1e-12,atol=1e-7)
            for scope in ['total','G1','G2']:
                i={'G1':0,'G2':1}.get(scope)
                b=daily[0].sum(axis=0) if i is None else daily[0][i]
                q=daily[1].sum(axis=0) if i is None else daily[1][i]
                entry=r['folds'][fold]['subsets'][subset][scope]
                np.testing.assert_allclose([b.sum(),q.sum()],[entry['baseline_W'],entry['candidate_W']],rtol=1e-12,atol=1e-5)
                point=100*(1-q.sum()/b.sum()) if b.sum()>0 else None
                if point is None:assert entry['point_improvement_percent'] is None
                else:np.testing.assert_allclose(point,entry['point_improvement_percent'],rtol=0,atol=1e-10)
                for block in [7,28]:
                    boot=entry['bootstrap'][str(block)];assert boot['seed']==20261105+1000*fi+block
                    replay(b,q,boot)
            verified[subset]=dict(original_history_and_cohort_counts_verified=True,all_bootstrap_replicates_match=True)
            print(f'{fold}/{subset}: raw cohort and all 7/28-day bootstrap replicates verified',flush=True)
        # Independent sensitivity list; it is diagnostic and cannot change the candidate.
        policies={}
        for i in range(3):
            for j in range(3):
                if i!=j:
                    w=[.5,.2,.3];w[i]-=.05;w[j]+=.05;policies[f'move_05_{i}_to_{j}']=w
        for i in range(3):
            w=[.5,.2,.3];w[i]=0.;total=sum(w);policies[f'exclude_{i}']=[x/total for x in w]
        assert set(policies)==set(r['folds'][fold]['sensitivity_diagnostic_only'])
        e=saved['errors_all'];weight=saved['weight']
        for name,w in policies.items():
            entry=r['folds'][fold]['sensitivity_diagnostic_only'][name]
            np.testing.assert_allclose(w,entry['weights'],rtol=0,atol=1e-12)
            value=e[:,0]*w[0]+e[:,1]*w[1]+e[:,2]*w[2]
            W=math.fsum(float(x) for x in np.abs(value)*weight)
            np.testing.assert_allclose(W,entry['W'],rtol=1e-12,atol=1e-5)
        b=saved['base_all'].sum(axis=0);q=saved['candidate_all'].sum(axis=0)
        months=[(c+pd.Timedelta(days=i)-pd.Timedelta(hours=1)).strftime('%Y-%m') for i in range(days)]
        for month,point in r['folds'][fold]['leave_one_month_out_improvement_percent'].items():
            keep=np.array([m!=month for m in months]);np.testing.assert_allclose(100*(1-q[keep].sum()/b[keep].sum()),point,rtol=0,atol=1e-10)
        worst=sorted(range(days),key=lambda i:(-b[i],i))[:math.ceil(.1*days)]
        np.testing.assert_allclose(100*(1-q[worst].sum()/b[worst].sum()),r['folds'][fold]['worst_baseline_10pct_days']['improvement_percent'],rtol=0,atol=1e-10)
        result['folds'][fold]=verified
    f3=r['folds']['F3']['subsets'];reasons=[]
    for block in [7,28]:
        ci=f3['all']['total']['bootstrap'][str(block)]['ci95_percent']
        if ci is None or ci[0]<=0:reasons.append(f'F3 overall {block}-day interval includes non-improvement')
    if f3['seen']['total']['point_improvement_percent'] is None or f3['seen']['total']['point_improvement_percent']<0:reasons.append('F3 previously observed subset not improved')
    for g in ['G1','G2']:
        if f3['all'][g]['point_improvement_percent'] is None or f3['all'][g]['point_improvement_percent']<0:reasons.append(f'F3 {g} not improved')
    expected='mixed_historical_support' if reasons else 'strong_historical_support'
    if f3['all']['total']['point_improvement_percent'] is None or f3['all']['total']['point_improvement_percent']<=0:expected='no_historical_support'
    assert r['support']['classification']==expected
    assert not r['support']['independent_final_test'] and not r['support']['automatic_model_replacement']
    assert original.sha256(args.source_dir/'selection.json')==r['selection_sha256']
    original.check_data();result.update(status='complete',support_classification_verified=expected)
    write_json(folder/'verification.json',result)
    print('A5 independent verification complete',flush=True)


if __name__=='__main__':main()
