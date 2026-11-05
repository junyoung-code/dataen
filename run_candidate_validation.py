"""A5 fixed blend: paired block bootstrap, cohort risks and fixed sensitivity."""
import argparse,json,platform,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import train_smoke as original
from annual_neural_data import check_contract,key_mask_digest
from export_candidate_subsets import mask_hash
from candidate_validation import WEIGHTS,BLOCKS,REPLICATES,SEED,SUBSETS,SCOPES,mixed,daily_amounts,improvement,block_bootstrap,perturbations,support
from run_peer_w import TZ,bounds,write_json

ROOT=Path(__file__).resolve().parent


def align(frame,index):
    frame=frame.copy();frame.ts=pd.to_datetime(frame.ts,utc=True).dt.tz_convert(TZ)
    keys=pd.MultiIndex.from_frame(frame[['group','ts']])
    if len(keys)!=len(index) or keys.duplicated().any() or not keys.isin(index).all():raise ValueError('Aggregate keys changed')
    return frame.set_index(['group','ts']).reindex(index).reset_index()


def cohorts(fold,args,contract,manifest,source,sites):
    _,_,c,d=bounds(fold);axis=pd.date_range(c,d,freq='h',inclusive='left')
    index=pd.MultiIndex.from_product([['G1','G2'],axis],names=['group','ts'])
    meta=manifest['folds'][fold];path=args.subset_dir/meta['file']
    assert original.sha256(path)==meta['sha256']
    base=align(pd.DataFrame(json.loads(path.read_text())),index)
    saved=align(pd.read_parquet(args.source_dir/f'{fold}_team_errors.parquet'),index)
    np.testing.assert_allclose(base.weight,saved.weight,rtol=0,atol=1e-9)
    np.testing.assert_allclose(base.error_all,saved.error_p4,rtol=1e-12,atol=1e-8)
    matrices={s:[base[f'error_{s}'].to_numpy()] for s in SUBSETS}
    mapping=sites.set_index('site_id').current_group.map({'A':'G1','B':'G2'})
    inputs={}
    for family in ['tabm','tft']:
        path=args.a1_dir/family/fold/'site_predictions.parquet'
        h=original.sha256(path);assert h==source['folds'][fold]['input_sources'][family]['site_predictions_sha256']
        rows=pd.read_parquet(path)
        assert key_mask_digest(rows)==contract['folds'][fold]['evaluation_key_mask_sha256']
        assert mask_hash(rows)==meta['key_seen_sha256']
        rows['group']=rows.site_id.map(mapping);rows['error']=rows.ensemble-rows.energy_peer.astype(float)
        for subset in SUBSETS:
            keep=rows.has_peer.to_numpy(bool).copy()
            if subset!='all':keep &= rows.seen.to_numpy(bool) if subset=='seen' else ~rows.seen.to_numpy(bool)
            agg=rows.loc[keep].groupby(['group','ts']).agg(error=('error','sum'),observed=('site_id','size')).reindex(index,fill_value=0.)
            np.testing.assert_array_equal(agg.observed,base[f'observed_{subset}'])
            matrices[subset].append(agg.error.to_numpy())
        inputs[family]=h
    matrices={s:np.column_stack(x) for s,x in matrices.items()}
    np.testing.assert_allclose(matrices['all'],matrices['seen']+matrices['new'],rtol=1e-12,atol=1e-8)
    np.testing.assert_array_equal(base.observed_all,base.observed_seen+base.observed_new)
    return base,matrices,inputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--a1-dir',type=Path,required=True)
    p.add_argument('--subset-dir',type=Path,required=True);args=p.parse_args()
    args.output_dir.resolve().relative_to((ROOT/'outputs').resolve());args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();contract=check_contract();original.check_data()
    source=json.loads((args.source_dir/'summary.json').read_text());assert source['status']=='complete'
    assert original.sha256(args.source_dir/'verification.json')==source['verification_sha256']
    assert original.sha256(args.source_dir/'selection.json')==source['selection_sha256']
    assert source['selection']['policies']['a3_global']['fallback']==WEIGHTS.tolist()
    for name,h in source['source_sha256'].items():assert original.sha256(ROOT/name)==h
    manifest=json.loads((args.subset_dir/'manifest.json').read_text())
    assert manifest['contract_sha256']==original.sha256(ROOT/'AB_EVALUATION_CONTRACT.json')
    names=['candidate_validation.py','export_candidate_subsets.py','run_candidate_validation.py','verify_candidate_validation.py','test_candidate_validation.py','launch_a5.sh']
    hashes={n:original.sha256(ROOT/n) for n in names};snapshot=args.output_dir/'source';snapshot.mkdir()
    for n in names:(snapshot/n).write_bytes((ROOT/n).read_bytes())
    report=dict(status='running',experiment='A5',fixed_weights=WEIGHTS.tolist(),model_order=['p4','tabm','tft'],
        source_sha256=hashes,source_summary_sha256=original.sha256(args.source_dir/'summary.json'),
        selection_sha256=source['selection_sha256'],subset_manifest_sha256=original.sha256(args.subset_dir/'manifest.json'),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        method=dict(blocks_days=BLOCKS,replicates=REPLICATES,base_seed=SEED,ci='95% percentile',
            seed_rule='base_seed+1000*fold_index+block_days; same resampling for all cohorts/groups',
            definition='paired circular moving blocks, same day indices for P4/candidate',
            limitations='historical only; seasonality and model-selection optimism not removed'),
        retrained_models=0,reselected_weights=False,independent_final_test=False,folds={})
    sites=pd.read_csv(original.DATA/'sites.csv')
    for fold_index,fold in enumerate(['F2','F3']):
        base,matrices,inputs=cohorts(fold,args,contract,manifest,source,sites)
        price=base.weight.to_numpy();ndays=len(base)//48
        stats={};daily={};arrays=dict(weight=price)
        for subset in SUBSETS:
            errors=matrices[subset];candidate=mixed(errors)
            b=daily_amounts(errors[:,0],price);c=daily_amounts(candidate,price)
            arrays[f'errors_{subset}']=errors;arrays[f'base_{subset}']=b;arrays[f'candidate_{subset}']=c
            stats[subset]={}
            for scope in SCOPES:
                i={'G1':0,'G2':1}.get(scope)
                bs=b.sum(axis=0) if i is None else b[i]
                cs=c.sum(axis=0) if i is None else c[i]
                stats[subset][scope]=dict(baseline_W=float(bs.sum()),candidate_W=float(cs.sum()),
                    point_improvement_percent=improvement(bs,cs),bootstrap={str(block):block_bootstrap(bs,cs,block,SEED+1000*fold_index+block) for block in BLOCKS})
            print(f'{fold}/{subset}: W={c.sum():.6f}; improvement={stats[subset]["total"]["point_improvement_percent"]}',flush=True)
        for name,target in [('p4','baseline_W'),('a3_global','candidate_W')]:
            np.testing.assert_allclose(stats['all']['total'][target],source['folds'][fold]['metrics'][name]['W'],rtol=1e-12,atol=1e-5)
        sensitivity={}
        for name,w in perturbations().items():
            value=float(np.sum(np.abs(mixed(matrices['all'],w))*price))
            sensitivity[name]=dict(weights=w,W=value,change_percent_vs_fixed=100*(value/stats['all']['total']['candidate_W']-1),
                improvement_percent_vs_p4=100*(1-value/stats['all']['total']['baseline_W']))
        days=pd.DatetimeIndex(base.ts.iloc[:len(base)//2:24])-pd.Timedelta(hours=1)
        month=days.strftime('%Y-%m');b=arrays['base_all'].sum(axis=0);c=arrays['candidate_all'].sum(axis=0)
        excluded={m:improvement(b[month!=m],c[month!=m]) for m in sorted(set(month))}
        worst=np.argsort(-b,kind='stable')[:int(np.ceil(.1*ndays))]
        result=dict(subsets=stats,sensitivity_diagnostic_only=sensitivity,leave_one_month_out_improvement_percent=excluded,
            worst_baseline_10pct_days=dict(days=len(worst),improvement_percent=improvement(b[worst],c[worst])),
            observed_site_hours={s:int(base[f'observed_{s}'].sum()) for s in SUBSETS},
            days=ndays,input_prediction_sha256=inputs,subset_key_sha256=manifest['folds'][fold]['key_seen_sha256'])
        report['folds'][fold]=result
        np.savez_compressed(args.output_dir/f'{fold}_arrays.npz',**arrays)
        write_json(args.output_dir/'summary.json',report)
    report['support']=support({s:x for s,x in report['folds']['F3']['subsets'].items()})
    assert all(original.sha256(ROOT/n)==h for n,h in hashes.items())
    assert original.sha256(args.source_dir/'selection.json')==source['selection_sha256']
    report.update(status='computed_pending_verification',total_seconds=time.perf_counter()-started,peak_process_memory_mib=original.peak_memory_mib())
    write_json(args.output_dir/'summary.json',report)
    print('A5 computed; historical support:',report['support'],flush=True)


if __name__=='__main__':main()
