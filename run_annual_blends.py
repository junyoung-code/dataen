"""A2 diagnosis, A3 frozen global blend and A4 forecast/clock conditional blend."""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd

import train_smoke as original
from annual_neural_data import check_contract, key_mask_digest
from annual_blend_core import MODELS, RULES, apply, coefficients, fit_policy, grid_fit, select_conditional
from run_peer_w import TZ, bounds, write_json

ROOT=Path(__file__).resolve().parent


def digest(path):return original.sha256(Path(path))


def aligned(frame,axis):
    frame=frame.copy();frame.ts=pd.to_datetime(frame.ts).dt.tz_convert(TZ)
    index=pd.MultiIndex.from_product([['G1','G2'],axis],names=['group','ts'])
    if frame.duplicated(['group','ts']).any():raise ValueError('Duplicate aggregate keys')
    if len(frame)!=len(index) or not pd.MultiIndex.from_frame(frame[['group','ts']]).isin(index).all():
        raise ValueError('Aggregate row universe changed')
    result=frame.set_index(['group','ts']).reindex(index)
    if result[['error','weight','observed_sites']].isna().any().any():raise ValueError('Missing team-hours')
    return result.reset_index()


def forecast_condition(sites,forecast,axis):
    # National summary across static capacities; independent of generation/mask.
    table=forecast.loc[(forecast['round']==17)&forecast.ts.isin(axis),['wa_station','ts','wa_cloud']].copy()
    if table.duplicated(['wa_station','ts']).any():raise ValueError('Duplicate provided forecast keys')
    capacity=sites.groupby('wa_station').capacity_kw.sum()
    table['capacity']=table.wa_station.map(capacity).fillna(0.)
    table.loc[~np.isfinite(table.wa_cloud),'capacity']=0.
    table['numerator']=table.wa_cloud.fillna(0.)*table.capacity
    agg=table.groupby('ts')[['numerator','capacity']].sum().reindex(axis,fill_value=0.)
    cloud=np.divide(agg.numerator,agg.capacity,out=np.full(len(axis),np.nan),where=agg.capacity.to_numpy()>0)
    missing=1-agg.capacity.to_numpy()/float(sites.capacity_kw.sum())
    if (missing < -1e-10).any() or (missing>1+1e-10).any():raise ValueError('Invalid forecast availability')
    # Stable physical threshold across equivalent station/site summation orders.
    return np.round(np.asarray(cloud),10),np.clip(missing,0,1)


def metrics(error,weights,groups,timestamps,observed):
    amounts=np.abs(error)*weights
    months=(pd.DatetimeIndex(timestamps)-pd.Timedelta(hours=1)).strftime('%Y-%m').to_numpy()
    return dict(W=float(amounts.sum()),W_1e8=float(amounts.sum()/1e8),
        W_by_group={g:float(amounts[groups==g].sum()) for g in ['G1','G2']},
        W_by_month={m:float(amounts[months==m].sum()) for m in sorted(set(months))},
        observed_site_hours=int(observed.sum()),team_hours=len(error),weight_sum=float(weights.sum()))


def diagnostic(errors,weights,hours,cloud,missing,groups,timestamps):
    conditions=dict(hour=hours,cloud=np.where(~np.isfinite(cloud),'missing',np.where(cloud<25,'below25',np.where(cloud<75,'25to75','75plus'))),
        forecast_availability=np.where(missing>0,'partly_missing','complete'),group=groups)
    result=dict(by_condition={},weighted_error_correlation={},opposite_sign_weight_fraction={})
    active=np.any(errors!=0,axis=1)&(weights>0)
    for name,labels in conditions.items():
        result['by_condition'][name]={str(label):dict(team_hours=int(np.sum(labels==label)),
            W={model:float(np.sum(np.abs(errors[labels==label,i])*weights[labels==label])) for i,model in enumerate(MODELS)}) for label in sorted(set(labels))}
    for i,first in enumerate(MODELS):
        for j,second in enumerate(MODELS):
            if j<=i:continue
            x,y=errors[active,i],errors[active,j];w=weights[active]
            mx=np.average(x,weights=w);my=np.average(y,weights=w)
            den=np.sqrt(np.average((x-mx)**2,weights=w)*np.average((y-my)**2,weights=w))
            key=f'{first}/{second}'
            result['weighted_error_correlation'][key]=float(np.average((x-mx)*(y-my),weights=w)/den) if den>0 else None
            result['opposite_sign_weight_fraction'][key]=float(w[(x*y)<0].sum()/w.sum())
    return result


def site_diagnostic(rows,median_capacity):
    frame=rows.loc[rows.has_peer&rows.daylight].copy()
    error=frame.ensemble-frame.energy_peer.astype(float)
    groups=dict(history=np.where(frame.seen,'seen','new'),capacity=np.where(frame.capacity_kw<=median_capacity,'small','large'))
    result={}
    for name,labels in groups.items():
        result[name]={str(label):dict(rows=int(np.sum(labels==label)),
            mae_kwh=float(np.abs(error[labels==label]).mean()),bias_kwh=float(error[labels==label].mean())) for label in sorted(set(labels))}
    return result


def load_fold(fold,args,contract,manifest,sites,forecast):
    _,_,c,d=bounds(fold);axis=pd.date_range(c,d,freq='h',inclusive='left')
    expected=contract['folds'][fold]
    info=manifest['folds'][fold];path=args.p4_aggregate_dir/info['file']
    assert digest(path)==info['sha256']
    data=json.loads(path.read_text())
    assert data['evaluation_key_mask_sha256']==expected['evaluation_key_mask_sha256']
    base=aligned(pd.DataFrame(data['rows']),axis)
    np.testing.assert_allclose(np.sum(np.abs(base.error)*base.weight),expected['baseline_W'],rtol=1e-12)
    assert int(base.observed_sites.sum())==expected['observed_site_hours']
    errors=[base.error.to_numpy()];source={};site_stats={}
    a1=json.loads((args.a1_dir/'summary.json').read_text())
    for family in ['tabm','tft']:
        folder=args.a1_dir/family/fold
        group_path=folder/'group_errors.csv';row_path=folder/'site_predictions.parquet'
        group=aligned(pd.read_csv(group_path),axis)
        np.testing.assert_array_equal(group.observed_sites,base.observed_sites)
        np.testing.assert_allclose(group.weight,base.weight,rtol=0,atol=1e-9)
        np.testing.assert_allclose(np.sum(np.abs(group.error)*group.weight),a1['results'][family][fold]['metrics']['ensemble']['W'],rtol=1e-12)
        rows=pd.read_parquet(row_path)
        assert key_mask_digest(rows)==expected['evaluation_key_mask_sha256']
        site_stats[family]=site_diagnostic(rows,float(sites.capacity_kw.median()))
        source[family]=dict(group_errors_sha256=digest(group_path),site_predictions_sha256=digest(row_path))
        errors.append(group.error.to_numpy());del rows
    cloud,missing=forecast_condition(sites,forecast,axis)
    return dict(errors=np.column_stack(errors),weights=base.weight.to_numpy(dtype=float),groups=base.group.to_numpy(),
        ts=base.ts,observed=base.observed_sites.to_numpy(),hours=base.ts.dt.hour.to_numpy(),
        cloud=np.tile(cloud,2),missing=np.tile(missing,2),sources=source,site_stats=site_stats)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--a1-dir',type=Path,required=True)
    p.add_argument('--p4-aggregate-dir',type=Path,required=True)
    args=p.parse_args()
    args.output_dir.resolve().relative_to((ROOT/'outputs').resolve())
    args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();contract=check_contract()
    a1=json.loads((args.a1_dir/'summary.json').read_text());assert a1['status']=='complete'
    for name,h in a1['verification_sha256'].items():
        assert digest(args.a1_dir/name)==h
        assert json.loads((args.a1_dir/name).read_text())['status']=='complete'
    original.check_data()
    manifest=json.loads((args.p4_aggregate_dir/'p4_manifest.json').read_text())
    assert manifest['contract_sha256']==digest(ROOT/'AB_EVALUATION_CONTRACT.json')
    names=['annual_blend_core.py','run_annual_blends.py','verify_annual_blends.py','test_annual_blends.py','export_p4_aggregates.py','launch_a234.sh']
    source=args.output_dir/'source';source.mkdir()
    hashes={name:digest(ROOT/name) for name in names}
    for name in names:(source/name).write_bytes((ROOT/name).read_bytes())
    report=dict(status='running',experiment='A2-A4',models=MODELS,source_sha256=hashes,
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        contract_sha256=digest(ROOT/'AB_EVALUATION_CONTRACT.json'),p4_manifest_sha256=digest(args.p4_aggregate_dir/'p4_manifest.json'),
        a1_summary_sha256=digest(args.a1_dir/'summary.json'),versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        historical_comparison_not_independent_final_test=True,model_retraining=False,folds={})
    write_json(args.output_dir/'summary.json',report)
    sites=pd.read_csv(original.DATA/'sites.csv')
    forecast=pd.read_parquet(original.DATA/'train/forecast_wa.parquet',columns=['wa_station','round','ts','wa_cloud'])
    forecast.ts=pd.to_datetime(forecast.ts).dt.tz_convert(TZ)
    f2=load_fold('F2',args,contract,manifest,sites,forecast)
    cutoff=pd.Timestamp('2023-12-01 01:00',tz=TZ)
    calibration=(f2['ts']<cutoff).to_numpy();validation=~calibration
    global_fit=grid_fit(f2['errors'],f2['weights'])
    conditional=select_conditional(f2['errors'],f2['weights'],f2['hours'],f2['cloud'],calibration,validation)
    policies=dict(a3_global=fit_policy(f2['errors'],f2['weights'],f2['hours'],f2['cloud'],'global'),
        a4_selected=fit_policy(f2['errors'],f2['weights'],f2['hours'],f2['cloud'],conditional['selected_rule']),
        a4_best_conditional_diagnostic=fit_policy(f2['errors'],f2['weights'],f2['hours'],f2['cloud'],conditional['best_conditional_rule']))
    selection=dict(fit_fold='F2',confirmation_fold='F3',model_order=MODELS,policies=policies,
        global_candidates=global_fit['candidates'],conditional_selection=conditional,
        calibration_start='2023-06-01 01:00:00+09:00',calibration_end_exclusive=str(cutoff),
        rule_validation_end_exclusive='2024-06-01 01:00:00+09:00',source_sha256=hashes,
        condition='17:00 provided WA forecast; static capacity weighted national cloud; threshold 50%; peak 11..15 KST; missing fallback global')
    write_json(args.output_dir/'selection.json',selection)
    selection_hash=digest(args.output_dir/'selection.json')
    print(f'F2 selection frozen before F3 load: global={policies["a3_global"]["fallback"]}; conditional={conditional["selected_rule"]}',flush=True)
    report['selection_sha256']=selection_hash;report['selection']=selection
    for fold,data in [('F2',f2),('F3',None)]:
        assert digest(args.output_dir/'selection.json')==selection_hash
        if data is None:data=load_fold(fold,args,contract,manifest,sites,forecast)
        result=dict(metrics={},input_sources=data['sources'],diagnostics=diagnostic(data['errors'],data['weights'],data['hours'],data['cloud'],data['missing'],data['groups'],data['ts']),
            site_daylight_diagnostics=data['site_stats'],evaluation_key_mask_sha256=contract['folds'][fold]['evaluation_key_mask_sha256'])
        outputs={name:data['errors'][:,i] for i,name in enumerate(MODELS)}
        outputs.update({name:apply(policy,data['errors'],data['hours'],data['cloud']) for name,policy in policies.items()})
        saved=pd.DataFrame(dict(group=data['groups'],ts=data['ts'],weight=data['weights'],observed_sites=data['observed'],hour=data['hours'],cloud=data['cloud'],missing_capacity_fraction=data['missing']))
        for name,error in outputs.items():
            saved[f'error_{name}']=error
            result['metrics'][name]=metrics(error,data['weights'],data['groups'],data['ts'],data['observed'])
            result['metrics'][name]['change_percent_vs_p4']=(result['metrics'][name]['W']/contract['folds'][fold]['baseline_W']-1)*100
        saved.to_parquet(args.output_dir/f'{fold}_team_errors.parquet',index=False)
        report['folds'][fold]=result
        write_json(args.output_dir/'summary.json',report)
        print(f'{fold}: A3 W={result["metrics"]["a3_global"]["W"]:.6f}; A4 W={result["metrics"]["a4_selected"]["W"]:.6f}',flush=True)
    assert all(digest(ROOT/name)==h for name,h in hashes.items())
    assert digest(args.output_dir/'selection.json')==selection_hash
    report.update(status='computed_pending_verification',total_seconds=time.perf_counter()-started,peak_process_memory_mib=original.peak_memory_mib())
    write_json(args.output_dir/'summary.json',report)


if __name__=='__main__':main()
