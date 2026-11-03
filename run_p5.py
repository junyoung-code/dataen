"""Freeze a tune-only team allocation, then compare it once on historical confirmation."""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import train_smoke as original
import validation_protocol as protocol
import evaluate_temporal as evaluation
from run_p1 import write_json,single_group_weighted
from team_assignment import optimize
ROOT=Path(__file__).resolve().parent


def check_assignment(assignment,sites):
    if assignment.site_id.duplicated().any() or set(assignment.site_id)!=set(sites.site_id):
        raise ValueError('Each of the 253 source sites must be assigned exactly once')
    if not assignment.group.isin(['G1','G2']).all():raise ValueError('Only G1/G2 permitted')
    combined=sites[['site_id','capacity_kw']].merge(assignment,on='site_id',validate='one_to_one')
    capacities=combined.groupby('group').capacity_kw.sum().reindex(['G1','G2'])
    if capacities.isna().any() or (capacities<20000).any():raise ValueError('Each group needs at least 20000kW')
    return {g:dict(capacity_kw=float(capacities[g]),sites=int((combined.group==g).sum())) for g in capacities.index}


def load_window(path,window,gen,sites,folder):
    frame=pd.read_parquet(path)
    _,valid=protocol.split(gen,window)
    raw=valid.set_index(['site_id','ts']).energy.sort_index()
    saved=frame.set_index(['site_id','ts']).energy.sort_index()
    if not raw.index.equals(saved.index) or not np.array_equal(raw.to_numpy(),saved.to_numpy()):
        raise ValueError('Stored predictions differ from raw observation universe')
    if frame.duplicated(['site_id','ts']).any() or not np.isfinite(frame.ensemble).all() or (frame.ensemble<0).any():
        raise ValueError('Invalid saved prediction')
    np.testing.assert_allclose(frame.ensemble,.25*frame.pred_10+.75*frame.pred_17,rtol=1e-12)
    hours=protocol.complete_hours(valid,sites.site_id)
    folder.mkdir(parents=True,exist_ok=False)
    pd.DataFrame({'ts':hours.astype(str)}).to_csv(folder/'hours.csv',index=False)
    expected=json.loads((ROOT/'docs/experiments/2026-11-03-p0-coverage.json').read_text())['windows'][window]
    if protocol.digest(folder/'hours.csv')!=expected['mask_sha256']:raise ValueError('P0 mask changed')
    return frame,hours


def evaluate(frame,hours,assignment,sites,prices,folder,name):
    metadata=sites[['site_id','capacity_kw']].merge(assignment,on='site_id',validate='one_to_one')
    metadata['current_group']=metadata.group.map({'G1':'A','G2':'B'})
    groups=evaluation.aggregate_groups(frame,metadata,['ensemble'])
    groups=groups.loc[groups.ts.isin(hours)].merge(prices,on='ts',how='left',validate='many_to_one')
    score=protocol.weighted_group_mae(groups,hours,'ensemble')
    groups['month']=(groups.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')
    metric=lambda f:evaluation.metrics(f.energy,f.ensemble,f.capacity_kw)
    monthly=[]
    for month,f in groups.groupby('month'):
        monthly.append(dict(month=month,group='both',hours=int(f.ts.nunique()),weighted_group_mae=single_group_weighted(f,'ensemble'),**metric(f)))
        for g,z in f.groupby('group'):
            monthly.append(dict(month=month,group=g,hours=len(z),weighted_group_mae=single_group_weighted(z,'ensemble'),**metric(z)))
    groups.to_parquet(folder/f'{name}_group_predictions.parquet',index=False)
    return dict(weighted_group_mae=score,group_metrics=metric(groups),common_hours=len(hours),
                by_group={g:dict(weighted_group_mae=single_group_weighted(z,'ensemble'),**metric(z)) for g,z in groups.groupby('group')},
                monthly=monthly)


def run(source,folder,threads):
    started=time.perf_counter();original.check_data()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit code before running')
    source=source.resolve();source.relative_to((ROOT/'outputs').resolve())
    p4=json.loads((source/'summary.json').read_text())
    if p4['status']!='complete' or p4['selected']['variant']!='l1_capacity':raise ValueError('Frozen P4 selection required')
    sites=pd.read_csv(original.DATA/'sites.csv').sort_values('site_id').reset_index(drop=True)
    if len(sites)!=253:raise ValueError('Expected 253 sites')
    reference=sites[['site_id']].copy();reference['group']=sites.current_group.map({'A':'G1','B':'G2'})
    ref_stats=check_assignment(reference,sites)
    gen=pd.read_parquet(original.DATA/'train/generation.parquet',columns=['site_id','ts','energy'])
    gen.ts=pd.to_datetime(gen.ts).dt.tz_convert(protocol.TZ)
    prices=pd.read_csv(original.DATA/'train/smp.csv');prices.ts=pd.to_datetime(prices.ts).dt.tz_convert(protocol.TZ)
    report=dict(status='running',protocol=protocol.PROTOCOL,code_commit=commit,dirty=False,threads=threads,
                source_run=source.name,source_code_commit=p4['code_commit'],confirmation_evaluated=False,
                versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),evaluations={})
    write_json(folder/'summary.json',report)
    try:
        tune_path=source/'tune/l1_capacity/site_predictions.parquet'
        frame,hours=load_window(tune_path,'tune',gen,sites,folder/'tune')
        base=evaluate(frame,hours,reference,sites,prices,folder/'tune','reference')
        np.testing.assert_allclose(base['weighted_group_mae'],p4['selected']['weighted_group_mae'],rtol=1e-12)
        frame['residual']=frame.ensemble-frame.energy
        residual=frame.loc[frame.ts.isin(hours)].pivot(index='ts',columns='site_id',values='residual').reindex(index=hours,columns=sites.site_id).to_numpy()
        weight=prices.set_index('ts').smp.reindex(hours).to_numpy()*np.where(hours.hour.isin([11,12,13,14,15]),3,1)
        initial=reference.group.eq('G1').to_numpy()
        callback=lambda row:print(f"search step {row['step']}: score={row['weighted_group_mae']:.6f}, {row['kind']}, capacities={row['capacity_g1']:.2f}/{row['capacity_g2']:.2f}",flush=True)
        chosen,search=optimize(residual,sites.capacity_kw.to_numpy(),weight,initial,max_steps=50,callback=callback)
        assignment=sites[['site_id']].copy();assignment['group']=np.where(chosen,'G1','G2')
        chosen_stats=check_assignment(assignment,sites)
        reference.to_csv(folder/'reference_assignment.csv',index=False)
        assignment.to_csv(folder/'optimized_assignment.csv',index=False)
        frozen_hash=protocol.digest(folder/'optimized_assignment.csv')
        selection=dict(selected_using='tune only',assignment_sha256=frozen_hash,code_commit=commit,
                       model=dict(variant='l1_capacity',trees=200,learning_rate=0.03,weight_17=0.75,feature_bundle='station_distance'),
                       search=search,changed_sites=int(np.sum(initial!=chosen)),reference_groups=ref_stats,optimized_groups=chosen_stats,
                       source_tune_predictions_sha256=protocol.digest(tune_path),confirmation_seen_at_selection=False)
        write_json(folder/'selection.json',selection)
        report['selection']=selection
        optimized=evaluate(frame,hours,assignment,sites,prices,folder/'tune','optimized')
        np.testing.assert_allclose(optimized['weighted_group_mae'],search['final_score'],rtol=1e-12)
        report['evaluations']['tune']=dict(reference=base,optimized=optimized)
        write_json(folder/'summary.json',report)
        print(f"Assignment frozen: {selection['changed_sites']} sites changed; SHA256={frozen_hash}",flush=True)
        # No search after this point, including when diagnostic/confirmation performance worsens.
        frame,hours=load_window(source/'seasonal_diagnostic/l1_capacity/site_predictions.parquet','seasonal_diagnostic',gen,sites,folder/'seasonal_diagnostic')
        report['evaluations']['seasonal_diagnostic']={name:evaluate(frame,hours,comp,sites,prices,folder/'seasonal_diagnostic',name) for name,comp in [('reference',reference),('optimized',assignment)]}
        print('Retraining only the frozen P4 model for March-May historical confirmation',flush=True)
        subprocess.run([sys.executable,str(ROOT/'confirm_p5.py'),'--output-dir',str(folder/'confirmation_model'),
                        '--selection-file',str(folder/'selection.json'),'--threads',str(threads)],check=True,cwd=ROOT)
        frame,hours=load_window(folder/'confirmation_model/site_predictions.parquet','confirmation',gen,sites,folder/'confirmation')
        report['evaluations']['confirmation']={name:evaluate(frame,hours,comp,sites,prices,folder/'confirmation',name) for name,comp in [('reference',reference),('optimized',assignment)]}
        report['confirmation_model']=json.loads((folder/'confirmation_model/summary.json').read_text())
        if frozen_hash!=protocol.digest(folder/'optimized_assignment.csv'):raise ValueError('Frozen assignment changed')
        report.update(status='complete',confirmation_evaluated=True,assignment_unchanged_after_confirmation=True)
    except BaseException as exc:
        report['status']='failed';report['error']=str(exc);raise
    finally:
        report['total_seconds']=time.perf_counter()-started
        report['peak_parent_memory_mib']=original.peak_memory_mib()
        write_json(folder/'summary.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args();folder=args.output_dir.resolve();folder.relative_to((ROOT/'outputs').resolve())
    if args.threads<1:parser.error('Positive threads required')
    folder.mkdir(parents=True,exist_ok=False)
    run(args.source_dir,folder,args.threads)
