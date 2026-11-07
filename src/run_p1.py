"""Six fixed LightGBM configurations under p0-v1; confirmation is never evaluated."""
from __future__ import annotations
import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import evaluate_temporal as evaluation
import train_smoke as original
import validation_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [(n, lr) for n in (50, 200, 600) for lr in (0.03, 0.06)]
PREDICTIONS = ['pred_10', 'pred_17', 'ensemble']


def config_id(trees, rate):
    return f'trees{trees}_lr{rate:.2f}'


def choose(rows):
    if len(rows) != 6 or {(r['trees'], r['learning_rate']) for r in rows} != set(CONFIGS):
        raise ValueError('Expected all six distinct configurations')
    if not all(np.isfinite(r['weighted_group_mae']) for r in rows):
        raise ValueError('Missing/nonfinite selection score')
    best = min(r['weighted_group_mae'] for r in rows)
    tied = [r for r in rows if r['weighted_group_mae'] <= best + 1e-9]
    return min(tied, key=lambda r: (r['trees'], -r['learning_rate']))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def metric_table(frame):
    return {p: evaluation.metrics(frame.energy, frame[p], frame.capacity_kw) for p in PREDICTIONS}


def single_group_weighted(frame, prediction):
    if not len(frame):
        return None
    k = np.where(frame.ts.dt.hour.between(11, 15), 3, 1)
    w = frame.smp.to_numpy() * k
    return float(np.dot(w, np.abs(frame[prediction]-frame.energy))/w.sum()) if w.sum() else None


def worker(folder, window, trees, learning_rate, threads):
    started = time.perf_counter()
    if window not in ('tune', 'seasonal_diagnostic'):
        raise ValueError('P1 may not inspect confirmation')
    a,b,c,d = protocol.bounds(window)
    baseline = original.load_baseline()
    sites = pd.read_csv(original.DATA/'sites.csv')[
        ['site_id','current_group','capacity_kw','lat','lon','wa_station','wb_station']]
    gen = pd.read_parquet(original.DATA/'train/generation.parquet')
    gen.ts = pd.to_datetime(gen.ts).dt.tz_convert(protocol.TZ)
    train, valid = protocol.split(gen, window)
    del gen
    if not len(train) or not len(valid):
        raise ValueError('Empty chronological split')
    if train.duplicated(['site_id','ts']).any() or not np.isfinite(train.energy).all():
        raise ValueError('Invalid training labels')
    hours = protocol.complete_hours(valid, sites.site_id)
    mask = folder/'hours.csv'
    pd.DataFrame({'ts': hours.astype(str)}).to_csv(mask,index=False)
    expected = json.loads((ROOT/'docs/experiments/evaluation/2026-11-03-p0-coverage.json').read_text())
    if protocol.digest(mask) != expected['windows'][window]['mask_sha256']:
        raise ValueError('Evaluation mask differs from P0')
    fa = pd.read_parquet(original.DATA/'train/forecast_wa.parquet', columns=['wa_station','round','ts']+baseline.WA_COLS)
    fb = pd.read_parquet(original.DATA/'train/forecast_wb.parquet', columns=['wb_station','round','ts']+baseline.WB_COLS)
    for f in (fa,fb):
        f.ts = pd.to_datetime(f.ts).dt.tz_convert(protocol.TZ)
    fa = fa.loc[(fa.ts>=a)&(fa.ts<d)]
    fb = fb.loc[(fb.ts>=a)&(fb.ts<d)]
    for f,key in [(fa,'wa_station'),(fb,'wb_station')]:
        if f.duplicated([key,'round','ts']).any():
            raise ValueError('Duplicate forecast keys')
    output = valid.merge(sites[['site_id','capacity_kw']],on='site_id',validate='many_to_one')
    params = dict(n_estimators=trees, learning_rate=learning_rate, num_leaves=127,
                  min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, verbosity=-1, n_jobs=threads, random_state=0)
    training = {}
    unseen = sorted(set(valid.site_id)-set(train.site_id))
    for r in (10,17):
        features = baseline.features(train,sites,fa,fb,None,r)
        if len(features)!=len(train):
            raise ValueError('Training join changed row count')
        features = features.loc[features.elev>0]
        x = features[baseline.FEATS].astype('float32')
        y = (features.energy/features.cap).clip(0,1.2).astype('float32')
        t = time.perf_counter()
        model = lgb.LGBMRegressor(**params).fit(x,y)
        training[str(r)] = dict(rows=len(x),fit_seconds=time.perf_counter()-t,trees=model.booster_.num_trees())
        if model.booster_.num_trees()!=trees:
            raise ValueError('Requested tree count was not built')
        model_path = folder/f'model_round_{r}.txt'
        model.booster_.save_model(str(model_path))
        del features,x,y
        gc.collect()
        features = baseline.features(valid,sites,fa,fb,None,r)
        if not features[['site_id','ts']].reset_index(drop=True).equals(output[['site_id','ts']].reset_index(drop=True)):
            raise ValueError('Prediction rows lost alignment')
        x = features[baseline.FEATS].astype('float32')
        raw = model.predict(x)
        restored = lgb.Booster(model_file=str(model_path))
        np.testing.assert_allclose(restored.predict(x.iloc[:1024]),raw[:1024],rtol=1e-12)
        output[f'pred_{r}'] = np.maximum(np.where(features.elev>0,raw,0),0)*features.cap.to_numpy()
        output['daylight'] = features.elev.to_numpy()>0
        training[str(r)]['missing_feature_fraction'] = x.isna().mean().to_dict()
        del features,x,raw,restored,model
        gc.collect()
    del fa,fb
    output['ensemble'] = (output.pred_10+output.pred_17)/2
    if not np.isfinite(output[PREDICTIONS].to_numpy()).all():
        raise ValueError('Nonfinite predictions')
    groups = evaluation.aggregate_groups(output,sites,PREDICTIONS)
    prices = pd.read_csv(original.DATA/'train/smp.csv')
    prices.ts = pd.to_datetime(prices.ts).dt.tz_convert(protocol.TZ)
    groups = groups.merge(prices,on='ts',how='left',validate='many_to_one')
    fixed = groups.loc[groups.ts.isin(hours)]
    scores = {p: protocol.weighted_group_mae(fixed,hours,p) for p in PREDICTIONS}
    for frame in (output,groups):
        frame['month'] = (frame.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')
    monthly = []
    for month,rows in output.groupby('month'):
        for scope,subset in [('site_all',rows),('site_daylight',rows.loc[rows.daylight])]:
            for pred,values in metric_table(subset).items():
                monthly.append(dict(month=month,scope=scope,model=pred,**values))
    for month in sorted(output.month.unique()):
        complete = groups.loc[(groups.month==month)&groups.complete]
        for group in ('G1','G2'):
            subset = complete.loc[complete.group==group]
            for pred,values in metric_table(subset).items():
                monthly.append(dict(month=month,scope=f'reference_{group}_complete',model=pred,
                                    weighted_mae_kwh=single_group_weighted(subset,pred),**values))
    # Save individual predictions so P2 can reuse them without retraining.
    output.to_parquet(folder/'site_predictions.parquet',index=False,compression='zstd')
    groups.to_parquet(folder/'group_predictions.parquet',index=False,compression='zstd')
    pd.DataFrame(monthly).to_csv(folder/'monthly_metrics.csv',index=False)
    report = dict(status='complete',protocol=protocol.PROTOCOL,window=window,parameters=params,
        split=dict(train_start=str(a),train_end_exclusive=str(b),eval_start=str(c),eval_end_exclusive=str(d)),
        train_rows=len(train),eval_rows=len(valid),common_hours=len(hours),mask_sha256=protocol.digest(mask),
        training=training,weighted_group_mae=scores,group_fixed=metric_table(fixed),
        by_group={g:dict(metrics=metric_table(rows),weighted_mae={p:single_group_weighted(rows,p) for p in PREDICTIONS}) for g,rows in fixed.groupby('group')},
        site_daylight=metric_table(output.loc[output.daylight]),
        unseen_sites_count=len(unseen),unseen_daylight=metric_table(output.loc[output.daylight&output.site_id.isin(unseen)]),
        seen_daylight=metric_table(output.loc[output.daylight&~output.site_id.isin(unseen)]),
        peak_memory_mib=original.peak_memory_mib(),seconds_including_prediction_writes=time.perf_counter()-started,
        versions=dict(python=platform.python_version(),lightgbm=lgb.__version__,numpy=np.__version__,pandas=pd.__version__),
        machine=platform.machine(),saved_model_reload_checks_passed=True,confirmation_evaluated=False)
    write_json(folder/'summary.json',report)
    print(f'{window} {config_id(trees,learning_rate)} weighted MAE={scores["ensemble"]:.6f}',flush=True)


def orchestrate(folder,threads):
    start=time.perf_counter()
    original.check_data()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    if dirty:
        raise ValueError('Commit the experiment code before running the grid')
    report=dict(status='running',protocol=protocol.PROTOCOL,code_commit=commit,dirty=False,threads=threads,
                configuration_count=6,confirmation_evaluated=False,results=[])
    write_json(folder/'summary.json',report)
    try:
        for window in ('tune','seasonal_diagnostic'):
            for trees,rate in CONFIGS:
                dest=folder/window/config_id(trees,rate)
                cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--window',window,
                     '--trees',str(trees),'--learning-rate',str(rate),'--threads',str(threads),'--output-dir',str(dest)]
                print(f'Start {window} {config_id(trees,rate)}',flush=True)
                subprocess.run(cmd,check=True,cwd=ROOT)
                result=json.loads((dest/'summary.json').read_text())
                report['results'].append(dict(window=window,trees=trees,learning_rate=rate,
                    weighted_group_mae=result['weighted_group_mae']['ensemble'],
                    group_mae=result['group_fixed']['ensemble']['mae_kwh'],
                    site_daylight_nmae=result['site_daylight']['ensemble']['nmae_capacity_pct'],
                    fit_seconds=sum(x['fit_seconds'] for x in result['training'].values()),
                    total_seconds=result['seconds_including_prediction_writes'],peak_memory_mib=result['peak_memory_mib'],
                    output=str(dest.relative_to(folder))))
                write_json(folder/'summary.json',report)
        report['selected']=choose([r for r in report['results'] if r['window']=='tune'])
        report['selection_basis']='tune weighted ensemble MAE only; seasonal results are diagnostics'
        report['status']='complete'
    except BaseException as exc:
        report['status']='failed';report['error']=str(exc)
        raise
    finally:
        report['total_seconds']=time.perf_counter()-start
        write_json(folder/'summary.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--window',choices=['tune','seasonal_diagnostic'],default='tune')
    parser.add_argument('--trees',type=int,choices=[50,200,600],default=50)
    parser.add_argument('--learning-rate',type=float,choices=[0.03,0.06],default=0.06)
    args=parser.parse_args()
    folder=args.output_dir.resolve();data=original.DATA.resolve()
    if folder==data or data in folder.parents or args.threads<1:
        parser.error('Require positive threads and output outside original data')
    folder.mkdir(parents=True,exist_ok=False)
    if args.worker:worker(folder,args.window,args.trees,args.learning_rate,args.threads)
    else:orchestrate(folder,args.threads)


if __name__=='__main__':main()
