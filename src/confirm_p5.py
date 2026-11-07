"""Fixed P4 model retraining after P5 assignment freeze; never selects a model."""
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
from features_p3 import build
from training_weights import VARIANTS, make_weights, choose
from run_p1 import PREDICTIONS, metric_table, single_group_weighted, write_json
ROOT = Path(__file__).resolve().parents[1]


def worker(folder, threads):
    window, variant = 'confirmation', 'l1_capacity'
    objective, weighting = VARIANTS[variant]
    trees, learning_rate = 200, 0.03
    started = time.perf_counter()
    if window != 'confirmation':
        raise ValueError('Only frozen P5 confirmation is permitted')
    a,b,c,d = protocol.bounds(window)
    baseline = original.load_baseline()
    sites = pd.read_csv(original.DATA/'sites.csv')[
        ['site_id','current_group','capacity_kw','lat','lon','wa_station','wb_station','wa_dist','wb_dist']]
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
    prices = pd.read_csv(original.DATA/'train/smp.csv')
    prices.ts = pd.to_datetime(prices.ts).dt.tz_convert(protocol.TZ)
    params = dict(objective=objective,n_estimators=trees, learning_rate=learning_rate, num_leaves=127,
                  min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, verbosity=-1, n_jobs=threads, random_state=0)
    training = {}
    unseen = sorted(set(valid.site_id)-set(train.site_id))
    for r in (10,17):
        print(f'{window}/{variant}: round {r} build training features',flush=True)
        features, columns = build(baseline,train,sites,fa,fb,r,'station_distance')
        if len(features)!=len(train):
            raise ValueError('Training join changed row count')
        features = features.loc[features.elev>0]
        x = features[columns].astype('float32')
        y = (features.energy/features.cap).clip(0,1.2).astype('float32')
        train_missing=x.isna().mean().to_dict()
        print(f'{window}/{variant}: round {r} fitting {len(x):,} rows x {len(columns)} features',flush=True)
        weights, weight_info = make_weights(features,prices,weighting,a,b)
        assert 'smp' not in columns
        t = time.perf_counter()
        model = lgb.LGBMRegressor(**params).fit(x,y,sample_weight=weights)
        training[str(r)] = dict(rows=len(x),fit_seconds=time.perf_counter()-t,trees=model.booster_.num_trees(),train_missing_feature_fraction=train_missing,weights=weight_info)
        if model.booster_.num_trees()!=trees:
            raise ValueError('Requested tree count was not built')
        model_path = folder/f'model_round_{r}.txt'
        model.booster_.save_model(str(model_path))
        del features,x,y,weights
        gc.collect()
        features, eval_columns = build(baseline,valid,sites,fa,fb,r,'station_distance')
        assert eval_columns == columns
        if not features[['site_id','ts']].reset_index(drop=True).equals(output[['site_id','ts']].reset_index(drop=True)):
            raise ValueError('Prediction rows lost alignment')
        x = features[columns].astype('float32')
        raw = model.predict(x)
        restored = lgb.Booster(model_file=str(model_path))
        np.testing.assert_allclose(restored.predict(x.iloc[:1024]),raw[:1024],rtol=1e-12)
        output[f'pred_{r}'] = np.maximum(np.where(features.elev>0,raw,0),0)*features.cap.to_numpy()
        output['daylight'] = features.elev.to_numpy()>0
        training[str(r)]['missing_feature_fraction'] = x.isna().mean().to_dict()
        del features,x,raw,restored,model
        gc.collect()
    del fa,fb
    output['ensemble'] = output.pred_10*0.25+output.pred_17*0.75
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
    # Save site predictions for later fixed-mask team allocation experiments.
    output.to_parquet(folder/'site_predictions.parquet',index=False,compression='zstd')
    groups.to_parquet(folder/'group_predictions.parquet',index=False,compression='zstd')
    pd.DataFrame(monthly).to_csv(folder/'monthly_metrics.csv',index=False)
    report = dict(status='complete',protocol=protocol.PROTOCOL,window=window,variant=variant,weighting=weighting,feature_columns=columns,weight_17=0.75,parameters=params,
        split=dict(train_start=str(a),train_end_exclusive=str(b),eval_start=str(c),eval_end_exclusive=str(d)),
        train_rows=len(train),eval_rows=len(valid),common_hours=len(hours),mask_sha256=protocol.digest(mask),
        training=training,weighted_group_mae=scores,group_fixed=metric_table(fixed),
        by_group={g:dict(metrics=metric_table(rows),weighted_mae={p:single_group_weighted(rows,p) for p in PREDICTIONS}) for g,rows in fixed.groupby('group')},
        site_daylight=metric_table(output.loc[output.daylight]),
        unseen_sites_count=len(unseen),unseen_daylight=metric_table(output.loc[output.daylight&output.site_id.isin(unseen)]),
        seen_daylight=metric_table(output.loc[output.daylight&~output.site_id.isin(unseen)]),
        peak_memory_mib=original.peak_memory_mib(),seconds_including_prediction_writes=time.perf_counter()-started,
        versions=dict(python=platform.python_version(),lightgbm=lgb.__version__,numpy=np.__version__,pandas=pd.__version__),
        machine=platform.machine(),saved_model_reload_checks_passed=True,confirmation_evaluated=True)
    write_json(folder/'summary.json',report)
    print(f'{window}/{variant}: complete, weighted MAE={scores["ensemble"]:.6f}',flush=True)


def main():
    parser=argparse.ArgumentParser(description='Retrain only the frozen P4 model for P5 historical confirmation')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--selection-file',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    selection=json.loads(args.selection_file.read_text())
    assignment=args.selection_file.parent/'optimized_assignment.csv'
    if selection['assignment_sha256']!=protocol.digest(assignment):
        raise ValueError('Frozen assignment changed')
    if selection['model']!={'variant':'l1_capacity','trees':200,'learning_rate':0.03,'weight_17':0.75,'feature_bundle':'station_distance'}:
        raise ValueError('Unexpected frozen model')
    folder=args.output_dir.resolve();folder.relative_to((ROOT/'outputs').resolve())
    if args.threads<1:parser.error('Positive thread count required')
    folder.mkdir(parents=True,exist_ok=False)
    worker(folder,args.threads)
    if selection['assignment_sha256']!=protocol.digest(assignment):raise ValueError('Assignment changed during confirmation')


if __name__=='__main__':main()
