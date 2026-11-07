"""XGBoost vs fixed LightGBM/CatBoost, with common data and historical checks."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import xgboost
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

import evaluate_temporal as evaluation
import train_smoke as original
import validation_protocol as protocol
from features_p3 import build
from training_weights import make_weights
from run_p1 import PREDICTIONS, metric_table, single_group_weighted, write_json

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 'lightgbm_p4'
CAT_REFERENCE = 'cat_d6_n400'
COMPARATORS = (REFERENCE, CAT_REFERENCE)
HISTORY = ROOT/'docs/experiments/models/2026-11-03-catboost.json'
CANDIDATES = {
    'xgb_d6_n400': dict(n_estimators=400, max_depth=6, learning_rate=0.03),
    'xgb_d6_n1000': dict(n_estimators=1000, max_depth=6, learning_rate=0.03),
    'xgb_d8_n1000': dict(n_estimators=1000, max_depth=8, learning_rate=0.03),
}


def model_parameters(config, threads):
    return dict(**config, objective='reg:absoluteerror', tree_method='hist', device='cpu',
                random_state=0, n_jobs=threads, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=1., reg_alpha=0., reg_lambda=1., max_bin=256,
                grow_policy='depthwise', verbosity=1)


class Progress(xgboost.callback.TrainingCallback):
    def __init__(self, trees):
        self.trees = trees
        self.started = time.perf_counter()

    def after_iteration(self, model, epoch, evals_log):
        if epoch == 0 or (epoch+1) % 100 == 0 or epoch+1 == self.trees:
            print(f'{epoch+1}/{self.trees} trees, {time.perf_counter()-self.started:.1f}s', flush=True)
        return False


def choose(rows):
    candidates = [r for r in rows if r['variant'] in CANDIDATES]
    if len(candidates) != len(CANDIDATES) or {r['variant'] for r in candidates} != set(CANDIDATES):
        raise ValueError('Every XGBoost candidate must be evaluated exactly once')
    if any(r['window'] != 'tune' or not np.isfinite(r['weighted_group_mae']) for r in candidates):
        raise ValueError('Select using finite tune scores only')
    return min(candidates, key=lambda r: (r['weighted_group_mae'], list(CANDIDATES).index(r['variant'])))


def read_selection(path):
    selection = json.loads(path.read_text())
    variant = selection['selected_variant']
    if (selection['selection_window'] != 'tune' or variant not in CANDIDATES
            or selection['config'] != CANDIDATES[variant]
            or selection['weight_17'] != 0.75
            or selection['feature_bundle'] != 'station_distance'
            or selection['reference'] != REFERENCE
            or selection['comparators'] != list(COMPARATORS)
            or selection['sites_sha256'] != protocol.digest(original.DATA/'sites.csv')):
        raise ValueError('Frozen comparison conditions changed')
    return selection


def worker(folder, window, variant, threads):
    config = CANDIDATES[variant]
    weighting = 'capacity'
    trees = config['n_estimators']
    started = time.perf_counter()
    if window not in protocol.WINDOWS:
        raise ValueError('Unknown evaluation window')
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
    params = model_parameters(config, threads)
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
        model = XGBRegressor(**params, callbacks=[Progress(trees)]).fit(x,y,sample_weight=weights)
        training[str(r)] = dict(rows=len(x),fit_seconds=time.perf_counter()-t,trees=model.get_booster().num_boosted_rounds(),train_missing_feature_fraction=train_missing,weights=weight_info)
        if model.get_booster().num_boosted_rounds()!=trees:
            raise ValueError('Requested tree count was not built')
        model_path = folder/f'model_round_{r}.ubj'
        model.save_model(str(model_path))
        del features,x,y,weights
        gc.collect()
        features, eval_columns = build(baseline,valid,sites,fa,fb,r,'station_distance')
        assert eval_columns == columns
        if not features[['site_id','ts']].reset_index(drop=True).equals(output[['site_id','ts']].reset_index(drop=True)):
            raise ValueError('Prediction rows lost alignment')
        x = features[columns].astype('float32')
        raw = model.predict(x)
        restored = XGBRegressor(n_jobs=threads)
        restored.load_model(str(model_path))
        np.testing.assert_allclose(restored.predict(x.iloc[:1024]), raw[:1024], rtol=1e-12, atol=1e-12)
        training[str(r)]['resolved_parameters'] = json.loads(model.get_booster().save_config())
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
        versions=dict(python=platform.python_version(),xgboost=xgboost.__version__,numpy=np.__version__,pandas=pd.__version__),
        machine=platform.machine(),saved_model_reload_checks_passed=True,confirmation_evaluated=window=='confirmation')
    write_json(folder/'summary.json',report)
    print(f'{window}/{variant}: complete, weighted MAE={scores["ensemble"]:.6f}',flush=True)


def run_worker(folder, window, variant, threads):
    if variant == REFERENCE:
        if window == 'confirmation':
            from confirm_p5 import worker as reference_worker
            reference_worker(folder, threads)
        else:
            from run_p4 import worker as reference_worker
            reference_worker(folder, window, 'l1_capacity', threads)
    elif variant == CAT_REFERENCE:
        from run_catboost import worker as catboost_worker
        catboost_worker(folder, window, CAT_REFERENCE, threads)
    else:
        worker(folder, window, variant, threads)


def result_row(dest, root, window, variant):
    result = json.loads((dest/'summary.json').read_text())
    if result['status'] != 'complete':
        raise ValueError('Incomplete worker output')
    return dict(window=window, variant=variant,
                weighted_group_mae=result['weighted_group_mae']['ensemble'],
                group_mae=result['group_fixed']['ensemble']['mae_kwh'],
                site_daylight_nmae=result['site_daylight']['ensemble']['nmae_capacity_pct'],
                fit_seconds=sum(r['fit_seconds'] for r in result['training'].values()),
                total_seconds=result['seconds_including_prediction_writes'],
                peak_memory_mib=result['peak_memory_mib'], common_hours=result['common_hours'],
                mask_sha256=result['mask_sha256'], output=str(dest.relative_to(root)))


def orchestrate(folder, threads):
    started = time.perf_counter()
    original.check_data()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit experiment code before running')
    source_hashes = {str(p.relative_to(ROOT)): protocol.digest(p) for p in sorted((ROOT/'src').glob('*.py')) + sorted((ROOT/'tests').glob('*.py'))}
    report = dict(status='running', protocol=protocol.PROTOCOL, code_commit=commit, dirty=False,
                  threads=threads, candidates=CANDIDATES, reference=REFERENCE, comparators=list(COMPARATORS),
                  historical_comparison_sha256=protocol.digest(HISTORY),
                  tuning_opportunity_equal=False,
                  machine=dict(system=platform.system(),macos=platform.mac_ver()[0],architecture=platform.machine()),
                  feature_bundle='station_distance', weight_17=0.75,
                  source_hashes=source_hashes, labels_seen_previously=True, results=[])
    write_json(folder/'summary.json', report)

    def execute(window, variant, selection_file=None):
        dest = folder/window/variant
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--window', window,
                   '--variant', variant, '--threads', str(threads), '--output-dir', str(dest)]
        if selection_file:
            command += ['--selection-file', str(selection_file)]
        subprocess.run(command, cwd=ROOT, check=True)
        row = result_row(dest, folder, window, variant)
        if variant in COMPARATORS:
            historic = json.loads(HISTORY.read_text())
            expected = next(d['summary'] for d in historic['details']
                            if d['window']==window and d['variant']==variant)
            actual = json.loads((dest/'summary.json').read_text())
            for key in ('feature_columns','weight_17','weighting','split','train_rows',
                        'eval_rows','common_hours','mask_sha256'):
                if actual[key] != expected[key]:
                    raise ValueError(f'Comparison conditions changed: {variant}/{key}')
            for round_hour in ('10','17'):
                if actual['training'][round_hour]['weights'] != expected['training'][round_hour]['weights']:
                    raise ValueError('Historical training weights changed')
            # A different CPU can alter floating-point training. Record the actual
            # fresh score; never substitute a score measured on the other machine.
            old_score = expected['weighted_group_mae']['ensemble']
            row['historical_weighted_group_mae'] = old_score
            row['historical_score_delta'] = row['weighted_group_mae'] - old_score
        report['results'].append(row)
        write_json(folder/'summary.json', report)

    try:
        for variant in [*COMPARATORS, *CANDIDATES]:
            execute('tune', variant)
        chosen = choose(report['results'])
        selection_file = folder/'selection.json'
        write_json(selection_file, dict(selection_window='tune', selected_variant=chosen['variant'],
                   config=CANDIDATES[chosen['variant']], reference=REFERENCE, comparators=list(COMPARATORS),
                   feature_bundle='station_distance', weight_17=0.75,
                   sites_sha256=protocol.digest(original.DATA/'sites.csv'), code_commit=commit,
                   tune_weighted_group_mae=chosen['weighted_group_mae']))
        frozen_hash = protocol.digest(selection_file)
        report['selected'] = chosen
        report['selection_sha256'] = frozen_hash
        write_json(folder/'summary.json', report)
        for window in ('seasonal_diagnostic', 'confirmation'):
            for variant in (*COMPARATORS, chosen['variant']):
                if protocol.digest(selection_file) != frozen_hash:
                    raise ValueError('Selection changed after tuning')
                execute(window, variant, selection_file)
        if protocol.digest(selection_file) != frozen_hash:
            raise ValueError('Selection changed during confirmation')
        if any(protocol.digest(ROOT/name) != digest for name, digest in source_hashes.items()):
            raise ValueError('Source code changed during run')
        if protocol.digest(HISTORY) != report['historical_comparison_sha256']:
            raise ValueError('Historical comparison document changed during run')
        report['selection_unchanged'] = True
        report['confirmation_evaluated'] = True
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        raise
    finally:
        report['total_seconds'] = time.perf_counter()-started
        write_json(folder/'summary.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--window', choices=protocol.WINDOWS, default='tune')
    parser.add_argument('--variant', choices=[*COMPARATORS, *CANDIDATES], default=REFERENCE)
    parser.add_argument('--selection-file', type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('Positive thread count required')
    if args.worker and args.window != 'tune':
        if not args.selection_file:
            parser.error('Historical checks require a frozen tune selection')
        selection = read_selection(args.selection_file)
        if args.variant not in (*COMPARATORS, selection['selected_variant']):
            parser.error('Only the frozen candidate and reference may be checked')
    folder = args.output_dir.resolve()
    folder.relative_to((ROOT/'outputs').resolve())
    folder.mkdir(parents=True, exist_ok=False)
    if args.worker:
        try:
            run_worker(folder, args.window, args.variant, args.threads)
        except BaseException as error:
            write_json(folder/'summary.json', dict(status='failed', error=str(error)))
            raise
    else:
        orchestrate(folder, args.threads)


if __name__ == '__main__':
    main()
