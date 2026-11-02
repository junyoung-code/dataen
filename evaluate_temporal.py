"""Frozen-model chronological holdout; no tuning on the evaluated year."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import train_smoke as original

TZ = 'Asia/Seoul'
TRAIN_START = pd.Timestamp('2023-06-01 01:00', tz=TZ)
TRAIN_END = pd.Timestamp('2024-05-31 01:00', tz=TZ)  # exclusive; 24h embargo
VALID_START = pd.Timestamp('2024-06-01 01:00', tz=TZ)
VALID_END = pd.Timestamp('2025-06-01 01:00', tz=TZ)  # exclusive
PREDICTIONS = ['pred_10', 'pred_17', 'ensemble', 'climatology']


def split_data(frame):
    train = frame.loc[(frame.ts >= TRAIN_START) & (frame.ts < TRAIN_END)].copy()
    valid = frame.loc[(frame.ts >= VALID_START) & (frame.ts < VALID_END)].copy()
    return train, valid


def metrics(actual, prediction, capacity):
    y, p, c = (np.asarray(a, dtype=float) for a in (actual, prediction, capacity))
    if not (y.shape == p.shape == c.shape):
        raise ValueError('Metric arrays must have equal shapes')
    if not all(np.isfinite(a).all() for a in (y, p, c)) or (c <= 0).any():
        raise ValueError('Nonfinite values or invalid capacities')
    result = {'rows': len(y)}
    if not len(y):
        return dict(result, mae_kwh=None, rmse_kwh=None, nmae_capacity_pct=None,
                    wape_pct=None, bias_kwh=None)
    error = p - y
    ae = np.abs(error)
    return dict(result, mae_kwh=float(ae.mean()), rmse_kwh=float(np.sqrt(np.mean(error**2))),
                nmae_capacity_pct=float(np.mean(ae / c) * 100),
                wape_pct=float(ae.sum() / np.abs(y).sum() * 100) if np.abs(y).sum() else None,
                bias_kwh=float(error.mean()))


def aggregate_groups(rows, sites, predictions):
    if rows.duplicated(['site_id', 'ts']).any():
        raise ValueError('Duplicate site timestamps')
    joined = rows[['site_id', 'ts', 'energy'] + predictions].merge(
        sites[['site_id', 'current_group', 'capacity_kw']], on='site_id', validate='many_to_one')
    if len(joined) != len(rows) or joined.isna().any().any():
        raise ValueError('Missing metadata, labels, or predictions')
    joined['group'] = joined.current_group.map({'A': 'G1', 'B': 'G2'})
    result = joined.groupby(['group', 'ts'], as_index=False).agg(
        energy=('energy', 'sum'), observed_sites=('site_id', 'nunique'),
        observed_capacity_kw=('capacity_kw', 'sum'), **{p: (p, 'sum') for p in predictions})
    required = sites.assign(group=sites.current_group.map({'A': 'G1', 'B': 'G2'})).groupby('group')
    result['expected_sites'] = result.group.map(required.site_id.nunique())
    result['capacity_kw'] = result.group.map(required.capacity_kw.sum())
    result['complete'] = result.observed_sites == result.expected_sites
    return result


def climatology_predict(train, valid):
    """Site/month/hour mean; fallback site/hour, month/hour, training grand mean."""
    history = train[['site_id', 'month', 'hour']].copy()
    history['normalized'] = train.energy.to_numpy() / train.capacity_kw.to_numpy()
    prediction = pd.Series(np.nan, index=range(len(valid)))
    for keys in (['site_id', 'month', 'hour'], ['site_id', 'hour'], ['month', 'hour']):
        means = history.groupby(keys, as_index=False).normalized.mean()
        mapped = valid[keys].reset_index(drop=True).merge(means, on=keys, how='left',
                                                         sort=False, validate='many_to_one').normalized
        prediction = prediction.fillna(mapped)
    return prediction.fillna(history.normalized.mean()).to_numpy() * valid.capacity_kw.to_numpy()


def metric_table(frame, capacity='capacity_kw'):
    return {p: metrics(frame.energy, frame[p], frame[capacity]) for p in PREDICTIONS}


def run(folder):
    start = time.perf_counter()
    original.check_data()
    b = original.load_baseline()
    sites = pd.read_csv(original.DATA / 'sites.csv')[
        ['site_id', 'current_group', 'capacity_kw', 'lat', 'lon', 'wa_station', 'wb_station']]
    gen = pd.read_parquet(original.DATA / 'train/generation.parquet')
    gen.ts = pd.to_datetime(gen.ts).dt.tz_convert(TZ)
    assert not gen.duplicated(['site_id', 'ts']).any()
    assert gen.energy.notna().all() and np.isfinite(gen.energy).all()
    train, valid = split_data(gen)
    del gen
    assert len(train) and len(valid)
    assert train.ts.max() < VALID_START - pd.Timedelta(hours=24)
    first_issuance = VALID_START.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=17)
    assert train.ts.max() < first_issuance
    axis = pd.date_range(VALID_START, VALID_END, freq='h', inclusive='left')
    assert len(axis) == 8760
    original.log(f'Train {len(train):,} rows: {train.ts.min()} ~ {train.ts.max()}')
    original.log(f'Holdout {len(valid):,} observed rows: {valid.ts.min()} ~ {valid.ts.max()}')
    fa = pd.read_parquet(original.DATA / 'train/forecast_wa.parquet',
                         columns=['wa_station', 'round', 'ts'] + b.WA_COLS)
    fb = pd.read_parquet(original.DATA / 'train/forecast_wb.parquet',
                         columns=['wb_station', 'round', 'ts'] + b.WB_COLS)
    for frame in (fa, fb):
        frame.ts = pd.to_datetime(frame.ts).dt.tz_convert(TZ)
    fa = fa.loc[(fa.ts >= TRAIN_START) & (fa.ts < VALID_END)].copy()
    fb = fb.loc[(fb.ts >= TRAIN_START) & (fb.ts < VALID_END)].copy()
    assert not fa.duplicated(['wa_station', 'round', 'ts']).any()
    assert not fb.duplicated(['wb_station', 'round', 'ts']).any()
    output = valid.merge(sites[['site_id', 'capacity_kw']], on='site_id', validate='many_to_one')
    params = dict(n_estimators=50, learning_rate=0.06, num_leaves=127, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                  n_jobs=-1, random_state=0)
    training = {}
    for r in b.ROUNDS:
        original.log(f'Round {r}: build training features')
        d = b.features(train, sites, fa, fb, None, r)
        assert len(d) == len(train)
        d = d.loc[d.elev > 0]
        x = d[b.FEATS].astype('float32')
        y = (d.energy / d.cap).clip(0, 1.2).astype('float32')
        fit_start = time.perf_counter()
        model = lgb.LGBMRegressor(**params).fit(x, y)
        training[str(r)] = {'daytime_rows': len(x), 'trees': model.booster_.num_trees(),
                             'fit_seconds': time.perf_counter() - fit_start,
                             'train_mae_normalized': float(np.abs(model.predict(x) - y).mean())}
        assert model.booster_.num_trees() == 50
        model_path = folder / f'model_round_{r}.txt'
        model.booster_.save_model(str(model_path))
        del d, x, y
        gc.collect()
        original.log(f'Round {r}: predict untouched holdout')
        d = b.features(valid, sites, fa, fb, None, r)
        assert d[['site_id', 'ts']].reset_index(drop=True).equals(output[['site_id', 'ts']].reset_index(drop=True))
        x = d[b.FEATS].astype('float32')
        raw = model.predict(x)
        restored = lgb.Booster(model_file=str(model_path))
        np.testing.assert_allclose(restored.predict(x.iloc[:1024]), raw[:1024], rtol=1e-12)
        output[f'pred_{r}'] = np.clip(np.where(d.elev.to_numpy() > 0, raw, 0), 0, None) * d.cap.to_numpy()
        if r == 10:
            output['daylight'] = d.elev.to_numpy() > 0
        training[str(r)]['holdout_missing_feature_fraction'] = x.isna().mean().to_dict()
        del d, x, raw, model, restored
        gc.collect()
    del fa, fb
    output['ensemble'] = (output.pred_10 + output.pred_17) / 2
    history = train.merge(sites[['site_id', 'capacity_kw']], on='site_id', validate='many_to_one')
    for frame in (history, output):
        frame['month'] = frame.ts.dt.month
        frame['hour'] = frame.ts.dt.hour
    output['climatology'] = np.maximum(climatology_predict(history, output), 0)
    output.loc[~output.daylight, 'climatology'] = 0
    assert np.isfinite(output[PREDICTIONS].to_numpy()).all()
    assert (output[PREDICTIONS] >= 0).all().all()
    groups = aggregate_groups(output, sites, PREDICTIONS)
    prices = pd.read_csv(original.DATA / 'train/smp.csv')
    prices.ts = pd.to_datetime(prices.ts).dt.tz_convert(TZ)
    groups = groups.merge(prices, on='ts', how='left', validate='many_to_one')
    assert groups.smp.notna().all() and np.isfinite(groups.smp).all() and (groups.smp >= 0).all()
    groups['k'] = np.where(groups.ts.dt.hour.between(11, 15), 3, 1)
    complete = groups.loc[groups.complete]
    # Attribute the right-closed midnight interval to the preceding calendar day/month.
    output['period_month'] = ((output.ts - pd.Timedelta(hours=1)).dt.year * 100
                              + (output.ts - pd.Timedelta(hours=1)).dt.month)
    groups['period_month'] = ((groups.ts - pd.Timedelta(hours=1)).dt.year * 100
                              + (groups.ts - pd.Timedelta(hours=1)).dt.month)
    monthly = []
    for month, rows in output.groupby('period_month'):
        for scope, subset in [('site_all', rows), ('site_daylight', rows.loc[rows.daylight])]:
            for model, values in metric_table(subset).items():
                monthly.append(dict(month=int(month), scope=scope, model=model, **values))
    for month, rows in groups.loc[groups.complete].groupby('period_month'):
        for model, values in metric_table(rows).items():
            monthly.append(dict(month=int(month), scope='group_complete', model=model, **values))
    proxy = {}
    for p in PREDICTIONS:
        revenue = float((complete.smp * complete.energy).sum())
        penalty = float((complete.smp * complete.k * np.abs(complete[p] - complete.energy)).sum() * 0.1)
        weighted_denominator = float((complete.smp * complete.k).sum())
        proxy[p] = {'evaluated_group_hours': len(complete), 'revenue_krw': revenue,
                    'error_penalty_krw': penalty, 'score_proxy_krw': revenue - penalty,
                    'penalty_pct_of_revenue': penalty / revenue * 100 if revenue else None,
                    'smp_k_weighted_mae_kwh': penalty / .1 / weighted_denominator if weighted_denominator else None}
    unseen_sites = sorted(set(valid.site_id) - set(train.site_id))
    report = {
        'status': 'complete', 'protocol': 'fixed 50-tree ensemble, frozen one-year chronological holdout; no tuning',
        'split': {'train_start_inclusive': str(TRAIN_START), 'train_end_exclusive': str(TRAIN_END),
                  'validation_start_inclusive': str(VALID_START), 'validation_end_exclusive': str(VALID_END),
                  'train_observed_min': str(train.ts.min()), 'train_observed_max': str(train.ts.max()),
                  'first_forecast_issuance': str(first_issuance), 'embargo_hours': 24,
                  'train_rows': len(train), 'holdout_observed_rows': len(valid),
                  'holdout_expected_rows': len(sites) * len(axis), 'unseen_sites': unseen_sites},
        'parameters': params, 'features': b.FEATS, 'training': training,
        'coverage': {'site_observation_pct': len(valid) / (len(sites) * len(axis)) * 100,
                     'complete_group_hours': len(complete), 'expected_group_hours': 2 * len(axis),
                     'complete_group_hour_pct': len(complete) / (2 * len(axis)) * 100,
                     'complete_hours_by_group': complete.groupby('group').size().to_dict()},
        'site_all_observed': metric_table(output),
        'site_daylight_observed': metric_table(output.loc[output.daylight]),
        'group_complete_hours': metric_table(complete),
        'group_complete_by_group': {g: metric_table(rows) for g, rows in complete.groupby('group')},
        'group_observed_subset_only': metric_table(groups, 'observed_capacity_kw'),
        'historical_score_proxy_complete_hours_only': proxy,
        'definitions': {'nmae_capacity_pct': '100 * mean(abs(prediction - raw actual) / capacity_kw)',
                        'wape_pct': '100 * sum(abs(prediction - raw actual)) / sum(abs(raw actual))',
                        'climatology': 'training-only normalized means: site/month/hour, site/hour, month/hour, grand mean; night=0',
                        'score_proxy': 'sum(SMP * actual) - 0.1 * sum(SMP * k * abs(pred - actual)); complete group-hours only'},
        'limitations': ['One historical year, not the hidden competition period; historical model has less training data than the full-data model.',
                       'Missing labels excluded, never filled with zero; complete-hour group results may have selection bias.',
                       'No hyperparameter selection on holdout; future optimization requires a separate evaluation period.'],
        'versions': {'python': platform.python_version(), 'lightgbm': lgb.__version__,
                     'pandas': pd.__version__, 'numpy': np.__version__},
        'saved_model_reload_checks_passed': True,
        'total_seconds': time.perf_counter() - start, 'peak_process_memory_mib': original.peak_memory_mib()}
    output.to_parquet(folder / 'site_predictions.parquet', index=False, compression='zstd')
    groups.to_csv(folder / 'group_predictions.csv', index=False)
    pd.DataFrame(monthly).to_csv(folder / 'monthly_metrics.csv', index=False)
    (folder / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    original.log(f'Complete: {folder}; {report["total_seconds"]:.1f}s')
    print(json.dumps({k: report[k] for k in ['coverage', 'site_daylight_observed', 'group_complete_hours']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    if folder == original.DATA or original.DATA in folder.parents:
        parser.error('Output must be outside original data')
    folder.mkdir(parents=True, exist_ok=True)
    if any(p.name != 'run.log' for p in folder.iterdir()):
        parser.error('Output must be a new empty folder')
    run(folder)


if __name__ == '__main__':
    main()
