"""Fresh annual forecast grids with strictly bounded, partially masked labels."""
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_smoke as original
from features_p3 import build
from neural_data import Batches, daily_starts, row_indices
from neural_models import Scaler
from run_peer_w import TZ, bounds, peer_labels, write_json
from training_weights import make_weights

ROOT = Path(__file__).resolve().parent
CONTRACT_SHA256 = 'd65965390a3027d6b3267c352d7222cdfd76a9dca128e7d319bab3c77465b1ec'


def check_contract():
    path = ROOT/'AB_EVALUATION_CONTRACT.json'
    assert original.sha256(path) == CONTRACT_SHA256
    contract = json.loads(path.read_text())
    for name, digest in contract['fixed_source_sha256'].items():
        assert original.sha256(ROOT/name) == digest, name
    assert original.sha256(original.DATA/'sites.csv') == contract['sites_file_sha256']
    return contract


def key_mask_digest(output):
    frame = output.sort_values(['site_id', 'ts']).reset_index(drop=True)
    payload = frame[['site_id', 'ts', 'has_peer']].to_csv(index=False, lineterminator='\n')
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def bounded_starts(weights, n_sites, n_hours, first, last_inclusive):
    """Retain the last partial decoder day; no labels after the cutoff exist."""
    end = ((last_inclusive+1+23)//24)*24
    starts = daily_starts(n_sites, n_hours, first, end)
    indices = starts[:, None]+np.arange(24)[None, :]
    return starts[np.any(weights[indices] > 0, axis=1)]


def prepare(folder, fold, round_hour):
    folder = Path(folder)
    contract = check_contract()
    expected = contract['folds'][fold]
    manifest = folder/'metadata.json'
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        assert meta['status'] == 'complete' and meta['fold'] == fold and meta['round'] == round_hour
        assert meta['contract_sha256'] == CONTRACT_SHA256
        assert meta['builder_sha256'] == original.sha256(Path(__file__))
        return meta
    folder.mkdir(parents=True, exist_ok=False)
    a, cutoff, c, d = bounds(fold)
    axis_start = a-pd.Timedelta(hours=24)
    axis = pd.date_range(axis_start, d, freq='h', inclusive='left')
    n_hours = len(axis)
    baseline = original.load_baseline()
    sites = pd.read_csv(original.DATA/'sites.csv')
    site_ids = sites.site_id.tolist()
    gen = pd.read_parquet(original.DATA/'train/generation.parquet')
    gen.ts = pd.to_datetime(gen.ts).dt.tz_convert(TZ)
    assert not gen.duplicated(['site_id', 'ts']).any() and np.isfinite(gen.energy).all()
    train = gen.loc[(gen.ts >= a) & (gen.ts <= cutoff)].copy()
    valid = gen.loc[(gen.ts >= c) & (gen.ts < d)].copy()
    cap = gen.site_id.map(sites.set_index('site_id').capacity_kw)
    _, has = peer_labels(gen.energy, cap)
    seen = set(gen.loc[(gen.ts < c) & has, 'site_id'])
    del gen, cap, has
    assert len(train) == expected['train_raw_rows']
    assert len(valid) == expected['eval_raw_rows']
    fa = pd.read_parquet(original.DATA/'train/forecast_wa.parquet', columns=['wa_station','round','ts']+baseline.WA_COLS)
    fb = pd.read_parquet(original.DATA/'train/forecast_wb.parquet', columns=['wb_station','round','ts']+baseline.WB_COLS)
    for f in (fa, fb):
        f.ts = pd.to_datetime(f.ts).dt.tz_convert(TZ)
    fa = fa.loc[(fa.ts >= axis_start) & (fa.ts < d) & (fa['round'] == round_hour)]
    fb = fb.loc[(fb.ts >= axis_start) & (fb.ts < d) & (fb['round'] == round_hour)]
    columns = list(baseline.FEATS)+['wa_dist', 'wb_dist']
    assert columns == expected['feature_columns']
    shape = (len(sites)*n_hours, len(columns))
    x = np.lib.format.open_memmap(folder/'x.npy', mode='w+', dtype='float32', shape=shape)
    for begin in range(0, len(sites), 16):
        subset = sites.iloc[begin:begin+16]
        grid = pd.MultiIndex.from_product([subset.site_id, axis], names=['site_id','ts']).to_frame(index=False)
        grid['energy'] = 0.  # Join placeholder, never a model feature.
        features, actual_columns = build(baseline, grid, subset, fa, fb, round_hour, 'station_distance')
        assert actual_columns == columns
        x[begin*n_hours:(begin+len(subset))*n_hours] = features[columns].to_numpy(dtype='float32')
        del grid, features
    del fa, fb
    x.flush()
    train_idx = row_indices(train, site_ids, axis_start, n_hours)
    valid_idx = row_indices(valid, site_ids, axis_start, n_hours)
    daylight = x[:, columns.index('elev')] > 0
    used = daylight[train_idx]
    train_idx = train_idx[used]
    train_day = train.loc[used].copy()
    train_day['cap'] = train_day.site_id.map(sites.set_index('site_id').capacity_kw).astype('float32')
    weights, weight_info = make_weights(train_day, None, 'capacity', a, cutoff+pd.Timedelta(nanoseconds=1))
    target = (train_day.energy/train_day.cap).clip(0, 1.2).to_numpy(dtype='float32')
    y = np.zeros(shape[0], dtype='float32')
    w = np.zeros_like(y)
    y[train_idx] = target
    w[train_idx] = weights.astype('float32')
    missing = np.isnan(x[train_idx]).mean(axis=0)
    scaler = Scaler().fit(x[train_idx])
    np.savez(folder/'scaler.npz', median=scaler.median, mean=scaler.mean, std=scaler.std)
    for begin in range(0, len(x), 100_000):
        x[begin:begin+100_000] = scaler.transform(x[begin:begin+100_000])
    x.flush()
    last = int((cutoff-axis_start).total_seconds()/3600)
    first_eval = int((c-axis_start).total_seconds()/3600)
    starts = bounded_starts(w, len(sites), n_hours, 24, last)
    eval_starts = daily_starts(len(sites), n_hours, first_eval, n_hours)
    # Every positive label occurs once in sequence loss, including final 01:00..17:00.
    decoders = (starts[:, None]+np.arange(24)[None, :]).ravel()
    np.testing.assert_array_equal(np.sort(decoders[w[decoders] > 0]), np.sort(train_idx))
    assert not w.reshape(len(sites), n_hours)[:, last+1:].any()
    assert not y.reshape(len(sites), n_hours)[:, last+1:].any()
    output = valid.rename(columns={'energy':'energy_raw'}).merge(sites[['site_id','capacity_kw']], on='site_id', validate='many_to_one')
    output['energy_peer'], output['has_peer'] = peer_labels(output.energy_raw, output.capacity_kw)
    output['seen'] = output.site_id.isin(seen)
    output['daylight'] = daylight[valid_idx]
    mask_digest = key_mask_digest(output)
    assert mask_digest == expected['evaluation_key_mask_sha256']
    assert int(output.has_peer.sum()) == expected['observed_site_hours']
    output.to_parquet(folder/'evaluation.parquet', index=False)
    for name, array in dict(y=y, w=w, train_rows=train_idx, eval_rows=valid_idx, train_starts=starts, eval_starts=eval_starts).items():
        np.save(folder/f'{name}.npy', array)
    meta = dict(status='complete', fold=fold, round=round_hour, columns=columns,
        contract_sha256=CONTRACT_SHA256, builder_sha256=original.sha256(Path(__file__)),
        n_sites=len(sites), n_hours=n_hours, axis_start=str(axis_start),
        train_rows=len(train), train_daylight_rows=len(train_idx), eval_rows=len(valid),
        sequence_training_days=len(starts), train_weights=weight_info,
        train_missing_feature_fraction=dict(zip(columns, missing.tolist())),
        split=dict(train_start=str(a), train_end_inclusive=str(cutoff), eval_start=str(c), eval_end_exclusive=str(d)),
        evaluation_key_mask_sha256=mask_digest, train_min=str(train_day.ts.min()), train_max=str(train_day.ts.max()),
        positive_labels_after_cutoff=0, sequence_positive_labels=len(train_idx),
        final_day_positive_labels=int(((train_idx % n_hours)//24 == last//24).sum()),
        target='(energy/float32(cap)).clip(0,1.2).astype(float32)', input_contains_targets=False,
        preprocessing_fit='observed daytime training rows only; cutoff inclusive',
        decoder_day='01:00 through next 00:00; final training day masked after 17:00')
    write_json(manifest, meta)
    del x, y, w, train, train_day, valid, output
    gc.collect()
    return meta
