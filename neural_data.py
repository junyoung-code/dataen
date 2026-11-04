"""Memory-mapped forecast sequences. No observed target is ever an input."""
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import train_smoke as original
import validation_protocol as protocol
from features_p3 import build
from training_weights import make_weights
from neural_models import Scaler
from run_p1 import write_json


def row_indices(rows, site_ids, axis_start, n_hours):
    site = rows.site_id.map({v:i for i,v in enumerate(site_ids)}).to_numpy()
    hour_float = (rows.ts-axis_start).dt.total_seconds().to_numpy()/3600
    if pd.isna(site).any() or not np.equal(hour_float, np.floor(hour_float)).all():
        raise ValueError('Unknown site or non-hourly timestamp')
    hour = hour_float.astype('int64')
    if (hour < 0).any() or (hour >= n_hours).any():
        raise ValueError('Row outside feature grid')
    return site.astype('int64')*n_hours+hour


def daily_starts(n_sites, n_hours, first, last):
    if (last-first)%24 or first < 24 or last > n_hours:
        raise ValueError('Whole days and a full 24-hour encoder required')
    return (np.arange(n_sites)[:,None]*n_hours + np.arange(first,last,24)[None,:]).ravel()


def sequence_indices(starts):
    return np.asarray(starts)[:,None] + np.arange(-24,24)[None,:]


def prepare(folder, window, round_hour):
    """One shared grid per window/round, built in site chunks to bound RAM."""
    folder = Path(folder)
    manifest = folder/'metadata.json'
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        if meta['status'] != 'complete' or meta['window'] != window or meta['round'] != round_hour:
            raise ValueError('Invalid cache')
        return meta
    folder.mkdir(parents=True, exist_ok=False)
    a,b,c,d = protocol.bounds(window)
    axis_start = a-pd.Timedelta(hours=24)
    axis = pd.date_range(axis_start,d,freq='h',inclusive='left')
    n_hours = len(axis)
    baseline = original.load_baseline()
    sites = pd.read_csv(original.DATA/'sites.csv')
    site_ids = sites.site_id.tolist()
    gen = pd.read_parquet(original.DATA/'train/generation.parquet')
    gen.ts = pd.to_datetime(gen.ts).dt.tz_convert(protocol.TZ)
    train,valid = protocol.split(gen,window)
    del gen
    fa = pd.read_parquet(original.DATA/'train/forecast_wa.parquet',columns=['wa_station','round','ts']+baseline.WA_COLS)
    fb = pd.read_parquet(original.DATA/'train/forecast_wb.parquet',columns=['wb_station','round','ts']+baseline.WB_COLS)
    for f in (fa,fb):
        f.ts = pd.to_datetime(f.ts).dt.tz_convert(protocol.TZ)
    fa = fa.loc[(fa.ts>=axis_start)&(fa.ts<d)&(fa['round']==round_hour)]
    fb = fb.loc[(fb.ts>=axis_start)&(fb.ts<d)&(fb['round']==round_hour)]
    columns = list(baseline.FEATS)+['wa_dist','wb_dist']
    shape = (len(sites)*n_hours,len(columns))
    x = np.lib.format.open_memmap(folder/'x.npy',mode='w+',dtype='float32',shape=shape)
    for begin in range(0,len(sites),16):
        subset = sites.iloc[begin:begin+16]
        grid = pd.MultiIndex.from_product([subset.site_id,axis],names=['site_id','ts']).to_frame(index=False)
        grid['energy'] = 0.  # Required join column only; excluded from model inputs.
        features, actual_columns = build(baseline,grid,subset,fa,fb,round_hour,'station_distance')
        assert actual_columns == columns
        x[begin*n_hours:(begin+len(subset))*n_hours] = features[columns].to_numpy(dtype='float32')
        del grid,features
    del fa,fb
    x.flush()
    train_idx = row_indices(train,site_ids,axis_start,n_hours)
    valid_idx = row_indices(valid,site_ids,axis_start,n_hours)
    daylight = x[:,columns.index('elev')]>0
    used = daylight[train_idx]
    train_idx = train_idx[used]
    train_day = train.loc[used].copy()
    train_day['cap'] = train_day.site_id.map(sites.set_index('site_id').capacity_kw).astype('float32')
    weights, weight_info = make_weights(train_day,None,'capacity',a,b)
    target = (train_day.energy/train_day.cap).clip(0,1.2).to_numpy(dtype='float32')
    y = np.zeros(shape[0],dtype='float32')
    w = np.zeros_like(y)
    y[train_idx] = target
    w[train_idx] = weights.astype('float32')
    missing = np.isnan(x[train_idx]).mean(axis=0)
    scaler = Scaler().fit(x[train_idx])
    np.savez(folder/'scaler.npz',median=scaler.median,mean=scaler.mean,std=scaler.std)
    for begin in range(0,len(x),100_000):
        x[begin:begin+100_000] = scaler.transform(x[begin:begin+100_000])
    x.flush()
    train_last = int((b-axis_start).total_seconds()/3600)
    eval_first = int((c-axis_start).total_seconds()/3600)
    starts = daily_starts(len(sites),n_hours,24,train_last)
    # A day with missing labels is kept when it has at least one observed daytime label.
    counts = (w.reshape(len(sites),-1,24)>0).sum(axis=-1)
    starts = starts[counts[:,1:train_last//24].ravel()>0]
    eval_starts = daily_starts(len(sites),n_hours,eval_first,n_hours)
    output = valid.merge(sites[['site_id','capacity_kw']],on='site_id',validate='many_to_one')
    output['daylight'] = daylight[valid_idx]
    output.to_parquet(folder/'evaluation.parquet',index=False)
    hours = protocol.complete_hours(valid,site_ids)
    pd.DataFrame({'ts':hours.astype(str)}).to_csv(folder/'hours.csv',index=False)
    expected = json.loads((original.ROOT/'docs/experiments/2026-11-03-p0-coverage.json').read_text())
    assert protocol.digest(folder/'hours.csv') == expected['windows'][window]['mask_sha256']
    for name, array in dict(y=y,w=w,train_rows=train_idx,eval_rows=valid_idx,
                             train_starts=starts,eval_starts=eval_starts).items():
        np.save(folder/f'{name}.npy',array)
    meta = dict(status='complete',window=window,round=round_hour,columns=columns,
        n_sites=len(sites),n_hours=n_hours,axis_start=str(axis_start),
        train_rows=len(train),train_daylight_rows=len(train_idx),eval_rows=len(valid),
        sequence_training_days=len(starts),train_weights=weight_info,
        train_missing_feature_fraction=dict(zip(columns,missing.tolist())),
        split=dict(train_start=str(a),train_end_exclusive=str(b),eval_start=str(c),eval_end_exclusive=str(d)),
        mask_sha256=protocol.digest(folder/'hours.csv'),common_hours=len(hours),
        target='(energy/cap).clip(0,1.2).astype(float32)',input_contains_targets=False,
        preprocessing_fit='observed daytime training rows only')
    write_json(manifest,meta)
    del x,y,w,train,valid
    gc.collect()
    return meta


class Batches:
    def __init__(self, folder, family):
        folder=Path(folder)
        self.meta=json.loads((folder/'metadata.json').read_text())
        self.family=family
        for name in ('x','y','w','train_rows','eval_rows','train_starts','eval_starts'):
            setattr(self,name,np.load(folder/f'{name}.npy',mmap_mode='r'))

    def training(self, selected):
        if self.family=='tabm':
            ix=self.train_rows[selected]
            return np.array(self.x[ix]),np.array(self.y[ix]),np.array(self.w[ix])
        start=self.train_starts[selected]
        context=sequence_indices(start)
        target=start[:,None]+np.arange(24)[None,:]
        return np.array(self.x[context]),np.array(self.y[target]),np.array(self.w[target])

    def evaluation(self, selected):
        if self.family=='tabm':
            return np.array(self.x[self.eval_rows[selected]])
        return np.array(self.x[sequence_indices(self.eval_starts[selected])])

    def align(self, pred):
        if self.family=='tabm':
            return pred
        # Each evaluation site/day has exactly 24 decoder positions.
        indices=(self.eval_starts[:,None]+np.arange(24)[None,:]).ravel()
        positions=np.searchsorted(indices,self.eval_rows)
        if not np.array_equal(indices[positions],self.eval_rows):
            raise ValueError('Predictions do not cover raw evaluation rows')
        return pred.reshape(-1)[positions]
