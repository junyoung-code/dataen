"""Evaluate the unchanged P4 model on the peer's annual folds and W mask.

No tuning, submission, network, or raw-data modification. The first prediction
is issued at May 31 17:00, so training stops there rather than the peer's June 1
00:00 cutoff. Peer scores are historical references, not locally rerun models.
"""
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
import pyarrow

import train_smoke as original
from features_p3 import build
from training_weights import make_weights

ROOT = Path(__file__).resolve().parents[1]
TZ = 'Asia/Seoul'
FOLD_YEARS = {'F2': 2023, 'F3': 2024}
PREDICTIONS = ('pred_10', 'pred_17', 'ensemble')


def bounds(fold):
    start = pd.Timestamp(f'{FOLD_YEARS[fold]}-06-01 01:00', tz=TZ)
    return (start-pd.DateOffset(years=2),
            start.normalize()-pd.Timedelta(hours=7),
            start, start+pd.DateOffset(years=1))


def parameters(threads):
    return dict(objective='regression_l1', n_estimators=200, learning_rate=0.03,
                num_leaves=127, min_child_samples=200, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, verbosity=-1,
                n_jobs=threads, random_state=0)


def peer_labels(energy, capacity):
    """Peer data.py: cap float32; exclude missing/negative/>1.05cap; y float32."""
    e = np.asarray(energy, dtype=np.float64)
    c = np.asarray(capacity, dtype=np.float32).astype(np.float64)
    has = np.isfinite(e) & (e >= 0) & (e <= c*1.05)
    return np.where(has, e, np.nan).astype(np.float32), has


def score(rows, sites, prices, start, end, prediction='ensemble', seen_only=False,
          raw_labels=False):
    """Sum signed errors inside each group, then abs and SMP*k; missing sites omitted."""
    if rows.duplicated(['site_id', 'ts']).any():
        raise ValueError('Duplicate site/time')
    if not np.isfinite(rows[prediction]).all():
        raise ValueError('Nonfinite prediction')
    axis = pd.date_range(start, end, freq='h', inclusive='left')
    if prices.ts.duplicated().any():
        raise ValueError('Duplicate prices')
    p = prices.set_index('ts').reindex(axis)
    smp = p.smp.to_numpy(dtype=float)
    if not np.isfinite(smp).all() or (smp < 0).any():
        raise ValueError('Missing/invalid evaluation price')
    k = p.k.to_numpy(dtype=float) if 'k' in p else np.where(axis.hour.isin(range(11,16)),3.,1.)
    w = smp*k
    if not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('Invalid evaluation weights')
    keep = np.isfinite(rows.energy_raw) if raw_labels else rows.has_peer.to_numpy(dtype=bool,copy=True)
    if seen_only:
        keep &= rows.seen.to_numpy(bool)
    v = rows.loc[keep, ['site_id','ts',prediction,'energy_raw','energy_peer']].copy()
    if not v.ts.isin(axis).all():
        raise ValueError('Rows outside evaluation period')
    mapping = sites.set_index('site_id').current_group
    v['group'] = v.site_id.map(mapping).map({'A':'G1','B':'G2'})
    if v.group.isna().any():
        raise ValueError('Unknown group')
    v['error'] = v[prediction]-v['energy_raw' if raw_labels else 'energy_peer'].astype(float)
    index = pd.MultiIndex.from_product([['G1','G2'],axis], names=['group','ts'])
    agg = v.groupby(['group','ts']).agg(error=('error','sum'), observed_sites=('site_id','nunique'))
    agg = agg.reindex(index, fill_value=0).reset_index()
    agg['weight'] = np.tile(w,2)
    agg['weighted_absolute_error'] = agg.error.abs()*agg.weight
    required = sites.groupby(sites.current_group.map({'A':'G1','B':'G2'})).site_id.nunique()
    agg['complete'] = agg.observed_sites.eq(agg.group.map(required))
    W = float(agg.weighted_absolute_error.sum())
    month = (agg.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')
    metrics = dict(W=W, W_1e8_won=W/1e8, score_deduction=0.1*W,
                   weight_sum=float(agg.weight.sum()), weighted_mae_kwh=W/agg.weight.sum(),
                   evaluation_hours=len(axis), observed_site_hours=len(v),
                   active_sites=int(v.site_id.nunique()), complete_group_hours=int(agg.complete.sum()),
                   W_by_group=agg.groupby('group').weighted_absolute_error.sum().to_dict(),
                   W_by_month=agg.groupby(month).weighted_absolute_error.sum().to_dict())
    return metrics, agg


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def worker(folder, fold, threads):
    started = time.perf_counter()
    a, cutoff, c, d = bounds(fold)
    b = original.load_baseline()
    sites = pd.read_csv(original.DATA/'sites.csv')
    gen = pd.read_parquet(original.DATA/'train/generation.parquet')
    gen.ts = pd.to_datetime(gen.ts).dt.tz_convert(TZ)
    if gen.duplicated(['site_id','ts']).any() or not np.isfinite(gen.energy).all():
        raise ValueError('Unexpected generation keys/labels')
    cap = gen.site_id.map(sites.set_index('site_id').capacity_kw)
    _, has = peer_labels(gen.energy, cap)
    # Evaluation diagnostic only; this set never enters the model's inputs.
    peer_end = c-pd.Timedelta(hours=1)
    seen = set(gen.loc[(gen.ts<=peer_end)&has,'site_id'])
    train = gen.loc[(gen.ts>=a)&(gen.ts<=cutoff)].copy()
    valid = gen.loc[(gen.ts>=c)&(gen.ts<d)].copy()
    excluded = gen.loc[(gen.ts>cutoff)&(gen.ts<=peer_end)]
    removed = len(excluded)
    lat = excluded.site_id.map(sites.set_index('site_id').lat).to_numpy()
    lon = excluded.site_id.map(sites.set_index('site_id').lon).to_numpy()
    elev,_,_ = b.solar(pd.DatetimeIndex(excluded.ts),lat,lon)
    removed_day = int((elev>0).sum())
    del elev,lat,lon,excluded,gen,cap,has
    prices = pd.read_csv(original.DATA/'train/smp.csv')
    prices.ts = pd.to_datetime(prices.ts).dt.tz_convert(TZ)
    fa = pd.read_parquet(original.DATA/'train/forecast_wa.parquet',columns=['wa_station','round','ts']+b.WA_COLS)
    fb = pd.read_parquet(original.DATA/'train/forecast_wb.parquet',columns=['wb_station','round','ts']+b.WB_COLS)
    for f in (fa,fb):
        f.ts = pd.to_datetime(f.ts).dt.tz_convert(TZ)
    fa = fa.loc[(fa.ts>=a)&(fa.ts<d)]
    fb = fb.loc[(fb.ts>=a)&(fb.ts<d)]
    output = valid.rename(columns={'energy':'energy_raw'}).merge(
        sites[['site_id','capacity_kw']],on='site_id',validate='many_to_one')
    output['energy_peer'],output['has_peer'] = peer_labels(output.energy_raw,output.capacity_kw)
    output['seen'] = output.site_id.isin(seen)
    params = parameters(threads)
    model_info = {}
    for r in (10,17):
        print(f'{fold} r{r}: build training features',flush=True)
        f, columns = build(b,train,sites,fa,fb,r,'station_distance')
        f = f.loc[f.elev>0]
        assert f.ts.max() <= cutoff
        x = f[columns].astype('float32')
        y = (f.energy/f.cap).clip(0,1.2).astype('float32')
        w, wi = make_weights(f,prices,'capacity',a,cutoff+pd.Timedelta(nanoseconds=1))
        t = time.perf_counter()
        model = lgb.LGBMRegressor(**params).fit(x,y,sample_weight=w)
        fit_seconds = time.perf_counter()-t
        model_path = folder/f'model_round_{r}.txt'
        model.booster_.save_model(str(model_path))
        assert model.booster_.num_trees()==200
        model_info[str(r)] = dict(train_rows=len(f),fit_seconds=fit_seconds,
                                 train_min=str(f.ts.min()),train_max=str(f.ts.max()),weights=wi,
                                 missing_fraction=x.isna().mean().to_dict())
        del f,x,y,w
        gc.collect()
        print(f'{fold} r{r}: predict; fit {fit_seconds:.2f}s',flush=True)
        f, cols = build(b,valid,sites,fa,fb,r,'station_distance')
        assert cols==columns
        assert f[['site_id','ts']].reset_index(drop=True).equals(output[['site_id','ts']].reset_index(drop=True))
        x = f[columns].astype('float32')
        raw = model.predict(x)
        restored = lgb.Booster(model_file=str(model_path))
        np.testing.assert_allclose(raw[:1024],restored.predict(x.iloc[:1024]),rtol=1e-12)
        output[f'pred_{r}'] = np.maximum(np.where(f.elev>0,raw,0),0)*f.cap.to_numpy()
        output['daylight'] = f.elev.to_numpy()>0
        del f,x,raw,restored,model
        gc.collect()
    output['ensemble'] = 0.25*output.pred_10+0.75*output.pred_17
    scores = {}
    for pred in PREDICTIONS:
        scores[pred], agg = score(output,sites,prices,c,d,prediction=pred)
        if pred=='ensemble':
            agg.to_csv(folder/'group_errors.csv',index=False)
    seen_score,_ = score(output,sites,prices,c,d,seen_only=True)
    raw_score,_ = score(output,sites,prices,c,d,raw_labels=True)
    output.to_parquet(folder/'site_predictions.parquet',index=False)
    report = dict(status='complete',fold=fold,train_start=str(a),train_end_inclusive=str(cutoff),
                  peer_train_end_inclusive=str(peer_end),eval_start=str(c),eval_end_exclusive=str(d),
                  removed_after_issuance_rows=removed,removed_after_issuance_daylight_rows=removed_day,
                  parameters=params,feature_columns=columns,weight_17=0.75,models=model_info,
                  train_raw_rows=len(train),eval_raw_rows=len(valid),
                  peer_invalid_eval_rows=int((~output.has_peer).sum()),
                  metrics=scores,seen_metrics=seen_score,raw_label_metrics=raw_score,
                  total_seconds=time.perf_counter()-started,peak_process_memory_mib=original.peak_memory_mib())
    write_json(folder/'summary.json',report)
    print(f'{fold} W={scores["ensemble"]["W"]:.3f}',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--fold',choices=list(FOLD_YEARS))
    args = p.parse_args()
    folder = args.output_dir.resolve()
    if not folder.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Results must stay inside outputs/')
    if args.fold:
        worker(folder,args.fold,args.threads)
        return
    folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    report=dict(status='running',protocol='peer-annual-w-causal-v1',
                note='P4 unchanged; peer annual validation/mask; training cutoff limited to first issuance. No tuning.',
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
                versions=dict(python=platform.python_version(),lightgbm=lgb.__version__,
                              numpy=np.__version__,pandas=pd.__version__,pyarrow=pyarrow.__version__),
                machine=dict(platform=platform.platform(),architecture=platform.machine()),folds={})
    source=folder/'source';source.mkdir()
    names=['src/run_peer_w.py','src/features_p3.py','src/training_weights.py','src/train_smoke.py','tests/test_peer_w.py']
    report['source_sha256']={n:original.sha256(ROOT/n) for n in names}
    for n in names:
        (source/n).parent.mkdir(parents=True,exist_ok=True)
        (source/n).write_bytes((ROOT/n).read_bytes())
    write_json(folder/'summary.json',report)
    original.self_test()
    original.check_data()
    report['data_manifest_verified']=True
    for fold in FOLD_YEARS:
        target=folder/fold;target.mkdir()
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--fold',fold,'--threads',str(args.threads),
             '--output-dir',str(target)]
        subprocess.run(cmd,check=True,cwd=ROOT)
        report['folds'][fold]=json.loads((target/'summary.json').read_text())
        write_json(folder/'summary.json',report)
    report.update(status='complete',total_seconds=time.perf_counter()-started)
    write_json(folder/'summary.json',report)


if __name__=='__main__':
    main()
