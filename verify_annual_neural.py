"""Independent raw-label, training-boundary, NumPy W and checkpoint audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import train_smoke as original
from annual_neural_data import check_contract, key_mask_digest
from features_p3 import build
from neural_data import Batches
from neural_models import Scaler, make_model, predictions
from run_peer_w import TZ, bounds, write_json

ROOT=Path(__file__).resolve().parent


def audit_cache(cache,fold,r,gen,sites,fa,fb,baseline,expected):
    folder=cache/fold/f'r{r}'
    data=Batches(folder,'tft')
    meta=data.meta
    a,cutoff,c,d=bounds(fold)
    train=gen.loc[(gen.ts>=a)&(gen.ts<=cutoff)].copy()
    f,columns=build(baseline,train,sites,fa,fb,r,'station_distance')
    f=f.loc[f.elev>0].copy()
    site_order={s:i for i,s in enumerate(sites.site_id)}
    axis_start=pd.Timestamp(meta['axis_start'])
    indices=f.site_id.map(site_order).to_numpy()*meta['n_hours']+((f.ts-axis_start).dt.total_seconds()/3600).to_numpy(dtype='int64')
    np.testing.assert_array_equal(indices,data.train_rows)
    assert len(indices)==(476693 if fold=='F2' else 962913)
    y=(f.energy/f.cap).clip(0,1.2).to_numpy(dtype='float32')
    np.testing.assert_array_equal(data.y[indices],y)
    cap=f.cap.to_numpy(dtype='float64')
    w=(cap/cap.mean()).astype('float32')
    np.testing.assert_array_equal(data.w[indices],w)
    assert int((data.w>0).sum())==len(indices)
    reconstructed=Scaler().fit(f[columns].to_numpy(dtype='float32'))
    saved=np.load(folder/'scaler.npz')
    for name in ('median','mean','std'):
        np.testing.assert_array_equal(getattr(reconstructed,name),saved[name])
    last=int((cutoff-axis_start).total_seconds()/3600)
    assert not data.w.reshape(meta['n_sites'],meta['n_hours'])[:,last+1:].any()
    assert not data.y.reshape(meta['n_sites'],meta['n_hours'])[:,last+1:].any()
    decoder=(data.train_starts[:,None]+np.arange(24)[None,:]).ravel()
    positive=decoder[data.w[decoder]>0]
    np.testing.assert_array_equal(np.sort(positive),np.sort(indices))
    # Rebuild independent raw feature samples throughout train/eval and boundary.
    chosen=np.unique(np.r_[np.linspace(0,len(data.eval_rows)-1,512,dtype=int)])
    grid_idx=data.eval_rows[chosen]
    sample=pd.DataFrame(dict(site_id=sites.site_id.to_numpy()[grid_idx//meta['n_hours']],
        ts=axis_start+pd.to_timedelta(grid_idx%meta['n_hours'],unit='h'),energy=0.))
    sample_f,cols=build(baseline,sample,sites,fa,fb,r,'station_distance')
    np.testing.assert_array_equal(data.x[grid_idx],reconstructed.transform(sample_f[cols].to_numpy(dtype='float32')))
    return dict(daylight_labels=len(indices),post_cutoff_positive_labels=0,
        sequence_labels_exactly_once=True,training_targets_weights_scaler_match_raw=True,
        raw_feature_samples=len(chosen),train_max=str(f.ts.max()))


def numpy_score(rows,sites,prices,c,d):
    axis=pd.date_range(c,d,freq='h',inclusive='left')
    price=prices.set_index('ts').reindex(axis)
    k=price.k.to_numpy() if 'k' in price else np.where(axis.hour.isin(range(11,16)),3.,1.)
    weight=price.smp.to_numpy(dtype=float)*k
    group=rows.site_id.map(sites.set_index('site_id').current_group).map({'A':0,'B':1}).to_numpy()
    hour=((rows.ts-c).dt.total_seconds()/3600).to_numpy(dtype=int)
    valid=rows.has_peer.to_numpy(bool)
    error=rows.ensemble.to_numpy(dtype=float)-rows.energy_peer.to_numpy(dtype=float)
    totals=np.zeros((2,len(axis)),dtype='float64')
    np.add.at(totals,(group[valid],hour[valid]),error[valid])
    amounts=np.abs(totals)*weight
    months=(axis-pd.Timedelta(hours=1)).strftime('%Y-%m')
    return dict(W=float(amounts.sum()),W_by_group={g:float(amounts[i].sum()) for i,g in enumerate(['G1','G2'])},
        W_by_month={m:float(amounts[:,months==m].sum()) for m in sorted(set(months))})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    args=p.parse_args()
    contract=check_contract()
    report=json.loads((args.output_dir/'summary.json').read_text())
    assert report['status']=='trained_pending_independent_verification'
    for name,h in report['source_sha256'].items():
        assert original.sha256(ROOT/name)==h
        assert original.sha256(args.output_dir/'source'/name)==h
    gen=pd.read_parquet(original.DATA/'train/generation.parquet')
    gen.ts=pd.to_datetime(gen.ts).dt.tz_convert(TZ)
    sites=pd.read_csv(original.DATA/'sites.csv')
    prices=pd.read_csv(original.DATA/'train/smp.csv')
    prices.ts=pd.to_datetime(prices.ts).dt.tz_convert(TZ)
    baseline=original.load_baseline()
    fa=pd.read_parquet(original.DATA/'train/forecast_wa.parquet',columns=['wa_station','round','ts']+baseline.WA_COLS)
    fb=pd.read_parquet(original.DATA/'train/forecast_wb.parquet',columns=['wb_station','round','ts']+baseline.WB_COLS)
    for frame in (fa,fb):frame.ts=pd.to_datetime(frame.ts).dt.tz_convert(TZ)
    audit=dict(status='running',source_hashes_verified=True,caches={},results={})
    for fold in ('F2','F3'):
        expected=contract['folds'][fold]
        for r in (10,17):
            audit['caches'][f'{fold}/r{r}']=audit_cache(args.cache_dir,fold,r,gen,sites,fa,fb,baseline,expected)
            print(f'{fold}/r{r}: raw labels/weights/scaler/features/sequence boundary verified',flush=True)
        _,_,c,d=bounds(fold)
        raw=gen.loc[(gen.ts>=c)&(gen.ts<d)].rename(columns={'energy':'energy_raw'})
        for family in ('tabm','tft'):
            dest=args.output_dir/family/fold
            saved=json.loads((dest/'summary.json').read_text())
            rows=pd.read_parquet(dest/'site_predictions.parquet')
            assert key_mask_digest(rows)==expected['evaluation_key_mask_sha256']
            joined=rows.merge(raw,on=['site_id','ts'],suffixes=('','_original'),validate='one_to_one')
            assert len(joined)==len(raw)==expected['eval_raw_rows']
            np.testing.assert_array_equal(joined.energy_raw,joined.energy_raw_original)
            cap=rows.site_id.map(sites.set_index('site_id').capacity_kw).to_numpy(dtype='float32').astype('float64')
            energy=rows.energy_raw.to_numpy(dtype='float64')
            has=np.isfinite(energy)&(energy>=0)&(energy<=cap*1.05)
            np.testing.assert_array_equal(rows.has_peer,has)
            np.testing.assert_array_equal(rows.energy_peer,np.where(has,energy,np.nan).astype('float32'))
            assert int(has.sum())==expected['observed_site_hours']
            assert np.isfinite(rows[['pred_10','pred_17','ensemble']]).all().all()
            assert (rows[['pred_10','pred_17','ensemble']]>=0).all().all()
            assert not rows.loc[~rows.daylight,['pred_10','pred_17','ensemble']].to_numpy().any()
            np.testing.assert_array_equal(rows.ensemble,.25*rows.pred_10+.75*rows.pred_17)
            recomputed=numpy_score(rows,sites,prices,c,d)
            for metric in ('W_by_group','W_by_month'):
                for key,value in recomputed[metric].items():
                    np.testing.assert_allclose(value,saved['metrics']['ensemble'][metric][key],rtol=1e-12,atol=1e-5)
            np.testing.assert_allclose(recomputed['W'],saved['metrics']['ensemble']['W'],rtol=1e-12,atol=1e-5)
            for r in (10,17):
                data=Batches(args.cache_dir/fold/f'r{r}',family)
                epoch=10 if family=='tabm' else 20
                checkpoint=torch.load(dest/f'model_round_{r}_e{epoch:03}.pt',map_location='cpu',weights_only=True)
                assert checkpoint['epochs']==epoch and checkpoint['columns']==expected['feature_columns']
                model=make_model(family,checkpoint['columns']).to('mps')
                model.load_state_dict(checkpoint['state_dict']);model.eval()
                old=np.load(dest/f'prediction_round_{r}_e{epoch:03}.npy')
                count=len(data.eval_rows) if family=='tabm' else len(data.eval_starts)
                selected=np.linspace(0,count-1,512 if family=='tabm' else 32,dtype=int)
                with torch.inference_mode():
                    prediction=predictions(model,torch.from_numpy(data.evaluation(selected)).to('mps'),family)
                    if family=='tabm':prediction=prediction.mean(dim=1)
                    prediction=prediction.cpu().numpy()
                if family=='tabm':
                    np.testing.assert_allclose(prediction,old[selected],rtol=2e-4,atol=2e-5)
                else:
                    indices=(data.eval_starts[selected,None]+np.arange(24)).ravel()
                    # Select existing evaluation labels in the sampled full decoder days.
                    common,positions,rawpos=np.intersect1d(indices,data.eval_rows,return_indices=True)
                    assert len(common)>0
                    np.testing.assert_allclose(prediction.ravel()[positions],old[rawpos],rtol=2e-4,atol=2e-5)
                restored=np.maximum(np.where(rows.daylight,old.astype('float64'),0.),0.)*cap.astype('float32')
                np.testing.assert_array_equal(restored,rows[f'pred_{r}'])
                history=json.loads((dest/f'training_round_{r}.json').read_text())
                assert [v['epoch'] for v in history]==list(range(1,epoch+1))
                del model
                torch.mps.empty_cache()
            audit['results'][f'{family}/{fold}']=dict(**recomputed,raw_labels_and_mask_verified=True,
                saved_checkpoint_samples_verified=True,night_zero_blend_and_capacity_verified=True)
            print(f'{family}/{fold}: independent NumPy W and both checkpoints verified',flush=True)
        write_json(args.output_dir/'verification.json',audit)
    original.check_data()
    audit['status']='complete'
    write_json(args.output_dir/'verification.json',audit)
    print('Annual independent verification complete',flush=True)


if __name__=='__main__':main()
