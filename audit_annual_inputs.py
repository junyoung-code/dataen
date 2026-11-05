"""Supplement: raw full daylight mask, train input samples and saved group files."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import train_smoke as original
from annual_neural_data import check_contract
from features_p3 import build
from neural_data import Batches
from run_peer_w import TZ, bounds, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    args=parser.parse_args()
    check_contract()
    baseline=original.load_baseline()
    sites=pd.read_csv(original.DATA/'sites.csv')
    gen=pd.read_parquet(original.DATA/'train/generation.parquet')
    gen.ts=pd.to_datetime(gen.ts).dt.tz_convert(TZ)
    fa=pd.read_parquet(original.DATA/'train/forecast_wa.parquet',columns=['wa_station','round','ts']+baseline.WA_COLS)
    fb=pd.read_parquet(original.DATA/'train/forecast_wb.parquet',columns=['wb_station','round','ts']+baseline.WB_COLS)
    for frame in (fa,fb):frame.ts=pd.to_datetime(frame.ts).dt.tz_convert(TZ)
    result=dict(status='running',source_sha256=original.sha256(Path(__file__)),folds={})
    for fold in ('F2','F3'):
        _,_,c,d=bounds(fold)
        raw=gen.loc[(gen.ts>=c)&(gen.ts<d)]
        lat=raw.site_id.map(sites.set_index('site_id').lat).to_numpy()
        lon=raw.site_id.map(sites.set_index('site_id').lon).to_numpy()
        elevation,_,_=baseline.solar(pd.DatetimeIndex(raw.ts),lat,lon)
        daytime=elevation>0
        for r in (10,17):
            cache=args.cache_dir/fold/f'r{r}'
            data=Batches(cache,'tabm')
            chosen=np.linspace(0,len(data.train_rows)-1,512,dtype=int)
            indices=data.train_rows[chosen]
            sample=pd.DataFrame(dict(site_id=sites.site_id.to_numpy()[indices//data.meta['n_hours']],
                ts=pd.Timestamp(data.meta['axis_start'])+pd.to_timedelta(indices%data.meta['n_hours'],unit='h'),energy=0.))
            f,cols=build(baseline,sample,sites,fa,fb,r,'station_distance')
            scaler=np.load(cache/'scaler.npz')
            x=f[cols].to_numpy(dtype='float32')
            x=np.where(np.isfinite(x),x,scaler['median'])
            standardized=np.clip((x-scaler['mean'])/scaler['std'],-10,10).astype('float32')
            np.testing.assert_array_equal(data.x[indices],standardized)
        for family in ('tabm','tft'):
            folder=args.output_dir/family/fold
            rows=pd.read_parquet(folder/'site_predictions.parquet')
            np.testing.assert_array_equal(rows.daylight,daytime)
            groups=pd.read_csv(folder/'group_errors.csv')
            summary=json.loads((folder/'summary.json').read_text())['metrics']['ensemble']
            assert len(groups)==2*summary['evaluation_hours']
            np.testing.assert_allclose(groups.weighted_absolute_error,groups.error.abs()*groups.weight,rtol=1e-12,atol=1e-6)
            np.testing.assert_allclose(groups.weighted_absolute_error.sum(),summary['W'],rtol=1e-12)
        result['folds'][fold]=dict(full_raw_daylight_rows=len(raw),training_feature_samples_per_round=512,
            all_daylight_masks_match_raw_solar=True,saved_group_errors_match_W=True)
        print(f'{fold}: full daylight mask, train input samples, saved group errors verified',flush=True)
    result['status']='complete'
    write_json(args.output_dir/'supplemental_verification.json',result)


if __name__=='__main__':main()
