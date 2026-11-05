"""Supplementary independent training-balance and transformed-input audit."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import train_smoke as original
from run_peer_w import TZ,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--a1-dir',type=Path,required=True);args=p.parse_args()
    report=json.loads((args.output_dir/'summary.json').read_text());assert report['status']=='complete'
    sites=pd.read_csv(original.DATA/'sites.csv');site_order={s:i for i,s in enumerate(sites.site_id)}
    result=dict(status='running',source_name=Path(__file__).name,source_sha256=original.sha256(Path(__file__)),folds={})
    for fold in ['F2','F3']:
        cache=args.cache_dir/fold/'r17';meta=json.loads((cache/'metadata.json').read_text())
        train=np.load(cache/'train_rows.npy',mmap_mode='r');site_idx=train//meta['n_hours']
        known=set(sites.site_id.to_numpy()[np.unique(site_idx)])
        balance={group:dict(supervised_rows=int(np.sum(sites.current_group.to_numpy()[site_idx]==label)),
            supervised_sites=int(np.sum((sites.current_group==label)&sites.site_id.isin(known)))) for group,label in [('G1','A'),('G2','B')]}
        rows=pd.read_parquet(args.a1_dir/'tft'/fold/'site_predictions.parquet',columns=['site_id','ts','daylight','has_peer'])
        positions=rows.site_id.map(site_order).to_numpy()*meta['n_hours']+((rows.ts-pd.Timestamp(meta['axis_start'])).dt.total_seconds()/3600).to_numpy(dtype=int)
        x=np.load(cache/'x.npy',mmap_mode='r');clipping={}
        for label,history in [('known_G2',True),('cold_G2',False)]:
            keep=rows.daylight&rows.has_peer&rows.site_id.map(sites.set_index('site_id').current_group).eq('B')
            keep &= rows.site_id.isin(known) if history else ~rows.site_id.isin(known)
            values=x[positions[keep.to_numpy()]]
            fractions=np.count_nonzero(np.abs(values)>=9.99999,axis=0)/len(values)
            clipping[label]={name:float(f) for name,f in zip(meta['columns'],fractions) if f>0}
            assert clipping[label]==report['folds'][fold]['training_support']['transformed_feature_clip_fraction'][label]
        result['folds'][fold]=dict(training_balance=balance,independent_clip_fraction=clipping,
            metadata_sha256=original.sha256(cache/'metadata.json'),transformed_grid_sha256=original.sha256(cache/'x.npy'))
        print(f'{fold}: supervised sites G1={balance["G1"]["supervised_sites"]}, G2={balance["G2"]["supervised_sites"]}; independent clip fractions match',flush=True)
    result['status']='complete';write_json(args.output_dir/'distribution_audit.json',result)
    (args.output_dir/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
