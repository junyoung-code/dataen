"""Read-only P4 subset audit; export team aggregates without site rows."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from export_p4_aggregates import digest

ROOT=Path(__file__).resolve().parent


def mask_hash(rows):
    frame=rows.sort_values(['site_id','ts']).reset_index(drop=True)[['site_id','ts','has_peer','seen']]
    return hashlib.sha256(frame.to_csv(index=False,lineterminator='\n').encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-dir',type=Path,required=True)
    p.add_argument('--sites-file',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.resolve().relative_to((ROOT/'outputs').resolve());args.output_dir.mkdir(parents=True,exist_ok=True)
    contract=json.loads((ROOT/'AB_EVALUATION_CONTRACT.json').read_text())
    old=json.loads((ROOT/'outputs/a234_inputs/p4_manifest.json').read_text())
    assert digest(args.sites_file)==contract['sites_file_sha256']
    sites=pd.read_csv(args.sites_file);mapping=sites.set_index('site_id').current_group.map({'A':'G1','B':'G2'})
    manifest=dict(contract_sha256=digest(ROOT/'AB_EVALUATION_CONTRACT.json'),exporter_sha256=digest(Path(__file__)),
        subset_definition='existing seen flag: valid observations before evaluation start; diagnostic only',folds={})
    for fold,expected in contract['folds'].items():
        path=args.baseline_dir/fold/'site_predictions.parquet'
        assert digest(path)==old['folds'][fold]['source_prediction_sha256']
        rows=pd.read_parquet(path,columns=['site_id','ts','has_peer','seen','energy_peer','ensemble'])
        frame=rows.sort_values(['site_id','ts']).reset_index(drop=True)
        assert hashlib.sha256(frame[['site_id','ts','has_peer']].to_csv(index=False,lineterminator='\n').encode()).hexdigest()==expected['evaluation_key_mask_sha256']
        axis=pd.date_range(expected['eval_start'],expected['eval_end_exclusive'],freq='h',inclusive='left')
        index=pd.MultiIndex.from_product([['G1','G2'],axis],names=['group','ts'])
        agg=pd.DataFrame(index=index)
        rows['group']=rows.site_id.map(mapping);rows['error']=rows.ensemble-rows.energy_peer.astype(float)
        for subset in ['all','seen','new']:
            keep=rows.has_peer.to_numpy(bool).copy()
            if subset!='all':keep &= rows.seen.to_numpy(bool) if subset=='seen' else ~rows.seen.to_numpy(bool)
            part=rows.loc[keep].groupby(['group','ts']).agg(error=('error','sum'),observed=('site_id','size')).reindex(index,fill_value=0)
            agg[f'error_{subset}']=part.error;agg[f'observed_{subset}']=part.observed
        np.testing.assert_allclose(agg.error_all,agg.error_seen+agg.error_new,rtol=1e-12,atol=1e-8)
        np.testing.assert_array_equal(agg.observed_all,agg.observed_seen+agg.observed_new)
        baseline=pd.DataFrame(json.loads((ROOT/'outputs/a234_inputs'/old['folds'][fold]['file']).read_text())['rows'])
        baseline.ts=pd.to_datetime(baseline.ts);baseline=baseline.set_index(['group','ts']).reindex(index)
        agg['weight']=baseline.weight
        np.testing.assert_allclose(agg.error_all,baseline.error,rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(np.sum(np.abs(agg.error_all)*agg.weight),expected['baseline_W'],rtol=1e-12)
        output=args.output_dir/f'p4_{fold}_subsets.json'
        output.write_text(agg.reset_index().to_json(orient='records',date_format='iso',date_unit='ns',double_precision=15)+'\n')
        manifest['folds'][fold]=dict(file=output.name,sha256=digest(output),source_prediction_sha256=digest(path),
            key_seen_sha256=mask_hash(rows),evaluation_key_mask_sha256=expected['evaluation_key_mask_sha256'],
            observations={s:int(agg[f'observed_{s}'].sum()) for s in ['all','seen','new']})
        print(f'{fold}: P4 all/seen/new aggregation and original mask verified',flush=True)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
