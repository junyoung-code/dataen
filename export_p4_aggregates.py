"""Read-only local P4 audit; export only team/time error aggregates, no sites."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    args.output_dir.resolve().relative_to((ROOT/'outputs').resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    contract=json.loads((ROOT/'AB_EVALUATION_CONTRACT.json').read_text())
    verification=args.baseline_dir/'verification.json'
    assert json.loads(verification.read_text())['status']=='passed'
    manifest=dict(kind='team-time aggregate errors only; no site predictions, labels or coordinates',
        contract_sha256=digest(ROOT/'AB_EVALUATION_CONTRACT.json'),exporter_sha256=digest(Path(__file__)),
        baseline_verification_sha256=digest(verification),pandas=pd.__version__,folds={})
    for fold,expected in contract['folds'].items():
        folder=args.baseline_dir/fold
        source=folder/'site_predictions.parquet'
        keys=pd.read_parquet(source,columns=['site_id','ts','has_peer'])
        keys=keys.sort_values(['site_id','ts']).reset_index(drop=True)
        mask=hashlib.sha256(keys.to_csv(index=False,lineterminator='\n').encode()).hexdigest()
        assert mask==expected['evaluation_key_mask_sha256'],f'{fold}: key/mask mismatch'
        assert len(keys)==expected['eval_raw_rows']
        assert int(keys.has_peer.sum())==expected['observed_site_hours']
        with (folder/'group_errors.csv').open() as f:raw=list(csv.DictReader(f))
        rows=[dict(group=x['group'],ts=x['ts'],error=float(x['error']),
            observed_sites=int(x['observed_sites']),weight=float(x['weight']),complete=x['complete']=='True') for x in raw]
        assert len(rows)==2*expected['evaluation_hours']
        assert len({(x['group'],x['ts']) for x in rows})==len(rows)
        assert sum(x['observed_sites'] for x in rows)==expected['observed_site_hours']
        W=math.fsum(abs(x['error'])*x['weight'] for x in rows)
        assert math.isclose(W,expected['baseline_W'],rel_tol=1e-12)
        for x,y in zip(rows,raw):
            assert math.isclose(abs(x['error'])*x['weight'],float(y['weighted_absolute_error']),rel_tol=1e-10,abs_tol=1e-7)
        path=args.output_dir/f'p4_{fold}_group_errors.json'
        path.write_text(json.dumps(dict(fold=fold,evaluation_key_mask_sha256=mask,W=W,rows=rows),allow_nan=False,separators=(',',':'))+'\n')
        manifest['folds'][fold]=dict(file=path.name,sha256=digest(path),W=W,evaluation_key_mask_sha256=mask,
            observed_site_hours=expected['observed_site_hours'],rows=len(rows),
            source_group_csv_sha256=digest(folder/'group_errors.csv'),source_prediction_sha256=digest(source),
            source_summary_sha256=digest(folder/'summary.json'))
        print(f'{fold}: P4 aggregate export verified; {len(rows)} team-hours; contract mask matched',flush=True)
    (args.output_dir/'p4_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
