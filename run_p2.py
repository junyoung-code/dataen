"""Compare three blends of the P1-selected model; never train or inspect confirmation."""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import evaluate_temporal as evaluation
import validation_protocol as protocol
from run_p1 import config_id, single_group_weighted, write_json

ROOT = Path(__file__).resolve().parent
WEIGHTS = (0.5, 0.75, 1.0)


def blend(early, late, weight):
    a, b = np.asarray(early, dtype=float), np.asarray(late, dtype=float)
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Blend weight must be finite and between zero and one')
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Predictions must have equal shapes and finite values')
    return (1-weight)*a + weight*b


def choose(rows):
    if len(rows) != len(WEIGHTS) or {r['weight_17'] for r in rows} != set(WEIGHTS):
        raise ValueError('Expected all three distinct blend candidates')
    if not all(np.isfinite(r['weighted_group_mae']) for r in rows):
        raise ValueError('Missing/nonfinite selection score')
    best = min(r['weighted_group_mae'] for r in rows)
    return max((r for r in rows if r['weighted_group_mae'] <= best+1e-9),
               key=lambda r: r['weight_17'])


def metric(frame, pred):
    return evaluation.metrics(frame.energy, frame[pred], frame.capacity_kw)


def run(source, output):
    started = time.perf_counter()
    source, output = source.resolve(), output.resolve()
    # Keep derived outputs outside raw data and preserve existing results.
    output.relative_to((ROOT/'outputs').resolve())
    source.relative_to((ROOT/'outputs').resolve())
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit experiment code before running')
    p1 = json.loads((source/'summary.json').read_text())
    if p1['status'] != 'complete' or p1['protocol'] != protocol.PROTOCOL:
        raise ValueError('Completed P1 with the frozen protocol required')
    selected = p1['selected']
    from run_p1 import choose as select_p1
    if select_p1([r for r in p1['results'] if r['window']=='tune']) != selected:
        raise ValueError('P1 selection differs from its tuning results')
    config = config_id(selected['trees'],selected['learning_rate'])
    data = ROOT/'data'
    manifest = json.loads((ROOT/'data-manifest.json').read_text())
    hashes = {name:protocol.digest(data/name) for name in
              ['train/generation.parquet','sites.csv','train/smp.csv']}
    if any(hashes[name] != manifest[name]['sha256'] for name in hashes):
        raise ValueError('Source data differs from manifest')
    sites = pd.read_csv(data/'sites.csv')
    gen = pd.read_parquet(data/'train/generation.parquet',columns=['site_id','ts','energy'])
    gen.ts = pd.to_datetime(gen.ts).dt.tz_convert(protocol.TZ)
    prices = pd.read_csv(data/'train/smp.csv')
    prices.ts = pd.to_datetime(prices.ts).dt.tz_convert(protocol.TZ)
    expected = json.loads((ROOT/'docs/experiments/2026-11-03-p0-coverage.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    report = dict(status='running',protocol=protocol.PROTOCOL,code_commit=commit,dirty=False,
                  source_run=source.name,source_code_commit=p1['code_commit'],
                  model=dict(trees=selected['trees'],learning_rate=selected['learning_rate']),
                  training_performed=False,confirmation_evaluated=False,data_hashes=hashes,
                  versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
                  windows={})
    write_json(output/'summary.json',report)
    for window in ('tune','seasonal_diagnostic'):
        folder=output/window; folder.mkdir()
        input_path=source/window/config/'site_predictions.parquet'
        old=json.loads((input_path.parent/'summary.json').read_text())
        if old['status']!='complete' or old['window']!=window or old['protocol']!=protocol.PROTOCOL:
            raise ValueError('Invalid prediction provenance')
        if old['parameters']['n_estimators']!=selected['trees'] or old['parameters']['learning_rate']!=selected['learning_rate']:
            raise ValueError('Source model differs from P1 selection')
        frame=pd.read_parquet(input_path)
        train, valid=protocol.split(gen,window)
        raw=valid.set_index(['site_id','ts']).energy.sort_index()
        saved=frame.set_index(['site_id','ts']).energy.sort_index()
        if not raw.index.equals(saved.index) or not np.array_equal(raw.to_numpy(),saved.to_numpy()):
            raise ValueError('Saved observations differ from raw validation labels')
        capacity=frame.site_id.map(sites.set_index('site_id').capacity_kw)
        np.testing.assert_array_equal(capacity.to_numpy(),frame.capacity_kw.to_numpy())
        if (frame[['pred_10','pred_17']]<0).any().any() or not frame.daylight.isin([True,False]).all():
            raise ValueError('Invalid source predictions/daylight flags')
        if frame.loc[~frame.daylight,['pred_10','pred_17']].to_numpy().any():
            raise ValueError('Nonzero night prediction')
        hours=protocol.complete_hours(valid,sites.site_id)
        pd.DataFrame({'ts':hours.astype(str)}).to_csv(folder/'hours.csv',index=False)
        if protocol.digest(folder/'hours.csv')!=expected['windows'][window]['mask_sha256']:
            raise ValueError('P0 mask changed')
        cols={weight:f'blend_{weight:g}' for weight in WEIGHTS}
        for weight,col in cols.items(): frame[col]=blend(frame.pred_10,frame.pred_17,weight)
        np.testing.assert_allclose(frame[cols[0.5]],frame.ensemble,rtol=1e-12)
        groups=evaluation.aggregate_groups(frame,sites,list(cols.values()))
        groups=groups.merge(prices,on='ts',how='left',validate='many_to_one')
        fixed=groups.loc[groups.ts.isin(hours)]
        for f in (frame,groups): f['month']=(f.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')
        unseen=set(valid.site_id)-set(train.site_id)
        rows=[]; monthly=[]
        for weight,col in cols.items():
            score=protocol.weighted_group_mae(fixed,hours,col)
            rows.append(dict(weight_17=weight,weighted_group_mae=score,group=metric(fixed,col),
                             site_daylight=metric(frame.loc[frame.daylight],col),
                             unseen_daylight=metric(frame.loc[frame.daylight&frame.site_id.isin(unseen)],col),
                             by_group={g:dict(metrics=metric(f,col),weighted_mae=single_group_weighted(f,col))
                                       for g,f in fixed.groupby('group')}))
            for month,part in frame.groupby('month'):
                scopes=[('site_daylight',part.loc[part.daylight])]
                for g in ('G1','G2'):
                    scopes.append((f'reference_{g}_complete',groups.loc[(groups.month==month)&(groups.group==g)&groups.complete]))
                for scope,part in scopes:
                    monthly.append(dict(month=month,scope=scope,weight_17=weight,**metric(part,col),
                                        weighted_mae=single_group_weighted(part,col) if scope.startswith('reference_') else None))
        np.testing.assert_allclose(rows[0]['weighted_group_mae'],old['weighted_group_mae']['ensemble'],rtol=1e-12)
        groups.to_parquet(folder/'group_predictions.parquet',index=False)
        pd.DataFrame(monthly).to_csv(folder/'monthly_metrics.csv',index=False)
        report['windows'][window]=dict(common_hours=len(hours),eval_rows=len(frame),unseen_sites=len(unseen),
                                      mask_sha256=protocol.digest(folder/'hours.csv'),
                                      source_predictions_sha256=protocol.digest(input_path),
                                      candidates=rows,monthly=monthly)
    report.update(status='complete',selected=choose(report['windows']['tune']['candidates']),
                  selection_basis='tune weighted group MAE; ties within 1e-9 prefer larger weight_17',
                  total_seconds=time.perf_counter()-started)
    write_json(output/'summary.json',report)
    print(json.dumps({'status':report['status'],'selected':report['selected'],
                      'total_seconds':report['total_seconds']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    run(args.source_dir,args.output_dir)
