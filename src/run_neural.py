"""Train TabM/LSTM/TFT from scratch under the existing historical protocol."""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import evaluate_temporal as evaluation
import train_smoke as original
import validation_protocol as protocol
from neural_data import Batches, prepare
from neural_models import CONFIGS, EPOCHS, make_model, predictions, weighted_loss
from run_p1 import PREDICTIONS, metric_table, single_group_weighted, write_json

ROOT=Path(__file__).resolve().parents[1]
BATCH_SIZE={'tabm':4096,'lstm':512,'tft':1024}


def synchronize(device):
    if device=='mps':
        torch.mps.synchronize()


def choose(rows):
    if len(rows)!=len(EPOCHS) or {r['epochs'] for r in rows}!=set(EPOCHS):
        raise ValueError('All epoch candidates must be present exactly once')
    if any(r['window']!='tune' or not np.isfinite(r['weighted_group_mae']) for r in rows):
        raise ValueError('Choose using finite tune metrics only')
    return min(rows,key=lambda r:(r['weighted_group_mae'],r['epochs']))


def predict(model,batches,family,device,batch_size):
    model.eval()
    n=len(batches.eval_rows) if family=='tabm' else len(batches.eval_starts)
    outputs=[]
    with torch.inference_mode():
        for begin in range(0,n,batch_size):
            x=torch.from_numpy(batches.evaluation(slice(begin,begin+batch_size))).to(device)
            p=predictions(model,x,family)
            if family=='tabm':
                p=p.mean(dim=1)
            outputs.append(p.cpu().numpy())
    aligned=batches.align(np.concatenate(outputs))
    if not np.isfinite(aligned).all():
        raise ValueError('Nonfinite predictions')
    return aligned


def fit_round(folder,cache,family,round_hour,epochs,device,threads):
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    torch.set_num_threads(threads)
    data=Batches(cache,family)
    model=make_model(family,data.meta['columns']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    n=len(data.train_rows) if family=='tabm' else len(data.train_starts)
    batch_size=BATCH_SIZE[family]
    rng=np.random.default_rng(0)
    history=[]; checkpoints={}; fit_seconds=0.
    started=time.perf_counter()
    for epoch in range(1,max(epochs)+1):
        model.train(); t=time.perf_counter(); order=rng.permutation(n)
        error_sum=torch.zeros((),device=device);weight_sum=torch.zeros((),device=device)
        for begin in range(0,n,batch_size):
            x,y,w=(torch.from_numpy(v).to(device) for v in data.training(order[begin:begin+batch_size]))
            optimizer.zero_grad(set_to_none=True)
            p=predictions(model,x,family)
            loss=weighted_loss(p,y,w,family)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                error_sum+=loss.detach()*w.sum();weight_sum+=w.sum()
        synchronize(device)
        elapsed=time.perf_counter()-t;fit_seconds+=elapsed
        train_loss=float((error_sum/weight_sum).cpu())
        if not np.isfinite(train_loss):
            raise ValueError('Nonfinite training loss')
        row=dict(epoch=epoch,weighted_training_mae=train_loss,seconds=elapsed)
        history.append(row)
        write_json(folder/f'training_round_{round_hour}.json',history)
        print(f'{family}/{folder.name}/r{round_hour} epoch {epoch}/{max(epochs)} loss={train_loss:.6f} {elapsed:.1f}s',flush=True)
        if epoch in epochs:
            model_path=folder/f'model_round_{round_hour}_e{epoch:03}.pt'
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save(dict(state_dict=state,family=family,config=CONFIGS[family],columns=data.meta['columns'],epochs=epoch),model_path)
            raw=predict(model,data,family,device,batch_size)
            np.save(folder/f'prediction_round_{round_hour}_e{epoch:03}.npy',raw)
            # Reconstruct the architecture and compare actual saved weights.
            cpu_rng=torch.get_rng_state()
            mps_rng=torch.mps.get_rng_state() if device=='mps' else None
            restored=make_model(family,data.meta['columns']).to(device)
            restored.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True)['state_dict'])
            restored.eval()
            with torch.inference_mode():
                check=torch.from_numpy(data.evaluation(slice(0,min(32,len(data.eval_starts))))).to(device)
                np.testing.assert_allclose(predictions(restored,check,family).cpu().numpy(),
                    predictions(model,check,family).cpu().numpy(),rtol=1e-5,atol=1e-6)
            torch.set_rng_state(cpu_rng)
            if mps_rng is not None:torch.mps.set_rng_state(mps_rng)
            checkpoints[str(epoch)]=dict(fit_seconds=fit_seconds,weighted_training_mae=train_loss,
                parameter_count=sum(p.numel() for p in model.parameters()),saved_model_reload=True)
            del raw,restored,state
            gc.collect()
    result=dict(rows=data.meta['train_daylight_rows'],weights=data.meta['train_weights'],
        input_examples_per_epoch=n,batch_size=batch_size,history=history,checkpoints=checkpoints,
        train_missing_feature_fraction=data.meta['train_missing_feature_fraction'],
        seconds_with_prediction=time.perf_counter()-started)
    write_json(folder/f'round_{round_hour}.json',result)
    del data,model,optimizer
    gc.collect()
    if device=='mps':torch.mps.empty_cache()
    return result


def evaluate_checkpoint(folder,cache_root,window,family,epoch,training,elapsed):
    dest=folder/f'e{epoch:03}';dest.mkdir()
    meta=json.loads((cache_root/window/'r10/metadata.json').read_text())
    output=pd.read_parquet(cache_root/window/'r10/evaluation.parquet')
    sites=pd.read_csv(original.DATA/'sites.csv')
    prices=pd.read_csv(original.DATA/'train/smp.csv')
    prices.ts=pd.to_datetime(prices.ts).dt.tz_convert(protocol.TZ)
    hours=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(cache_root/window/'r10/hours.csv').ts))
    for r in (10,17):
        raw=np.load(folder/f'prediction_round_{r}_e{epoch:03}.npy')
        # Match the original pipeline's float32 cap, not a new target definition.
        output[f'pred_{r}']=np.maximum(np.where(output.daylight,raw,0.),0.)*output.capacity_kw.to_numpy(dtype='float32')
    output['ensemble']=output.pred_10*.25+output.pred_17*.75
    groups=evaluation.aggregate_groups(output,sites,PREDICTIONS)
    groups=groups.merge(prices,on='ts',how='left',validate='many_to_one')
    fixed=groups.loc[groups.ts.isin(hours)]
    scores={p:protocol.weighted_group_mae(fixed,hours,p) for p in PREDICTIONS}
    for frame in (output,groups):frame['month']=(frame.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')
    monthly=[]
    for month, rows in fixed.groupby((fixed.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')):
        h=pd.DatetimeIndex(rows.ts.unique()).sort_values()
        monthly.append(dict(month=month,common_hours=len(h),weighted_group_mae=protocol.weighted_group_mae(rows,h,'ensemble')))
    output.to_parquet(dest/'site_predictions.parquet',index=False,compression='zstd')
    groups.to_parquet(dest/'group_predictions.parquet',index=False,compression='zstd')
    pd.DataFrame(monthly).to_csv(dest/'monthly_metrics.csv',index=False)
    report=dict(status='complete',window=window,variant=f'{family}_e{epoch:03}',family=family,epochs=epoch,
        protocol=protocol.PROTOCOL,feature_columns=meta['columns'],weight_17=.75,weighting='capacity',
        split=meta['split'],train_rows=meta['train_rows'],eval_rows=meta['eval_rows'],common_hours=meta['common_hours'],
        mask_sha256=meta['mask_sha256'],weighted_group_mae=scores,group_fixed=metric_table(fixed),
        site_daylight=metric_table(output.loc[output.daylight]),monthly_common=monthly,
        by_group={g:single_group_weighted(rows,'ensemble') for g,rows in fixed.groupby('group')},
        training={r:dict(rows=t['rows'],weights=t['weights'],**t['checkpoints'][str(epoch)]) for r,t in training.items()},
        peak_memory_mib=original.peak_memory_mib(),window_seconds_including_all_checkpoints=elapsed,
        evaluation_labels_seen_previously=True,
        sequence_context_hours=0 if family=='tabm' else 24,preprocessing_fit='training observed daytime rows only')
    write_json(dest/'summary.json',report)
    print(f'{window}/{family}/e{epoch:03}: team weighted MAE {scores["ensemble"]:.6f}',flush=True)
    return dict(window=window,family=family,epochs=epoch,weighted_group_mae=scores['ensemble'],
        fit_seconds=sum(v['checkpoints'][str(epoch)]['fit_seconds'] for v in training.values()),
        output=str(dest),peak_memory_mib=report['peak_memory_mib'])


def worker(folder,cache_root,window,family,epochs,device,threads):
    started=time.perf_counter();training={}
    for r in (10,17):
        cache=cache_root/window/f'r{r}'
        print(f'{family}/{window}: prepare round {r}',flush=True)
        meta=prepare(cache,window,r)
        # Check against prior results before new training.
        history=json.loads((ROOT/'docs/experiments/models/2026-11-03-xgboost.json').read_text())
        expected=next(x['summary'] for x in history['details'] if x['window']==window and x['variant']=='lightgbm_p4')
        for k in ('split','train_rows','eval_rows','common_hours','mask_sha256'):
            if meta[k]!=expected[k]:raise ValueError(f'Comparison condition changed: {k}')
        assert meta['columns']==expected['feature_columns']
        assert meta['train_weights']==expected['training'][str(r)]['weights']
        assert meta['train_daylight_rows']==expected['training'][str(r)]['rows']
        training[str(r)]=fit_round(folder,cache,family,r,epochs,device,threads)
    rows=[evaluate_checkpoint(folder,cache_root,window,family,e,training,time.perf_counter()-started) for e in epochs]
    write_json(folder/'summary.json',dict(status='complete',results=rows,total_seconds=time.perf_counter()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=CONFIGS,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','mps'],default='mps')
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--window',choices=protocol.WINDOWS,default='tune')
    parser.add_argument('--epochs',type=int,nargs='+',default=list(EPOCHS))
    args=parser.parse_args()
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no silent CPU fallback')
    folder=args.output_dir.resolve();folder.relative_to((ROOT/'outputs').resolve())
    cache=args.cache_dir.resolve();cache.relative_to((ROOT/'outputs').resolve())
    folder.mkdir(parents=True,exist_ok=False)
    if args.worker:
        try:worker(folder,cache,args.window,args.family,args.epochs,args.device,args.threads)
        except BaseException as e:
            write_json(folder/'failure.json',dict(error=str(e)));raise
        return
    original.check_data()
    files=sorted((ROOT/'src').glob('*.py')) + sorted((ROOT/'tests').glob('*.py'))+[ROOT/'requirements/requirements-neural.txt',ROOT/'docs/experiments/models/neural-design.md']
    hashes={str(p.relative_to(ROOT)):protocol.digest(p) for p in files}
    report=dict(status='running',family=args.family,config=CONFIGS[args.family],epochs=list(EPOCHS),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),source_hashes=hashes,
        machine=dict(chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
            memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),macos=platform.mac_ver()[0]),
        device=args.device,threads=args.threads,
        versions={p:importlib.metadata.version(p) for p in ['torch','tabm','pytorch-forecasting','numpy','pandas','lightgbm']},
        python=platform.python_version(),results=[])
    write_json(folder/'summary.json',report)
    snapshot=folder/'source';snapshot.mkdir()
    for path in files:
        target=snapshot/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    started=time.perf_counter()
    try:
        for window in ('tune','seasonal_diagnostic','confirmation'):
            epochs=list(EPOCHS) if window=='tune' else [report['selected_epochs']]
            if window!='tune' and protocol.digest(folder/'selection.json')!=report['selection_sha256']:
                raise ValueError('Frozen selection changed')
            dest=folder/window
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--family',args.family,
                '--output-dir',str(dest),'--cache-dir',str(cache),'--window',window,
                '--epochs',*[str(e) for e in epochs],'--device',args.device,'--threads',str(args.threads)],check=True)
            rows=json.loads((dest/'summary.json').read_text())['results']
            report['results']+=rows
            if window=='tune':
                chosen=choose(rows);report['selected_epochs']=chosen['epochs']
                write_json(folder/'selection.json',dict(family=args.family,config=CONFIGS[args.family],
                    selected_epochs=chosen['epochs'],selection_window='tune',score=chosen['weighted_group_mae'],
                    source_hashes=hashes,sites_sha256=protocol.digest(original.DATA/'sites.csv')))
                report['selection_sha256']=protocol.digest(folder/'selection.json')
            write_json(folder/'summary.json',report)
        if any(protocol.digest(ROOT/p)!=v for p,v in hashes.items()):raise ValueError('Source changed during run')
        if protocol.digest(folder/'selection.json')!=report['selection_sha256']:raise ValueError('Selection changed')
        report['status']='complete'
    except BaseException as e:
        report['status']='failed';report['error']=str(e);raise
    finally:
        report['total_seconds']=time.perf_counter()-started;write_json(folder/'summary.json',report)


if __name__=='__main__':main()
