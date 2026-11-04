"""Download only a fixed public Korea-area GEFS subset; never read competition data."""
from __future__ import annotations
import argparse, concurrent.futures, datetime as dt, hashlib, json, os, platform, time, traceback
from pathlib import Path
import icechunk
import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parent
CONFIG=ROOT/'b1_gefs_config.json'
TZ='Asia/Seoul'

def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def status(phase,stage,log,**extra):
    dump(ROOT/'outputs/status.json',dict(phase=phase,device=platform.node(),stage=stage,
         log_path=str(log),updated_at=dt.datetime.now(dt.timezone.utc).isoformat(),**extra))

def interval_values(values,leads):
    """Input [member,lead,lat,lon], output non-overlapping preceding 3h means."""
    out=np.array(values,dtype=np.float64,copy=True)
    for i,h in enumerate(leads):
        if h%6==0:
            if i==0 or leads[i-1]!=h-3: raise ValueError('Missing preceding 3h average')
            out[:,i]=2*values[:,i]-values[:,i-1]
    return out

def summarize(values,latitudes,prefix,leads):
    if not np.isfinite(values).all(): raise ValueError('Nonfinite public forecast subset')
    v=interval_values(values,leads)
    if prefix=='gefs_cloud': v=np.clip(v,0,100)
    else: v=np.maximum(v,0)
    weights=np.cos(np.deg2rad(latitudes))[None,None,:,None]
    member_means=(v*weights).sum(axis=(2,3))/(weights.sum()*v.shape[3])
    return member_means

def hourly(day, members, cfg):
    day=pd.Timestamp(day).tz_localize(TZ)
    axis=pd.date_range(day+pd.Timedelta(hours=1),periods=24,freq='h')
    init=(day.tz_localize(None)-pd.Timedelta(days=2)).tz_localize('UTC')
    deadline=day-pd.Timedelta(hours=7)
    availability=init+pd.Timedelta(hours=cfg['availability_lag_hours'])
    if availability>deadline: raise ValueError('Future forecast issue')
    lead=(axis.tz_convert('UTC')-init)/pd.Timedelta(hours=1)
    idx=np.searchsorted(cfg['lead_hours'],np.ceil(lead/3)*3)
    if (idx>=len(cfg['lead_hours'])).any(): raise ValueError('Lead coverage')
    rows=pd.DataFrame({'ts':axis,'init_time':init,'available_at_assumed':availability,'deadline':deadline})
    for prefix,m in members.items():
        rows[prefix+'_mean']=m.mean(axis=0)[idx].astype('float32')
        rows[prefix+'_std']=m.std(axis=0,ddof=0)[idx].astype('float32')
    return rows

def one_day(ds,day,folder,cfg,cfg_hash):
    file=folder/'days'/f'{day}.parquet'
    proof=file.with_suffix('.json')
    if file.exists() and proof.exists():
        saved=json.loads(proof.read_text())
        if saved['config_sha256']!=cfg_hash: raise ValueError('Cache config changed')
        if hashlib.sha256(file.read_bytes()).hexdigest()!=saved['parquet_sha256']: raise ValueError('Cache checksum')
        return saved|{'cached':True}
    t=time.perf_counter()
    init=pd.Timestamp(day)-pd.Timedelta(days=2)
    sub=ds[list(cfg['variables'])].sel(init_time=init.to_numpy(),
        lead_time=pd.to_timedelta(cfg['lead_hours'],unit='h'),
        latitude=slice(cfg['latitude_max'],cfg['latitude_min']),
        longitude=slice(cfg['longitude_min'],cfg['longitude_max']))
    if sub.sizes!=dict(ensemble_member=31,lead_time=9,latitude=33,longitude=33):
        raise ValueError(f'Unexpected subset dimensions: {dict(sub.sizes)}')
    sub=sub.load()
    members={prefix:summarize(sub[var].transpose('ensemble_member','lead_time','latitude','longitude').values,
                sub.latitude.values,prefix,cfg['lead_hours']) for var,prefix in cfg['variables'].items()}
    rows=hourly(day,members,cfg)
    file.parent.mkdir(parents=True,exist_ok=True)
    tmp=file.with_suffix('.tmp.parquet');rows.to_parquet(tmp,index=False);tmp.replace(file)
    np.savez_compressed(file.with_suffix('.members.npz'),**members)
    audit=dict(day=day,init_time=str(init),config_sha256=cfg_hash,parquet_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        subset_dimensions=dict(sub.sizes),seconds=time.perf_counter()-t,selected_float32_bytes=sum(sub[v].nbytes for v in cfg['variables']),
        cached=False,finite=True,min_values={k:float(rows[k].min()) for k in cfg['feature_columns']},
        max_values={k:float(rows[k].max()) for k in cfg['feature_columns']})
    dump(proof,audit)
    return audit

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',required=True);p.add_argument('--end',required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();folder=args.output_dir.resolve()
    if not folder.is_relative_to(ROOT/'outputs'): raise ValueError('B outputs required')
    folder.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(CONFIG.read_text());cfg_hash=hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    days=pd.date_range(args.start,args.end,freq='D').strftime('%Y-%m-%d').tolist()
    status('downloading',f'GEFS subset 0/{len(days)} days',str(folder/'run.log'),result_dir=str(folder))
    repo=icechunk.Repository.open(icechunk.http_storage(cfg['icechunk_https']))
    s=repo.readonly_session(snapshot_id=cfg['snapshot_id'])
    ds=xr.open_zarr(s.store,chunks=None,consolidated=False)
    results=[];errors=[];t=time.perf_counter()
    def task(day):
        for retry in range(3):
            try:return one_day(ds,day,folder,cfg,cfg_hash)
            except Exception:
                if retry==2:raise
                time.sleep(2*(retry+1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(task,day):day for day in days}
        for f in concurrent.futures.as_completed(futures):
            day=futures[f]
            try:results.append(f.result())
            except Exception as e:
                errors.append(dict(day=day,error=repr(e)));print(traceback.format_exc(),flush=True)
            n=len(results)+len(errors)
            if n%10==0 or len(days)<=3 or n==len(days):
                stage=f'GEFS subset {len(results)}/{len(days)} days, failures {len(errors)}'
                print(stage, f'elapsed={time.perf_counter()-t:.1f}s',flush=True)
                status('downloading',stage,str(folder/'run.log'),result_dir=str(folder),completed_days=len(results),expected_days=len(days))
                dump(folder/'collection_summary.json',dict(status='downloading',config_sha256=cfg_hash,completed_days=len(results),expected_days=len(days),errors=errors,seconds=time.perf_counter()-t))
    if errors:
        dump(folder/'errors.json',errors);status('blocked','GEFS missing days; evaluation not reduced',str(folder/'run.log'));raise RuntimeError(f'{len(errors)} failed public days')
    frames=[pd.read_parquet(folder/'days'/f'{d}.parquet') for d in days]
    frame=pd.concat(frames,ignore_index=True)
    assert not frame.ts.duplicated().any() and len(frame)==24*len(days)
    assert np.isfinite(frame[cfg['feature_columns']]).all().all()
    assert (frame.available_at_assumed<=frame.deadline).all()
    frame.to_parquet(folder/'hourly.parquet',index=False)
    report=dict(status='complete',config=cfg,config_sha256=cfg_hash,days=len(days),hours=len(frame),
        start=str(frame.ts.min()),end=str(frame.ts.max()),seconds=time.perf_counter()-t,
        failures=errors,new_download_days=sum(not x['cached'] for x in results),
        local_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),
        selected_float32_bytes=sum(x['selected_float32_bytes'] for x in results))
    dump(folder/'collection_summary.json',report)
    status('preparing','GEFS subset collected; join/training next',str(folder/'run.log'),result_dir=str(folder))
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
