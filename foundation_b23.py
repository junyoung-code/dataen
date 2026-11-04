"""B-only local foundation-model adapters and immutable evaluation helpers."""
from __future__ import annotations
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',TABPFN_DISABLE_TELEMETRY='1',
 TABPFN_MODEL_CACHE_DIR=str(ROOT/'outputs/b23_models'),HF_HOME=str(ROOT/'.runtime/huggingface'),SKB_DATA_DIRECTORY=str(ROOT/'.runtime/skrub_data'),
 TORCH_HOME=str(ROOT/'.runtime/torch'),XDG_CACHE_HOME=str(ROOT/'.runtime/cache'),MPLCONFIGDIR=str(ROOT/'outputs/b23_research/mpl'),
 OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',VECLIB_MAXIMUM_THREADS='4')
import datetime as dt,hashlib,json,platform,socket
import numpy as np
import pandas as pd
import torch
import train_smoke as original
from run_peer_w import bounds,peer_labels,score,write_json
from features_p3 import build

CONFIG=ROOT/'b23_foundation_config.json'
CONTRACT=ROOT/'AB_EVALUATION_CONTRACT.json'
CONTRACT_SHA='d65965390a3027d6b3267c352d7222cdfd76a9dca128e7d319bab3c77465b1ec'
BASE=ROOT.parent.parent/'peer_w_20261103_163512_rescore'

def dump(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def status(phase,stage,log,**extra):
 complete={}
 p=ROOT/'outputs/b1_annual/verification.json'
 if p.exists():complete['B1']=json.loads(p.read_text())
 for model in ['tabpfn','tabicl']:
  p=ROOT/f'outputs/b23_{model}/verification.json'
  if p.exists():complete['B2' if model=='tabpfn' else 'B3']=json.loads(p.read_text())
 dump(ROOT/'outputs/status.json',dict(phase=phase,device=platform.node(),stage=stage,log_path=str(log),
       pid=os.getpid(),updated_at=dt.datetime.now(dt.timezone.utc).isoformat(),completed_experiments=complete,**extra))

def check_frozen():
 if original.sha256(CONTRACT)!=CONTRACT_SHA:raise ValueError('Contract changed')
 c=json.loads(CONTRACT.read_text())
 for name,digest in c['fixed_source_sha256'].items():
  if original.sha256(ROOT/name)!=digest:raise ValueError('Frozen common source changed')
 if original.sha256(original.DATA/'sites.csv')!=c['sites_file_sha256']:raise ValueError('Site assignment changed')
 return c

def key_hash(rows):
 return hashlib.sha256(rows.sort_values(['site_id','ts']).reset_index(drop=True)[['site_id','ts','has_peer']].to_csv(index=False,lineterminator='\n').encode()).hexdigest()

def select_context(capacity,n,seed):
 c=np.asarray(capacity,dtype=float)
 if len(c)<n or not np.isfinite(c).all() or (c<=0).any():raise ValueError('Context capacities')
 return np.sort(np.random.default_rng(seed).choice(len(c),n,replace=False,p=c/c.sum()))

def train_medians(x):
 x=np.asarray(x,dtype='float32')
 with np.errstate(all='ignore'):m=np.nanmedian(np.where(np.isfinite(x),x,np.nan),axis=0)
 return np.where(np.isfinite(m),m,0).astype('float32')

def fill(x,medians):
 x=np.asarray(x,dtype='float32');return np.where(np.isfinite(x),x,np.asarray(medians,dtype='float32')).astype('float32')

def make_model(name,cfg):
 spec=cfg[name];path=ROOT/spec['weights']
 if original.sha256(path)!=spec['sha256']:raise ValueError('Public checkpoint checksum failed')
 torch.set_num_threads(cfg['cpu_threads'])
 if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; do not silently switch device')
 torch.mps.set_per_process_memory_fraction(0.7)
 if name=='tabpfn':
  from tabpfn import TabPFNRegressor
  from tabpfn.constants import ModelVersion
  return TabPFNRegressor.create_default_for_version(ModelVersion.V2,n_estimators=cfg['n_estimators'],model_path=path,
    device='mps',fit_mode='fit_with_cache',inference_precision=torch.float32,random_state=cfg['seed'],n_preprocessing_jobs=1,show_progress_bar=False)
 if name=='tabicl':
  from tabicl import TabICLRegressor
  return TabICLRegressor(n_estimators=cfg['n_estimators'],norm_methods='none',feat_shuffle_method='none',batch_size=1,kv_cache=True,
    model_path=path,allow_auto_download=False,device='mps',use_amp=False,use_fa3=False,offload_mode=False,random_state=cfg['seed'],n_jobs=1,verbose=False)
 raise ValueError('Unknown foundation model')

def load_inputs(fold):
 a,cutoff,c,d=bounds(fold);b=original.load_baseline();sites=pd.read_csv(original.DATA/'sites.csv')
 gen=pd.read_parquet(original.DATA/'train/generation.parquet');gen.ts=pd.to_datetime(gen.ts).dt.tz_convert('Asia/Seoul')
 train=gen.loc[(gen.ts>=a)&(gen.ts<=cutoff)].copy();valid=gen.loc[(gen.ts>=c)&(gen.ts<d)].copy()
 prices=pd.read_csv(original.DATA/'train/smp.csv');prices.ts=pd.to_datetime(prices.ts).dt.tz_convert('Asia/Seoul')
 fa=pd.read_parquet(original.DATA/'train/forecast_wa.parquet',columns=['wa_station','round','ts']+b.WA_COLS)
 fb=pd.read_parquet(original.DATA/'train/forecast_wb.parquet',columns=['wb_station','round','ts']+b.WB_COLS)
 for f in (fa,fb):f.ts=pd.to_datetime(f.ts).dt.tz_convert('Asia/Seoul')
 fa=fa.loc[(fa.ts>=a)&(fa.ts<d)];fb=fb.loc[(fb.ts>=a)&(fb.ts<d)]
 return b,sites,train,valid,prices,fa,fb

def deny_network_probe():
 # Assert the enclosing OS profile actually denies Internet sockets before reading competition data.
 try:
  with socket.create_connection(('1.1.1.1',443),timeout=0.5):pass
 except PermissionError:return 'OS network deny verified'
 except OSError as e:
  if e.errno in [1,13]:return 'OS network deny verified'
  raise RuntimeError('Network denied status not established') from e
 raise RuntimeError('Network was available; offline model job refused')
