"""Join the same timestamp-only public GEFS summaries to every site."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'b1_gefs_config.json').read_text())
COLUMNS=CFG['feature_columns']

def load_table(path):
    table=pd.read_parquet(path)
    table.ts=pd.to_datetime(table.ts).dt.tz_convert('Asia/Seoul')
    if table.ts.duplicated().any() or not np.isfinite(table[COLUMNS]).all().all():
        raise ValueError('Invalid GEFS table keys/values')
    day=(table.ts-pd.Timedelta(hours=1)).dt.normalize()
    deadline=day-pd.Timedelta(hours=7)
    expected=(day.dt.tz_localize(None)-pd.Timedelta(days=2)).dt.tz_localize('UTC')
    if not (pd.to_datetime(table.init_time,utc=True)==expected).all():raise ValueError('Incorrect initialization day')
    available=pd.to_datetime(table.init_time,utc=True)+pd.Timedelta(hours=CFG['availability_lag_hours'])
    if not (available<=deadline).all():raise ValueError('Unavailable forecast at deadline')
    if not (pd.to_datetime(table.available_at_assumed,utc=True)==available).all():raise ValueError('Availability audit mismatch')
    return table[['ts']+COLUMNS]

def attach(frame, table):
    keys=frame[['site_id','ts']].reset_index(drop=True)
    if keys.duplicated().any() or table.ts.duplicated().any():raise ValueError('Duplicate feature keys')
    out=frame.merge(table,on='ts',how='left',sort=False,validate='many_to_one')
    if not keys.equals(out[['site_id','ts']].reset_index(drop=True)):raise ValueError('GEFS changed rows/order')
    if not np.isfinite(out[COLUMNS]).all().all():raise ValueError('Uncovered GEFS timestamp; cannot reduce evaluation set')
    return out
