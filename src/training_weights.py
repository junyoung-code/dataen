"""P4 frozen objectives and training-only sample weights for normalized targets."""
import numpy as np
import pandas as pd

VARIANTS = {
    'l2_uniform': ('regression', 'uniform'),
    'l1_uniform': ('regression_l1', 'uniform'),
    'l1_capacity': ('regression_l1', 'capacity'),
    'l1_capacity_peak': ('regression_l1', 'capacity_peak'),
    'l1_capacity_peak_smp': ('regression_l1', 'capacity_peak_smp'),
}


def make_weights(rows, prices, mode, train_start, train_end):
    if mode not in {v[1] for v in VARIANTS.values()}:
        raise ValueError('Unknown weighting mode')
    if rows.empty or rows.ts.dt.tz is None or rows.ts.isna().any():
        raise ValueError('Nonempty timezone-aware training timestamps required')
    if not ((rows.ts>=train_start)&(rows.ts<train_end)).all():
        raise ValueError('Sample weights may only use rows inside training window')
    if mode=='uniform':
        return None,dict(mode=mode,rows=len(rows),mean=1.,min=1.,max=1.,zero_rows=0,
                         raw_mean=1.,price_rows_used=0)
    capacity=rows.cap.to_numpy(dtype=float)
    if not np.isfinite(capacity).all() or (capacity<=0).any():
        raise ValueError('Positive finite site capacities required')
    raw=capacity.copy()
    if mode in ('capacity_peak','capacity_peak_smp'):
        hours=rows.ts.dt.tz_convert('Asia/Seoul').dt.hour
        raw*=np.where(hours.between(11,15),3.,1.)
    price_rows=0
    if mode=='capacity_peak_smp':
        # Discard evaluation/future prices before lookup or any normalization.
        if prices.ts.dt.tz is None or prices.ts.isna().any():
            raise ValueError('Timezone-aware price timestamps required')
        historic=prices.loc[(prices.ts>=train_start)&(prices.ts<train_end)]
        if historic.ts.duplicated().any():raise ValueError('Duplicate historical prices')
        values=rows.ts.map(historic.set_index('ts').smp).to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values<0).any():
            raise ValueError('Missing, negative or nonfinite training prices')
        raw*=values;price_rows=rows.ts.nunique()
    mean=float(raw.mean())
    if not np.isfinite(raw).all() or not np.isfinite(mean) or mean<=0:
        raise ValueError('Positive finite total sample weight required')
    weights=raw/mean
    return weights,dict(mode=mode,rows=len(rows),mean=float(weights.mean()),min=float(weights.min()),
                        max=float(weights.max()),zero_rows=int((weights==0).sum()),raw_mean=mean,
                        price_rows_used=int(price_rows))


def choose(rows):
    if len(rows)!=len(VARIANTS) or {r['variant'] for r in rows}!=set(VARIANTS):
        raise ValueError('Expected every frozen P4 candidate exactly once')
    if not all(np.isfinite(r['weighted_group_mae']) for r in rows):
        raise ValueError('Invalid candidate score')
    best=min(r['weighted_group_mae'] for r in rows)
    return min((r for r in rows if r['weighted_group_mae']<=best+1e-9),
               key=lambda r:list(VARIANTS).index(r['variant']))
