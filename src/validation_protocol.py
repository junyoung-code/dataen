"""P0 frozen evaluation windows, masks and score; never trains a model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TZ = 'Asia/Seoul'
PROTOCOL = 'p0-v1'
# All ends are exclusive. Full 24-hour embargo before each evaluation.
WINDOWS = {
    'seasonal_diagnostic': ('2023-06-01 01:00', '2024-06-30 01:00', '2024-07-01 01:00', '2025-01-01 01:00'),
    'tune': ('2023-06-01 01:00', '2024-12-31 01:00', '2025-01-01 01:00', '2025-03-01 01:00'),
    'confirmation': ('2023-06-01 01:00', '2025-02-28 01:00', '2025-03-01 01:00', '2025-06-01 01:00'),
}


def bounds(name):
    return tuple(pd.Timestamp(x, tz=TZ) for x in WINDOWS[name])


def split(frame, name):
    a, b, c, d = bounds(name)
    if frame.ts.dt.tz is None:
        raise ValueError('Timezone-aware timestamps required')
    first_issuance = c.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=17)
    if not (a < b <= first_issuance < c < d and c-b >= pd.Timedelta(hours=24)):
        raise ValueError('Invalid chronological boundary')
    return (frame.loc[(frame.ts >= a) & (frame.ts < b)].copy(),
            frame.loc[(frame.ts >= c) & (frame.ts < d)].copy())


def complete_hours(rows, site_ids):
    """Mask uses observed labels only, independent of predictions/group assignment."""
    expected = set(site_ids)
    if not expected or rows.duplicated(['site_id', 'ts']).any():
        raise ValueError('Empty site universe or duplicate observations')
    if not set(rows.site_id).issubset(expected):
        raise ValueError('Unknown site')
    if rows.ts.dt.tz is None or rows.ts.isna().any() or not np.isfinite(rows.energy).all():
        raise ValueError('Invalid timestamps or raw observations')
    counts = rows.groupby('ts').site_id.nunique()
    return pd.DatetimeIndex(counts[counts == len(expected)].index).as_unit('ns').sort_values()


def weighted_group_mae(groups, hours, prediction='prediction'):
    """One row per G1/G2 and fixed timestamp; raw kWh labels, SMP >= 0."""
    if len(hours) == 0 or hours.has_duplicates or hours.tz is None:
        raise ValueError('Nonempty unique timezone-aware mask required')
    expected = pd.MultiIndex.from_product([['G1', 'G2'], hours], names=['group', 'ts'])
    if groups.duplicated(['group', 'ts']).any():
        raise ValueError('Duplicate group timestamp')
    indexed = groups.set_index(['group', 'ts'])
    if len(indexed) != len(expected) or len(expected.difference(indexed.index)):
        raise ValueError('Group rows must match the frozen mask exactly')
    indexed = indexed.reindex(expected)
    values = indexed[['energy', prediction, 'smp']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (indexed.smp < 0).any():
        raise ValueError('Invalid labels, predictions or price')
    prices = indexed.smp.unstack('group')
    if not np.array_equal(prices.G1.to_numpy(), prices.G2.to_numpy()):
        raise ValueError('Both groups must use the same timestamp price')
    h = indexed.index.get_level_values('ts').tz_convert(TZ).hour
    weights = indexed.smp.to_numpy() * np.where((h >= 11) & (h <= 15), 3, 1)
    if weights.sum() <= 0:
        raise ValueError('Positive total evaluation weight required')
    errors = np.abs(indexed[prediction].to_numpy() - indexed.energy.to_numpy())
    return float(np.dot(weights, errors) / weights.sum())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(output):
    data = ROOT / 'data'
    manifest = json.loads((ROOT/'configs/data-manifest.json').read_text())
    inputs = ['train/generation.parquet', 'sites.csv', 'train/smp.csv']
    hashes = {name: digest(data/name) for name in inputs}
    for name in inputs:
        if hashes[name] != manifest[name]['sha256']:
            raise ValueError(f'Original data hash mismatch: {name}')
    sites = pd.read_csv(data/'sites.csv')
    if len(sites) != 253 or sites.site_id.duplicated().any():
        raise ValueError('Expected 253 unique sites')
    gen = pd.read_parquet(data/'train/generation.parquet', columns=['site_id','ts','energy'])
    gen.ts = pd.to_datetime(gen.ts).dt.tz_convert(TZ)
    complete_hours(gen, sites.site_id)  # Validate whole source before splitting.
    prices = pd.read_csv(data/'train/smp.csv')
    prices.ts = pd.to_datetime(prices.ts).dt.tz_convert(TZ)
    if prices.ts.duplicated().any():
        raise ValueError('Duplicate prices')
    prices = prices.set_index('ts').smp
    output = output.resolve()
    if output == data.resolve() or data.resolve() in output.parents:
        raise ValueError('Output must be outside original data')
    output.mkdir(parents=True, exist_ok=False)
    report = {'protocol': PROTOCOL, 'data_hashes': hashes,
              'labels_seen_previously': True, 'model_training_performed': False, 'windows': {}}
    for name in WINDOWS:
        a,b,c,d = bounds(name)
        train, valid = split(gen, name)
        hours = complete_hours(valid, sites.site_id)
        axis = pd.date_range(c,d,freq='h',inclusive='left')
        if name != 'seasonal_diagnostic' and not hours.equals(axis):
            raise ValueError(f'{name}: expected full 253-site coverage')
        price = prices.reindex(hours)
        if not np.isfinite(price).all() or (price < 0).any() or price.sum() <= 0:
            raise ValueError('Missing/invalid evaluation prices')
        mask_path = output/f'{name}_hours.csv'
        pd.DataFrame({'ts': hours.astype(str)}).to_csv(mask_path,index=False)
        report['windows'][name] = dict(
            train_start=str(a),train_end_exclusive=str(b),eval_start=str(c),eval_end_exclusive=str(d),
            train_rows=len(train),eval_observed_rows=len(valid),expected_hours=len(axis),
            common_hours=len(hours),common_coverage_pct=100*len(hours)/len(axis),
            weighted_peak_hours=int(((hours.hour>=11)&(hours.hour<=15)).sum()),
            unseen_sites=len(set(valid.site_id)-set(train.site_id)),
            mask_sha256=digest(mask_path),
            monthly_common_hours={str(k):int(v) for k,v in pd.Series(1,index=hours-pd.Timedelta(hours=1)).resample('MS').sum().items()})
    (output/'coverage.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    audit(parser.parse_args().output_dir)
