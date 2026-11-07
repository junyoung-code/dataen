"""Forecast-only feature bundles available at the day-before 17:00 cutoff."""
import numpy as np

BUNDLES = ('baseline', 'revision', 'provider_gap', 'station_distance')


def extra_columns(baseline, bundle):
    if bundle == 'baseline': return []
    if bundle == 'revision':
        names = [f'revision_{c}' for c in baseline.WA_COLS + baseline.WB_COLS]
        return names + [f'{c}_missing' for c in names]
    if bundle == 'provider_gap':
        names = ['gap_cloud', 'gap_temp', 'gap_humidity', 'gap_wind_ms']
        return names + [f'{c}_missing' for c in names]
    if bundle == 'station_distance': return ['wa_dist', 'wb_dist']
    raise ValueError('Unknown feature bundle')


def revision_table(forecast, station, columns):
    keys = [station, 'ts']
    if forecast.duplicated(keys + ['round']).any():
        raise ValueError('Duplicate station/target/issuance keys')
    # Never fill an absent issue from another timestamp or a later forecast.
    early = forecast.loc[forecast['round']==10, keys+columns]
    late = forecast.loc[forecast['round']==17, keys+columns]
    pair = early.merge(late,on=keys,how='outer',suffixes=('_10','_17'),validate='one_to_one')
    result = pair[keys].copy()
    for col in columns:
        result[f'revision_{col}'] = pair[f'{col}_17'] - pair[f'{col}_10']
    return result


def build(baseline, gen, sites, fa, fb, round_hour, bundle):
    if round_hour not in (10,17): raise ValueError('Only prescribed forecast issues allowed')
    extras = extra_columns(baseline,bundle)
    if gen.duplicated(['site_id','ts']).any() or sites.site_id.duplicated().any():
        raise ValueError('Duplicate site keys')
    for forecast,station in [(fa,'wa_station'),(fb,'wb_station')]:
        if forecast.duplicated([station,'round','ts']).any():
            raise ValueError('Duplicate forecast keys')
    result = baseline.features(gen,sites,fa,fb,None,round_hour)
    if bundle == 'revision':
        for forecast,station,cols in [(fa,'wa_station',baseline.WA_COLS),(fb,'wb_station',baseline.WB_COLS)]:
            result=result.merge(revision_table(forecast,station,cols),on=[station,'ts'],
                                how='left',sort=False,validate='many_to_one')
    elif bundle == 'provider_gap':
        result['gap_cloud']=result.wb_cloud-result.wa_cloud
        result['gap_temp']=result.wb_temp-result.wa_temp
        result['gap_humidity']=result.wb_rel_hum-result.wa_humidity
        result['gap_wind_ms']=result.wb_wind_speed/3.6-result.wa_wind_speed
    elif bundle == 'station_distance':
        values=result[['wa_dist','wb_dist']].to_numpy()
        if not np.isfinite(values).all() or (values<0).any():
            raise ValueError('Invalid provided station distances')
    for col in extras:
        if col.endswith('_missing'):result[col]=result[col.removesuffix('_missing')].isna().astype('int8')
    if not result[['site_id','ts']].reset_index(drop=True).equals(gen[['site_id','ts']].reset_index(drop=True)):
        raise ValueError('Feature join changed the row universe/order')
    columns=list(baseline.FEATS)+extras
    if 'energy' in columns or 'current_group' in columns:
        raise ValueError('Labels/group allocation cannot be model inputs')
    return result,columns
