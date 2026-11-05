"""Independent raw-mask, source-score, policy-fit and scalar blend-W audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import train_smoke as original
from annual_neural_data import check_contract, key_mask_digest
from run_peer_w import TZ, bounds, score, write_json

ROOT=Path(__file__).resolve().parent


def optimum(errors,weights):
    rows=[]
    for t in range(21):
        for f in range(21-t):
            w=[1-(t+f)/20,t/20,f/20]
            e=errors[:,0]*w[0]+errors[:,1]*w[1]+errors[:,2]*w[2]
            rows.append((float(np.dot(np.abs(e),weights)),w))
    smallest=min(r[0] for r in rows)
    return min((r for r in rows if r[0]<=smallest+1e-6),key=lambda r:(1-r[1][0],r[1][2],r[1][1]))[1]


def labels(rule,hours,cloud):
    peak=np.isin(hours,[11,12,13,14,15]).astype(int)
    if rule=='global':return np.zeros(len(hours),dtype=int)
    if rule=='peak':return peak
    if rule=='cloud':return np.where(np.isfinite(cloud),(cloud>=50).astype(int),-1)
    if rule=='peak_cloud':return np.where(np.isfinite(cloud),2*peak+(cloud>=50).astype(int),-1)
    raise AssertionError('Unknown policy')


def independently_apply(policy,errors,hours,cloud):
    lab=labels(policy['rule'],hours,cloud)
    result=[]
    for i,label in enumerate(lab):
        w=policy['bins'].get(str(label),policy['fallback'])
        assert len(w)==3 and min(w)>=0 and math.isclose(sum(w),1.,abs_tol=1e-12)
        result.append(math.fsum(float(errors[i,j])*w[j] for j in range(3)))
    return np.asarray(result)


def audit_policy(policy,errors,weights,hours,cloud):
    np.testing.assert_allclose(policy['fallback'],optimum(errors,weights),rtol=0,atol=1e-12)
    lab=labels(policy['rule'],hours,cloud)
    expected=set()
    if policy['rule']!='global':
        for label in set(lab)-{-1}:
            keep=(lab==label)&(weights>0)&np.any(errors!=0,axis=1)
            if int(keep.sum())>=200:
                expected.add(str(label))
                np.testing.assert_allclose(policy['bins'][str(label)],optimum(errors[keep],weights[keep]),rtol=0,atol=1e-12)
    assert set(policy['bins'])==expected


def raw_forecast_condition(sites,forecast,axis):
    f=forecast.loc[(forecast['round']==17)&forecast.ts.isin(axis)]
    table=f.pivot(index='ts',columns='wa_station',values='wa_cloud').reindex(index=axis,columns=sites.wa_station.to_numpy())
    x=table.to_numpy(dtype=float);capacity=sites.capacity_kw.to_numpy(dtype=float)
    finite=np.isfinite(x)
    den=np.sum(finite*capacity[None,:],axis=1)
    num=np.sum(np.where(finite,x,0.)*capacity[None,:],axis=1)
    cloud=np.divide(num,den,out=np.full(len(axis),np.nan),where=den>0)
    return np.round(cloud,10),1-den/capacity.sum()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--a1-dir',type=Path,required=True)
    p.add_argument('--p4-aggregate-dir',type=Path,required=True)
    args=p.parse_args()
    contract=check_contract();folder=args.output_dir
    report=json.loads((folder/'summary.json').read_text())
    assert report['status']=='computed_pending_verification'
    for name,h in report['source_sha256'].items():
        assert original.sha256(ROOT/name)==h
        assert original.sha256(folder/'source'/name)==h
    assert original.sha256(args.a1_dir/'summary.json')==report['a1_summary_sha256']
    assert original.sha256(args.p4_aggregate_dir/'p4_manifest.json')==report['p4_manifest_sha256']
    assert original.sha256(folder/'selection.json')==report['selection_sha256']
    selection=json.loads((folder/'selection.json').read_text())
    assert selection==report['selection'] and selection['fit_fold']=='F2' and selection['confirmation_fold']=='F3'
    a1=json.loads((args.a1_dir/'summary.json').read_text())
    assert a1['status']=='complete'
    for name,h in a1['verification_sha256'].items():assert original.sha256(args.a1_dir/name)==h
    manifest=json.loads((args.p4_aggregate_dir/'p4_manifest.json').read_text())
    sites=pd.read_csv(original.DATA/'sites.csv')
    prices=pd.read_csv(original.DATA/'train/smp.csv');prices.ts=pd.to_datetime(prices.ts).dt.tz_convert(TZ)
    gen=pd.read_parquet(original.DATA/'train/generation.parquet');gen.ts=pd.to_datetime(gen.ts).dt.tz_convert(TZ)
    forecast=pd.read_parquet(original.DATA/'train/forecast_wa.parquet',columns=['wa_station','round','ts','wa_cloud']);forecast.ts=pd.to_datetime(forecast.ts).dt.tz_convert(TZ)
    result=dict(status='running',source_hashes_verified=True,selection_only_F2=True,folds={})
    for fold in ['F2','F3']:
        expected=contract['folds'][fold];_,_,c,d=bounds(fold)
        axis=pd.date_range(c,d,freq='h',inclusive='left')
        index=pd.MultiIndex.from_product([['G1','G2'],axis],names=['group','ts'])
        saved=pd.read_parquet(folder/f'{fold}_team_errors.parquet')
        assert pd.MultiIndex.from_frame(saved[['group','ts']]).equals(index)
        actual=gen.loc[(gen.ts>=c)&(gen.ts<d)].rename(columns={'energy':'energy_raw'}).copy()
        actual['capacity_kw']=actual.site_id.map(sites.set_index('site_id').capacity_kw)
        e=actual.energy_raw.to_numpy(dtype=float);cap=actual.capacity_kw.to_numpy(dtype='float32').astype(float)
        actual['has_peer']=np.isfinite(e)&(e>=0)&(e<=cap*1.05)
        assert key_mask_digest(actual)==expected['evaluation_key_mask_sha256']
        assert int(actual.has_peer.sum())==expected['observed_site_hours']
        source=manifest['folds'][fold]
        assert original.sha256(args.p4_aggregate_dir/source['file'])==source['sha256']
        p4=pd.DataFrame(json.loads((args.p4_aggregate_dir/source['file']).read_text())['rows'])
        p4.ts=pd.to_datetime(p4.ts).dt.tz_convert(TZ);p4=p4.set_index(['group','ts']).reindex(index)
        np.testing.assert_array_equal(saved.error_p4,p4.error)
        np.testing.assert_array_equal(saved.observed_sites,p4.observed_sites)
        np.testing.assert_allclose(saved.weight,p4.weight,rtol=0,atol=1e-9)
        assert int(saved.observed_sites.sum())==expected['observed_site_hours']
        price=prices.set_index('ts').reindex(axis)
        k=price.k.to_numpy() if 'k' in price else np.where(axis.hour.isin([11,12,13,14,15]),3.,1.)
        np.testing.assert_allclose(saved.weight,np.tile(price.smp.to_numpy()*k,2),rtol=0,atol=1e-9)
        cloud,missing=raw_forecast_condition(sites,forecast,axis)
        np.testing.assert_allclose(saved.cloud,np.tile(cloud,2),rtol=0,atol=1e-9,equal_nan=True)
        np.testing.assert_allclose(saved.missing_capacity_fraction,np.tile(missing,2),rtol=0,atol=1e-12)
        np.testing.assert_array_equal(saved.hour,saved.ts.dt.hour)
        for family in ['tabm','tft']:
            source_folder=args.a1_dir/family/fold
            inp=report['folds'][fold]['input_sources'][family]
            assert original.sha256(source_folder/'site_predictions.parquet')==inp['site_predictions_sha256']
            assert original.sha256(source_folder/'group_errors.csv')==inp['group_errors_sha256']
            rows=pd.read_parquet(source_folder/'site_predictions.parquet')
            assert key_mask_digest(rows)==expected['evaluation_key_mask_sha256']
            joined=rows[['site_id','ts','energy_raw','has_peer']].merge(actual[['site_id','ts','energy_raw','has_peer']],on=['site_id','ts'],suffixes=('','_original'),validate='one_to_one')
            assert len(joined)==len(actual)
            np.testing.assert_array_equal(joined.energy_raw,joined.energy_raw_original)
            np.testing.assert_array_equal(joined.has_peer,joined.has_peer_original)
            metrics,groups=score(rows,sites,prices,c,d)
            groups=groups.set_index(['group','ts']).reindex(index)
            np.testing.assert_allclose(saved[f'error_{family}'],groups.error,rtol=1e-12,atol=1e-8)
            np.testing.assert_allclose(metrics['W'],report['folds'][fold]['metrics'][family]['W'],rtol=1e-12)
            del rows,joined,groups
        errors=saved[['error_p4','error_tabm','error_tft']].to_numpy()
        weights=saved.weight.to_numpy();hours=saved.hour.to_numpy();cloud=saved.cloud.to_numpy()
        if fold=='F2':
            cal=(saved.ts<pd.Timestamp('2023-12-01 01:00',tz=TZ)).to_numpy();val=~cal
            global_candidates=selection['global_candidates']
            assert len(global_candidates)==231
            for candidate in global_candidates:
                w=candidate['weights']
                assert min(w)>=0 and math.isclose(sum(w),1,abs_tol=1e-12)
                s=float(np.dot(np.abs(errors[:,0]*w[0]+errors[:,1]*w[1]+errors[:,2]*w[2]),weights))
                np.testing.assert_allclose(s,candidate['W'],rtol=1e-12,atol=1e-5)
            conditional=selection['conditional_selection'];scores={}
            for item in conditional['validation_candidates']:
                policy=item['calibration_policy']
                audit_policy(policy,errors[cal],weights[cal],hours[cal],cloud[cal])
                pred=independently_apply(policy,errors[val],hours[val],cloud[val])
                value=math.fsum(float(abs(x)*w) for x,w in zip(pred,weights[val]))
                np.testing.assert_allclose(value,item['validation_W'],rtol=1e-12,atol=1e-5)
                scores[item['rule']]=value
            rule_order=['global','peak','cloud','peak_cloud']
            best=min(rule_order[1:],key=lambda rule:(scores[rule],rule_order.index(rule)))
            chosen=best if scores[best]<scores['global']*(1-.0025) else 'global'
            assert conditional['selected_rule']==chosen and conditional['best_conditional_rule']==best
            for policy in selection['policies'].values():audit_policy(policy,errors,weights,hours,cloud)
        output={name:errors[:,i] for i,name in enumerate(['p4','tabm','tft'])}
        output.update({name:independently_apply(policy,errors,hours,cloud) for name,policy in selection['policies'].items()})
        verified={}
        for name,error in output.items():
            np.testing.assert_allclose(saved[f'error_{name}'],error,rtol=1e-12,atol=1e-8)
            amounts=np.abs(error)*weights
            W=math.fsum(float(v) for v in amounts)
            metric=report['folds'][fold]['metrics'][name]
            np.testing.assert_allclose(W,metric['W'],rtol=1e-12,atol=1e-5)
            for group in ['G1','G2']:
                np.testing.assert_allclose(math.fsum(float(v) for v in amounts[saved.group==group]),metric['W_by_group'][group],rtol=1e-12,atol=1e-5)
            month=(saved.ts-pd.Timedelta(hours=1)).dt.strftime('%Y-%m')
            for m in set(month):np.testing.assert_allclose(math.fsum(float(v) for v in amounts[month==m]),metric['W_by_month'][m],rtol=1e-12,atol=1e-5)
            verified[name]=dict(W=W,scalar_blend_and_month_group_sums_match=True)
        result['folds'][fold]=dict(raw_mask_and_labels_verified=True,forecast_condition_independent_site_aggregation=True,
            neural_source_score_replayed=True,policies=verified)
        print(f'{fold}: raw mask/source W/forecast conditions/scalar blend and policy checks passed',flush=True)
    assert original.sha256(folder/'selection.json')==report['selection_sha256']
    original.check_data()
    result['status']='complete'
    write_json(folder/'verification.json',result)
    print('A2-A4 independent verification complete',flush=True)


if __name__=='__main__':main()
