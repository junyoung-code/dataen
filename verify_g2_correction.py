"""Independent A6 raw support, attribution, fixed policy and bootstrap audit."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import train_smoke as original
from annual_neural_data import check_contract
from run_peer_w import TZ,bounds,write_json
from verify_candidate_validation import replay

ROOT=Path(__file__).resolve().parent
POLICIES=['g2_cold_p4','g2_no_tabm','g2_cold_no_tabm']


def compute(e,k,c,name):
    out=.5*e[:,0]+.2*e[:,1]+.3*e[:,2];n=len(out)//2
    if name=='original':return out
    if name=='p4':return e[:,0]
    if name=='g2_cold_p4':out[n:]=.5*k[n:,0]+.2*k[n:,1]+.3*k[n:,2]+c[n:,0]
    elif name=='g2_no_tabm':out[n:]=.625*e[n:,0]+.375*e[n:,2]
    elif name=='g2_cold_no_tabm':out[n:]=.5*k[n:,0]+.2*k[n:,1]+.3*k[n:,2]+.625*c[n:,0]+.375*c[n:,2]
    else:raise AssertionError(name)
    return out


def metrics(error,w,keep):
    amount=np.abs(error)*w;n=len(error)//2
    return dict(W=float(amount[keep].sum()),G1_W=float(amount[:n][keep[:n]].sum()),G2_W=float(amount[n:][keep[n:]].sum()))


def shapley(base,a,b,w):
    zero=np.abs(base);first=np.abs(base+a);second=np.abs(base+b);both=np.abs(base+a+b)
    return ((first-zero+both-second)*w/2,(second-zero+both-first)*w/2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--a5-dir',type=Path,required=True)
    p.add_argument('--a1-dir',type=Path,required=True);p.add_argument('--cache-dir',type=Path,required=True);args=p.parse_args()
    folder=args.output_dir;r=json.loads((folder/'summary.json').read_text());assert r['status']=='computed_pending_verification'
    check_contract()
    for name,h in r['source_sha256'].items():
        assert original.sha256(ROOT/name)==h
        assert original.sha256(folder/'source'/name)==h
    assert original.sha256(args.a5_dir/'summary.json')==r['a5_summary_sha256']
    assert original.sha256(folder/'selection.json')==r['selection_sha256']
    assert json.loads((folder/'selection.json').read_text())==r['selection']
    sites=pd.read_csv(original.DATA/'sites.csv');gen=pd.read_parquet(original.DATA/'train/generation.parquet')
    gen.ts=pd.to_datetime(gen.ts).dt.tz_convert(TZ);baseline=original.load_baseline();calibration=None
    result=dict(status='running',source_hashes_verified=True,folds={})
    for fi,fold in enumerate(['F2','F3']):
        path=args.a5_dir/f'{fold}_arrays.npz';assert original.sha256(path)==r['folds'][fold]['input_arrays_sha256']
        z=np.load(path);e,k,c=z['errors_all'],z['errors_seen'],z['errors_new'];w=z['weight'];hours=len(w)//2;days=hours//24
        a,cutoff,start,end=bounds(fold);axis=pd.date_range(start,end,freq='h',inclusive='left');ts=axis.append(axis)
        frame=gen.loc[(gen.ts>=a)&(gen.ts<=cutoff)]
        lat=frame.site_id.map(sites.set_index('site_id').lat).to_numpy();lon=frame.site_id.map(sites.set_index('site_id').lon).to_numpy()
        elev,_,_=baseline.solar(pd.DatetimeIndex(frame.ts),lat,lon);known=set(frame.loc[elev>0,'site_id'])
        assert len(known)==r['folds'][fold]['training_support']['supervised_sites']
        for hour in [10,17]:
            cache=args.cache_dir/fold/f'r{hour}';meta=json.loads((cache/'metadata.json').read_text());train=np.load(cache/'train_rows.npy',mmap_mode='r')
            assert set(sites.site_id.to_numpy()[np.unique(train//meta['n_hours'])])==known
            assert original.sha256(cache/'train_rows.npy')==r['folds'][fold]['training_support']['cache_sha256'][str(hour)]['train_rows']
            assert original.sha256(cache/'metadata.json')==r['folds'][fold]['training_support']['cache_sha256'][str(hour)]['metadata']
        for family in ['tabm','tft']:
            rows=pd.read_parquet(args.a1_dir/family/fold/'site_predictions.parquet')
            np.testing.assert_array_equal(rows.seen,rows.site_id.isin(known))
            for label,mask in [('known_G2',rows.site_id.isin(known)),('cold_G2',~rows.site_id.isin(known))]:
                use=mask&rows.has_peer&rows.daylight&rows.site_id.map(sites.set_index('site_id').current_group).eq('B')
                selected=rows.loc[use];err=selected.ensemble.to_numpy()-selected.energy_peer.to_numpy(dtype=float)
                cap=selected.capacity_kw.to_numpy(dtype='float32');saved=r['folds'][fold]['training_support']['neural_daylight_residuals'][family][label]
                assert len(err)==saved['daylight_rows']
                np.testing.assert_allclose([err.mean(),np.mean(np.abs(err)),np.mean(err/cap)*100,np.mean(np.abs(err)/cap)*100],
                    [saved['bias_kwh'],saved['mae_kwh'],saved['normalized_bias_percentage_points'],saved['normalized_mae_percent']],rtol=1e-12,atol=1e-10)
        policies={name:compute(e,k,c,name) for name in ['original','p4',*POLICIES]}
        if fold=='F2':
            keep=ts<pd.Timestamp('2023-12-01 01:00',tz=TZ)
            calibration={name:metrics(value,w,keep) for name,value in policies.items()}
            for name,entry in calibration.items():
                for key,value in entry.items():np.testing.assert_allclose(value,r['selection']['calibration'][name][key],rtol=1e-12,atol=1e-5)
            eligible=[name for name in POLICIES if calibration[name]['G2_W']<=calibration['p4']['G2_W']+1e-5]
            chosen=min(eligible,key=lambda name:(calibration[name]['W'],POLICIES.index(name))) if eligible else 'original'
            assert chosen==r['selection']['selected_policy']
            for name,value in policies.items():
                for key,amount in metrics(value,w,~keep).items():np.testing.assert_allclose(amount,r['F2_validation_metrics'][name][key],rtol=1e-12,atol=1e-5)
        policies['selected']=policies[r['selection']['selected_policy']]
        output=np.load(folder/f'{fold}_policy_errors.npz')
        for name,value in policies.items():
            np.testing.assert_allclose(value,output[name],rtol=1e-12,atol=1e-8)
            for key,amount in metrics(value,w,np.ones(len(w),dtype=bool)).items():np.testing.assert_allclose(amount,r['folds'][fold]['metrics'][name][key],rtol=1e-12,atol=1e-5)
            months=(ts-pd.Timedelta(hours=1)).strftime('%Y-%m').to_numpy()
            for month in set(months):np.testing.assert_allclose(float(np.sum(np.abs(value[months==month])*w[months==month])),r['folds'][fold]['metrics'][name]['W_by_month'][month],rtol=1e-12,atol=1e-5)
        p=e[:,0];a=.2*(e[:,1]-p);b=.3*(e[:,2]-p);pt,pf=shapley(p,a,b,w)
        dk=.5*k[:,0]+.2*k[:,1]+.3*k[:,2]-k[:,0];dc=.5*c[:,0]+.2*c[:,1]+.3*c[:,2]-c[:,0];pk,pc=shapley(p,dk,dc,w)
        for group,sl in [('G1',slice(0,hours)),('G2',slice(hours,None))]:
            diag=r['folds'][fold]['diagnosis'][group];delta=(np.abs(policies['original'])-np.abs(p))*w
            np.testing.assert_allclose([delta[sl].sum(),pt[sl].sum(),pf[sl].sum(),pk[sl].sum(),pc[sl].sum()],
                [diag['delta_W'],diag['model_attribution']['tabm'],diag['model_attribution']['tft'],diag['cohort_attribution']['known'],diag['cohort_attribution']['cold']],rtol=1e-12,atol=1e-5)
        selected_daily=(np.abs(policies['selected'])*w).reshape(2,days,24).sum(axis=2)
        for reference in ['p4','original']:
            base_daily=(np.abs(policies[reference])*w).reshape(2,days,24).sum(axis=2)
            for group,bs,qs in [('total',base_daily.sum(axis=0),selected_daily.sum(axis=0)),('G2',base_daily[1],selected_daily[1])]:
                for block in ['7','28']:replay(bs,qs,r['folds'][fold]['selected_bootstrap'][reference][group][block])
        result['folds'][fold]=dict(strict_training_support_from_raw=True,attribution_identity_verified=True,all_fixed_policies_and_W_verified=True,bootstrap_replayed=True)
        print(f'{fold}: causal training support, attribution, policy selection/W and bootstrap verified',flush=True)
    m2=r['folds']['F2']['metrics'];m3=r['folds']['F3']['metrics'];mv=r['F2_validation_metrics'];reasons=[]
    if m2['selected']['G2_W']>m2['p4']['G2_W']+1e-5:reasons.append('F2 full G2 worse than P4')
    if mv['selected']['G2_W']>mv['p4']['G2_W']+1e-5:reasons.append('F2 later-half G2 worse than P4')
    if m3['selected']['W']>m3['p4']['W']+1e-5:reasons.append('F3 total worse than P4')
    if m3['selected']['G2_W']>m3['p4']['G2_W']+1e-5:reasons.append('F3 G2 worse than P4')
    if m3['selected']['W']>m3['original']['W']*1.0025:reasons.append('F3 total costs more than 0.25% versus original mixture')
    assert r['support']['reasons']==reasons and r['support']['classification']==('hold' if reasons else 'supported_historical_correction')
    used=gen.loc[(gen.ts>=pd.Timestamp('2023-06-01 01:00',tz=TZ))&(gen.ts<=pd.Timestamp('2025-05-31 17:00',tz=TZ))]
    elev,_,_=baseline.solar(pd.DatetimeIndex(used.ts),used.site_id.map(sites.set_index('site_id').lat).to_numpy(),used.site_id.map(sites.set_index('site_id').lon).to_numpy())
    known=set(used.loc[elev>0,'site_id']);assert len(known)==r['proposed_final_training_support']['supervised_daylight_sites']
    assert int(((sites.current_group=='B')&~sites.site_id.isin(known)).sum())==r['proposed_final_training_support']['cold_G2_sites']
    original.check_data();assert original.sha256(folder/'selection.json')==r['selection_sha256']
    result.update(status='complete',support_verified=r['support']['classification']);write_json(folder/'verification.json',result)
    print('A6 independent verification complete',flush=True)


if __name__=='__main__':main()
