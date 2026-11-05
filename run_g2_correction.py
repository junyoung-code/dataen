"""A6: mathematical root-cause diagnosis, fixed G2 corrections, historical guards."""
import argparse,json,platform,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import train_smoke as original
from annual_neural_data import check_contract
from candidate_validation import mixed,daily_amounts,block_bootstrap
from g2_correction import POLICIES,apply_policy,attribution,select_policy,support
from run_peer_w import TZ,bounds,write_json

ROOT=Path(__file__).resolve().parent


def metrics(error,price,ts,keep=None):
    if keep is None:keep=np.ones(len(error),dtype=bool)
    amount=np.abs(error)*price;half=len(error)//2
    month=(pd.DatetimeIndex(ts)-pd.Timedelta(hours=1)).strftime('%Y-%m').to_numpy()
    return dict(W=float(amount[keep].sum()),G1_W=float(amount[:half][keep[:half]].sum()),G2_W=float(amount[half:][keep[half:]].sum()),
        W_by_month={m:float(amount[keep&(month==m)].sum()) for m in sorted(set(month[keep]))})


def strict_support(fold,args,sites,expected_prediction_hashes):
    supported=[];cache_hashes={}
    for hour in [10,17]:
        folder=args.cache_dir/fold/f'r{hour}';meta=json.loads((folder/'metadata.json').read_text())
        _,cutoff,_,_=bounds(fold);axis_start=pd.Timestamp(meta['axis_start'])
        assert pd.Timestamp(meta['train_max'])<=cutoff
        train=np.load(folder/'train_rows.npy',mmap_mode='r');n=meta['n_hours']
        assert np.all(train%n<=int((cutoff-axis_start).total_seconds()/3600))
        ids=set(sites.site_id.to_numpy()[np.unique(train//n)]);supported.append(ids)
        cache_hashes[str(hour)]=dict(metadata=original.sha256(folder/'metadata.json'),train_rows=original.sha256(folder/'train_rows.npy'))
    assert supported[0]==supported[1]
    result={};known=supported[0]
    for family in ['tabm','tft']:
        path=args.a1_dir/family/fold/'site_predictions.parquet'
        assert original.sha256(path)==expected_prediction_hashes[family]
        rows=pd.read_parquet(path)
        is_known=rows.site_id.isin(known)
        np.testing.assert_array_equal(is_known,rows.seen)
        is_g2=rows.site_id.map(sites.set_index('site_id').current_group).eq('B')
        stats={}
        for key,keep in [('known_G2',is_known&is_g2),('cold_G2',~is_known&is_g2)]:
            frame=rows.loc[keep&rows.has_peer&rows.daylight]
            error=frame.ensemble-frame.energy_peer.astype(float)
            cap=frame.capacity_kw.to_numpy(dtype='float32')
            stats[key]=dict(daylight_rows=len(frame),sites=int(frame.site_id.nunique()),bias_kwh=float(error.mean()),
                mae_kwh=float(error.abs().mean()),normalized_bias_percentage_points=float(np.mean(error/cap)*100),
                normalized_mae_percent=float(np.mean(np.abs(error)/cap)*100))
        result[family]=stats
    # Aggregate transformed-input clipping diagnostic; no raw feature/coordinate output.
    folder=args.cache_dir/fold/'r17';meta=json.loads((folder/'metadata.json').read_text())
    x=np.load(folder/'x.npy',mmap_mode='r');ix=np.load(folder/'eval_rows.npy',mmap_mode='r')
    site_order={s:i for i,s in enumerate(sites.site_id)}
    row_idx=rows.site_id.map(site_order).to_numpy()*meta['n_hours']+((rows.ts-pd.Timestamp(meta['axis_start'])).dt.total_seconds()/3600).to_numpy(dtype='int64')
    np.testing.assert_array_equal(ix,row_idx)
    clips={}
    for key,keep in [('known_G2',is_known&is_g2),('cold_G2',~is_known&is_g2)]:
        use=(keep&rows.has_peer&rows.daylight).to_numpy()
        v=x[ix[use]];fraction=np.mean(np.abs(v)>=9.99999,axis=0)
        clips[key]={name:float(f) for name,f in zip(meta['columns'],fraction) if f>0}
    return dict(supervised_sites=len(known),flag_discrepancies=0,cache_sha256=cache_hashes,
        neural_daylight_residuals=result,transformed_feature_clip_fraction=clips)


def diagnosis(errors,known,cold,price,ts):
    half=len(price)//2;p=errors[:,0];t=.2*(errors[:,1]-p);f=.3*(errors[:,2]-p)
    pt,pf=attribution(p,t,f,price)
    dk=mixed(known)-known[:,0];dc=mixed(cold)-cold[:,0];pk,pc=attribution(p,dk,dc,price)
    delta=(np.abs(mixed(errors))-np.abs(p))*price
    month=(pd.DatetimeIndex(ts)-pd.Timedelta(hours=1)).strftime('%Y-%m').to_numpy()
    result={}
    for group,idx in [('G1',np.arange(half)),('G2',np.arange(half,len(price)))]:
        result[group]=dict(delta_W=float(delta[idx].sum()),model_attribution=dict(tabm=float(pt[idx].sum()),tft=float(pf[idx].sum())),
            cohort_attribution=dict(known=float(pk[idx].sum()),cold=float(pc[idx].sum())),
            by_month={m:dict(delta_W=float(delta[idx][month[idx]==m].sum()),tabm=float(pt[idx][month[idx]==m].sum()),tft=float(pf[idx][month[idx]==m].sum()),cold=float(pc[idx][month[idx]==m].sum())) for m in sorted(set(month))},
            by_hour={str(h):float(delta[idx][pd.DatetimeIndex(ts)[idx].hour==h].sum()) for h in range(24)},
            weighted_signed_error_kwh={name:float(np.sum(errors[idx,i]*price[idx])/price[idx].sum()) for i,name in enumerate(['p4','tabm','tft'])})
        np.testing.assert_allclose(pt[idx].sum()+pf[idx].sum(),delta[idx].sum(),atol=1e-5)
        np.testing.assert_allclose(pk[idx].sum()+pc[idx].sum(),delta[idx].sum(),atol=1e-5)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--a5-dir',type=Path,required=True)
    p.add_argument('--a1-dir',type=Path,required=True);p.add_argument('--cache-dir',type=Path,required=True);args=p.parse_args()
    args.output_dir.resolve().relative_to((ROOT/'outputs').resolve());args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();check_contract();original.check_data()
    a5=json.loads((args.a5_dir/'summary.json').read_text());assert a5['status']=='complete'
    assert original.sha256(args.a5_dir/'verification.json')==a5['verification_sha256']
    for name,h in a5['source_sha256'].items():assert original.sha256(ROOT/name)==h
    names=['g2_correction.py','run_g2_correction.py','verify_g2_correction.py','test_g2_correction.py','launch_a6.sh']
    source=args.output_dir/'source';source.mkdir();hashes={n:original.sha256(ROOT/n) for n in names}
    for n in names:(source/n).write_bytes((ROOT/n).read_bytes())
    report=dict(status='running',experiment='A6',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
        source_sha256=hashes,a5_summary_sha256=original.sha256(args.a5_dir/'summary.json'),retrained_models=0,original_weights=[.5,.2,.3],
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),folds={})
    sites=pd.read_csv(original.DATA/'sites.csv');all_metrics={}
    selection_hash=None
    for fi,fold in enumerate(['F2','F3']):
        arrays=args.a5_dir/f'{fold}_arrays.npz';z=np.load(arrays);price=z['weight']
        e,k,c=z['errors_all'],z['errors_seen'],z['errors_new']
        _,_,a,b=bounds(fold);axis=pd.date_range(a,b,freq='h',inclusive='left');ts=axis.append(axis)
        actual_support=strict_support(fold,args,sites,a5['folds'][fold]['input_prediction_sha256'])
        output={p:apply_policy(e,k,c,p) for p in ['original',*POLICIES]};output['p4']=e[:,0]
        values={name:metrics(v,price,ts) for name,v in output.items()}
        if fold=='F2':
            cal=ts<pd.Timestamp('2023-12-01 01:00',tz=TZ);val=~cal
            calibration={name:metrics(v,price,ts,cal) for name,v in output.items()}
            selected=select_policy(calibration)
            selection=dict(selected_policy=selected,calibration=calibration,source_sha256=hashes,
                selection_period='F2 first six months',confirmation_periods=['F2 later six months','F3 full'],
                input_sha256=original.sha256(arrays),policies=list(POLICIES),definition='G1 unchanged; G2 causal supervised-training support or fixed removal of TabM',
                warning='Base 50:20:30 already selected using full F2; historical comparison only')
            write_json(args.output_dir/'selection.json',selection);selection_hash=original.sha256(args.output_dir/'selection.json')
            all_metrics['F2_validation']={name:metrics(v,price,ts,val) for name,v in output.items()}
            all_metrics['F2_validation']['selected']=all_metrics['F2_validation'][selected]
            report['selection']=selection;report['selection_sha256']=selection_hash
            print('A6 policy selected on first-half F2, frozen before F3:',selected,flush=True)
        assert original.sha256(args.output_dir/'selection.json')==selection_hash
        output['selected']=output[selected];values['selected']=values[selected];all_metrics[fold]=values
        boots={}
        q=daily_amounts(output['selected'],price)
        for ref in ['p4','original']:
            b=daily_amounts(output[ref],price);boots[ref]={}
            for group,bs,qs in [('total',b.sum(axis=0),q.sum(axis=0)),('G2',b[1],q[1])]:
                boots[ref][group]={str(block):block_bootstrap(bs,qs,block,20261106+fi*1000+block) for block in [7,28]}
        report['folds'][fold]=dict(metrics=values,diagnosis=diagnosis(e,k,c,price,ts),training_support=actual_support,
            input_arrays_sha256=original.sha256(arrays),selected_bootstrap=boots)
        np.savez_compressed(args.output_dir/f'{fold}_policy_errors.npz',**output)
        write_json(args.output_dir/'summary.json',report)
        print(f'{fold}: G2 P4={values["p4"]["G2_W"]:.3f}; original={values["original"]["G2_W"]:.3f}; selected={values["selected"]["G2_W"]:.3f}',flush=True)
    # Check proposed final two-year support footprint only; no final model is fitted.
    gen=pd.read_parquet(original.DATA/'train/generation.parquet');gen.ts=pd.to_datetime(gen.ts).dt.tz_convert(TZ)
    used=gen.loc[(gen.ts>=pd.Timestamp('2023-06-01 01:00',tz=TZ))&(gen.ts<=pd.Timestamp('2025-05-31 17:00',tz=TZ))]
    baseline=original.load_baseline();lat=used.site_id.map(sites.set_index('site_id').lat).to_numpy();lon=used.site_id.map(sites.set_index('site_id').lon).to_numpy()
    elev,_,_=baseline.solar(pd.DatetimeIndex(used.ts),lat,lon)
    present=set(used.loc[elev>0,'site_id'])
    report['proposed_final_training_support']=dict(supervised_daylight_sites=len(present),total_sites=len(sites),
        cold_G2_sites=int(((sites.current_group=='B')&~sites.site_id.isin(present)).sum()),no_final_models_trained=True)
    report['F2_validation_metrics']=all_metrics['F2_validation'];report['support']=support(all_metrics)
    assert all(original.sha256(ROOT/n)==h for n,h in hashes.items())
    assert original.sha256(args.output_dir/'selection.json')==selection_hash
    report.update(status='computed_pending_verification',total_seconds=time.perf_counter()-started,peak_process_memory_mib=original.peak_memory_mib())
    write_json(args.output_dir/'summary.json',report);print('A6 computed:',report['support'],flush=True)


if __name__=='__main__':main()
