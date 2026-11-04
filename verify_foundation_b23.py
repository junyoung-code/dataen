"""Independent raw-label/mask/NumPy score checks and replay from saved training contexts."""
from __future__ import annotations
import argparse,gc,hashlib,json,time
from pathlib import Path
from foundation_b23 import ROOT,CONFIG,BASE,original,torch,np,pd,build,dump,status,make_model,load_inputs,deny_network_probe


def close(a,b):
 if not np.isclose(a,b,rtol=1e-12,atol=1e-3):raise ValueError('Independent aggregate mismatch')

def independent_scores(rows,sites,prices,start,end,prediction):
 axis=pd.date_range(start,end,freq='h',inclusive='left');n=len(axis);times=axis.get_indexer(rows.ts)
 assert (times>=0).all()
 g=rows.site_id.map(sites.set_index('site_id').current_group).map({'A':0,'B':1}).to_numpy(int)
 has=rows.has_peer.to_numpy(bool);idx=g[has]*n+times[has]
 sums=np.bincount(idx,weights=rows[prediction].to_numpy(float)[has]-rows.energy_peer.to_numpy(float)[has],minlength=2*n).reshape(2,n)
 p=prices.set_index('ts').reindex(axis);k=p.k.to_numpy(float) if 'k' in p else np.where(axis.hour.isin(range(11,16)),3.,1.)
 weights=p.smp.to_numpy(float)*k;assert np.isfinite(weights).all() and (weights>=0).all()
 contrib=np.abs(sums)*weights[None,:];months=(axis-pd.Timedelta(hours=1)).strftime('%Y-%m')
 return dict(W=float(contrib.sum()),W_by_group={g:float(contrib[i].sum()) for i,g in enumerate(['G1','G2'])},
  W_by_month={m:float(contrib[:,months==m].sum()) for m in sorted(set(months))})


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['tabpfn','tabicl'],required=True);args=p.parse_args()
 network=deny_network_probe();folder=ROOT/f'outputs/b23_{args.model}';started=time.perf_counter()
 status('verifying',f'{args.model}: independent raw data/NumPy W/context/replay checks',folder.with_suffix('.verify.log'),active_experiment='B2' if args.model=='tabpfn' else 'B3')
 summary=json.loads((folder/'summary.json').read_text());cfg=json.loads(CONFIG.read_text());contract=json.loads((ROOT/'AB_EVALUATION_CONTRACT.json').read_text())
 for name,digest in summary['source_sha256'].items():
  assert original.sha256(ROOT/name)==digest and original.sha256(folder/'source'/name)==digest
 for name,digest in contract['fixed_source_sha256'].items():assert original.sha256(ROOT/name)==digest
 assert original.sha256(ROOT/cfg[args.model]['weights'])==cfg[args.model]['sha256']
 verified={};comparison=[]
 for fold,c in contract['folds'].items():
  report=summary['folds'][fold];b,sites,train,valid,prices,fa,fb=load_inputs(fold)
  rows=pd.read_parquet(folder/fold/'site_predictions.parquet');ref=pd.read_parquet(BASE/fold/'site_predictions.parquet')
  assert len(rows)==len(valid)==len(ref)==c['eval_raw_rows'] and len(train)==c['train_raw_rows']
  assert not rows.duplicated(['site_id','ts']).any()
  assert rows[['site_id','ts']].reset_index(drop=True).equals(valid[['site_id','ts']].reset_index(drop=True))
  assert rows[['site_id','ts']].equals(ref[['site_id','ts']])
  e=valid.energy.to_numpy(float);cap=valid.site_id.map(sites.set_index('site_id').capacity_kw).to_numpy('float32').astype(float)
  has=np.isfinite(e)&(e>=0)&(e<=1.05*cap);y=np.where(has,e,np.nan).astype('float32')
  assert np.array_equal(rows.energy_raw,e) and np.array_equal(rows.has_peer,has)
  np.testing.assert_allclose(rows.energy_peer,y,rtol=0,atol=0,equal_nan=True)
  for col in ['energy_raw','energy_peer','capacity_kw','has_peer']:assert np.array_equal(rows[col],ref[col],equal_nan=True)
  keys=rows.sort_values(['site_id','ts']).reset_index(drop=True)[['site_id','ts','has_peer']]
  digest=hashlib.sha256(keys.to_csv(index=False,lineterminator='\n').encode()).hexdigest();assert digest==c['evaluation_key_mask_sha256']
  assert np.isfinite(rows[['pred_10','pred_17','ensemble']]).all().all()
  assert (rows[['pred_10','pred_17','ensemble']]>=0).all().all()
  assert (rows.loc[~rows.daylight,['pred_10','pred_17','ensemble']]==0).all().all()
  np.testing.assert_allclose(rows.ensemble,0.25*rows.pred_10+0.75*rows.pred_17,rtol=1e-14)
  computed={};start=pd.Timestamp(c['eval_start']);end=pd.Timestamp(c['eval_end_exclusive'])
  for prediction in ['pred_10','pred_17','ensemble']:
   computed[prediction]=independent_scores(rows,sites,prices,start,end,prediction);stored=report['metrics'][prediction]
   close(computed[prediction]['W'],stored['W'])
   for metric in ['W_by_group','W_by_month']:
    for key,value in computed[prediction][metric].items():close(value,stored[metric][key])
  base=independent_scores(ref,sites,prices,start,end,'ensemble');close(base['W'],c['baseline_W'])
  b1_rows=pd.read_parquet(ROOT/'outputs/b1_annual'/fold/'site_predictions.parquet')
  b1=independent_scores(b1_rows,sites,prices,start,end,'ensemble')
  for level in ['W_by_group','W_by_month']:
   for key,value in computed['ensemble'][level].items():
    p4=base[level][key];g=b1[level][key]
    comparison.append(dict(model=args.model,fold=fold,level=level,key=key,P4_W=p4,B1_W=g,model_W=value,P4_change_percent=(value/p4-1)*100,B1_change_percent=(value/g-1)*100))
  replay={}
  for r in [10,17]:
   ctx_path=folder/fold/f'context_round_{r}.npz';key_path=folder/fold/f'context_keys_round_{r}.parquet';saved=np.load(ctx_path)
   info=report['models'][str(r)]
   assert original.sha256(ctx_path)==info['context_sha256'] and original.sha256(key_path)==info['context_keys_sha256']
   f,cols=build(b,train,sites,fa,fb,r,'station_distance');assert cols==c['feature_columns']
   f=f.loc[f.elev>0].reset_index(drop=True)
   capacities=f.cap.to_numpy(float);positions=np.sort(np.random.default_rng(cfg['seed']).choice(len(f),cfg['context_rows'],replace=False,p=capacities/capacities.sum()))
   assert len(f)==info['train_rows'] and np.array_equal(saved['positions'],positions)
   original_x=f[cols].to_numpy('float32');med=np.nanmedian(np.where(np.isfinite(original_x),original_x,np.nan),axis=0);med=np.where(np.isfinite(med),med,0).astype('float32')
   np.testing.assert_array_equal(saved['medians'],med)
   context=np.where(np.isfinite(original_x[positions]),original_x[positions],med).astype('float32')
   np.testing.assert_array_equal(saved['x'],context)
   target=(f.iloc[positions].energy/f.iloc[positions].cap).clip(0,1.2).to_numpy('float32');np.testing.assert_array_equal(saved['y'],target)
   assert f.iloc[positions][['site_id','ts']].reset_index(drop=True).equals(pd.read_parquet(key_path))
   assert f.ts.min()>=pd.Timestamp(c['train_start']) and f.ts.max()<=pd.Timestamp(c['train_end_inclusive'])
   if args.model=='tabicl':
    other=np.load(ROOT/f'outputs/b23_tabpfn/{fold}/context_round_{r}.npz')
    for name in ['x','y','medians','positions']:np.testing.assert_array_equal(saved[name],other[name])
   del f,original_x;gc.collect()
   sample_positions=np.unique(np.linspace(0,len(valid)-1,1024).astype(int));sample=valid.iloc[sample_positions]
   sf,_=build(b,sample,sites,fa,fb,r,'station_distance');day=sf.elev.to_numpy()>0
   assert np.array_equal(day,rows.iloc[sample_positions].daylight)
   q=sf[cols].to_numpy('float32');q=np.where(np.isfinite(q),q,med).astype('float32')
   model=make_model(args.model,cfg);model.fit(context,target)
   predictions=np.zeros(len(sample));predictions[day]=np.maximum(model.predict(q[day],output_type='median'),0)
   actual=rows.iloc[sample_positions][f'pred_{r}'].to_numpy()/sf.cap.to_numpy()
   np.testing.assert_allclose(predictions,actual,rtol=1e-4,atol=1e-5)
   progress=json.loads((folder/fold/f'progress_round_{r}.json').read_text());assert progress['completed_day_rows']==progress['expected_day_rows']
   replay[str(r)]=dict(rows=len(sample),max_abs_normalized_diff=float(np.max(np.abs(predictions-actual))))
   del model;gc.collect();torch.mps.empty_cache()
  verified[fold]=dict(status='passed',evaluation_key_mask_sha256=digest,eval_rows=len(rows),independent_W=computed['ensemble']['W'],baseline_W=base['W'],B1_W=b1['W'],
    P4_change_percent=(computed['ensemble']['W']/base['W']-1)*100,B1_change_percent=(computed['ensemble']['W']/b1['W']-1)*100,replay=replay)
  print(args.model,fold,'raw labels/mask/NumPy W/context/replay passed',flush=True)
 pd.DataFrame(comparison).to_csv(folder/'monthly_group_comparison.csv',index=False)
 result=dict(status='passed',model=args.model,network_policy=network,folds=verified,seconds=time.perf_counter()-started,
   context_rows_per_model=cfg['context_rows'],historical_publication_caveat=cfg['publication_caveat'])
 dump(folder/'verification.json',result);summary['status']='complete';summary['verification']=result
 for fold in summary['folds']:summary['folds'][fold]['status']='complete'
 dump(folder/'summary.json',summary)
 status('preparing' if args.model=='tabpfn' else 'complete',f'{args.model}: annual comparison and independent checks passed',folder.with_suffix('.verify.log'),
  active_experiment='B3 next' if args.model=='tabpfn' else 'B1-B3 complete')
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
