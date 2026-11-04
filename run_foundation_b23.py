"""Retrospective local TabPFNv2/TabICLv2 comparison; fixed full evaluation mask, sampled past context."""
from __future__ import annotations
import argparse,gc,json,platform,subprocess,sys,time
from importlib.metadata import version
from pathlib import Path
from foundation_b23 import (ROOT,CONFIG,BASE,original,torch,np,pd,bounds,peer_labels,score,build,
 dump,status,check_frozen,key_hash,select_context,train_medians,fill,make_model,load_inputs,deny_network_probe)

SOURCE_NAMES=['run_foundation_b23.py','foundation_b23.py','verify_foundation_b23.py','test_foundation_b23.py',
 'b23_foundation_config.json','requirements-foundation.txt','run_peer_w.py','features_p3.py','training_weights.py','train_smoke.py','AB_EVALUATION_CONTRACT.json']

def worker(name,fold,folder):
 network=deny_network_probe();cfg=json.loads(CONFIG.read_text());contract=check_frozen()['folds'][fold]
 started=time.perf_counter();a,cutoff,c,d=bounds(fold)
 status('preparing',f'{name} {fold}: build past context and original evaluation universe',folder.parent.with_suffix('.log'),active_experiment='B2' if name=='tabpfn' else 'B3')
 b,sites,train,valid,prices,fa,fb=load_inputs(fold)
 assert len(train)==contract['train_raw_rows'] and len(valid)==contract['eval_raw_rows']
 output=valid.rename(columns={'energy':'energy_raw'}).merge(sites[['site_id','capacity_kw']],on='site_id',validate='many_to_one')
 output['energy_peer'],output['has_peer']=peer_labels(output.energy_raw,output.capacity_kw)
 digest=key_hash(output);assert digest==contract['evaluation_key_mask_sha256']
 baseline=json.loads((BASE/fold/'summary.json').read_text())
 infos={};common_indices=None;cfg_sha=original.sha256(CONFIG)
 for r in [10,17]:
  print(f'{name} {fold} r{r}: build training features',flush=True)
  f,columns=build(b,train,sites,fa,fb,r,'station_distance');assert columns==contract['feature_columns']
  f=f.loc[f.elev>0].reset_index(drop=True)
  assert f.ts.max()<=cutoff and len(f)==baseline['models'][str(r)]['train_rows']
  indices=select_context(f.cap,cfg['context_rows'],cfg['seed'])
  if common_indices is None:common_indices=indices
  else:assert np.array_equal(indices,common_indices)
  medians=train_medians(f[columns].to_numpy('float32'))
  ctx=fill(f.iloc[indices][columns],medians);target=(f.iloc[indices].energy/f.iloc[indices].cap).clip(0,1.2).to_numpy('float32')
  keys=f.iloc[indices][['site_id','ts']].reset_index(drop=True)
  context_path=folder/f'context_round_{r}.npz';keys_path=folder/f'context_keys_round_{r}.parquet'
  if context_path.exists():
   saved=np.load(context_path)
   for k,val in dict(x=ctx,y=target,medians=medians,positions=indices).items():assert np.array_equal(saved[k],val)
   assert keys.equals(pd.read_parquet(keys_path))
  else:
   np.savez_compressed(context_path,x=ctx,y=target,medians=medians,positions=indices)
   keys.to_parquet(keys_path,index=False)
  info=dict(train_rows=len(f),context_rows=len(ctx),train_min=str(f.ts.min()),train_max=str(f.ts.max()),
     context_sha256=original.sha256(context_path),context_keys_sha256=original.sha256(keys_path),
     context_sampling='capacity-proportional without replacement; no exact weighted-L1 optimization',
     imputation_fit='full daylight training rows only',context_fit_min=str(keys.ts.min()),context_fit_max=str(keys.ts.max()))
  del f;gc.collect()
  status('training',f'{name} {fold} round{r}: fit cached context8192',folder.parent.with_suffix('.log'),active_experiment='B2' if name=='tabpfn' else 'B3')
  model=make_model(name,cfg);t=time.perf_counter();model.fit(ctx,target);torch.mps.synchronize()
  info['context_fit_seconds']=time.perf_counter()-t
  print(f'{name} {fold} r{r}: cached fit {info["context_fit_seconds"]:.3f}s',flush=True)
  f,cols=build(b,valid,sites,fa,fb,r,'station_distance');assert cols==columns
  assert f[['site_id','ts']].reset_index(drop=True).equals(output[['site_id','ts']].reset_index(drop=True))
  day=f.elev.to_numpy()>0;positions=np.flatnonzero(day)
  x=fill(f.loc[day,columns],medians);capacity=f.cap.to_numpy()
  output['daylight']=day
  del f;gc.collect()
  values_path=folder/f'pred_round_{r}.npy';progress_path=folder/f'progress_round_{r}.json'
  if values_path.exists() and progress_path.exists():
   prog=json.loads(progress_path.read_text());assert prog['config_sha256']==cfg_sha
   assert prog['context_sha256']==info['context_sha256'] and prog['expected_day_rows']==len(x)
   predictions=np.load(values_path,mmap_mode='r+');completed=prog['completed_day_rows'];active_seconds=prog['predict_active_seconds']
   assert predictions.shape==(len(output),) and np.isfinite(predictions[positions[:completed]]).all()
  else:
   predictions=np.lib.format.open_memmap(values_path,mode='w+',shape=(len(output),),dtype='float64')
   predictions[:]=0;predictions[positions]=np.nan;predictions.flush();completed=0;active_seconds=0.
  peak_driver=torch.mps.driver_allocated_memory()/1024**2
  # Test real-feature query independence once before annual scoring; no validation labels used.
  sample=x[:min(cfg['batch_rows'],len(x))]
  t=time.perf_counter();first=np.asarray(model.predict(sample,output_type='median'));small=np.asarray(model.predict(sample[:8],output_type='median'))
  consistency=float(np.max(np.abs(first[:8]-small)))
  assert consistency<1e-4 and np.isfinite(first).all(),'Batch-dependent/nonfinite model output'
  info['real_feature_batch_max_abs_diff']=consistency
  for start in range(completed,len(x),cfg['batch_rows']):
   end=min(start+cfg['batch_rows'],len(x));t=time.perf_counter()
   raw=np.asarray(model.predict(x[start:end],output_type='median'));torch.mps.synchronize()
   if not np.isfinite(raw).all():raise ValueError('Nonfinite foundation prediction')
   predictions[positions[start:end]]=np.maximum(raw,0)*capacity[positions[start:end]]
   active_seconds+=time.perf_counter()-t
   peak_driver=max(peak_driver,torch.mps.driver_allocated_memory()/1024**2)
   if (start//cfg['batch_rows'])%8==7 or end==len(x):
    predictions.flush()
    prog=dict(config_sha256=cfg_sha,context_sha256=info['context_sha256'],expected_day_rows=len(x),completed_day_rows=end,
              predict_active_seconds=active_seconds,observed_max_mps_driver_mib=peak_driver)
    dump(progress_path,prog)
    stage=f'{name} {fold} r{r}: predicted {end}/{len(x)} daylight rows'
    print(stage,f'active_seconds={active_seconds:.1f}',flush=True)
    status('training',stage,folder.parent.with_suffix('.log'),active_experiment='B2' if name=='tabpfn' else 'B3',
        fold=fold,round=r,completed_prediction_rows=end,expected_prediction_rows=len(x))
  assert np.isfinite(predictions).all() and (predictions>=0).all()
  output[f'pred_{r}']=np.asarray(predictions)
  info.update(predict_active_seconds=active_seconds,daylight_eval_rows=len(x),observed_max_mps_driver_mib=peak_driver,
     peak_process_memory_mib=original.peak_memory_mib())
  infos[str(r)]=info
  dump(folder/f'model_round_{r}_info.json',info)
  del x,model,predictions;gc.collect();torch.mps.empty_cache()
 output['ensemble']=0.25*output.pred_10+0.75*output.pred_17
 scores={}
 for pred in ['pred_10','pred_17','ensemble']:
  scores[pred],agg=score(output,sites,prices,c,d,prediction=pred)
  if pred=='ensemble':agg.to_csv(folder/'group_errors.csv',index=False)
 output.to_parquet(folder/'site_predictions.parquet',index=False)
 b1=json.loads((ROOT/'outputs/b1_annual'/fold/'summary.json').read_text())['metrics']['ensemble']['W']
 report=dict(status='awaiting_verification',model=name,fold=fold,network_policy=network,train_start=str(a),train_end_inclusive=str(cutoff),
  eval_start=str(c),eval_end_exclusive=str(d),train_raw_rows=len(train),eval_raw_rows=len(valid),
  evaluation_key_mask_sha256=digest,feature_columns=columns,weight_17=0.75,models=infos,metrics=scores,baseline_W=contract['baseline_W'],
  baseline_change_percent=(scores['ensemble']['W']/contract['baseline_W']-1)*100,B1_W=b1,B1_change_percent=(scores['ensemble']['W']/b1-1)*100,
  total_seconds=time.perf_counter()-started,peak_process_memory_mib=original.peak_memory_mib())
 dump(folder/'summary.json',report)
 print(f'{name} {fold} W={scores["ensemble"]["W"]:.6f} P4_change={report["baseline_change_percent"]:+.4f}%',flush=True)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['tabpfn','tabicl'],required=True)
 p.add_argument('--fold',choices=['F2','F3']);p.add_argument('--output-dir',type=Path)
 args=p.parse_args();folder=(args.output_dir or ROOT/f'outputs/b23_{args.model}').resolve()
 if not folder.is_relative_to(ROOT/'outputs'):raise ValueError('B outputs only')
 if args.fold:
  folder.mkdir(parents=True,exist_ok=True);worker(args.model,args.fold,folder);return
 network=deny_network_probe();check_frozen();folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
 hashes={n:original.sha256(ROOT/n) for n in SOURCE_NAMES}
 summary=folder/'summary.json'
 if summary.exists():
  report=json.loads(summary.read_text());assert report['source_sha256']==hashes,'Source changed; use a new run folder'
 else:
  source=folder/'source';source.mkdir()
  for n in SOURCE_NAMES:(source/n).write_bytes((ROOT/n).read_bytes())
  report=dict(status='running',model=args.model,config=json.loads(CONFIG.read_text()),network_policy=network,
   source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
   source_sha256=hashes,versions={n:version(n) for n in ['numpy','pandas','pyarrow','lightgbm','torch','tabpfn','tabicl']},
   machine=dict(platform=platform.platform(),architecture=platform.machine(),device='MPS'),folds={})
 dump(summary,report)
 for fold in ['F2','F3']:
  target=folder/fold;target.mkdir(exist_ok=True)
  if not (target/'summary.json').exists():
   subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--model',args.model,'--fold',fold,'--output-dir',str(target)],cwd=ROOT,check=True)
  report['folds'][fold]=json.loads((target/'summary.json').read_text());dump(summary,report)
 report.update(status='awaiting_independent_verification',last_invocation_seconds=time.perf_counter()-started)
 dump(summary,report)
 status('verifying',f'{args.model}: F2/F3 ready for independent checks',folder.with_suffix('.log'),active_experiment='B2' if args.model=='tabpfn' else 'B3')

if __name__=='__main__':main()
