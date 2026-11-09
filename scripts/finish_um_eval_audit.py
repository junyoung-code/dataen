"""Wait for this UM collection, audit raw18, notify once; never train/upload."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_s929_um import save, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root',type=Path,required=True)
    args=ap.parse_args(); r=args.run_root.resolve()
    lock=(r/'audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (r/'audit_completion.json').exists():return
    try:
        while True:
            p=r/'um_eval_collection/status.json'
            state=json.loads(p.read_text()) if p.exists() else {}
            if state.get('status')=='failed':raise RuntimeError('UM collector failed; inspect collection status/log')
            if state.get('status')=='collected':break
            save(r/'audit_status.json',{'status':'waiting_for_collection','pid':os.getpid(),
                                     'completed_days':state.get('completed_days',0),'time':time.time()})
            time.sleep(30)
        save(r/'audit_status.json',{'status':'auditing','pid':os.getpid(),'time':time.time()})
        builder=Path(__file__).resolve().with_name('build_um_features.py')
        with (r/'audit_build.log').open('a') as log:
            subprocess.run([sys.executable,str(builder),'--collection',str(r/'um_eval_collection'),
                '--points',str(r/'um_fixed_points.json'),'--reference',str(r/'um_core_reference_verification.json'),
                '--output-dir',str(r/'um_eval_features'),'--raw-only'],stdout=log,stderr=subprocess.STDOUT,check=True)
        meta=json.loads((r/'um_eval_features/um_features.json').read_text())
        assert meta['status']=='verified' and meta['raw_only'] and meta['expected_days']==365
        assert meta['start']=='2025-06-01' and meta['end']=='2026-05-31' and meta['finite_core_cells']>0
        assert sha(r/'um_eval_features/um_core_hourly.parquet')==meta['raw_sha256']
        summary={'status':'complete','time':time.time(),'expected_days':365,'complete_days':meta['complete_days'],
                 'finite_core_cells':meta['finite_core_cells'],'expected_core_cells':meta['expected_core_cells'],
                 'training_started':False,'raw_sha256':meta['raw_sha256']}
        save(r/'audit_completion.json',summary);save(r/'audit_status.json',summary)
        message='UM 추가 수집·검산이 완료됐습니다.'
    except Exception as e:
        save(r/'audit_status.json',{'status':'failed','error':str(e),'time':time.time(),'training_started':False})
        message='UM 수집·검산이 중단됐습니다. 오류 확인이 필요합니다.'
    apple='display notification '+json.dumps(message,ensure_ascii=False)+' with title "UM 수집·검산" sound name "Glass"'
    n=subprocess.run(['/usr/bin/osascript','-e',apple],capture_output=True,text=True,timeout=20)
    save(r/'audit_notification.json',{'time':time.time(),'exit_code':n.returncode,'stderr':n.stderr})


if __name__=='__main__':main()
