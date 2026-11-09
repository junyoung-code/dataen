"""One-job macOS completion notification using SSH, without any model/API calls.

The Mac mini training is independent of this process. A sleeping MacBook checks
again after waking; restarting the MacBook requires restarting this notifier.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time


REMOTE_CHECK = '''
from pathlib import Path
import hashlib,json,sys
r=Path.home()/sys.argv[1]
s=json.loads((r/'status.json').read_text())
if s['status']=='complete':
 c=json.loads((r/'completion.json').read_text())
 p=r/'comparison/results.json'
 assert hashlib.sha256(p.read_bytes()).hexdigest()==c['results_sha256']
 result=json.loads(p.read_text())
 assert result['status']=='verified' and len(result['scores'])==4
 s['comparison']=result
 s['completion']=c
print(json.dumps(s))
'''


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', default='macmini')
    ap.add_argument('--remote-root', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'notifier.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output / 'notification_sent.json').exists():
        return
    command = shlex.join(['python3', '-c', REMOTE_CHECK, args.remote_root])
    while True:
        try:
            proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                   args.host, command], capture_output=True, text=True, timeout=30, check=True)
            state = json.loads(proc.stdout)
            save(args.output / 'notifier_latest.json', {**state, 'checked_at': time.time()})
            if state['status'] in ('complete', 'failed'):
                if state['status'] == 'complete':
                    save(args.output / 'results.json', state['comparison'])
                    save(args.output / 'completion.json', state['completion'])
                message = 'S929 + UM 비교가 완료됐습니다.' if state['status'] == 'complete' else 'S929 + UM 실행이 중단됐습니다. 오류 확인이 필요합니다.'
                apple = 'display notification ' + json.dumps(message, ensure_ascii=False) + ' with title "S929 + UM" sound name "Glass"'
                notification = subprocess.run(['/usr/bin/osascript', '-e', apple], capture_output=True, text=True, timeout=20)
                save(args.output / 'notification_sent.json', {'job_status': state['status'], 'time': time.time(),
                     'notification_exit_code': notification.returncode, 'stderr': notification.stderr,
                     'note': 'OS command accepted does not certify notification visibility/settings'})
                return
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            save(args.output / 'notifier_connection.json', {'time': time.time(), 'error': str(error)})
        time.sleep(60)


if __name__ == '__main__':
    main()
