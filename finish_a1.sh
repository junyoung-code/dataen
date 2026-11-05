#!/bin/zsh
set -u
cd "${0:A:h}"
A1_PYTHON=../oibc_mac_mini/.runtime/neural-venv/bin/python
A1_RUN=outputs/a1_20261104_01
[[ "$(cat "$A1_RUN.train.exit")" == 0 && "$(cat "$A1_RUN.verify.exit")" == 0 ]] || exit 1
/usr/bin/caffeinate -i env DYLD_LIBRARY_PATH="$PWD/../oibc_mac_mini/.venv/lib/python3.12/site-packages/lightgbm/lib" MPLCONFIGDIR="$PWD/outputs/preflight/mpl" "$A1_PYTHON" -u audit_annual_inputs.py --output-dir "$A1_RUN" --cache-dir outputs/annual_cache_20261104_01 > "$A1_RUN.supplement.log" 2>&1
result=$?
printf '%s\n' "$result" > "$A1_RUN.supplement.exit"
(( result == 0 )) || exit "$result"
"$A1_PYTHON" - <<'PY'
import datetime,hashlib,json
from pathlib import Path
root=Path('outputs/a1_20261104_01')
for name in ['verification.json','supplemental_verification.json']:
 assert json.loads((root/name).read_text())['status']=='complete'
summary=root/'summary.json'
original=summary.read_bytes()
backup=root/'summary.training.json'
assert not backup.exists()
backup.write_bytes(original)
report=json.loads(original)
assert report['status']=='trained_pending_independent_verification'
report['status']='complete'
report['verified_at']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
report['verification_sha256']={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['verification.json','supplemental_verification.json']}
report['supplemental_source_sha256']={}
for name in ['audit_annual_inputs.py','launch_a1.sh','finish_a1.sh']:
 value=Path(name).read_bytes()
 (root/'source'/name).write_bytes(value)
 report['supplemental_source_sha256'][name]=hashlib.sha256(value).hexdigest()
summary.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
p=Path('outputs/status.json')
status=json.loads(p.read_text())
status.update(phase='complete',step='A1 네 연간 평가·원본 및 체크포인트 독립 검산·전체 낮/밤 추가 검산 완료',updated_at=report['verified_at'],verification_paths=[str(root/'verification.json'),str(root/'supplemental_verification.json')])
p.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print('A1 aggregate status finalized: complete')
PY
