#!/bin/zsh
set -u
cd "${0:A:h}"
A234_PY=../oibc_mac_mini/.runtime/neural-venv/bin/python
A234_LIB="$PWD/../oibc_mac_mini/.venv/lib/python3.12/site-packages/lightgbm/lib"
A234_RUN=outputs/a234_20261104_02
status_update() {
 "$A234_PY" - "$1" "$2" <<'PY'
import datetime,json,sys
from pathlib import Path
p=Path('outputs/status.json')
v=dict(phase=sys.argv[1],step=sys.argv[2],device='Mac mini M6 CPU; A1 predictions reused',
 remote_result='outputs/a234_20261104_02',log_paths=['outputs/a234_20261104_02.log','outputs/a234_20261104_02.verify.log'],
 updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat())
p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
PY
}
status_update preparing 'A2 원본/입력 검증 및 오류·특징 진단; A3/A4 선택·고정·평가'
env DYLD_LIBRARY_PATH="$A234_LIB" MPLCONFIGDIR="$PWD/outputs/preflight/mpl" "$A234_PY" -u run_annual_blends.py --output-dir "$A234_RUN" --a1-dir outputs/a1_20261104_01 --p4-aggregate-dir outputs/a234_inputs > "$A234_RUN.log" 2>&1
result=$?
printf '%s\n' "$result" > "$A234_RUN.compute.exit"
if (( result != 0 )); then status_update blocked 'A2-A4 계산 실패; 로그 확인'; exit "$result"; fi
status_update verifying '원본 마스크·기준 W·혼합 선택·독립 scalar W 검산'
env DYLD_LIBRARY_PATH="$A234_LIB" MPLCONFIGDIR="$PWD/outputs/preflight/mpl" "$A234_PY" -u verify_annual_blends.py --output-dir "$A234_RUN" --a1-dir outputs/a1_20261104_01 --p4-aggregate-dir outputs/a234_inputs > "$A234_RUN.verify.log" 2>&1
result=$?
printf '%s\n' "$result" > "$A234_RUN.verify.exit"
if (( result != 0 )); then status_update blocked 'A2-A4 독립 검산 실패; 검산 로그 확인'; exit "$result"; fi
"$A234_PY" - <<'PY'
import datetime,hashlib,json
from pathlib import Path
folder=Path('outputs/a234_20261104_02');p=folder/'summary.json'
assert json.loads((folder/'verification.json').read_text())['status']=='complete'
raw=p.read_bytes();(folder/'summary.computed.json').write_bytes(raw)
r=json.loads(raw);assert r['status']=='computed_pending_verification'
r.update(status='complete',verified_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),verification_sha256=hashlib.sha256((folder/'verification.json').read_bytes()).hexdigest())
p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
PY
status_update complete 'A2 진단·A3 고정 혼합·A4 조건별 혼합 및 독립 검산 완료'
