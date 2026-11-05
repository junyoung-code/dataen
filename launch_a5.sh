#!/bin/zsh
set -u
cd "${0:A:h}"
A5_PY=../oibc_mac_mini/.runtime/neural-venv/bin/python
A5_LIB="$PWD/../oibc_mac_mini/.venv/lib/python3.12/site-packages/lightgbm/lib"
A5_RUN=outputs/a5_validation_20261105_01
status_update() {
 "$A5_PY" - "$1" "$2" <<'PY'
import datetime,json,sys
from pathlib import Path
r=dict(phase=sys.argv[1],step=sys.argv[2],device='Mac mini M6 CPU; no training',result_path='outputs/a5_validation_20261105_01',log_paths=['outputs/a5_validation_20261105_01.log','outputs/a5_validation_20261105_01.verify.log'],updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat())
Path('outputs/status.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
PY
}
status_update preparing 'A5: 고정 후보7/28일5000회 재표집·관측이력·민감도 진단'
env DYLD_LIBRARY_PATH="$A5_LIB" MPLCONFIGDIR="$PWD/outputs/preflight/mpl" "$A5_PY" -u run_candidate_validation.py --output-dir "$A5_RUN" --source-dir outputs/a234_20261104_02 --a1-dir outputs/a1_20261104_01 --subset-dir outputs/a5_inputs > "$A5_RUN.log" 2>&1
result=$?
printf '%s\n' "$result" > "$A5_RUN.compute.exit"
if (( result != 0 )); then status_update blocked 'A5 계산 실패'; exit "$result"; fi
status_update verifying '원본 관측이력·집합·재표집5000회 독립 재생 검산'
env DYLD_LIBRARY_PATH="$A5_LIB" MPLCONFIGDIR="$PWD/outputs/preflight/mpl" "$A5_PY" -u verify_candidate_validation.py --output-dir "$A5_RUN" --source-dir outputs/a234_20261104_02 --a1-dir outputs/a1_20261104_01 --subset-dir outputs/a5_inputs > "$A5_RUN.verify.log" 2>&1
result=$?
printf '%s\n' "$result" > "$A5_RUN.verify.exit"
if (( result != 0 )); then status_update blocked 'A5 독립 검산 실패'; exit "$result"; fi
"$A5_PY" - <<'PY'
import datetime,hashlib,json
from pathlib import Path
folder=Path('outputs/a5_validation_20261105_01');p=folder/'summary.json'
assert json.loads((folder/'verification.json').read_text())['status']=='complete'
raw=p.read_bytes();(folder/'summary.computed.json').write_bytes(raw)
r=json.loads(raw);r.update(status='complete',verified_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),verification_sha256=hashlib.sha256((folder/'verification.json').read_bytes()).hexdigest())
p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
PY
status_update complete 'A5 고정 후보의 안정성·관측이력·민감도 검사 및 독립 검산 완료'
