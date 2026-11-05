#!/bin/zsh
set -u
cd "${0:A:h}"
A6_PY=../oibc_mac_mini/.runtime/neural-venv/bin/python
A6_LIB="$PWD/../oibc_mac_mini/.venv/lib/python3.12/site-packages/lightgbm/lib"
A6_RUN=outputs/a6_g2_diagnosis_20261105_01
status_update() {
 "$A6_PY" - "$1" "$2" <<'PY'
import datetime,json,sys
from pathlib import Path
v=dict(phase=sys.argv[1],step=sys.argv[2],device='Mac mini M6 CPU; existing predictions',result_path='outputs/a6_g2_diagnosis_20261105_01',log_paths=['outputs/a6_g2_diagnosis_20261105_01.log','outputs/a6_g2_diagnosis_20261105_01.verify.log'],updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat())
Path('outputs/status.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
PY
}
status_update preparing 'A6 G2 원인 기여분해·학습지원 보완3정책 확인'
env DYLD_LIBRARY_PATH="$A6_LIB" MPLCONFIGDIR="$PWD/outputs/preflight/mpl" "$A6_PY" -u run_g2_correction.py --output-dir "$A6_RUN" --a5-dir outputs/a5_validation_20261105_01 --a1-dir outputs/a1_20261104_01 --cache-dir outputs/annual_cache_20261104_01 > "$A6_RUN.log" 2>&1
result=$?
printf '%s\n' "$result" > "$A6_RUN.compute.exit"
if (( result != 0 )); then status_update blocked 'A6 계산 실패'; exit "$result"; fi
status_update verifying '원본 학습지원·기여항등식·보완정책/W·재표집 독립검산'
env DYLD_LIBRARY_PATH="$A6_LIB" MPLCONFIGDIR="$PWD/outputs/preflight/mpl" "$A6_PY" -u verify_g2_correction.py --output-dir "$A6_RUN" --a5-dir outputs/a5_validation_20261105_01 --a1-dir outputs/a1_20261104_01 --cache-dir outputs/annual_cache_20261104_01 > "$A6_RUN.verify.log" 2>&1
result=$?
printf '%s\n' "$result" > "$A6_RUN.verify.exit"
if (( result != 0 )); then status_update blocked 'A6 독립 검산 실패'; exit "$result"; fi
"$A6_PY" - <<'PY'
import datetime,hashlib,json
from pathlib import Path
folder=Path('outputs/a6_g2_diagnosis_20261105_01');p=folder/'summary.json'
assert json.loads((folder/'verification.json').read_text())['status']=='complete'
raw=p.read_bytes();(folder/'summary.computed.json').write_bytes(raw)
r=json.loads(raw);r.update(status='complete',verified_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),verification_sha256=hashlib.sha256((folder/'verification.json').read_bytes()).hexdigest())
p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
PY
status_update complete 'A6 원인분석·개선확인 및 독립검산 완료'
