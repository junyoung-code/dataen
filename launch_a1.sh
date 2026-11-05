#!/bin/zsh
set -u
cd "${0:A:h}"
A1_PYTHON=../oibc_mac_mini/.runtime/neural-venv/bin/python
A1_LIB="$PWD/../oibc_mac_mini/.venv/lib/python3.12/site-packages/lightgbm/lib"
A1_RUN=outputs/a1_20261104_01
A1_CACHE=outputs/annual_cache_20261104_01
mkdir -p outputs/preflight
status_update() {
  "$A1_PYTHON" - "$1" "$2" <<'PY'
import datetime,json,sys
from pathlib import Path
v=dict(phase=sys.argv[1],device='Mac mini M6 MPS',step=sys.argv[2],
       log_paths=['outputs/a1_20261104_01.log','outputs/a1_20261104_01.verify.log'],
       result_path='outputs/a1_20261104_01',updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat())
p=Path('outputs/status.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
PY
}
status_update training 'TabM10 F2/F3 → TFT20 F2/F3'
env DYLD_LIBRARY_PATH="$A1_LIB" MPLCONFIGDIR="$PWD/outputs/preflight/mpl" "$A1_PYTHON" -u run_annual_neural.py --output-dir "$A1_RUN" --cache-dir "$A1_CACHE" --device mps --threads 8 > "$A1_RUN.log" 2>&1
result=$?
printf '%s\n' "$result" > "$A1_RUN.train.exit"
if (( result != 0 )); then
  status_update blocked '학습 실행 실패; 로그와 train.exit 확인'
  exit "$result"
fi
status_update verifying '원본 라벨·학습 경계·체크포인트·독립 NumPy W 검산'
env DYLD_LIBRARY_PATH="$A1_LIB" MPLCONFIGDIR="$PWD/outputs/preflight/mpl" "$A1_PYTHON" -u verify_annual_neural.py --output-dir "$A1_RUN" --cache-dir "$A1_CACHE" > "$A1_RUN.verify.log" 2>&1
result=$?
printf '%s\n' "$result" > "$A1_RUN.verify.exit"
if (( result == 0 )); then
  status_update complete 'A1 네 연간 평가 및 독립 검증 완료'
else
  status_update blocked '독립 검증 실패; verify.log 확인'
fi
exit "$result"
