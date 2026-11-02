#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  echo "먼저 bash setup.sh 를 실행하세요." >&2
  exit 1
fi
# 외부에서 가져온 다른 컴퓨터의 가상환경을 실행하지 않도록 확인합니다.
.venv/bin/python -c 'import platform; assert platform.machine()=="arm64", "ARM 환경을 사용하세요."'
.venv/bin/python train_smoke.py --check-only
mkdir -p outputs
RUN_DIR="$(mktemp -d "$ROOT/outputs/smoke_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
echo "결과 폴더: $RUN_DIR"
echo "학습 중 이 터미널을 닫거나 Mac을 수동으로 잠자기/종료하지 마세요. 중단: Ctrl+C"
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i .venv/bin/python -u train_smoke.py --output-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/run.log"
else
  .venv/bin/python -u train_smoke.py --output-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/run.log"
fi
echo "시험 학습 완료: $RUN_DIR"
