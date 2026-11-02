#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "이 설치 스크립트는 Apple Silicon Mac용입니다. Rosetta 터미널은 사용하지 마세요." >&2
  exit 1
fi
if [[ -x /opt/homebrew/bin/brew ]]; then
  BREW=/opt/homebrew/bin/brew
elif command -v brew >/dev/null 2>&1; then
  BREW="$(command -v brew)"
else
  echo "Homebrew가 없습니다. https://brew.sh 의 공식 안내로 설치한 뒤 다시 실행하세요." >&2
  exit 1
fi
"$BREW" install python@3.12 libomp
PYTHON="$("$BREW" --prefix python@3.12)/bin/python3.12"
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -c 'import platform,sys; assert platform.machine()=="arm64" and sys.version_info[:2]==(3,12), "기존 .venv는 ARM Python 3.12가 아닙니다. 새 폴더에서 설치하세요."'
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip freeze > environment-lock.txt
.venv/bin/python train_smoke.py --self-test
.venv/bin/python train_smoke.py --check-only
echo "설치와 데이터 확인 완료. 다음 명령: bash run_smoke.sh"
