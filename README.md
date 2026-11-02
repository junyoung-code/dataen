# dataen — 태양광 발전량 예측

대회 제공 데이터를 이용해 LightGBM을 학습하고 발전소별 예측을 두 팀의 발전량으로 합산하는 프로젝트입니다.
현재 단계는 **Mac mini CPU에서 모델당 50개 트리로 환경과 실행 시간을 확인하는 시험**입니다.

- 대회: https://dataen.ai/en/challenge/2026
- 현재 결정과 진행 상태: [CONTEXT.md](CONTEXT.md)
- Codex 작업 지침: [AGENTS.md](AGENTS.md)
- 실험 기록: [EXPERIMENTS.md](EXPERIMENTS.md)
- 초기 실행 검증: [VERIFICATION.md](VERIFICATION.md)

## 다른 Mac에서 시작하기

### 1. 저장소 받기

```bash
git clone https://github.com/junyoung-code/dataen.git
cd dataen
```

기존 작업 폴더에서는 미커밋 변경을 확인하고 정리한 뒤 `git pull --ff-only`로 업데이트합니다.
GitHub에는 코드와 문서만 있습니다. 데이터, 학습 결과, 가상환경, AirDrop ZIP은 포함하지 않습니다.

### 2. 데이터 별도 배치

대회에서 받은 `01_participant_dataset` 폴더의 **내용 전체**를 이 프로젝트의 `data/`에 복사합니다.
이미 AirDrop 패키지가 있다면 그 안의 `oibc_mac_mini/data/` 내용 전체를 복사해도 됩니다.
폴더가 이중으로 중첩되지 않도록 아래 구조를 확인합니다.

```text
data/
├── README.md
├── baseline.py
├── sites.csv
├── smp.csv
├── train/
│   ├── generation.parquet
│   ├── forecast_wa.parquet
│   ├── forecast_wb.parquet
│   ├── observed_wa.parquet
│   ├── observed_wb.parquet
│   └── smp.csv
└── test/
    ├── forecast_wa.parquet
    └── forecast_wb.parquet
```

`baseline.py`도 필요한 파일입니다. 학습 코드가 이 파일의 특징 생성 함수를 재사용합니다.
원본 데이터는 변경하지 않습니다. `data-manifest.json`으로 제공 파일 12개의 해시와 행 수를 검사합니다.
기존 맥북의 `data/`는 로컬 패키지로 연결된 링크일 수 있습니다. 다른 기기에서는 독립된 폴더로 배치하면 됩니다.

### 3. 최초 환경 설치

설치 스크립트는 Apple Silicon macOS용입니다. [Homebrew](https://brew.sh)가 없다면 공식 안내로 먼저 설치합니다.

```bash
bash setup.sh
```

Python 3.12와 OpenMP(libomp), 프로젝트 전용 `.venv`, `requirements.txt`의 패키지를 설치합니다.
작은 CPU 학습·모델 저장·재로딩과 데이터 무결성 검사까지 수행합니다.
설치에는 인터넷이 필요합니다. `.venv`는 기기마다 새로 만들며 설치 버전 목록은 로컬 `environment-lock.txt`에 저장됩니다.
Linux·Intel Mac용 설치는 아직 준비·검증하지 않았습니다.

### 4. 시험 학습 실행

```bash
bash run_smoke.sh
```

가상환경을 따로 활성화할 필요는 없습니다. 각 모델의 트리 10개마다 로그가 나옵니다.
특징 연결 중에는 트리 로그가 잠시 없을 수 있습니다. `Ctrl+C`로 중단할 수 있습니다.
실행 중 터미널을 닫거나 Mac을 수동 잠자기·종료하지 않습니다. 스크립트는 자동 유휴 잠자기를 억제합니다.

## 학습 방식

- 제공 기준선과 같은 24개 특징: 기상 예보, 위치, 설비용량, 시간, 태양 위치 등.
- 2023-06-01 01:00 KST 이후의 최근 2년, 낮 시간 자료 사용. 기본 실행은 임의의 작은 표본만 사용하는 방식이 아닙니다.
- 10시·17시 예보용 모델 각각 트리 50개, 두 모델 합계 최대 100개.
- 발전소별 설비이용률 예측을 kWh로 환산하고 두 모델을 평균 낸 뒤 팀별로 합산. 밤은 0.
- 참조 배정 A→G1, B→G2 사용. 팀 배정 최적화는 아직 하지 않습니다.
- 관측 기상 파일은 데이터에 포함되지만 현재 학습 입력으로 사용하지 않습니다.

이번 실행에는 시간순 검증 분할이 없습니다. **학습 오차는 대회 점수나 미래 예측 정확도가 아닙니다.**
트리 수나 모델을 비교하기 전에 검증 기간과 지표를 추가할 예정입니다.

## 결과 확인

매 실행마다 새로운 `outputs/smoke_날짜_시간_고유값/` 폴더가 생깁니다.

| 파일 | 내용 |
|---|---|
| `run.log` | 실행 기록 |
| `summary.json` | 성공/실패, 모델별 학습 시간·트리 수, 학습 오차, 최대 메모리, 버전 |
| `model_round_10.txt`, `model_round_17.txt` | 학습한 모델 |
| `composition.csv` | 발전소 253개의 팀 배정 |
| `forecast.csv` | 팀 2개 × 8,760시간 = 17,520행, 발전량 kWh |
| `site_forecasts.parquet` | 발전소별 시간별 예측 |

행 수는 CSV 헤더 제외입니다. 예측 시각, 중복·누락, 팀별 20MW 이상 용량, 음수·비유한 예측값을 검사합니다.
자동 제출 기능은 없습니다. `summary.json`의 `status=complete`가 정상 완료입니다.
`fit_seconds`는 모델 학습 시간, `total_seconds`는 처리·예측·저장을 포함한 시간,
`peak_process_memory_mib`는 Python 프로세스의 최대 메모리입니다.
중단 시 처음부터 다시 실행합니다. 체크포인트 재개 기능은 없습니다.

## 두 기기에서 Codex와 이어서 작업하기

1. 작업 시작 전 Git 상태를 확인하고 최신 변경 사항을 받습니다.
2. Codex에게 `AGENTS.md`, `CONTEXT.md`, `EXPERIMENTS.md`를 읽고 현재 상태를 파악하도록 요청합니다.
3. 맥북에서 개발하고 맥 미니에서 학습하는 흐름을 기본으로 합니다. 같은 파일을 동시에 수정할 때는 작업별 브랜치를 사용합니다.
4. 작업 종료 시 현재 상태와 다음 할 일을 `CONTEXT.md`에, 실험은 `EXPERIMENTS.md`에 기록합니다.
5. 코드와 문서를 commit/push하고 다른 기기에서 pull합니다. 수정 중인 파일을 덮어쓰거나 force push하지 않습니다.

GitHub는 대화나 실행 프로세스를 자동으로 동기화하지 않습니다. 모델과 예측 파일은 실행한 기기의 `outputs/`에 남습니다.
실험 직전에 `git rev-parse HEAD`와 `git status --short`를 확인해 실험 기록에 남기면 코드와 결과를 연결할 수 있습니다.

원격 작업은 Tailscale + macOS 원격 로그인(SSH) + tmux 구성을 추천한 상태이며 아직 설정하지 않았습니다.
원격 연결을 설정한 뒤에는 맥 미니에서 `tmux new -s training` 안에서 실행하고,
재접속 시 `tmux attach -t training`으로 같은 작업창을 볼 수 있습니다. 전원 종료 후 복구와는 다릅니다.

## 문제 해결

- 데이터 누락: 위 폴더 구조와 `data/baseline.py`를 확인하고 원본을 다시 복사합니다.
- ARM Python 오류: Rosetta가 아닌 기본 터미널에서 Apple Silicon Python 3.12로 설치합니다.
- libomp 오류: `brew reinstall libomp` 후 `bash setup.sh`를 다시 실행합니다.
- 학습 실패: 해당 실행 폴더의 `run.log` 및 `summary.json` 확인. 강제 종료에서는 summary가 없을 수 있습니다.
- 다른 경로로 이동: `.venv`는 절대 경로를 포함할 수 있으므로 이동한 위치에서 환경을 새로 설치합니다.

데이터·모델·ZIP과 계정 비밀값은 저장소에 올리지 않습니다. `.gitignore`는 이미 Git으로 추적 중인 파일을 자동 제거하지는 않습니다.
