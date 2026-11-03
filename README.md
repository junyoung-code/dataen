# dataen — 태양광 발전량 예측

대회 제공 데이터를 이용해 LightGBM을 학습하고 발전소별 예측을 두 팀의 발전량으로 합산하는 프로젝트입니다.
현재는 **Mac mini에서 P0~P5의 계획한 1차 실험과 팀 배정 고정 후 역사적 확인을 완료**했습니다. P5 새 배정은 1~2월에는 개선했으나 3~5월 확인에서 0.35% 악화해 채택 보류했습니다. 현재 작업 기준은 P4 모델 후보(거리 포함 26특징·200트리·0.03·25:75·L1+용량 가중)와 기존 참조 배정입니다. CatBoost 1차 비교도 완료했으며 1~2월·3~5월 팀 오차가 악화해 LightGBM을 유지합니다. 최종 전체 학습·공식 제출은 아직 하지 않았습니다.

- 대회: https://dataen.ai/en/challenge/2026
- 현재 결정과 진행 상태: [CONTEXT.md](CONTEXT.md)
- Codex 작업 지침: [AGENTS.md](AGENTS.md)
- 실험 기록: [EXPERIMENTS.md](EXPERIMENTS.md)
- 확정 평가 기준: [P0 프로토콜](docs/experiments/validation-protocol.md) — 설정 비교·확인 기간, 공통 관측 시각, 가중 오차
- 다음 실험과 세션 인수인계: [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)
- 다른 모델 비교 계획: [MODEL_COMPARISON_PLAN.md](MODEL_COMPARISON_PLAN.md) — CatBoost 1차 완료, XGBoost 미실행
- 실행·환경 검증: [VERIFICATION.md](VERIFICATION.md)
- CatBoost 비교 결과: [2026-11-03 결과](docs/experiments/2026-11-03-catboost.md) — 세 설정 비교·역사적 확인 완료, 교체 보류
- P5 팀 배정 결과: [2026-11-03 결과](docs/experiments/2026-11-03-p5-assignment.md) — tune 개선·확인 악화로 새 배정 채택 보류
- P4 학습 방식 결과: [2026-11-03 결과](docs/experiments/2026-11-03-p4-objectives.md) — L1+용량 가중 선택, 계절 악화와 검증 기록
- P3 특징 비교 결과: [2026-11-03 결과](docs/experiments/2026-11-03-p3-features.md) — 거리 2개 추가 선택, 실제 실행·로그 확인 방법
- P2 혼합 비율 결과: [2026-11-03 결과](docs/experiments/2026-11-03-p2-blend.md) — 25:75 선택, 개선 폭과 월별 한계
- P1 설정 비교 결과: [2026-11-03 결과](docs/experiments/2026-11-03-p1-grid.md) — 실행 명령, 전체 결과, 한계와 다음 작업
- 시간순 검증 보고서: [2026-11-02 결과](docs/experiments/2026-11-02-temporal-evaluation.md)

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

### 현재 맥 미니의 환경 (2026-11-02)

기존 작업 폴더의 `.venv`는 바로 사용할 수 있습니다. Python 3.12.16 / arm64,
LightGBM 4.6.0이며, Homebrew가 없어 uv로 Python을 `.runtime/python/`에 설치했습니다.
공식 Homebrew OpenMP 23.1.4 ARM 배포 파일을 검증한 뒤 가상환경 내부에 배치하고
LightGBM에 로컬 라이브러리 검색 경로와 ad-hoc 서명을 적용했습니다.
이 `.venv`는 `.runtime`에 의존하므로 둘 다 유지해야 합니다. 다른 기기에는 가상환경을
복사하지 말고 위 최초 설치 절차를 따릅니다. 두 폴더는 Git에서 제외합니다.

2026-11-03에는 CatBoost 1.2.10을 추가해 설치·데이터 검사와 비교를 완료했습니다. 새 환경에서 CatBoost를 사용할 때는 `python -m pip install -r requirements-catboost.txt`로 설치합니다. 이번 비교는 기존 폴더 옆 `catboost_compare_199bada/`에서 실행했으며, 코드·결과 문서는 원격 main push가 거절되어 아직 GitHub에 반영되지 않았습니다. 자세한 경로와 재실행 명령은 CatBoost 결과 보고서에 있습니다.

### 4. 시험 학습 실행

```bash
bash run_smoke.sh
```

가상환경을 따로 활성화할 필요는 없습니다. 각 모델의 트리 10개마다 로그가 나옵니다.
특징 연결 중에는 트리 로그가 잠시 없을 수 있습니다. `Ctrl+C`로 중단할 수 있습니다.
실행 중 터미널을 닫거나 Mac을 수동 잠자기·종료하지 않습니다. 스크립트는 자동 유휴 잠자기를 억제합니다.

## 기존 시험 학습 방식

- 제공 기준선과 같은 24개 특징: 기상 예보, 위치, 설비용량, 시간, 태양 위치 등.
- 2023-06-01 01:00 KST 이후의 최근 2년, 낮 시간 자료 사용. 기본 실행은 임의의 작은 표본만 사용하는 방식이 아닙니다.
- 10시·17시 예보용 모델 각각 트리 50개, 두 모델 합계 최대 100개.
- 발전소별 설비이용률 예측을 kWh로 환산하고 두 모델을 평균 낸 뒤 팀별로 합산. 밤은 0.
- 참조 배정 A→G1, B→G2 사용. 팀 배정 최적화는 아직 하지 않습니다.
- 관측 기상 파일은 데이터에 포함되지만 현재 학습 입력으로 사용하지 않습니다.

`run_smoke.sh` 실행에는 검증 분할이 없습니다. **학습 오차는 대회 점수나 미래 예측 정확도가 아닙니다.**
미래 구간 성능은 아래 별도 실행기로 확인합니다.

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

## 시간순 검증

```bash
# 결과 덮어쓰기를 막기 위해 매번 새 출력 폴더 사용
.venv/bin/python evaluate_temporal.py --output-dir outputs/temporal_new
.venv/bin/python -m unittest discover -v
```

- 학습: 2023-06-01 01:00 ~ 2024-05-31 00:00 KST.
- 24시간 구간을 비운 뒤 검증: 2024-06-01 01:00 ~ 2025-06-01 00:00 KST.
- 두 모델 각각 50트리로 고정. 검증 기간으로 튜닝·조기 종료·재학습하지 않습니다.
- 발전소별 지표는 관측된 원본 실측값, 팀별 주요 지표는 모든 소속 발전소의 실측이 있는 시간으로 계산합니다. 결측은 0으로 채우지 않습니다.
- 과거 데이터만으로 구한 월·시간대 평균 예측과 비교하며 낮 시간 지표를 별도 표시합니다.

현재 평균 모델의 낮 시간 발전소 MAE는 12.972kWh, 팀별 MAE는 425.246kWh입니다.
팀별 평가 범위는 전체 팀×시간의 66.16%이므로 공식 대회 점수와 직접 비교하지 않습니다.
상세 수치와 한계는 [검증 보고서](docs/experiments/2026-11-02-temporal-evaluation.md)를 참조하세요.

검증 실행기는 `summary.json`, `monthly_metrics.csv`, `site_predictions.parquet`,
`group_predictions.csv`, 모델 파일 두 개를 저장합니다. `group_predictions.csv`의
`complete=false` 행은 관측된 일부 발전소의 합계이며 전체 팀 지표로 사용하지 않습니다.
`REPORT.md`와 독립 검증용 `verification.json`은 이번 실행 뒤 별도로 작성한 기록으로,
실행기가 자동 생성하는 파일은 아닙니다. 원본 출력은 Git에 올리지 않습니다.

## 두 기기에서 Codex와 이어서 작업하기

1. 작업 시작 전 Git 상태를 확인하고 최신 변경 사항을 받습니다.
2. Codex에게 `AGENTS.md`, `CONTEXT.md`, `EXPERIMENT_PLAN.md`, `EXPERIMENTS.md`를 읽고 현재 상태와 다음 실험을 파악하도록 요청합니다.
3. 노트북의 VS Code Remote-SSH로 맥 미니 폴더를 열면 코드·데이터·학습은 맥 미니에 유지됩니다. 두 기기에서 별도 clone을 사용하는 경우에는 commit/push/pull로 맞춥니다. 같은 파일의 동시 수정은 피하거나 작업별 브랜치를 사용합니다.
4. 작업 종료 시 현재 상태와 다음 할 일을 `CONTEXT.md`에, 실험은 `EXPERIMENTS.md`에 기록합니다.
5. 코드와 문서를 commit/push하고 다른 기기에서 pull합니다. 수정 중인 파일을 덮어쓰거나 force push하지 않습니다.

GitHub는 대화나 실행 프로세스를 자동으로 동기화하지 않습니다. 모델과 예측 파일은 실행한 기기의 `outputs/`에 남습니다.
실험 직전에 `git rev-parse HEAD`와 `git status --short`를 확인해 실험 기록에 남기면 코드와 결과를 연결할 수 있습니다.

맥 미니의 Tailscale, 본인 계정만 허용한 macOS 원격 로그인, 노트북 공개키 인증,
로그인 시 Tailscale 실행, tmux 3.8 설치를 완료했습니다. 사용자가 외부망 SSH 접속 성공을 확인했습니다.
접속 주소·공개키·계정 인증 정보는 이 공개 저장소에 기록하지 않습니다.

긴 학습은 맥 미니의 프로젝트 폴더에서 다음처럼 실행합니다.

```bash
tmux new -s training
bash run_smoke.sh
# 검증 실행은 위 evaluate_temporal.py 명령 사용
```

`Ctrl+B` 다음 `D`로 분리하고, 재접속 후 `tmux attach -t training`으로 복귀합니다.
tmux는 현재 맥 미니의 `~/.local/bin`에 설치되어 새 zsh 세션의 PATH에 포함됩니다.
설치만으로 모든 학습이 tmux에서 실행되는 것은 아닙니다.

현재 자동 본체 잠자기는 꺼져 있고, 비활성 화면 꺼짐 설정은 10분입니다.
화면 꺼짐·화면 잠금 상태의 SSH 응답 및 분리된 tmux 세션의 작업 유지를 확인했습니다.
수동 잠자기·로그아웃·재시작·종료는 접속이나 계산을 중단할 수 있습니다.
Tailscale의 로그인 시 실행 등록은 재부팅 직후 로그인 전 접속을 보장하지 않습니다.
재부팅 후 무인 복구와 학습 체크포인트 재개는 구성하지 않았습니다.

## 문제 해결

- 데이터 누락: 위 폴더 구조와 `data/baseline.py`를 확인하고 원본을 다시 복사합니다.
- ARM Python 오류: Rosetta가 아닌 기본 터미널에서 Apple Silicon Python 3.12로 설치합니다.
- libomp 오류: `brew reinstall libomp` 후 `bash setup.sh`를 다시 실행합니다.
- 학습 실패: 해당 실행 폴더의 `run.log` 및 `summary.json` 확인. 강제 종료에서는 summary가 없을 수 있습니다.
- 다른 경로로 이동: `.venv`는 절대 경로를 포함할 수 있으므로 이동한 위치에서 환경을 새로 설치합니다.

데이터·모델·ZIP과 계정 비밀값은 저장소에 올리지 않습니다. `.gitignore`는 이미 Git으로 추적 중인 파일을 자동 제거하지는 않습니다.
