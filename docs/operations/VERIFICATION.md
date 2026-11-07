# 패키지 검증 기록

검증일: 2026-11-02. 현재 작업 컴퓨터의 ARM macOS, Python 3.12.13 임시 가상환경에서 수행했습니다.
사용자의 맥 미니에서 실행한 결과는 아닙니다.

- `setup.sh`, `run_smoke.sh`: Bash 구문 검사 통과.
- `train_smoke.py`: Python 구문 검사 통과.
- `requirements.txt`: 별도 임시 가상환경에 설치 성공.
- CPU/OpenMP: 합성 데이터 학습, 모델 저장·재로딩, 예측 일치 검사 통과.
- 동봉 데이터 12개: 원본과 SHA-256 일치. Parquet 행 수, 발전소 253개 및 평가 시각 8,760개 확인.
- 실제 데이터 일부로 전체 처리 경로 검증: 발전소 253개, 학습 2025-05-25 01:00 ~ 2025-06-01 00:00 KST,
  평가 예보 최초 48시간을 사용. 낮 시간 학습 행렬은 모델별 25,453행 × 24열.
- 10시·17시 모델 모두 실제 생성 트리 50개, 저장 후 재로딩 성공.
- 팀 배정 253행, 팀별 예측 96행, 발전소별 예측 12,144행 확인.
- 발전소별 예측을 합산한 값과 팀별 예측의 일치 확인.
- 팀별 설비용량: G1 20,465.655kW, G2 22,189.715kW로 각각 20MW 이상.

전체 2년 구간 학습과 1년 예측은 여기서 실행하지 않았습니다. Mac mini에서 `bash scripts/run_smoke.sh`로 실행합니다.
Homebrew 설치 스크립트 자체는 현재 컴퓨터의 전역 환경을 변경하지 않기 위해 실행하지 않았습니다.
검증에는 공식 Homebrew OpenMP 파일을 임시 위치에 풀어 사용했습니다.
별도 검증 구간의 정확도나 대회 점수는 아직 측정하지 않았습니다.

## 저장소 구성 후 재확인 — 2026-11-02 KST

- 실행 코드·설치 스크립트·의존성·데이터 manifest가 초기 검증 패키지와 바이트 단위로 같은지 확인했습니다.
- 저장소 루트에서 Bash/Python 구문 검사, CPU 학습·저장·재로딩 검사, 전체 데이터 무결성 검사를 다시 통과했습니다.
- Git에 포함할 파일을 코드·문서 11개로 한정하고 데이터, 기존 패키지, ZIP, outputs, 가상환경 제외를 확인했습니다.
- 작은 실제 데이터의 전체 처리 경로 검사는 위 초기 검증 기록이며, 코드가 같아 이번에는 반복 실행하지 않았습니다.
- 데이터는 GitHub에 포함하지 않습니다. 다른 기기에서 별도로 배치해야 합니다.

## 맥 미니 환경 설정 — 2026-11-02


- macOS 27.0.1 / arm64, 프로젝트 전용 Python 3.12.16.
- `.runtime/python`의 Python으로 `.venv` 생성. Homebrew 전역 설치 없이 구성.
- `requirements.txt`의 5개 패키지 버전 일치, `pip check` 통과.
- OpenMP 23.1.4: 공식 Homebrew `arm64_sequoia` bottle 다운로드 및 SHA-256 검증.
  검증 해시: `af8425bbe39630ec9b2022b67449ac737e0ed6d48e98a6309f841295d3c6f0f2`.
- OpenMP를 `.venv/lib/python3.12/site-packages/lightgbm/lib/`에 배치하고
  `lib_lightgbm.dylib`에 `@loader_path` RPATH 추가 및 ad-hoc 서명 적용.
- `train_smoke.py --self-test`: CPU 학습·저장·재로딩 및 예측 일치 검사 통과.
- `train_smoke.py --check-only`: 데이터 12개, 발전소 253개, 평가 시각 8,760개 검사 통과.
- 별도 검사로 데이터 12개의 파일 크기와 SHA-256 일치 확인.
- `bash -n setup.sh run_smoke.sh` 통과. 이 환경 설치 시점에는 전체 학습 전이었으며, 아래 후속 실행에서 완료함.
- 설치 버전 기록: `environment-lock.txt`. 실행 명령: `bash scripts/run_smoke.sh`.

설치 방식 참고: [uv Python 설치](https://docs.astral.sh/uv/guides/install-python/),
[uv 저장 위치](https://docs.astral.sh/uv/reference/storage/),
[Homebrew OpenMP 배포 정보](https://formulae.brew.sh/api/formula/libomp.json).

## 맥 미니 전체 학습 및 시간순 검증 — 2026-11-02

- 전체 시험 학습 `smoke_20261102_200221_LPmZAS`: 두 모델 각각 50트리, 팀 예측 17,520행, 발전소 예측 2,216,280행. 출력 형식·시간축·결측/중복·음수/비유한값·팀 합계 일치 검사 통과.
- 저장된 모델로 최초 48시간의 발전소 예측 12,144행을 다시 계산해 저장 예측과 일치 확인.
- 시간순 검증 `temporal_20261102_200911_C859iM`: 원본 라벨 유지, 학습 시점 경계, 모든 주요 지표 및 역사적 가중 오차 패널티 독립 재계산 통과.
- 완전 관측 팀×시간 11,592개(66.16%). 2024년 6월은 완전 관측 팀×시간이 없어 팀별 월 지표를 계산하지 않음. 발전소별 월 지표는 12개월 모두 존재.
- 테스트: `.venv/bin/python -m unittest discover -v` — 5개 통과.
- 코드 검토에서 수정이 필요한 오류 없음. 성능 수치·한계는 EXPERIMENTS 및 검증 보고서 참조.
- 원본 결과 파일은 맥 미니 `outputs/`에 보관하며 Git에는 요약 문서만 포함.

## CatBoost 설치·비교 검사 — 2026-11-03

- 맥북의 Python 3.12.13 임시 가상환경과 맥 미니의 Python 3.12.16 프로젝트 `.venv`에 CatBoost 1.2.10 설치. 기존 5개 고정 패키지 버전 유지, 두 환경 `pip check`, `train_smoke.py --self-test`, `--check-only` 통과.
- 기본 Exact의 합성 가중치 테스트가 최초 실패해 원인 재현 후 Gradient로 명시했다. 두 환경 단위 테스트 각각 37개 통과(건너뛰기 없음). 상세 재현과 한계는 [보고서](../experiments/models/2026-11-03-catboost.md)에 기록.
- 맥 미니의 8개 비교 실행·16개 모델 학습 정상 종료. 3기간 기존 LightGBM 점수 재현, 원본 실측·마스크·주/월별 지표·학습 가중치·선택 해시 독립 확인. 저장 모델 각 512행 재예측 일치.
- 실행 `catboost_20261103_144056`, 코드 `199bada`, 결과는 맥 미니 분리 worktree `catboost_compare_199bada/outputs/`에 보관. 기존 폴더의 `outputs/catboost_env_20261103/`에 설치 전후 버전 기록.

## XGBoost·세 모델 비교 검사 — 2026-11-03

- 현재 맥북 M1 Pro의 Python3.12.13 임시 환경에 XGBoost3.4.1만 추가, 기존 버전 유지. pip check·self-test·check-only·원본12개 manifest 검사 및 단위42개 통과(건너뛰기 없음). 맥 미니 SSH 인증 실패로 맥 미니 환경은 변경하지 않았다.
- 최초 caffeinate 실행에서 라이브러리 경로가 전달되지 않아 학습 전에 실패했다. 경로 전달 방식을 재현·수정한 뒤 새 실행 `xgboost_20261103_150530`에서22개 모델 학습을 완료했다. 코드 `1fbf67e`.
- 같은 맥북에서 두 참조 모델을 재학습해 이전 맥 미니 점수 재현. 11개 실행 라벨·마스크·지표·학습 행/가중치·선택 해시 검사와22개 저장 모델 각512행 재예측 일치. 학습·독립 검증 종료0.
- 결과·실패·환경 기록은 현재 맥북 outputs에 보관. 상세한 동일 조건 범위와 누적 튜닝 차이는 [비교 보고서](../experiments/models/2026-11-03-xgboost.md)에 기록했다.


## 2026-11-03 — 신경망 세 모델, 두 기기

- 맥북·맥 미니 별도 환경의 pip check·self-test·check-only·51개 단위 검사 통과. MPS 정방향/역전파 확인 후 TabM·LSTM·TFT 모두 정상 종료0.
- 15개 평가 결과의 원본 라벨·마스크·팀 점수·학습 타깃/가중치·원본에서 재생성한 입력 표본, 30개 저장 체크포인트의 재예측 검증 통과. 소스 해시·선택 파일 고정 확인.
- 검증 스크립트·로그는 각 기기의 `outputs/neural_env_20261103/`, 결과별 `verification.json`은 각 학습 폴더에 있다. [결과·환경·사전 실패 기록](../experiments/models/2026-11-03-neural.md).


## 전체 폴더 정리 검증 — 2026-11-05

- 이동 전 55개, 이동 후 기존 55개와 새 경로 검사 4개를 합친 **59개 테스트 통과**(43.623초, 건너뛰기 없음). 새 진입점으로 13개 명령의 도움말을 다른 작업 디렉터리에서 실행했고, 중첩 경로의 실행 소스 사본 생성도 확인했다.
- CPU 합성 데이터 학습·저장·재로딩 self-test와 원본 12파일 check-only 통과. 전체 성능 실험/모델 재학습은 하지 않았다.
- 코드·문서·설정 67개 이동, 최상위 일반 파일 44→6개. 결과·전달 자료 437개 파일의 크기·SHA-256 보존, 문서 상대 링크 134개와 requirements 연결 확인. B 내부 기존 4,795개 자료도 자체 검사를 통과했다.
- 초기 새 테스트 1개는 Git 응답을 대체하는 코드가 platform의 subprocess 호출까지 가로채 실패했다. Git 명령에만 대체를 적용한 후 해당 검사와 전체 59개가 통과했다. 실제 학습/점수 코드 오류는 아니며 실패 로그도 보존했다.
- 기존 미커밋 작업은 이동 전 사본·패치와 새 위치에 보존했다. Git index, A/B 루트 파일, 원본 data 링크, 가상환경 위치를 유지했다. commit/push 없음.
- 기록: `outputs/maintenance/structure_20261105/{completion.json,layout-check.json,tests-before.log,tests-after.log,tests-final.log,self-test.log,data-check.log}`. 이전 실험 점수·기간·기기는 그대로다.

현재 [구조와 명령](../PROJECT_STRUCTURE.md)을 사용한다.
