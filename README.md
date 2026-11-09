# dataen — 태양광 발전량 예측

발전소 253개의 예측을 두 팀으로 합쳐 **다음 날 시간별 발전량**을 예측하는 프로젝트입니다. [대회 안내](https://dataen.ai/en/challenge/2026).

## 무엇을 보면 되나요?

| 알고 싶은 것 | 문서 |
|---|---|
| 지금 어디까지 했는지 | [현재 상태](CONTEXT.md) |
| 다음에 무엇을 할지 | [작업 계획](EXPERIMENT_PLAN.md) |
| 무엇을 실험했고 결과가 어땠는지 | [실험 이력](EXPERIMENTS.md) |
| UM 요청·시간 정렬·모델·재현 코드 | [UM 팀원 인계](docs/experiments/parallel/2026-11-09-um-handoff.md) |
| P4·G315·F3가 무슨 뜻인지 | [용어와 번호 설명](docs/GLOSSARY.md) |
| 코드·데이터·결과가 어디 있는지 | [폴더 안내](docs/PROJECT_STRUCTURE.md) |
| Codex가 지킬 작업 규칙 | [AGENTS.md](AGENTS.md) |

진행 상황은 CONTEXT에서만 관리합니다. 과거 실험 설정을 현재 실행 기준으로 혼동하지 않도록 이 문서에는 반복하지 않습니다.

## 새 기기에서 처음 시작하기

1. 저장소를 받습니다. 다른 기기로 옮길 때는 `git status`와 원격 변경을 확인하고, 해당 기기의 미커밋 작업을 보존한 뒤 동기화합니다.

   ```bash
   git clone https://github.com/junyoung-code/dataen.git
   cd dataen
   ```

2. 대회 제공 `01_participant_dataset`의 **내용 전체**를 `data/`에 둡니다. `baseline.py`, `sites.csv`, `smp.csv`, `train/`, `test/` 등이 필요하며 원본은 수정하지 않습니다. 데이터·결과·가상환경은 Git으로 전달되지 않습니다.
3. 새 Apple Silicon Mac에서는 Homebrew 준비 후 아래 설치를 실행합니다. Python 3.12·OpenMP·기본 패키지 설치와 설치/데이터 검사를 수행합니다. 이미 작동하는 기기 환경은 그대로 사용합니다.

   ```bash
   bash scripts/setup.sh
   ```

기존 맥미니 환경은 `.venv`와 `.runtime`이 함께 필요합니다. 가상환경을 다른 기기로 복사하지 않습니다. Linux·Intel Mac 설치는 검증하지 않았습니다.

## 기본 확인 명령

아래 `python`은 **해당 작업의 가상환경 Python**입니다. 새로 위 설치를 했다면 `.venv/bin/python`을 사용합니다.

```bash
python manage.py --help                    # 사용 가능한 실행 명령
python manage.py train_smoke --self-test   # 작은 합성 자료로 설치 검사
python manage.py train_smoke --check-only  # 원본 데이터 검사
python manage.py test -v                   # 코드 테스트
```

시험 학습이 필요할 때는 `bash scripts/run_smoke.sh`를 사용합니다. 결과는 `outputs/runs/smoke_<날짜·시간·고유값>/`에 저장됩니다. **50트리 시험의 학습 오차는 검증 점수가 아닙니다.** 이미 완료한 학습을 문서 확인 목적으로 다시 실행하지 않습니다.

시간순 검증·모델 비교는 [실험 이력](EXPERIMENTS.md)의 해당 보고서를 먼저 확인합니다. 긴 학습은 해당 기기의 tmux에서 실행하며, 잠자기·종료·재부팅은 계산을 중단할 수 있습니다.

## 오류가 나면

- 데이터 문제: `data/`가 이중 폴더가 아닌지, `baseline.py`를 포함한 제공 파일이 있는지 확인합니다.
- 학습 실패: 해당 실행의 `run.log`·`summary.json`·종료 코드를 확인합니다. 강제 종료 시 summary가 없을 수 있습니다.
- 환경 문제: 사용 중인 Python과 OpenMP 경로를 먼저 확인합니다. [환경 검증 이력](docs/operations/VERIFICATION.md)을 참고하고 기존 환경을 임의로 재설치하지 않습니다.

과거 설치·SSH 설정·시험 출력의 상세 설명은 [정리 전 README](docs/archive/handoff_20261107/README.md)에 보관했습니다. 당시 상태를 현재 상태로 간주하지 않습니다.
