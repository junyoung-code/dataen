# 폴더와 실행 파일 안내

2026-11-07 정리. 처음 시작하는 방법은 [README](../README.md), 현재 맥미니 실행 위치는 [CONTEXT](../CONTEXT.md)를 봅니다.

## 파일은 어디에 있나요?

| 위치 | 용도 |
|---|---|
| `src/` | 모델 학습·예측·평가·입력 생성 코드 |
| `scripts/` | 설치, 기상자료 수집, 자동 비교, 구조 검사 |
| `tests/` | 코드 검사 |
| `requirements/` | 기본·모델별 패키지 목록 |
| `configs/` | 원본 데이터 목록/해시와 폴더 이동표 |
| `data/` | 대회 원본. 수정하지 않음. 이 기기에서는 기존 자료로 연결된 링크 |
| `docs/GLOSSARY.md` | [실험 번호·평가 기간·용어 설명](GLOSSARY.md) |
| `docs/experiments/` | 평가 기준, P0~P5, 모델 비교, A/B·혼합 실험의 상세 보고서 |
| `docs/plans/`, `docs/operations/` | 상세 계획, 환경·운영 검증 이력 |
| `docs/archive/` | 정리 전 문서. 과거 근거용이며 현재 작업 지시가 아님 |
| `docs/superpowers/` | 과거 설계 기록 |
| `workspaces/` | 별도 A/B 작업 저장소로 들어가는 로컬 연결. [안내](../workspaces/README.md) |
| `.venv/`, `.runtime/` | 기기별 실행 환경. 다른 기기로 복사하지 않음 |
| `private/oibc_mac_mini/` | 이 기기의 `data/`가 가리키는 기존 원본 보관 위치 |
| `archive/transfers/` | 과거 전달용 ZIP·자료 사본 |

## 결과는 어디에 있나요?

모두 Git 제외 로컬 자료입니다. 원격에서 실행했다면 해당 기기에 있습니다.

| 위치 | 용도 |
|---|---|
| `outputs/runs/<실행명>/` | 실행별 결과·상태·검증. 폴더가 있다고 성공한 것은 아님 |
| `outputs/cache/` | 재사용할 입력·중간 계산 |
| `outputs/logs/` | 실행 로그·종료 코드. 실행에 따라 해당 runs 폴더 안에 저장하기도 함 |
| `outputs/checks/`, `outputs/environments/`, `outputs/metadata/` | 점검·환경·장치 기록 |
| `outputs/archive/` | 실패·검산 전 결과. 유효 결과와 구분 |
| `outputs/maintenance/` | 구조 정리 전 사본·보존 검사 |
| `outputs/ab_20261104/`, `outputs/blend_20261105/` | A/B 및 BL1/BL2의 별도 작업 폴더 |

각 실험의 정확한 결과 경로는 [실험 이력](../EXPERIMENTS.md)과 연결된 보고서를 확인합니다. 이전 경로의 심볼릭 링크는 호환용이므로 임의로 삭제하지 않습니다.

## 코드는 어떻게 실행하나요?

프로젝트 가상환경의 Python으로 `python manage.py <명령>`을 실행합니다. 상대 경로는 프로젝트 루트 기준입니다.

```bash
python manage.py --help
python manage.py train_smoke --check-only
python manage.py test -v
```

`manage.py`에 등록된 명령만 이 방식으로 실행합니다. 직접 경로 실행도 가능합니다(`python src/train_smoke.py --check-only`). 기상 자동 비교는 별도 `scripts/continue_weather_ablation.py`가 담당하며, 현재 작업을 중복 시작하지 않습니다.

새 실행기는 지원하는 경우 `--output-dir outputs/runs/<새 실행명>`으로 결과를 분리합니다. 과거 기록의 `python run_p1.py` 같은 명령은 현재 `python src/run_p1.py` 또는 `python manage.py run_p1` 경로에 해당합니다.

구조·문서 링크 검사: `python scripts/check_project_layout.py`. 이동표는 [project-layout.json](../configs/project-layout.json), 이전 이동·검증 상세는 [보관본](archive/handoff_20261107/PROJECT_STRUCTURE.md)에 있습니다.
