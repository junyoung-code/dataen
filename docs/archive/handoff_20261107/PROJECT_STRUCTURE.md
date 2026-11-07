> **과거 안내 보관본 — 현재 실행 지시가 아닙니다.**
> 2026-11-07 정리 전 `docs/PROJECT_STRUCTURE.md`입니다. 본문은 보존하고 상대 링크만 조정했습니다.
> [현재 README](../../../README.md)와 [현재 폴더 안내](../../../docs/PROJECT_STRUCTURE.md)를 우선하세요. 코드·결과 경로는 당시 프로젝트 루트 기준입니다.

# 프로젝트 폴더 안내

2026-11-05 KST. 이 문서는 상위 `dataen` 프로젝트의 현재 구조다. B 저장소 내부 정리와 별개로 전체 코드·테스트·문서·실행 결과를 분류했다.

```text
dataen/
├── README.md                  # 사용법·보고서 입구
├── AGENTS.md                  # 작업 지침
├── CONTEXT.md                 # 현재 상태
├── EXPERIMENT_PLAN.md         # 계획·진행표
├── EXPERIMENTS.md             # 실험 이력
├── manage.py                  # 실행·테스트 공통 입구
├── src/                       # 학습 실행기·특징·평가·공통 함수
├── tests/                     # 단위·경로·명령행 검사
├── scripts/                   # 설치·시험 학습·구조 검사
├── requirements/              # 기본·CatBoost·XGBoost·신경망 의존성
├── configs/                   # 데이터 manifest·경로 이동표
├── docs/
│   ├── experiments/
│   │   ├── evaluation/        # 시간순·연간 W·평가 프로토콜
│   │   ├── p0-p5/             # P0~P5 비교·설계·집계
│   │   ├── models/            # CatBoost·XGBoost·신경망
│   │   └── parallel/          # A/B 공통 계획
│   ├── plans/                 # 모델 비교 계획
│   ├── operations/            # 설치·운영 검증 이력
│   └── superpowers/           # 기존 설계 기록
├── outputs/                   # 로컬 결과 (Git 제외)
│   ├── runs/                  # 최종 유효 실행 결과
│   ├── cache/                 # 재사용 가능한 파생 입력
│   ├── logs/                  # 실행 로그·종료 코드
│   ├── checks/                # 평가·참조·접속 점검 기록
│   ├── environments/          # 환경 설치·검사 기록
│   ├── metadata/              # 실행 장치 등 메타데이터
│   ├── archive/               # 실패·검산 전 결과 (삭제 아님)
│   ├── maintenance/           # 구조 변경 전 사본·해시·검증
│   ├── ab_20261104/           # 실제 A/B 저장소·공통 준비 기록
│   └── 이전 경로의 연결들     # 과거 참조 및 A/B 호환
├── workspaces/                # A/B/control로 들어가는 로컬 연결
├── archive/transfers/         # 기존 ZIP·전달용 데이터 사본
├── data/                      # 기존 원본 링크: 이동·수정 없음
├── private/oibc_mac_mini/     # data 링크가 참조하는 기존 보관 위치
└── .runtime/                  # 기존 기기 전용 환경: 이동 없음
```

## 실행하는 방법

프로젝트 가상환경의 Python을 사용한다. 아래 `python`은 해당 환경의 실행 파일을 뜻한다. 현재 맥북의 `.runtime/neural-venv`를 새로 설치하거나 다른 환경으로 바꾸지는 않았다.

```bash
python manage.py --help
python manage.py train_smoke --self-test
python manage.py train_smoke --check-only
python manage.py run_peer_w --help
python manage.py test -v

# 파일 경로를 직접 지정해도 된다.
python src/train_smoke.py --check-only
python -m unittest discover -s tests -t . -v

# 최초 환경 설치와 시험 학습 (필요할 때만 실행)
bash scripts/setup.sh
bash scripts/run_smoke.sh
```

`manage.py`는 프로젝트 루트를 작업 디렉터리로 사용한다. 그 뒤 전달한 상대 경로는 프로젝트 기준이다. 기존 `python run_p1.py ...` 명령은 `python src/run_p1.py ...` 또는 `python manage.py run_p1 ...`로 바꾼다. 루트에 예전 이름의 Python 파일을 다시 나열하지 않았다. 예전 실험 로그의 명령은 실행 당시 기록이며, 현재 경로 안내를 함께 참고한다.

새 결과는 `--output-dir outputs/runs/<새 실행 이름>`에 저장한다. 기존 완료 모델을 구조 검사 목적으로 다시 학습하지 않는다. 실행기가 요구하는 커밋 상태·출력 폴더 중복 검사 등 기존 보호 조건은 유지했다.

## 결과와 별도 저장소를 찾는 방법

- 유효 P4 연간 기준선: `outputs/runs/peer_w_20261103_163512_rescore/`.
- 검산 전 P4 기록: `outputs/archive/peer_w_20261103_163512/`. 유효 결과로 사용하지 않는다.
- 신경망 결과: `outputs/runs/{tabm,lstm}_20261103_155606/`.
- 신경망 입력 캐시: `outputs/cache/neural_cache_20261103/`.
- B 결과 보고서: 로컬 `workspaces/b/docs/reports/b-process-results.md` 또는 [GitHub 보고서](https://github.com/junyoung-code/dataen/blob/codex/b-gefs-20261104/docs/reports/b-process-results.md).
- A/B 저장소 안내: [workspaces](../../../workspaces/README.md).

A/B는 자체 Git 저장소이며 기존 상대 데이터 경로와 환경을 사용한다. 실제 `outputs/ab_20261104`를 옮기지 않고 `workspaces`에서 찾도록 했다. 다른 기기의 원격 A 실행 상태는 이번 구조 정리에서 확인하거나 변경하지 않았다.

`outputs`의 이전 이름은 새 실제 위치로 향하는 상대 심볼릭 링크다. 파일을 이중 복사한 것이 아니다. 과거 스크립트·결과 문서·A/B가 이 경로를 참조하므로 연결은 유지한다. 링크나 실제 결과를 정리 목적으로 삭제하지 않는다.

## 보존·공유 범위

원본 `data/`와 그 실제 대상, A/B 소스·계약, 가상환경, 기존 실행 소스 사본·예측·모델·점수는 유지했다. ZIP·전달 데이터 사본은 `archive/transfers`로 이동만 했으며 중복이라고 판단해 삭제하지 않았다. 학습식·특징 계산·평가식은 바꾸지 않았다. 코드 수정은 루트/문서/설정 위치와 실행 소스 기록 경로, 새 진입점에 한정한다.

기존 미커밋 내용은 새 경로로 보존했으며 이동 전 사본과 패치는 `outputs/maintenance/structure_20261105/`에 있다. 이번 전체 정리는 로컬 작업이다. 상위 저장소의 기존 미커밋 변경을 일괄 commit/push하지 않았다. Git에서는 정리 전 파일 삭제와 새 경로 추가로 보일 수 있으며, 나중에 변경을 함께 기록하면 이동으로 비교할 수 있다.

이동표: [configs/project-layout.json](../../../configs/project-layout.json). 검사:

```bash
python scripts/check_project_layout.py
python scripts/check_project_layout.py --local
```

기본 검사는 코드·문서·설정 경로를 확인한다. `--local`은 이 기기의 이동 전 해시와 실제 결과/전달 자료, 호환 연결, A/B 루트 파일 보존까지 비교한다. 새 clone에는 로컬 결과·링크·이동 전 사본이 없으므로 기본 검사만 가능하다.

최종 검증: 기존 55개와 경로 검사 4개를 합친 **59개 테스트**, 원본 12파일 무결성, 이동 자료 437개 해시, B 기존 자료 4,795개 보존 검사를 통과했다. 상위 Git index와 A/B 루트 파일도 보존했다.
