# B 저장소 폴더 안내

2026-11-05 KST. 이 안내의 기준 위치는 B 전용 저장소(`outputs/ab_20261104/b`)다. 상위 dataen 프로젝트와 A 작업 폴더는 이번 정리 대상에 포함하지 않았다.

## 먼저 볼 파일

| 목적 | 위치 |
|---|---|
| 왜 했고, 어떻게 했고, 어떤 결과인지 | [B 과정 종합 보고서](reports/b-process-results.md) |
| 연간·월별·팀별 수치를 프로그램으로 읽기 | [B 집계 JSON](experiments/b/2026-11-04-b-lane-results.json) |
| GEFS 출처·시간 처리·실험 상세 | [B1 상세](experiments/b/2026-11-04-b1-gefs.md) |
| 사전학습 모델 조건·라이선스·한계 | [B2/B3 상세](experiments/b/2026-11-04-b23-foundation.md) |
| 현재 상태·다음 작업 | [CONTEXT](../CONTEXT.md), [계획](../EXPERIMENT_PLAN.md) |

## 실제 폴더 구조

```text
B 저장소/
├── README.md / AGENTS.md / CONTEXT.md / EXPERIMENT*.md
├── docs/
│   ├── reports/                 # 읽기 쉬운 종합 결과
│   ├── experiments/b/           # B 상세 보고서와 공유 집계
│   ├── experiments/             # 기존 P·모델 비교·공통 평가 문서
│   └── superpowers/             # 기존 설계 기록
├── configs/
│   └── b-artifact-layout.json   # 이전 출력 경로 → 실제 보관 경로
├── scripts/
│   └── check_b_artifacts.py     # 코드·집계·이동 결과 검사 (재학습 없음)
├── outputs/                    # 아래 전체는 로컬 전용, Git 제외
│   ├── runs/
│   │   ├── b1_gefs/            # 학습 모델·예측·검산·실행 소스 사본
│   │   ├── b2_tabpfn/          # 문맥·예측·검산·실행 소스 사본
│   │   └── b3_tabicl/
│   ├── datasets/gefs/{full,trial}/
│   ├── models/foundation/     # 내려받은 사전학습 가중치
│   ├── research/{gefs,foundation}/  # 출처·환경·실패·시험 기록
│   ├── logs/{b1,b2,b3}/
│   ├── reorganization/        # 이동 전 해시·이동 후 검증 기록
│   ├── status.json
│   └── 이전 이름의 심볼릭 링크들
├── data/                       # 기존 원본 읽기 전용 링크
└── .runtime/                   # 기존 기기 전용 가상환경
```

결과·데이터 캐시·가중치·조사 자료·로그의 실제 파일을 위 폴더로 이동했다. 예를 들어 `outputs/b1_annual`은 이제 `outputs/runs/b1_gefs`를 가리키는 상대 심볼릭 링크다. 복제본이 아니므로 이중으로 디스크를 차지하지 않는다. 기존 실행 코드와 과거 기록이 그대로 읽히도록 이전 이름의 링크는 유지한다. **링크나 이동된 결과를 임의로 삭제하지 않는다.**

Python 실행 코드, 실행 당시 설정·requirements, 공통 평가 계약은 원래 루트 위치와 내용을 보존했다. 실행 당시 SHA-256과 재검산기가 이 경로를 직접 확인하므로, 이번 파일 정리와 학습 코드 재구성을 섞지 않았다. 가상환경은 내부 절대 경로가 있어 이동하지 않았다. `data/` 원본도 이동·수정하지 않았다.

## 코드 역할

| 범주 | 루트 파일 |
|---|---|
| B1 자료 수집·특징 | `collect_gefs_b1.py`, `features_gefs_b1.py`, `b1_gefs_config.json` |
| B1 실행·검산·검사 | `run_gefs_b1.py`, `verify_gefs_b1.py`, `test_gefs_b1.py` |
| B2/B3 공통·설정 | `foundation_b23.py`, `b23_foundation_config.json` |
| B2/B3 실행·검산·검사 | `run_foundation_b23.py`, `verify_foundation_b23.py`, `test_foundation_b23.py` |
| 고정 비교 기준 | `AB_EVALUATION_CONTRACT.json`, `run_peer_w.py`, `features_p3.py`, `training_weights.py`, `train_smoke.py`, `data-manifest.json` |
| 설치 조건 | `requirements-gefs.txt`, `requirements-foundation.txt`와 기존 requirements |

## 정리 상태 확인

저장소 루트에서 실행한다. 학습·다운로드·원본 수정은 하지 않는다.

```bash
# Git으로 공유한 코드·집계·문서 링크 확인
python3 scripts/check_b_artifacts.py

# 이 기기의 기존 결과까지 있는 경우: 이동 전후 모든 파일 해시·별칭 확인
python3 scripts/check_b_artifacts.py --local
```

공유 대상은 코드·문서·설정·집계뿐이다. 원본, 행별 예측, 모델 가중치, 캐시, 로그, 환경은 GitHub에 보내지 않는다. 새 clone에는 `outputs/`의 실제 결과와 호환 링크가 없으며 로컬 검사를 실행하려면 별도로 보관한 결과가 필요하다. 과거 실행기는 기존 B 폴더 배치와 `../../peer_w_20261103_163512_rescore` 기준선에 의존하므로, 새 clone만으로 전체 재실행이 된다고 가정하지 않는다.


정리 후 검증(2026-11-05 KST): 이동한 4,795개 파일의 크기·SHA-256이 이동 전과 모두 같고, 15개 이전 경로가 새 위치로 연결됨을 확인했다. 실행 당시 소스/고정 계약 해시, 연간 6개 결과, 월·팀 집계 84개, 문서 링크를 대조했다. 관련 검사 28개(공통/사전학습 24개 + GEFS 4개)를 다시 실행해 통과했다. 상위 프로젝트의 미커밋 변경은 작업 전과 동일하다. 새 학습이나 모델 재예측은 하지 않았다.
