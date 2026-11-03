# CatBoost 1차 비교

2026-11-03 KST, 결과 확인 전 작성. 기존 P4 LightGBM과 26개 입력·참조 그룹·L1+용량 가중·25:75 예보 혼합을 맞춘다.

| 후보 | 트리 수 | 깊이 | 학습률 |
|---|---:|---:|---:|
| cat_d6_n400 | 400 | 6 | 0.03 |
| cat_d6_n1000 | 1000 | 6 | 0.03 |
| cat_d8_n1000 | 1000 | 8 | 0.03 |

학습량 증가와 깊이 증가를 먼저 비교하는 작은 탐색이다. CatBoost 1.2.10, CPU 8스레드, seed 0, MAE, 조기 종료 없음. 나머지 실제 적용값도 결과에 저장한다. CatBoost에만 ID를 추가하지 않는다.

설치 검사에서 CPU 1.2.10의 기본 `leaf_estimation_method=Exact`가 가중 중앙값 0.9인 합성 데이터에서 30트리 후 0.133914로 이동하는 현상을 재현했다. 같은 조건의 `Gradient`는 0.899999, 가중 MAE 0.133334로 가중치를 반영했다(bootstrap을 끈 대조 실험). 이번 비교는 **MAE + Gradient**로 실행하며 기본 Exact 결과로 일반화하지 않는다. 회귀 테스트로 이 가중치 동작을 확인한다. 실제 대회 성능 수치는 아직 미측정이다.

1. p0-v1의 1~2월에서 기존 LightGBM과 세 후보를 학습·평가한다. LightGBM은 기존 점수 재현을 검사한다.
2. tune 팀 가중 MAE 최솟값으로 CatBoost 한 후보를 선택한다. 정확한 동률이면 위 순서를 따른다.
3. 선택 설정·참조 배정 해시를 저장한 뒤, 계절 진단과 3~5월에서 선택 후보와 기존 LightGBM을 각각 학습·비교한다. 확인 후 재탐색하지 않는다.

기간·지표는 [p0-v1](validation-protocol.md)을 재사용한다. 3~5월은 이미 결과를 본 역사적 확인 기간이고 계절 공통 시각은 12월 일부 178시간뿐이다. 모델별 월·팀 결과, 시간·메모리, 원본 실측·평가 마스크·저장 재로딩을 확인한다. 공식 제출은 하지 않는다.

실행: `python run_catboost.py --threads 8 --output-dir outputs/catboost_<새실행ID>`.
설치: 프로젝트 가상환경에서 `python -m pip install -r requirements-catboost.txt`.

공식 문서 확인(2026-11-03): [MAE](https://catboost.ai/docs/en/concepts/loss-functions-regression), [표본 가중치](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit), [macOS ARM 설치](https://catboost.ai/docs/en/installation/python-installation-method-pip-install).
