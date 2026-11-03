# XGBoost 1차 비교 설계

2026-11-03 KST, 결과 확인 전 작성. 사용자가 같은 조건의 비교와 LightGBM의 누적 튜닝 우위를 함께 확인하도록 요청했다.

현재 비교는 **같은 데이터·입력·평가 조건에서 확보한 설정들의 성능 비교**다. LightGBM은 P1~P4에서 트리·학습률·혼합·입력·손실 등을 선택했고, CatBoost는 세 설정만 탐색했다. 이 차이는 남아 있다. 입력·손실·혼합도 LightGBM을 통해 선택했으므로 모델별 최적 설정이나 동등한 탐색 예산의 순위로 해석하지 않는다.

| XGBoost 후보 | 트리 수 | 최대 깊이 | 학습률 |
|---|---:|---:|---:|
| xgb_d6_n400 | 400 | 6 | 0.03 |
| xgb_d6_n1000 | 1000 | 6 | 0.03 |
| xgb_d8_n1000 | 1000 | 8 | 0.03 |

- XGBoost 3.4.1, `reg:absoluteerror`(L1), CPU `hist`, 8스레드, seed0, 행/열 표본비 각각0.8, min_child_weight1, L2정규화1, max_bin256, depthwise. 조기 종료 없음. CatBoost와 같은 깊이·트리 후보 수지만 구조·연산량은 동일하지 않다.
- 기존 26개 수치 특징, 정규화 발전량 목표, 낮 시간 학습, 용량 가중 평균1, 야간0, 10시/17시 예측25:75, 기존 A→G1/B→G2 고정. 미래 발전량·실측 기상·새 ID 입력 없음.
- [p0-v1](validation-protocol.md) 분할·공통 관측 시각·팀 가중 MAE 유지. 1~2월에서 세 후보 중 최저 오차 설정을 선택하고 고정한 후 계절 진단·3~5월에 적용한다. 정확한 동률은 위 순서. 이 기간들은 이미 본 역사적 평가 기간이다.
- 비교 기준은 P4 LightGBM 200트리·0.03·L1+용량과 기존 선택 CatBoost 깊이6·400트리·0.03·MAE+Gradient. 이번에 두 모델을 다시 튜닝하지 않는다.
- 맥 미니 SSH 인증 실패로 현재 맥북 M1 Pro/16GiB에서 실행한다. 세 모델을 같은 맥북·8스레드에서 순차 학습해 정확도·학습 비용을 비교한다. 과거 맥 미니 실행 시간과 직접 비교하지 않는다.
- 모델·원본 예측은 새 outputs 폴더, 코드 버전·실제 파라미터·점수·시간·메모리·기존 결과와의 차이를 기록한다. 표본 가중치·저장 재로딩·원본 실측·마스크·팀 지표·선택을 검사한다.

설치: 프로젝트 환경에서 `python -m pip install -r requirements-xgboost.txt`. 실행: `python run_xgboost.py --threads 8 --output-dir outputs/xgboost_<새ID>`.

공식 문서 확인(2026-11-03): [L1·hist·표본/열 비율](https://xgboost.readthedocs.io/en/stable/parameter.html), [표본 가중치·모델 저장](https://xgboost.readthedocs.io/en/stable/python/python_api.html), [Apple Silicon CPU 설치](https://xgboost.readthedocs.io/en/stable/install.html), [설치 버전](https://pypi.org/project/xgboost/3.4.1/).

실행 완료: [세 모델 결과·검증](2026-11-03-xgboost.md). 코드 `1fbf67e`, 실행 `xgboost_20261103_150530`.
