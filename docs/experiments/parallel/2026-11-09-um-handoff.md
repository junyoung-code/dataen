# UM 팀원 인계 — 2026-11-09

이 문서는 실제 실행한 S929+UM의 API·시간 규약·모델·결과와 재현에 필요한 파일을 정리한다. Git에는 코드, 설정, 실험 요약만 저장한다. 기상 원본/Parquet, 대회 데이터, 모델, 예측, 인증키, 팀원 산출물은 포함하지 않는다.

## 1. 요청과 격자

인증키를 뺀 실제 요청 형식:

```text
https://apihub.kma.go.kr/api/typ06/url/um_grib_pt_tmfc.php?group=UMKR&nwp=N512&data=U&varn=4200,6213&tmfc=2024070100&ef=24,32,1&X=199&Y=122&help=1
```

`4200=tdsw` 하향 단파 일사, `6213=tcam` 전체 운량을 함께 요청한다. `tmfc`는 UTC 초기화 시각, `ef=24,32,1`은 +24~32h를 1시간 간격으로 요청한다. 인증은 로컬 `.env`의 `KMA_API_KEY`를 수집기가 읽어 처리한다. 기상청 해당 API 활용신청/승인이 필요하다.

발전소 위치를 쓰지 않고 위도34/36/38 × 경도125/127/129의 고정9점을 정한 뒤 UM 1.5km 격자의 가까운 점에 대응시켰다. 모든 발전소에 같은 시각의 동일18개 값을 제공한다. [고정점 설정](../../../configs/um_fixed_points.json)은 실제 사용 파일의 바이트를 보존한다.

| 위도 | 경도125의 X,Y | 경도127의 X,Y | 경도129의 X,Y |
|---|---|---|---|
| 34 | 199,122 | 320,122 | 441,125 |
| 36 | 201,268 | 318,268 | 436,271 |
| 38 | 203,413 | 317,413 | 430,415 |

11/8~9에 대상일2021-01-19~2025-05-31(1,594일), 2025-06-01~2026-04-01(305일)을 확보했다. 날짜 파일이 있어도 일부 값은 결측이다. 평가305일 중 완전278일, 나머지27일에238값 결측이 있다.

11/9 11:23:21 KST에 `tmfc=2026040100`을 1회 조회해 HTTP200 본문으로 “파일생산 중단으로 인해 2026.03.31 12:00 이후 호출을 할 수 없습니다.”를 받았다. 이는 과거 전체 재조회 금지가 아니라 생산종료 경계다. 현 회차 규칙의 마지막 대상일은4/1, 4/2~5/31은 미확보다. 다른 회차의 장기예보 전체가 없다고 단정하지 않는다. [공식 API 안내](https://apihub.kma.go.kr/apiList.do?seqApi=9)의 UM 2026.3.31 생산종료도11/9 재확인했다. 이번 Git 정리에서는 인증 API를 다시 호출하지 않았다.

## 2. 공개시각과 시간 정렬

**D-1 17KST 이전 실제 공개 증빙은 미확인이다.** D-1 00UTC=09KST는 초기화 시각이다. 11월에 과거 예보를 받은 기록도 당시 공개시각 증명이 아니다. 규정 적격성 최종확인과 파일형식 검산을 구분한다.

`tdsw`는 직전1시간 평균, 단위 W/m²다. 저장 GRIB 검산기록28건 모두 `stepType=avg`, `startStep=endStep-1`이며 +24h는 +23~24h 평균이다. [기상청 국지모델 설명서](https://apihub.kma.go.kr/getAttachFile.do?fileName=nwp_grib_guidance.pdf)의1시간 평균 및 [변수표](https://apihub.kma.go.kr/getAttachFile.do?fileName=um_var_inf.pdf)의 단위와 일치한다. 전 기간 모든 GRIB를 재해독했다는 의미는 아니다. tcam은 운량0~1이며 저장 표본은 순간값이다.

`build_um_features.py`는 `tmef`를 UTC→Asia/Seoul로 변환해 동일 `ts`에 붙인다. 한 시간 이동, 누적 차분, 에너지 단위 변환은 없다. 예: 7/1 00UTC +24h → 7/2 09KST, tdsw 평균구간08~09KST → `ts=09KST`. 발전량 ts가 직전1시간 양이라는 규약과 구간이 맞는다.

학습·F2/F3 검증·평가 모두 D-1 00UTC/+24~32h를 사용했다. 09~17KST 밖은 NaN이다. 이 시간 제한은 점수 확인 전 야간 API/원본 불일치 때문에 정했고, 낮9점×9시간×2변수162값은 원본과 대조했다. 최대차이 일사0.0004626694 W/m², 운량0.00000495782. 부분결측도0으로 채우지 않는다.

## 3. 모델과 실제 seen W

F153/F155 발전소 LightGBM 두 모델만 UM을 추가해 재학습했다. XGB/N530x20/직접 모델은 기존 예측을 재사용했다. 발전소 혼합 F153:XGB:F155:N530=2:1:3:3, 직접비중0.15, 보정shrink0.5를 고정했다. raw18은9점×2변수, PCA10은 변수당5개이며 fold 학습구간에서만 PCA를 적합했다.

| 혼합·보정 후 조건 | F2 seen W | F3 seen W |
|---|---:|---:|
| 제공 S929 | 431449313.563 | 1358831463.806 |
| 같은 마감으로 재학습한 control | 431396292.637 | 1364373273.803 |
| raw18 | 420422428.171 | 1337059829.787 |
| PCA10 | 424168091.019 | 1343918497.779 |
| UM18값 완전 시 raw18, 나머지 S929 | 420817028.213 | 1336816790.260 |

raw18의 control 대비 개선은 F2 2.5438%/F3 2.0019%, 최종 결측 대체의 제공 S929 대비 개선은2.4643%/1.6201%다. W는 낮을수록 좋고 개별 트리 점수가 아니다. F2=2023-06~2024-05, F3=2024-06~2025-05. 새 트리 학습마감은 각 시작 직전5/31 17KST. F2는 미래F3 보정을 쓴 회고적 진단, F3는 F2 보정이다. F1 미실행이고 F2/F3는 이미 본 기간이라 독립 최종시험이 아니다.

Mac mini에서12회 비교학습과 최종F153/F155 학습을 수행했다. 최종 학습마감2025-05-31 17KST, 각236입력/2359146행. 최종5310개 팀×시간은 UM 예측,12210개는 S929 대체다. 제공 S929 구성요소를 재현한 기준이며 원래 제출 forecast와 바이트동일 확인은 못 했다. 사용자가 보고한 서버 결과와 상세 한계는 CONTEXT/EXPERIMENTS를 참고한다.

## 4. 코드와 실행 준비

| 역할 | 코드 |
|---|---|
| 원본 소량 조회 | `scripts/probe_kma_api.py` |
| 지점 수집·공유한도 | `scripts/collect_um_points.py`, `scripts/kma_shared_budget.py` |
| 해시/회차/시간 검산·raw18/PCA 변환 | `scripts/build_um_features.py` |
| S929 control/raw18/PCA 비교 | `scripts/run_s929_um.py`, `scripts/compare_s929_um.py` |
| 결측 대체 검증·최종 학습/조립 | `scripts/build_s929_um_submission.py` |
| 제출 결과 독립검산 | `scripts/audit_s929_submission.py` |

원 실행의 프로젝트 기준은 `f492bdc`+당시 미커밋 코드, 팀원 고정 소스는 `47be97916da88a5a60d910d96219c882853bc8c6`이다. 이번 커밋은 현재 코드/기록을 고정하며 원 실행 당시 코드해시 기록을 대체하지 않는다. 기존 S927/KIM 코드와 관련 검사도 보존한다.

수집 예시(프로젝트 루트, 기존 Python3.12 가상환경, 키와 승인 준비 후 별도 실행폴더):

```bash
python scripts/collect_um_points.py \
  --points configs/um_fixed_points.json \
  --output-dir outputs/runs/um_example/collection \
  --start 2024-07-02 --end 2024-07-02 \
  --lead-start 24 --lead-end 32 \
  --shared-budget outputs/runs/kma_shared_budget/daily_budget.json
```

수집기의 일반 기본값은+16~39h이므로 본 실험 재현에는 **반드시+24~32h를 지정**한다. 현재 다른 수집기가 실행 중이면 같은 장부를 사용하고 중복 실행하지 않는다. 위 명령은 안내이며 이번 Git 작업에서 실행하지 않았다.

학습 실행기는 독립 설치 패키지가 아니다. 팀원 `oibc` 소스/`reproduce/params_S929.json`, 기존 학습 캐시·OOF·직접 예측·기상 입력이 필요하다. `--run-root` 아래 `peer_source/`, `work/{cache,outputs,external}/`, `received/direct/outputs/`, `source.json`, 기기별 `runtime_paths.json`을 준비해야 한다. runtime 설정의 PYTHONPATH/OIBC_DATA_DIR/OIBC_WORK_DIR는 각 기기 경로로 작성하며 저장소에 개인 절대경로를 넣지 않는다. 컨트롤러가 호출할 코드도 실행폴더 `scripts/`에 복사한다. 최종 조립은 추가로 `inputs/um_history.parquet`, `inputs/um_combined.parquet`, 이전 비교를 가리키는 `paths.json` 등이 필요하다. 준비되지 않은 새 clone에서 즉시 전체 재현되는 것으로 안내하지 않는다.

원본 감사 근거는 로컬 `outputs/runs/kma_api_probe_20261108/verification.json`, `s927_kma_20261108/um_core_reference_verification.json`, `um_available_recovery_20261109/boundary_verification.json`, 점수는 `s929_um_compare_20261108/comparison/results.json`과 `s929_um_submission_20261109/validation/results.json`에 있다(모두 `outputs/runs/` 기준). Git에는 이 문서의 요약만 포함한다.

Drive의 `UM_features_for_training_20261109.zip`은 UM Parquet·고정점·결합 도우미 `um_features.py`·무결성 목록이다. 전체 실험 코드/모델/정답이 포함된 묶음이 아니다. 팀원은 Drive 자료를 자기 모델에 붙이거나 이 저장소의 실행기를 필요한 별도 입력과 함께 사용한다.

## 5. 커밋 전 검증

2026-11-09 맥북 기존 Python3.12 neural-venv에서 아래34검사 통과:

```bash
python -m unittest tests.test_kim_remote tests.test_kma_points \
  tests.test_kma_shared_budget tests.test_s929_submission \
  tests.test_s929_submission_format tests.test_s929_um tests.test_um_raw_audit -v
```

실패0. 예외주입 시 잠금파일 ResourceWarning 및 sandbox의 Arrow sysctl 경고가 있었지만 검사 실패는 없었다. 환경 설치·새 학습·API 수집·대회 제출은 수행하지 않았다. 다음 재현 작업은 팀원 환경의 위 입력을 준비하고 공개시각 증빙과 원본 S929 제출파일 일치를 확인하는 것이다.
