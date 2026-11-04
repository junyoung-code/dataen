# B2 / B3 — 로컬 사전학습 회귀 모델 비교

2026-11-04 KST. 사용자의 최신 요청은 B1을 마친 뒤 마지막 단계까지 진행하는 것이다. 이 B 채팅에서는 기존 계획의 B2 TabPFN v2, B3 TabICLv2까지 실제 실행·검산을 이어간다. 초기 LANE_START의 B2/B3 제외보다 최신 지시를 우선한다. B 전용 저장소·맥북만 사용하고 A/부모·원본·공통 평가 계약은 수정하지 않는다. GitHub push·공식 제출은 하지 않는다.

## 결과 전에 고정한 비교 조건

- B1은 [완료](2026-11-04-b1-gefs.md)했고 재학습하지 않는다. B2/B3는 **기존 P4의26입력**만 사용해 모델을 바꾸는 첫 후보를 비교한다. GEFS4입력을 추가한 B1과는 입력 차이도 있으므로 별도 열로 비교한다.
- F2/F3 학습 기간·인과적 학습 끝·평가 키/마스크·기존 팀 배정·야간0·25:75·W는 공통 계약 그대로다. 전체 평가 행을 예측하고 채점한다. 한 번에 MPS 작업 하나만 실행한다.
- 각 분할의 낮 학습 행에서 **용량에 비례하는 확률로8,192행을 중복 없이 선택**한다(seed0). 선택된 위치를 정렬하고 두 예보 라운드와 두 모델에 동일 문맥을 준다. 평가 자료/실측 기상/가격으로 선택하지 않는다.
- **P4는 전체 낮 학습 행을 이용하며 이 후보들은8,192개의 라벨만 문맥으로 쓴다.** 같은 기간이지만 학습 정보량과 방식은 다르다. 사전학습 가중치를 고정한 in-context 추론이며 이 과제에서 가중치를 역전파로 갱신하지 않는다. 용량 비례 샘플링은 P4의 용량 가중L1 최적화와 정확히 같지 않다.
- 결측 입력은 각 라운드의 전체 낮 학습 행만으로 계산한 중앙값(전체 결측 열0)으로 채운다. 라이브러리 전처리도 fit 문맥에만 맞춘다. 예측값은 중앙값(`median`), 음수만0으로 자르며 상한은 두지 않는다.
- 사전학습 회귀 가중치1개, 내부앙상블1개, Torch float32, MPS, CPU4스레드,1,024행 추론 배치. TabPFN은fit_with_cache, TabICL은kv_cache=True/none 정규화/열 순서 유지로 고정한다. 기본8개 앙상블의 최대 성능 탐색은 아니다.
- 합성 입력에서 문맥2,048→8,192의 실행성만 확인하고 연간 점수를 보기 전에 조건을 고정했다. 작은 시험은 연간 성능 결과가 아니다. 결과를 보고 문맥/특징/앙상블/혼합을 다시 선택하지 않는다.
- 연간 예측은 라운드별 memmap와 진행 JSON으로 저장한다. 중단 시 같은 코드/설정/문맥 checksum을 확인한 뒤 마지막 저장 구간부터 재개한다. 전체 평가 표본은 줄이지 않는다.

## 공개 소스·라이선스·시점

확인2026-11-04 KST. 모델 파일/metadata/license/wheel은 `outputs/b23_research/`, 원래 가중치는 `outputs/b23_models/`. 공개 주소에서 다시 받을 수 있고 로그인·유료 API 없이 다운로드했다. 대회 데이터는 외부로 보내지 않았다.

### B2 TabPFN v2

- [공식 회귀 가중치](https://huggingface.co/Prior-Labs/TabPFN-v2-reg), 고정 revision `02a44705ff7790b50cded404bc59adb94da456a1`.
- `tabpfn-v2-regressor-v2_default.ckpt`,44,390,977바이트, SHA256 `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`.
- [정확한 가중치 라이선스](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/02a44705ff7790b50cded404bc59adb94da456a1/LICENSE.txt): Prior Labs License1.1, Apache2.0에 추가 출처 표시 조건. 상업 이용을 일괄 금지하는 후속 버전의 비상업 라이선스와 구분한다. 라이선스/저작권/변경사항 고지를 유지한다. 이 결과의 출처 표시: **Built with PriorLabs-TabPFN**. 내부 시험만이면 추가10조 표시 예외가 있지만 결과 문서에도 표시했다.
- 코드 `tabpfn==9.2.0`의 라이선스는 wheel의Apache2.0 고지 확인. 최신 버전의 기본 모델로 자동 전환하지 않고 `create_default_for_version(ModelVersion.V2)`와 로컬 가중치 경로를 명시한다. TabPFN2.5/2.6/3/3.5 가중치는 사용하지 않는다.
- 저장소 createdAt `2025-01-04T14:33:42Z` 확인. 현재 사용 파일 각각의 최초 공개일을 모두 검증한 것은 아니며, 이 날짜를 가중치별 발표 시각으로 바꾸어 쓰지 않는다.

### B3 TabICLv2

- [공식 회귀 가중치](https://huggingface.co/jingang/TabICL), 고정 revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`.
- `tabicl-regressor-v2-20260212.ckpt`,114,324,594바이트, SHA256 `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`.
- 모델 카드가BSD3-Clause를 명시하고, `tabicl==2.2.0` wheel의LICENSE/metadata에서 BSD3-Clause 확인. Copyright(c)2025 Soda team@Inria. 출처·라이선스/면책 고지를 유지하고 저작자 보증·제휴를 암시하지 않는다.
- 저장소 createdAt `2026-02-12T07:29:20Z`, checkpoint 파일명도20260212이다. [공식 문서](https://tabicl.readthedocs.io/en/latest/)의 v2 회귀 모델과 일치한다.

### 평가 시점과 공개 시점의 차이

TabPFN v2 저장소는2025년, TabICLv2는2026년 공개여서 평가 기간의 상당 부분이 가중치 공개보다 앞선다. **이번 비교는 나중에 공개된 방법을 과거 분할에 적용하는 기술 비교이며, 당시 배포 가능했던 시스템의 재현이 아니다.** 사용자 제공 운영팀 공지는 조건부 사전학습 모델 사용을 허용하지만, 가중치/라이브러리 각 버전에 대한 공개시점 조건 해석은 미확인이다. 운영팀에 확인하지 않고 공식 제출 적격이라고 확정하지 않는다. 데이터를 새로 공개된 모델에 실제 전송하지 않고 로컬에서만 계산한다.

## 환경·실행성·네트워크 차단

- B 전용 `.runtime/foundation-venv`: Python3.12.13, Torch2.15.0, TabPFN9.2.0, TabICL2.2.0. 공통 NumPy2.2.6/pandas2.2.3/PyArrow19.0.1/LightGBM4.6.0/scikit-learn1.6.1 유지. `requirements-foundation.txt`와 실제 environment-lock를 보관한다. pip check/self-test/check-only 통과.
- 일반 실행 샌드박스에서MPS가false였고, 합성 데이터의 권한 분리 시험에서MPS가true임을 확인했다. 표본을CPU로 몰래 바꾸지 않는다.
- 라이브러리의 기본skrub 캐시가B밖을 향하는 것을 발견하여 `SKB_DATA_DIRECTORY`를B .runtime으로 고정했다. 최초 시험에서 생긴 새 빈외부 폴더는rmdir로 제거했다. HF/Torch/XDG/Matplotlib 캐시도B 안으로 지정한다.
- 합성8,192문맥/1,024배치/2,048쿼리: TabPFN 문맥fit13.855초/추론3.101초/관찰MPS driver4,331.7MiB, TabICL4.126초/0.527초/2,414.6MiB. CPU peakRSS499.1/716.0MiB. GPU driver와CPU RSS를 합산해 동시 전체 사용량으로 보고하지 않는다.
- 배치/순서 불변성: TabPFN 최대 정규화 차이1.59e-7/0, TabICL8.34e-7/0. 실제 특징에서도 첫 배치와 소규모 배치를 추가 대조하고 연간 채점한다.
- 실제 자료 작업은 `/usr/bin/sandbox-exec -p '(version 1) (allow default) (deny network*)'` 아래에서 실행한다. 네트워크 소켓이OS에서PermissionError로 거부되고MPS 산술이 성공하는 합성 시험을 통과했다. 각 실행기/검산기도 데이터 로딩 전에 이 차단을 검사하고, 확인되지 않으면 중단한다. HF offline/로컬 가중치/자동 다운로드false도 적용한다.
- 첫 네트워크 차단 시험은 sandbox-exec가DYLD_LIBRARY_PATH를 지워OpenMP import에 실패했다. 내부`env`에서Python 직전에변수를 전달하여 해결했다. 소스/모델 변경 없이OS 네트워크 거부와MPS 동시 사용을 다시 검증했다.

## 진행과 결과

- 코드: `foundation_b23.py`, `run_foundation_b23.py`, `verify_foundation_b23.py`, `test_foundation_b23.py`.
- B2 `outputs/b23_tabpfn/`, 로그 `outputs/b23_tabpfn.log`/`.verify.log`.
- B3 `outputs/b23_tabicl/`, 로그 `outputs/b23_tabicl.log`/`.verify.log`.
- `outputs/status.json`은 완료B1과 현재B2/B3 단계를 함께 남긴다.
- B2/B3의 전체 평가·독립 검산이 완료됐으며 아래 최종 결과에 실제 측정값을 기록했다.

- 최초 연간 실행은 Path import 누락으로 명령행 인자 정의에서 종료1(실제자료 로딩/예측 전)했다. CLI --help 회귀 검사에서 실패를 재현하고 import 수정 후24검사 통과. 원래 실패 로그는 outputs/b23_research/{pipeline,tabpfn}_initial_failure.log에 보존하고 재시작했다.

- 전원 대기: AC연결 표시에도 배터리2%/방전/OS추정5분을 확인하여B2 F3-r17 GPU worker PID52919만SIGSTOP했다. 저장된예측/8192문맥/고정설정은보존. 충전상태안정후SIGCONT로이어간다. 최신진행은outputs/status.json, 중단기록은outputs/b23_research/power_interruption.json. 원시활성추론시간에일시중단대기가들어갈수있으므로시간보고시별도표시한다.

- 전원 재개: 사용자 충전 확인 후 AC/배터리87%를 양 실행 환경에서 확인했다. 충전감시기는 ps 실행 권한 오류로 종료되어, B worker PID/명령/정지상태를 직접 검증한 뒤 2026-11-04T14:40 UTC경SIGCONT했다. 저장770048행부터 동일모델/설정으로계속하며 네트워크차단유지. 전원대기벽시계와실행기perf_counter시간은동일하지않으므로원시시간을그대로보존하고대기포함가능성을표시한다.

## 최종 결과 — 완료

| 실험 | F2 W | P4 대비 | F3 W | P4 대비 |
|---|---:|---:|---:|---:|
| P4+GEFS | 1,039,175,004.310 | -7.339% | 1,998,466,851.636 | -3.162% |
| TabPFN v2 (8192 context) | 1,168,621,543.477 | +4.203% | 2,084,775,339.748 | +1.020% |
| TabICLv2 (8192 context) | 1,182,359,161.396 | +5.428% | 2,166,110,322.429 | +4.961% |

[종합 결과와 측정/한계](../../reports/b-process-results.md). 모든 모델의 verification.json은passed다. 위의 진행 중·전원 대기는 이력이며 현재 계산은 완료했다.

최종 확인은 2026-11-05 KST. B2 검산92.978초, B3 검산51.933초, 최종24검사 통과. B2의 전원 대기 포함 시간과 실제 GPU 관측/월·팀 변화는 [종합 결과](../../reports/b-process-results.md)에 정리했다.
