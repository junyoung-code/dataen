# 2026-11-05 혼합 실험 두 채팅 시작

사용자 요청으로 1번/2번을 새 채팅에서 실행한다. 두 채팅 모두 gpt-6-astra / high로 생성하고 active 상태 및 작업 시작 메시지를 확인했다. 초기 상태는 인계 문서·입력 자산 확인이며 수치 계산의 시작·완료는 각 outputs/status.json과 로그로 별도 확인한다.

| 실험 | 계산 기기 | 범위 | 독립 작업 폴더 |
|---|---|---|---|
| BL1 | 맥북 | 기존 B1 GEFS + TabM/TFT 예측 혼합 | outputs/blend_20261105/local_ab |
| BL2 | 맥미니 (SSH) | 팀원 S914 방식 기준선 재현 + 우리 예측 소량 혼합 | outputs/blend_20261105/peer_mini; 원격 ~/공모전/dataen/blend_peer_20261105 |

- BL1 채팅: 01a1aa74-a973-7ab3-a9a3-d0bfb3f31b07, 브랜치 codex/blend-ab-20261105.
- BL2 채팅: 01a1aa74-bc60-7410-bdf4-d3a16396bc23, 브랜치 codex/blend-peer-20261105.
- 두 독립 clone의 출발점: 완료된 A dbf8040dbc1750333e9fe303bcf76faf5c402417. 원격 origin은 제거했고 부모 미커밋 구조 정리와 기존 A/B는 보존한다. 각 lane은 자기 clone의 코드/문서/outputs만 수정한다. 부모 루트 문서는 부모 담당.
- 사용자 SSH 인증 후 macmini BatchMode 접속과 기존 A 결과 존재를 검증했다. BL1의 경량 신경망 예측 추출은 별도 blend_ab_export_20261105 폴더만 사용한다.
- 공통 평가 계약: AB_EVALUATION_CONTRACT.json SHA256 d65965390a3027d6b3267c352d7222cdfd76a9dca128e7d319bab3c77465b1ec. 같은 F2/F3·부분 관측 마스크·W로 비교한다. F2로 선택하고 F3는 비중을 고정해 보고하며 둘 다 이미 본 역사적 검증이다.
- BL1은 기존 참조 배정, BL2는 팀원 C_strat_v1 배정을 고정하고 각 비교 안에서 모든 후보를 동일하게 재합산한다. 서로 다른 배정의 집계 예측을 그대로 혼합하지 않는다. 팀원의 학습 경계 차이도 기록하고 예측시점 이후 실측을 새 학습에 쓰지 않는다.
- Fast 속도는 생성 도구에 옵션이 없고 Codex 앱 화면 제어도 제한돼 적용하지 못했다. Astra High 지정만 확인됐으며 속도는 사용자가 각 채팅에서 설정할 수 있다.
- 준비 기록: outputs/blend_20261105/control/dispatch.json. 세부 지시는 각 clone LANE_START.md. 각 outputs/status.json과 실제 로그가 준비 시점 문서보다 우선한다.

## 다음 기기의 시작점

해당 lane의 LANE_START.md와 outputs/status.json을 읽고 진행 중 작업에 이어서 수행한다. 기존 모델/원본/공유 환경을 덮어쓰지 않는다. BL2의 팀원 OOF 자산 가용성과 기준선 재현 여부가 주요 미확인 사항이다. 완료 시 실제 W, 비교 기준, 시간, 코드/입력 해시, 실패·미실행을 각 clone EXPERIMENTS/CONTEXT/EXPERIMENT_PLAN에 기록한다. 원격 push·공식 제출은 수행하지 않는다.

## 배정 후 실제 진행 확인

- BL1: active, 12개 후보 사전 고정, 상태 파일 running / transferring_saved_predictions. 기존 OpenMP 검색 경로를 실행 명령에 지정해 환경을 재사용 중이다. 혼합 결과는 아직 미확인이다.
- BL2: active, 맥미니 SSH 접속 및 기존 신경망 예측 확인. 팀원 일부 보정의 역방향 fold 사용 가능성을 발견해 시간 인과성을 확인 중이다. 팀원 기록 W를 그대로 재현 유효 기준선으로 확정하지 않았고 실제 학습/수치 결과는 아직 미확인이다.
