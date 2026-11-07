# TabM·LSTM·TFT 1차 비교

2026-11-03, 사용자 실행 요청. 맥북은 TabM→LSTM, 맥 미니는 TFT를 실행한다. 기존 환경과 분리한 `.runtime/neural-venv`를 사용한다. 외부 학습 가중치 없이 제공 데이터로 처음부터 학습한다.

- 공통 평가: p0-v1의 세 기간·원본 라벨·공통 시간·팀 가중 MAE, 기존 그룹, 10시/17시 25:75. 이미 본 역사적 검증이며 독립 최종 테스트가 아니다.
- 학습: 기존과 같은 관측된 낮 행, 정규화 발전량·용량 가중 L1. 결측 라벨·밤은 신경망 손실에서 제외한다. 입력의 결측 대체·표준화는 해당 학습 구간의 관측된 낮 행만으로 맞춘다.
- TabM은 26개 수치 입력을 행별로 사용한다. LSTM·TFT는 발전소별 이전 24시간 예보를 인코딩하고 대상일 01시~다음날00시의 24시간 예보로 예측한다. 같은 26개 변수지만 시간 문맥이 추가되므로 입력 구성이 완전히 같다고 부르지 않는다. 실제 발전량·실측 기상·발전소 ID는 입력에 넣지 않는다. 각 예보는 해당 대상일 전에 제공된 10시/17시 회차만 사용한다.
- 1차 구조: TabM 3블록·폭128·내부8개 모델·dropout0.1, LSTM 인코더/디코더 각1층·폭64, TFT 폭16·연속변수폭8·헤드4·LSTM1층·dropout0.1. AdamW 학습률0.001, weight_decay0.0001, gradient clip1, seed0.
- 각 구조에서 5/10/20 epoch 체크포인트를 tune의 팀 지표로 비교하고 한 epoch를 고정한다. 이후 계절·확인 기간은 선택 epoch만 새로 학습한다. 초기화·학습 순서가 다른 독립 3회 실험은 아니다. 학습 중 확인 기간으로 설정을 조정하지 않는다.
- 설치 검사·가중 손실·시간/발전소 경계·미래 라벨 차단·저장/재로딩 검사와 짧은 처리량 시험 후 실행한다. GPU가 지원되는지는 실제 정방향/역전파로 확인한다. 실행 시간·메모리·학습곡선·실패와 실제 기기를 기록한다. 다른 기기의 학습 시간을 모델 자체 속도로 직접 비교하지 않는다.

구현 출처(2026-11-03 확인): [공식 TabM](https://github.com/yandex-research/tabm), [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html), [PyTorch Forecasting TFT](https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting/models/temporal_fusion_transformer/_tft.html). 사전학습 파일은 받지 않는다. 기존 LightGBM의 누적 튜닝 기회가 더 많다는 한계는 유지한다.
