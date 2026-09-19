# 03. 학습 과정 — 모델은 어떻게 "배우는가"

관련 코드: `train.py`

## 학습의 4단계 사이클

`train.py`의 for 루프는 아래 네 단계를 `max_iters`번 반복합니다.

```python
x, y = get_batch(...)          # 1. 배치 뽑기
_, loss = model(x, y)          # 2. 순전파(forward) + loss 계산
optimizer.zero_grad()
loss.backward()                # 3. 역전파(backward) — gradient 계산
optimizer.step()               # 4. 가중치 업데이트
```

### 1. 배치 뽑기 — `get_batch`

전체 토큰 스트림에서 무작위 위치를 `batch_size`개 골라, 각 위치에서
`block_size`개 길이의 (입력 x, 정답 y) 쌍을 만듭니다.

**핵심: y는 x를 한 칸 뒤로 민 것.**

```
data = "HELLO WORLD"
block_size = 4, 시작 위치 i = 0 이라면
  x = "HELL"   (인덱스 0..3)
  y = "ELLO"   (인덱스 1..4)
```

이렇게 하면 x의 위치 t마다 "x[t] 다음에 실제로 온 글자"가 y[t]에 들어있게
되어, 한 시퀀스 안에서 `block_size`개의 "다음 글자 맞히기" 학습 샘플을
동시에 얻습니다 (`model.py`의 Causal masking 덕분에 이 병렬 계산이
"미래를 훔쳐보지 않고" 안전하게 이루어짐).

매 스텝 무작위 위치를 새로 뽑는 것은 **확률적 경사하강법(SGD, Stochastic
Gradient Descent)**의 "확률적" 부분입니다 — 전체 데이터를 매번 다 보는 대신
무작위 표본으로 gradient를 추정해도 평균적으로는 올바른 방향으로 수렴합니다.

### 2. 순전파 + Loss — 얼마나 틀렸는지 측정

`model(x, y)`는 내부에서 `model.py`에 설명된 대로 예측 logits을 만들고,
**Cross-Entropy Loss**로 정답과의 차이를 하나의 숫자로 요약합니다.

직관적으로 Cross-Entropy는 "모델이 정답 글자에 부여한 확률이 높을수록
loss가 작고, 낮을수록(엉뚱한 글자에 확신을 가질수록) loss가 크다"는 지표입니다.
`-log(정답 글자에 부여된 확률)` 형태이므로, 정답에 확률 1.0을 주면 loss=0,
확률이 0에 가까울수록 loss는 무한히 커집니다.

### 3. 역전파 — 어느 방향으로 고쳐야 할지 계산

`loss.backward()` 한 줄이 하는 일: **연쇄법칙(chain rule)**을 이용해
"이 loss를 줄이려면 모델의 각 가중치(임베딩, attention의 Q/K/V 투영,
MLP의 가중치 등 수십만 개 숫자)를 어느 방향으로, 얼마나 민감하게 바꿔야
하는가"를 전부 계산합니다. 이 결과가 각 파라미터의 `.grad`에 저장됩니다.

`optimizer.zero_grad()`를 매 스텝 먼저 호출하는 이유: PyTorch는 기본적으로
gradient를 덮어쓰지 않고 "누적"하기 때문에, 새 스텝을 시작하기 전 이전
스텝의 잔여 gradient를 지워야 합니다.

### 4. 가중치 업데이트 — Optimizer(AdamW)

가장 단순한 경사하강법은 `가중치 -= lr * gradient` 인데, `train.py`는
더 발전된 **AdamW**를 씁니다. AdamW는 파라미터마다 과거 gradient의
평균과 분산을 추적해서 파라미터별로 실질적인 학습 속도를 자동 조절하고
(자주/크게 바뀌어야 하는 파라미터는 더 크게, 안정적인 파라미터는 더 작게),
weight decay라는 항으로 가중치가 지나치게 커지는 것도 억제합니다.
현재 딥러닝, 특히 Transformer 계열에서 사실상 표준 옵티마이저입니다.

`lr(learning rate)=3e-4`는 한 번에 얼마나 크게 이동할지를 정하는 값으로,
너무 크면 loss가 진동/발산하고 너무 작으면 학습이 지나치게 느립니다.

## 검증(validation)과 과적합(overfitting) 감시

`estimate_loss` 함수는 `eval_interval` 스텝마다 train loss와 val loss를
각각 여러 배치 평균으로 추정합니다.

- **train loss**: 학습에 실제로 쓰인 데이터에 대한 성능
- **val loss**: 학습에 전혀 쓰이지 않은 데이터에 대한 성능

두 값을 함께 보는 이유:

```
정상적인 학습                     과적합(overfitting) 시작
train loss ↓, val loss ↓          train loss ↓ 계속, val loss ↑ 로 반전
(패턴을 잘 배우는 중)              (데이터를 "암기"하기 시작, 일반화 능력 저하)
```

`train.py`는 **val loss가 지금까지 중 가장 낮았을 때만** 체크포인트를
저장하므로, 과적합이 시작된 이후의 상태가 아니라 "가장 일반화가 잘 된"
시점의 모델이 자동으로 `checkpoints/ckpt.pt`에 남습니다.

## 메모리 효율 — `np.memmap`

`data/train.bin`은 numpy의 `memmap`으로 열립니다. 이는 파일 전체를
한 번에 메모리로 읽어들이는 대신, 실제로 접근하는 부분만 그때그때
디스크에서 읽어오는 방식입니다. 말뭉치가 수 GB로 커져도 RAM 사용량이
거의 늘지 않는다는 장점이 있습니다 (지금 규모의 Tiny Shakespeare에서는
체감이 크지 않지만, 실제 LLM 학습에서 필수적인 기법입니다).

## 실행 시 로그 읽는 법

```
device: mps
model params: 826,368
iter 250: train loss 2.4345, val loss 2.4390 (9.3s)
...
iter 3000: train loss 1.4564, val loss 1.6511 (109.5s)
done. best val loss: 1.6511
```

- `device`: 실제 연산이 어디서 도는지 (`mps`=Apple GPU, `cuda`=Nvidia GPU, `cpu`)
- loss는 절대값 자체보다 **추세**가 중요합니다. 문자 단위 모델에서 loss가
  약 1.0~1.5 근처까지 내려오면 (65개 후보 중 아무거나 찍는 것보다 훨씬 나은,
  `-log(1/65) ≈ 4.17`이 "완전 무작위" 수준의 loss) 눈에 띄게 그럴듯한
  단어/문장 구조가 나타나기 시작합니다.

## 주요 하이퍼파라미터 요약

| 인자 | 의미 | 늘리면 | 줄이면 |
|---|---|---|---|
| `--max_iters` | 총 학습 스텝 수 | 더 오래 학습 → 보통 더 좋아짐(단, 과적합 위험도 증가) | 빠르지만 덜 학습됨 |
| `--n_layer`, `--n_embd`, `--n_head` | 모델 크기 | 표현력↑, 느려짐, 더 많은 데이터 필요 | 빠름, 표현력↓ |
| `--block_size` | 문맥 길이 | 더 긴 문맥 참고 가능, 메모리/연산량↑ | 짧은 문맥만 참고 |
| `--batch_size` | 한 스텝에 처리하는 시퀀스 수 | 학습이 더 안정적(gradient 추정이 덜 흔들림), 메모리↑ | 불안정하지만 가벼움 |
| `--lr` | 학습률 | 너무 크면 발산 | 너무 작으면 느림 |

실험해볼 때는 한 번에 여러 값을 바꾸지 말고, 하나씩 바꿔가며 val loss가
어떻게 변하는지 관찰하는 것을 추천합니다.
