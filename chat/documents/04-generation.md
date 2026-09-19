# 04. 텍스트 생성 — 자기회귀(Autoregressive) 샘플링

관련 코드: `model.py`의 `GPT.generate`, `chat.py`

## 자기회귀란

학습된 모델이 할 수 있는 것은 딱 하나: **지금까지의 문자열을 보고, 바로
다음 한 글자의 확률분포를 예측하는 것**뿐입니다. 여러 글자로 된 문장을
만들려면 이 "한 글자 예측"을 반복해서 이어 붙여야 합니다.

```
프롬프트: "First Citizen:"

1회차: "First Citizen:"                    다음 글자 예측 → "\n"
2회차: "First Citizen:\n"                  다음 글자 예측 → "M"
3회차: "First Citizen:\nM"                 다음 글자 예측 → "y"
...                                        (max_new_tokens번 반복)
```

매번 "지금까지 생성된 전체 문자열"을 다시 모델에 넣고, 마지막 위치의
예측만 사용해 한 글자를 뽑고, 그 글자를 문자열 뒤에 붙인 뒤 또 반복하는
것 — 이것이 `generate` 함수가 하는 일의 전부입니다. 스스로 만든 출력을
다시 자신의 입력으로 사용하기 때문에 "자기회귀(auto-regressive)"라고
부릅니다.

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -self.config.block_size:]   # 최근 block_size개만 사용
    logits, _ = self(idx_cond)
    logits = logits[:, -1, :]                       # 마지막 위치(=다음 글자) 예측만 사용
    ...
    next_id = torch.multinomial(probs, num_samples=1)  # 확률분포에서 샘플링
    idx = torch.cat((idx, next_id), dim=1)              # 이어붙이기
```

`idx[:, -block_size:]`로 자르는 이유: 모델은 `block_size`보다 긴 문맥은
애초에 본 적이 없어서(위치 임베딩 테이블 크기가 `block_size`) 처리할 수
없기 때문에, 항상 "최근 block_size개 글자"만 잘라서 넣습니다.

## 왜 확률이 가장 높은 글자를 항상 고르지 않는가 (샘플링 vs Greedy)

가장 단순한 방법은 매번 확률이 제일 높은 글자만 고르는 것(**greedy
decoding**, `argmax`)입니다. 하지만 이렇게 하면:

- 같은 프롬프트를 넣으면 항상 완전히 같은 문장이 나옴 (다양성 없음)
- 특정 패턴이 반복 루프에 빠지기 쉬움 ("the the the the ...")

그래서 `torch.multinomial(probs, num_samples=1)`로 **확률에 비례해서
무작위로** 하나를 뽑습니다. 확률이 높은 글자가 뽑힐 가능성이 크지만,
매번 정확히 같은 글자만 나오지는 않습니다 — 실제 ChatGPT/Claude 등에서
같은 질문을 다시 물으면 답이 조금씩 달라지는 것과 같은 원리입니다.

## temperature — 무작위성 조절 손잡이

```python
logits = logits / max(temperature, 1e-5)
```

softmax에 들어가기 전 logits을 temperature로 나눕니다.

- **temperature < 1** (예: 0.5): logits 간 차이가 상대적으로 더 벌어짐
  → softmax 후 확률분포가 더 "뾰족"해짐 → 1등 후보에 확률이 더 쏠림
  → 더 보수적이고 반복적이지만 안정적인 문장
- **temperature = 1**: 모델이 계산한 확률을 그대로 사용
- **temperature > 1** (예: 1.5): 확률분포가 더 "평평"해짐 → 낮은 확률의
  후보도 뽑힐 기회가 늘어남 → 더 다양하지만 문법이 깨질 위험도 커짐

```
온도가 낮을 때(뾰족함):  ▇▇▇▇▇▇ ▂ ▁ ▁ ▁ ▁         → 거의 항상 1등만 뽑힘
온도가 높을 때(평평함):  ▄▄ ▃▃ ▃▃ ▃▃ ▃▃ ▃▃         → 여러 후보가 고르게 뽑힘
```

## top_k — 말이 안 되는 후보를 아예 제거

```python
v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
logits[logits < v[:, [-1]]] = -float("inf")
```

temperature만으로는 "확률이 아주 낮지만 0은 아닌" 이상한 후보가 아주
가끔 뽑혀 문장을 망칠 수 있습니다. `top_k`는 매 스텝마다 **확률 상위
k개** 후보만 남기고 나머지는 아예 후보에서 제외(`-inf`로 만들어 softmax
확률을 0으로)한 뒤, 그 안에서만 temperature 샘플링을 합니다. 안정성과
다양성 사이의 절충점을 제공하는 장치입니다.

## chat.py에서의 실제 흐름

```
사용자 입력(prompt)
   │  encode(): 문자열 → 정수 인덱스 (학습 때 없던 문자는 무시됨)
   ▼
idx = [[...]]  (배치 차원 1개 추가, shape=(1, T))
   │  model.generate(idx, max_new_tokens, temperature, top_k)
   ▼
out = [[프롬프트 인덱스... , 새로 생성된 인덱스...]]
   │  decode(): 정수 인덱스 → 문자열
   ▼
텍스트 전체 중 "프롬프트 이후 부분"만 잘라서 출력
```

## 이 생성 방식의 한계 (그리고 왜 "대화"처럼 안 느껴질 수 있는가)

- 모델은 "질문에 답하라"고 학습된 적이 없고, 그냥 셰익스피어풍 텍스트의
  통계적 다음 글자를 이어 쓸 뿐입니다. 프롬프트가 질문형이어도 모델은
  그것을 그냥 "이어 쓸 텍스트의 시작 부분"으로만 취급합니다.
- `block_size`(기본 128자)보다 긴 맥락은 아예 기억하지 못합니다.
- 파라미터 수가 적고 학습 데이터도 작아, 사실관계나 논리적 일관성을
  기대할 수 없습니다 — 이 프로젝트의 목적은 "그럴듯한 대화"가 아니라
  **Transformer/GPT가 내부적으로 어떻게 동작하는지 직접 보는 것**입니다.

진짜 질의응답형 챗봇에 가깝게 만들려면 [05-next-steps.md](05-next-steps.md)의
"instruction tuning" 방향을 참고하세요.
