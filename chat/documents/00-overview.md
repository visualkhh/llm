# 00. 전체 개요 — 이 프로젝트는 무엇을 만드는가

이 프로젝트는 **문자 단위(character-level) 미니 GPT**를 처음부터(from scratch)
직접 구현하고, 학습시키고, 그 결과로 텍스트를 생성해보는 교육용 프로젝트입니다.
ChatGPT/Claude 같은 실제 서비스형 LLM과 "같은 원리"로 동작하지만, 규모가
수만 배 이상 작습니다 (파라미터 약 80만 개 vs 실제 LLM은 수십억~수조 개).

## 왜 작게 만들었나

- 노트북/개인 PC의 GPU(Apple Silicon의 MPS 등)에서 몇 분~몇십 분 안에 학습이 끝남
- 원리를 이해하는 게 목적이므로, 크기보다 **구조를 눈으로 확인할 수 있는 것**이 중요
- 같은 코드에서 `n_layer`, `n_embd` 등 숫자만 키우면 원리 그대로 더 큰 모델도 됨

## 전체 파이프라인

```
data/input.txt (원문 텍스트)
        │
        ▼  data/prepare.py  — 문자를 숫자로 변환(토큰화)
data/train.bin, data/val.bin, data/meta.pkl
        │
        ▼  train.py  — 모델(model.py)을 학습시킴
checkpoints/ckpt.pt (학습된 가중치 + 설정 + 토크나이저 정보)
        │
        ▼  chat.py  — 체크포인트를 불러와 텍스트 생성
사용자 입력(프롬프트) → 모델이 이어서 생성한 텍스트
```

## 파일별 역할 한눈에 보기

| 파일 | 역할 | 자세한 설명 |
|---|---|---|
| `data/prepare.py` | 텍스트 → 숫자 배열로 변환(토큰화) | [01-tokenization.md](01-tokenization.md) |
| `model.py` | GPT 모델 구조(Transformer) 정의 | [02-transformer-architecture.md](02-transformer-architecture.md) |
| `train.py` | 모델을 실제로 학습시키는 루프 | [03-training.md](03-training.md) |
| `chat.py` | 학습된 모델로 텍스트 생성(추론) | [04-generation.md](04-generation.md) |

각 코드 파일에도 위 문서 내용에 대응하는 상세 주석이 달려 있습니다.
문서를 먼저 훑고 코드를 보거나, 코드를 보다가 막히는 부분의 문서를
찾아 읽는 방식 모두 괜찮습니다.

## 이 모델이 "대화"를 하는가?

엄밀히는 아닙니다. 이 모델이 학습한 것은 오직 하나:

> "지금까지 입력된 글자들 다음에, 학습 데이터(셰익스피어 희곡)에서
> 통계적으로 가장 자주/자연스럽게 나타났던 글자는 무엇인가?"

즉 **다음 글자 예측**만 반복할 뿐, 질문에 "답하도록" 학습된 적은 없습니다.
실제 ChatGPT류 서비스가 대화처럼 느껴지는 이유는:

1. 훨씬 큰 모델을 훨씬 많은(웹 전체 규모) 텍스트로 사전학습(pretraining)하고,
2. 그 위에 "질문-답변" 형식 데이터로 추가 학습(instruction tuning)하고,
3. 사람 피드백으로 한 번 더 다듬기(RLHF 등) 때문입니다.

이 프로젝트의 코드/구조는 위 1번 단계(사전학습)의 축소판입니다.
[05-next-steps.md](05-next-steps.md)에 2번 방향으로 확장하는 방법을 정리해두었습니다.

## 실행 방법 요약

```bash
# 1) 말뭉치를 문자 단위로 토큰화
python3 data/prepare.py

# 2) 학습 (기본 3000 스텝, Mac이면 자동으로 MPS GPU 사용)
python3 train.py --max_iters 3000

# 3) 학습된 모델과 대화(=텍스트 이어쓰기)
python3 chat.py
```

더 자세한 실행/실험 방법은 [05-next-steps.md](05-next-steps.md)를 참고하세요.
