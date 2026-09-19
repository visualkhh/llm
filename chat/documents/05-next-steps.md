# 05. 다음 실험 아이디어 & 진짜 "대화"에 가깝게 만들기

## 지금 바로 해볼 수 있는 실험

1. **더 오래 학습시키기**
   ```bash
   python3 train.py --max_iters 10000
   ```
   `documents/03-training.md`에서 설명한 대로, val loss가 계속 떨어지는
   동안은 늘려볼 가치가 있습니다. 어느 시점부터 val loss가 다시 오르기
   시작하면 그게 이 모델 크기/데이터로 낼 수 있는 한계에 가깝다는 신호입니다.

2. **모델 크기 키워보기**
   ```bash
   python3 train.py --n_layer 6 --n_head 6 --n_embd 192 --max_iters 5000
   ```
   `n_layer`/`n_head`/`n_embd`를 키우면 파라미터 수가 늘어 표현력은
   커지지만, 학습 시간도 늘고 지금처럼 작은 데이터(백만 자 수준)에서는
   오히려 과적합이 더 빨리 올 수도 있습니다. val loss를 보며 판단하세요.

3. **temperature / top_k 바꿔가며 생성 품질 비교**
   ```bash
   python3 chat.py --temperature 0.5 --top_k 20   # 보수적/안정적
   python3 chat.py --temperature 1.2 --top_k 200  # 다양하지만 불안정
   ```

4. **본인만의 말뭉치로 교체**
   - `data/input.txt`를 원하는 텍스트(예: 좋아하는 소설, 블로그 글 모음,
     혹은 한국어 텍스트)로 통째로 교체
   - `python3 data/prepare.py` 다시 실행 (vocab이 새로 만들어짐 — 한국어를
     넣으면 한글 자모/음절이 문자 단위로 그대로 vocab에 포함됨)
   - `python3 train.py`로 재학습
   - 문자 단위 토큰화라 어떤 언어를 넣어도 코드 수정 없이 그대로 동작합니다.

## 진짜 "질문-답변" 형태의 대화에 가깝게 만들려면

지금 모델은 "다음 글자 이어쓰기"만 학습했습니다(=사전학습, pretraining과
같은 성격). ChatGPT/Claude 같은 서비스가 실제로 대화를 하는 것처럼
느껴지는 이유는 그 위에 몇 단계가 더 있기 때문입니다.

```
1) 사전학습 (Pretraining)          ← 지금 이 프로젝트가 하는 것
   대량의 일반 텍스트로 "다음 토큰 예측"만 학습

2) 지도 미세조정 (SFT, Instruction Tuning)
   "질문 -> 좋은 답변" 형태의 (프롬프트, 응답) 쌍 데이터로 추가 학습
   → 모델이 "질문에는 답변 형식으로 응답"하는 패턴을 배움

3) 사람 피드백 기반 강화학습 (RLHF) 등
   여러 답변 후보 중 사람이 선호하는 것을 골라주는 피드백으로 한 번 더 다듬음
```

이 미니 프로젝트를 2단계 방향으로 확장해보고 싶다면:

1. `(질문, 답변)` 쌍으로 된 데이터를 특정 구분자로 만듭니다. 예:
   ```
   ### 질문: 오늘 날씨 어때?
   ### 답변: 저는 날씨 정보를 실시간으로 알 수 없어요.
   <|endofturn|>
   ### 질문: ...
   ```
2. `data/input.txt`를 이런 형식의 텍스트 여러 개로 채우고 `prepare.py`로
   재토큰화합니다.
3. `chat.py`에서 사용자 입력을 받을 때 `### 질문: {입력}\n### 답변:` 형태로
   감싸서 프롬프트를 만들고, `<|endofturn|>`이 나오면 생성을 멈추도록
   `generate` 함수를 조금 손보면(EOS 토큰 감지) 훨씬 "대화"에 가까운
   행동을 보이기 시작합니다.

단, 이 미니 모델(파라미터 수백만~수천만 개, 문자 단위 토큰화)로는
데이터가 아주 많지 않은 이상 여전히 품질에 한계가 있습니다. 이 방향은
"instruction tuning이 무엇을 하는지 원리를 체험"하는 데 의미가 있고,
실용적인 챗봇 품질을 원한다면 처음 대화에서 다뤘던 "오픈소스 LLM
파인튜닝" 경로(HuggingFace의 사전학습된 모델 + LoRA 등)가 훨씬 효율적입니다.

## 서브워드(BPE) 토크나이저로 바꿔보기

문자 단위 대신 실제 GPT류가 쓰는 BPE(Byte-Pair Encoding)를 붙여보는 것도
좋은 다음 실습입니다.

- `pip install tiktoken` 후 `tiktoken.get_encoding("gpt2")`로 인코딩/디코딩
- `data/prepare.py`의 `encode`/`decode`를 이걸로 교체, `vocab_size`는
  `enc.n_vocab`으로 대체
- 같은 텍스트 길이 대비 토큰 수가 훨씬 줄어들어(한 토큰이 여러 글자를
  포함) 같은 `block_size`로 더 긴 실제 문맥을 다룰 수 있게 됩니다.

## 참고하면 좋은 자료

- Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out"
  (이 프로젝트의 구조가 크게 참고한 nanoGPT 계열 설명 영상/코드)
- "Attention Is All You Need" (Vaswani et al., 2017) — Transformer 원논문
- OpenAI GPT-2 논문 — Pre-LN, GELU 등 이 코드에 쓰인 설계 선택들의 출처
