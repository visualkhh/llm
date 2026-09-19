# ============================================================================
# model.py — 미니 GPT(Decoder-only Transformer) 구현
#
# 전체 데이터 흐름 요약:
#   문자 인덱스(idx) --[토큰 임베딩 + 위치 임베딩]--> 벡터
#     --[Transformer Block x n_layer]--> 문맥이 반영된 벡터
#     --[LayerNorm + Linear(head)]--> 각 문자에 대한 점수(logits)
#     --[softmax]--> 다음 문자가 무엇일지에 대한 확률분포
#
# GPT는 "다음 토큰을 예측"하는 것 하나만 학습한다. 대화처럼 보이는 것도
# 결국 "지금까지 입력된 글자들 다음에 올 확률이 가장 그럴듯한 글자"를
# 하나씩 이어붙이는 것뿐이다. (=자기회귀, autoregressive)
# ============================================================================

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """모델 크기/구조를 정의하는 하이퍼파라미터 모음.

    vocab_size : 다룰 수 있는 토큰(여기서는 문자) 종류의 개수.
                 data/prepare.py 가 만든 meta.pkl 의 vocab_size 와 반드시 같아야 함.
    block_size : 모델이 한 번에 볼 수 있는 최대 문맥 길이(context length).
                 "이 글자 다음에 뭐가 올까"를 예측할 때 최근 block_size개
                 글자까지만 참고한다는 뜻. (ChatGPT의 "context window"와 같은 개념)
    n_layer    : Transformer Block을 몇 겹 쌓을지. 깊을수록 더 복잡한 패턴을
                 학습할 수 있지만 느려지고 데이터도 더 많이 필요함.
    n_head     : Self-Attention을 몇 개의 "머리(head)"로 나눠서 병렬로 볼지.
                 각 head는 서로 다른 종류의 관계(문법, 반복, 거리 등)에 집중하도록
                 학습된다.
    n_embd     : 각 토큰을 표현하는 벡터의 차원(embedding dimension).
    dropout    : 학습 중 일부 뉴런을 확률적으로 꺼서 과적합(overfitting)을 막는 비율.
    """
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Masked(=Causal) Multi-Head Self-Attention.

    Self-Attention이 하는 일을 한 문장으로 요약하면:
    "문장 안의 각 토큰이, 같은 문장 안의 다른 토큰들을 얼마나 '참고'할지
    가중치를 스스로 계산해서, 그 가중 평균으로 자신의 표현을 갱신한다."

    구체적으로는 각 토큰 벡터에서 세 가지 벡터를 만든다.
      Q (Query) : "나는 지금 무엇을 찾고 있는가?"
      K (Key)   : "나는 어떤 정보를 갖고 있다고 남에게 알려줄까?"
      V (Value) : "실제로 참고할 내용물"

    어떤 토큰 i의 새 표현 = sum_j softmax(Q_i · K_j / sqrt(d)) * V_j
    즉 Q_i와 K_j의 내적(유사도)이 클수록 V_j를 더 많이 반영한다.

    "Causal(=인과적)"이라는 말은, 문장을 왼쪽에서 오른쪽으로 생성해야 하므로
    토큰 i는 자기 자신과 그 이전 토큰들(j <= i)만 볼 수 있고 미래 토큰(j > i)은
    절대 못 보게 마스킹한다는 뜻. 학습 때 정답(다음 글자)을 미리 훔쳐보는 것을
    막기 위한 장치다. (F.scaled_dot_product_attention 의 is_causal=True 가 이 역할)

    "Multi-Head"는 위 과정을 n_embd 차원 전체로 한 번에 하지 않고, n_head개의
    작은 부분공간(head_dim = n_embd / n_head)으로 나누어 각각 독립적으로
    수행한 뒤 결과를 이어붙이는 것. 서로 다른 head가 서로 다른 관계 패턴
    (예: "바로 앞 글자", "문장의 시작 글자", "따옴표 짝" 등)을 전문적으로
    학습하게 하려는 목적이다.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # 하나의 Linear로 Q, K, V를 한 번에 계산 (3배 크기로 뽑은 뒤 나눠 씀 -> 효율적)
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 여러 head의 결과를 이어붙인 뒤, 다시 원래 차원으로 섞어주는 투영
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: (B, T, C)
        #   B = batch size (한 번에 처리하는 문장 개수)
        #   T = 이 배치에서의 시퀀스 길이 (block_size 이하)
        #   C = n_embd (임베딩 차원)
        B, T, C = x.shape

        # (B, T, C) -> (B, T, 3C) 로 만든 뒤 Q/K/V로 3등분
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        # head 차원을 앞으로 옮겨서, 이후 attention 연산이 "각 head를 독립된
        # 배치처럼" 병렬로 계산되게 만든다.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled_dot_product_attention 내부에서 일어나는 일 (개념적으로):
        #   attn_scores = (q @ k.transpose(-2, -1)) / sqrt(head_dim)   # 유사도
        #   attn_scores[j > i] = -inf                                  # 미래 마스킹 (is_causal)
        #   attn_weights = softmax(attn_scores, dim=-1)                # 확률화
        #   out = attn_weights @ v                                     # 가중 평균
        # 위 네 줄을 PyTorch가 최적화된 커널로 한 번에 처리해주는 것이 이 함수.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
        )

        # 다시 (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, C)
        # 로 head들을 이어붙여 원래 모양으로 복원
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    """Position-wise Feed-Forward Network (각 토큰 위치마다 독립적으로 적용되는 MLP).

    Self-Attention이 "토큰들 사이의 관계"를 보는 역할이라면,
    이 MLP는 attention이 모아온 정보를 바탕으로 "토큰 하나하나를 개별적으로
    더 깊게 가공"하는 역할을 한다. Transformer는 이 두 가지(관계 파악 + 개별 가공)를
    번갈아 쌓아서 표현력을 키운다.

    구조는 단순한 2층 MLP: n_embd -> 4*n_embd -> n_embd.
    중간을 4배로 넓혔다가 다시 줄이는 것은 GPT/Transformer 논문들의 관례적 설계로,
    더 넓은 중간 표현이 다양한 패턴을 담을 여유를 준다.
    GELU는 ReLU와 비슷하지만 더 부드럽게(smooth) 꺾이는 비선형 활성함수.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Transformer Block 하나 = (LayerNorm -> Attention -> 잔차연결) + (LayerNorm -> MLP -> 잔차연결)

    "잔차연결(residual connection)"이란 x + f(x) 형태로, f(x)의 결과를 x에
    그대로 더해주는 것. 이렇게 하면:
      1) 층을 깊게 쌓아도 gradient가 사라지지(vanishing) 않고 잘 흘러간다.
      2) f(x)가 "x에 무엇을 더할지"만 학습하면 되므로 학습이 더 쉬워진다.

    LayerNorm을 attention/MLP "이전에" 적용하는 방식(Pre-LN)을 쓰는데,
    이는 원조 GPT-2 이후 표준이 된 방식으로, 학습이 더 안정적이다.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # 토큰들 사이의 관계를 반영
        x = x + self.mlp(self.ln2(x))   # 반영된 정보를 개별 토큰 단위로 가공
        return x


class GPT(nn.Module):
    """전체 모델: 임베딩 -> Transformer Block들 -> 최종 정규화 -> 어휘 전체에 대한 점수.

    이 모델이 학습하는 것은 오직 하나: "지금까지의 문자열이 주어졌을 때
    바로 다음 문자가 무엇일 확률이 높은가?" 라는 조건부 확률
    P(다음 문자 | 이전 문자들) 를 근사하는 것.

    입력은 문자를 숫자로 바꾼 정수 인덱스(idx)이며, data/prepare.py 에서 만든
    stoi(문자->숫자) 매핑을 그대로 따른다.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 토큰 임베딩: "이 문자가 무엇인가"에 대한 정보를 담은 학습 가능한 벡터표
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # 위치 임베딩: Self-Attention 자체는 순서 개념이 없기 때문에(집합처럼 다룸),
        # "이 문자가 문장에서 몇 번째 위치인가"를 별도로 알려줘야 한다.
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer Block을 n_layer개 쌓음
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)  # 마지막 출력 직전 정규화
        # 최종적으로 각 위치의 벡터(n_embd차원)를 "어휘 전체 크기"의 점수로 변환.
        # 이 점수(logits)를 softmax하면 "다음 문자 확률분포"가 된다.
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 표준편차 0.02의 정규분포로 초기화 (GPT-2 논문에서 쓰인 값).
        # 초기값이 너무 크면 학습 초반에 gradient가 폭발하고,
        # 너무 작으면 신호가 죽어 학습이 잘 안 되므로 경험적으로 정해진 값을 사용.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx     : (B, T) 정수 텐서. 각 값은 문자 하나에 대응하는 vocab 인덱스.
        targets : (B, T) 정수 텐서 또는 None.
                  주어지면 "idx의 각 위치 다음에 실제로 온 문자"가 담겨 있어야 하며
                  (즉 idx를 한 칸 뒤로 민 것), 이를 이용해 학습 loss를 계산한다.
                  생성(추론)할 때는 None으로 둔다.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence longer than block_size"

        # 위치 인덱스 [0, 1, 2, ..., T-1] 을 만들어 위치 임베딩을 조회
        pos = torch.arange(0, T, device=idx.device)

        # "이 문자가 무엇인가"(tok_emb) + "몇 번째 위치인가"(pos_emb) 를 더해서
        # 각 토큰의 초기 표현 벡터를 만든다.
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        # Transformer Block을 순서대로 통과시키며 문맥 정보를 계속 섞어 넣는다.
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size) : 각 위치에서 "다음 문자" 점수

        loss = None
        if targets is not None:
            # Cross-Entropy Loss: 모델이 예측한 확률분포(logits를 softmax한 것)와
            # 실제 정답 문자 사이의 차이를 측정. 이 값을 최소화하도록 역전파(backprop)로
            # 모든 가중치(임베딩, attention, MLP 등)를 업데이트하는 것이 "학습"이다.
            #
            # view(-1, vocab_size) 로 (B, T, vocab) -> (B*T, vocab) 로 펼치는 이유는
            # cross_entropy가 "배치 축 하나 + 클래스 축 하나" 형태를 기대하기 때문.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()  # 생성 시에는 gradient를 계산할 필요가 없으므로 메모리/속도 절약
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """자기회귀(autoregressive) 생성: 한 글자를 예측해 이어붙이고,
        그 결과를 다시 입력으로 넣어 다음 글자를 예측하는 과정을 반복한다.

        idx            : (B, T) 시작 프롬프트의 토큰 인덱스.
        max_new_tokens : 새로 생성할 글자 수.
        temperature    : 확률분포를 얼마나 "날카롭게/평평하게" 만들지 조절.
                          - 1.0 : 모델이 계산한 확률을 그대로 사용.
                          - < 1.0 (예: 0.7) : logits를 나누기 전보다 더 큰 값 차이로
                            벌어지므로 softmax 후 확률이 더 뾰족해짐 -> 가장 그럴듯한
                            선택지에 몰림 -> 더 "보수적/반복적"인 출력.
                          - > 1.0 : 확률이 평평해짐 -> 더 "다양하지만 엉뚱한" 출력.
        top_k          : 매 스텝마다 확률이 가장 높은 top_k개 후보만 남기고
                          나머지는 후보에서 제외(-inf 처리)한 뒤 그 안에서만 샘플링.
                          너무 낮은 확률의 이상한 토큰이 뽑히는 것을 방지.
        """
        for _ in range(max_new_tokens):
            # 모델은 block_size보다 긴 문맥을 볼 수 없으므로, 최근 block_size개만 잘라서 사용
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)          # (B, T, vocab_size)
            logits = logits[:, -1, :]            # 마지막 위치(=다음 글자 예측)만 사용 -> (B, vocab_size)
            logits = logits / max(temperature, 1e-5)

            if top_k is not None:
                # 상위 top_k개 값 중 가장 작은 값(v[:, -1])보다 작은 모든 logit을
                # -inf로 만들어 softmax에서 확률이 0이 되게 함
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            # argmax로 항상 1등만 뽑으면(=greedy) 매번 똑같고 단조로운 문장이 나오므로,
            # 확률분포에 따라 "무작위로" 하나를 뽑는다 (샘플링). 이것이 같은 프롬프트를
            # 넣어도 매번 다른 문장이 나오는 이유다.
            next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)  # 생성된 글자를 이어붙여 다음 루프의 입력으로 사용
        return idx
