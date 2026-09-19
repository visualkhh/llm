# ============================================================================
# visualize_generation.py — 내가 입력한 글로 시작해서, 실제 학습된 모델이
#                            "한 글자씩" 만들어가는 과정을 눈으로 보기
#
# visualize_learning.py와 다른 점: 이 파일은 "학습 과정을 흉내낸 가짜"가
# 아니라, chat/train.py로 실제로 학습시킨 진짜 체크포인트(checkpoints/ckpt.pt)를
# 불러와 **진짜로 추론(generate)**한다. 다만 "학습"은 안 하고 이미 학습이
# 끝난 모델을 가지고 "생성"만 한다 — 그래서 몇 초 안에 실행되고, 매번 결과가
# 조금씩 달라질 수 있다(확률적으로 샘플링하기 때문. model.py의 generate()
# 함수, documents/04-generation.md 참고).
#
# 화면 구성:
#   - 위: 지금까지 만들어진 글자들 (내가 입력한 부분은 회색, 모델이 새로
#     만든 부분은 검정으로 구분)
#   - 아래: "바로 다음 글자로 무엇이 올지"에 대한 모델의 확률 분포
#     (상위 8개만 막대그래프로) — 매 프레임(=매 글자)마다 갱신되고,
#     실제로 뽑힌 글자는 초록색으로 표시된다.
#
# model.py의 generate() 함수는 이 과정을 한 번에 다 돌려버리는데, 여기서는
# "한 글자 예측 -> 뽑기 -> 이어붙이기"를 한 프레임에 하나씩 직접 수행해서
# 그 반복 구조(자기회귀, autoregressive) 자체를 눈으로 보여준다
# (documents/04-generation.md의 자기회귀 설명과 정확히 같은 과정).
# ============================================================================

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import GPT

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def load_model(ckpt_name):
    ckpt = torch.load(os.path.join(CKPT_DIR, f"{ckpt_name}.pt"), map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["meta"]


def visible(ch):
    """공백/줄바꿈처럼 막대그래프 라벨로 그대로 쓰면 안 보이는 문자를
    눈에 보이는 기호로 바꿔준다 (실제 생성 결과에는 영향 없음, 화면 표시용).
    """
    return {"\n": "↵", " ": "␣", "\t": "⇥"}.get(ch, ch)


def wrap_text(text, width=70):
    """긴 글을 화면 폭에 맞춰 여러 줄로 자른다 (실제 줄바꿈 문자는 그대로 살림)."""
    lines = []
    for raw_line in text.split("\n"):
        while len(raw_line) > width:
            lines.append(raw_line[:width])
            raw_line = raw_line[width:]
        lines.append(raw_line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_name", type=str, default="ckpt", help="checkpoints/{ckpt_name}.pt 를 불러온다")
    parser.add_argument("--prompt", type=str, default=None,
                         help="시작 문장. 생략하면 실행할 때 터미널에서 물어본다.")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="새로 생성할 글자 수")
    parser.add_argument("--temperature", type=float, default=0.8,
                         help="높을수록 다양하지만 엉뚱해지고, 낮을수록 보수적/반복적이 됨 "
                              "(documents/04-generation.md 참고)")
    parser.add_argument("--top_k", type=int, default=50, help="매 스텝 후보를 상위 몇 개로 제한할지")
    parser.add_argument("--interval_ms", type=int, default=220, help="한 글자가 나오는 데 걸리는 화면 시간(ms)")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(CKPT_DIR, f"{args.ckpt_name}.pt")):
        raise SystemExit(
            f"checkpoints/{args.ckpt_name}.pt 를 찾을 수 없습니다. "
            f"먼저 chat/train.py로 모델을 학습시켜 주세요 (documents/03-training.md 참고)."
        )

    model, meta = load_model(args.ckpt_name)
    stoi, itos = meta["stoi"], meta["itos"]

    prompt = args.prompt
    if prompt is None:
        prompt = input("시작 문장을 입력하세요 (엔터만 누르면 'First Citizen:' 사용): ").strip()
        if not prompt:
            prompt = "First Citizen:"

    # 학습 때 없던 문자는 조용히 무시 (chat.py와 동일한 처리)
    prompt_ids = [stoi[c] for c in prompt if c in stoi]
    if not prompt_ids:
        raise SystemExit("입력한 문장에 모델이 아는 문자가 하나도 없습니다. 다른 문장으로 시도해 보세요.")

    idx = torch.tensor([prompt_ids], dtype=torch.long)
    prompt_len = len(prompt_ids)

    @torch.no_grad()
    def gen_step():
        """model.py의 generate() 안에서 한 글자만큼만 떼어낸 것과 동일한 계산.
        여기서는 그 결과(선택된 글자, 후보 확률 분포)를 애니메이션에 쓰기
        위해 반환값으로 노출한다.
        """
        nonlocal idx
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(args.temperature, 1e-5)
        if args.top_k is not None:
            v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)[0]
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        idx = torch.cat([idx, torch.tensor([[next_id]])], dim=1)

        k = min(8, probs.shape[0])
        topk_probs, topk_ids = torch.topk(probs, k)
        candidates = [(visible(itos[int(i)]), float(p)) for p, i in zip(topk_probs, topk_ids)]
        return itos[next_id], candidates

    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(9, 6))
    fig.suptitle("학습된 모델이 한 글자씩 문장을 생성하는 과정", fontsize=12)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1])
    ax_text = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    ax_text.axis("off")
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    # 내가 입력한 부분(프롬프트)은 위쪽에 회색으로 고정 표시하고, 모델이
    # 새로 만들어내는 부분만 그 아래에 검정색으로 한 글자씩 늘려간다 —
    # "어디까지가 내가 준 것이고 어디부터가 모델이 만든 것인지"를 명확히
    # 구분하기 위함.
    #
    # 한글 라벨("입력:", "생성:")과 실제 생성 내용(영어 텍스트)의 폰트를
    # 분리한 이유: family="monospace"로 지정하면 그 폰트에 한글 글리프가
    # 없어서 라벨이 네모(□)로 깨진다. 그래서 한글 라벨은 기본 폰트
    # (AppleGothic, 한글 지원)로 두고, 생성된 영어 텍스트만 별도 텍스트
    # 객체로 떼어내 monospace를 적용한다.
    ax_text.text(0.02, 0.97, "입력:", transform=ax_text.transAxes, va="top", ha="left",
                 fontsize=11, color="#777777")
    ax_text.text(
        0.10, 0.97, wrap_text(prompt), transform=ax_text.transAxes, va="top", ha="left",
        fontsize=11, family="monospace", color="#777777",
    )
    ax_text.text(0.02, 0.72, "생성:", transform=ax_text.transAxes, va="top", ha="left",
                 fontsize=11, color="#777777")
    gen_display = ax_text.text(
        0.02, 0.62, "", transform=ax_text.transAxes, va="top", ha="left",
        fontsize=12, family="monospace", color="black",
    )
    step_label = ax_text.text(0.02, 0.03, "", transform=ax_text.transAxes, fontsize=9, color="#888888")

    ax_bar.set_ylim(0, 1)
    ax_bar.set_title("바로 다음 글자 확률 (상위 8개, 초록 = 실제로 뽑힌 글자)")
    bar_container = ax_bar.bar(range(8), [0] * 8, color="#999999")

    generated_so_far = {"value": ""}

    def update(frame):
        chosen, candidates = gen_step()

        generated_so_far["value"] += chosen
        gen_display.set_text(wrap_text(generated_so_far["value"]))
        step_label.set_text(f"생성된 글자 수: {frame + 1}/{args.max_new_tokens}   (온도={args.temperature}, top_k={args.top_k})")

        labels = [c for c, _ in candidates] + [""] * (8 - len(candidates))
        values = [p for _, p in candidates] + [0.0] * (8 - len(candidates))
        for bar, v, lab in zip(bar_container, values, labels):
            bar.set_height(v)
            bar.set_color("#55a868" if lab == visible(chosen) else "#dd8452")
        ax_bar.set_xticks(range(8))
        ax_bar.set_xticklabels(labels)

        return list(bar_container) + [gen_display, step_label]

    ani = animation.FuncAnimation(
        fig, update, frames=args.max_new_tokens, interval=args.interval_ms, blit=False, repeat=False
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
