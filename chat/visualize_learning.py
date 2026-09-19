# ============================================================================
# visualize_learning.py — "문장 하나가 어떻게 학습되는지" 전체 흐름을 보여주는
#                          교육용 애니메이션
#
# chat/model.py, train.py의 TensorFlow는 전혀 쓰지 않지만, 이번 버전은
# "가짜로 흉내만 내는" 것이 아니라 **진짜 경사하강법(역전파를 직접 numpy로
# 계산)**으로 실제 학습을 한다 — 다만 모델이 아주 작고(글자 하나 -> 글자
# 하나만 보는 아주 단순한 구조) 데이터도 문장 하나뿐이라 순식간에 끝난다.
#
# 전체 흐름은 실제 프로젝트와 정확히 같은 순서를 따른다
# (documents/01-tokenization.md, 03-training.md 참고):
#
#   1) [화면 맨 위] 문장 -> 토큰화: 문장의 각 글자를 정수로 바꾼다
#      (data/prepare.py의 stoi와 완전히 같은 방식 — 등장하는 고유 문자를
#      정렬해서 번호를 매김).
#   2) [왼쪽] 그 숫자들을 가지고 "글자 하나를 보고 바로 다음 글자를
#      예측"하도록 작은 신경망(입력=현재 글자의 원-핫 벡터, 은닉층,
#      출력=다음 글자 확률)을 학습시킨다. 매 스텝 문장 안에서 무작위로
#      (현재 글자, 다음 글자) 쌍을 하나 뽑아 학습하는 것도 실제
#      train.py의 get_batch()가 무작위로 구간을 뽑는 것과 같은 원리다.
#   3) [오른쪽 위] Loss 곡선, [오른쪽 아래] 이번 스텝에 뽑힌 글자 다음에
#      무엇이 올지에 대한 확률 분포 — 학습될수록 실제 정답에 확신을
#      갖게 되는 모습을 보여준다.
#
# 실제 model.py의 GPT는 여러 글자로 이루어진 문맥 전체(최대 block_size개)를
# 보고 예측하지만, 여기서는 흐름을 눈으로 따라가기 쉽도록 "글자 하나만
# 보고 다음 글자를 예측"하는 가장 단순한 버전(바이그램, bigram)으로
# 단순화했다. "문장 -> 숫자 -> 신경망 -> 확률 -> 학습 반복"이라는 큰
# 골격은 실제 프로젝트와 동일하다.
# ============================================================================

import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)

TOTAL_FRAMES = 400

# ---------------------------------------------------------------------------
# 1) 문장 -> 토큰화 (data/prepare.py와 완전히 같은 방식)
# ---------------------------------------------------------------------------
sentence = input("학습시킬 문장을 입력하세요 (영문 권장, 엔터만 누르면 'COMPUTER' 사용): ").strip()
if not sentence:
    sentence = "COMPUTER"
if len(sentence) < 2:
    raise SystemExit("적어도 두 글자 이상의 문장을 입력해 주세요.")

chars = sorted(set(sentence))          # 등장하는 고유 문자를 정렬 (data/prepare.py와 동일)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)

# "y = x를 한 칸 뒤로 민 것" — documents/03-training.md에서 설명한 get_batch()의
# 핵심 아이디어를 문장 전체에 그대로 적용한다.
pairs = [(stoi[sentence[i]], stoi[sentence[i + 1]]) for i in range(len(sentence) - 1)]

# 이 모델은 "바로 앞 글자 하나"만 보고 다음 글자를 예측한다 (더 긴 문맥은
# 안 봄). 그래서 같은 글자 뒤에 서로 다른 글자가 여러 번 나오면(예:
# "HELLO"의 'L' 다음엔 'L'도 오고 'O'도 옴) 모델이 아무리 학습해도 그
# 글자에 대해서만큼은 100% 확신할 수 없다 — 이건 버그가 아니라 "글자
# 하나만 보는 모델"이 가질 수밖에 없는 근본적인 한계다 (실제 GPT가 여러
# 글자로 된 문맥 전체를 보는 이유가 바로 이 한계를 극복하기 위해서다).
ambiguous = {}
for x_idx, y_idx in pairs:
    ambiguous.setdefault(x_idx, set()).add(y_idx)
ambiguous_chars = [itos[x] for x, ys in ambiguous.items() if len(ys) > 1]
if ambiguous_chars:
    print(
        f"참고: 이 문장은 {ambiguous_chars} 다음에 서로 다른 글자가 여러 번 나와서, "
        f"'바로 앞 글자 하나만 보는' 이 모델은 그 글자들에 대해서는 loss가 "
        f"완전히 0까지 내려가지 않습니다 (정상입니다 — 아래 애니메이션의 자막 참고)."
    )


def visible(ch):
    return {"\n": "↵", " ": "·", "\t": "⇥"}.get(ch, ch)


# ---------------------------------------------------------------------------
# 2) 아주 작은 신경망 — 입력(현재 글자, 원-핫) -> 은닉층 -> 출력(다음 글자 확률)
# ---------------------------------------------------------------------------
N_IN = V
N_HID = max(6, V)
N_OUT = V

W1 = rng.normal(0, 0.5, size=(N_IN, N_HID))
W2 = rng.normal(0, 0.5, size=(N_HID, N_OUT))


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def forward(x_idx):
    x = np.zeros(N_IN)
    x[x_idx] = 1.0
    h = np.tanh(x @ W1)
    probs = softmax(h @ W2)
    return x, h, probs


def train_step():
    """진짜 순전파 + 역전파 + 경사하강법 한 스텝.

    이 함수가 하는 일은 model.py/train.py가 하는 일과 원리상 완전히 같다:
      1) 무작위로 학습 데이터 한 쌍을 뽑고 (get_batch)
      2) 순전파로 예측하고 (model(x))
      3) 정답과 비교해 loss를 구하고 (cross entropy)
      4) 역전파로 기울기(gradient)를 구하고 (loss.backward())
      5) 그 반대 방향으로 가중치를 아주 조금 이동시킨다 (optimizer.step())

    문장 왼쪽부터 순서대로 학습하는 게 아니라, get_batch()처럼 매 스텝마다
    문장 안의 위치(pos)를 무작위로 하나 뽑는다 — 그래서 "지금 몇 번째
    글자를 보고 있는지"는 화면(맨 위 토큰 줄의 강조 박스)으로 직접
    확인해야 알 수 있다.
    """
    global W1, W2
    pos = rng.integers(len(pairs))
    x_idx, y_idx = pairs[pos]
    x, h, probs = forward(x_idx)
    loss = -np.log(probs[y_idx] + 1e-9)

    # --- 역전파 (softmax + cross-entropy의 기울기는 "예측확률 - 정답(원-핫)"로 깔끔하게 정리됨) ---
    dlogits = probs.copy()
    dlogits[y_idx] -= 1.0
    dW2 = np.outer(h, dlogits)
    dh = dlogits @ W2.T
    dtanh = dh * (1 - h ** 2)           # tanh의 미분: 1 - tanh(x)^2
    dW1 = np.outer(x, dtanh)

    lr = 0.6
    W1 -= lr * dW1
    W2 -= lr * dW2
    return pos, x_idx, y_idx, probs, loss


loss_history = []

# ---------------------------------------------------------------------------
# 화면 구성: 맨 위 - 토큰화 표시, 왼쪽 - 신경망, 오른쪽 위 - loss, 오른쪽 아래 - 확률분포
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 7.5))
fig.suptitle("문장 하나로 미니 언어모델 학습시키기 (진짜 경사하강법, 교육용 numpy 구현)", fontsize=12)
gs = fig.add_gridspec(4, 3, height_ratios=[0.45, 0.35, 1, 1], width_ratios=[1.1, 1, 1],
                       left=0.05, right=0.98, top=0.90, bottom=0.13)

ax_token = fig.add_subplot(gs[0, :])
ax_status = fig.add_subplot(gs[1, :])
ax_net = fig.add_subplot(gs[2:, 0])
ax_w1 = fig.add_subplot(gs[2, 1])
ax_w2 = fig.add_subplot(gs[3, 1])
ax_loss = fig.add_subplot(gs[2, 2])
ax_prob = fig.add_subplot(gs[3, 2])

# --- 1) 토큰화 패널: 문장의 각 글자와 그 숫자(인덱스)를 나란히 표시 ---
ax_token.axis("off")
ax_token.set_xlim(0, 1)
ax_token.set_ylim(0, 1)
ax_token.text(0.01, 0.85, "문장 → 토큰화 (문자를 숫자로):", fontsize=10, color="#555555", transform=ax_token.transAxes)

n_show = len(sentence)
x_positions = np.linspace(0.02, 0.98, n_show)
box_w = (x_positions[1] - x_positions[0]) * 0.6 if n_show > 1 else 0.08
for x, ch in zip(x_positions, sentence):
    ax_token.text(x, 0.5, visible(ch), fontsize=13, family="monospace", ha="center",
                  transform=ax_token.transAxes, fontweight="bold")
    ax_token.text(x, 0.15, str(stoi[ch]), fontsize=10, family="monospace", ha="center",
                  color="#4c72b0", transform=ax_token.transAxes)

# 지금 이 스텝이 문장의 "몇 번째 위치"를 학습하고 있는지 눈으로 바로 알 수
# 있도록, 입력 글자는 파란 박스, 정답(다음) 글자는 초록 박스로 감싸서
# 매 프레임 위치를 옮겨준다 (랜덤으로 위치를 뽑기 때문에 순서대로 움직이지
# 않고 문장 위를 이리저리 튀어다니는 것이 정상이다).
from matplotlib.patches import FancyBboxPatch
input_box = FancyBboxPatch((0, 0), box_w, 0.85, boxstyle="round,pad=0.01",
                            transform=ax_token.transAxes, fill=False,
                            edgecolor="#4c72b0", linewidth=2.2, zorder=5)
target_box = FancyBboxPatch((0, 0), box_w, 0.85, boxstyle="round,pad=0.01",
                             transform=ax_token.transAxes, fill=False,
                             edgecolor="#55a868", linewidth=2.2, zorder=5)
ax_token.add_patch(input_box)
ax_token.add_patch(target_box)
current_pair_text = ax_token.text(0.01, 0.0, "", fontsize=9, color="#c44e52", transform=ax_token.transAxes)

# --- 1-1) 전체 쌍(pair) 학습 현황판 ---
# "C->O가 나올 때까지 계속 고쳐서 맞춘 다음, M->P로 넘어가서 또 맞을 때까지
# 고친다" — 라고 오해하기 쉬운데, 실제로는 그렇지 않다. 위 train_step()은
# 매 스텝 7개 쌍(문장 예시가 COMPUTER면 C->O, O->M, ... 등) 중 완전
# 무작위로 하나를 뽑아 아주 조금만 가중치를 조정하고, 모든 쌍이 같은
# W1/W2를 공유하기 때문에 "한 쌍을 마스터하고 다음으로 넘어가는" 게 아니라
# "모든 쌍이 뒤섞여서 조금씩 같이 좋아지는" 방식이다. 이 패널은 W1/W2가
# 바뀔 때마다 (그 스텝에서 뽑히지 않은 쌍까지 포함해서) 7개 쌍 전부의
# "현재 정답 확률"을 동시에 보여줘서 이 사실을 직접 확인시켜준다.
ax_status.axis("off")
ax_status.set_xlim(0, 1)
ax_status.set_ylim(0, 1)
ax_status.text(0.01, 0.9, "전체 쌍 학습 현황 (한 쌍씩 순서대로 완성되는 게 아니라, 전부 같이 조금씩 좋아짐):",
                fontsize=8.5, color="#555555", transform=ax_status.transAxes)
n_pairs = len(pairs)
status_x = np.linspace(0.04, 0.98, n_pairs)
pair_status_texts = []
for x, (x_idx, y_idx) in zip(status_x, pairs):
    t = ax_status.text(
        x, 0.35, "", ha="center", va="center", fontsize=8, family="monospace",
        transform=ax_status.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eeeeee", edgecolor="#999999", linewidth=0.8),
    )
    pair_status_texts.append(t)


def _status_color(p):
    if p > 0.7:
        return "#c8e6c9"   # 초록: 거의 맞춤
    if p > 0.3:
        return "#fff3cd"   # 노랑: 애매함
    return "#f8d7da"       # 빨강: 아직 많이 틀림


def update_pair_status():
    """지금 뽑히지 않은 쌍까지 포함해서 7개 쌍 전부를 현재 가중치로 다시
    순전파해서 정답 확률을 계산한다 — '이번 스텝에 안 뽑힌 쌍도 가중치
    공유 때문에 같이 바뀐다'는 것을 보여주기 위함."""
    for t, (x_idx, y_idx) in zip(pair_status_texts, pairs):
        p_correct = forward(x_idx)[2][y_idx]
        t.set_text(f"{visible(itos[x_idx])}→{visible(itos[y_idx])}\n{p_correct*100:.0f}%")
        t.get_bbox_patch().set_facecolor(_status_color(p_correct))


update_pair_status()

# --- 2) 신경망 패널 ---
pos_in = [(0.0, y) for y in np.linspace(-1, 1, N_IN)]
pos_hid = [(1.0, y) for y in np.linspace(-1, 1, N_HID)]
pos_out = [(2.0, y) for y in np.linspace(-1, 1, N_OUT)]

ax_net.set_xlim(-0.6, 2.6)
ax_net.set_ylim(-1.3, 1.3)
ax_net.axis("off")
ax_net.set_title("신경망 가중치 (선 두께/색 = 가중치 크기·부호)")
ax_net.text(0.0, 1.15, "입력\n(현재 글자)", ha="center", fontsize=9)
ax_net.text(1.0, 1.15, "은닉층", ha="center", fontsize=9)
ax_net.text(2.0, 1.15, "출력\n(다음 글자)", ha="center", fontsize=9)

for (x, y), ch in zip(pos_in, chars):
    ax_net.text(x - 0.15, y, visible(ch), ha="right", va="center", fontsize=10)
for (x, y), ch in zip(pos_out, chars):
    ax_net.text(x + 0.15, y, visible(ch), ha="left", va="center", fontsize=10)

edge_lines_1 = []
for i, p_in in enumerate(pos_in):
    for j, p_hid in enumerate(pos_hid):
        line, = ax_net.plot([p_in[0], p_hid[0]], [p_in[1], p_hid[1]], lw=1, alpha=0.5, zorder=1)
        edge_lines_1.append((line, i, j))
edge_lines_2 = []
for j, p_hid in enumerate(pos_hid):
    for k, p_out in enumerate(pos_out):
        line, = ax_net.plot([p_hid[0], p_out[0]], [p_hid[1], p_out[1]], lw=1, alpha=0.5, zorder=1)
        edge_lines_2.append((line, j, k))

in_scatter = ax_net.scatter(*zip(*pos_in), s=280, c="#4c72b0", zorder=2)
ax_net.scatter(*zip(*pos_hid), s=200, c="#999999", zorder=2)
out_scatter = ax_net.scatter(*zip(*pos_out), s=280, c="#dd8452", zorder=2)
step_text = ax_net.text(0.02, 0.02, "", transform=ax_net.transAxes, va="bottom", fontsize=10)

# --- 2-1) 가중치 실제 숫자값 (히트맵) ---
# "히트맵"은 그냥 숫자를 색으로 칠한 표다 (일기예보의 온도 지도와 같은
# 원리) — 숫자가 크게 +면 빨강, 크게 -면 파랑, 0에 가까우면 흰색.
# 글자와 글자 사이의 "거리/격차"를 재는 게 아니라, 신경망이 학습하며
# 저장해두는 내부 숫자(가중치) 그 자체다. 줄(행)과 칸(열)에 실제로 어떤
# 글자/은닉 뉴런인지 라벨을 붙여서, "C라는 글자가 입력됐을 때 h3번
# 은닉 뉴런에 주는 영향"처럼 정확히 어느 값인지 바로 읽을 수 있게 한다.
# 은닉 뉴런(h0, h1, ...)은 특정 의미가 있는 게 아니라 신경망이 계산 도중에
# 쓰는 "이름 없는 중간 저장 칸"이라고 생각하면 된다.
_show_text = (N_IN * N_HID <= 120)
hidden_labels = [f"h{i}" for i in range(N_HID)]


def _make_heatmap(ax, mat, title, xlabel, ylabel, xticklabels, yticklabels):
    im = ax.imshow(mat, cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, fontsize=7)
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=7)
    texts = None
    if _show_text:
        texts = [[ax.text(c, r, "", ha="center", va="center", fontsize=6)
                  for c in range(mat.shape[1])] for r in range(mat.shape[0])]
    return im, texts


im_w1, texts_w1 = _make_heatmap(
    ax_w1, W1, "W1 실제 값 (입력 글자 → 은닉 뉴런)", "은닉 뉴런", "입력 글자",
    xticklabels=hidden_labels, yticklabels=[visible(c) for c in chars],
)
im_w2, texts_w2 = _make_heatmap(
    ax_w2, W2, "W2 실제 값 (은닉 뉴런 → 출력 글자)", "출력 글자", "은닉 뉴런",
    xticklabels=[visible(c) for c in chars], yticklabels=hidden_labels,
)
fig.colorbar(im_w1, ax=ax_w1, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
fig.colorbar(im_w2, ax=ax_w2, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)


def _update_heatmap(im, texts, mat):
    im.set_data(mat)
    if texts is not None:
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                texts[r][c].set_text(f"{mat[r, c]:.1f}")

# --- 3) Loss 곡선 ---
ax_loss.set_title("Loss (모델이 얼마나 틀리고 있는지)")
ax_loss.set_xlim(0, TOTAL_FRAMES)
ax_loss.set_ylim(0, max(3, np.log(V) * 2))
ax_loss.set_xlabel("학습 스텝")
loss_line, = ax_loss.plot([], [], color="#c44e52")

# --- 4) 다음 글자 확률분포 ---
ax_prob.set_title("이번 스텝 입력 글자 다음에 올 확률 (초록 = 정답)")
ax_prob.set_ylim(0, 1)
bars = ax_prob.bar([visible(c) for c in chars], [1 / V] * V, color="#dd8452")


def w_to_style(w):
    color = "#4c72b0" if w >= 0 else "#c44e52"
    lw = float(np.clip(abs(w) * 2.2, 0.3, 6.0))
    alpha = float(np.clip(abs(w) / 2.5, 0.15, 1.0))
    return color, lw, alpha


def print_final_weights():
    np.set_printoptions(precision=3, suppress=True)
    print("\n=== 학습 완료: 이게 마지막(최종) 가중치 값입니다 ===")
    print(f"W1 (입력 {N_IN} -> 은닉 {N_HID}):")
    print(W1)
    print(f"W2 (은닉 {N_HID} -> 출력 {N_OUT}):")
    print(W2)


def do_train_step(frame):
    """실제 학습 한 스텝을 수행하고 화면의 모든 패널을 갱신한다."""
    pos, x_idx, y_idx, probs, loss = train_step()
    loss_history.append(loss)

    input_box.set_bounds(x_positions[pos] - box_w / 2, 0.02, box_w, 0.85)
    target_box.set_bounds(x_positions[pos + 1] - box_w / 2, 0.02, box_w, 0.85)

    for line, i, j in edge_lines_1:
        color, lw, alpha = w_to_style(W1[i, j])
        line.set_color(color); line.set_linewidth(lw); line.set_alpha(alpha)
    for line, j, k in edge_lines_2:
        color, lw, alpha = w_to_style(W2[j, k])
        line.set_color(color); line.set_linewidth(lw); line.set_alpha(alpha)

    _update_heatmap(im_w1, texts_w1, W1)
    _update_heatmap(im_w2, texts_w2, W2)
    update_pair_status()

    loss_line.set_data(range(len(loss_history)), loss_history)

    for bar, p in zip(bars, probs):
        bar.set_height(p)
        bar.set_color("#999999")
    bars[y_idx].set_color("#55a868")   # 정답
    bars[x_idx].set_edgecolor("#333333")
    bars[x_idx].set_linewidth(1.5)

    step = len(loss_history)
    step_text.set_text(f"학습 스텝 {step}/{TOTAL_FRAMES}\nloss = {loss:.3f}")
    current_pair_text.set_text(
        f"문장 내 위치 {pos}번째: '{visible(itos[x_idx])}'({x_idx}) → '{visible(itos[y_idx])}'({y_idx})   "
        f"모델 예측 확률(정답) = {probs[y_idx]*100:.1f}%   (위 파란 박스=입력, 초록 박스=정답)"
    )

    if step >= TOTAL_FRAMES:
        state["finished"] = True
        state["running"] = False
        step_text.set_text(step_text.get_text() + "\n=== 학습 완료 ===")
        print_final_weights()


# ---------------------------------------------------------------------------
# 재생 / 멈춤 / 다음 스텝 컨트롤
#
# FuncAnimation은 45ms마다 계속 update()를 호출하지만, 실제로 가중치를
# 갱신할지는 여기 state 값으로 결정한다 — "멈춤"을 누르면 화면 갱신 없이
# 가만히 있고, "다음 스텝"을 누르면 멈춘 상태에서 딱 한 스텝만 학습을
# 진행한다. "학습 완료" 후에는 재생을 눌러도 더 이상 진행되지 않고, 그때
# 화면(과 콘솔)에 보이는 값이 곧 최종 가중치다.
# ---------------------------------------------------------------------------
state = {"running": True, "step_once": False, "finished": False}


def update(frame):
    if state["finished"]:
        return []
    if not (state["running"] or state["step_once"]):
        return []
    state["step_once"] = False
    do_train_step(frame)
    return (list(bars) + [loss_line, step_text, current_pair_text, input_box, target_box]
            + [im_w1, im_w2] + pair_status_texts)


def on_play(event):
    if not state["finished"]:
        state["running"] = True


def on_pause(event):
    state["running"] = False


def on_step(event):
    if not state["finished"]:
        state["running"] = False
        state["step_once"] = True


from matplotlib.widgets import Button

ax_play = fig.add_axes([0.30, 0.02, 0.12, 0.05])
ax_pause = fig.add_axes([0.44, 0.02, 0.12, 0.05])
ax_step = fig.add_axes([0.58, 0.02, 0.14, 0.05])
btn_play = Button(ax_play, "재생 ▶")
btn_pause = Button(ax_pause, "멈춤 ⏸")
btn_step = Button(ax_step, "다음 스텝 ⏭")
btn_play.on_clicked(on_play)
btn_pause.on_clicked(on_pause)
btn_step.on_clicked(on_step)

ani = animation.FuncAnimation(fig, update, frames=itertools.count(), interval=45, blit=False, repeat=False,
                               cache_frame_data=False)
plt.show()
