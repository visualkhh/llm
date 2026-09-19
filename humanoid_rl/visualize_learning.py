# ============================================================================
# visualize_learning.py — "로봇이 강화학습으로 점점 잘 걷게 되는 모습"을
#                          보여주는 교육용 애니메이션
#
# ⚠️ 중요: 이 파일은 실제로 MuJoCo도, TensorFlow도, envs/biped_env.py나
# rl/ppo.py의 진짜 물리 시뮬레이션/신경망도 전혀 쓰지 않는다. "강화학습으로
# 정책이 점점 좋아진다는 것이 어떤 느낌인가"를 2D 평면 위에서 단순화해서
# 보여주기 위해, 그럴듯한 움직임을 직접 만들어낸(시뮬레이션) 것뿐이다.
#
# 그래도 화면에 나오는 요소들은 실제 rl/ppo.py의 학습 과정과 개념적으로
# 정확히 대응된다:
#
#   1) [왼쪽] 로봇(점)이 목표(★)를 향해 장애물(■)을 피해 이동하는 궤적.
#      에피소드가 진행될수록(=더 많이 학습할수록) 궤적이 "제멋대로
#      비틀거리다 금방 멈춤"에서 "장애물을 피해 목표로 곧장 향함"으로
#      바뀌는 걸 보여준다. 이게 documents/00-overview.md에서 말하는
#      "쓰러지지 않고 버티기 -> 목표로 이동하기" 학습 순서를 흉내낸 것.
#   2) [오른쪽 위] 정책 신경망의 가중치 그림 (chat/visualize_learning.py와
#      같은 방식) — 관측값(observation) -> 행동(action)으로 이어지는
#      연결의 가중치가 학습될수록 뚜렷해지는 모습.
#   3) [오른쪽 아래] 에피소드별 보상(reward) 곡선 — 학습이 될수록 점점
#      올라간다 (rl/ppo.py의 실제 학습 로그에서 봤던 "ep_return"이 바로 이것).
#
# 실제로는 이 "점점 나아지는 행동"을 PPO 알고리즘(수집한 경험으로 정책
# 신경망을 조금씩 업데이트하는 것을 수백~수천 번 반복)이 만들어낸다
# (documents/03-ppo-algorithm.md, 04-training-loop.md 참고). 여기서는 그
# 결과만 그럴듯하게 흉내낸다.
# ============================================================================

import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(1)

# --- 세계 설정 (envs/biped_env.py의 기본 맵과 비슷한 느낌으로) ---
START = np.array([0.0, 0.0])
GOAL = np.array([6.0, 1.0])
OBSTACLES = [
    (2.0, 0.6, 0.35),   # (x, y, half_size) — 정사각형 장애물로 단순화
    (3.6, -0.6, 0.35),
]

N_EPISODES = 40          # "학습 에피소드" 수 (실제로는 수백~수천 번 필요하지만 여기선 40개로 압축)
STEPS_PER_EPISODE = 60   # 한 에피소드를 몇 프레임으로 그릴지
LIDAR_MAX_RANGE = 6.0    # 실제 프로젝트의 LIDAR_MAX_RANGE(documents/08-lidar-sensor.md)와 같은 개념

# 관측값/행동에 붙일 이름표. 실제 envs/biped_env.py는 45차원 관측값(관절
# 10개×2 + 기울기 + 속도 6개 + 목표 3개 + 라이다 15개)과 10차원 행동(다리
# 관절 10개)을 쓰지만, 그림에 다 넣으면 너무 빽빽해지므로 그중 "지금
# 이동 결정과 직접적으로 관련된" 6개/4개만 뽑아 보여준다.
OBS_LABELS = ["목표거리", "목표각도", "라이다-전", "라이다-좌", "라이다-우", "기울기"]
ACT_LABELS = ["L_hip_pitch", "L_knee", "R_hip_pitch", "R_knee"]


def skill_level(ep):
    """에피소드 번호 -> '정책이 얼마나 능숙해졌는지'를 0~1로. 실제 학습
    곡선은 이렇게 매끈하지 않고 들쭉날쭉하지만, 큰 추세는 이런 S자
    곡선(처음엔 느리게, 중반에 빠르게, 후반에 다시 완만하게)을 따르는
    경우가 많다 (documents/04-training-loop.md의 학습 로그 해석 참고).
    """
    x = (ep / (N_EPISODES - 1)) * 10 - 5  # -5 ~ 5 범위로 매핑
    return 1 / (1 + np.exp(-x))  # 시그모이드


def lidar_like(pos, heading, obstacles, max_range=LIDAR_MAX_RANGE):
    """실제 envs/biped_env.py의 _lidar_scan()과 같은 개념을 2D로 단순화한 것.
    heading(정면 방향) 기준 전방/좌측/우측 각 방향에서 가장 가까운 장애물까지의
    거리를 구한다. 아무것도 없으면 max_range(=아주 멀다)를 반환한다.
    """
    h_angle = np.arctan2(heading[1], heading[0])
    sectors = {"front": (h_angle - 0.35, h_angle + 0.35),
               "left": (h_angle + 0.35, h_angle + 1.4),
               "right": (h_angle - 1.4, h_angle - 0.35)}
    result = {}
    for name, (lo, hi) in sectors.items():
        best = max_range
        for ox, oy, hs in obstacles:
            d = np.array([ox, oy]) - pos
            dist = np.linalg.norm(d)
            ang = np.arctan2(d[1], d[0])
            # 각도 차이를 -pi~pi 범위로 정규화해서 이 장애물이 그 방향
            # 부채꼴 안에 있는지 확인
            rel = (ang - lo) % (2 * np.pi)
            span = (hi - lo) % (2 * np.pi)
            if rel <= span:
                best = min(best, max(dist - hs, 0.05))
        result[name] = best
    return result


def simulate_episode(skill, rng):
    """단순화된 '한 에피소드' 궤적을 만든다.

    - skill=0(학습 초반): 목표 방향으로 가려는 의지가 거의 없고, 무작위로
      비틀거리다 금방 "넘어져서"(=에피소드 조기 종료) 멈춘다.
    - skill=1(학습 후반): 목표를 향해 곧장 이동하면서, 장애물을 만나면
      옆으로 피해가는 경로를 만든다.

    path와 함께, 매 스텝의 "관측값"과 "행동"에 해당하는 실제 숫자도 함께
    기록해서 반환한다 — 진짜 envs/biped_env.py의 45차원 관측값/10차원
    행동을 그대로 흉내낼 순 없지만, 같은 성격(목표까지 거리/방향, 라이다
    거리, 관절 제어값)의 숫자를 만들어 신경망 그림 옆에 표시하기 위함이다.
    """
    pos = START.copy()
    path = [pos.copy()]
    obs_log = []
    act_log = []
    heading = None
    fell = False

    max_steps = int(10 + skill * (STEPS_PER_EPISODE - 10))  # 능숙할수록 안 넘어지고 오래/많이 감
    step_size = 0.15 + 0.05 * skill

    for t in range(max_steps):
        to_goal = GOAL - pos
        dist = np.linalg.norm(to_goal) + 1e-6
        goal_dir = to_goal / dist
        cur_heading = heading if heading is not None else goal_dir
        # "목표각도": 지금 향하고 있는 방향 대비 목표가 얼마나 좌/우로
        # 틀어져 있는지 (라디안, 실제 프로젝트의 로봇-기준 상대좌표와 같은 개념)
        goal_angle = np.arctan2(
            cur_heading[0] * goal_dir[1] - cur_heading[1] * goal_dir[0],
            cur_heading[0] * goal_dir[0] + cur_heading[1] * goal_dir[1],
        )
        lidar = lidar_like(pos, cur_heading, OBSTACLES)

        # 장애물 회피: 가까운 장애물에서 멀어지는 방향으로 밀어내는 힘
        avoid = np.zeros(2)
        for ox, oy, hs in OBSTACLES:
            o = np.array([ox, oy])
            d = pos - o
            dist_o = np.linalg.norm(d) + 1e-6
            if dist_o < hs + 0.6:
                avoid += (d / dist_o) * (hs + 0.6 - dist_o) * 2.0

        # skill이 높을수록 "목표로 가려는 의지 + 장애물 회피"가 강해지고,
        # 무작위성(노이즈)은 줄어든다 — 이게 이 시뮬레이션에서 "학습됨"을
        # 표현하는 핵심 트릭이다.
        purpose = 0.15 + 0.85 * skill
        wobble = 0.35 * (1 - skill) + 0.03  # 몸이 얼마나 흔들리는지 (기울기 계산에도 재사용)
        noise = rng.normal(0, wobble, size=2)
        move = purpose * (goal_dir * step_size) + purpose * avoid * 0.3 + noise

        tilt = float(np.clip(1.0 - np.linalg.norm(noise) / 0.5, 0.0, 1.0))

        # "행동"(모터 제어값, -1~1): 앞으로 나아가려는 성분을 hip_pitch에,
        # 장애물 회피/방향 전환 성분을 knee에 대응시켜 -1~1로 눌러 담는다.
        # 실제 로봇의 관절 매핑과 정확히 같지는 않지만, "전진 의지가
        # 강할수록 hip_pitch가 커지고, 회피가 필요할수록 knee가 커진다"는
        # 방향성은 실제와 같은 개념이다.
        fwd = float(np.tanh((purpose * step_size + move[0] * 0.3)))
        turn = float(np.tanh(avoid[1] * purpose + noise[1]))
        l_hip = float(np.clip(fwd + 0.05 * np.sin(t * 0.9), -1, 1))
        r_hip = float(np.clip(fwd - 0.05 * np.sin(t * 0.9), -1, 1))  # 좌우 다리가 번갈아 움직이는 느낌
        l_knee = float(np.clip(-abs(turn) - 0.2 * skill, -1, 1))
        r_knee = float(np.clip(-abs(turn) - 0.2 * skill, -1, 1))

        obs_log.append([dist, goal_angle, lidar["front"], lidar["left"], lidar["right"], tilt])
        act_log.append([l_hip, l_knee, r_hip, r_knee])

        pos_new = pos + move
        heading = (pos_new - pos) / (np.linalg.norm(pos_new - pos) + 1e-6)
        pos = pos_new
        path.append(pos.copy())

        # 장애물에 정통으로 부딪히면(=회피 실패) 넘어진 것으로 처리, 에피소드 종료
        for ox, oy, hs in OBSTACLES:
            if abs(pos[0] - ox) < hs * 0.6 and abs(pos[1] - oy) < hs * 0.6:
                fell = True
                break
        if fell:
            break
        # 학습 초반에는 그냥 "중심을 못 잡고 쓰러지는" 상황도 무작위로 발생
        if rng.random() < 0.05 * (1 - skill):
            fell = True
            break
        if dist < 0.3:
            break  # 목표 도달

    reached = np.linalg.norm(pos - GOAL) < 0.3
    reward = -np.linalg.norm(pos - GOAL) * 2 + (15 if reached else 0) - (5 if fell else 0)
    return np.array(path), np.array(obs_log), np.array(act_log), reached, fell, reward


# --- 정책 신경망 시각화용 (chat/visualize_learning.py와 동일한 방식) ---
N_IN, N_HID, N_OUT = 6, 6, 4  # 관측(단순화) -> 은닉 -> 행동(단순화) 개수
pos_in = [(0.0, y) for y in np.linspace(-1, 1, N_IN)]
pos_hid = [(1.0, y) for y in np.linspace(-1, 1, N_HID)]
pos_out = [(2.0, y) for y in np.linspace(-1, 1, N_OUT)]

target_w1 = rng.normal(0, 1, size=(N_IN, N_HID))
target_w2 = rng.normal(0, 1, size=(N_HID, N_OUT))
w1 = rng.normal(0, 1, size=(N_IN, N_HID))
w2 = rng.normal(0, 1, size=(N_HID, N_OUT))

reward_history = []

# ---------------------------------------------------------------------------
# 화면 구성 — chat/visualize_learning.py와 같은 구도:
#   맨 위: 전체 에피소드 진행 상황을 한눈에 보는 현황판 (초록=도달, 빨강=넘어짐,
#          회색=시간초과, 흰색=아직 안 겪음)
#   왼쪽: 로봇 이동 경로
#   가운데: 정책 신경망 그림 + 가중치 실제 숫자(히트맵)
#   오른쪽: 보상(reward) 곡선
#   맨 아래: 재생 / 멈춤 / 다음 스텝 버튼
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 7.5))
fig.suptitle("강화학습으로 로봇이 점점 잘 걷게 되는 모습 (교육용 시뮬레이션 — 실제 학습 아님)", fontsize=11)
gs = fig.add_gridspec(3, 3, height_ratios=[0.35, 1, 1], width_ratios=[1.3, 1, 1],
                       left=0.05, right=0.98, top=0.90, bottom=0.13)

ax_history = fig.add_subplot(gs[0, :])
ax_world = fig.add_subplot(gs[1:, 0])
ax_net = fig.add_subplot(gs[1, 1])
ax_reward = fig.add_subplot(gs[2, 1])
ax_w1 = fig.add_subplot(gs[1, 2])
ax_w2 = fig.add_subplot(gs[2, 2])

# --- 0) 전체 에피소드 진행 상황 현황판 ---
# "에피소드 하나씩 순서대로 완벽해지는 게 아니라, 들쭉날쭉하지만 전체
# 추세가 점점 좋아진다"는 강화학습의 실제 느낌을 처음부터 끝까지 한눈에
# 보여주기 위한 것 — chat 쪽의 "전체 쌍 학습 현황판"과 같은 목적이다.
ax_history.axis("off")
ax_history.set_xlim(0, 1)
ax_history.set_ylim(0, 1)
ax_history.text(0.01, 0.85, "전체 에피소드 진행 상황 (초록=목표 도달, 빨강=넘어짐, 회색=시간 초과):",
                 fontsize=8.5, color="#555555", transform=ax_history.transAxes)
hist_x = np.linspace(0.01, 0.99, N_EPISODES)
hist_w = (hist_x[1] - hist_x[0]) * 0.7 if N_EPISODES > 1 else 0.1
history_boxes = []
for x in hist_x:
    box = Rectangle((x - hist_w / 2, 0.05), hist_w, 0.55, transform=ax_history.transAxes,
                     facecolor="white", edgecolor="#bbbbbb", linewidth=0.8)
    ax_history.add_patch(box)
    history_boxes.append(box)

ax_world.set_xlim(-0.8, 7.5)
ax_world.set_ylim(-2.2, 2.2)
ax_world.set_aspect("equal")
ax_world.set_title("로봇의 이동 경로 (에피소드가 진행될수록 점점 능숙해짐)")

for ox, oy, hs in OBSTACLES:
    ax_world.add_patch(Rectangle((ox - hs, oy - hs), 2 * hs, 2 * hs, color="#c44e52", alpha=0.8))
ax_world.plot(*GOAL, marker="*", markersize=22, color="#55a868", zorder=3)
ax_world.text(GOAL[0], GOAL[1] + 0.35, "목표", ha="center", fontsize=9)

path_line, = ax_world.plot([], [], "-", color="#4c72b0", lw=2, zorder=2)
robot_dot, = ax_world.plot([], [], "o", color="#4c72b0", markersize=12, zorder=3)
episode_text = ax_world.text(0.02, 0.97, "", transform=ax_world.transAxes, va="top", fontsize=10)
result_text = ax_world.text(0.02, 0.03, "", transform=ax_world.transAxes, va="bottom", fontsize=10)

ax_net.set_xlim(-1.5, 3.7)
ax_net.set_ylim(-1.3, 1.3)
ax_net.axis("off")
ax_net.set_title("정책 신경망 — 관측값(입력) → 행동(모터 제어값, 출력)")
ax_net.text(0.0, 1.15, "관측값", ha="center", fontsize=9)
ax_net.text(1.0, 1.15, "은닉층", ha="center", fontsize=9)
ax_net.text(2.0, 1.15, "행동", ha="center", fontsize=9)

# 각 입력/출력 노드 옆에 "이름표: 실제 값" 텍스트를 붙인다. 값 자체는
# update()에서 매 프레임 갱신된다 (아래 in_value_texts / out_value_texts).
in_value_texts = []
for (x, y), label in zip(pos_in, OBS_LABELS):
    ax_net.text(x - 0.08, y, label, ha="right", va="center", fontsize=8, color="#555555")
    t = ax_net.text(x - 0.08, y - 0.13, "", ha="right", va="center", fontsize=8, color="#4c72b0", fontweight="bold")
    in_value_texts.append(t)

out_value_texts = []
for (x, y), label in zip(pos_out, ACT_LABELS):
    ax_net.text(x + 0.08, y, label, ha="left", va="center", fontsize=8, color="#555555")
    t = ax_net.text(x + 0.08, y - 0.13, "", ha="left", va="center", fontsize=8, color="#dd8452", fontweight="bold")
    out_value_texts.append(t)

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
ax_net.scatter(*zip(*pos_in), s=200, c="#4c72b0", zorder=2)
ax_net.scatter(*zip(*pos_hid), s=200, c="#999999", zorder=2)
ax_net.scatter(*zip(*pos_out), s=200, c="#dd8452", zorder=2)

ax_reward.set_title("에피소드별 보상 (ep_return)")
ax_reward.set_xlim(0, N_EPISODES)
ax_reward.set_ylim(-15, 20)
ax_reward.set_xlabel("에피소드")
ax_reward.axhline(0, color="gray", lw=0.5)
reward_line, = ax_reward.plot([], [], "o-", color="#c44e52", markersize=3)

# --- 가중치 실제 숫자값 (히트맵) — chat/visualize_learning.py와 동일한 목적 ---
# "히트맵"은 그냥 숫자를 색으로 칠한 표다 (일기예보 온도 지도와 같은 원리)
# — 숫자가 크게 +면 빨강, 크게 -면 파랑, 0에 가까우면 흰색. 위 신경망
# 그림의 선 두께/색은 "느낌"만 보여줄 뿐이고, 실제 값이 어떻게 바뀌는지는
# 여기 숫자로 직접 확인한다. 줄(행)과 칸(열)에 실제로 어떤 관측값/행동인지
# 라벨을 붙여서 "목표거리가 h3번 은닉 뉴런에 주는 영향"처럼 정확히 어느
# 값인지 바로 읽을 수 있게 한다 (은닉 뉴런은 특정 의미가 있는 게 아니라
# 신경망이 계산 도중에 쓰는 "이름 없는 중간 저장 칸"이다).
#
# 마지막 에피소드가 끝나고 "학습 완료"가 뜬 순간 여기 보이는 값이 곧 최종
# 가중치이고, 콘솔에도 그대로 출력된다. (단, 맨 위 파일 설명대로 이
# 가중치 자체는 실제 역전파가 아니라 목표값(target_w1/target_w2)으로
# 서서히 다가가도록 흉내낸 것이다 — "학습될수록 값이 안정된다"는 느낌만
# 실제와 같다.)
hidden_labels = [f"h{i}" for i in range(N_HID)]


def _make_heatmap(ax, mat, title, xlabel, ylabel, xticklabels, yticklabels):
    im = ax.imshow(mat, cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, fontsize=6, rotation=30, ha="right")
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=6)
    texts = [[ax.text(c, r, "", ha="center", va="center", fontsize=7)
              for c in range(mat.shape[1])] for r in range(mat.shape[0])]
    return im, texts


def _update_heatmap(im, texts, mat):
    im.set_data(mat)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            texts[r][c].set_text(f"{mat[r, c]:.1f}")


im_w1, texts_w1 = _make_heatmap(
    ax_w1, w1, "w1 실제 값 (관측 → 은닉 뉴런)", "은닉 뉴런", "관측값",
    xticklabels=hidden_labels, yticklabels=OBS_LABELS,
)
im_w2, texts_w2 = _make_heatmap(
    ax_w2, w2, "w2 실제 값 (은닉 뉴런 → 행동)", "행동", "은닉 뉴런",
    xticklabels=ACT_LABELS, yticklabels=hidden_labels,
)
fig.colorbar(im_w1, ax=ax_w1, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
fig.colorbar(im_w2, ax=ax_w2, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)


def w_to_style(w):
    color = "#4c72b0" if w >= 0 else "#c44e52"
    lw = float(np.clip(abs(w) * 1.6, 0.3, 5.5))
    alpha = float(np.clip(abs(w) / 3.0, 0.15, 1.0))
    return color, lw, alpha


# 애니메이션은 "에피소드 하나를 여러 프레임에 걸쳐 재생"하는 구조라, 프레임
# 번호를 (에피소드 번호, 그 안의 스텝 번호)로 변환해가며 진행한다.
current_ep = {"idx": -1, "path": None, "obs": None, "act": None, "reached": False, "fell": False}

TOTAL_FRAMES = N_EPISODES * STEPS_PER_EPISODE


def print_final_weights():
    np.set_printoptions(precision=3, suppress=True)
    print("\n=== 학습 완료: 이게 마지막(최종) 정책 가중치 값입니다 ===")
    print(f"w1 (관측 {N_IN} -> 은닉 {N_HID}):")
    print(w1)
    print(f"w2 (은닉 {N_HID} -> 행동 {N_OUT}):")
    print(w2)


def do_frame_step():
    """실제로 화면을 한 프레임 진행시킨다 (에피소드 시작 시 가중치도 갱신)."""
    global w1, w2
    frame = sim_state["frame"]
    ep_idx = frame // STEPS_PER_EPISODE
    step_idx = frame % STEPS_PER_EPISODE

    if ep_idx != current_ep["idx"]:
        # 새 에피소드 시작: 궤적을 새로 만들고, 그 사이 정책 가중치도 한 걸음 갱신
        skill = skill_level(ep_idx)
        path, obs_log, act_log, reached, fell, reward = simulate_episode(skill, rng)
        current_ep.update(idx=ep_idx, path=path, obs=obs_log, act=act_log, reached=reached, fell=fell)
        reward_history.append(reward)

        lr = 0.15
        noise_scale = 0.2 * (1 - skill)
        w1 += lr * (target_w1 - w1) + rng.normal(0, noise_scale, w1.shape)
        w2 += lr * (target_w2 - w2) + rng.normal(0, noise_scale, w2.shape)

        # 전체 진행 상황 현황판: 이 에피소드의 결과를 바로 색칠해서
        # "지금까지 학습이 들쭉날쭉하지만 전체적으로 좋아지는지"를 한눈에
        # 볼 수 있게 한다 (한 에피소드씩 순서대로 완벽해지는 게 아님).
        color = "#55a868" if reached else ("#c44e52" if fell else "#999999")
        history_boxes[ep_idx].set_facecolor(color)

    path = current_ep["path"]
    shown = path[: min(step_idx + 1, len(path))]
    path_line.set_data(shown[:, 0], shown[:, 1])
    robot_dot.set_data([shown[-1, 0]], [shown[-1, 1]])

    # 이 스텝의 실제 관측값/행동 숫자를 노드 옆에 표시 (마지막 스텝을
    # 넘어가면 마지막으로 기록된 값을 그대로 유지해서 보여준다)
    obs_idx = min(step_idx, len(current_ep["obs"]) - 1)
    obs_now = current_ep["obs"][obs_idx]
    act_now = current_ep["act"][obs_idx]
    for t, v in zip(in_value_texts, obs_now):
        t.set_text(f"{v:+.2f}")
    for t, v in zip(out_value_texts, act_now):
        t.set_text(f"{v:+.2f}")

    skill = skill_level(ep_idx)
    episode_text.set_text(f"에피소드 {ep_idx + 1}/{N_EPISODES}   (정책 능숙도 {skill*100:.0f}%)")
    if step_idx >= len(path) - 1:
        if current_ep["reached"]:
            result_text.set_text("결과: 목표 도달!")
            result_text.set_color("#55a868")
        elif current_ep["fell"]:
            result_text.set_text("결과: 넘어짐")
            result_text.set_color("#c44e52")
        else:
            result_text.set_text("결과: 시간 초과")
            result_text.set_color("#999999")
    else:
        result_text.set_text("")

    for line, i, j in edge_lines_1:
        color, lw, alpha = w_to_style(w1[i, j])
        line.set_color(color); line.set_linewidth(lw); line.set_alpha(alpha)
    for line, j, k in edge_lines_2:
        color, lw, alpha = w_to_style(w2[j, k])
        line.set_color(color); line.set_linewidth(lw); line.set_alpha(alpha)

    _update_heatmap(im_w1, texts_w1, w1)
    _update_heatmap(im_w2, texts_w2, w2)

    reward_line.set_data(range(len(reward_history)), reward_history)

    sim_state["frame"] += 1
    if sim_state["frame"] >= TOTAL_FRAMES:
        sim_state["finished"] = True
        sim_state["running"] = False
        episode_text.set_text(episode_text.get_text() + "\n=== 학습 완료 ===")
        print_final_weights()


# ---------------------------------------------------------------------------
# 재생 / 멈춤 / 다음 스텝 컨트롤 (chat/visualize_learning.py와 동일한 방식)
#
# FuncAnimation은 30ms마다 계속 update()를 호출하지만, 실제로 화면을
# 진행시킬지는 여기 sim_state 값으로 결정한다. "멈춤"을 누르면 그 자리에서
# 정지하고, "다음 스텝"을 누르면 멈춘 상태에서 딱 한 프레임만 진행한다
# (로봇이 한 걸음 움직이는 것까지 하나씩 뜯어볼 수 있다). 마지막 에피소드가
# 끝나면 "학습 완료"가 뜨고, 그 순간 화면(과 콘솔)에 보이는 가중치가 곧
# 최종 값이다.
# ---------------------------------------------------------------------------
sim_state = {"frame": 0, "running": True, "step_once": False, "finished": False}


def update(_frame):
    if sim_state["finished"]:
        return []
    if not (sim_state["running"] or sim_state["step_once"]):
        return []
    sim_state["step_once"] = False
    do_frame_step()
    return [path_line, robot_dot, episode_text, result_text, reward_line, im_w1, im_w2] + list(history_boxes)


def on_play(event):
    if not sim_state["finished"]:
        sim_state["running"] = True


def on_pause(event):
    sim_state["running"] = False


def on_step(event):
    if not sim_state["finished"]:
        sim_state["running"] = False
        sim_state["step_once"] = True


ax_play = fig.add_axes([0.30, 0.02, 0.12, 0.05])
ax_pause = fig.add_axes([0.44, 0.02, 0.12, 0.05])
ax_step = fig.add_axes([0.58, 0.02, 0.14, 0.05])
btn_play = Button(ax_play, "재생 ▶")
btn_pause = Button(ax_pause, "멈춤 ⏸")
btn_step = Button(ax_step, "다음 스텝 ⏭")
btn_play.on_clicked(on_play)
btn_pause.on_clicked(on_pause)
btn_step.on_clicked(on_step)

ani = animation.FuncAnimation(fig, update, frames=itertools.count(), interval=30, blit=False, repeat=False,
                               cache_frame_data=False)
plt.show()
