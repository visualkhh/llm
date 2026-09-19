# ============================================================================
# visualize_mpc.py — "학습된 신경망이 즉시 반응"하는 게 아니라, "그때그때
#                     실시간으로 미래를 시뮬레이션해보고 계획을 세우는"
#                     방식(모델 기반 제어, Model Predictive Control)을
#                     보여주는 교육용 애니메이션
#
# visualize_learning.py(가짜 PPO 흉내)와 정확히 반대되는 개념을 보여주기
# 위한 파일이다:
#
#   visualize_learning.py (모델-프리 RL, PPO)
#     - "학습(training)" 단계에서 수많은 시행착오를 거쳐 가중치에 노하우를
#       미리 다 구워넣는다.
#     - 실행할 땐 신경망에 센서값을 넣고 한 번 통과시켜(forward pass)
#       나온 값을 그대로 쓴다 — 계산량이 거의 없는 즉각 반응(반사작용).
#     - 그래서 "학습 곡선"이 있다: 초반 에피소드는 못하고, 갈수록 좋아진다.
#
#   visualize_mpc.py (이 파일, 모델 기반 제어/Model-based RL)
#     - "학습"이라는 개념 자체가 없다. 가중치도, 정답 데이터도 없다.
#     - 대신 로봇이 "내가 이렇게 움직이면 세상이 이렇게 반응할 것이다"라는
#       수학적 모델(step_dynamics 함수)을 갖고 있다.
#     - 매 스텝마다 여러 개의 "만약 이렇게 조향하면?" 후보를 그 모델로
#       직접 미리 시뮬레이션(rollout)해보고, 목표에 가까워지고 장애물을
#       피하는 정도를 점수(cost)로 매겨서 제일 나은 후보를 고른다.
#     - 근데 그 계획 전체를 실행하는 게 아니라 **딱 첫 한 스�텝만** 실행하고,
#       다음 순간 또 처음부터 계획을 새로 짠다 (이걸 receding horizon,
#       "매번 지평선을 다시 그리며 전진"이라고 부른다).
#     - 그래서 "학습 곡선"이 없다 — 첫 번째 시도부터 마지막 시도까지
#       실력 차이가 나지 않는다 (연습으로 나아지는 게 아니라, 매번 계산으로
#       잘하는 것이기 때문). 화면 위쪽 현황판에서 이걸 직접 확인할 수 있다.
#
# 로봇 머리에 3D 라이다가 달려 있다고 가정하고(위아래 여러 각도 x 좌우
# 여러 각도로 광선을 쏘는 방식, 실제 Velodyne류 라이다와 같은 개념), 그
# 라이다가 만들어내는 3D 점군(point cloud)을 오른쪽 위 3D 패널에 실시간으로
# 보여준다. (단, 교육용 단순화를 위해 "계획을 세울 때 쓰는 장애물 위치"는
# 이미 정확히 안다고 가정한다 — 라이다 점들로부터 지도를 직접 만드는
# 과정(SLAM/occupancy mapping)은 이 데모의 범위를 넘어선다.)
# ============================================================================

import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D 프로젝션 등록용, 직접 쓰진 않음)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(2)

# ---------------------------------------------------------------------------
# 세계 설정 — 장애물을 원기둥(x, y, 반지름, 높이)으로 표현한다. 반지름은
# 2D 상단뷰 회피 계산에, 높이는 3D 라이다가 장애물의 "옆면"에 부딪히는지
# 판정하는 데 쓰인다.
# ---------------------------------------------------------------------------
START = np.array([0.0, 0.0])
GOAL = np.array([6.0, 1.0])
OBSTACLES = [
    (2.0, 0.6, 0.45, 1.4),
    (3.6, -0.6, 0.45, 1.6),
    (4.8, 1.3, 0.4, 1.2),
]

STEP_SIZE = 0.22            # 한 스텝에 전진하는 거리
MAX_TURN = 0.5              # 한 스텝에 꺾을 수 있는 최대 각도(rad, 약 29도)
HORIZON = 10                # 몇 스텝 앞까지 미리 시뮬레이션(rollout)해볼지
CANDIDATE_TURNS = np.linspace(-MAX_TURN, MAX_TURN, 15)   # 매번 검토하는 조향각 후보(고정)
MAX_STEPS_PER_ATTEMPT = 120
N_ATTEMPTS = 6               # 데모를 몇 번 반복할지 (학습 곡선이 없다는 걸 보여주기 위함)

SENSOR_HEIGHT = 1.5
LIDAR_MAX_RANGE = 6.0
LIDAR_ELEVATIONS_DEG = [-15.0, 0.0, 12.0]              # 위/아래 3개 층(ring)
LIDAR_AZIMUTHS_DEG = np.linspace(-100.0, 100.0, 25)    # 전방 200도, 25개 광선


def step_dynamics(pos, heading, turn_rate, step_size=STEP_SIZE):
    """유니사이클(unicycle) 모델: 방향을 turn_rate만큼 꺾은 뒤 그 방향으로
    step_size만큼 전진한다.

    실제로 로봇을 한 스텝 움직일 때도, 아래 rollout()에서 "미래를 미리
    예측"할 때도 **이 함수를 똑같이 사용한다** — 이게 '모델 기반
    (model-based)'이라는 말의 정확한 의미다: 로봇이 자기 몸이 어떻게
    움직일지 아는 수학적 모델을 갖고 있어서, 실제로 움직여보지 않고도
    "이렇게 하면 이렇게 될 것이다"를 미리 계산해볼 수 있다는 뜻이다.
    """
    new_heading = heading + turn_rate
    new_pos = pos + step_size * np.array([np.cos(new_heading), np.sin(new_heading)])
    return new_pos, new_heading


def clearance(pos, obstacles):
    """가장 가까운 장애물 표면까지 남은 여유 거리 (음수면 이미 충돌)."""
    best = np.inf
    for ox, oy, r, _h in obstacles:
        d = np.linalg.norm(pos - np.array([ox, oy])) - r
        best = min(best, d)
    return best


def rollout(pos, heading, turn_rate, horizon=HORIZON):
    """이 조향각(turn_rate)을 horizon 스텝 동안 유지했다고 가정하고,
    실제로 움직여보지 않은 채 결과 경로를 미리 계산해본다. 이게 바로
    MPC의 "계획 세우기(planning)"다."""
    traj = [pos.copy()]
    p, h = pos.copy(), heading
    for _ in range(horizon):
        p, h = step_dynamics(p, h, turn_rate)
        traj.append(p.copy())
    return np.array(traj)


def trajectory_cost(traj, obstacles, turn_rate):
    """이 경로가 얼마나 "나쁜지" 점수를 매긴다 (낮을수록 좋음).
      1) 목표와 멀리 끝날수록 나쁨 (경로의 '끝점'이 아니라, 경로가
         지나가는 모든 지점 중 목표에 가장 가까웠던 지점 기준 — 끝점만
         보면 목표를 스쳐 지나가는 경로도 "끝점 기준"으로는 나쁘게
         평가돼서, 목표 바로 앞에서 계속 원을 그리며 맴돌기만 하고 절대
         안착하지 못하는 문제가 생긴다)
      2) 장애물에 가까이 지나갈수록 나쁨 (부딪히면 아주 크게 나쁨)
      3) 조향을 너무 세게 꺾을수록 살짝 나쁨 (부드러운 움직임 선호)
    """
    dists_to_goal = np.linalg.norm(traj - GOAL, axis=1)
    goal_cost = dists_to_goal.min() * 3.0
    obstacle_cost = 0.0
    for p in traj[1:]:
        c = clearance(p, obstacles)
        if c < 0.5:
            obstacle_cost += (0.5 - c) ** 2 * 40.0
        if c < 0:
            obstacle_cost += 200.0
    control_cost = abs(turn_rate) * 0.5
    return goal_cost + obstacle_cost + control_cost


def mpc_plan(pos, heading, obstacles):
    """지금 이 순간, 가능한 조향각 후보 CANDIDATE_TURNS를 전부 시뮬레이션
    해보고 비용이 제일 낮은 것을 고른다.

    핵심: 이 함수는 **매 스텝마다 처음부터 다시 호출된다** (receding
    horizon). 방금 세운 계획의 나머지 부분은 다음 스텝에서 그냥 버려지고
    완전히 새로 계산한다 — "학습해서 점점 잘하는" 게 아니라 "매번 계산해서
    바로 잘하는" 방식이라는 뜻이다.
    """
    trajs = [rollout(pos, heading, tr) for tr in CANDIDATE_TURNS]
    costs = np.array([trajectory_cost(traj, obstacles, tr) for traj, tr in zip(trajs, CANDIDATE_TURNS)])
    best_idx = int(np.argmin(costs))
    return trajs, costs, best_idx


def lidar3d_scan(pos, heading, obstacles, sensor_height=SENSOR_HEIGHT,
                  elevations_deg=LIDAR_ELEVATIONS_DEG, azimuths_deg=LIDAR_AZIMUTHS_DEG,
                  max_range=LIDAR_MAX_RANGE):
    """머리에 달린 3D 라이다 흉내. 여러 '층(elevation, 위아래 각도)' x 여러
    '수평각(azimuth)' 방향으로 광선을 쏴서, 장애물 옆면이나 바닥에 부딪히는
    지점들을 모은다 — 실제 3D 라이다(예: Velodyne)가 만들어내는 점군
    (point cloud)과 같은 개념이다. 아무것도 안 맞고 하늘로 날아가는
    광선은 (실제 라이다처럼) 점을 남기지 않는다.
    """
    points = []
    for el_deg in elevations_deg:
        el = np.radians(el_deg)
        # 아래를 보고 있을 때만 바닥에 닿을 수 있다.
        ground_dist = sensor_height / (-np.tan(el)) if el < -1e-6 else np.inf

        for az_deg in azimuths_deg:
            az = heading + np.radians(az_deg)
            direction = np.array([np.cos(az), np.sin(az)])

            obs_dist = np.inf
            obs_z = None
            for ox, oy, r, h in obstacles:
                f = pos - np.array([ox, oy])
                b = 2 * np.dot(f, direction)
                c = np.dot(f, f) - r ** 2
                disc = b ** 2 - 4 * c
                if disc < 0:
                    continue
                t = (-b - np.sqrt(disc)) / 2
                if t <= 0 or t > max_range:
                    continue
                z = sensor_height + t * np.tan(el)
                if 0 <= z <= h and t < obs_dist:
                    obs_dist = t
                    obs_z = z

            if obs_dist < ground_dist and obs_dist <= max_range:
                hit_xy = pos + obs_dist * direction
                points.append((hit_xy[0], hit_xy[1], obs_z))
            elif ground_dist <= max_range:
                hit_xy = pos + ground_dist * direction
                points.append((hit_xy[0], hit_xy[1], 0.0))
            # 둘 다 아니면: 광선이 아무것도 못 맞히고 하늘로 날아감 -> 점 없음
    return np.array(points) if points else np.zeros((0, 3))


# ---------------------------------------------------------------------------
# 화면 구성
#   맨 위: 전체 시도(attempt) 결과 현황판 — "학습 곡선이 없다"는 것을
#          한눈에 보여준다 (1번째 시도부터 초록불이 뜨는 게 정상).
#   왼쪽: 상단뷰 — 로봇, 목표, 장애물, 이번 스텝에 검토한 모든 후보 경로
#         (색=비용), 그중 선택된 경로(굵은 파랑), 지금까지 실제로 걸어온 길.
#   가운데 위: 머리에 달린 3D 라이다가 지금 보고 있는 점군.
#   가운데 아래: 이번 스텝에 검토한 15개 조향각 후보와 각각의 비용(막대).
#   오른쪽 위: 그동안 실제로 선택된 조향각의 변화(부드럽게 이어지는지).
#   맨 아래: 재생 / 멈춤 / 다음 스텝 버튼.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 8))
fig.suptitle("모델 기반 제어(MPC) — 학습 없이, 매 순간 미래를 시뮬레이션해서 계획하는 방식", fontsize=11)
gs = fig.add_gridspec(3, 3, height_ratios=[0.32, 1, 1], width_ratios=[1.3, 1, 1],
                       left=0.045, right=0.98, top=0.90, bottom=0.13, hspace=0.55, wspace=0.35)

ax_history = fig.add_subplot(gs[0, :])
ax_world = fig.add_subplot(gs[1:, 0])
ax_lidar3d = fig.add_subplot(gs[1, 1], projection="3d")
ax_cost = fig.add_subplot(gs[2, 1])
ax_turn = fig.add_subplot(gs[1, 2])
ax_info = fig.add_subplot(gs[2, 2])

# --- 0) 전체 시도 결과 현황판 ---
ax_history.axis("off")
ax_history.set_xlim(0, 1)
ax_history.set_ylim(0, 1)
ax_history.text(
    0.01, 0.85,
    "전체 시도(attempt) 결과 (초록=목표 도달, 빨강=충돌, 회색=시간 초과) — "
    "학습이 아니므로 1번째 시도부터 실력 차이가 없는 게 정상:",
    fontsize=8.5, color="#555555", transform=ax_history.transAxes,
)
hist_x = np.linspace(0.03, 0.97, N_ATTEMPTS)
hist_w = (hist_x[1] - hist_x[0]) * 0.6 if N_ATTEMPTS > 1 else 0.1
history_boxes = []
for i, x in enumerate(hist_x):
    box = Rectangle((x - hist_w / 2, 0.1), hist_w, 0.55, transform=ax_history.transAxes,
                     facecolor="white", edgecolor="#bbbbbb", linewidth=0.8)
    ax_history.add_patch(box)
    ax_history.text(x, 0.02, f"시도{i + 1}", ha="center", fontsize=7, color="#888888",
                     transform=ax_history.transAxes)
    history_boxes.append(box)

# --- 1) 상단뷰 (World) ---
ax_world.set_xlim(-1.0, 7.5)
ax_world.set_ylim(-2.2, 2.5)
ax_world.set_aspect("equal")
ax_world.set_title("상단뷰: 매 스텝 검토한 후보 경로(색=비용) + 선택된 경로(굵은 파랑)")

for ox, oy, r, _h in OBSTACLES:
    ax_world.add_patch(Circle((ox, oy), r, color="#c44e52", alpha=0.75, zorder=2))
ax_world.plot(*GOAL, marker="*", markersize=22, color="#55a868", zorder=3)
ax_world.text(GOAL[0], GOAL[1] + 0.35, "목표", ha="center", fontsize=9)

cmap_cost = plt.get_cmap("RdYlGn_r")
candidate_lines = [ax_world.plot([], [], "-", lw=1.0, alpha=0.55, zorder=1)[0] for _ in CANDIDATE_TURNS]
best_line, = ax_world.plot([], [], "-", color="#1f5fa8", lw=2.6, zorder=4)
path_line, = ax_world.plot([], [], "-", color="#333333", lw=1.6, zorder=3)
robot_dot, = ax_world.plot([], [], "o", color="#333333", markersize=11, zorder=5)
attempt_text = ax_world.text(0.02, 0.97, "", transform=ax_world.transAxes, va="top", fontsize=10)
result_text = ax_world.text(0.02, 0.03, "", transform=ax_world.transAxes, va="bottom", fontsize=10)

# --- 2) 3D 라이다 점군 패널 ---
ax_lidar3d.set_title("머리 3D 라이다가 지금 보는 점군(point cloud)", fontsize=9)


def _draw_lidar3d(points, pos, heading):
    ax_lidar3d.cla()
    ax_lidar3d.set_title("머리 3D 라이다가 지금 보는 점군(point cloud)", fontsize=9)
    ax_lidar3d.set_xlim(pos[0] - 1, pos[0] + LIDAR_MAX_RANGE * 0.6)
    ax_lidar3d.set_ylim(pos[1] - 3, pos[1] + 3)
    ax_lidar3d.set_zlim(0, 2.2)
    ax_lidar3d.set_xlabel("x", fontsize=7, labelpad=-8)
    ax_lidar3d.set_ylabel("y", fontsize=7, labelpad=-8)
    ax_lidar3d.tick_params(labelsize=6, pad=0)
    ax_lidar3d.view_init(elev=22, azim=-60)
    ax_lidar3d.scatter([pos[0]], [pos[1]], [SENSOR_HEIGHT], color="#333333", s=35, marker="^")
    if len(points):
        ax_lidar3d.scatter(points[:, 0], points[:, 1], points[:, 2],
                            c=points[:, 2], cmap="viridis", s=8)


_draw_lidar3d(np.zeros((0, 3)), START, 0.0)

# --- 3) 후보 조향각 비용 막대그래프 ---
cand_deg = np.degrees(CANDIDATE_TURNS)
ax_cost.set_title("이번 스텝에 검토한 조향각 후보와 비용 (초록=선택됨)", fontsize=9)
ax_cost.set_xlabel("조향각(도) — 0=직진, +좌회전/-우회전", fontsize=7.5)
ax_cost.set_ylabel("비용(낮을수록 좋음)", fontsize=7.5)
bar_width = (cand_deg[1] - cand_deg[0]) * 0.8
cost_bars = ax_cost.bar(cand_deg, [0] * len(cand_deg), width=bar_width, color="#999999")

# --- 4) 선택된 조향각 히스토리 (이번 시도 동안) ---
ax_turn.set_title("선택된 조향각 변화 (이번 시도)", fontsize=9)
ax_turn.set_xlabel("스텝", fontsize=7.5)
ax_turn.set_ylabel("조향각(도)", fontsize=7.5)
ax_turn.set_xlim(0, MAX_STEPS_PER_ATTEMPT)
ax_turn.set_ylim(-np.degrees(MAX_TURN) * 1.2, np.degrees(MAX_TURN) * 1.2)
ax_turn.axhline(0, color="gray", lw=0.5)
turn_line, = ax_turn.plot([], [], "-", color="#4c72b0", lw=1.3)

# --- 5) 현재 상태 정보 패널 ---
ax_info.axis("off")
ax_info.set_xlim(0, 1)
ax_info.set_ylim(0, 1)
info_text = ax_info.text(0.02, 0.95, "", transform=ax_info.transAxes, va="top", fontsize=9, family="monospace")


def print_final_summary(results):
    print("\n=== 데모 종료: 시도별 결과 ===")
    for i, r in enumerate(results):
        print(f"  시도 {i + 1}: {r}")
    n_ok = sum(1 for r in results if r == "목표 도달")
    print(f"학습 없이도 {n_ok}/{len(results)}번 성공 — 시도 순서와 성공 여부에 뚜렷한 상관관계가 없는 게 정상입니다.")


# ---------------------------------------------------------------------------
# 시도(attempt) 진행 상태
# ---------------------------------------------------------------------------
attempt_state = {
    "idx": 0,
    "pos": None,
    "heading": 0.0,
    "path": None,
    "turn_history": [],
    "step": 0,
    "results": [],
}


def reset_attempt(idx):
    goal_dir = GOAL - START
    base_heading = np.arctan2(goal_dir[1], goal_dir[0])
    heading0 = base_heading + rng.uniform(-0.4, 0.4)   # 매번 살짝 다른 초기 방향
    pos0 = START + rng.normal(0, 0.05, size=2)         # 매번 살짝 다른 시작 위치
    attempt_state.update(idx=idx, pos=pos0, heading=heading0,
                          path=[pos0.copy()], turn_history=[], step=0)
    attempt_text.set_text(f"시도 {idx + 1}/{N_ATTEMPTS}")
    result_text.set_text("")
    turn_line.set_data([], [])


reset_attempt(0)


def do_frame_step():
    pos = attempt_state["pos"]
    heading = attempt_state["heading"]

    # 1) 지금 이 순간 다시 계획을 세운다 (receding horizon) — 이전 스텝의
    #    계획은 전부 버리고 매번 처음부터 다시 계산한다.
    trajs, costs, best_idx = mpc_plan(pos, heading, OBSTACLES)
    chosen_turn = CANDIDATE_TURNS[best_idx]

    # 2) 머리 3D 라이다로 지금 보이는 것을 스캔한다 (표시용).
    points = lidar3d_scan(pos, heading, OBSTACLES)

    # 3) 계획의 '첫 한 스텝'만 실제로 실행한다.
    new_pos, new_heading = step_dynamics(pos, heading, chosen_turn)
    attempt_state["pos"] = new_pos
    attempt_state["heading"] = new_heading
    attempt_state["path"].append(new_pos.copy())
    attempt_state["turn_history"].append(np.degrees(chosen_turn))
    attempt_state["step"] += 1

    # --- 화면 갱신: 상단뷰 후보 경로들 ---
    cost_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-9)
    for line, traj, cn in zip(candidate_lines, trajs, cost_norm):
        line.set_data(traj[:, 0], traj[:, 1])
        line.set_color(cmap_cost(cn))
    best_line.set_data(trajs[best_idx][:, 0], trajs[best_idx][:, 1])

    path_arr = np.array(attempt_state["path"])
    path_line.set_data(path_arr[:, 0], path_arr[:, 1])
    robot_dot.set_data([new_pos[0]], [new_pos[1]])

    # --- 3D 라이다 패널 ---
    _draw_lidar3d(points, new_pos, new_heading)

    # --- 비용 막대그래프 ---
    for bar, c in zip(cost_bars, costs):
        bar.set_height(c)
        bar.set_color("#999999")
    cost_bars[best_idx].set_color("#55a868")
    ax_cost.set_ylim(0, max(costs.max() * 1.15, 1.0))

    # --- 조향각 히스토리 ---
    hist = attempt_state["turn_history"]
    turn_line.set_data(range(len(hist)), hist)

    # --- 정보 패널 ---
    dist = np.linalg.norm(new_pos - GOAL)
    info_text.set_text(
        f"시도       : {attempt_state['idx'] + 1}/{N_ATTEMPTS}\n"
        f"스텝       : {attempt_state['step']}/{MAX_STEPS_PER_ATTEMPT}\n"
        f"위치       : ({new_pos[0]:+.2f}, {new_pos[1]:+.2f})\n"
        f"목표까지   : {dist:.2f}\n"
        f"선택 조향각: {np.degrees(chosen_turn):+.1f}도\n"
        f"선택 비용  : {costs[best_idx]:.2f}\n"
        f"검토 후보 수: {len(CANDIDATE_TURNS)}개 (매 스텝 전부 재계산)"
    )

    # --- 종료 판정 ---
    outcome = None
    if dist < 0.3:
        outcome = "목표 도달"
    elif clearance(new_pos, OBSTACLES) < 0:
        outcome = "충돌"
    elif attempt_state["step"] >= MAX_STEPS_PER_ATTEMPT:
        outcome = "시간 초과"

    if outcome is not None:
        color = {"목표 도달": "#55a868", "충돌": "#c44e52", "시간 초과": "#999999"}[outcome]
        result_text.set_text(f"결과: {outcome}")
        result_text.set_color(color)
        history_boxes[attempt_state["idx"]].set_facecolor(color)
        attempt_state["results"].append(outcome)

        next_idx = attempt_state["idx"] + 1
        if next_idx >= N_ATTEMPTS:
            sim_state["finished"] = True
            sim_state["running"] = False
            attempt_text.set_text(attempt_text.get_text() + "\n=== 데모 종료 ===")
            print_final_summary(attempt_state["results"])
        else:
            reset_attempt(next_idx)


# ---------------------------------------------------------------------------
# 재생 / 멈춤 / 다음 스텝 컨트롤 (chat, humanoid_rl의 다른 시각화 파일들과
# 동일한 방식) — "다음 스텝"을 누르면 MPC가 다시 계획을 세우고 딱 한 걸음만
# 내딛는 과정을 하나씩 뜯어볼 수 있다.
# ---------------------------------------------------------------------------
sim_state = {"running": True, "step_once": False, "finished": False}


def update(_frame):
    if sim_state["finished"]:
        return []
    if not (sim_state["running"] or sim_state["step_once"]):
        return []
    sim_state["step_once"] = False
    do_frame_step()
    return ([path_line, robot_dot, best_line, attempt_text, result_text, turn_line, info_text]
            + candidate_lines + list(cost_bars) + list(history_boxes))


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

ani = animation.FuncAnimation(fig, update, frames=itertools.count(), interval=60, blit=False, repeat=False,
                               cache_frame_data=False)
plt.show()
