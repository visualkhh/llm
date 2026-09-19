# ============================================================================
# visualize_inference.py — 실제로 학습된 정책이 실제 MuJoCo 물리 시뮬레이션
#                           위에서 어떤 관측값을 받아 어떤 모터 제어값을
#                           내는지 실시간으로 보여주기
#
# visualize_learning.py와의 관계: chat 프로젝트의 visualize_learning.py(가짜
# 학습 시뮬레이션) / visualize_generation.py(진짜 추론)와 정확히 같은
# 구도다.
#
#   humanoid_rl/visualize_learning.py  = "학습이 되면 이런 느낌"을 2D로
#                                         흉내낸 것 (가짜, 실제 MuJoCo/PPO 없음)
#   humanoid_rl/visualize_inference.py = 이 파일. 실제 checkpoints/의 학습된
#                                         가중치 + 진짜 BipedEnv(MuJoCo 물리)로
#                                         진짜 추론(inference)을 돌린다.
#
# play.py와 다른 점: play.py는 MuJoCo의 인터랙티브 3D 뷰어(mjpython 필요)를
# 쓰지만, 이 파일은 matplotlib 한 창 안에 (1) 로봇을 위에서 오프스크린으로
# 렌더링한 이미지 (2) 그 순간의 실제 관측값 숫자들 (3) 그 순간 정책이 낸
# 실제 모터 제어값 숫자들을 함께 보여준다 — "지금 이 순간 모델이 뭘 보고
# (관측값) 뭘 하기로 했는지(행동)"를 한눈에 확인하기 위함이다.
# ============================================================================

import argparse
import json
import os

import numpy as np
import mujoco
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from envs.biped_env import BipedEnv, LEG_JOINTS, N_LIDAR_RAYS
from rl.ppo import ActorCritic

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

# envs/biped_env.py의 _get_obs()가 만드는 45차원 관측값의 실제 배치
# (documents/02-environment-and-reward.md 표와 정확히 일치):
#   [0:10] 관절 각도, [10:20] 관절 각속도, [20] 기울기,
#   [21:24] torso 선속도, [24:27] torso 각속도,
#   [27:29] 목표 상대위치(x,y), [29] 목표거리, [30:45] 라이다 15개
TILT_IDX = 20
GOAL_XY_IDX = (27, 29)
GOAL_DIST_IDX = 29
LIDAR_START = 30


def load_policy(obs_dim, act_dim, ckpt_name):
    ac = ActorCritic(obs_dim, act_dim)
    ac.actor(np.zeros((1, obs_dim), dtype=np.float32))
    ac.critic(np.zeros((1, obs_dim), dtype=np.float32))
    ac.actor.load_weights(os.path.join(CKPT_DIR, f"{ckpt_name}_actor.weights.h5"))
    ac.critic.load_weights(os.path.join(CKPT_DIR, f"{ckpt_name}_critic.weights.h5"))
    return ac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_name", type=str, default="ppo_biped")
    parser.add_argument("--scenario", type=str, default=None,
                         help="select_scenario.py로 만든 scenario.json 경로 (map_path/goal 고정)")
    parser.add_argument("--goal", type=float, nargs=2, default=None)
    parser.add_argument("--map_path", type=str, default=None)
    parser.add_argument("--interval_ms", type=int, default=60)
    args = parser.parse_args()

    ckpt_path = os.path.join(CKPT_DIR, f"{args.ckpt_name}_actor.weights.h5")
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"checkpoints/{args.ckpt_name}_actor.weights.h5 를 찾을 수 없습니다. "
            f"먼저 train.py로 학습시켜 주세요."
        )

    map_path = args.map_path
    goal = tuple(args.goal) if args.goal else None
    if args.scenario:
        with open(args.scenario) as f:
            scenario = json.load(f)
        map_path = scenario["map_path"]
        goal = tuple(scenario["goal"])

    env = BipedEnv(max_episode_steps=2000, frame_skip=5, map_path=map_path)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    ac = load_policy(obs_dim, act_dim, args.ckpt_name)

    options = {"goal": goal} if goal is not None else {}
    obs, _ = env.reset(options=options)

    renderer = mujoco.Renderer(env.model, height=360, width=480)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.5
    cam.azimuth = 130
    cam.elevation = -15

    joint_labels = [j.replace("_joint", "") for j in LEG_JOINTS]

    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 6.5))
    fig.suptitle(f"실제 학습된 정책(checkpoints/{args.ckpt_name})의 실시간 추론 — 진짜 MuJoCo 물리 시뮬레이션", fontsize=11)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1])
    ax_img = fig.add_subplot(gs[:, 0])
    ax_obs = fig.add_subplot(gs[0, 1])
    ax_act = fig.add_subplot(gs[1, 1])

    ax_img.axis("off")
    ax_img.set_title("실제 MuJoCo 렌더링")
    img_artist = ax_img.imshow(np.zeros((360, 480, 3), dtype=np.uint8))
    info_text = ax_img.text(0.02, 0.02, "", transform=ax_img.transAxes, va="bottom", color="white",
                             fontsize=10, bbox=dict(facecolor="black", alpha=0.5, pad=4))

    ax_obs.set_title("실제 관측값 (관절 각도 10개는 생략, 핵심만 표시)")
    ax_obs.axis("off")
    obs_labels = ["목표거리", "목표-앞", "목표-좌우", "기울기"] + [f"라이다{i}" for i in range(N_LIDAR_RAYS)]
    obs_texts = []
    for i, lab in enumerate(obs_labels):
        y = 0.95 - i * (0.9 / len(obs_labels))
        ax_obs.text(0.02, y, lab, fontsize=8, color="#555555", transform=ax_obs.transAxes, va="top")
        t = ax_obs.text(0.55, y, "", fontsize=8, color="#4c72b0", fontweight="bold",
                         transform=ax_obs.transAxes, va="top")
        obs_texts.append(t)

    ax_act.set_title("실제 모터 제어값 (ctrl, -1~1)")
    ax_act.set_xlim(-1, 1)
    ax_act.set_yticks(range(len(joint_labels)))
    ax_act.set_yticklabels(joint_labels, fontsize=8)
    ax_act.axvline(0, color="gray", lw=0.6)
    act_bars = ax_act.barh(range(len(joint_labels)), [0] * len(joint_labels), color="#dd8452")

    state = {"reward_sum": 0.0, "step": 0, "done_msg": ""}

    def step_and_render(frame):
        nonlocal obs
        if state["done_msg"]:
            return [img_artist, info_text] + obs_texts + list(act_bars)

        mean_action = ac.actor(obs[None, :].astype(np.float32)).numpy()[0]
        obs, reward, terminated, truncated, info = env.step(mean_action)
        state["reward_sum"] += reward
        state["step"] += 1

        renderer.update_scene(env.data, camera=cam)
        img_artist.set_data(renderer.render())

        values = [obs[GOAL_DIST_IDX], obs[GOAL_XY_IDX[0]], obs[GOAL_XY_IDX[1]], obs[TILT_IDX]]
        values += list(obs[LIDAR_START:LIDAR_START + N_LIDAR_RAYS])
        for t, v in zip(obs_texts, values):
            t.set_text(f"{v:+.2f}")

        for bar, v in zip(act_bars, mean_action):
            bar.set_width(v)
            bar.set_color("#55a868" if v >= 0 else "#c44e52")

        info_text.set_text(
            f"step {state['step']}   reward_sum {state['reward_sum']:.1f}\n"
            f"목표까지 {obs[GOAL_DIST_IDX]:.2f}m"
        )

        done = terminated or truncated
        if done:
            msg = "목표 도달!" if info.get("reached_goal") else ("넘어짐" if info.get("fell") else "시간 초과")
            state["done_msg"] = msg
            info_text.set_text(info_text.get_text() + f"\n=== {msg} (애니메이션 정지) ===")

        return [img_artist, info_text] + obs_texts + list(act_bars)

    ani = animation.FuncAnimation(fig, step_and_render, frames=2000, interval=args.interval_ms,
                                   blit=False, repeat=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
