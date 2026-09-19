# ============================================================================
# play.py — 학습된 정책을 MuJoCo 인터랙티브 뷰어로 직접 눈으로 확인하기
#
# train.py가 저장한 Actor 신경망 가중치를 불러와, 실제로 화면에 3D로 로봇을
# 띄우고 목표 지점(하늘색 원반)을 향해 걸어가는 모습을 실시간으로 보여준다.
#
# 학습 때(train.py)는 정책에서 "무작위로 샘플링"한 행동을 썼지만(탐험을 위해),
# 평가/시연할 때는 각 관절에 대해 정책이 예측한 "평균(mean)" 행동을 그대로
# 사용한다 (deterministic). 이게 그 시점 정책이 낼 수 있는 "가장 자신있는"
# 행동이기 때문에, 실제 성능을 보여주기에 더 적합하다.
# ============================================================================

import argparse
import json
import os
import time

import numpy as np
import mujoco
import mujoco.viewer

from envs.biped_env import BipedEnv, LEG_JOINTS
from rl.ppo import ActorCritic

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def load_policy(obs_dim, act_dim, ckpt_name):
    ac = ActorCritic(obs_dim, act_dim)
    # 가중치를 실제로 만들려면 신경망을 한 번 "호출"해서 각 층의 크기를
    # 확정지어야 한다 (Keras의 지연 빌드 방식 때문에 build 호출 또는 한 번의
    # forward pass가 필요함).
    ac.actor(np.zeros((1, obs_dim), dtype=np.float32))
    ac.critic(np.zeros((1, obs_dim), dtype=np.float32))
    ac.actor.load_weights(os.path.join(CKPT_DIR, f"{ckpt_name}_actor.weights.h5"))
    ac.critic.load_weights(os.path.join(CKPT_DIR, f"{ckpt_name}_critic.weights.h5"))
    return ac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_name", type=str, default="ppo_biped")
    parser.add_argument("--goal", type=float, nargs=2, default=None,
                         help="목표 지점을 직접 지정 (예: --goal 5.0 -1.0). 생략하면 무작위.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--map_path", type=str, default=None,
                         help="사용할 맵(.xml) 파일 경로. 생략하면 maps/map_default.xml 사용")
    parser.add_argument("--randomize_map", action="store_true",
                         help="매 에피소드 무작위 장애물 배치 사용")
    parser.add_argument("--scenario", type=str, default=None,
                         help="select_scenario.py로 만든 scenario.json 경로 (map_path/goal을 이걸로 고정)")
    parser.add_argument("--print_ctrl", action="store_true",
                         help="관절별 컨트롤 시그널(ctrl, -1~1) 값을 터미널에 주기적으로 출력")
    parser.add_argument("--print_ctrl_interval", type=int, default=25,
                         help="몇 스텝마다 출력할지 (--print_ctrl과 함께 사용)")
    args = parser.parse_args()

    map_path = args.map_path
    goal = tuple(args.goal) if args.goal is not None else None
    if args.scenario:
        with open(args.scenario) as f:
            scenario = json.load(f)
        map_path = scenario["map_path"]
        goal = tuple(scenario["goal"])
        print(f"시나리오 로드: map={map_path}  goal={goal}")

    env = BipedEnv(
        max_episode_steps=2000, frame_skip=5,
        map_path=map_path, randomize_map=args.randomize_map,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    ac = load_policy(obs_dim, act_dim, args.ckpt_name)

    options = {"goal": goal} if goal is not None else {}
    joint_labels = [j.replace("_joint", "") for j in LEG_JOINTS]  # 예: "l_hip_yaw_joint" -> "l_hip_yaw"

    def print_ctrl(action):
        # 관절별 컨트롤 시그널(ctrl, -1~1)을 "이름=값" 형태로 나열해 출력.
        # 이 값이 실제로 env.step() 안에서 각 액추에이터의 torque = gain*ctrl
        # 계산에 그대로 쓰인다 (documents/02-environment-and-reward.md 참고).
        parts = [f"{name}={v:+.2f}" for name, v in zip(joint_labels, action)]
        print("  ctrl: " + "  ".join(parts))

    def run_episode(viewer):
        obs, _ = env.reset(options=options)
        print(f"goal = {env._goal_xy}  obstacles = {env._obstacle_xy}")
        done = False
        info = {}
        step_count = 0
        while not done and viewer.is_running():
            step_start = time.time()

            obs_tf = obs[None, :].astype(np.float32)
            mean_action = ac.actor(obs_tf).numpy()[0]  # 샘플링 없이 평균 행동만 사용
            obs, reward, terminated, truncated, info = env.step(mean_action)
            done = terminated or truncated

            step_count += 1
            if args.print_ctrl and step_count % args.print_ctrl_interval == 0:
                print_ctrl(mean_action)

            viewer.sync()
            # 물리 시뮬레이션 속도와 실제 시계(wall-clock) 속도를 맞춰서
            # 너무 빠르거나 느리지 않게, 사람 눈에 자연스러운 실시간으로 재생
            elapsed = time.time() - step_start
            sleep_time = env.frame_skip * env.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if viewer.is_running():
            print(f"  -> {'GOAL 도달!' if info.get('reached_goal') else '넘어짐' if info.get('fell') else '시간 초과'}")
        return viewer.is_running()

    try:
        if args.randomize_map:
            # randomize_map=True일 때는 reset()마다 env.model/env.data 자체가
            # 완전히 새 객체로 교체된다 (장애물 배치가 바뀌면 모델을 통째로
            # 재컴파일하기 때문). MuJoCo 뷰어는 특정 model/data 객체에 묶여서
            # 열리므로, 그런 경우에만 매 에피소드 뷰어를 새로 열어야 새
            # 지형이 화면에 반영된다. macOS(mjpython)에서는 뷰어를 닫자마자
            # 바로 다시 열면 "another MuJoCo viewer is already open" 에러가
            # 나서(이전 창의 리소스 해제가 비동기라 살짝 시간이 걸림), 짧게
            # 쉬었다가 다시 연다.
            for ep in range(args.episodes):
                print(f"episode {ep}:", end=" ")
                with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                    still_running = run_episode(viewer)
                if not still_running:
                    break
                time.sleep(0.3)
        else:
            # 맵이 에피소드 사이에 바뀌지 않는 일반적인 경우: 뷰어 하나를
            # 계속 재사용한다 (매번 새로 열 필요가 없고, 그게 더 안전하다).
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                for ep in range(args.episodes):
                    if not viewer.is_running():
                        break
                    print(f"episode {ep}:", end=" ")
                    if not run_episode(viewer):
                        break
    except RuntimeError as e:
        if "mjpython" in str(e):
            print(
                "\n[macOS 안내] 인터랙티브 뷰어는 일반 python이 아니라 "
                "mjpython으로 실행해야 합니다. 아래처럼 다시 실행해 보세요:\n"
                "  uv run mjpython play.py ...\n"
            )
            return
        raise

    env.close()


if __name__ == "__main__":
    main()
