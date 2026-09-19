# ============================================================================
# select_scenario.py — 맵을 고르고, 뷰어에서 키보드로 목표 지점을 "찍어서"
#                       scenario.json으로 저장하는 사전 준비 도구
#
# 사용 흐름:
#   1) maps/ 폴더에 있는 맵 파일 중 하나를 번호로 선택 (또는 무작위 생성)
#   2) MuJoCo 3D 뷰어가 열리고, 로봇과 함께 초록색(청록색) 원반(목표 후보)이 보임
#   3) 방향키(↑↓←→) 또는 W/A/S/D로 원반을 지면 위에서 움직이고, Enter로 확정
#   4) 확정한 (맵 경로, 목표 좌표)가 scenario.json에 저장됨
#
# train.py --scenario scenario.json / play.py --scenario scenario.json 로
# 이 설정을 그대로 불러와 쓸 수 있다.
#
# (지난 버전에서 겪은 버그) 처음엔 목표 마커를 "site"로 만들고 매 프레임
# model.site_pos만 갱신했는데, site_pos는 컴파일 시점 값으로 취급되어
# mj_forward를 다시 불러도 실제 렌더링 좌표(data.site_xpos)에 반영되지
# 않았다. 그래서 방향키를 눌러도 화면에서 커서가 안 움직이는 것처럼
# 보였다 (원인은 키 입력이 아니라 마커 갱신 방식이었음). 지금은
# envs/biped_env.py에서 목표 마커를 "mocap body"로 만들어, 매 프레임
# data.mocap_pos만 갱신하면 mj_forward가 항상 정확히 반영해준다.
#
# (참고) 진짜 마우스 클릭으로 지면을 찍는 방식은 MuJoCo의 공식 뷰어
# API가 지원하지 않는다 — 키보드 콜백(key_callback)만 공식 지원되고,
# 화면 클릭 좌표를 3D 월드 좌표로 변환하는 "레이캐스팅"은 저수준 렌더링
# API를 직접 다뤄야 하는 별도 작업이라 방향키 방식을 쓴다.
# ============================================================================

import argparse
import glob
import json
import os
import time

import numpy as np
import mujoco
import mujoco.viewer

from envs.biped_env import BipedEnv
from maps.generate_map import random_map_spec

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(ROOT_DIR, "maps")
SCENARIO_PATH = os.path.join(ROOT_DIR, "scenario.json")

CURSOR_STEP = 0.25   # 키 한 번에 커서가 움직이는 거리(m)


def list_map_files():
    return sorted(glob.glob(os.path.join(MAPS_DIR, "*.xml")))


def choose_map(preselected=None):
    if preselected is not None:
        return preselected

    maps = list_map_files()
    print("\n사용할 맵을 선택하세요:")
    for i, path in enumerate(maps):
        print(f"  [{i}] {os.path.basename(path)}")
    print(f"  [r] 무작위로 새 맵 생성해서 미리보기")

    choice = input("번호 입력 (기본 0): ").strip().lower()
    if choice == "r":
        rng = np.random.default_rng()
        spec = random_map_spec(rng)
        spec.compile()
        tmp_path = os.path.join(MAPS_DIR, "_preview_random.xml")
        spec.to_file(tmp_path)
        print(f"  -> 무작위 맵을 {tmp_path} 로 저장했습니다 (필요하면 이 파일을 직접 편집하세요)")
        return tmp_path
    if choice == "":
        return maps[0]
    return maps[int(choice)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, default=None, help="번호 선택 없이 바로 이 맵 사용")
    parser.add_argument("--out", type=str, default=SCENARIO_PATH)
    args = parser.parse_args()

    map_path = choose_map(args.map_path)
    print(f"\n선택된 맵: {map_path}")
    print("방향키(↑↓←→) 또는 W/A/S/D로 목표 지점(청록 원반)을 움직이고, Enter로 확정하세요.")
    print("Esc를 누르거나 창을 닫으면 취소합니다.\n")

    env = BipedEnv(map_path=map_path, max_episode_steps=10 ** 9)
    env.reset()
    # 목표 지점을 미리 골라야 하므로, 이 화면에서는 로봇을 넘어지지
    # 않는 정지된 자세로 고정해두고 물리 시뮬레이션은 진행시키지 않는다
    # (키보드로 목표만 고르는 동안 로봇이 쓰러지면 산만하기만 함).

    cursor = np.array([4.0, 0.0])
    state = {"confirmed": False, "cancelled": False}

    def apply_cursor():
        env.data.mocap_pos[env._goal_mocap_id] = [cursor[0], cursor[1], 0.02]
        # mocap body 위치를 바꾼 뒤 mj_forward를 다시 불러야 화면(및
        # data.xpos)에 새 위치가 반영된다.
        mujoco.mj_forward(env.model, env.data)
        print(f"  목표 후보: x={cursor[0]:+.2f}  y={cursor[1]:+.2f}   (Enter로 확정)")

    def key_callback(keycode):
        glfw = mujoco.viewer.glfw
        moved = True
        if keycode in (glfw.KEY_UP, glfw.KEY_W):
            cursor[0] += CURSOR_STEP
        elif keycode in (glfw.KEY_DOWN, glfw.KEY_S):
            cursor[0] -= CURSOR_STEP
        elif keycode in (glfw.KEY_LEFT, glfw.KEY_A):
            cursor[1] += CURSOR_STEP
        elif keycode in (glfw.KEY_RIGHT, glfw.KEY_D):
            cursor[1] -= CURSOR_STEP
        elif keycode in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            state["confirmed"] = True
            moved = False
        elif keycode == glfw.KEY_ESCAPE:
            state["cancelled"] = True
            moved = False
        else:
            moved = False

        if moved:
            with viewer.lock():
                apply_cursor()

    try:
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
            apply_cursor()
            while viewer.is_running() and not state["confirmed"] and not state["cancelled"]:
                viewer.sync()
                time.sleep(0.02)
    except RuntimeError as e:
        if "mjpython" in str(e):
            print(
                "\n[macOS 안내] 인터랙티브 뷰어는 일반 python이 아니라 "
                "mjpython으로 실행해야 합니다. 아래처럼 다시 실행해 보세요:\n"
                "  uv run mjpython select_scenario.py\n"
            )
            return
        raise

    env.close()

    if state["confirmed"]:
        scenario = {"map_path": map_path, "goal": [float(cursor[0]), float(cursor[1])]}
        with open(args.out, "w") as f:
            json.dump(scenario, f, indent=2)
        print(f"\n저장 완료: {args.out}")
        print(f"  {scenario}")
        print(f"\n이제 아래처럼 실행하면 이 시나리오를 그대로 씁니다:")
        print(f"  uv run python train.py --scenario {args.out} --watch")
        print(f"  uv run mjpython play.py --scenario {args.out}")
    else:
        print("\n취소되었습니다. scenario.json이 저장되지 않았습니다.")


if __name__ == "__main__":
    main()
