# ============================================================================
# maps/generate_map.py — "맵(지면+장애물)"을 URDF와 분리된 독립 파일로 다루기
#
# robot/biped.urdf가 "로봇 자체"만 담당하듯, 이 파일은 "로봇이 그 위에서
# 움직일 세상(맵)"만 담당한다. MuJoCo 생태계에서 이런 씬(scene) 묘사에
# 쓰이는 표준 포맷은 USD(NVIDIA Omniverse/Isaac 계열)가 아니라
# **MJCF(.xml, MuJoCo의 네이티브 포맷)**이다. (자세한 설명은
# documents/06-map-files.md 참고)
#
# 이 파일 하나로 세 가지를 모두 할 수 있다:
#   1) 맵을 코드로 직접 구성 (build_map_spec)
#   2) 무작위 장애물 배치를 만들어 즉석에서 사용 (random_map_spec) — 학습 중
#      매 에피소드 새로운 지형에 적응하도록 훈련시키고 싶을 때 씀
#   3) 만든 맵을 실제 .xml 파일로 저장해서(export), 나중에 사람이 텍스트
#      에디터로 열어 직접 장애물을 추가/이동/삭제하는 식으로 편집하고,
#      그 파일을 다시 불러와 쓸 수도 있음 (--out 옵션으로 CLI 실행)
#
# 맵 파일이 지켜야 하는 "규칙(스키마)"은 매우 단순하다 (envs/biped_env.py가
# 이 규칙에 따라 맵을 해석하기 때문에, 직접 맵을 작성할 때도 지켜야 함):
#   - 지면 geom의 이름은 정확히 "ground"
#   - 장애물 geom들은 이름이 "obstacle_0", "obstacle_1", "obstacle_2", ...
#     처럼 0부터 순서대로 이어져야 함 (env가 obstacle_0부터 순서대로 찾다가
#     이름이 끊기는 지점에서 멈춤)
#   - 로봇은 항상 맵의 월드 원점 (0, 0) 근처에서 스폰된다고 가정
#     (장애물 좌표를 이 원점 기준으로 배치하면 됨)
# ============================================================================

import argparse
import numpy as np
import mujoco

GROUND_SIZE = 30.0
GROUND_FRICTION = [1.2, 0.005, 0.0001]

# 로봇 자신의 몸(팔다리 등)은 기본 group(0)에 남겨두고, "센서가 감지해야 할
# 대상"인 지면/장애물은 전부 group=1로 분류해둔다. 이렇게 나눠두면
# envs/biped_env.py의 라이다 센서(mj_ray)가 geomgroup 필터로 로봇 자기
# 몸은 무시하고 장애물/지면만 정확히 감지할 수 있다
# (documents/08-lidar-sensor.md 참고).
SENSED_GEOM_GROUP = 1


def add_ground(spec):
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[GROUND_SIZE, GROUND_SIZE, 0.1],
        rgba=[0.55, 0.6, 0.55, 1.0],
        friction=GROUND_FRICTION,
        group=SENSED_GEOM_GROUP,
    )


def add_obstacles(spec, obstacles):
    """obstacles: [(x, y, half_x, half_y, height), ...] — 장애물 상자들을 맵에 추가.
    이름을 obstacle_0, obstacle_1, ... 순서로 붙이는 것이 env가 맵을 읽어들이는
    유일한 규칙이므로 반드시 이 순서/이름 규칙을 지켜야 한다.
    """
    for i, (x, y, hx, hy, hz) in enumerate(obstacles):
        spec.worldbody.add_geom(
            name=f"obstacle_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, y, hz],
            size=[hx, hy, hz],
            rgba=[0.75, 0.25, 0.2, 1.0],
            group=SENSED_GEOM_GROUP,
        )


def build_map_spec(obstacles):
    """정해진 장애물 목록으로 맵 하나를 만든다 (결정론적, 재현 가능)."""
    spec = mujoco.MjSpec()
    spec.modelname = "map"
    add_ground(spec)
    add_obstacles(spec, obstacles)
    return spec


def random_obstacles(
    rng,
    n_range=(1, 4),
    x_range=(1.5, 6.0),
    y_range=(-2.5, 2.5),
    half_xy_range=(0.15, 0.3),
    height_range=(0.5, 0.9),
    min_gap=0.9,
    max_tries=200,
):
    """무작위 장애물 배치를 하나 생성.

    단순 거부 샘플링(rejection sampling)을 쓴다: 새 장애물 후보를 무작위로
    뽑아보고, 로봇이 스폰되는 원점(0,0)이나 이미 배치된 다른 장애물과
    너무 가까우면(= min_gap보다 가까우면) 버리고 다시 뽑는다. 이렇게 하면
    "로봇이 스폰되자마자 장애물에 끼여버리는" 등의 말이 안 되는 배치를
    자연스럽게 피할 수 있다.
    """
    n = rng.integers(n_range[0], n_range[1] + 1)
    obstacles = []
    placed_xy = [(0.0, 0.0)]  # 로봇 스폰 지점도 '이미 차지된 자리'로 취급

    for _ in range(n):
        for _try in range(max_tries):
            x = rng.uniform(*x_range)
            y = rng.uniform(*y_range)
            if all(np.hypot(x - px, y - py) >= min_gap for px, py in placed_xy):
                hx = hy = rng.uniform(*half_xy_range)
                hz = rng.uniform(*height_range)
                obstacles.append((x, y, hx, hy, hz))
                placed_xy.append((x, y))
                break
        # max_tries 안에 자리를 못 찾으면 이번 장애물은 포기하고 다음으로 넘어감
    return obstacles


def random_map_spec(rng=None, **kwargs):
    """무작위 장애물 배치를 가진 맵 spec을 즉석에서(파일로 저장하지 않고) 생성.
    envs/biped_env.py가 매 에피소드 새 지형을 원할 때(randomize_map=True) 사용.
    """
    rng = rng or np.random.default_rng()
    obstacles = random_obstacles(rng, **kwargs)
    return build_map_spec(obstacles)


DEFAULT_OBSTACLES = [
    (2.0, 0.6, 0.2, 0.2, 0.8),
    (3.5, -0.7, 0.2, 0.2, 0.8),
]


def main():
    parser = argparse.ArgumentParser(description="맵(.xml) 파일을 생성해서 저장한다")
    parser.add_argument("--out", type=str, required=True, help="저장할 경로, 예: maps/random_01.xml")
    parser.add_argument("--fixed", action="store_true", help="무작위 대신 기본(2장애물) 배치 사용")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_min", type=int, default=1)
    parser.add_argument("--n_max", type=int, default=4)
    args = parser.parse_args()

    if args.fixed:
        spec = build_map_spec(DEFAULT_OBSTACLES)
    else:
        rng = np.random.default_rng(args.seed)
        spec = random_map_spec(rng, n_range=(args.n_min, args.n_max))

    spec.compile()  # MuJoCo는 "컴파일된" spec만 XML로 내보낼 수 있도록 되어 있음
    spec.to_file(args.out)
    print(f"map saved to {args.out}")


if __name__ == "__main__":
    main()
