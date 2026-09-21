# ============================================================================
# nav/nav_node.py — Nav2의 "목표 지점까지 실제로 움직이기"에 해당하는 노드
#
# 동작 순서 (Nav2와 동일한 큰 흐름):
#   1) 목표 좌표(x, y)를 받으면 planner.a_star()로 전역 경로(웨이포인트
#      나열)를 한 번 계산한다.
#   2) 매 컨트롤 주기마다 "지금 내가 어디 있다고 생각하는지"(=odom 추정
#      위치 — Nav2도 실제로는 map의 '진�자' 위치가 아니라 로컬라이제이션이
#      추정한 위치를 쓴다)와 "다음 웨이포인트"를 비교해서, 그쪽을 향하도록
#      바퀴 속도를 계산해 발행한다 (아주 단순한 pure-pursuit 스타일 P제어).
#
# 중요한 포인트: 이 노드는 "지금 내가 어디 있다고 생각하는지"를 직접
# 계산하지 않고, 밖에서 넘겨준 pose_fn()을 그대로 믿는다 (의존성 주입 —
# 어떤 로컬라이제이션을 쓰든 NavNode 코드는 안 바뀐다). run.py에서는
# localization/lidar_localization_node.py의 corrected_pose를 넘겨준다
# (odom 날것 그대로가 아니라 라이다 보정이 적용된 값). 그래도 그 추정이
# 100% 정확하진 않으므로, "내비게이션이 목표에 도착했다고 생각하는 지점"과
# "실제로 도착한 지점"이 완전히 같지는 않을 수 있다 — 이게 바로 실제
# 로봇에서 로컬라이제이션 정확도가 중요한 이유다.
# ============================================================================

import numpy as np

from nodes.core import Node
from nav.planner import OccupancyGrid, a_star
from msgs import Float64, PoseGoal


class NavNode(Node):
    def __init__(self, name, world_map, pose_fn, drive_params, bus,
                 rate_hz=20.0, goal_tolerance=0.12, waypoint_tolerance=0.35,
                 max_v=0.6, max_omega=1.5):
        super().__init__(f"nav_{name}", bus)
        self.grid = OccupancyGrid(world_map)
        self.pose_fn = pose_fn  # () -> (x, y, yaw), "지금 내가 어디 있다고 믿는지"
        self.wheel_radius, self.track_width = drive_params
        self.goal_tolerance = goal_tolerance
        self.waypoint_tolerance = waypoint_tolerance
        self.max_v, self.max_omega = max_v, max_omega

        self.path = None
        self.wp_idx = 0
        self.status = "idle"  # idle | planning_failed | moving | reached
        self._stopped = True  # 마지막으로 "정지" 명령을 이미 내렸는지 (중복 발행 방지)

        self.create_subscription("/nav/goal", self.set_goal, msg_type=PoseGoal)
        self.publish_left = self.create_publisher_pair("front_left", "rear_left")
        self.publish_right = self.create_publisher_pair("front_right", "rear_right")
        self.create_timer(rate_hz, self._tick)

    def create_publisher_pair(self, name_a, name_b):
        """스키드-스티어라 한쪽 바퀴 2개(앞/뒤)는 항상 같은 속도로 도므로,
        cmd_vel을 두 바퀴에 동시에 보내는 헬퍼."""
        pub_a = self.create_publisher(f"/wheel/{name_a}/cmd_vel", msg_type=Float64)
        pub_b = self.create_publisher(f"/wheel/{name_b}/cmd_vel", msg_type=Float64)
        return lambda v: (pub_a(Float64(data=v)), pub_b(Float64(data=v)))

    def set_goal(self, goal: PoseGoal):
        x, y, _ = self.pose_fn()
        path = a_star(self.grid, (x, y), (goal.x, goal.y))
        if path is None:
            self.status = "planning_failed"
            self.path = None
            return
        self.path = path
        self.wp_idx = 0
        self.status = "moving"
        self._stopped = False

    def _tick(self):
        # idle이거나 이미 정지 명령을 낸 상태라면 바퀴 토픽에 아예 손대지
        # 않는다 — 그래야 수동 조작(teleop) 같은 다른 명령 소스가 계속
        # 바퀴를 제어할 수 있다 (실제 ROS2에서 twist_mux가 하는 "누가 지금
        # 바퀴의 주인인가" 중재를 여기서는 "내가 할 일이 없으면 아무것도
        # 발행하지 않는다"는 규칙으로 아주 단순화한 것).
        if self.status != "moving" or not self.path:
            if not self._stopped and self.status in ("reached", "planning_failed"):
                self.publish_left(0.0)
                self.publish_right(0.0)
                self._stopped = True
            return

        x, y, yaw = self.pose_fn()
        target = self.path[self.wp_idx]
        dist = np.hypot(target[0] - x, target[1] - y)

        if self.wp_idx == len(self.path) - 1 and dist < self.goal_tolerance:
            self.status = "reached"
            self.publish_left(0.0)
            self.publish_right(0.0)
            self._stopped = True
            return
        if dist < self.waypoint_tolerance and self.wp_idx < len(self.path) - 1:
            self.wp_idx += 1
            target = self.path[self.wp_idx]

        target_yaw = np.arctan2(target[1] - y, target[0] - x)
        yaw_error = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))

        # 목표 방향과 많이 틀어져 있으면 제자리에서 먼저 돌고, 어느 정도
        # 정면을 향하면 전진하면서 미세 조향 (실제 로봇 내비게이션에서도
        # 흔한 전략 — 급격한 회전 중에 전진까지 섞으면 경로를 크게 벗어남).
        if abs(yaw_error) > 0.6:
            v, omega = 0.0, np.clip(2.0 * yaw_error, -self.max_omega, self.max_omega)
        else:
            v = self.max_v
            omega = np.clip(2.0 * yaw_error, -self.max_omega, self.max_omega)

        left = (v - omega * self.track_width / 2.0) / self.wheel_radius
        right = (v + omega * self.track_width / 2.0) / self.wheel_radius
        self.publish_left(left)
        self.publish_right(right)
