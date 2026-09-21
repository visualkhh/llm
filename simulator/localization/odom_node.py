# ============================================================================
# localization/odom_node.py — 바퀴 속도만으로 "내가 어디 있는지" 추정하는
#                              오도메트리 ComposableNode
#
# ROS2 개념 정리 (사용자 질문에 대한 답): 이런 "여러 센서값을 조합해서
# 더 고차원의 추정치(위치)를 만드는" 노드가 바로 ComposableNode다.
# nodes/wheel_node.py, lidar_node.py 같은 "센서 원값을 만드는" 노드와
# 구분해서 localization/ 아래에 모아둔 이유이기도 하다.
#
# 핵심 아이디어: 이 노드는 world/robot의 "진짜" 위치(ground truth)를
# 절대 들여다보지 않는다 — 오직 바퀴 속도 값만 갖고 적분한다. 그런데
# 그 바퀴 속도를 "읽는" 과정 자체에 실제 로봇처럼 두 종류의 오차를
# 일부러 섞는다:
#   1) 바이어스(계통 오차): 바퀴 반지름을 실제보다 살짝 잘못 알고 있다고
#      가정(캘리브레이션 오차) — 항상 같은 방향으로 조금씩 틀리게 만든다.
#   2) 노이즈(무작위 오차): 엔코더 측정값에 매 틱 작은 무작위 잡음을 더한다.
#
# 그 결과 이 노드가 계산하는 "odom 좌표계 기준 위치"는 시간이 지날수록
# (=로봇이 많이 움직일수록) 실제 위치(ground truth, map 좌표계)에서
# 점점 벌어진다 — 이게 바로 "캘리브레이션이 안 되어 있으면 화면에 보이는
# 자기 위치가 실제 지도와 어긋나 보인다"는 사용자의 말 그대로의 현상이다.
# 이 어긋남을 다시 맞추는 게 다음 단계인 SLAM/로컬라이제이션의 역할이고,
# 지금은 아직 구현 전이라 (slam_stub.py 참고) 어긋난 채로 보이는 게 정상이다.
# ============================================================================

import numpy as np

from nodes.core import ComposableNode
from robot.drive import SkidSteerDrive
from msgs import Float64, Header, Odometry, Pose2D


class WheelOdometryNode(ComposableNode):
    def __init__(self, bus, wheel_radius, track_width,
                 radius_bias=1.03, noise_std=0.01, rate_hz=50.0):
        super().__init__("odom_wheel", bus)
        # 로봇이 "스스로 알고 있다고 믿는" 바퀴 반지름은 실제보다
        # radius_bias배만큼 틀리다 (예: 1.03 = 3% 과대평가 — 타이어가
        # 마모되거나 공기압이 달라 실제 반지름이 변했는데 소프트웨어
        # 설정은 안 바뀐 상황을 흉내낸 것).
        self.estimator = SkidSteerDrive(wheel_radius * radius_bias, track_width)
        self.noise_std = noise_std
        self.rate_hz = rate_hz
        self.left_vel_raw = 0.0
        self.right_vel_raw = 0.0
        self.rng = np.random.default_rng(1)

        self.create_subscription("/wheel/front_left/velocity", self._set_left, msg_type=Float64)
        self.create_subscription("/wheel/rear_left/velocity", self._set_left, msg_type=Float64)
        self.create_subscription("/wheel/front_right/velocity", self._set_right, msg_type=Float64)
        self.create_subscription("/wheel/rear_right/velocity", self._set_right, msg_type=Float64)
        self.publish_odom = self.create_publisher("/odom/base_pose", msg_type=Odometry)
        self.create_timer(rate_hz, self._tick)

    def _set_left(self, msg: Float64):
        self.left_vel_raw = msg.data

    def _set_right(self, msg: Float64):
        self.right_vel_raw = msg.data

    def _tick(self):
        dt = 1.0 / self.rate_hz
        left_measured = self.left_vel_raw + self.rng.normal(0, self.noise_std)
        right_measured = self.right_vel_raw + self.rng.normal(0, self.noise_std)
        v, omega = self.estimator.step(left_measured, right_measured, dt)
        self.publish_odom(Odometry(
            header=Header(frame_id="odom"), child_frame_id="base_link",
            pose=Pose2D(x=self.estimator.x, y=self.estimator.y, theta=self.estimator.yaw),
            linear_velocity=v, angular_velocity=omega,
        ))

    @property
    def estimated_pose(self):
        return self.estimator.x, self.estimator.y, self.estimator.yaw

    @property
    def is_moving(self):
        return abs(self.left_vel_raw) + abs(self.right_vel_raw) > 1e-3
