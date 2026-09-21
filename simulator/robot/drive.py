# ============================================================================
# robot/drive.py — 바퀴 4개의 회전 속도로부터 로봇 몸체가 얼마나
#                   움직였는지 계산하는 스키드-스티어(skid-steer) 운동학
#
# 이 로봇은 4륜인데 조향 관절이 따로 없으므로(각 바퀴는 그냥 "회전"만
# 함), 탱크처럼 왼쪽 바퀴들과 오른쪽 바퀴들의 속도 차이로 방향을 튼다
# (skid-steer). 왼쪽 앞/뒤 바퀴는 항상 같은 속도로, 오른쪽 앞/뒤도
# 마찬가지로 같은 속도로 돈다고 가정한다(실제 로봇도 보통 좌/우를
# 기계적으로 동기화하거나 소프트웨어로 그렇게 명령한다).
#
#   v(전진속도)     = wheel_radius * (w_left + w_right) / 2
#   omega(회전속도) = wheel_radius * (w_right - w_left) / track_width
#
# 이걸 매 스텝(dt) 적분하면 "odom 좌표계 기준 base_link의 위치"가 나온다
# — 이게 바로 바퀴만으로 계산한 오도메트리(wheel odometry)의 핵심 수식이다.
# ============================================================================

import numpy as np

from math3d import xyzrpy_to_matrix


class SkidSteerDrive:
    def __init__(self, wheel_radius, track_width):
        self.wheel_radius = wheel_radius
        self.track_width = track_width
        # odom 좌표계 기준 (x, y, yaw) — 오도메트리 그 자체.
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

    def step(self, left_wheel_angular_vel, right_wheel_angular_vel, dt):
        v = self.wheel_radius * (left_wheel_angular_vel + right_wheel_angular_vel) / 2.0
        omega = self.wheel_radius * (right_wheel_angular_vel - left_wheel_angular_vel) / self.track_width
        # 회전 중심을 기준으로 곡선 경로를 적분 (오일러 적분으로도 이
        # 시뮬레이터 스케일에서는 충분히 정확함).
        self.x += v * np.cos(self.yaw) * dt
        self.y += v * np.sin(self.yaw) * dt
        self.yaw += omega * dt
        return v, omega

    def pose_matrix(self):
        return xyzrpy_to_matrix([self.x, self.y, 0.0], [0.0, 0.0, self.yaw])

    def body_velocity_command(self, v, omega):
        """"전진 속도 v, 회전 속도 omega로 가라"를 좌/우 바퀴 각속도로
        변환 (내비게이션 컨트롤러가 이 형태로 명령을 내리는 게 더 직관적)."""
        left = (v - omega * self.track_width / 2.0) / self.wheel_radius
        right = (v + omega * self.track_width / 2.0) / self.wheel_radius
        return left, right
