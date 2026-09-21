# ============================================================================
# nodes/imu_node.py — 몸통 IMU(관성측정장치) 노드
#
# 실제 IMU는 각속도(자이로)와 선형가속도(가속도계)를 아주 높은 주기로
# 측정한다. 이 시뮬레이터는 바퀴 운동학(SkidSteerDrive)으로 "실제" 속도를
# 이미 알고 있으므로, 그 값에 약간의 센서 노이즈를 섞어서 IMU 값을
# 만들어낸다 — 실제 IMU도 완벽하지 않고 노이즈/바이어스가 있다는 걸
# 그대로 흉내낸 것 (이 노이즈 때문에 IMU만으로 적분한 위치도 시간이
# 지나면 틀어진다 — 오도메트리 노드가 IMU 대신 바퀴를 주로 쓰는 이유).
# ============================================================================

import numpy as np

from nodes.core import Node
from msgs import Header, Imu, Vector3


class ImuNode(Node):
    def __init__(self, name, drive, bus, rate_hz=100.0, noise_std=0.02):
        super().__init__(f"imu_{name}", bus)
        self.drive = drive
        self.rate_hz = rate_hz
        self.noise_std = noise_std
        self._prev_yaw = drive.yaw
        self._prev_v = 0.0
        self.rng = np.random.default_rng(0)

        self.publish_imu = self.create_publisher(f"/imu/{name}/data", msg_type=Imu)
        self.create_timer(rate_hz, self._tick)

    def _tick(self):
        dt = 1.0 / self.rate_hz
        yaw = self.drive.yaw
        angular_velocity_z = (yaw - self._prev_yaw) / dt + self.rng.normal(0, self.noise_std)
        # 평지 주행이라 roll/pitch는 0에 노이즈만 섞어서 흉내낸다.
        angular_velocity = np.array([
            self.rng.normal(0, self.noise_std * 0.2),
            self.rng.normal(0, self.noise_std * 0.2),
            angular_velocity_z,
        ])
        linear_acceleration = np.array([
            self.rng.normal(0, self.noise_std),
            self.rng.normal(0, self.noise_std),
            9.81 + self.rng.normal(0, self.noise_std),  # 중력 (z 위쪽이 +)
        ])
        self._prev_yaw = yaw
        self.publish_imu(Imu(
            header=Header(frame_id="imu_link"),
            orientation_yaw=yaw,  # 단순화를 위해 롤/피치 없이 yaw만 (평지 주행 가정)
            angular_velocity=Vector3(*angular_velocity),
            linear_acceleration=Vector3(*linear_acceleration),
        ))
