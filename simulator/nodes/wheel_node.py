# ============================================================================
# nodes/wheel_node.py — 바퀴 4개, 각각 자기만의 노드
#
# 실제 4륜 로봇도 보통 바퀴마다 모터 드라이버가 따로 있고, 각 모터가
# "지금 이 각속도로 돌아라"라는 명령을 받아 독립적으로 회전한다 — 그래서
# 사용자 요청대로 바퀴 하나당 노드 하나(WheelNode)를 만든다.
#
# 이 노드가 하는 일 두 가지:
#   1) f"/wheel/{name}/cmd_vel" 토픽을 구독해서 "목표 각속도(rad/s)"를 받는다
#      (std_msgs/Float64 타입 — 바퀴 하나의 각속도는 숫자 하나면 충분하다).
#   2) 자기 주기(rate_hz)마다 그 각속도만큼 URDF 관절 각도를 돌리고
#      (바퀴가 빙글빙글 도는 시각적 효과), 지금 각속도를
#      f"/wheel/{name}/velocity"에 발행한다 — 이걸 GroundTruthDriveNode가
#      모아서 로봇 전체가 얼마나 움직였는지 적분한다.
#
# 실제 ros2_control이었다면 이 클래스 하나가 "하드웨어 인터페이스"
# (모터 드라이버에 명령 쓰기 + 엔코더 읽기를 한 번에 처리)에 해당한다 —
# 자세한 설명은 README.md의 "핵심 설계 결정" 참고.
# ============================================================================

import math

from nodes.core import Node
from msgs import Float64, Pose2D


class WheelNode(Node):
    def __init__(self, name, joint_name, robot, bus, rate_hz=50.0):
        super().__init__(f"wheel_{name}", bus)
        self.joint_name = joint_name
        self.robot = robot
        self.rate_hz = rate_hz
        self.angular_velocity = 0.0  # rad/s, 양수 = 전진 방향 회전

        self.create_subscription(f"/wheel/{name}/cmd_vel", self._on_cmd, msg_type=Float64)
        self.publish_velocity = self.create_publisher(f"/wheel/{name}/velocity", msg_type=Float64)
        self.create_timer(rate_hz, self._tick)

    def _on_cmd(self, msg: Float64):
        self.angular_velocity = msg.data

    def _tick(self):
        dt = 1.0 / self.rate_hz
        angle = self.robot.get_joint(self.joint_name) + self.angular_velocity * dt
        # continuous 관절이라 각도를 -pi~pi로 감쌀 필요는 없다 (회전 자체가
        # 시각화용일 뿐이라 계속 누적돼도 무방, 단 너무 커지지 않게 감아준다).
        angle = math.remainder(angle, 2 * math.pi)
        self.robot.set_joint(self.joint_name, angle)
        self.publish_velocity(Float64(data=self.angular_velocity))


class GroundTruthDriveNode(Node):
    """4개 바퀴 노드가 발행하는 (명령대로 정확한) 속도를 모아 "로봇이 실제로
    map 좌표계에서 어디에 있는지" 적분하는 노드.

    ⚠️ 이건 로봇 자신은 알 수 없는 "신(神)의 시점" 정답값이다 — 시뮬레이터가
    센서(라이다/카메라) 값을 만들거나, 화면에 "실제 위치"를 그릴 때만 쓴다.
    로봇이 스스로 추정하는 위치(오차가 섞인, odom 좌표계)는
    localization/odom_node.py가 따로 계산한다 — 이 둘이 시간이 지나며
    조금씩 벌어지는 것 자체가 "왜 SLAM/보정이 필요한가"를 보여준다.

    발행 타입이 nav_msgs/Odometry가 아니라 geometry_msgs/Pose2D인 이유:
    "오도메트리"라는 이름 자체가 이미 "추정치"라는 뜻을 담고 있어서,
    이건 추정이 아니라 정답이라는 걸 타입 이름으로도 구분해뒀다.
    """

    def __init__(self, drive, bus, rate_hz=50.0):
        super().__init__("ground_truth_drive", bus)
        self.drive = drive
        self.rate_hz = rate_hz
        self.left_vel = 0.0
        self.right_vel = 0.0

        self.create_subscription("/wheel/front_left/velocity", self._set_left, msg_type=Float64)
        self.create_subscription("/wheel/rear_left/velocity", self._set_left, msg_type=Float64)
        self.create_subscription("/wheel/front_right/velocity", self._set_right, msg_type=Float64)
        self.create_subscription("/wheel/rear_right/velocity", self._set_right, msg_type=Float64)
        self.publish_true_pose = self.create_publisher("/ground_truth/base_pose", msg_type=Pose2D)
        self.create_timer(rate_hz, self._tick)

    def _set_left(self, msg: Float64):
        self.left_vel = msg.data

    def _set_right(self, msg: Float64):
        self.right_vel = msg.data

    def _tick(self):
        dt = 1.0 / self.rate_hz
        self.drive.step(self.left_vel, self.right_vel, dt)
        self.publish_true_pose(Pose2D(x=self.drive.x, y=self.drive.y, theta=self.drive.yaw))
