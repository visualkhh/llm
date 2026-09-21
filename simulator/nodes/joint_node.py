# ============================================================================
# nodes/joint_node.py — 매니퓰레이터(팔) 관절 하나당 노드 하나
#
# 팔의 각 관절(어깨/팔꿈치/손목/그리퍼 손가락 등)도 바퀴와 같은 패턴:
# "목표 위치(각도 또는 길이)"를 std_msgs/Float64 타입 토픽으로 받아서,
# 정해진 최대 속도로 그 목표를 향해 서서히 움직인다 (실제 서보모터가
# 목표각으로 부드럽게 움직이는 것과 같은 느낌).
# ============================================================================

from nodes.core import Node
from msgs import Float64


class JointNode(Node):
    def __init__(self, name, joint_name, robot, bus, rate_hz=50.0, max_speed=1.5):
        super().__init__(f"joint_{name}", bus)
        self.joint_name = joint_name
        self.robot = robot
        self.rate_hz = rate_hz
        self.max_speed = max_speed  # rad/s (revolute) 또는 m/s (prismatic)
        self.target = robot.get_joint(joint_name)

        self.create_subscription(f"/joint/{name}/cmd_position", self._on_cmd, msg_type=Float64)
        self.publish_position = self.create_publisher(f"/joint/{name}/position", msg_type=Float64)
        self.create_timer(rate_hz, self._tick)

    def _on_cmd(self, msg: Float64):
        self.target = msg.data

    def _tick(self):
        dt = 1.0 / self.rate_hz
        current = self.robot.get_joint(self.joint_name)
        diff = self.target - current
        step = max(-self.max_speed * dt, min(self.max_speed * dt, diff))
        self.robot.set_joint(self.joint_name, current + step)
        self.publish_position(Float64(data=current + step))

    @property
    def at_target(self):
        return abs(self.robot.get_joint(self.joint_name) - self.target) < 1e-3
