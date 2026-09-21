# ============================================================================
# nodes/joint_state_aggregator.py — 모든 관절 값을 하나로 모아 /joint_states로
#
# 지금까지 각 WheelNode/JointNode는 자기 관절 하나의 값만 자기 토픽에
# 발행했다 (f"/joint/{name}/position" 등, 전부 std_msgs/Float64). 근데
# 실제 로봇은 관절마다 토픽을 따로 안 둔다 — ros2_control의
# joint_state_broadcaster가 로봇의 "모든" 관절 이름/위치/속도를 한 번에
# 읽어서 sensor_msgs/JointState 타입으로 f"/joint_states" 토픽 하나에
# 발행한다. RViz의 로봇 모델 표시, robot_state_publisher(TF 계산) 등
# 대부분의 소비자가 이 토픽 하나만 구독하면 로봇 전체 자세를 알 수 있다.
#
# 이 노드가 바로 그 역할이다: 개별 토픽을 다시 구독하는 대신(그래도
# 되지만, 매번 최신값을 여러 토픽에서 모으는 코드가 번거롭다), 이미
# 로봇 모델(robot.joint_positions)이 "지금 모든 관절 값"을 들고 있으므로
# 그걸 직접 읽어서 한 번에 묶어 발행한다.
# ============================================================================

from nodes.core import Node
from msgs import Header, JointState


class JointStateAggregatorNode(Node):
    def __init__(self, robot, joint_names, bus, rate_hz=30.0):
        super().__init__("joint_state_aggregator", bus)
        self.robot = robot
        self.joint_names = list(joint_names)
        self._prev_positions = {n: robot.get_joint(n) for n in self.joint_names}
        self.rate_hz = rate_hz

        self.publish_joint_states = self.create_publisher("/joint_states", msg_type=JointState)
        self.create_timer(rate_hz, self._tick)

    def _tick(self):
        dt = 1.0 / self.rate_hz
        positions, velocities = [], []
        for name in self.joint_names:
            pos = self.robot.get_joint(name)
            velocities.append((pos - self._prev_positions[name]) / dt)
            self._prev_positions[name] = pos
            positions.append(pos)

        self.publish_joint_states(JointState(
            header=Header(frame_id="base_link"),
            name=self.joint_names, position=positions, velocity=velocities,
        ))
