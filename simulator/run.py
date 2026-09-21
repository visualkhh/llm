# ============================================================================
# run.py — 이 시뮬레이터의 조립 지점(entry point)
#
# URDF와 맵 파일을 읽어서 로봇/월드를 만들고, nodes/*, localization/*,
# nav/*, moveit/* 의 모든 노드를 인스턴스화해서 Bus로 서로 연결한
# 뒤, Executor에 등록하고 viewer(GUI)를 띄운다. 이 파일 하나만 보면
# "이 시뮬레이터를 이루는 노드들이 전부 무엇이고 서로 어떻게 배선되어
# 있는지"를 한눈에 볼 수 있게 하는 게 목적이다 (ROS2의 launch 파일과
# 같은 역할).
# ============================================================================

import argparse
import os

from math3d import xyzrpy_to_matrix
from urdf.parser import parse_urdf
from robot.model import RobotModel
from robot.drive import SkidSteerDrive
from world.map_io import load_map
from nodes.core import Bus, Executor
from nodes.wheel_node import WheelNode, GroundTruthDriveNode
from nodes.joint_node import JointNode
from nodes.lidar_node import LidarNode
from nodes.depth_camera_node import DepthCameraNode
from nodes.imu_node import ImuNode
from nodes.joint_state_aggregator import JointStateAggregatorNode
from localization.odom_node import WheelOdometryNode
from localization.lidar_localization_node import LidarLocalizationNode
from localization.slam_stub import VisualOdometryNode
from nav.nav_node import NavNode
from moveit.arm_node import ArmNode
from msgs import Float64, Header, MoveItCommand, Pose2D, PoseGoal

HERE = os.path.dirname(os.path.abspath(__file__))

WHEEL_RADIUS = 0.08
TRACK_WIDTH = 0.38  # 왼쪽 바퀴 중심 <-> 오른쪽 바퀴 중심 거리

ARM_JOINTS = ["arm_base_yaw_joint", "shoulder_pitch_joint", "elbow_pitch_joint", "wrist_pitch_joint"]
ARM_JOINT_TOPICS = ["arm_yaw", "shoulder", "elbow", "wrist"]
FINGER_JOINTS = ["left_finger_joint", "right_finger_joint"]
FINGER_JOINT_TOPICS = ["left_finger", "right_finger"]


class Simulation:
    """조립된 시뮬레이터 전체를 들고 있는 컨테이너. viewer.py와
    test_headless.py 양쪽에서 이 클래스 하나로 전체를 초기화한다."""

    def __init__(self, urdf_path, map_path):
        self.description = parse_urdf(urdf_path)
        self.robot = RobotModel(self.description)
        self.world_map = load_map(map_path)
        self.bus = Bus()
        self.drive_true = SkidSteerDrive(WHEEL_RADIUS, TRACK_WIDTH)

        # ground truth 위치가 갱신될 때마다 robot(FK용 base_pose)에도 즉시
        # 반영 — 센서(라이다/카메라)는 항상 "진짜" 위치 기준으로 계산돼야
        # 하기 때문이다 (map/robot.py의 forward_kinematics 참고).
        def _on_true_pose(pose: Pose2D):
            self.robot.base_pose = xyzrpy_to_matrix([pose.x, pose.y, 0.0], [0.0, 0.0, pose.theta])
        self.bus.subscribe("/ground_truth/base_pose", _on_true_pose, msg_type=Pose2D)

        self.executor = Executor()
        self._build_nodes()

    def _build_nodes(self):
        bus, robot, world_map = self.bus, self.robot, self.world_map

        self.wheel_nodes = {
            "front_left": WheelNode("front_left", "front_left_wheel_joint", robot, bus),
            "front_right": WheelNode("front_right", "front_right_wheel_joint", robot, bus),
            "rear_left": WheelNode("rear_left", "rear_left_wheel_joint", robot, bus),
            "rear_right": WheelNode("rear_right", "rear_right_wheel_joint", robot, bus),
        }
        self.ground_truth_drive = GroundTruthDriveNode(self.drive_true, bus)

        self.arm_joint_nodes = {
            jn: JointNode(tn, jn, robot, bus)
            for jn, tn in zip(ARM_JOINTS, ARM_JOINT_TOPICS)
        }
        self.finger_joint_nodes = {
            jn: JointNode(tn, jn, robot, bus)
            for jn, tn in zip(FINGER_JOINTS, FINGER_JOINT_TOPICS)
        }

        self.lidar = LidarNode("main", "lidar_link", robot, world_map, bus, rate_hz=10.0)
        self.camera = DepthCameraNode("gripper", "camera_link", robot, world_map, bus, rate_hz=10.0)
        self.imu = ImuNode("main", self.drive_true, bus, rate_hz=100.0)

        self.odom = WheelOdometryNode(bus, WHEEL_RADIUS, TRACK_WIDTH)
        self.slam_lidar = LidarLocalizationNode(bus, world_map, self.odom)
        self.slam_visual = VisualOdometryNode(bus)

        # Nav2 역할의 NavNode는 이제 odom의 날것 추정치가 아니라, 라이다
        # 보정이 적용된 self.slam_lidar.corrected_pose()를 쓴다 — 실제
        # Nav2도 순수 odom이 아니라 amcl 등 로컬라이제이션이 보정한 위치를
        # 기준으로 움직인다.
        self.nav = NavNode("main", world_map, self.slam_lidar.corrected_pose,
                            (WHEEL_RADIUS, TRACK_WIDTH), bus)
        self.arm = ArmNode("main", robot, world_map, ARM_JOINTS, ARM_JOINT_TOPICS,
                            "gripper_base_link", FINGER_JOINTS, FINGER_JOINT_TOPICS, bus)

        all_joint_names = (["front_left_wheel_joint", "front_right_wheel_joint",
                             "rear_left_wheel_joint", "rear_right_wheel_joint"]
                            + ARM_JOINTS + FINGER_JOINTS)
        self.joint_state_aggregator = JointStateAggregatorNode(robot, all_joint_names, bus)

        for node in [*self.wheel_nodes.values(), self.ground_truth_drive,
                     *self.arm_joint_nodes.values(), *self.finger_joint_nodes.values(),
                     self.lidar, self.camera, self.imu,
                     self.odom, self.slam_lidar, self.slam_visual,
                     self.nav, self.arm, self.joint_state_aggregator]:
            self.executor.add_node(node)

    # -- 사용자/컨트롤러가 부를 편의 함수들 ----------------------------------
    def set_body_velocity(self, v, omega):
        """키보드 teleop 등에서 "이 속도로 가라"를 받아 좌/우 바퀴 각속도로
        변환해 발행한다 — 실제 ROS2의 /cmd_vel(geometry_msgs/Twist)을
        diff_drive_controller가 바퀴 명령으로 바꾸는 것과 같은 역할을
        여기서 간단히 인라인으로 수행한다."""
        left, right = self.drive_true.body_velocity_command(v, omega)
        for name in ("front_left", "rear_left"):
            self.bus.publish(f"/wheel/{name}/cmd_vel", Float64(data=left))
        for name in ("front_right", "rear_right"):
            self.bus.publish(f"/wheel/{name}/cmd_vel", Float64(data=right))

    def send_nav_goal(self, x, y):
        self.bus.publish("/nav/goal", PoseGoal(header=Header(frame_id="map"), x=x, y=y))

    def send_moveit_command(self, command: dict):
        self.bus.publish("/moveit/command", MoveItCommand(
            type=command["type"], target=command.get("target"), object_id=command.get("object_id"),
        ))

    def step(self):
        self.executor.spin_once()


def default_paths():
    return (os.path.join(HERE, "robots", "example_robot.urdf"),
            os.path.join(HERE, "maps", "example_room.json"))


def build_simulation(urdf_path=None, map_path=None):
    default_urdf, default_map = default_paths()
    return Simulation(urdf_path or default_urdf, map_path or default_map)


def main():
    parser = argparse.ArgumentParser(description="Pure-Python 로보틱스 시뮬레이터")
    parser.add_argument("--urdf", type=str, default=None, help="로봇 URDF 파일 경로 (생략 시 예시 로봇)")
    parser.add_argument("--map", type=str, default=None, help="맵 JSON 파일 경로 (생략 시 예시 방)")
    args = parser.parse_args()

    sim = build_simulation(args.urdf, args.map)
    from viewer import run_viewer
    run_viewer(sim)


if __name__ == "__main__":
    main()
