# ============================================================================
# msgs.py — ROS2의 표준 메시지 패키지(std_msgs, geometry_msgs, sensor_msgs,
#            nav_msgs)를 흉내낸, 타입이 있는 메시지 정의
#
# 지금까지 토픽에는 그냥 파이썬 튜플/딕셔너리/숫자를 아무거나 실어
# 보냈다 — 이건 진짜 ROS2가 하는 방식이 아니다. 실제 ROS2는 토픽마다
# "이 토픽엔 반드시 이 타입의 메시지만 흐른다"는 계약이 .msg 파일로
# 정의돼 있고(예: geometry_msgs/Twist, sensor_msgs/JointState), 다른
# 노드가 그 계약을 어기고 엉뚱한 타입을 보내면 만들 수조차 없다.
#
# 여기서는 그 대표적인 메시지 타입들을 실제 ROS2와 최대한 비슷한 이름/
# 필드 구성으로 dataclass로 정의한다. nodes/core.py의 Bus가 "이 토픽에
# 원래 선언된 타입과 실제로 보내는 메시지의 타입이 맞는지"를 검사하게
# 만들어서, 그 "계약"을 실제로 흉내낸다 (nodes/core.py의 Bus._declare_type
# 참고).
#
# 진짜 ROS2와 다른 점(의도적 단순화): .msg 파일 + 코드 생성 대신 그냥
# 파이썬 dataclass를 직접 쓴다. 그리고 DDS를 통한 실제 직렬화/네트워크
# 전송은 하지 않는다(다 같은 프로세스 안이라 필요 없다) — 하지만 "이
# 토픽은 이 타입"이라는 계약 자체는 진짜와 동일하게 강제한다.
# ============================================================================

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Header:
    """모든 ROS2 메시지가 흔히 갖는 공통 필드 — 언제 만들어졌는지(stamp)와
    어느 좌표계 기준인지(frame_id). std_msgs/Header에 해당."""
    stamp: float = field(default_factory=time.time)
    frame_id: str = ""


@dataclass
class Float64:
    """std_msgs/Float64 — 숫자 하나짜리 가장 단순한 메시지. 바퀴 하나의
    각속도 명령/피드백, 관절 하나의 목표각도처럼 "값 하나"만 필요한
    토픽에 쓴다."""
    data: float = 0.0


@dataclass
class StringMsg:
    """std_msgs/String"""
    data: str = ""


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Twist:
    """geometry_msgs/Twist — 몸체 전체의 선속도+각속도. 실제 로봇에서
    teleop/Nav2가 /cmd_vel에 이 타입으로 "이 속도로 가라"를 발행하면,
    diff_drive_controller 같은 컨트롤러가 이걸 좌/우 바퀴 각속도로
    변환한다."""
    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)


@dataclass
class Pose2D:
    """geometry_msgs/Pose2D — 평면(x, y, theta) 위치. 실제 ROS2에서는
    3D Pose가 표준이지만, 지면 주행 로봇의 "진짜 위치(ground truth)"처럼
    2D로 충분한 경우 이해하기 쉬운 이 형태를 쓴다."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass
class Odometry:
    """nav_msgs/Odometry — "내가 odom 좌표계 기준으로 어디 있다고
    추정하는지"를 담는, 오도메트리 노드의 대표 발행 메시지. ground truth
    (Pose2D)와 이름을 다르게 둔 이유: 오도메트리는 항상 "추정치"라는
    의미를 담고 있어서, 실제로는 100% 정확하지 않을 수 있다는 뜻이
    타입 이름에 그대로 드러난다."""
    header: Header
    child_frame_id: str
    pose: Pose2D
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0


@dataclass
class JointState:
    """sensor_msgs/JointState — 여러 관절의 이름/위치/속도를 한 번에 담는
    메시지. 실제 로봇도 관절마다 따로 토픽을 안 쓰고, 이 타입 하나로
    /joint_states라는 토픽에 모든 관절 값을 모아 발행한다
    (nodes/joint_state_aggregator.py 참고)."""
    header: Header
    name: list
    position: list
    velocity: list


@dataclass
class Imu:
    """sensor_msgs/Imu"""
    header: Header
    orientation_yaw: float
    angular_velocity: Vector3
    linear_acceleration: Vector3


@dataclass
class PointCloud:
    """sensor_msgs/PointCloud2를 단순화한 버전. 진짜 PointCloud2는 필드
    레이아웃까지 바이트 단위로 기술하는 이진 포맷인데, 그대로 흉내내면
    배우는 데 오히려 방해가 돼서 numpy (N,3) 배열을 그대로 들고 있는
    형태로 단순화했다 — 핵심 개념(센서 로컬 좌표계 기준 3D 점들의
    모음)은 동일하다."""
    header: Header
    points: np.ndarray


@dataclass
class Image:
    """sensor_msgs/Image + sensor_msgs/CameraInfo의 depth 버전을 하나로
    합친 단순화 버전 (실제로는 RGB와 Depth가 별도 토픽/타입인 경우가
    많지만, 이 카메라는 항상 세트로 쓰이므로 편의상 합쳤다)."""
    header: Header
    rgb: np.ndarray
    depth: np.ndarray


@dataclass
class PoseGoal:
    """geometry_msgs/PoseStamped를 내비게이션 목표 용도로 단순화 (2D만)."""
    header: Header
    x: float
    y: float


@dataclass
class MoveItCommand:
    """이 시뮬레이터 전용 커스텀 메시지. 실제 ROS2 프로젝트에서도
    표준 메시지로 표현 안 되는 자기 도메인 고유의 명령/데이터는 이렇게
    직접 .msg(여기서는 dataclass)를 새로 정의해서 쓰는 게 아주 흔하다."""
    type: str
    target: list = None
    object_id: str = None
