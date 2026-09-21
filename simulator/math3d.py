# ============================================================================
# math3d.py — 이 시뮬레이터 전체가 쓰는 3D 변환(transform) 수학
#
# ROS2/URDF에서 로봇의 모든 위치는 "어떤 좌표계(frame)를 기준으로 한
# 4x4 동차변환행렬(homogeneous transform matrix)"로 표현한다. 예를 들어
# "map -> odom -> base_link -> ... -> camera_link" 처럼 좌표계가
# 체인으로 연결되어 있고, 그 체인을 따라 행렬을 곱해나가면(forward
# kinematics) 최종적으로 "카메라가 map 좌표계에서 어디에, 어느 방향을
# 보고 있는지"를 알 수 있다.
#
# 여기서는 외부 로보틱스 라이브러리 없이 numpy만으로 그 최소한을 직접
# 구현한다: xyz(위치) + rpy(roll/pitch/yaw, 오일러각) -> 4x4 행렬 변환,
# 행렬 합성(자식 좌표계를 부모 좌표계 기준으로 바꾸기), 역행렬(회전만
# 있는 강체변환이라 전치+역변환으로 빠르게 계산 가능).
# ============================================================================

import numpy as np


def rpy_to_matrix(roll, pitch, yaw):
    """오일러각(roll=x축, pitch=y축, yaw=z축 순서 회전, URDF의 <origin rpy="..."/>
    규약과 동일: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)) -> 3x3 회전행렬."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def xyzrpy_to_matrix(xyz, rpy):
    """위치(xyz) + 오일러각(rpy) -> 4x4 동차변환행렬."""
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(*rpy)
    t[:3, 3] = xyz
    return t


def translation_matrix(xyz):
    t = np.eye(4)
    t[:3, 3] = xyz
    return t


def rotation_matrix_axis_angle(axis, angle):
    """임의의 회전축(axis, 단위벡터) 기준으로 angle(rad)만큼 회전하는 3x3
    행렬 (Rodrigues' rotation formula). URDF의 revolute/continuous
    관절은 "각 관절 고유의 회전축"을 가지므로(꼭 x/y/z축이 아닐 수 있음)
    이 일반형이 필요하다."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def joint_transform(joint_type, axis, value):
    """관절이 지금 value(각도 또는 길이) 상태일 때, 관절의 origin 좌표계
    기준으로 얼마나 더 움직였는지를 나타내는 4x4 행렬.
    (관절의 origin(고정 오프셋)은 별도로 곱해준다 — 여기서는 "관절 자체의
    움직임"만 계산한다.)
    """
    if joint_type == "fixed":
        return np.eye(4)
    if joint_type in ("revolute", "continuous"):
        t = np.eye(4)
        t[:3, :3] = rotation_matrix_axis_angle(np.asarray(axis, dtype=float), value)
        return t
    if joint_type == "prismatic":
        return translation_matrix(np.asarray(axis, dtype=float) * value)
    raise ValueError(f"알 수 없는 관절 타입: {joint_type}")


def invert_transform(t):
    """강체변환(회전+이동)의 역행렬. 일반 역행렬 계산보다 훨씬 빠르고
    수치적으로 안정적이다: R^-1 = R^T, 이동은 -R^T @ p."""
    r = t[:3, :3]
    p = t[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ p
    return inv


def transform_point(t, point):
    """4x4 변환행렬로 3D 점 하나(또는 (N,3) 배열)를 변환한다."""
    point = np.asarray(point, dtype=float)
    if point.ndim == 1:
        return (t[:3, :3] @ point) + t[:3, 3]
    return (t[:3, :3] @ point.T).T + t[:3, 3]


def transform_direction(t, direction):
    """방향벡터(이동 성분은 무시하고 회전만 적용)를 변환한다."""
    return t[:3, :3] @ np.asarray(direction, dtype=float)


def matrix_to_xyz_yaw(t):
    """4x4 행렬에서 2D 평면(바퀴 로봇의 위치 추정 등)에 필요한 x, y, yaw만
    뽑아낸다. roll/pitch는 지면 주행 로봇에서는 보통 0에 가깝다고 가정."""
    x, y = t[0, 3], t[1, 3]
    yaw = np.arctan2(t[1, 0], t[0, 0])
    return x, y, yaw
