# ============================================================================
# robot/model.py — URDF로 읽어들인 로봇을 "지금 관절 값이 이럴 때 각 부품이
#                   어디에 있는지" 계산할 수 있는 살아있는 객체로 만들기
#
# ROS2에서는 이 역할을 tf2(transform tree)가 한다: 각 관절의 현재 값을
# 알고 있으면, "map -> odom -> base_link -> arm_link1 -> ... -> camera_link"
# 체인을 따라 행렬을 곱해서 임의의 두 좌표계 사이의 변환을 구할 수 있다.
# 이 파일의 RobotModel이 그 역할을 한다: URDF 트리 + 각 관절의 현재값을
# 들고 있다가, 요청이 오면 forward kinematics(정기구학)를 계산해준다.
# ============================================================================

import numpy as np

from math3d import xyzrpy_to_matrix, joint_transform


class RobotModel:
    def __init__(self, description):
        self.desc = description
        # 관절 이름 -> 현재 값(revolute/continuous는 라디안, prismatic은 미터)
        self.joint_positions = {name: 0.0 for name in description.joints}
        # base_link가 "odom" 좌표계 기준으로 지금 어디 있는지 (바퀴로 움직이는
        # 로봇이므로 이것만 따로 관리한다 — 나머지 관절들은 URDF 트리를 그대로 탄다).
        self.base_pose = np.eye(4)

    def set_joint(self, name, value):
        joint = self.desc.joints[name]
        if joint.has_limit:
            value = float(np.clip(value, joint.lower, joint.upper))
        self.joint_positions[name] = value

    def get_joint(self, name):
        return self.joint_positions[name]

    def forward_kinematics(self):
        """지금 관절 값들로 모든 링크의 "odom 좌표계 기준" 4x4 변환행렬을
        계산해서 {link_name: matrix} 딕셔너리로 반환한다.

        base_link는 self.base_pose(바퀴 주행으로 갱신됨)에서 시작하고,
        그 뒤로는 URDF의 부모->자식 관절을 루트부터 순서대로 타고 내려가며
        "부모 변환 @ 관절 origin @ 관절 움직임"을 누적 곱셈한다.
        """
        transforms = {self.desc.root_link: self.base_pose.copy()}
        self._recurse(self.desc.root_link, transforms)
        return transforms

    def _recurse(self, link_name, transforms):
        parent_t = transforms[link_name]
        for joint in self.desc.children_of.get(link_name, []):
            origin_t = xyzrpy_to_matrix(joint.origin.xyz, joint.origin.rpy)
            value = self.joint_positions[joint.name]
            move_t = joint_transform(joint.type, joint.axis, value)
            child_t = parent_t @ origin_t @ move_t
            transforms[joint.child] = child_t
            self._recurse(joint.child, transforms)

    def link_pose(self, link_name):
        """특정 링크 하나의 변환행렬만 필요할 때 (매번 전체를 계산하긴
        하지만, 이 시뮬레이터 규모에서는 충분히 빠르다)."""
        return self.forward_kinematics()[link_name]

    def joints_with_type(self, joint_type):
        return [j for j in self.desc.joints.values() if j.type == joint_type]
