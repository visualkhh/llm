# ============================================================================
# moveit/arm_node.py — MoveIt처럼 "팔을 움직여라/집어라" 명령을
#                             받아서 IK로 풀고 각 관절 노드에 배분하는 노드
#
# f"/moveit/command" 토픽으로 아래 두 종류의 명령을 받는다:
#   {"type": "move_gripper", "target": [x, y, z]}   # 그리퍼를 이 좌표로
#   {"type": "pick", "object_id": "cube_1"}          # 이 물체를 집어라
#
# "집어라" 명령은 실제 로봇처럼 몇 단계로 나눠 진행한다 (hover -> descend
# -> grasp -> lift): 바로 순간이동하지 않고, 물체 위로 먼저 간 다음
# 내려가서 집고 다시 드는 순서 — 그래야 화면에서 "집는 동작"처럼 보인다.
#
# ⚠️ 실제로 부드럽게 "움직이는" 것처럼 보이려면 두 가지가 다 필요하다
# (처음엔 이 둘을 안 챙겨서 관절이 목표로 순간이동해버리는 버그가 있었다):
#   1) IK 자체는 robot.joint_positions를 직접 여러 번 바꿔가며 수렴시키는
#      알고리즘이라(moveit/ik.py 참고), 그 계산이 끝나는 순간 화면에 이미
#      "다 움직인 최종 자세"가 보여버린다 — 그래서 IK가 다 풀린 뒤에는
#      풀기 전 각도로 로봇을 원상복구시키고, 계산해낸 목표값만 각
#      JointNode의 cmd_position 토픽으로 "발행"한다. 실제 움직임은
#      JointNode가 담당하는 속도 제한 보간(목표까지 서서히 이동)이 그린다.
#   2) 시퀀스(hover -> descend -> ...)의 다음 단계로, 팔이 실제로 그
#      자세에 "도착"하기도 전에 바로 넘어가면 마찬가지로 중간 과정이
#      안 보인다. 그래서 각 단계마다 관절 값이 목표에 충분히 가까워질
#      때까지 기다렸다가 다음 단계로 넘어간다 (_arrived() 참고).
# ============================================================================

import numpy as np

from nodes.core import Node
from moveit.ik import IKSolver
from msgs import Float64, MoveItCommand, StringMsg

ARRIVE_TOLERANCE = 0.02   # rad(관절) — 목표와 이 정도 이내면 "도착"으로 침
MAX_WAIT_TICKS = 60       # 혹시 수렴이 안 돼도 이 틱 수가 지나면 강제로 다음 단계로 (무한 대기 방지)


class ArmNode(Node):
    def __init__(self, name, robot, world_map, arm_joint_names, joint_topic_names,
                 gripper_link, finger_joint_names, finger_joint_topics, bus, rate_hz=20.0):
        super().__init__(f"arm_{name}", bus)
        self.robot = robot
        self.world_map = world_map
        self.arm_joint_names = arm_joint_names
        self.joint_topic_names = joint_topic_names  # arm_joint_names와 순서 같은 "/joint/{topic}/..." 이름들
        self.finger_joint_names = finger_joint_names
        self.gripper_link = gripper_link
        self.ik = IKSolver(robot, arm_joint_names, gripper_link)

        self._publishers = {jn: self.create_publisher(f"/joint/{tn}/cmd_position", msg_type=Float64)
                             for jn, tn in zip(arm_joint_names, joint_topic_names)}
        # ⚠️ 여기서 관절 이름(예: "left_finger_joint")을 그대로 토픽 이름으로
        # 쓰면 안 된다 — JointNode는 run.py에서 FINGER_JOINT_TOPICS(짧은
        # 이름, 예: "left_finger")로 구독하고 있어서, 관절 이름 그대로
        # 발행하면 서로 다른 토픽이 되어 명령이 전달되지 않는다(실제로
        # 이 버그 때문에 손가락이 전혀 안 움직이고 있었다). 팔 관절과
        # 똑같이 "joint_name -> topic_name" 매핑을 그대로 써야 한다.
        self._finger_publishers = [self.create_publisher(f"/joint/{tn}/cmd_position", msg_type=Float64)
                                    for tn in finger_joint_topics]

        self.status = "idle"
        self._last_fail_distance = None  # 마지막으로 IK가 실패했을 때, 목표가 팔 축 기준 얼마나 멀었는지(m)
        self.attached_object = None   # 지금 그리퍼가 붙잡고 있는 Shape (world_map.objects 중 하나)
        self._sequence = []           # 진행 중인 다단계 동작 (콜백 리스트)
        self._pending_targets = None  # {joint_name: 목표값} — 지금 이 목표에 도착하길 기다리는 중이면 채워짐
        self._wait_ticks = 0
        self.publish_status = self.create_publisher(f"/moveit/{name}/status", msg_type=StringMsg)

        self.create_subscription("/moveit/command", self._on_command, msg_type=MoveItCommand)
        self.create_timer(rate_hz, self._tick)

    # -- 관절 목표를 실제로 로봇에 적용하는 공통 루틴 -----------------------
    def _move_to(self, target_xyz):
        # IK는 robot.joint_positions를 직접 수렴시키는 알고리즘이라, 풀고
        # 나면 화면에 "최종 자세"가 이미 보여버린다. 그래서 풀기 전 값을
        # 저장해뒀다가 계산이 끝나면 원상복구하고, 목표값만 JointNode에
        # 넘겨서 실제 움직임(=화면에 보이는 변화)은 JointNode의 서서히
        # 이동하는 로직이 담당하게 한다.
        backup = {name: self.robot.get_joint(name) for name in self.arm_joint_names}
        pivot_pos = self.robot.link_pose("arm_pivot_link")[:3, 3]
        ok = self.ik.solve(target_xyz)
        targets = {name: self.robot.get_joint(name) for name in self.arm_joint_names}
        for name, value in backup.items():
            self.robot.set_joint(name, value)
        if not ok:
            self._last_fail_distance = float(np.linalg.norm(np.asarray(target_xyz) - pivot_pos))
            return False
        for name, value in targets.items():
            self._publishers[name](Float64(data=value))
        self._pending_targets = targets
        return True

    def _set_gripper(self, opening):
        for pub in self._finger_publishers:
            pub(Float64(data=opening))
        self._pending_targets = {name: opening for name in self.finger_joint_names}
        return True

    # -- 외부 명령 처리 ------------------------------------------------------
    def _on_command(self, command: MoveItCommand):
        ctype = command.type
        if ctype == "move_gripper":
            self._sequence = [lambda: self._move_to(np.array(command.target))]
            self.status = "moving"
        elif ctype == "pick":
            obj = self.world_map.find_object(command.object_id)
            if obj is None:
                known = [o.id for o in self.world_map.objects]
                self.status = f"error: object '{command.object_id}' not found (있는 것: {known})"
                return
            above = obj.xyz + np.array([0, 0, 0.15])
            self._sequence = [
                lambda: self._set_gripper(0.04),          # 손가락 벌리기
                lambda: self._move_to(above),              # 물체 위로 이동
                lambda: self._move_to(obj.xyz),             # 내려가기
                lambda: self._set_gripper(0.0),             # 손가락 오므리기(집기)
                lambda: self._attach(obj),
                lambda: self._move_to(above),                # 들어올리기
            ]
            self.status = "picking"
        elif ctype == "place":
            target = np.array(command.target)
            self._sequence = [
                lambda: self._move_to(target + np.array([0, 0, 0.1])),
                lambda: self._move_to(target),
                lambda: self._set_gripper(0.04),
                lambda: self._detach(target),
            ]
            self.status = "placing"

    def _attach(self, obj):
        self.attached_object = obj
        return True

    def _detach(self, target_xyz):
        if self.attached_object is not None:
            self.attached_object.xyz = np.array(target_xyz)
        self.attached_object = None
        return True

    def _arrived(self):
        if self._pending_targets is None:
            return True
        return all(abs(self.robot.get_joint(name) - target) < ARRIVE_TOLERANCE
                   for name, target in self._pending_targets.items())

    def _tick(self):
        if self.attached_object is not None:
            # 집고 있는 물체는 매 틱 그리퍼 위치를 그대로 따라가게 한다
            # (진짜 물리적 접촉/마찰이 아니라 "운동학적으로 붙어있다"는
            # 단순화 — Kinematic-only 시뮬레이션 범위에 맞춘 결정이다).
            self.attached_object.xyz = self.robot.link_pose(self.gripper_link)[:3, 3]

        if self._pending_targets is not None:
            self._wait_ticks += 1
            if not self._arrived() and self._wait_ticks < MAX_WAIT_TICKS:
                self.publish_status(StringMsg(data=self.status))
                return  # 아직 이전 단계 동작이 안 끝났다 — 다음 단계로 넘어가지 않고 기다린다
            self._pending_targets = None
            self._wait_ticks = 0
            if not self._sequence:
                # 방금 끝낸 게 시퀀스의 마지막 단계였다 — 여기서 바로
                # idle로 바꿔야 한다. (아래 "if self._sequence:" 블록은
                # 시퀀스가 이미 비어있으면 아예 안 들어가므로, 여기서
                # 처리 안 하면 상태가 영원히 "picking" 등으로 남아버린다.)
                self.status = "idle"

        if self._sequence:
            step = self._sequence.pop(0)
            ok = step()
            if not ok:
                # IK가 목표에 못 닿았다 — 나머지 시퀀스(집기/놓기 등)를 다
                # 취소한다. 실제로 안 닿은 채로 "집었다"고 거짓 보고하면 안 되니까.
                self._sequence.clear()
                self._pending_targets = None
                dist_msg = (f" (팔 축 기준 {self._last_fail_distance:.2f}m 떨어져 있음 — "
                            f"goto로 더 가까이 간 다음 다시 시도)") if self._last_fail_distance else ""
                self.status = f"error: target unreachable{dist_msg}"
            elif not self._sequence and self._pending_targets is None:
                self.status = "idle"
        self.publish_status(StringMsg(data=self.status))
