# ============================================================================
# moveit/ik.py — MoveIt의 "역기구학(IK) 풀이"에 해당하는 최소 구현
#
# "그리퍼를 이 좌표로 보내고 싶다"는 목표(target_xyz)가 있을 때, 그걸
# 만족하는 각 관절의 각도를 거꾸로 계산하는 게 역기구학(IK)이다. 여기서는
# 수치적(numerical) 방법을 쓴다 — 해석적(공식으로 한 번에 풀기)으로
# 풀려면 팔의 기구학적 구조마다 사람이 매번 새로 수식을 유도해야 해서
# 범용성이 없다. 대신:
#
#   1) 지금 관절 각도에서 손끝(gripper) 위치를 계산한다 (FK).
#   2) 관절 하나하나를 아주 조금씩 움직여보면서 "이 관절이 손끝을 얼마나,
#      어느 방향으로 움직이는가"를 수치미분으로 근사한다 (야코비안, Jacobian).
#   3) "목표 방향으로 가려면 각 관절을 얼마나 움직여야 하는가"를 야코비안의
#      의사역행렬로 풀고, 그만큼 관절을 움직인다.
#   4) 손끝이 목표에 충분히 가까워질 때까지 1~3을 반복한다.
#
# 감쇠최소제곱(damped least squares)을 쓰는 이유: 팔이 목표에 닿을 수
# 없는 자세(특이점, singularity) 근처에서 야코비안이 불안정해지는데,
# 감쇠항(lambda)이 그 불안정을 눌러줘서 관절이 미친 듯이 튀는 걸 막아준다.
#
# ⚠️ 멀티스타트(multi-start)가 필요한 이유: 이 방법은 "지금 자세에서
# 시작해서 조금씩 목표로 다가가는" 국소(local) 탐색이다. 실제로 팔이
# 물리적으로 닿을 수 있는 위치인데도, 시작 자세가 "안 좋으면"(예: 로봇
# 몸체가 바라보는 방향과 정반대쪽에 있는 목표라서 관절을 크게 돌아가야
# 하는 경우) 중간에 국소최적점(local minimum)에 갇혀서 수렴에 실패할 수
# 있다 — 실제로 이 문제 때문에 "분명 닿는 거리인데 pick이 실패하는"
# 버그가 있었다. 그래서 처음 시도가 실패하면, 관절 한계 범위 안에서
# 무작위로 뽑은 시작 자세로 몇 번 더 재시도한다(진짜 산업용 IK 솔버들도
# 흔히 쓰는 방법). 그래도 다 실패하면 그중 제일 목표에 가까웠던 자세를
# 남겨두고 실패(False)를 반환한다.
# ============================================================================

import numpy as np


class IKSolver:
    def __init__(self, robot, joint_names, end_link, damping=0.02):
        self.robot = robot
        self.joint_names = joint_names
        self.end_link = end_link
        self.damping = damping
        self.rng = np.random.default_rng()

    def _end_pos(self):
        return self.robot.link_pose(self.end_link)[:3, 3]

    def _jacobian(self, eps=1e-4):
        base = self._end_pos()
        J = np.zeros((3, len(self.joint_names)))
        for i, name in enumerate(self.joint_names):
            original = self.robot.get_joint(name)
            self.robot.set_joint(name, original + eps)
            perturbed = self._end_pos()
            J[:, i] = (perturbed - base) / eps
            self.robot.set_joint(name, original)  # 원래 값으로 복구 (수치미분용 임시 이동이었을 뿐)
        return J

    def _joint_range(self, name):
        joint = self.robot.desc.joints[name]
        if joint.has_limit:
            return joint.lower, joint.upper
        return -np.pi, np.pi

    def _solve_from_current(self, target_xyz, max_iters, tol, step_scale):
        """지금 관절 각도에서 시작해서 max_iters만큼 국소 탐색. (수렴여부,
        최종 오차) 반환. robot.joint_positions는 마지막으로 찾은 자세로
        남는다(성공이든 실패든)."""
        for _ in range(max_iters):
            error_vec = target_xyz - self._end_pos()
            error = np.linalg.norm(error_vec)
            if error < tol:
                return True, error
            J = self._jacobian()
            lam2 = self.damping ** 2
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(3), error_vec)
            for name, delta in zip(self.joint_names, dq):
                self.robot.set_joint(name, self.robot.get_joint(name) + delta * step_scale)
        return False, np.linalg.norm(target_xyz - self._end_pos())

    def solve(self, target_xyz, max_iters=150, tol=1e-3, step_scale=0.6, n_restarts=6):
        target_xyz = np.asarray(target_xyz, dtype=float)
        start_backup = {name: self.robot.get_joint(name) for name in self.joint_names}

        best_error, best_config = np.inf, dict(start_backup)

        # 1차 시도: 지금 자세 그대로. 이전 동작에서 자연스럽게 이어지는
        # 경우가 대부분이라, 보통은 여기서 바로 풀린다(재시도 없이 빠름).
        ok, error = self._solve_from_current(target_xyz, max_iters, tol, step_scale)
        if error < best_error:
            best_error = error
            best_config = {name: self.robot.get_joint(name) for name in self.joint_names}
        if ok:
            return True

        # 실패하면 관절 한계 범위 안에서 무작위 시작 자세로 재시도.
        for _ in range(n_restarts):
            for name in self.joint_names:
                lo, hi = self._joint_range(name)
                self.robot.set_joint(name, self.rng.uniform(lo, hi))
            ok, error = self._solve_from_current(target_xyz, max_iters, tol, step_scale)
            if error < best_error:
                best_error = error
                best_config = {name: self.robot.get_joint(name) for name in self.joint_names}
            if ok:
                return True

        # 끝까지 못 풀었으면, 그나마 제일 가까웠던 자세로 남겨두고 실패 보고.
        for name, value in best_config.items():
            self.robot.set_joint(name, value)
        return best_error < tol * 5
