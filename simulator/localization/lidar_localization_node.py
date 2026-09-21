# ============================================================================
# localization/lidar_localization_node.py — 라이다 스캔으로 odom 드리프트를
#                                            실제로 보정하는 ComposableNode
#
# odom_node.py가 계산하는 위치는 바퀴 속도만 적분한 것이라(캘리브레이션
# 오차 + 노이즈 포함) 시간이 지날수록 실제 위치에서 벌어진다. 이 노드는
# 그 어긋남을 "라이다가 지금 본 벽들이, 내가 odom 기준으로 생각하는
# 위치에 놓고 보면 실제 지도(world_map)의 벽과 얼마나 잘 겹치는가"로
# 역으로 추정해서 보정한다 — 이게 스캔 매칭(scan matching)의 핵심 아이디어
# 이고, 실제 로봇의 AMCL/scan-to-map 로컬라이제이션이 하는 일과 같다.
#
# 알고리즘 (grid correlation scan matching — ICP보다 훨씬 단순하지만
# 원리는 같다: "후보 위치들을 여러 개 시도해보고, 라이다 점들이 실제
# 벽(점유 격자)에 가장 많이 들어맞는 후보를 고른다"):
#   1) 지금 odom이 믿는 위치(ex, ey, eyaw) 근처에서 작은 보정값 후보들
#      (dx, dy, dyaw)을 격자 모양으로 여러 개 만든다.
#   2) 각 후보마다: "라이다 점들 -> odom 프레임으로 옮기고 -> 그 후보
#      보정을 적용" 해서 map 좌표계 점들을 만든 뒤, 그 점들이 실제 맵의
#      점유 격자(장애물 칸)에 몇 개나 떨어지는지 센다.
#   3) 가장 많이 들어맞는 후보를 "이번 틱의 최선의 보정"으로 삼고, 지금
#      갖고 있던 보정값에서 그쪽으로 절반만 이동한다(급격히 튀지 않고
#      부드럽게 수렴하도록 — 실제 로컬라이제이션도 매 스캔마다 한 번에
#      확 바꾸지 않고 점진적으로 신뢰도를 쌓아간다).
#
# 결과적으로 이 보정값(self.correction)을 odom 위치에 더하면(정확히는
# SE(2) 합성) "map 좌표계 기준 최선의 추정 위치"가 나온다 — 이게 바로
# ROS2 tf 트리의 "map -> odom" 변환이 하는 역할이다. NavNode/viewer.py는
# 이제 odom_node의 날것 그대로가 아니라 이 노드의 corrected_pose()를
# 쓴다 (odom은 계속 드리프트하지만, 이 노드가 주기적으로 그 드리프트를
# 다시 깎아내는 구조).
# ============================================================================

import numpy as np

from nodes.core import ComposableNode
from nav.planner import OccupancyGrid
from math3d import xyzrpy_to_matrix, matrix_to_xyz_yaw
from msgs import PointCloud


class LidarLocalizationNode(ComposableNode):
    def __init__(self, bus, world_map, odom_node, rate_hz=2.0,
                 search_range=(0.15, 0.15, 0.12), search_steps=9,
                 horizontal_band=0.15, smoothing=0.5):
        super().__init__("slam_lidar", bus)
        self.world_map = world_map
        self.odom_node = odom_node
        # ⚠️ NavNode의 격자를 재사용하면 안 된다 — 그건 경로계획용으로 벽
        # 주변에 안전마진(기본 0.35m)을 덧씌워 "부풀린" 격자라서, 스캔
        # 매칭에 쓰면 실제 벽 위치와 상관없이 아무 후보나 "적중"으로 잘못
        # 판정돼 버린다(실제로 이 버그 때문에 보정이 위치를 더 틀어지게
        # 만드는 걸 확인했다). 스캔 매칭은 벽의 "진짜" 크기 그대로의
        # 격자가 필요하므로 안전마진을 거의 0으로 따로 만든다.
        self.grid = OccupancyGrid(world_map, inflate_radius=0.02)
        self.search_range = search_range
        self.search_steps = search_steps
        self.horizontal_band = horizontal_band  # 이 높이(z) 범위 안의 라이다 점만 2D 매칭에 사용
        self.smoothing = smoothing  # 0~1: 클수록 새 후보 쪽으로 더 빨리 이동

        self.correction = np.array([0.0, 0.0, 0.0])  # (dx, dy, dyaw): map = correction ∘ odom
        self.latest_scan_local = np.zeros((0, 3))
        self.last_fit_ratio = 0.0  # 이번에 찾은 최선의 보정이 라이다 점을 몇 %나 벽에 맞췄는지 (디버그/표시용)

        # "지금 이 순간의 스캔"과는 별개로, 지금까지 로봇이 돌아다니며 본
        # 라이다 점들을 계속 누적해서 "로봇이 스스로 파악한 지도"를
        # 만든다 — 이게 실제 SLAM이 최종적으로 만들어내는 산출물(지도)에
        # 해당한다. 매번 스캔이 올 때마다 world 좌표로 옮겨서 작은
        # 격자칸(map_cell_size) 단위로 중복 없이 모아둔다(같은 벽을 백날
        # 다시 봐도 점이 무한히 늘어나지 않게).
        self.map_cell_size = 0.08
        self.mapped_cells = set()

        self.create_subscription("/lidar/main/points", self._on_scan, msg_type=PointCloud)
        self.create_timer(rate_hz, self._tick)

    def _on_scan(self, msg: PointCloud):
        self.latest_scan_local = msg.points
        self._accumulate_map(msg.points)

    def _accumulate_map(self, points_local):
        if len(points_local) == 0:
            return
        x, y, yaw = self.corrected_pose()
        c, s = np.cos(yaw), np.sin(yaw)
        wx = x + points_local[:, 0] * c - points_local[:, 1] * s
        wy = y + points_local[:, 0] * s + points_local[:, 1] * c
        cs = self.map_cell_size
        for px, py in zip(wx, wy):
            self.mapped_cells.add((round(px / cs) * cs, round(py / cs) * cs))

    def mapped_points_array(self):
        if not self.mapped_cells:
            return np.zeros((0, 2))
        return np.array(list(self.mapped_cells))

    def corrected_pose(self):
        """odom의 날것 추정치에 지금까지 쌓은 보정을 합성한, "map 좌표계
        기준 최선의 추정 위치" — NavNode와 viewer.py가 여기서 위치를 읽는다."""
        ox, oy, oyaw = self.odom_node.estimated_pose
        odom_t = xyzrpy_to_matrix([ox, oy, 0.0], [0.0, 0.0, oyaw])
        cx, cy, cyaw = self.correction
        corr_t = xyzrpy_to_matrix([cx, cy, 0.0], [0.0, 0.0, cyaw])
        x, y, yaw = matrix_to_xyz_yaw(corr_t @ odom_t)
        return float(x), float(y), float(yaw)

    def _tick(self):
        # 로봇이 멈춰있을 때도 계속 후보를 재탐색하면, 격자 해상도/탐색
        # 스텝 크기 때문에 생기는 작은 양자화 잡음만으로 보정값이 미세하게
        # 계속 흔들린다(=실제로는 안 움직였는데 추정 위치가 제자리에서
        # 살짝씩 걸어다니는 것처럼 보임). 실제 로봇도 멈춰있으면 위치
        # 추정이 안정되는 게 정상이므로, 바퀴가 실제로 돌고 있을 때만
        # 다시 탐색한다.
        if not self.odom_node.is_moving:
            return
        pts = self.latest_scan_local
        if len(pts) < 8:
            return
        # 라이다는 3D(위아래 여러 층)지만, 2D 평면 지도와 맞춰볼 것이므로
        # 로봇 몸체 높이와 비슷한(z가 0에 가까운) 수평 링만 골라 쓴다.
        horizontal = pts[np.abs(pts[:, 2]) < self.horizontal_band][:, :2]
        if len(horizontal) < 8:
            return

        ox, oy, oyaw = self.odom_node.estimated_pose
        cos_o, sin_o = np.cos(oyaw), np.sin(oyaw)
        # 라이다 로컬 좌표 -> "odom이 믿는 위치" 기준 world 좌표로 한 번만 미리 변환
        odom_frame_x = ox + horizontal[:, 0] * cos_o - horizontal[:, 1] * sin_o
        odom_frame_y = oy + horizontal[:, 0] * sin_o + horizontal[:, 1] * cos_o

        dx_range, dy_range, dyaw_range = self.search_range
        candidates_dx = np.linspace(-dx_range, dx_range, self.search_steps)
        candidates_dy = np.linspace(-dy_range, dy_range, self.search_steps)
        candidates_dyaw = np.linspace(-dyaw_range, dyaw_range, self.search_steps)

        base_cx, base_cy, base_cyaw = self.correction
        best_score, best_corr = -1, self.correction

        for dyaw in candidates_dyaw:
            cyaw = base_cyaw + dyaw
            cc, cs = np.cos(cyaw), np.sin(cyaw)
            # yaw 후보 하나당 회전은 한 번만 계산하고, dx/dy는 벡터화해서 빠르게 훑는다.
            rx = odom_frame_x * cc - odom_frame_y * cs
            ry = odom_frame_x * cs + odom_frame_y * cc
            for dx in candidates_dx:
                for dy in candidates_dy:
                    mx = base_cx + dx + rx
                    my = base_cy + dy + ry
                    score = self._occupancy_score(mx, my)
                    if score > best_score:
                        best_score = score
                        best_corr = np.array([base_cx + dx, base_cy + dy, cyaw])

        self.last_fit_ratio = best_score / len(horizontal)
        # 한 번에 점프하지 않고 절반(smoothing)만 이동 — 노이즈 있는 한 번의
        # 스캔 매칭 결과에 과하게 흔들리지 않도록.
        self.correction = (1 - self.smoothing) * self.correction + self.smoothing * best_corr

    def _occupancy_score(self, xs, ys):
        cols = ((xs - self.grid.origin_x) / self.grid.resolution).astype(int)
        rows = ((ys - self.grid.origin_y) / self.grid.resolution).astype(int)
        valid = (cols >= 0) & (cols < self.grid.cols) & (rows >= 0) & (rows < self.grid.rows)
        hits = self.grid.occupied[rows[valid], cols[valid]]
        return int(hits.sum())
