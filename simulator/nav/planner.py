# ============================================================================
# nav/planner.py — Nav2의 "전역 경로계획(global planner)"에 해당하는 A*
#
# 맵의 3D 도형들을 2D 평면으로 내려다본 "장애물 격자(occupancy grid)"로
# 바꾼 뒤(2D 내비게이션이므로 높이는 무시하고 "여기 뭔가 있다/없다"만
# 필요), 그 격자 위에서 A* 탐색으로 시작점->목표점 최단 경로를 찾는다.
# 실제 Nav2도 내부적으로 costmap(장애물 격자) 위에서 이런 그래프 탐색을
# 한다 — 여기서는 그 핵심 아이디어를 그대로, 가장 단순한 형태로 구현한다.
# ============================================================================

import heapq

import numpy as np


class OccupancyGrid:
    # 로봇 몸체 반너비(대략 0.19m, base_link 0.5x0.35m 기준)에 여유를
    # 조금만 더한 값. 이보다 훨씬 크게(예전엔 0.35였음) 잡으면 장애물
    # 옆의 물체(예: 테이블 위 물체)에 팔이 닿을 수 있는 거리보다 더
    # 멀리서만 접근하게 돼서, "가까이 가서 집는" 것 자체가 불가능해지는
    # 문제가 실제로 있었다 (테이블 위 cube_1을 어떤 각도로 접근해도 팔이
    # 못 닿는 상황이 발생했었음).
    def __init__(self, world_map, resolution=0.1, inflate_radius=0.15):
        self.resolution = resolution
        self.origin_x = -world_map.width / 2
        self.origin_y = -world_map.depth / 2
        self.cols = int(world_map.width / resolution)
        self.rows = int(world_map.depth / resolution)
        self.occupied = np.zeros((self.rows, self.cols), dtype=bool)

        # 장애물 종류별로 실제 2D 형상에 맞게 격자를 채운다. 박스를 "외접원"
        # 하나로 뭉뚱그리면 벽처럼 길쭉한 도형은 반지름이 수 미터로
        # 부풀어서 지도 전체가 장애물이 되어버리는 심각한 오차가 생기므로
        # (실제로 이 버그가 있었다), 박스는 회전을 반영한 사각형으로,
        # 원기둥/구는 실제 반지름의 원으로 각각 정확히 래스터화한다.
        for shape in world_map.shapes:
            if shape.kind == "box":
                self._mark_box(shape.xyz[0], shape.xyz[1], shape.rpy[2],
                                shape.size[0], shape.size[1], inflate_radius)
            else:
                self._mark_circle(shape.xyz[0], shape.xyz[1], shape.radius + inflate_radius)

    def _mark_box(self, cx, cy, yaw, length_x, length_y, inflate):
        # 회전된 사각형을 정확히 래스터화하는 대신, "회전을 반영한 축
        # 정렬 바운딩 박스"로 살짝 보수적으로(더 넓게) 잡는다 — 경로계획
        # 단계에서는 이 정도 여유가 오히려 안전 마진이 되어 낫다.
        hx, hy = length_x / 2.0, length_y / 2.0
        c, s = np.cos(yaw), np.sin(yaw)
        corners_x = [cx + c * hx - s * hy, cx + c * hx + s * hy,
                     cx - c * hx - s * hy, cx - c * hx + s * hy]
        corners_y = [cy + s * hx + c * hy, cy + s * hx - c * hy,
                     cy - s * hx + c * hy, cy - s * hx - c * hy]
        min_x, max_x = min(corners_x) - inflate, max(corners_x) + inflate
        min_y, max_y = min(corners_y) - inflate, max(corners_y) + inflate
        c0, r0 = self.world_to_cell(min_x, min_y)
        c1, r1 = self.world_to_cell(max_x, max_y)
        r0, r1 = max(0, r0), min(self.rows - 1, r1)
        c0, c1 = max(0, c0), min(self.cols - 1, c1)
        if r0 <= r1 and c0 <= c1:
            self.occupied[r0:r1 + 1, c0:c1 + 1] = True

    def _mark_circle(self, cx, cy, radius):
        r_cells = int(np.ceil(radius / self.resolution))
        ccol, crow = self.world_to_cell(cx, cy)
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                if dr * dr + dc * dc > r_cells * r_cells:
                    continue
                r, c = crow + dr, ccol + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    self.occupied[r, c] = True

    def world_to_cell(self, x, y):
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return col, row

    def cell_to_world(self, col, row):
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    def is_free(self, col, row):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        return not self.occupied[row, col]


_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]


def a_star(grid: OccupancyGrid, start_xy, goal_xy):
    """격자 위에서 A* 탐색. 못 찾으면 None, 찾으면 [(x,y), ...] 웨이포인트 리스트."""
    start = grid.world_to_cell(*start_xy)
    goal = grid.world_to_cell(*goal_xy)
    if not grid.is_free(*goal):
        return None

    def h(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    open_set = [(h(start, goal), start)]
    came_from = {}
    g_score = {start: 0.0}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return _reconstruct_path(grid, came_from, current)

        for dc, dr in _NEIGHBORS:
            neighbor = (current[0] + dc, current[1] + dr)
            if not grid.is_free(*neighbor):
                continue
            step_cost = np.hypot(dc, dr)
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, np.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + h(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))
    return None  # 경로 없음


def _reconstruct_path(grid, came_from, current):
    cells = [current]
    while current in came_from:
        current = came_from[current]
        cells.append(current)
    cells.reverse()
    return [grid.cell_to_world(c, r) for c, r in cells]
