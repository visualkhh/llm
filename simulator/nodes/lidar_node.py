# ============================================================================
# nodes/lidar_node.py — 몸통에 달린 3D 라이다 노드
#
# 실제 회전형 3D 라이다(Velodyne류)처럼, 여러 '층(elevation, 위아래 각도)'
# x 여러 '수평각(azimuth, 360도 회전)'으로 광선을 쏴서 부딪힌 지점들을
# 모아 점군(point cloud)을 만든다. 매 틱마다:
#   1) robot.link_pose(lidar_link)로 "지금 라이다가 map 좌표계의 어디서,
#      어느 방향을 향하고 있는지" 구하고
#   2) 그 자세 기준으로 광선을 world_map의 모든 도형에 쏴서
#   3) 부딪힌 점들을(라이다 로컬 좌표계로) f"/lidar/{name}/points"에 발행한다.
#
# "라이다 로컬 좌표계로" 발행하는 이유: 실제 라이다 센서도 자기 몸체
# 기준 좌표로 점을 낸다. 그 점들을 map 좌표계로 옮기려면 "지금 로봇이
# map 기준 어디 있는지"를 알아야 하는데, 그게 바로 SLAM/로컬라이제이션이
# 풀어야 하는 문제다 — 즉, 이 노드는 "센서가 본 것"만 순수하게 책임지고,
# "그래서 내가 지도의 어디에 있는가"는 절대 여기서 답하지 않는다(이게
# ROS2에서 센서 노드와 로컬라이제이션 노드를 분리하는 이유이기도 하다).
# ============================================================================

import numpy as np

from nodes.core import Node
from world.raycast import raycast_scene_batch
from msgs import Header, PointCloud


class LidarNode(Node):
    def __init__(self, name, link_name, robot, world_map, bus, rate_hz=10.0,
                 n_rings=16, elevation_range_deg=(-15.0, 15.0),
                 n_azimuth=180, max_range=10.0):
        super().__init__(f"lidar_{name}", bus)
        self.link_name = link_name
        self.robot = robot
        self.world_map = world_map
        self.max_range = max_range
        self.latest_points = np.zeros((0, 3))

        elevations = np.linspace(np.radians(elevation_range_deg[0]),
                                  np.radians(elevation_range_deg[1]), n_rings)
        azimuths = np.linspace(-np.pi, np.pi, n_azimuth, endpoint=False)
        # 라이다 로컬 좌표계에서의 광선 방향들을 미리 다 계산해둔다 (매 틱
        # 재계산할 필요 없음 — 라이다 자체의 스캔 패턴은 안 바뀌니까).
        el_grid, az_grid = np.meshgrid(elevations, azimuths, indexing="ij")
        self._local_dirs = np.stack([
            np.cos(el_grid) * np.cos(az_grid),
            np.cos(el_grid) * np.sin(az_grid),
            np.sin(el_grid),
        ], axis=-1).reshape(-1, 3)

        self.publish_points = self.create_publisher(f"/lidar/{name}/points", msg_type=PointCloud)
        self.create_timer(rate_hz, self.scan)

    def scan(self):
        link_t = self.robot.link_pose(self.link_name)
        origin = link_t[:3, 3]
        rot = link_t[:3, :3]
        world_dirs = self._local_dirs @ rot.T  # 로컬 방향들을 월드 좌표계로 회전

        dists, hits = raycast_scene_batch(origin, world_dirs, self.world_map.all_shapes, self.max_range)
        hit_mask = np.array([h is not None for h in hits])
        self.latest_points = self._local_dirs[hit_mask] * dists[hit_mask, None]
        self.publish_points(PointCloud(header=Header(frame_id=self.link_name), points=self.latest_points))
