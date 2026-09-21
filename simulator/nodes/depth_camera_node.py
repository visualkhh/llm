# ============================================================================
# nodes/depth_camera_node.py — 그리퍼에 달린 RGB+Depth(뎁스) 카메라 노드
#
# 실제 RGB-D 카메라(RealSense류)를 흉내낸다: 카메라가 보는 시야각(FOV)
# 안에서 격자 모양으로 광선을 아주 많이 쏴서(픽셀 하나당 광선 하나),
# 각 픽셀이 무엇에 부딪혔는지(RGB용)와 얼마나 멀리서 부딪혔는지
# (Depth용)를 계산한다 — 라이다와 원리는 완전히 같고, "카메라 프레임
# (직사각형 격자) 모양으로 광선을 쏜다"는 점만 다르다.
#
# 이 카메라는 그리퍼(gripper_base_link)에 고정되어 있으므로, 팔 관절이
# 움직이면 robot.link_pose(camera_link)가 매번 달라지고, 그 결과 카메라가
# "보고 있는 장면"도 팔이 움직이는 대로 함께 바뀐다 — 실제로 손목에
# 카메라를 단 로봇과 똑같은 특성이다.
# ============================================================================

import numpy as np

from nodes.core import Node
from world.raycast import raycast_scene_batch
from msgs import Header, Image


class DepthCameraNode(Node):
    def __init__(self, name, link_name, robot, world_map, bus, rate_hz=10.0,
                 width=64, height=48, h_fov_deg=87.0, max_range=8.0):
        super().__init__(f"camera_{name}", bus)
        self.link_name = link_name
        self.robot = robot
        self.world_map = world_map
        self.max_range = max_range
        self.width, self.height = width, height

        h_fov = np.radians(h_fov_deg)
        v_fov = h_fov * height / width
        xs = np.tan(np.linspace(-h_fov / 2, h_fov / 2, width))
        ys = np.tan(np.linspace(v_fov / 2, -v_fov / 2, height))
        # 카메라 로컬 좌표계 관례: x=전방, y=왼쪽, z=위쪽 (URDF/ROS 공통 관례).
        # 픽셀별 광선 방향을 전부 미리 계산해둔다.
        gx, gy = np.meshgrid(xs, ys)
        dirs = np.stack([np.ones_like(gx), gx, gy], axis=-1)
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        self._local_dirs = dirs.reshape(-1, 3)

        self.rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self.depth = np.full((height, width), max_range, dtype=np.float32)

        self.publish_frame = self.create_publisher(f"/camera/{name}/frame", msg_type=Image)
        self.create_timer(rate_hz, self.render)

    def render(self):
        link_t = self.robot.link_pose(self.link_name)
        origin = link_t[:3, 3]
        rot = link_t[:3, :3]
        world_dirs = self._local_dirs @ rot.T

        depth_flat, hits = raycast_scene_batch(origin, world_dirs, self.world_map.all_shapes, self.max_range)

        rgb_flat = np.full((len(world_dirs), 3), 30, dtype=np.uint8)  # 기본값: 배경(하늘/먼 곳)
        rgb_flat[:, 2] = 40
        # 아주 단순한 셰이딩: 가까울수록 밝게 (실제 조명 계산은 생략 —
        # 이 시뮬레이터의 목적은 "위치가 맞는 관측"이지 그래픽스가 아니다).
        shade = np.clip(1.0 - depth_flat / self.max_range, 0.15, 1.0)
        for i, hit in enumerate(hits):
            if hit is not None:
                rgb_flat[i] = (hit.color[:3] * 255 * shade[i]).astype(np.uint8)

        self.rgb = rgb_flat.reshape(self.height, self.width, 3)
        self.depth = depth_flat.reshape(self.height, self.width).astype(np.float32)
        self.publish_frame(Image(header=Header(frame_id=self.link_name), rgb=self.rgb, depth=self.depth))
