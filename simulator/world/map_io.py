# ============================================================================
# world/map_io.py — 사용자가 직접 작성하는 "맵" JSON 파일을 읽어들이기
#
# 이 시뮬레이터의 맵은 3D 도형(박스/원기둥/구)들을 좌표와 함께 나열한
# JSON 파일이다. 라이다/뎁스카메라가 3D이므로 2D 점유격자(occupancy
# grid)로는 높이 정보를 표현할 수 없어서, 이 방식을 선택했다
# (문서 documents/00-overview.md 참고).
#
# 파일 형식 예시는 maps/example_room.json 참고. 스키마:
#   {
#     "name": "...",
#     "bounds": {"width": 10.0, "depth": 10.0},   # 2D 내비게이션 평면 크기
#     "shapes":  [ {도형...}, ... ],                # 고정 장애물 (벽, 기둥 등)
#     "objects": [ {도형..., "graspable": true}, ...] # 로봇이 집을 수 있는 물체
#   }
#   도형 하나: {"id": "...", "type": "box"|"cylinder"|"sphere",
#               "pose": {"xyz": [x,y,z], "rpy": [r,p,y]},
#               "size": [x,y,z] (box)  또는  "radius"/"height" (cylinder/sphere)}
# ============================================================================

import json

import numpy as np


class Shape:
    def __init__(self, id, kind, xyz, rpy, size=None, radius=None, height=None,
                 graspable=False, color=(0.6, 0.6, 0.65)):
        self.id = id
        self.kind = kind  # "box" | "cylinder" | "sphere"
        self.xyz = np.array(xyz, dtype=float)
        self.rpy = np.array(rpy, dtype=float)
        self.size = np.array(size, dtype=float) if size is not None else None
        self.radius = radius
        self.height = height
        self.graspable = graspable
        self.color = np.array(color, dtype=float)  # 뎁스카메라 렌더링용 RGB(0~1)

    @property
    def bounding_radius_2d(self):
        """내비게이션 평면(2D)에서 이 도형이 차지하는 대략적인 반지름
        (장애물 지도를 격자화할 때 사용)."""
        if self.kind == "box":
            return float(np.hypot(self.size[0], self.size[1]) / 2.0)
        return float(self.radius or 0.3)


class WorldMap:
    def __init__(self, name, width, depth, shapes, objects):
        self.name = name
        self.width = width
        self.depth = depth
        self.shapes = shapes      # 고정 장애물 (list[Shape])
        self.objects = objects    # 집을 수 있는 물체 (list[Shape])

    @property
    def all_shapes(self):
        return self.shapes + self.objects

    def find_object(self, object_id):
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None


def _shape_from_dict(d):
    pose = d.get("pose", {})
    return Shape(
        id=d["id"], kind=d["type"],
        xyz=pose.get("xyz", [0, 0, 0]), rpy=pose.get("rpy", [0, 0, 0]),
        size=d.get("size"), radius=d.get("radius"), height=d.get("height"),
        graspable=d.get("graspable", False), color=d.get("color", (0.6, 0.6, 0.65)),
    )


def load_map(path) -> WorldMap:
    with open(path) as f:
        data = json.load(f)
    bounds = data.get("bounds", {"width": 10.0, "depth": 10.0})
    shapes = [_shape_from_dict(s) for s in data.get("shapes", [])]
    objects = [_shape_from_dict(o) for o in data.get("objects", [])]
    return WorldMap(name=data.get("name", "map"), width=bounds["width"], depth=bounds["depth"],
                     shapes=shapes, objects=objects)
