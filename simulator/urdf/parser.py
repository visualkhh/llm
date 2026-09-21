# ============================================================================
# urdf/parser.py — URDF(.urdf) 파일을 직접 파싱하는 최소 구현
#
# URDF는 그냥 XML이다. 로봇을 <link>(강체 부품)들과 <joint>(부품 사이의
# 관절, 어느 축으로 얼마나 움직이는지)들의 트리 구조로 표현한다.
#
#   <robot>
#     <link name="base_link">...</link>
#     <link name="wheel_fl">...</link>
#     <joint name="wheel_fl_joint" type="continuous">
#       <parent link="base_link"/>
#       <child link="wheel_fl"/>
#       <origin xyz="0.2 0.2 0"/>          <!-- base_link 기준 바퀴 위치 -->
#       <axis xyz="0 1 0"/>                <!-- 이 관절이 회전하는 축 -->
#     </joint>
#     ...
#   </robot>
#
# 이 파일은 urdfpy 같은 외부 URDF 라이브러리를 쓰지 않고, 표준 라이브러리
# xml.etree.ElementTree만으로 직접 읽어서 아래 dataclass들로 바꿔준다.
# 실제 렌더링(mesh 파일 읽기)까지는 하지 않고, 라이다/카메라 레이캐스팅과
# 화면 표시에 필요한 만큼(box/cylinder/sphere 형상, origin, 질량중심은
# 생략 — 이 시뮬레이터는 운동학만 다루므로 관성/질량은 불필요)만 다룬다.
# ============================================================================

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np


def _parse_vec(text, default):
    if text is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


@dataclass
class Origin:
    xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @staticmethod
    def from_xml(elem):
        if elem is None:
            return Origin()
        return Origin(
            xyz=_parse_vec(elem.get("xyz"), [0, 0, 0]),
            rpy=_parse_vec(elem.get("rpy"), [0, 0, 0]),
        )


@dataclass
class Geometry:
    """시각화/충돌(=여기서는 레이캐스팅)용 형상. box/cylinder/sphere만 지원."""
    kind: str  # "box" | "cylinder" | "sphere"
    size: np.ndarray | None = None      # box: [x, y, z]
    radius: float | None = None          # cylinder/sphere
    length: float | None = None          # cylinder

    @staticmethod
    def from_xml(elem):
        if elem is None:
            return None
        box = elem.find("box")
        if box is not None:
            return Geometry(kind="box", size=_parse_vec(box.get("size"), [0.1, 0.1, 0.1]))
        cyl = elem.find("cylinder")
        if cyl is not None:
            return Geometry(kind="cylinder", radius=float(cyl.get("radius", 0.05)),
                             length=float(cyl.get("length", 0.1)))
        sph = elem.find("sphere")
        if sph is not None:
            return Geometry(kind="sphere", radius=float(sph.get("radius", 0.05)))
        return None


@dataclass
class Visual:
    origin: Origin
    geometry: Geometry | None
    color: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.7, 0.7, 1.0]))


@dataclass
class Link:
    name: str
    visuals: list = field(default_factory=list)   # list[Visual]


@dataclass
class Joint:
    name: str
    type: str                 # fixed | revolute | continuous | prismatic
    parent: str
    child: str
    origin: Origin
    axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    lower: float = 0.0
    upper: float = 0.0
    has_limit: bool = False


@dataclass
class RobotDescription:
    name: str
    links: dict          # name -> Link
    joints: dict          # name -> Joint
    root_link: str
    children_of: dict     # parent link name -> list[Joint] (자식으로 이어지는 관절들)


def parse_urdf(path) -> RobotDescription:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"루트 태그가 <robot>이 아닙니다: {root.tag}")

    links = {}
    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        visuals = []
        for vis_elem in link_elem.findall("visual"):
            origin = Origin.from_xml(vis_elem.find("origin"))
            geometry = Geometry.from_xml(vis_elem.find("geometry"))
            color = np.array([0.7, 0.7, 0.7, 1.0])
            mat = vis_elem.find("material")
            if mat is not None:
                col_elem = mat.find("color")
                if col_elem is not None:
                    color = _parse_vec(col_elem.get("rgba"), [0.7, 0.7, 0.7, 1.0])
            visuals.append(Visual(origin=origin, geometry=geometry, color=color))
        links[name] = Link(name=name, visuals=visuals)

    joints = {}
    children_of = {name: [] for name in links}
    for joint_elem in root.findall("joint"):
        name = joint_elem.get("name")
        jtype = joint_elem.get("type")
        parent = joint_elem.find("parent").get("link")
        child = joint_elem.find("child").get("link")
        origin = Origin.from_xml(joint_elem.find("origin"))
        axis_elem = joint_elem.find("axis")
        axis = _parse_vec(axis_elem.get("xyz") if axis_elem is not None else None, [1, 0, 0])
        lower = upper = 0.0
        has_limit = False
        limit_elem = joint_elem.find("limit")
        if limit_elem is not None and limit_elem.get("lower") is not None:
            lower = float(limit_elem.get("lower"))
            upper = float(limit_elem.get("upper"))
            has_limit = True
        joint = Joint(name=name, type=jtype, parent=parent, child=child,
                      origin=origin, axis=axis, lower=lower, upper=upper, has_limit=has_limit)
        joints[name] = joint
        children_of.setdefault(parent, []).append(joint)

    # 루트 링크 찾기: 어떤 joint의 child로도 등장하지 않는 유일한 link.
    all_children = {j.child for j in joints.values()}
    roots = [name for name in links if name not in all_children]
    if len(roots) != 1:
        raise ValueError(f"루트 링크를 정확히 하나 찾아야 하는데 {len(roots)}개 발견됨: {roots}")

    return RobotDescription(
        name=root.get("name", "robot"),
        links=links, joints=joints, root_link=roots[0], children_of=children_of,
    )
