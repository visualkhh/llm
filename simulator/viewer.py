# ============================================================================
# viewer.py — 시뮬레이터를 실제로 보고 조작하는 GUI
#
# 화면 구성:
#   왼쪽(큰 패널) — 맵 탑뷰. 회색 도형=장애물, 초록 별=집을 수 있는 물체,
#     파란 로봇 아이콘=진짜 위치(ground truth), 빨간 점선 로봇 아이콘=
#     로봇이 스스로 믿고 있는 위치(odom 추정치). 라이다 점들은 "로봇이
#     스스로 믿는 위치" 기준으로 그린다 — 그래서 오도메트리가 아직
#     보정(SLAM)되지 않은 상태에서는 라이다 점들이 실제 벽 모양과 어긋나
#     보인다(사용자가 요청한 "캘리브레이션 안 된 것처럼 비틀어져 보이는"
#     효과가 바로 이것).
#   오른쪽 위 — 그리퍼에 달린 카메라의 RGB 화면 (팔이 움직이면 그대로
#     따라 바뀐다).
#   오른쪽 가운데 — 같은 카메라의 Depth(뎁스) 화면.
#   아래 — 상태 텍스트 + 명령어 입력창.
#
# 조작:
#   W/S = 전진/후진, A/D = 좌/우 회전, Space = 정지 (키보드 teleop)
#   맵 패널을 클릭 = 그 좌표로 내비게이션 목표 설정 ("여기로 가라")
#   아래 입력창에 "pick <object_id>" 또는 "move x y z" 입력 후 Enter
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.widgets import TextBox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D 프로젝션 등록용)

from math3d import xyzrpy_to_matrix, transform_point

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

TELEOP_V = 0.4
TELEOP_OMEGA = 0.8


def _shape_patch(shape, facecolor, alpha=0.85):
    if shape.kind == "box":
        yaw_deg = np.degrees(shape.rpy[2])
        w, h = shape.size[0], shape.size[1]
        # matplotlib Rectangle은 "왼쪽 아래 꼭짓점 + 회전"이라 중심 기준으로 보정
        cx, cy = shape.xyz[0], shape.xyz[1]
        c, s = np.cos(shape.rpy[2]), np.sin(shape.rpy[2])
        corner = (cx - (w / 2) * c + (h / 2) * s, cy - (w / 2) * s - (h / 2) * c)
        return Rectangle(corner, w, h, angle=yaw_deg, facecolor=facecolor, alpha=alpha, edgecolor="#333333")
    return Circle((shape.xyz[0], shape.xyz[1]), shape.radius, facecolor=facecolor, alpha=alpha, edgecolor="#333333")


def _robot_triangle(x, y, yaw, size=0.22):
    """방향을 알 수 있게 삼각형(화살촉 모양)으로 로봇을 표시."""
    pts = np.array([[size, 0], [-size * 0.6, size * 0.55], [-size * 0.6, -size * 0.55]])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    world = (rot @ pts.T).T + np.array([x, y])
    return world


# -- 로봇 3D 미리보기 (지금 관절 값 그대로, URDF 형상을 실제로 그리기) --------
def _box_faces(transform, size):
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    corners_local = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])
    c = transform_point(transform, corners_local)
    idx = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    return [c[i] for i in idx]


def _cylinder_faces(transform, radius, length, n=10):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    bottom_local = np.column_stack([circle, np.full(n, -length / 2)])
    top_local = np.column_stack([circle, np.full(n, length / 2)])
    bottom = transform_point(transform, bottom_local)
    top = transform_point(transform, top_local)
    faces = [[bottom[i], bottom[(i + 1) % n], top[(i + 1) % n], top[i]] for i in range(n)]
    faces += [list(bottom), list(top)]
    return faces


def draw_robot_3d(ax, sim):
    """URDF에 적힌 실제 형상(box/cylinder)을, 지금 이 순간의 관절 값으로
    계산한 위치에 그대로 그린다 — "지금 로봇이 어떤 자세인지"를 추상적인
    삼각형이 아니라 실제 모양으로 보여주는 패널."""
    ax.cla()
    ax.set_title("로봇 현재 형태 (실시간, base_link 기준)", fontsize=9)
    fk = sim.robot.forward_kinematics()
    all_faces, all_colors = [], []
    for link_name, link in sim.description.links.items():
        link_t = fk.get(link_name)
        if link_t is None:
            continue
        for visual in link.visuals:
            if visual.geometry is None:
                continue
            vt = link_t @ xyzrpy_to_matrix(visual.origin.xyz, visual.origin.rpy)
            geom = visual.geometry
            if geom.kind == "box":
                faces = _box_faces(vt, geom.size)
            elif geom.kind == "cylinder":
                faces = _cylinder_faces(vt, geom.radius, geom.length)
            else:
                continue
            all_faces.extend(faces)
            all_colors.extend([tuple(visual.color[:3])] * len(faces))

    ax.add_collection3d(Poly3DCollection(all_faces, facecolor=all_colors, edgecolor="#222222", linewidths=0.3))

    base = fk[sim.description.root_link][:3, 3]
    ax.set_xlim(base[0] - 0.35, base[0] + 0.55)
    ax.set_ylim(base[1] - 0.45, base[1] + 0.45)
    ax.set_zlim(base[2] - 0.25, base[2] + 0.45)
    ax.set_box_aspect((0.9, 0.9, 0.7))
    ax.view_init(elev=22, azim=-60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def run_viewer(sim):
    print("=" * 64)
    print("조작법")
    print("  W/S = 전진/후진, A/D = 좌/우 회전, Space = 정지 (키보드 teleop)")
    print("  맵 패널(왼쪽) 클릭 = 그 자리로 내비게이션 이동")
    print("  화면 맨 아래 '명령' 입력창을 클릭해서 커서를 넣은 뒤 타이핑, Enter로 실행:")
    print("    pick <object_id>     예) pick cube_1")
    print("    move <x> <y> <z>     예) move 0.4 0.0 0.3   (그리퍼를 이 좌표로)")
    print("    place <x> <y> <z>    예) place 0.3 0.1 0.2  (쥔 물체를 이 좌표에 놓기)")
    print("    goto <x> <y>         예) goto 2.0 1.5       (맵 클릭과 동일)")
    print(f"  지금 맵에 있는 집을 수 있는 물체: {[o.id for o in sim.world_map.objects]}")
    print("=" * 64)

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Pure-Python 로보틱스 시뮬레이터 — 조작법은 터미널 출력 참고", fontsize=10)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.35], width_ratios=[1.5, 1, 1],
                           left=0.04, right=0.98, top=0.92, bottom=0.14, hspace=0.4, wspace=0.25)

    ax_map = fig.add_subplot(gs[:2, 0])
    ax_rgb = fig.add_subplot(gs[0, 1])
    ax_depth = fig.add_subplot(gs[0, 2])
    ax_robot3d = fig.add_subplot(gs[1, 1:], projection="3d")
    ax_info = fig.add_subplot(gs[2, :])

    half_w, half_d = sim.world_map.width / 2, sim.world_map.depth / 2
    ax_map.set_xlim(-half_w, half_w)
    ax_map.set_ylim(-half_d, half_d)
    ax_map.set_aspect("equal")
    ax_map.set_title("맵 탑뷰 (파랑=실제 로봇, 빨강 점선=odom 날것, 초록 점선=보정 후, "
                     "초록 점=이번 스캔, 옅은 파랑 점=누적된 지도)", fontsize=8.5)

    for shape in sim.world_map.shapes:
        ax_map.add_patch(_shape_patch(shape, "#888888"))
    object_patches = {}
    object_labels = {}
    for obj in sim.world_map.objects:
        patch = _shape_patch(obj, tuple(obj.color))  # 물체마다 맵 JSON에 적힌 고유 색으로 표시
        ax_map.add_patch(patch)
        object_patches[obj.id] = patch
        object_labels[obj.id] = ax_map.text(
            obj.xyz[0], obj.xyz[1] + 0.15, obj.id, fontsize=7, ha="center", va="bottom",
            color="#333333", zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7),
        )

    true_robot_poly = Polygon(_robot_triangle(0, 0, 0), closed=True, facecolor="#4c72b0", edgecolor="#1f3f66", zorder=5)
    ax_map.add_patch(true_robot_poly)
    # odom 날것(보정 전) — 계속 드리프트하는 것을 그대로 보여주는 기준선
    est_robot_poly = Polygon(_robot_triangle(0, 0, 0), closed=True, facecolor="none",
                              edgecolor="#c44e52", linewidth=1.4, linestyle="--", zorder=5, alpha=0.7)
    ax_map.add_patch(est_robot_poly)
    # 라이다 스캔 매칭으로 보정된 추정치 — NavNode/그리퍼가 실제로 쓰는 값,
    # 진짜 위치(파랑)에 odom(빨강)보다 더 가깝게 붙어있어야 정상.
    corrected_robot_poly = Polygon(_robot_triangle(0, 0, 0), closed=True, facecolor="none",
                                    edgecolor="#55a868", linewidth=1.8, linestyle="-.", zorder=6)
    ax_map.add_patch(corrected_robot_poly)

    # 지금 이 순간의 라이다 스캔(진한 초록, 매 프레임 갱신)과, 지금까지
    # 로봇이 돌아다니며 누적해서 스스로 파악한 지도(옅은 파랑, 계속
    # 쌓이기만 함)를 서로 다른 레이어로 분리해서 보여준다 — 후자가 바로
    # "라이다로 파악한 지도"에 해당한다 (SLAM이 최종적으로 만들어내는
    # 산출물). 실제 벽(회색 도형)과 이 파란 점들이 시간이 지나며 점점
    # 겹쳐지는 걸 보면 로컬라이제이션 보정이 잘 되고 있다는 뜻이다.
    mapped_scatter = ax_map.scatter([], [], s=4, color="#4c72b0", alpha=0.35, zorder=2)
    lidar_scatter = ax_map.scatter([], [], s=3, color="#55a868", alpha=0.6, zorder=4)
    path_line, = ax_map.plot([], [], "--", color="#8172b2", lw=1.3, zorder=3)
    goal_marker, = ax_map.plot([], [], "*", color="#dd8452", markersize=16, zorder=6)

    ax_rgb.set_title("그리퍼 카메라 — RGB", fontsize=9)
    ax_rgb.axis("off")
    rgb_im = ax_rgb.imshow(np.zeros((sim.camera.height, sim.camera.width, 3), dtype=np.uint8))

    ax_depth.set_title("그리퍼 카메라 — Depth", fontsize=9)
    ax_depth.axis("off")
    depth_im = ax_depth.imshow(sim.camera.depth, cmap="viridis", vmin=0, vmax=sim.camera.max_range)

    ax_info.axis("off")
    # family="monospace"를 쓰면 그 폰트(DejaVu Sans Mono)에 한글 글리프가
    # 없어서 한글만 네모(□)로 깨진다 — 숫자 정렬을 살짝 포기하는 대신
    # 기본 폰트(AppleGothic, 한글 지원)를 그대로 쓴다.
    info_text = ax_info.text(0.0, 1.0, "", transform=ax_info.transAxes, va="top", fontsize=9)

    draw_robot_3d(ax_robot3d, sim)

    # -- 입력창 (텍스트 명령어) ------------------------------------------------
    ax_box = fig.add_axes([0.30, 0.02, 0.55, 0.05])
    text_box = TextBox(ax_box, "명령(클릭 후 입력): ", initial="")

    def submit_command(text):
        text = text.strip()
        parts = text.split()
        if not parts:
            return
        if parts[0] == "pick" and len(parts) >= 2:
            sim.send_moveit_command({"type": "pick", "object_id": parts[1]})
        elif parts[0] == "move" and len(parts) >= 4:
            xyz = [float(v) for v in parts[1:4]]
            sim.send_moveit_command({"type": "move_gripper", "target": xyz})
        elif parts[0] == "place" and len(parts) >= 4:
            xyz = [float(v) for v in parts[1:4]]
            sim.send_moveit_command({"type": "place", "target": xyz})
        elif parts[0] == "goto" and len(parts) >= 3:
            sim.send_nav_goal(float(parts[1]), float(parts[2]))
        text_box.set_val("")

    text_box.on_submit(submit_command)

    # -- 키보드 teleop --------------------------------------------------------
    teleop_state = {"v": 0.0, "omega": 0.0}

    def on_key(event):
        k = event.key
        if k == "w":
            teleop_state["v"] = TELEOP_V
        elif k == "s":
            teleop_state["v"] = -TELEOP_V
        elif k == "a":
            teleop_state["omega"] = TELEOP_OMEGA
        elif k == "d":
            teleop_state["omega"] = -TELEOP_OMEGA
        elif k == " ":
            teleop_state["v"], teleop_state["omega"] = 0.0, 0.0
        else:
            return
        sim.set_body_velocity(teleop_state["v"], teleop_state["omega"])

    fig.canvas.mpl_connect("key_press_event", on_key)

    # -- 맵 클릭 = 내비게이션 목표 --------------------------------------------
    def on_click(event):
        if event.inaxes is not ax_map or event.xdata is None:
            return
        sim.send_nav_goal(event.xdata, event.ydata)

    fig.canvas.mpl_connect("button_press_event", on_click)

    # -- 매 프레임 --------------------------------------------------------------
    def update(_frame):
        sim.step()

        tx, ty, tyaw = sim.drive_true.x, sim.drive_true.y, sim.drive_true.yaw
        ex, ey, eyaw = sim.odom.estimated_pose
        rx, ry, ryaw = sim.slam_lidar.corrected_pose()

        true_robot_poly.set_xy(_robot_triangle(tx, ty, tyaw))
        est_robot_poly.set_xy(_robot_triangle(ex, ey, eyaw))
        corrected_robot_poly.set_xy(_robot_triangle(rx, ry, ryaw))

        # 라이다 점: 로봇 로컬 좌표계 값을 "라이다 보정이 적용된 최선의
        # 추정 위치" 기준으로 world 좌표로 옮겨서 그린다 — NavNode가 실제로
        # 내비게이션에 쓰는 것과 같은 프레임이라, 여기 찍힌 점들이 실제
        # 벽(회색 도형)에 얼마나 잘 들어맞는지가 곧 "보정이 잘 되고
        # 있는지"를 눈으로 보여준다 (처음엔 어긋나 보이다가 로봇이 움직여
        # 라이다 스캔이 몇 번 쌓이면 점점 벽에 들어맞기 시작한다).
        pts = sim.lidar.latest_points
        if len(pts):
            c, s = np.cos(ryaw), np.sin(ryaw)
            wx = rx + pts[:, 0] * c - pts[:, 1] * s
            wy = ry + pts[:, 0] * s + pts[:, 1] * c
            lidar_scatter.set_offsets(np.stack([wx, wy], axis=1))
        else:
            lidar_scatter.set_offsets(np.zeros((0, 2)))

        mapped_scatter.set_offsets(sim.slam_lidar.mapped_points_array())

        if sim.nav.path:
            path_arr = np.array(sim.nav.path)
            path_line.set_data(path_arr[:, 0], path_arr[:, 1])
        else:
            path_line.set_data([], [])

        rgb_im.set_data(sim.camera.rgb)
        depth_im.set_data(sim.camera.depth)
        draw_robot_3d(ax_robot3d, sim)

        for obj_id, patch in object_patches.items():
            obj = sim.world_map.find_object(obj_id)
            if isinstance(patch, Circle):
                patch.center = (obj.xyz[0], obj.xyz[1])
            else:
                patch.set_xy((obj.xyz[0] - obj.size[0] / 2, obj.xyz[1] - obj.size[1] / 2))
            object_labels[obj_id].set_position((obj.xyz[0], obj.xyz[1] + 0.15))

        odom_error = np.hypot(tx - ex, ty - ey)
        corrected_error = np.hypot(tx - rx, ty - ry)
        info_text.set_text(
            f"실제 위치(map)         : x={tx:+.2f} y={ty:+.2f} yaw={np.degrees(tyaw):+.1f}도\n"
            f"odom 날것(보정 전)     : x={ex:+.2f} y={ey:+.2f}   어긋난 정도 {odom_error:.2f}m\n"
            f"라이다 보정 후 추정치 : x={rx:+.2f} y={ry:+.2f}   어긋난 정도 {corrected_error:.2f}m   "
            f"(스캔 적중률 {sim.slam_lidar.last_fit_ratio*100:.0f}%)\n"
            f"내비게이션 상태: {sim.nav.status}   |   매니퓰레이션 상태: {sim.arm.status}   "
            f"|   그리퍼가 쥔 것: {sim.arm.attached_object.id if sim.arm.attached_object else '없음'}"
        )
        return [true_robot_poly, est_robot_poly, corrected_robot_poly, lidar_scatter, mapped_scatter,
                path_line, rgb_im, depth_im, info_text, *object_labels.values()]

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    from run import build_simulation
    run_viewer(build_simulation())
