"""단일 이미지 -> 3D 포인트클라우드 (깊이추정 + 오브젝트 검출/이름/3D 박스).

depth: HF transformers pipeline (Depth-Anything-V2-Small)
segmentation: ultralytics FastSAM (전체 오브젝트 커버리지, COCO 클래스 제한 없음)
naming: ultralytics YOLOv8-seg (COCO 클래스 이름) — FastSAM 마스크와 겹치는
        것만 이름을 붙이고, 안 겹치면 "object_N"으로 둔다 (YOLO는 80종류뿐이라
        커버리지가 FastSAM보다 훨씬 좁음 — 그래서 이름표만 붙이는 용도로만 씀).
뷰어: Open3D draw_geometries (GPU/하드웨어 가속, GLFW) — 포인트클라우드 +
      오브젝트별 색상 3D 바운딩박스를 한 창에서 바로 보여주고 드래그로
      빠르게 회전할 수 있다. (matplotlib mplot3d는 매 프레임 CPU로 모든
      점/선/텍스트를 다시 그려서 84개 박스+라벨 정도만 있어도 회전이
      버벅였음 — 그래픽카드를 못 씀. Open3D의 신형 GUI(O3DVisualizer)는
      3D 텍스트 라벨도 그릴 수 있지만 macOS에서 app.run() 중 크래시가
      나서 안정적인 구형 API(draw_geometries)를 씀 — 그래서 이름/좌표는
      3D 뷰 안 텍스트 대신 콘솔 출력 + objects.json으로 제공한다.)

python 3d/view3d.py [이미지경로]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

WEIGHTS_DIR = Path(__file__).parent  # 모델 가중치를 실행 위치(cwd)가 아니라 항상 여기(3d/)에 받게 고정
IOU_NAME_THRESHOLD = 0.3  # FastSAM 마스크가 YOLO 검출과 이 정도 이상 겹치면 그 이름을 물려받음
# ponytail: 실제 카메라 캘리브레이션 정보가 없어서 수평 시야각을 이 값으로
# 가정하고 카메라 위치/프러스텀을 역산한다 (스마트폰/웹캠 표준 화각 근사치).
ASSUMED_HFOV_DEG = 60.0


def estimate_depth(img):
    from transformers import pipeline
    depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    out = depth_pipe(img)
    return np.array(out["depth"].resize(img.size), dtype=np.float32)


def _mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def detect_objects(path):
    """반환: [(mask(h,w) bool, name, confidence), ...]. FastSAM으로 전체
    오브젝트를 다 잡고, YOLOv8-seg와 겹치는 것만 실제 이름을 붙인다."""
    from ultralytics import FastSAM, YOLO

    fast_result = FastSAM(str(WEIGHTS_DIR / "FastSAM-s.pt"))(path, verbose=False)[0]
    if fast_result.masks is None:
        return []
    fast_masks = fast_result.masks.data.cpu().numpy() > 0.5

    yolo_result = YOLO(str(WEIGHTS_DIR / "yolov8n-seg.pt"))(path, verbose=False)[0]
    yolo_masks = np.zeros((0, *fast_masks.shape[1:]), dtype=bool)
    yolo_names, yolo_confs = [], []
    if yolo_result.masks is not None:
        yolo_masks = yolo_result.masks.data.cpu().numpy() > 0.5
        yolo_names = [yolo_result.names[int(c)] for c in yolo_result.boxes.cls]
        yolo_confs = yolo_result.boxes.conf.cpu().numpy()

    detections = []
    for i, fm in enumerate(fast_masks):
        ious = [_mask_iou(fm, ym) for ym in yolo_masks]
        best_j = int(np.argmax(ious)) if ious else -1
        if best_j >= 0 and ious[best_j] > IOU_NAME_THRESHOLD:
            detections.append((fm, yolo_names[best_j], float(yolo_confs[best_j])))
        else:
            detections.append((fm, f"object_{i + 1}", 0.0))
    return detections


def estimate_vanishing_point(img_rgb):
    """Canny+Hough 직선들의 최소자승 교차점으로 소실점 근사 (여러 방향의
    소실점을 다 분리하진 않고, 가장 두드러진 교차점 하나만 잡는 단순 버전).
    반환: (px, py) 픽셀 좌표, 직선을 못 찾으면 None."""
    import cv2
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=10)
    if lines is None:
        return None
    A, b = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a, bb, c = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
        norm = np.hypot(a, bb) + 1e-9
        A.append([a / norm, bb / norm])
        b.append(-c / norm)
    vp, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return float(vp[0]), float(vp[1])


def resize_mask(mask, size_hw):
    if mask.shape == size_hw:
        return mask
    h, w = size_hw
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h))) > 127


def build_cloud(depth, full_masks, img_size, stride=1):
    w, h = img_size
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[ys, xs]

    seg_id = np.zeros_like(z, dtype=np.int32)
    # 큰 마스크부터 칠하고 작은 마스크를 나중에 덮어쓴다 — 순서대로(원래
    # FastSAM 출력 순서로) 칠하면 쿠션/꽃병처럼 작은 물체가 소파/테이블 같은
    # 큰 마스크에 완전히 가려져서 포인트를 하나도 못 받는 경우가 생긴다.
    order = np.argsort([m.sum() for m in full_masks])[::-1]  # 큰 것 -> 작은 것 순
    for i in order:
        seg_id[full_masks[i][ys, xs]] = i + 1

    x = (xs - w / 2).astype(np.float32)
    y = (h / 2 - ys).astype(np.float32)  # 이미지 위쪽 = +y
    z_scaled = (z - z.min()) / (np.ptp(z) + 1e-6) * (w / 3)  # 보기 좋게 스케일만 맞춤 (미터 단위 아님)

    assert x.shape == y.shape == z.shape == seg_id.shape
    return x.ravel(), y.ravel(), z_scaled.ravel(), seg_id.ravel()


def object_infos(x, y, z, seg_id, detections):
    """오브젝트별 (min/max/center 좌표 dict) 목록 — json.dump 가능,
    나중에 로봇 코드에서 좌표/이름 읽어 쓰기 위함."""
    infos = []
    for i, (_m, name, conf) in enumerate(detections, start=1):
        sel = seg_id == i
        if not sel.any():
            continue
        pts = np.stack([x[sel], y[sel], z[sel]], axis=1)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        center = ((lo + hi) / 2).tolist()
        infos.append({"seg_id": i, "name": name, "confidence": conf,
                      "center_xyz": center, "min_xyz": lo.tolist(), "max_xyz": hi.tolist()})
    return infos


def show_3d(x, y, z, seg_id, infos, palette, path, camera):
    """Open3D(GPU) 창 하나에 포인트클라우드 + 오브젝트별 색상 박스 +
    카메라 위치/시야각(FOV) 프러스텀 + 소실점을 그린다.
    O3DVisualizer(신형 GUI)의 add_3d_label로 텍스트도 시도해봤지만 이
    macOS 환경에서 파이썬 예외로도 못 잡는 네이티브 크래시(검정 화면+
    깜빡임)가 나서 포기 — 안정적인 구형 API(draw_geometries)만 쓴다.
    이름/좌표는 콘솔 출력 + objects.json으로 확인."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack([x, y, z], axis=1))
    colors = np.full((seg_id.size, 3), 0.65)  # 미분할 배경 = 회색
    for i in range(1, seg_id.max() + 1):
        colors[seg_id == i] = palette[(i - 1) % len(palette)]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    boxes = []
    for info in infos:
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound=info["min_xyz"], max_bound=info["max_xyz"])
        box.color = palette[(info["seg_id"] - 1) % len(palette)]
        boxes.append(box)

    marker_r = camera["marker_radius"]
    cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_r)
    cam_sphere.translate(camera["position"])
    cam_sphere.paint_uniform_color((1.0, 0.5, 0.0))  # 주황 = 카메라
    extra = [cam_sphere]

    # 카메라의 좌우(X,빨강)/상하(Y,초록)/정면(-Z,파랑) 방향 축 — 우리 좌표계가
    # 이미 x=오른쪽+, y=위쪽+, z=카메라 쪽으로 갈수록 커지게 구성돼 있어서
    # (build_cloud 참고) 월드 축과 그대로 정렬된 좌표축을 카메라 위치에
    # 그리는 것만으로 카메라 기준 좌우/상하/정면이 그대로 표시된다.
    cam_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=marker_r * 5, origin=camera["position"])
    extra.append(cam_axes)

    corners = camera["frustum_corners"]
    pts = [camera["position"], *corners]
    lines = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    frustum.paint_uniform_color((1.0, 1.0, 0.0))  # 노랑 = FOV 프러스텀
    extra.append(frustum)

    if camera["vanishing_point"] is not None:
        vp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_r)
        vp_sphere.translate(camera["vanishing_point"])
        vp_sphere.paint_uniform_color((1.0, 0.0, 1.0))  # 자홍 = 소실점
        extra.append(vp_sphere)

    o3d.visualization.draw_geometries(
        [pcd, *boxes, *extra],
        window_name=(f"{Path(path).name} — 객체 {len(infos)}개 (드래그=회전·스크롤=줌, "
                     f"주황=카메라·빨강/초록/파랑축=카메라 좌우/상하/정면·노랑=FOV·자홍=소실점, "
                     f"이름/좌표는 콘솔+objects.json)"),
    )


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(WEIGHTS_DIR / "img.png")
    img = Image.open(path).convert("RGB")
    w, h = img.size

    print("깊이 추정 중...")
    depth = estimate_depth(img)
    print("객체 검출 중 (FastSAM 전체 분할 + YOLO 이름 매칭)...")
    detections = detect_objects(path)
    full_masks = [resize_mask(m, (h, w)) for m, _name, _conf in detections]
    print(f"객체 {len(detections)}개 검출 (이름 있는 것: "
          f"{sum(1 for _, n, _ in detections if not n.startswith('object_'))}개)")

    x, y, z, seg_id = build_cloud(depth, full_masks, img.size)
    assert x.size > 0, "포인트클라우드가 비었음"

    cmap = plt.get_cmap("tab20")
    palette = [cmap(k)[:3] for k in range(20)]

    infos = object_infos(x, y, z, seg_id, detections)
    if len(infos) < len(detections):
        dropped = set(range(1, len(detections) + 1)) - {info["seg_id"] for info in infos}
        print(f"참고: {len(dropped)}개는 다른 마스크에 완전히 가려져 포인트 0개 "
              f"(seg_id: {sorted(dropped)})")

    z_near, z_far = float(z.max()), float(z.min())
    cam_dist = float((w / 2) / np.tan(np.radians(ASSUMED_HFOV_DEG / 2)))  # 가정 FOV로부터 역산한 카메라~이미지평면 거리
    camera_pos = (0.0, 0.0, z_near + cam_dist)
    frustum_corners = [(sx * w / 2, sy * h / 2, z_near) for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))]
    vp_px = estimate_vanishing_point(np.array(img))
    vanishing_point = (vp_px[0] - w / 2, h / 2 - vp_px[1], z_far - cam_dist * 0.5) if vp_px else None
    print(f"카메라 위치(가정 수평FOV {ASSUMED_HFOV_DEG:.0f}°): {tuple(round(v, 1) for v in camera_pos)}")
    print(f"소실점: {tuple(round(v, 1) for v in vanishing_point) if vanishing_point else '검출 안 됨'}")
    camera = {
        "position": camera_pos, "frustum_corners": frustum_corners, "vanishing_point": vanishing_point,
        "marker_radius": w * 0.01, "assumed_hfov_deg": ASSUMED_HFOV_DEG,
    }

    import json
    out_json = WEIGHTS_DIR / "objects.json"
    out_json.write_text(json.dumps({"objects": infos, "camera": camera}, ensure_ascii=False, indent=2))
    print(f"좌표/이름/카메라 저장: {out_json}")
    for info in infos:
        cx, cy, cz = info["center_xyz"]
        print(f"  {info['name']:20s} ({cx:6.0f}, {cy:6.0f}, {cz:6.0f})")

    show_3d(x, y, z, seg_id, infos, palette, path, camera)


if __name__ == "__main__":
    main()
