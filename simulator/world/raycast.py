# ============================================================================
# world/raycast.py — 광선(ray) 여러 개가 맵 안의 도형들과 어디서 처음
#                     부딪히는지 한꺼번에(벡터화) 계산하기
#
# 라이다도, 뎁스카메라도 원리는 똑같다: "센서 위치에서 어떤 방향으로 광선을
# 쏘면, 그 광선이 맨 처음 무엇에 부딪히는가?"를 아주 많이(라이다는 수천 개
# 방향, 카메라는 픽셀 수만큼) 반복하는 것뿐이다.
#
# 처음 버전은 광선 하나마다 파이썬 함수를 호출하는 순수 반복문이었는데,
# 라이다 한 번 스캔에 ~0.4초씩 걸릴 만큼 느렸다(광선 수 x 도형 수 만큼
# 파이썬 레벨 호출이 발생하니 당연한 결과). 그래서 "광선 여러 개를 한
# 번에 numpy 배열로 계산"하도록 바꿨다 — 도형 개수(보통 몇~몇십 개)만큼만
# 파이썬 루프를 돌고, 그 안에서는 광선 수천 개를 numpy 벡터 연산으로
# 한꺼번에 처리한다. 외부 레이트레이싱 라이브러리 없이 numpy만으로 충분히
# 빨라진다.
# ============================================================================

import numpy as np

from math3d import xyzrpy_to_matrix, invert_transform


def _ray_box_batch(origins, dirs, half_size):
    """origins, dirs: (N,3). half_size: (3,). 반환: (N,) 거리(없으면 inf)."""
    t_min = np.full(origins.shape[0], -np.inf)
    t_max = np.full(origins.shape[0], np.inf)
    for i in range(3):
        d = dirs[:, i]
        o = origins[:, i]
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-half_size[i] - o) / d
            t2 = (half_size[i] - o) / d
        lo = np.minimum(t1, t2)
        hi = np.maximum(t1, t2)
        parallel = np.abs(d) < 1e-12
        miss_parallel = parallel & ((o < -half_size[i]) | (o > half_size[i]))
        lo = np.where(parallel, -np.inf, lo)
        hi = np.where(parallel, np.inf, hi)
        t_min = np.maximum(t_min, lo)
        t_max = np.minimum(t_max, hi)
        t_min = np.where(miss_parallel, np.inf, t_min)  # 강제로 "충돌 없음" 처리

    result = np.where(t_min >= 0, t_min, t_max)
    invalid = (t_min > t_max) | (t_max < 0)
    result = np.where(invalid, np.inf, result)
    return result


def _ray_cylinder_batch(origins, dirs, radius, height):
    ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
    dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    n = origins.shape[0]
    best = np.full(n, np.inf)

    a = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        b = 2 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - radius * radius
        disc = b * b - 4 * a * c
        has_side = (a > 1e-12) & (disc >= 0)
        sq = np.sqrt(np.where(disc >= 0, disc, 0.0))
        t_lo = (-b - sq) / (2 * a)
        t_hi = (-b + sq) / (2 * a)
    for t in (t_lo, t_hi):
        z = oz + t * dz
        valid = has_side & (t >= 0) & (z >= 0) & (z <= height) & (t < best)
        best = np.where(valid, t, best)

    for cap_z in (0.0, height):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (cap_z - oz) / dz
        x = ox + t * dx
        y = oy + t * dy
        valid = (np.abs(dz) > 1e-12) & (t >= 0) & (x * x + y * y <= radius * radius) & (t < best)
        best = np.where(valid, t, best)
    return best


def _ray_sphere_batch(origins, dirs, radius):
    b = 2 * np.sum(origins * dirs, axis=1)
    c = np.sum(origins * origins, axis=1) - radius * radius
    disc = b * b - 4 * c
    ok = disc >= 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t1 = (-b - sq) / 2
    t2 = (-b + sq) / 2
    t = np.where(t1 >= 0, t1, t2)
    valid = ok & (t >= 0)
    return np.where(valid, t, np.inf)


def raycast_scene_batch(origin, directions, shapes, max_range):
    """origin: (3,) 하나의 센서 위치. directions: (N,3) 정규화된 월드 좌표계
    방향 벡터들. 반환: (dists, hit_shapes) — dists는 (N,) 배열(못 맞으면
    max_range), hit_shapes는 길이 N 리스트(못 맞으면 None)."""
    n = directions.shape[0]
    best_t = np.full(n, float(max_range))
    best_shape = [None] * n
    origins_b = np.broadcast_to(origin, directions.shape)

    for shape in shapes:
        shape_t = xyzrpy_to_matrix(shape.xyz, shape.rpy)
        inv = invert_transform(shape_t)
        local_origins = origins_b @ inv[:3, :3].T + inv[:3, 3]
        local_dirs = directions @ inv[:3, :3].T

        if shape.kind == "box":
            t = _ray_box_batch(local_origins, local_dirs, shape.size / 2.0)
        elif shape.kind == "cylinder":
            local_origins = local_origins + np.array([0, 0, shape.height / 2.0])
            t = _ray_cylinder_batch(local_origins, local_dirs, shape.radius, shape.height)
        elif shape.kind == "sphere":
            t = _ray_sphere_batch(local_origins, local_dirs, shape.radius)
        else:
            continue

        closer = (t > 0) & (t < best_t)
        best_t = np.where(closer, t, best_t)
        idx = np.nonzero(closer)[0]
        for i in idx:
            best_shape[i] = shape

    return best_t, best_shape


# -- 광선 하나짜리 간단 버전 (테스트/디버깅, 소량 호출용) --------------------
def raycast_scene(ray_origin, ray_direction, shapes, max_range):
    dists, hits = raycast_scene_batch(np.asarray(ray_origin), np.asarray([ray_direction]), shapes, max_range)
    return float(dists[0]), hits[0]
