# tool/fusion/frustum_fallback.py
"""
Frustum Fallback v2: YOLO-only 검출에 대해 LiDAR frustum에서 3D box를 생성.

frustum-pointnets 논문의 아이디어 + 실용적 구현:
  1. YOLO 2D bbox → frustum 포인트 추출
  2. Frustum rotation normalization (frustum-pointnets)
  3. DBSCAN 3D 클러스터링 → foreground/background 분리
  4. Heatmap BEV grid search → 정밀 center + yaw 추정
  5. 2D 재투영 → YOLO bbox IoU 검증
  6. 포인트 없으면 bbox 크기 기반 depth 추정 (최종 fallback)

참고:
  - frustum-pointnets (Charles Qi, 2018): 구조적 아이디어 차용
  - /home/a/frustum/: DBSCAN + heatmap 구현 참고
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# Class size prior — frustum-pointnets g_type_mean_size 기반
# (KITTI training set 전체 통계, [l, w, h] 순서)
# ─────────────────────────────────────────────────────────────
CLASS_PRIOR = {
    "Car":        {"l": 3.88, "w": 1.63, "h": 1.53},
    "Pedestrian": {"l": 0.84, "w": 0.66, "h": 1.76},
    "Cyclist":    {"l": 1.76, "w": 0.60, "h": 1.74},
}

CAMERA_HEIGHT = 1.65

_CLS_PRIOR_LOWER = {k.lower(): v for k, v in CLASS_PRIOR.items()}
_CLS_NAME_TO_ID  = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

# ─────────────────────────────────────────────────────────────
# DBSCAN / Heatmap 설정 (기본값, pipeline CLI에서 override 가능)
# ─────────────────────────────────────────────────────────────
DBSCAN_DEFAULTS = {
    "eps":         0.7,   # 이웃 반경 [m]
    "min_samples": 3,     # 코어 포인트 최소 이웃 수
    "min_points":  3,     # 클러스터링 시도 최소 포인트 수
}

HEATMAP_DEFAULTS = {
    "grid_size":    0.15,   # BEV grid 해상도 [m]
    "edge_band":    0.15,   # 박스 엣지 근방 판정 거리 [m]
    "lam":          1.0,    # 엣지 포인트 스코어 가중치
    "yaw_step_deg": 5,      # yaw 탐색 간격 [deg]
    "roi_margin":   0.5,    # 포인트 extent 추가 여백 [m]
}


# ─────────────────────────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────────────────────────

def _get_class_prior(cls_name: str) -> dict:
    if cls_name in CLASS_PRIOR:
        return CLASS_PRIOR[cls_name]
    lower = cls_name.lower()
    if lower in _CLS_PRIOR_LOWER:
        return _CLS_PRIOR_LOWER[lower]
    return CLASS_PRIOR["Car"]


def _cls_name_to_id(cls_name: str) -> int:
    return _CLS_NAME_TO_ID.get(cls_name, _CLS_NAME_TO_ID.get(cls_name.capitalize(), 0))


def _lidar_to_cam_rect(points_lidar: np.ndarray, calib) -> np.ndarray:
    """LiDAR XYZ (N, 3+) → 카메라 직교 좌표 (N, 3)."""
    N = len(points_lidar)
    pts_hom = np.hstack([points_lidar[:, :3], np.ones((N, 1), dtype=np.float32)])
    return pts_hom @ (calib.V2C.T @ calib.R0.T)


def _cam_rect_to_img(pts_rect: np.ndarray, calib):
    """카메라 직교 좌표 (N, 3) → 이미지 픽셀 (N, 2), depth (N,)."""
    N = len(pts_rect)
    pts_hom = np.hstack([pts_rect, np.ones((N, 1), dtype=np.float32)])
    pts_2d  = pts_hom @ calib.P.T
    depth   = pts_rect[:, 2].copy()
    pts_img = np.zeros((N, 2), dtype=np.float32)
    valid   = depth > 0
    pts_img[valid] = pts_2d[valid, :2] / depth[valid, np.newaxis]
    return pts_img, depth


def _estimate_depth_from_bbox(cls_name: str, bbox2d: list, focal: float) -> float:
    """포인트 없을 때 2D box 높이 기반 depth 추정."""
    prior   = _get_class_prior(cls_name)
    _, y1, _, y2 = bbox2d
    bbox_h  = max(1.0, y2 - y1)
    return focal * prior["h"] / bbox_h


def _estimate_heading_ego(cx: float, cz: float) -> float:
    """객체가 ego vehicle을 향해 있다고 가정."""
    return float(np.arctan2(cx, cz))


# ─────────────────────────────────────────────────────────────
# Frustum Rotation Normalization (frustum-pointnets 핵심 아이디어)
# ─────────────────────────────────────────────────────────────

def _compute_frustum_angle(bbox2d: list, calib) -> float:
    """
    2D bbox center의 frustum angle 계산.
    frustum-pointnets: frustum_angle = -arctan2(z_rect, x_rect)
    """
    x1, y1, x2, y2 = bbox2d
    u_center = (x1 + x2) / 2.0
    v_center = (y1 + y2) / 2.0
    # 이미지 좌표 → 카메라 직교 좌표 (임의 depth=20)
    fx = float(calib.P[0, 0])
    fy = float(calib.P[1, 1])
    cx = float(calib.P[0, 2])
    cy = float(calib.P[1, 2])
    x_rect = (u_center - cx) * 20.0 / fx
    z_rect = 20.0
    return float(-np.arctan2(z_rect, x_rect))


def _rotate_pc_along_y(pts: np.ndarray, rot_angle: float) -> np.ndarray:
    """
    Y축 기준 회전 (frustum-pointnets provider.py).
    XZ 평면에서 rot_angle만큼 회전.
    """
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    pts_out = pts.copy()
    pts_out[:, 0] = cosval * pts[:, 0] + sinval * pts[:, 2]
    pts_out[:, 2] = -sinval * pts[:, 0] + cosval * pts[:, 2]
    return pts_out


# ─────────────────────────────────────────────────────────────
# DBSCAN 클러스터링 (foreground/background 분리)
# ─────────────────────────────────────────────────────────────

def _dbscan_cluster(pts_rect: np.ndarray, eps=0.7, min_samples=3, min_points=3):
    """
    DBSCAN으로 frustum 포인트를 클러스터링하여 foreground 클러스터 선택.

    선택 전략 (frustum-pointnets의 masking에 대응):
      1. 포인트 수가 가장 많은 클러스터
      2. 동률이면 depth(z) 중앙값이 가장 가까운 클러스터

    Returns:
        selected_pts: (M, 3) 선택된 클러스터 포인트
        all_labels:   (N,)  클러스터 레이블 (-1 = noise)
        n_clusters:   int
    """
    n = len(pts_rect)
    if n < min_points:
        return pts_rect, np.zeros(n, dtype=np.int32), 0

    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        # sklearn 없으면 전체 포인트 반환 (graceful fallback)
        return pts_rect, np.zeros(n, dtype=np.int32), 1

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(pts_rect).labels_
    unique_labels = sorted(set(labels.tolist()) - {-1})
    n_clusters = len(unique_labels)

    if n_clusters == 0:
        return pts_rect, labels, 0

    # 포인트 수 최대 클러스터 선택 (동률 시 z 최소)
    cluster_sizes = {lbl: int((labels == lbl).sum()) for lbl in unique_labels}
    max_size = max(cluster_sizes.values())
    candidates = [lbl for lbl, sz in cluster_sizes.items() if sz == max_size]

    if len(candidates) == 1:
        best_label = candidates[0]
    else:
        best_label = min(
            candidates,
            key=lambda lbl: float(np.median(pts_rect[labels == lbl, 2])),
        )

    return pts_rect[labels == best_label], labels, n_clusters


# ─────────────────────────────────────────────────────────────
# Heatmap BEV Grid Search (정밀 center + yaw 추정)
# ─────────────────────────────────────────────────────────────

def _heatmap_box_estimate(pts_rect, cls_name, init_yaw=None,
                          grid_size=0.15, edge_band=0.15, lam=1.0,
                          yaw_step_deg=5, roi_margin=0.5):
    """
    BEV x-z 그리드에서 앵커 적합도 탐색으로 center + yaw 추정.

    스코어 = log1p(N_in) + λ·log1p(N_edge)
      N_in:   앵커 박스 내부 포인트 수
      N_edge: 앵커 박스 엣지 근방 포인트 수

    Returns:
        (cx, cz, yaw, score) 또는 None
    """
    if len(pts_rect) < 2:
        return None

    prior = _get_class_prior(cls_name)
    anchor_l, anchor_w = prior["l"], prior["w"]
    half_l, half_w = anchor_l / 2.0, anchor_w / 2.0

    pts_xz = pts_rect[:, [0, 2]].astype(np.float32)

    # grid 범위
    grid_cx = np.arange(pts_xz[:, 0].min() - roi_margin,
                        pts_xz[:, 0].max() + roi_margin + 1e-6,
                        grid_size, dtype=np.float32)
    grid_cz = np.arange(pts_xz[:, 1].min() - roi_margin,
                        pts_xz[:, 1].max() + roi_margin + 1e-6,
                        grid_size, dtype=np.float32)

    if len(grid_cx) == 0 or len(grid_cz) == 0:
        return None

    # grid 크기 제한 (Jetson 메모리 보호: 최대 ~50x50 grid)
    MAX_GRID = 50
    if len(grid_cx) > MAX_GRID:
        step = grid_size * (len(grid_cx) / MAX_GRID)
        grid_cx = np.arange(grid_cx[0], grid_cx[-1] + 1e-6, step, dtype=np.float32)
    if len(grid_cz) > MAX_GRID:
        step = grid_size * (len(grid_cz) / MAX_GRID)
        grid_cz = np.arange(grid_cz[0], grid_cz[-1] + 1e-6, step, dtype=np.float32)

    # yaw 후보
    yaw_step = np.radians(yaw_step_deg)
    if init_yaw is not None:
        # OBB yaw hint 주변 ±30° + 180° flip
        half_range = np.radians(30)
        band1 = np.arange(init_yaw - half_range, init_yaw + half_range + 1e-9, yaw_step)
        band2 = np.arange(init_yaw + np.pi - half_range,
                          init_yaw + np.pi + half_range + 1e-9, yaw_step)
        yaw_candidates = np.concatenate([band1, band2]).astype(np.float32)
    else:
        yaw_candidates = np.arange(-np.pi, np.pi, yaw_step, dtype=np.float32)

    # grid centers
    CX, CZ = np.meshgrid(grid_cx, grid_cz)
    gc = np.stack([CX.ravel(), CZ.ravel()], axis=1).astype(np.float32)  # (G, 2)
    G = len(gc)

    best_scores = np.full(G, -1e9, dtype=np.float32)
    best_yaws   = np.zeros(G, dtype=np.float32)

    for yaw in yaw_candidates:
        cr, sr = float(np.cos(yaw)), float(np.sin(yaw))
        R = np.array([[cr, sr], [-sr, cr]], dtype=np.float32)

        pts_rot = pts_xz @ R    # (N, 2)
        gc_rot  = gc @ R        # (G, 2)

        # diff[g, n] = pts_rot[n] - gc_rot[g]
        diff = pts_rot[None, :, :] - gc_rot[:, None, :]  # (G, N, 2)

        inside = (np.abs(diff[..., 0]) <= half_l) & (np.abs(diff[..., 1]) <= half_w)
        n_in   = inside.sum(axis=1).astype(np.float32)

        dist_to_edge = np.minimum(
            half_l - np.abs(diff[..., 0]),
            half_w - np.abs(diff[..., 1]),
        )
        n_edge = (inside & (dist_to_edge <= edge_band)).sum(axis=1).astype(np.float32)

        score = np.where(n_in > 0,
                         np.log1p(n_in) + lam * np.log1p(n_edge),
                         np.float32(-1e9))

        mask = score > best_scores
        best_scores[mask] = score[mask]
        best_yaws[mask]   = yaw

    flat_idx = int(np.argmax(best_scores))
    if best_scores[flat_idx] <= -1e8:
        return None

    iy, ix = np.unravel_index(flat_idx, (len(grid_cz), len(grid_cx)))
    return (float(grid_cx[ix]), float(grid_cz[iy]),
            float(best_yaws[flat_idx]), float(best_scores[flat_idx]))


def _obb_initial_yaw(pts_rect: np.ndarray):
    """Open3D OBB로 수평 방향 초기 yaw 추정 (heatmap 탐색 범위 힌트)."""
    try:
        import open3d as o3d
    except ImportError:
        return None

    if len(pts_rect) < 4:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_rect.astype(np.float64))

    try:
        obb = pcd.get_oriented_bounding_box()
    except Exception:
        return None

    extents = np.asarray(obb.extent)
    R = np.asarray(obb.R)

    y_cam = np.array([0.0, 1.0, 0.0])
    y_idx = int(np.argmax(np.abs(R.T @ y_cam)))
    horiz = [i for i in range(3) if i != y_idx]
    l_idx = horiz[0] if extents[horiz[0]] >= extents[horiz[1]] else horiz[1]
    l_axis = R[:, l_idx]

    return float(np.arctan2(-l_axis[2], l_axis[0]))


# ─────────────────────────────────────────────────────────────
# 3D 박스 관련 유틸
# ─────────────────────────────────────────────────────────────

def _box3d_corners(cx, cy, cz, h, w, l, ry):
    """3D 박스 8개 꼭짓점 (카메라 좌표). frustum-pointnets get_3d_box와 동일."""
    cos_r, sin_r = np.cos(ry), np.sin(ry)
    R = np.array([[ cos_r, 0, sin_r],
                  [     0, 1,     0],
                  [-sin_r, 0, cos_r]])
    # l→X, h→Y, w→Z (frustum-pointnets convention)
    x_corners = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
    y_corners = np.array([ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2])
    z_corners = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2])
    corners = R @ np.vstack([x_corners, y_corners, z_corners])
    corners[0, :] += cx
    corners[1, :] += cy
    corners[2, :] += cz
    return corners.T  # (8, 3)


def _project_3d_box_to_2d(location, dimensions, ry, calib):
    """3D 박스 → 2D bbox [x1, y1, x2, y2]."""
    cx, cy, cz = location
    h, w, l = dimensions
    corners = _box3d_corners(cx, cy, cz, h, w, l, ry)

    valid = corners[:, 2] > 0.1
    if not valid.any():
        return None

    corners_v = corners[valid]
    pts_hom = np.hstack([corners_v, np.ones((len(corners_v), 1))])
    pts_2d = pts_hom @ calib.P.T
    pts_img = pts_2d[:, :2] / pts_2d[:, 2:3]

    return [float(pts_img[:, 0].min()), float(pts_img[:, 1].min()),
            float(pts_img[:, 0].max()), float(pts_img[:, 1].max())]


# ─────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────

def extract_frustum_points(
    points_lidar: np.ndarray,
    calib,
    bbox2d: list,
    near: float = 0.5,
    far:  float = 60.0,
):
    """
    YOLO 2D bbox에 해당하는 frustum 내 LiDAR 포인트를 필터링.

    Returns:
        pts_lidar_in (N', 3): frustum 내 LiDAR 포인트
        pts_cam_in   (N', 3): 동일 포인트의 카메라 직교 좌표
        depths_in    (N',):   camera Z (forward depth)
    """
    x1, y1, x2, y2 = bbox2d

    pts_cam = _lidar_to_cam_rect(points_lidar, calib)
    pts_img, depths = _cam_rect_to_img(pts_cam, calib)

    mask = (
        (depths  >= near)      & (depths  <= far)  &
        (pts_img[:, 0] >= x1)  & (pts_img[:, 0] <= x2) &
        (pts_img[:, 1] >= y1)  & (pts_img[:, 1] <= y2)
    )
    return points_lidar[mask, :3], pts_cam[mask], depths[mask]


def generate_frustum_box(
    points_lidar: np.ndarray,
    calib,
    yolo_det: dict,
    near: float = 0.5,
    far:  float = 60.0,
    # DBSCAN 파라미터
    dbscan_eps:         float = 0.7,
    dbscan_min_samples: int   = 3,
    dbscan_min_points:  int   = 3,
    # Heatmap 파라미터
    heatmap_grid_size:  float = 0.15,
    heatmap_edge_band:  float = 0.15,
    heatmap_lam:        float = 1.0,
    heatmap_yaw_step:   int   = 5,
    heatmap_roi_margin: float = 0.5,
    # 기존 호환
    depth_bin_size:     float = 1.0,
    min_pts_per_bin:    int   = 2,
    min_pts_for_pca:    int   = 5,
    debug: bool = False,
) -> dict | None:
    """
    YOLO 검출 1개에 대해 frustum fallback 3D box를 생성.

    파이프라인:
      1. Frustum 포인트 추출
      2. Frustum rotation normalization (frustum-pointnets)
      3. DBSCAN 클러스터링 → foreground 분리
      4. Heatmap BEV grid search → center + yaw 정밀 추정
      5. 정규화 역변환 → 원래 카메라 좌표계로 복원
      6. 2D 재투영 → YOLO bbox IoU 검증
      7. 포인트 없으면 bbox 크기 기반 depth 추정

    Returns:
        KITTI 16-field 호환 dict, 또는 None
    """
    from matcher import compute_iou

    bbox2d    = yolo_det["bbox"]
    cls_name  = yolo_det["cls_name"]
    yolo_conf = float(yolo_det["score"])
    focal     = float(calib.P[0, 0])

    # ── 1. Frustum 포인트 추출 ────────────────────────────────
    pts_lidar_in, pts_cam_in, depths_in = extract_frustum_points(
        points_lidar, calib, bbox2d, near=near, far=far
    )
    n_pts = len(pts_lidar_in)

    if debug:
        print(f"[FRUSTUM] cls={cls_name} frustum_pts={n_pts}")

    # ── 2~6. 포인트가 있는 경우: DBSCAN + Heatmap ────────────
    if n_pts >= dbscan_min_points:

        # ── 2. Frustum rotation normalization ─────────────────
        frustum_angle = _compute_frustum_angle(bbox2d, calib)
        rot_angle = np.pi / 2.0 + frustum_angle  # frustum-pointnets convention

        pts_normalized = _rotate_pc_along_y(pts_cam_in, rot_angle)

        # ── 3. DBSCAN 클러스터링 ──────────────────────────────
        fg_pts, labels, n_clusters = _dbscan_cluster(
            pts_normalized,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            min_points=dbscan_min_points,
        )
        n_fg = len(fg_pts)

        if debug:
            print(f"[FRUSTUM] DBSCAN: {n_clusters} clusters, "
                  f"fg={n_fg}/{n_pts} pts")

        # ── 4. Heatmap BEV grid search (정규화된 좌표에서) ────
        # OBB yaw hint (선택적)
        init_yaw = _obb_initial_yaw(fg_pts) if n_fg >= 4 else None

        heatmap_result = _heatmap_box_estimate(
            fg_pts, cls_name, init_yaw=init_yaw,
            grid_size=heatmap_grid_size,
            edge_band=heatmap_edge_band,
            lam=heatmap_lam,
            yaw_step_deg=heatmap_yaw_step,
            roi_margin=heatmap_roi_margin,
        )

        if heatmap_result is not None:
            cx_norm, cz_norm, yaw_norm, score = heatmap_result
            cy_norm = float(fg_pts[:, 1].max())  # Y-down → max = 물체 하단

            if debug:
                print(f"[FRUSTUM] Heatmap: center=({cx_norm:.2f},{cz_norm:.2f}) "
                      f"yaw={np.degrees(yaw_norm):.1f}° score={score:.3f}")

            # ── 5. 역변환: 정규화 좌표 → 원래 카메라 좌표 ────
            center_norm = np.array([[cx_norm, cy_norm, cz_norm]])
            center_cam = _rotate_pc_along_y(center_norm, -rot_angle)[0]
            cx, cy, cz = float(center_cam[0]), float(center_cam[1]), float(center_cam[2])
            ry = yaw_norm - rot_angle  # heading도 역회전

            prior = _get_class_prior(cls_name)
            h, w, l = prior["h"], prior["w"], prior["l"]
            loc  = [cx, cy, cz]
            dims = [h, w, l]

        else:
            # Heatmap 실패 → centroid fallback (정규화 좌표 기반)
            cx_norm = float(fg_pts[:, 0].mean())
            cy_norm = float(fg_pts[:, 1].max())
            cz_norm = float(fg_pts[:, 2].mean())

            if n_fg >= min_pts_for_pca:
                xz = fg_pts[:, [0, 2]]
                cov = np.cov((xz - xz.mean(axis=0)).T)
                if cov.ndim >= 2 and not np.all(cov == 0):
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    principal = eigvecs[:, np.argmax(eigvals)]
                    yaw_norm = float(np.arctan2(principal[0], principal[1]))
                else:
                    yaw_norm = _estimate_heading_ego(cx_norm, cz_norm)
            else:
                yaw_norm = _estimate_heading_ego(cx_norm, cz_norm)

            center_norm = np.array([[cx_norm, cy_norm, cz_norm]])
            center_cam = _rotate_pc_along_y(center_norm, -rot_angle)[0]
            cx, cy, cz = float(center_cam[0]), float(center_cam[1]), float(center_cam[2])
            ry = yaw_norm - rot_angle

            prior = _get_class_prior(cls_name)
            h, w, l = prior["h"], prior["w"], prior["l"]
            loc  = [cx, cy, cz]
            dims = [h, w, l]

            if debug:
                print(f"[FRUSTUM] Centroid fallback: center=({cx:.2f},{cz:.2f})")

        # ── 6. 2D 재투영 → YOLO bbox IoU 검증 ────────────────
        proj = _project_3d_box_to_2d(loc, dims, ry, calib)
        if proj is not None:
            iou = compute_iou(proj, bbox2d)
            if debug:
                print(f"[FRUSTUM] 2D IoU={iou:.3f}")
        else:
            iou = 0.0

        # Confidence 계산
        if n_fg >= 10:
            point_factor = 0.9
        elif n_fg >= dbscan_min_points:
            point_factor = 0.8
        else:
            point_factor = 0.6
        final_conf = yolo_conf * point_factor

        if debug:
            print(f"[FRUSTUM] SELECTED cls={cls_name} n_fg={n_fg} "
                  f"loc=({loc[0]:.1f},{loc[2]:.1f}) ry={np.degrees(ry):.1f}° "
                  f"conf={final_conf:.4f}")

    else:
        # ── 7. 포인트 없음: bbox 크기 기반 depth 추정 ────────
        depth = _estimate_depth_from_bbox(cls_name, bbox2d, focal)
        if depth is None or not (near <= depth <= far):
            if debug:
                print(f"[FRUSTUM] size-based depth={depth}, out of range, skip")
            return None

        x1, y1, x2, y2 = bbox2d
        u    = (x1 + x2) / 2.0
        fx   = float(calib.P[0, 0])
        cx_p = float(calib.P[0, 2])
        cx   = (u - cx_p) * depth / fx
        cy   = CAMERA_HEIGHT
        cz   = depth

        prior = _get_class_prior(cls_name)
        h, w, l = prior["h"], prior["w"], prior["l"]
        ry = _estimate_heading_ego(cx, cz)
        loc  = [cx, cy, cz]
        dims = [h, w, l]

        final_conf = yolo_conf * 0.3 * 0.5

        if debug:
            print(f"[FRUSTUM] SIZE-BASED cls={cls_name} depth={depth:.2f} "
                  f"cx={cx:.2f} conf={final_conf:.4f}")

    x1, y1, x2, y2 = bbox2d
    return {
        "cls_name":   cls_name,
        "cls_id":     _cls_name_to_id(cls_name),
        "truncated":  0.0,
        "occluded":   0,
        "alpha":      -10.0,
        "bbox":       [float(x1), float(y1), float(x2), float(y2)],
        "dimensions": [float(dims[0]), float(dims[1]), float(dims[2])],
        "location":   [float(loc[0]), float(loc[1]), float(loc[2])],
        "rotation_y": float(ry),
        "score":      float(final_conf),
    }


# ─────────────────────────────────────────────────────────────
# BEV 중복 제거
# ─────────────────────────────────────────────────────────────

def filter_overlapping_fallbacks(fallback_boxes, pp_preds, debug=False):
    """
    Frustum fallback box가 기존 PP box와 BEV에서 겹치면 제거.

    동일 클래스끼리만 비교하며, BEV 중심 거리가 해당 클래스 prior의
    max(l, w) × 0.75 이내이면 중복으로 판정.
    """
    if not fallback_boxes or not pp_preds:
        return list(fallback_boxes)

    filtered = []
    for fb in fallback_boxes:
        fb_x, _, fb_z = fb["location"]
        fb_cls = fb["cls_name"]

        is_dup = False
        for pp in pp_preds:
            if pp["cls_name"] != fb_cls:
                continue
            pp_x, _, pp_z = pp["location"]
            dist = np.sqrt((fb_x - pp_x) ** 2 + (fb_z - pp_z) ** 2)
            h_pp, w_pp, l_pp = pp["dimensions"]
            thr = max(l_pp, w_pp) * 0.75
            if dist < thr:
                is_dup = True
                if debug:
                    print(
                        f"[FRUSTUM-DEDUP] {fb_cls} fallback at "
                        f"({fb_x:.1f},{fb_z:.1f}) too close to PP at "
                        f"({pp_x:.1f},{pp_z:.1f}), dist={dist:.2f}<{thr:.2f}, drop"
                    )
                break

        if not is_dup:
            filtered.append(fb)

    return filtered
