# tool/fusion/frustum_fallback.py
"""
Frustum Fallback: YOLO-only 검출에 대해 LiDAR frustum에서 BEV box를 생성.

흐름:
  1. YOLO 2D bbox로 frustum 정의 (near/far plane)
  2. LiDAR → 카메라 좌표 변환 후 frustum 내 포인트 필터링
  3. 포인트 수에 따라 depth 결정 (히스토그램 / median / bbox 크기 추정)
  4. BEV 중심점 결정
  5. Class prior로 box 크기 결정
  6. PCA 또는 ego 방향으로 heading 추정
  7. Confidence = yolo_conf × point_factor (× 0.5 if size-based)
"""

import numpy as np

# KITTI 통계 기반 class 크기 prior (쉽게 튜닝할 수 있도록 최상단에 노출)
CLASS_PRIOR = {
    "Car":        {"w": 1.8, "l": 4.0, "h": 1.5},
    "Pedestrian": {"w": 0.6, "l": 0.6, "h": 1.7},
    "Cyclist":    {"w": 0.6, "l": 1.8, "h": 1.7},
}

# KITTI 카메라가 지면으로부터 약 1.65m 높이에 위치 — cy 추정 fallback에 사용
CAMERA_HEIGHT = 1.65

_CLS_PRIOR_LOWER = {k.lower(): v for k, v in CLASS_PRIOR.items()}
_CLS_NAME_TO_ID  = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


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
    """LiDAR XYZ (N, 3+) → 카메라 직교 좌표 (N, 3). pipeline.get_fov_flag와 동일 변환."""
    N = len(points_lidar)
    pts_hom = np.hstack([points_lidar[:, :3], np.ones((N, 1), dtype=np.float32)])
    return pts_hom @ (calib.V2C.T @ calib.R0.T)   # (N, 3)


def _cam_rect_to_img(pts_rect: np.ndarray, calib):
    """카메라 직교 좌표 (N, 3) → 이미지 픽셀 (N, 2), depth (N,)."""
    N = len(pts_rect)
    pts_hom = np.hstack([pts_rect, np.ones((N, 1), dtype=np.float32)])
    pts_2d  = pts_hom @ calib.P.T           # (N, 3)
    depth   = pts_rect[:, 2].copy()         # camera Z
    pts_img = np.zeros((N, 2), dtype=np.float32)
    valid   = depth > 0
    pts_img[valid] = pts_2d[valid, :2] / depth[valid, np.newaxis]
    return pts_img, depth


def _histogram_depth_centroid(depths: np.ndarray, bin_size: float = 1.0) -> float:
    """depth 히스토그램에서 가장 밀집된 bin의 centroid 반환."""
    d_min, d_max = depths.min(), depths.max()
    if d_max - d_min < bin_size:
        return float(np.median(depths))

    bins = np.arange(d_min, d_max + bin_size, bin_size)
    counts, edges = np.histogram(depths, bins=bins)
    peak = np.argmax(counts)
    lo, hi = edges[peak], edges[peak + 1]
    in_bin = depths[(depths >= lo) & (depths < hi)]
    return float(np.mean(in_bin)) if len(in_bin) > 0 else float(np.median(depths))


def _estimate_depth_from_bbox(cls_name: str, bbox2d: list, focal: float) -> float:
    """포인트 없을 때 2D box 높이 기반 depth 추정: depth = f * h_prior / bbox_h."""
    prior   = _get_class_prior(cls_name)
    _, y1, _, y2 = bbox2d
    bbox_h  = max(1.0, y2 - y1)
    return focal * prior["h"] / bbox_h


def _estimate_heading_pca(pts_cam: np.ndarray) -> float:
    """카메라 XZ 평면의 PCA → 첫 번째 주성분 방향을 rotation_y (KITTI 기준)로 반환."""
    xz = pts_cam[:, [0, 2]]          # (N, 2): camera X, Z
    if len(xz) < 2:
        cx, cz = xz[0] if len(xz) == 1 else (0.0, 1.0)
        return float(np.arctan2(cx, cz))

    cov = np.cov((xz - xz.mean(axis=0)).T)
    if cov.ndim < 2 or np.all(cov == 0):
        cx, cz = xz.mean(axis=0)
        return float(np.arctan2(cx, cz))

    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]   # [x_dir, z_dir]
    return float(np.arctan2(principal[0], principal[1]))


def _estimate_heading_ego(cx: float, cz: float) -> float:
    """객체가 ego vehicle을 향해 있다고 가정 → atan2(cx, cz)."""
    return float(np.arctan2(cx, cz))


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

    pts_cam = _lidar_to_cam_rect(points_lidar, calib)       # (N, 3)
    pts_img, depths = _cam_rect_to_img(pts_cam, calib)      # (N, 2), (N,)

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
    depth_bin_size:          float = 1.0,
    min_pts_for_histogram:   int   = 5,
    min_pts_for_pca:         int   = 5,
    debug: bool = False,
) -> dict | None:
    """
    YOLO 검출 1개에 대해 frustum fallback BEV box dict를 생성.

    Args:
        points_lidar: 이미 FOV 필터링된 LiDAR 포인트 (N, 4) 또는 (N, 3)
        calib:        kitti_util.Calibration 객체
        yolo_det:     {"bbox": [x1,y1,x2,y2], "cls_name": str, "score": float}
        near/far:     frustum 깊이 범위 (m)
        depth_bin_size: 히스토그램 bin 크기 (m)
        min_pts_for_histogram: 히스토그램 전략 최소 포인트 수
        min_pts_for_pca:       PCA heading 최소 포인트 수
        debug:        상세 로그 출력

    Returns:
        KITTI 16-field 호환 dict, 또는 생성 실패 시 None
    """
    bbox2d    = yolo_det["bbox"]
    cls_name  = yolo_det["cls_name"]
    yolo_conf = float(yolo_det["score"])
    focal     = float(calib.P[0, 0])

    # ── 1. Frustum 포인트 추출 ────────────────────────────────
    pts_lidar_in, pts_cam_in, depths_in = extract_frustum_points(
        points_lidar, calib, bbox2d, near=near, far=far
    )
    n_pts = len(pts_lidar_in)

    # ── 2. Depth 결정 ─────────────────────────────────────────
    size_based = False
    if n_pts >= min_pts_for_histogram:
        depth        = _histogram_depth_centroid(depths_in, bin_size=depth_bin_size)
        # 밀집도 보너스: 10+ 포인트가 좁은 depth 범위에 모여있으면 신뢰도 상향
        if n_pts >= 10 and np.std(depths_in) < 2.0:
            point_factor = 0.9
        else:
            point_factor = 0.8
    elif n_pts >= 1:
        depth        = float(np.median(depths_in))
        point_factor = 0.6
    else:
        depth        = _estimate_depth_from_bbox(cls_name, bbox2d, focal)
        point_factor = 0.3
        size_based   = True

    if depth is None or not (near <= depth <= far):
        if debug:
            print(f"[FRUSTUM] invalid depth={depth}, skip")
        return None

    # ── 3. BEV 중심점 결정 ───────────────────────────────────
    if n_pts >= min_pts_for_histogram:
        # 가장 밀집된 depth bin 내 포인트만으로 중심 계산
        lo = depth - depth_bin_size / 2
        hi = depth + depth_bin_size / 2
        mask_bin = (depths_in >= lo) & (depths_in <= hi)
        pts_center = pts_cam_in[mask_bin] if mask_bin.any() else pts_cam_in

        cx  = float(pts_center[:, 0].mean())
        cy  = float(pts_center[:, 1].max())   # camera Y-down → 최대값 = 물체 하단
        cz  = float(pts_center[:, 2].mean())
    elif n_pts >= 1:
        cx  = float(pts_cam_in[:, 0].mean())
        cy  = float(pts_cam_in[:, 1].max())
        cz  = float(pts_cam_in[:, 2].mean())
    else:
        # 2D bbox 중심을 추정 depth로 역투영
        x1, y1, x2, y2 = bbox2d
        u   = (x1 + x2) / 2.0
        fx  = float(calib.P[0, 0])
        cx_p = float(calib.P[0, 2])
        cx  = (u - cx_p) * depth / fx
        cy  = CAMERA_HEIGHT      # 카메라 지면 높이로 대체
        cz  = depth

    # ── 4. Box 크기 (class prior) ─────────────────────────────
    prior = _get_class_prior(cls_name)
    h, w, l = prior["h"], prior["w"], prior["l"]

    # ── 5. Heading 추정 ───────────────────────────────────────
    if n_pts >= min_pts_for_pca and not size_based:
        ry = _estimate_heading_pca(pts_cam_in)
    else:
        ry = _estimate_heading_ego(cx, cz)

    # ── 6. Confidence ─────────────────────────────────────────
    conf_scale = 0.5 if size_based else 1.0
    final_conf = yolo_conf * point_factor * conf_scale

    if debug:
        strategy = "histogram" if n_pts >= min_pts_for_histogram \
                   else ("median" if n_pts >= 1 else "size-based")
        print(
            f"[FRUSTUM] cls={cls_name} n_pts={n_pts} strategy={strategy} "
            f"depth={depth:.2f} cx={cx:.2f} cz={cz:.2f} ry={ry:.3f} "
            f"yolo={yolo_conf:.3f} factor={point_factor} "
            f"conf_scale={conf_scale} final={final_conf:.4f}"
        )

    x1, y1, x2, y2 = bbox2d
    return {
        "cls_name":   cls_name,
        "cls_id":     _cls_name_to_id(cls_name),
        "truncated":  0.0,
        "occluded":   0,
        "alpha":      -10.0,
        "bbox":       [float(x1), float(y1), float(x2), float(y2)],
        "dimensions": [float(h), float(w), float(l)],   # KITTI 순서: h, w, l
        "location":   [float(cx), float(cy), float(cz)], # KITTI 순서: x, y, z
        "rotation_y": float(ry),
        "score":      float(final_conf),
    }


# ─────────────────────────────────────────────────────────────
# BEV 중복 제거
# ─────────────────────────────────────────────────────────────

def filter_overlapping_fallbacks(fallback_boxes, pp_preds, debug=False):
    """
    Frustum fallback box가 기존 PP box와 BEV에서 겹치면 제거.

    PP가 해당 객체를 이미 잡았지만 2D 매칭에서 빠진 경우를 방지한다.
    동일 클래스끼리만 비교하며, BEV 중심 거리가 해당 클래스 prior의
    max(l, w) × 0.75 이내이면 중복으로 판정한다.

    Returns:
        filtered: 중복이 아닌 fallback box 리스트
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
