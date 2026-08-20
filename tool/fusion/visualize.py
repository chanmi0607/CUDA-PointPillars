# tool/fusion/visualize.py
"""
Fusion 디버그 시각화 모듈.

생성물: 좌측 이미지뷰 + 우측 BEV뷰를 나란히 합성한 PNG.

색상 규칙 (BGR):
  PP-only     : 주황 — PointPillars만 잡은 박스
  PP matched  : 시안 — PP + YOLO 매칭 성공
  Fallback    : 빨강 — YOLO-only → frustum fallback으로 생성
  YOLO        : 초록 — YOLO 2D 검출 (얇은 선, 배경 참조용)
"""

import cv2
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 색상 팔레트 (BGR)
# ─────────────────────────────────────────────────────────────
COLORS = {
    "pp":             (255, 150, 50),    # 주황-파랑
    "pp_matched":     (255, 255, 0),     # 시안
    "fallback":       (0, 50, 255),      # 빨강
    "yolo":           (50, 220, 50),     # 초록
    "yolo_unmatched": (0, 150, 255),     # 주황 — PP와 매칭 안 된 YOLO 검출
    "grid":           (60, 60, 60),      # 진회색
    "ego":            (255, 255, 255),   # 흰색
    "text_bg":        (0, 0, 0),
    "point_cloud":    (200, 200, 200),   # 포인트 클라우드 (밝은 회색)
    "gt":             (100, 100, 100),   # GT 박스 색상
    "cluster_cand":   (50, 220, 50),     # 클러스터 후보 (초록)
    "cluster_sel":    (0, 0, 255),       # 클러스터 선택 (빨강)
}

# ─────────────────────────────────────────────────────────────
# 표시할 클래스 필터 (None이면 전부 표시)
# ─────────────────────────────────────────────────────────────
SHOW_CLASSES = {"Pedestrian"}
# SHOW_CLASSES = {"Car"}
# SHOW_CLASSES = {"Cyclist"}
# SHOW_CLASSES = {"Car", "Pedestrian", "Cyclist"}
# SHOW_CLASSES = None   # 전부 표시


def _class_visible(cls_name):
    """표시 대상 클래스인지 확인."""
    if SHOW_CLASSES is None:
        return True
    return cls_name in SHOW_CLASSES


# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────

def _put_label(img, text, pos, font_scale=0.45, color=(255, 255, 255),
               bg_color=(0, 0, 0), thickness=1):
    """글자 뒤에 반투명 배경 사각형을 깔아 가독성 확보."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(pos[0]), int(pos[1])
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 4), bg_color, -1)
    cv2.putText(img, text, (x + 2, y), font, font_scale, color,
                thickness, cv2.LINE_AA)


def _project_lidar_to_img(points, calib):
    """LiDAR 포인트를 이미지 평면에 투영. (N,3) 픽셀좌표 + depth 반환."""
    pts_hom = np.hstack([points[:, :3], np.ones((len(points), 1))])
    pts_rect = pts_hom @ (calib.V2C.T @ calib.R0.T)
    pts_rect_hom = np.hstack([pts_rect, np.ones((len(pts_rect), 1))])
    pts_2d_hom = pts_rect_hom @ calib.P.T
    depth = pts_2d_hom[:, 2]
    mask = depth > 0
    pts_img = pts_2d_hom[mask, :2] / depth[mask, np.newaxis]
    return pts_img, depth[mask]


# ─────────────────────────────────────────────────────────────
# 이미지 뷰 (2D bbox overlay + point cloud projection)
# ─────────────────────────────────────────────────────────────

def draw_image_view(img, yolo_preds, fused_preds, matched_yolo_indices,
                    points=None, calib=None):
    """
    카메라 이미지 위에 포인트 클라우드 프로젝션 + 2D bbox를 오버레이.

    Args:
        img:                   원본 이미지 (BGR)
        yolo_preds:            YOLO 검출 리스트
        fused_preds:           fused 예측 리스트 (각각 "source" 필드 보유)
        matched_yolo_indices:  PP와 매칭된 YOLO 인덱스 set
        points:                LiDAR 포인트 (N,4) — None이면 프로젝션 생략
        calib:                 kitti_util.Calibration 객체
    """
    vis = img.copy()

    # 포인트 클라우드 프로젝션
    if points is not None and calib is not None:
        pts_img, depth = _project_lidar_to_img(points, calib)
        h, w = vis.shape[:2]
        # 이미지 범위 내 필터
        valid = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < w) & \
                (pts_img[:, 1] >= 0) & (pts_img[:, 1] < h)
        pts_img = pts_img[valid]
        depth = depth[valid]
        # depth 기반 색상 (가까울수록 밝은 노랑, 멀수록 어두운 파랑)
        if len(depth) > 0:
            d_min, d_max = depth.min(), max(depth.max(), depth.min() + 1)
            d_norm = (depth - d_min) / (d_max - d_min)  # 0~1
            for i in range(len(pts_img)):
                px, py = int(pts_img[i, 0]), int(pts_img[i, 1])
                t = d_norm[i]
                # 가까우면 (0,255,255) 노랑, 멀면 (255,50,0) 파랑 (BGR)
                b = int(255 * t)
                g = int(255 * (1 - t))
                r = int(255 * (1 - t))
                cv2.circle(vis, (px, py), 1, (b, g, r), -1)

    # YOLO 박스 (배경 — 얇은 선)
    for yi, yolo in enumerate(yolo_preds):
        if not _class_visible(yolo.get("cls_name", "")):
            continue
        x1, y1, x2, y2 = [int(v) for v in yolo["bbox"]]
        matched = yi in matched_yolo_indices
        if not matched:
            continue
        color = COLORS["yolo"] if matched else COLORS["yolo_unmatched"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

    # Fallback 예측만 그리기 (PP 박스는 생략)
    for pred in fused_preds:
        source = pred.get("source", "pp")
        if source != "fallback":
            continue
        if not _class_visible(pred.get("cls_name", "")):
            continue
        bbox = pred["bbox"]
        if bbox == [0.0, 0.0, 0.0, 0.0]:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]

        cls   = pred["cls_name"]
        score = pred["score"]
        color = COLORS["fallback"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        _put_label(vis, f"FB:{cls[:3]} {score:.2f}",
                   (x1, y1 - 2), color=color)

    return vis


# ─────────────────────────────────────────────────────────────
# BEV IoU 계산
# ─────────────────────────────────────────────────────────────

def _bev_corners(cx, cz, w, l, ry):
    """BEV 상의 oriented 박스 4개 꼭짓점 반환 (X-Z 평면, float32).

    kitti_util.compute_box_3d 와 동일 컨벤션: X ← l, Z ← w
    """
    cos_r, sin_r = np.cos(ry), np.sin(ry)
    corners_local = np.array([
        [ l / 2,  w / 2],
        [ l / 2, -w / 2],
        [-l / 2, -w / 2],
        [-l / 2,  w / 2],
    ], dtype=np.float32)
    R = np.array([[cos_r, sin_r], [-sin_r, cos_r]], dtype=np.float32)
    return (R @ corners_local.T).T + np.array([cx, cz], dtype=np.float32)


def _bev_iou(cx1, cz1, w1, l1, ry1, cx2, cz2, w2, l2, ry2):
    """두 oriented BEV 박스 간 IoU."""
    c1 = _bev_corners(cx1, cz1, w1, l1, ry1)
    c2 = _bev_corners(cx2, cz2, w2, l2, ry2)
    ret, inter_pts = cv2.intersectConvexConvex(c1, c2)
    inter_area = cv2.contourArea(inter_pts) if ret > 0 and len(inter_pts) >= 3 else 0.0
    area1 = w1 * l1
    area2 = w2 * l2
    union = area1 + area2 - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def best_bev_iou(pred, gt_boxes):
    """
    pred 박스와 같은 클래스 GT 박스들 사이의 최대 BEV IoU 반환.

    Args:
        pred:     fused_pred 딕셔너리 (location, dimensions, rotation_y, cls_name)
        gt_boxes: GT 박스 딕셔너리 리스트 (같은 필드 구조)
    Returns:
        best_iou (float)
    """
    if not gt_boxes:
        return 0.0
    cx, _, cz = pred["location"]
    _, w, l = pred["dimensions"]
    ry  = pred["rotation_y"]
    cls = pred["cls_name"].lower()

    best = 0.0
    for gt in gt_boxes:
        if gt["cls_name"].lower() != cls:
            continue
        gx, _, gz = gt["location"]
        _, gw, gl = gt["dimensions"]
        gr = gt["rotation_y"]
        iou = _bev_iou(cx, cz, w, l, ry, gx, gz, gw, gl, gr)
        if iou > best:
            best = iou
    return best


# ─────────────────────────────────────────────────────────────
# BEV 뷰 (Bird's-Eye View)
# ─────────────────────────────────────────────────────────────

def _world_to_px(x, z, ppm, origin):
    """카메라 좌표 (X right, Z forward) → BEV 픽셀 (북쪽=위: Z→위, X→오른쪽)."""
    px = int(origin[0] + x * ppm)
    py = int(origin[1] - z * ppm)
    return px, py


def _draw_bev_box(canvas, cx, cz, w, l, ry, color, ppm, origin, thickness=1):
    """Oriented BEV 직사각형 + 방향 화살표.

    kitti_util.compute_box_3d 와 동일한 컨벤션:
      X방향(로컬) = l (length), Z방향(로컬) = w (width)
    roty 행렬: [[cos, sin], [-sin, cos]]
    """
    cos_r, sin_r = np.cos(ry), np.sin(ry)
    # kitti_util.compute_box_3d 기준: X ← l, Z ← w
    corners_local = np.array([
        [ l / 2,  w / 2],   # front-right
        [ l / 2, -w / 2],   # front-left
        [-l / 2, -w / 2],   # rear-left
        [-l / 2,  w / 2],   # rear-right
    ])
    R = np.array([[cos_r, sin_r],
                  [-sin_r, cos_r]])
    corners_world = (R @ corners_local.T).T + np.array([cx, cz])

    pts_px = np.array([_world_to_px(cw[0], cw[1], ppm, origin)
                       for cw in corners_world], dtype=np.int32)
    cv2.polylines(canvas, [pts_px], isClosed=True, color=color,
                  thickness=thickness)

    # 전면 방향 화살표 (front edge 중점 → center)
    front_mid = ((pts_px[0] + pts_px[1]) / 2).astype(int)
    center_px = np.array(_world_to_px(cx, cz, ppm, origin), dtype=int)
    cv2.arrowedLine(canvas, tuple(center_px), tuple(front_mid),
                    color, 1, tipLength=0.3)


def _draw_bev_points(canvas, points, calib, ppm, origin, bev_h, bev_w):
    """BEV 캔버스에 LiDAR 포인트를 camera rect 좌표로 변환 후 그리기."""
    pts_hom = np.hstack([points[:, :3], np.ones((len(points), 1))])
    pts_rect = pts_hom @ (calib.V2C.T @ calib.R0.T)  # (N, 3) — camera rect

    x_rect = pts_rect[:, 0]
    z_rect = pts_rect[:, 2]

    for i in range(len(x_rect)):
        px, py = _world_to_px(x_rect[i], z_rect[i], ppm, origin)
        if 0 <= px < bev_w and 0 <= py < bev_h:
            canvas[py, px] = COLORS["point_cloud"]


def _draw_bev_clusters(canvas, fused_preds, ppm, origin, bev_h, bev_w):
    """BEV에 DBSCAN 클러스터 포인트를 그리기.

    초록: 후보 클러스터 (미선택), 빨강: 선택된 클러스터, noise(-1)는 생략.
    """
    for pred in fused_preds:
        if pred.get("source") != "fallback":
            continue
        pts_cam = pred.get("_cluster_pts_cam")
        labels  = pred.get("_cluster_labels")
        sel     = pred.get("_cluster_selected")
        if pts_cam is None or labels is None:
            continue

        x_rect = pts_cam[:, 0]
        z_rect = pts_cam[:, 2]

        for i in range(len(labels)):
            lbl = int(labels[i])
            if lbl == -1:          # noise 스킵
                continue
            color = COLORS["cluster_sel"] if lbl == sel else COLORS["cluster_cand"]
            px, py = _world_to_px(x_rect[i], z_rect[i], ppm, origin)
            if 0 <= px < bev_w and 0 <= py < bev_h:
                cv2.circle(canvas, (px, py), 2, color, -1)


def draw_bev_view(fused_preds, bev_h=500, x_range=(-20, 20), z_range=(0, 40),
                  gt_boxes=None, points=None, calib=None):
    """
    BEV 캔버스에 LiDAR 포인트 + oriented 3D box를 그린다.

    Args:
        fused_preds: fused 예측 리스트 (source 태깅된)
        bev_h:       캔버스 세로 크기 (px)
        x_range:     BEV X축 범위 (m, 좌우)
        z_range:     BEV Z축 범위 (m, 전방)
        gt_boxes:    GT 박스 리스트 (있으면 회색 외곽선 + 예측 박스에 IoU 표시)
        points:      LiDAR 포인트 (N,4) — None이면 포인트 생략
        calib:       kitti_util.Calibration 객체

    Returns:
        canvas (H, W, 3) BGR 이미지
    """
    x_span = x_range[1] - x_range[0]
    z_span = z_range[1] - z_range[0]
    ppm    = bev_h / z_span                   # 높이 = Z축 범위 (전방)
    bev_w  = int(x_span * ppm)               # 너비 = X축 범위 (좌우)

    canvas = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    # ego = 하단 중앙 (z=0 → py=bev_h, x=0 → px=중앙)
    origin = (int(-x_range[0] * ppm), bev_h)

    # 포인트 클라우드 (박스/그리드보다 먼저 그려서 배경 역할)
    if points is not None and calib is not None:
        _draw_bev_points(canvas, points, calib, ppm, origin, bev_h, bev_w)

    # 그리드 (10m 간격) — Z방향: 가로선, X방향: 세로선
    for d in range(0, int(z_span) + 1, 10):
        _, py = _world_to_px(0, d, ppm, origin)
        if 0 <= py < bev_h:
            cv2.line(canvas, (0, py), (bev_w, py), COLORS["grid"], 1)
            _put_label(canvas, f"{d}m", (2, py - 3),
                       font_scale=0.35, color=(150, 150, 150))
    for xi in range(int(x_range[0]), int(x_range[1]) + 1, 10):
        px, _ = _world_to_px(xi, 0, ppm, origin)
        if 0 <= px < bev_w:
            cv2.line(canvas, (px, 0), (px, bev_h), COLORS["grid"], 1)

    # Ego 마커 (하단 중앙, 위쪽 방향 삼각형)
    ex, ey = origin
    cv2.drawMarker(canvas, (ex, min(ey - 5, bev_h - 5)),
                   COLORS["ego"], cv2.MARKER_TRIANGLE_UP, 14, 2)

    # DBSCAN 클러스터 포인트 (초록: 후보, 빨강: 선택)
    _draw_bev_clusters(canvas, fused_preds, ppm, origin, bev_h, bev_w)

    # GT 박스 (회색 얇은 선)
    if gt_boxes:
        for gt in gt_boxes:
            if not _class_visible(gt["cls_name"]):
                continue
            gx, _, gz = gt["location"]
            _, gw, gl = gt["dimensions"]
            gr = gt["rotation_y"]
            _draw_bev_box(canvas, gx, gz, gw, gl, gr,
                          COLORS["gt"], ppm, origin, thickness=1)
            lx, ly = _world_to_px(gx, gz, ppm, origin)
            _put_label(canvas, f"GT:{gt['cls_name'][:3]}",
                       (lx + 3, ly + 8), font_scale=0.28, color=COLORS["gt"])

    # Fallback 예측 박스만 그리기 (PP 박스는 생략)
    for pred in fused_preds:
        source = pred.get("source", "pp")
        if source != "fallback":
            continue
        if not _class_visible(pred.get("cls_name", "")):
            continue
        cx, _, cz = pred["location"]
        h_dim, w_dim, l_dim = pred["dimensions"]
        ry = pred["rotation_y"]
        score = pred["score"]

        color = COLORS["fallback"]

        _draw_bev_box(canvas, cx, cz, w_dim, l_dim, ry,
                      color, ppm, origin, 1)

        # BEV IoU (GT 있을 때만)
        iou_str = ""
        if gt_boxes:
            iou = best_bev_iou(pred, gt_boxes)
            iou_str = f" IoU:{iou:.2f}"

        lbl = f"FB:{pred['cls_name'][:3]} {score:.2f}{iou_str}"
        lx, ly = _world_to_px(cx, cz, ppm, origin)
        _put_label(canvas, lbl, (lx + 3, ly - 3),
                   font_scale=0.3, color=color)

    return canvas


# ─────────────────────────────────────────────────────────────
# 합성 + 저장
# ─────────────────────────────────────────────────────────────

def save_fusion_vis(
    img_path,
    yolo_preds,
    fused_preds,
    matched_yolo_indices,
    save_path,
    frame_id="",
    bev_x_range=(-20, 20),
    bev_z_range=(0, 25),
    gt_boxes=None,
    points=None,
    calib=None,
):
    """
    이미지뷰 + BEV뷰를 좌우로 합성해 PNG로 저장.

    Args:
        img_path:              카메라 이미지 경로
        yolo_preds:            YOLO 검출 전체
        fused_preds:           fused 예측 (source 태깅된)
        matched_yolo_indices:  매칭된 YOLO 인덱스 set
        save_path:             저장 경로
        frame_id:              프레임 식별자 (오버레이 텍스트)
        bev_x_range:           BEV X축 범위 (m)
        bev_z_range:           BEV Z축 범위 (m)
        gt_boxes:              GT 박스 리스트 (있으면 BEV에 회색 박스 + IoU 표시)
        points:                LiDAR 포인트 (N,4) — None이면 포인트 프로젝션 생략
        calib:                 kitti_util.Calibration 객체
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[VIS] Cannot read image: {img_path}")
        return

    h_img, w_img = img.shape[:2]

    # ── 이미지 뷰 ──
    img_vis = draw_image_view(img, yolo_preds, fused_preds, matched_yolo_indices,
                              points=points, calib=calib)

    # ── BEV 뷰 (이미지 높이에 맞춤) ──
    bev_h = max(h_img, 400)
    bev_vis = draw_bev_view(fused_preds, bev_h=bev_h,
                            x_range=bev_x_range, z_range=bev_z_range,
                            gt_boxes=gt_boxes, points=points, calib=calib)
    # BEV 높이를 이미지 높이에 맞춤
    if bev_vis.shape[0] != h_img:
        scale = h_img / bev_vis.shape[0]
        bev_vis = cv2.resize(bev_vis,
                             (int(bev_vis.shape[1] * scale), h_img),
                             interpolation=cv2.INTER_AREA)

    # ── 합성 ──
    composite = np.hstack([img_vis, bev_vis])

    # ── 정보 텍스트 ──
    n_pp      = sum(1 for p in fused_preds
                    if p.get("source") in ("pp", "pp_matched"))
    n_matched = sum(1 for p in fused_preds if p.get("source") == "pp_matched")
    n_fb      = sum(1 for p in fused_preds if p.get("source") == "fallback")
    info = (f"Frame:{frame_id}  PP:{n_pp}(matched:{n_matched})  "
            f"YOLO:{len(yolo_preds)}  Fallback:{n_fb}")
    _put_label(composite, info, (5, 15), font_scale=0.5, color=(255, 255, 255))

    # 범례
    y_leg = 35
    for label, key in [("Fallback", "fallback"), ("YOLO", "yolo"),
                       ("GT", "gt"),
                       ("Cluster:cand", "cluster_cand"),
                       ("Cluster:sel", "cluster_sel")]:
        _put_label(composite, label, (5, y_leg),
                   font_scale=0.4, color=COLORS[key])
        y_leg += 18

    # ── 저장 ──
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), composite)
