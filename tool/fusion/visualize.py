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
    "yolo_unmatched": (0, 150, 255),     # 주황-빨강
    "grid":           (60, 60, 60),      # 진회색
    "ego":            (255, 255, 255),   # 흰색
    "text_bg":        (0, 0, 0),
}

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


# ─────────────────────────────────────────────────────────────
# 이미지 뷰 (2D bbox overlay)
# ─────────────────────────────────────────────────────────────

def draw_image_view(img, yolo_preds, fused_preds, matched_yolo_indices):
    """
    카메라 이미지 위에 2D bbox를 색상별로 오버레이.

    Args:
        img:                   원본 이미지 (BGR)
        yolo_preds:            YOLO 검출 리스트
        fused_preds:           fused 예측 리스트 (각각 "source" 필드 보유)
        matched_yolo_indices:  PP와 매칭된 YOLO 인덱스 set
    """
    vis = img.copy()

    # YOLO 박스 (배경 — 얇은 선)
    for yi, yolo in enumerate(yolo_preds):
        x1, y1, x2, y2 = [int(v) for v in yolo["bbox"]]
        matched = yi in matched_yolo_indices
        color = COLORS["yolo"] if matched else COLORS["yolo_unmatched"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

    # Fused 예측 (전경 — 두꺼운 선 + 라벨)
    for pred in fused_preds:
        source = pred.get("source", "pp")
        bbox   = pred["bbox"]
        # bbox가 전부 0이면 2D 투영이 없는 것 — skip
        if bbox == [0.0, 0.0, 0.0, 0.0]:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]

        cls   = pred["cls_name"]
        score = pred["score"]

        if source == "fallback":
            color = COLORS["fallback"]
            thick = 3
            tag   = "FB"
        elif source == "pp_matched":
            color = COLORS["pp_matched"]
            thick = 2
            tag   = "M"
        else:
            color = COLORS["pp"]
            thick = 2
            tag   = "PP"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        _put_label(vis, f"{tag}:{cls[:3]} {score:.2f}",
                   (x1, y1 - 2), color=color)

    return vis


# ─────────────────────────────────────────────────────────────
# BEV 뷰 (Bird's-Eye View)
# ─────────────────────────────────────────────────────────────

def _world_to_px(x, z, ppm, origin):
    """카메라 좌표 (X right, Z forward) → BEV 픽셀."""
    px = int(origin[0] + x * ppm)
    py = int(origin[1] - z * ppm)
    return px, py


def _draw_bev_box(canvas, cx, cz, w, l, ry, color, ppm, origin, thickness=2):
    """Oriented BEV 직사각형 + 방향 화살표."""
    cos_r, sin_r = np.cos(ry), np.sin(ry)
    # 로컬 꼭짓점 (w: X방향 폭, l: Z방향 길이)
    corners_local = np.array([
        [ w / 2,  l / 2],
        [-w / 2,  l / 2],
        [-w / 2, -l / 2],
        [ w / 2, -l / 2],
    ])
    # rotation_y 회전 (Y축 기준, KITTI 컨벤션)
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


def draw_bev_view(fused_preds, bev_h=500, x_range=(-20, 20), z_range=(0, 40)):
    """
    BEV 캔버스에 oriented 3D box를 그린다.

    Args:
        fused_preds: fused 예측 리스트 (source 태깅된)
        bev_h:       캔버스 세로 크기 (px)
        x_range:     BEV X축 범위 (m, 좌우)
        z_range:     BEV Z축 범위 (m, 전방)

    Returns:
        canvas (H, W, 3) BGR 이미지
    """
    x_span = x_range[1] - x_range[0]
    z_span = z_range[1] - z_range[0]
    ppm    = bev_h / z_span                   # pixels per meter
    bev_w  = int(x_span * ppm)

    canvas = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    origin = (bev_w // 2, bev_h)              # ego = 하단 중앙

    # 그리드 (10m 간격)
    for d in range(0, int(z_span) + 1, 10):
        _, py = _world_to_px(0, d, ppm, origin)
        if 0 <= py < bev_h:
            cv2.line(canvas, (0, py), (bev_w, py), COLORS["grid"], 1)
            _put_label(canvas, f"{d}m", (5, py),
                       font_scale=0.35, color=(150, 150, 150))
    for xi in range(int(x_range[0]), int(x_range[1]) + 1, 10):
        px, _ = _world_to_px(xi, 0, ppm, origin)
        if 0 <= px < bev_w:
            cv2.line(canvas, (px, 0), (px, bev_h), COLORS["grid"], 1)

    # Ego 마커
    ex, ey = origin
    cv2.drawMarker(canvas, (ex, min(ey - 5, bev_h - 5)),
                   COLORS["ego"], cv2.MARKER_TRIANGLE_UP, 14, 2)

    # 박스 그리기
    for pred in fused_preds:
        source = pred.get("source", "pp")
        cx, _, cz = pred["location"]
        h_dim, w_dim, l_dim = pred["dimensions"]
        ry = pred["rotation_y"]
        score = pred["score"]

        color = COLORS.get(source, COLORS["pp"])
        thick = 3 if source == "fallback" else 2

        _draw_bev_box(canvas, cx, cz, w_dim, l_dim, ry,
                      color, ppm, origin, thick)

        # 라벨
        tag = {"fallback": "FB", "pp_matched": "M"}.get(source, "PP")
        lbl = f"{tag}:{pred['cls_name'][:3]} {score:.2f}"
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
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[VIS] Cannot read image: {img_path}")
        return

    h_img, w_img = img.shape[:2]

    # ── 이미지 뷰 ──
    img_vis = draw_image_view(img, yolo_preds, fused_preds, matched_yolo_indices)

    # ── BEV 뷰 (이미지 높이에 맞춤) ──
    bev_h = max(h_img, 400)
    bev_vis = draw_bev_view(fused_preds, bev_h=bev_h,
                            x_range=bev_x_range, z_range=bev_z_range)
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
    for label, key in [("PP-only", "pp"), ("Matched", "pp_matched"),
                       ("Fallback", "fallback"), ("YOLO", "yolo")]:
        _put_label(composite, label, (5, y_leg),
                   font_scale=0.4, color=COLORS[key])
        y_leg += 18

    # ── 저장 ──
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), composite)
