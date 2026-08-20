#!/usr/bin/env python3
"""
디버그용: fusion(PP+YOLO) 결과에서 놓친 GT 객체만 이미지에 표시해서 저장.

매칭 기준: GT와 예측 박스의 BEV 중심거리 < threshold (기본 2m)

사용법:
    python tool/fusion/debug_missed_gt.py \
        --pred_dir   data/kitti/pred \
        --label_dir  /home/a/OpenPCDet_my/data/kitti/training/label_2 \
        --image_dir  /home/a/OpenPCDet_my/data/kitti/training/image_2 \
        --calib_dir  /home/a/OpenPCDet_my/data/kitti/training/calib \
        --split_file tool/eval/val_easy.txt \
        --out_dir    debug_missed \
        --cls Pedestrian
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from io_utils import load_pp_predictions
from pipeline import load_gt_boxes
import kitti_util


COLORS = {
    "missed_gt": (0, 0, 255),    # 빨강 (BGR) — 놓친 GT
    "matched_gt": (0, 200, 0),   # 초록 — 매칭된 GT (참고용, 얇게)
    "pred":       (255, 200, 0), # 시안 — 예측 박스 (참고용, 얇게)
}


def project_3d_to_2d(location, dimensions, ry, P):
    """3D box → 2D bbox [x1,y1,x2,y2]."""
    cx, cy, cz = location
    h, w, l = dimensions
    cos_r, sin_r = np.cos(ry), np.sin(ry)
    R = np.array([[cos_r, 0, sin_r], [0, 1, 0], [-sin_r, 0, cos_r]])

    x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])

    corners = R @ np.vstack([x_corners, y_corners, z_corners])
    corners[0] += cx
    corners[1] += cy
    corners[2] += cz

    if np.any(corners[2] < 0.1):
        return None

    pts_hom = np.vstack([corners, np.ones((1, 8))])  # (4, 8)
    pts_2d = P @ pts_hom  # (3, 8)
    pts_2d[:2] /= pts_2d[2:3]

    x1, y1 = pts_2d[0].min(), pts_2d[1].min()
    x2, y2 = pts_2d[0].max(), pts_2d[1].max()
    return [x1, y1, x2, y2]


def bev_dist(a, b):
    """BEV (X-Z) 중심 거리."""
    ax, _, az = a["location"]
    bx, _, bz = b["location"]
    return np.sqrt((ax - bx) ** 2 + (az - bz) ** 2)


def find_missed_gt(gt_boxes, preds, cls_filter, dist_thr=2.0):
    """GT 중 예측과 매칭되지 않은 것만 반환."""
    missed = []
    matched = []
    for gt in gt_boxes:
        if cls_filter and gt["cls_name"] not in cls_filter:
            continue
        is_matched = False
        for pred in preds:
            if pred["cls_name"] != gt["cls_name"]:
                continue
            if bev_dist(gt, pred) < dist_thr:
                is_matched = True
                break
        if is_matched:
            matched.append(gt)
        else:
            missed.append(gt)
    return missed, matched


def draw_box_2d(img, bbox2d, color, thickness, label=""):
    x1, y1, x2, y2 = [int(v) for v in bbox2d]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3), font, 0.45, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Missed GT visualizer")
    parser.add_argument("--pred_dir", type=str, default="data/kitti/pred")
    parser.add_argument("--label_dir", type=str,
                        default="/home/a/OpenPCDet_my/data/kitti/training/label_2")
    parser.add_argument("--image_dir", type=str,
                        default="/home/a/OpenPCDet_my/data/kitti/training/image_2")
    parser.add_argument("--calib_dir", type=str,
                        default="/home/a/OpenPCDet_my/data/kitti/training/calib")
    parser.add_argument("--split_file", type=str, default="/home/a/CUDA-PointPillars/tool/eval/val_occ0.txt")
    parser.add_argument("--out_dir", type=str, default="debug_missed")
    parser.add_argument("--cls", type=str, nargs="+", default=["Pedestrian"],
                        help="표시할 클래스 (예: Pedestrian Car)")
    parser.add_argument("--dist_thr", type=float, default=2.0,
                        help="BEV 매칭 거리 임계값 (m)")
    parser.add_argument("--show_matched", action="store_true",
                        help="매칭된 GT도 얇은 초록선으로 표시")
    parser.add_argument("--show_pred", action="store_true",
                        help="예측 박스도 얇은 시안선으로 표시")
    args = parser.parse_args()

    pred_dir  = Path(args.pred_dir)
    label_dir = Path(args.label_dir)
    image_dir = Path(args.image_dir)
    calib_dir = Path(args.calib_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cls_filter = set(args.cls)

    frame_ids = [l.strip() for l in open(args.split_file) if l.strip()]
    print(f"[INFO] {len(frame_ids)} frames, cls={cls_filter}, dist_thr={args.dist_thr}m")

    total_missed = 0
    total_gt = 0
    saved = 0

    for frame_id in frame_ids:
        gt_boxes = load_gt_boxes(label_dir / f"{frame_id}.txt")
        preds    = load_pp_predictions(pred_dir / f"{frame_id}.txt")

        missed, matched = find_missed_gt(gt_boxes, preds, cls_filter, args.dist_thr)
        total_gt += len(missed) + len(matched)
        total_missed += len(missed)

        if not missed:
            continue

        img_path = image_dir / f"{frame_id}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        calib = kitti_util.Calibration(str(calib_dir / f"{frame_id}.txt"))

        # 참고용: 예측 박스 (얇은 시안)
        if args.show_pred:
            for pred in preds:
                if pred["cls_name"] not in cls_filter:
                    continue
                bbox2d = project_3d_to_2d(pred["location"], pred["dimensions"],
                                          pred["rotation_y"], calib.P)
                if bbox2d:
                    draw_box_2d(img, bbox2d, COLORS["pred"], 1)

        # 참고용: 매칭된 GT (얇은 초록)
        if args.show_matched:
            for gt in matched:
                bbox2d = project_3d_to_2d(gt["location"], gt["dimensions"],
                                          gt["rotation_y"], calib.P)
                if bbox2d:
                    draw_box_2d(img, bbox2d, COLORS["matched_gt"], 1, "GT:ok")

        # 놓친 GT (굵은 빨강)
        for gt in missed:
            bbox2d = project_3d_to_2d(gt["location"], gt["dimensions"],
                                      gt["rotation_y"], calib.P)
            if bbox2d:
                loc = gt["location"]
                label = f"MISS:{gt['cls_name'][:3]} z={loc[2]:.1f}m"
                draw_box_2d(img, bbox2d, COLORS["missed_gt"], 2, label)

        # 프레임 정보
        info = f"{frame_id}  missed:{len(missed)}  matched:{len(matched)}"
        cv2.putText(img, info, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(str(out_dir / f"{frame_id}.png"), img)
        saved += 1

    print(f"[DONE] missed {total_missed}/{total_gt} GT objects, saved {saved} images → {out_dir}/")


if __name__ == "__main__":
    main()
