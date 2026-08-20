#!/usr/bin/env python3
"""
Custom BEV Evaluation: PP+YOLO fusion 결과를 Rotated BEV IoU로 평가.

pipeline.py의 fusion 결과(data/kitti/pred_fused/ 등)를 읽어서
pipeline_eval_lgbm.py와 동일한 평가 방식 적용:
  - Rotated BEV IoU (Sutherland-Hodgman polygon clipping)
  - Greedy confidence-ordered 매칭
  - 11-point interpolated AP (KITTI style)
  - 여러 IoU threshold

사용법:
    python tool/fusion/pipeline_custom_eval.py \
        --pred_dir   data/kitti/pred_fused \
        --split_file tool/eval/val_easy.txt \
        --max_dist   30
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from io_utils import load_pp_predictions, load_frame_ids
import kitti_util


# ═════════════════════════════════════════════════════════════════════════
# 설정
# ═════════════════════════════════════════════════════════════════════════
# TARGET_CLASSES = {"car", "pedestrian", "cyclist"}
TARGET_CLASSES = {"car", "pedestrian"}
IOU_THRESHOLDS = [0.5, 0.3, 0.25]
DIST_RANGES = [(0, 10), (10, 20), (20, 30)]


# ═════════════════════════════════════════════════════════════════════════
# GT 로드
# ═════════════════════════════════════════════════════════════════════════
KITTI_CLS_NAMES = {"Car", "Pedestrian", "Cyclist", "Van", "Person_sitting",
                   "Truck", "Tram", "Misc"}


def load_gt_labels(label_path):
    """KITTI label → list of dicts with cls, h, w, l, x, y, z, ry."""
    objects = []
    if not Path(label_path).exists():
        return objects
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls not in KITTI_CLS_NAMES:
                continue
            if cls.lower() not in TARGET_CLASSES:
                continue
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            objects.append({"cls": cls.lower(), "h": h, "w": w, "l": l,
                            "x": x, "y": y, "z": z, "ry": ry})
    return objects


# ═════════════════════════════════════════════════════════════════════════
# 좌표 변환: Camera rect → Velodyne BEV 코너
# ═════════════════════════════════════════════════════════════════════════

def _rect_to_velo(pts_rect, calib):
    """Camera rect (N, 3) → velodyne (N, 3).
    R0: (3,3), V2C: (3,4) → C2V = inv(V2C_4x4) @ inv(R0_4x4)
    """
    R0_4 = np.eye(4)
    R0_4[:3, :3] = calib.R0
    Tr_4 = np.eye(4)
    Tr_4[:3, :] = calib.V2C

    # rect → velo: inv(Tr) @ inv(R0) @ [x,y,z,1]^T
    T = np.linalg.inv(Tr_4) @ np.linalg.inv(R0_4)
    ones = np.ones((len(pts_rect), 1))
    pts_h = np.hstack([pts_rect, ones])
    return (T @ pts_h.T).T[:, :3]


def _box_corners_cam(l, w, x, y, z, ry):
    """3D box params → (4, 3) BEV 코너 in camera rect."""
    corners = np.array([
        [ l/2, 0,  w/2], [ l/2, 0, -w/2],
        [-l/2, 0, -w/2], [-l/2, 0,  w/2],
    ])
    c, s = np.cos(ry), np.sin(ry)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    corners = (R @ corners.T).T
    corners[:, 0] += x
    corners[:, 1] += y
    corners[:, 2] += z
    return corners


def gt_box_to_velo_bev(obj, calib):
    """GT 3D box (camera rect) → velodyne BEV 코너 (4, 2)."""
    corners_cam = _box_corners_cam(
        obj["l"], obj["w"], obj["x"], obj["y"], obj["z"], obj["ry"])
    corners_velo = _rect_to_velo(corners_cam, calib)
    return corners_velo[:, :2]


def pred_box_to_velo_bev(pred, calib):
    """Fusion prediction dict (camera rect) → velodyne BEV 코너 (4, 2)."""
    h, w, l = pred["dimensions"]
    cx, cy, cz = pred["location"]
    ry = pred["rotation_y"]
    corners_cam = _box_corners_cam(l, w, cx, cy, cz, ry)
    corners_velo = _rect_to_velo(corners_cam, calib)
    return corners_velo[:, :2]


# ═════════════════════════════════════════════════════════════════════════
# Rotated BEV IoU (Sutherland-Hodgman)
# ═════════════════════════════════════════════════════════════════════════

def _cross2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _line_intersect(a, b, c, d):
    denom = (a[0]-b[0])*(c[1]-d[1]) - (a[1]-b[1])*(c[0]-d[0])
    if abs(denom) < 1e-12:
        return a
    t = ((a[0]-c[0])*(c[1]-d[1]) - (a[1]-c[1])*(c[0]-d[0])) / denom
    return (a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]))


def _clip_polygon_by_edge(polygon, p1, p2):
    if not polygon:
        return []
    output = []
    for i in range(len(polygon)):
        curr = polygon[i]
        prev = polygon[i - 1]
        c_in = _cross2d(p1, p2, curr) >= 0
        p_in = _cross2d(p1, p2, prev) >= 0
        if c_in:
            if not p_in:
                output.append(_line_intersect(prev, curr, p1, p2))
            output.append(curr)
        elif p_in:
            output.append(_line_intersect(prev, curr, p1, p2))
    return output


def _polygon_area(vertices):
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def _ensure_ccw(poly):
    n = len(poly)
    area2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        area2 += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    if area2 < 0:
        poly = poly[::-1]
    return poly


def rotated_bev_iou(corners_a, corners_b):
    """두 회전 사각형의 BEV IoU (Sutherland-Hodgman clipping)."""
    poly_a = [(corners_a[i, 0], corners_a[i, 1]) for i in range(4)]
    poly_b = [(corners_b[i, 0], corners_b[i, 1]) for i in range(4)]
    poly_a = _ensure_ccw(poly_a)
    poly_b = _ensure_ccw(poly_b)

    clipped = list(poly_b)
    for i in range(len(poly_a)):
        if not clipped:
            break
        clipped = _clip_polygon_by_edge(
            clipped, poly_a[i], poly_a[(i + 1) % len(poly_a)])

    if len(clipped) < 3:
        return 0.0

    inter_area = _polygon_area(clipped)
    area_a = _polygon_area(poly_a)
    area_b = _polygon_area(poly_b)
    union_area = area_a + area_b - inter_area

    if union_area < 1e-8:
        return 0.0
    return inter_area / union_area


# ═════════════════════════════════════════════════════════════════════════
# AP 계산 (KITTI 11-point interpolation)
# ═════════════════════════════════════════════════════════════════════════

def compute_ap(scores, tp_flags, n_gt):
    if n_gt == 0:
        return 0.0
    sorted_idx = np.argsort(-np.array(scores))
    tp_sorted = np.array(tp_flags)[sorted_idx]

    tp_cumsum = np.cumsum(tp_sorted)
    fp_cumsum = np.cumsum(1 - tp_sorted)

    recall = tp_cumsum / n_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_recall = precision[recall >= t]
        if len(prec_at_recall) > 0:
            ap += np.max(prec_at_recall)
    ap /= 11.0
    return ap


# ═════════════════════════════════════════════════════════════════════════
# 메인 평가
# ═════════════════════════════════════════════════════════════════════════

def run_eval(pred_dir, label_dir, calib_dir, split_file, max_dist=30.0,
             iou_thresholds=None, score_thr=0.5):
    if iou_thresholds is None:
        iou_thresholds = IOU_THRESHOLDS

    pred_dir  = Path(pred_dir)
    label_dir = Path(label_dir)
    calib_dir = Path(calib_dir)

    frame_ids = load_frame_ids(split_file)
    print(f"[EVAL] {len(frame_ids)} frames, dist≤{max_dist}m, "
          f"score≥{score_thr}, IoU thresholds={iou_thresholds}")
    print(f"  pred_dir:  {pred_dir}")
    print(f"  label_dir: {label_dir}")
    print("=" * 75)

    # 누적 통계
    stats_by_thr = {}
    for thr in iou_thresholds:
        stats_by_thr[thr] = {
            "all_scores": defaultdict(list),
            "all_tp": defaultdict(list),
        }
    n_gt_total = defaultdict(int)

    # 거리별 누적 통계
    dist_stats = {}
    for d_lo, d_hi in DIST_RANGES:
        dist_stats[(d_lo, d_hi)] = {"n_gt": defaultdict(int)}
        for thr in iou_thresholds:
            dist_stats[(d_lo, d_hi)][(thr, "scores")] = defaultdict(list)
            dist_stats[(d_lo, d_hi)][(thr, "tp")] = defaultdict(list)

    for idx, frame_id in enumerate(frame_ids):
        # ── 예측 로드 ──
        preds = load_pp_predictions(pred_dir / f"{frame_id}.txt")

        # ── GT 로드 ──
        gt_objects = load_gt_labels(label_dir / f"{frame_id}.txt")

        # ── Calib 로드 ──
        calib_path = calib_dir / f"{frame_id}.txt"
        if not calib_path.exists():
            continue
        calib = kitti_util.Calibration(str(calib_path))

        # ── GT → velodyne BEV 코너 (거리 필터) ──
        gt_bev = []  # [(cls, corners, dist), ...]
        for obj in gt_objects:
            corners = gt_box_to_velo_bev(obj, calib)
            center = corners.mean(axis=0)
            gt_dist = float(np.linalg.norm(center))
            if gt_dist > max_dist:
                continue
            gt_bev.append((obj["cls"], corners, gt_dist))
            n_gt_total[obj["cls"]] += 1
            for (d_lo, d_hi), ds in dist_stats.items():
                if d_lo <= gt_dist < d_hi:
                    ds["n_gt"][obj["cls"]] += 1

        # ── Pred → velodyne BEV 코너 (score + 거리 필터) ──
        pred_bev = []
        for pred in preds:
            if pred["score"] < score_thr:
                continue
            if pred["cls_name"].lower() not in TARGET_CLASSES:
                continue
            corners = pred_box_to_velo_bev(pred, calib)
            center = corners.mean(axis=0)
            if np.linalg.norm(center) > max_dist:
                continue
            pred_bev.append((pred["cls_name"].lower(), pred["score"], corners))

        # ── Greedy 매칭 (confidence 내림차순) ──
        preds_sorted = sorted(enumerate(pred_bev), key=lambda p: -p[1][1])

        # 각 pred의 best IoU 사전 계산
        pred_best = {}
        for pi, (cls_name, conf, pred_corners) in preds_sorted:
            best_iou = 0.0
            best_gt_idx = -1
            for gi, (gt_cls, gt_corners, gt_d) in enumerate(gt_bev):
                if gt_cls != cls_name:
                    continue
                iou = rotated_bev_iou(pred_corners, gt_corners)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi
            pred_best[pi] = (best_iou, best_gt_idx)

        # 각 IoU 임계값별 greedy 매칭
        for thr in iou_thresholds:
            gt_matched = [False] * len(gt_bev)
            for pi, (cls_name, conf, pred_corners) in preds_sorted:
                best_iou, best_gt_idx = pred_best[pi]

                # 이미 매칭된 GT면 재탐색
                if best_gt_idx >= 0 and gt_matched[best_gt_idx]:
                    best_iou2 = 0.0
                    best_gt_idx2 = -1
                    for gi, (gt_cls, gt_corners, gt_d) in enumerate(gt_bev):
                        if gt_matched[gi] or gt_cls != cls_name:
                            continue
                        iou = rotated_bev_iou(pred_corners, gt_corners)
                        if iou > best_iou2:
                            best_iou2 = iou
                            best_gt_idx2 = gi
                    best_iou, best_gt_idx = best_iou2, best_gt_idx2

                if best_iou >= thr and best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
                    stats_by_thr[thr]["all_scores"][cls_name].append(conf)
                    stats_by_thr[thr]["all_tp"][cls_name].append(1)
                    # 거리별: 매칭된 GT 거리로 분류
                    matched_dist = gt_bev[best_gt_idx][2]
                    for (d_lo, d_hi), ds in dist_stats.items():
                        if d_lo <= matched_dist < d_hi:
                            ds[(thr, "scores")][cls_name].append(conf)
                            ds[(thr, "tp")][cls_name].append(1)
                else:
                    stats_by_thr[thr]["all_scores"][cls_name].append(conf)
                    stats_by_thr[thr]["all_tp"][cls_name].append(0)
                    # 거리별: FP는 pred 중심 거리로 분류
                    pred_dist = float(np.linalg.norm(pred_corners.mean(axis=0)))
                    for (d_lo, d_hi), ds in dist_stats.items():
                        if d_lo <= pred_dist < d_hi:
                            ds[(thr, "scores")][cls_name].append(conf)
                            ds[(thr, "tp")][cls_name].append(0)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(frame_ids)}] {frame_id}")

    # ═══════════════════════════════════════════════════════════════════
    # 결과 출력
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print(f"  Rotated BEV IoU Evaluation (Sutherland-Hodgman)")
    print(f"  Distance: 0~{max_dist}m | IoU thresholds: {iou_thresholds}")
    print(f"  Frames: {len(frame_ids)}")
    print("=" * 75)

    for thr in iou_thresholds:
        all_scores_t = stats_by_thr[thr]["all_scores"]
        all_tp_t = stats_by_thr[thr]["all_tp"]

        total_tp = 0
        total_fp = 0
        total_fn = 0
        aps = {}

        print(f"\n{'Class':<15} {'GT':>6} {'TP':>6} {'FP':>6} {'FN':>6} "
              f"{'Prec':>8} {'Recall':>8} {'AP@'+str(thr):>8}")
        print("-" * 75)

        for cls in sorted(TARGET_CLASSES):
            n_gt = n_gt_total[cls]
            scores = all_scores_t[cls]
            tp_flags = all_tp_t[cls]

            tp_count = sum(tp_flags)
            fp_count = len(tp_flags) - tp_count
            fn_count = n_gt - tp_count

            prec = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            rec = tp_count / n_gt if n_gt > 0 else 0.0
            ap = compute_ap(scores, tp_flags, n_gt)
            aps[cls] = ap

            total_tp += tp_count
            total_fp += fp_count
            total_fn += fn_count

            print(f"{cls:<15} {n_gt:>6} {tp_count:>6} {fp_count:>6} {fn_count:>6} "
                  f"{prec:>8.4f} {rec:>8.4f} {ap:>8.4f}")

        map_val = np.mean(list(aps.values())) if aps else 0.0
        total_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        total_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        print("-" * 75)
        print(f"{'Total':<15} {sum(n_gt_total.values()):>6} {total_tp:>6} "
              f"{total_fp:>6} {total_fn:>6} "
              f"{total_prec:>8.4f} {total_rec:>8.4f} {map_val:>8.4f}")
        print(f"\n  mAP@{thr}: {map_val:.4f}")

    # 인스턴스 통계 (첫 번째 임계값 기준)
    thr0 = iou_thresholds[0]
    tp0 = sum(sum(stats_by_thr[thr0]["all_tp"][cls]) for cls in TARGET_CLASSES)
    n_pred0 = sum(len(stats_by_thr[thr0]["all_tp"][cls]) for cls in TARGET_CLASSES)
    fp0 = n_pred0 - tp0
    fn0 = sum(n_gt_total.values()) - tp0
    print(f"\n{'─'*40}")
    print(f"  Instance Statistics (IoU={thr0})")
    print(f"{'─'*40}")
    print(f"  Total GT (≤{max_dist}m):  {sum(n_gt_total.values())}")
    print(f"  Total predictions:  {n_pred0}")
    print(f"  TP: {tp0}  FP: {fp0}  FN: {fn0}")

    # ── 거리별 평가 ──
    print(f"\n{'═'*75}")
    print(f"  Distance-based Evaluation")
    print(f"{'═'*75}")
    for thr in iou_thresholds:
        print(f"\n  ── AP@{thr} by distance ──")
        header = f"  {'Range':<12}"
        for cls in sorted(TARGET_CLASSES):
            header += f" {'GT_'+cls:>10} {'TP_'+cls:>8} {'AP_'+cls:>8}"
        header += f" {'mAP':>8}"
        print(header)
        print(f"  {'-'*len(header)}")
        for d_lo, d_hi in DIST_RANGES:
            ds = dist_stats[(d_lo, d_hi)]
            row = f"  {d_lo:>2}~{d_hi:<2}m     "
            aps_d = {}
            for cls in sorted(TARGET_CLASSES):
                n_gt_d = ds["n_gt"][cls]
                scores_d = ds[(thr, "scores")][cls]
                tp_d = ds[(thr, "tp")][cls]
                tp_count_d = sum(tp_d)
                ap_d = compute_ap(scores_d, tp_d, n_gt_d) if n_gt_d > 0 else 0.0
                aps_d[cls] = ap_d
                row += f" {n_gt_d:>10} {tp_count_d:>8} {ap_d:>8.4f}"
            map_d = np.mean(list(aps_d.values())) if aps_d else 0.0
            row += f" {map_d:>8.4f}"
            print(row)

    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Rotated BEV IoU evaluation for fusion predictions")
    parser.add_argument("--pred_dir", type=str, default="data/kitti/pred")
    parser.add_argument("--label_dir", type=str,
                        default="/home/a/OpenPCDet_my/data/kitti/training/label_2")
    parser.add_argument("--calib_dir", type=str,
                        default="/home/a/OpenPCDet_my/data/kitti/training/calib")
    parser.add_argument("--split_file", type=str,
                        default="/home/a/CUDA-PointPillars/tool/eval/val_occ0.txt")
    parser.add_argument("--max_dist", type=float, default=30.0)
    parser.add_argument("--iou_thr", type=float, nargs="+", default=[0.5, 0.3, 0.25])
    parser.add_argument("--score_thr", type=float, default=0.7,
                        help="최소 confidence score (이하 예측 제외)")
    args = parser.parse_args()

    run_eval(
        pred_dir=args.pred_dir,
        label_dir=args.label_dir,
        calib_dir=args.calib_dir,
        split_file=args.split_file,
        max_dist=args.max_dist,
        iou_thresholds=args.iou_thr,
        score_thr=args.score_thr,
    )


if __name__ == "__main__":
    main()
