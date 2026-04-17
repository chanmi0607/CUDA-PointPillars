# tool/fusion/pipeline.py

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from io_utils import (
    load_pp_predictions,
    save_pp_predictions,
    load_frame_ids,
    ensure_dir,
)
from yolo_wrapper import YoloTRTDetector
from matcher import match_pp_with_yolo, get_unmatched_yolo
from score_fusion import calibrate_pp_scores
from frustum_fallback import generate_frustum_box, filter_overlapping_fallbacks
from visualize import save_fusion_vis

# kitti_util은 tool/eval/ 아래에 있으므로 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
import kitti_util


KITTI_CLS_NAMES = {"Car", "Pedestrian", "Cyclist", "Van", "Person_sitting",
                   "Truck", "Tram", "Misc"}

def load_gt_boxes(label_path):
    """
    KITTI label 파일을 읽어 GT 박스 리스트 반환.
    형식: type trunc occ alpha x1 y1 x2 y2 h w l x y z ry
    """
    boxes = []
    label_path = Path(label_path)
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls_name = parts[0]
            if cls_name not in KITTI_CLS_NAMES:
                continue
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            boxes.append({
                "cls_name":   cls_name,
                "location":   [x, y, z],
                "dimensions": [h, w, l],
                "rotation_y": ry,
            })
    return boxes


def get_fov_flag(points, calib, img_shape):
    """OpenPCDet calibration_kitti.py 의 rect_to_img + get_fov_flag와 동일한 로직."""
    # lidar → rect (OpenPCDet: lidar_to_rect)
    pts_lidar_hom = np.hstack([points[:, :3], np.ones((len(points), 1))])
    pts_rect = pts_lidar_hom @ (calib.V2C.T @ calib.R0.T)  # (N, 3)

    # rect → image (OpenPCDet: rect_to_img)
    pts_rect_hom = np.hstack([pts_rect, np.ones((len(pts_rect), 1))])
    pts_2d_hom   = pts_rect_hom @ calib.P.T                 # (N, 3)
    pts_img      = pts_2d_hom[:, :2] / pts_rect_hom[:, 2:3] # (N, 2)
    pts_rect_depth = pts_2d_hom[:, 2] - calib.P.T[3, 2]     # OpenPCDet depth

    val_x = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_y = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_d = pts_rect_depth >= 0

    return np.logical_and(np.logical_and(val_x, val_y), val_d)


def filter_velodyne_fov(velodyne_dir, calib_dir, image_dir, out_dir):
    """
    velodyne_dir 의 모든 .bin 파일을 카메라 FOV 기준으로 필터링해
    out_dir 에 저장한다. OpenPCDet FOV_POINTS_ONLY: True 와 동일.
    """
    velodyne_dir = Path(velodyne_dir)
    calib_dir    = Path(calib_dir)
    image_dir    = Path(image_dir)
    out_dir      = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(velodyne_dir.glob("*.bin"))
    print(f"[FOV] Filtering {len(bin_files)} frames ...")

    t0 = time.perf_counter()
    for bin_path in bin_files:
        frame_id   = bin_path.stem
        calib_path = calib_dir / f"{frame_id}.txt"
        img_path   = image_dir / f"{frame_id}.png"

        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

        if not calib_path.exists() or not img_path.exists():
            # calib/image 없으면 필터링 없이 그대로 복사
            points.tofile(out_dir / bin_path.name)
            continue

        import cv2
        img       = cv2.imread(str(img_path))
        img_shape = img.shape[:2]  # (H, W)

        calib    = kitti_util.Calibration(str(calib_path))
        fov_flag = get_fov_flag(points, calib, img_shape)
        points[fov_flag].tofile(out_dir / bin_path.name)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[FOV] Done. {elapsed:.0f} ms total / {elapsed/len(bin_files):.2f} ms/frame")


def run_command(cmd, cwd=None):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def run_pointpillars(project_root, velodyne_dir, pred_dir, skip_pp=False,
                     calib_dir=None, image_dir=None, fov_filter=False):
    """Returns (ms_per_frame, num_frames) or (None, 0) if skipped."""
    if skip_pp:
        print("[INFO] Skipping PointPillars inference.")
        return None, 0

    build_dir = project_root / "build"
    exe_path  = build_dir / "pointpillar"

    if not exe_path.exists():
        raise FileNotFoundError(f"PointPillars executable not found: {exe_path}")

    # 절대경로로 변환
    velodyne_dir_abs = str((project_root / velodyne_dir).resolve()) \
        if not Path(velodyne_dir).is_absolute() else str(velodyne_dir)
    pred_dir_abs = str((project_root / pred_dir).resolve()) \
        if not Path(pred_dir).is_absolute() else str(pred_dir)

    Path(pred_dir_abs).mkdir(parents=True, exist_ok=True)

    # FOV 필터링
    tmp_dir = None
    input_dir = velodyne_dir_abs
    if fov_filter and calib_dir and image_dir:
        tmp_dir  = tempfile.mkdtemp(prefix="pp_fov_")
        calib_dir_abs = str((project_root / calib_dir).resolve()) \
            if not Path(calib_dir).is_absolute() else str(calib_dir)
        image_dir_abs = str((project_root / image_dir).resolve()) \
            if not Path(image_dir).is_absolute() else str(image_dir)
        filter_velodyne_fov(velodyne_dir_abs, calib_dir_abs, image_dir_abs, tmp_dir)
        input_dir = tmp_dir

    import glob
    num_bins = len(glob.glob(input_dir + "/*.bin"))

    t_pp_start = time.perf_counter()
    run_command(
        [str(exe_path), input_dir + "/", pred_dir_abs + "/"],
        cwd=str(build_dir),
    )
    t_pp_end = time.perf_counter()

    if tmp_dir:
        shutil.rmtree(tmp_dir)

    ms_per_frame = None
    if num_bins > 0:
        pp_total_ms  = (t_pp_end - t_pp_start) * 1000
        ms_per_frame = pp_total_ms / num_bins
        print(f"[TIMING] PointPillars: {pp_total_ms:.1f} ms total / "
              f"{ms_per_frame:.2f} ms/frame / "
              f"{num_bins/(t_pp_end-t_pp_start):.1f} FPS  ({num_bins} frames)")
    return ms_per_frame, num_bins


def run_kitti_format(project_root, split_file="tool/eval/val.txt"):
    run_command(
        ["python3", "tool/eval/kitti_format.py", "--split_file", split_file],
        cwd=str(project_root),
    )


def run_evaluate(project_root, result_path, max_dist=-1, pr_save_dir=None,
                  split_file="tool/eval/val.txt"):
    cmd = [
        "python3",
        "tool/eval/evaluate.py",
        "evaluate",
        "--label_path=data/kitti/training/label_2/",
        f"--result_path={result_path}",
        f"--label_split_file={split_file}",
        "--current_class=0,1,2",
        "--coco=False",
    ]
    if max_dist > 0:
        cmd.append(f"--max_dist={max_dist}")
    if pr_save_dir is not None:
        cmd.append(f"--pr_save_dir={pr_save_dir}")
    run_command(cmd, cwd=str(project_root))


def copy_baseline_pred(project_root):
    pred_dir     = project_root / "data/kitti/pred"
    baseline_dir = project_root / "data/kitti/pred_baseline"

    if baseline_dir.exists():
        shutil.rmtree(baseline_dir)

    shutil.copytree(pred_dir, baseline_dir)
    print(f"[INFO] Baseline predictions copied to: {baseline_dir}")


def run_fusion(
    project_root,
    yolo_engine,
    image_dir,
    split_file,
    save_dir,
    class_map,
    yolo_device=0,
    yolo_conf=0.001,
    yolo_iou=0.7,
    yolo_imgsz=640,
    match_iou_thr=0.5,
    gamma=0.25,
    min_yolo_score=0.5,
    min_match_iou=0.5,
    max_frames=-1,
    debug=False,
    # ── Frustum Fallback ──────────────────────────
    frustum_fallback=False,
    velodyne_dir=None,
    calib_dir=None,
    frustum_near=0.5,
    frustum_far=60.0,
    frustum_bin_size=1.0,
    min_yolo_score_fallback=0.3,
    # ── DBSCAN / Heatmap 파라미터 ────────────────
    dbscan_eps=0.7,
    dbscan_min_samples=3,
    heatmap_grid_size=0.15,
    heatmap_yaw_step=5,
    # ── 시각화 ────────────────────────────────────
    vis_dir=None,
    vis_all=False,
    bev_x_range=(-20, 20),
    bev_z_range=(0, 25),
    label_dir=None,
):
    pred_dir = project_root / "data/kitti/pred"
    save_dir = project_root / save_dir
    ensure_dir(save_dir)

    frame_ids = load_frame_ids(project_root / split_file)
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]

    detector = YoloTRTDetector(
        engine_path=project_root / yolo_engine,
        device=yolo_device,
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=yolo_imgsz,
        class_map=class_map,
        verbose=False,
    )

    print(f"[INFO] #frames to fuse: {len(frame_ids)}")

    total_frames = 0
    t_yolo_total = 0.0
    t_match_total = 0.0
    t_total = 0.0

    # frustum fallback 통계
    fb_stats = {"frames": 0, "total": 0, "by_cls": {}, "dedup": 0}

    # frustum fallback 사용 시 필요한 경로 검증
    if frustum_fallback:
        if velodyne_dir is None or calib_dir is None:
            raise ValueError("--frustum_fallback 사용 시 --velodyne_dir, --calib_dir 필수")
        velodyne_dir = Path(velodyne_dir)
        calib_dir    = Path(calib_dir)

    # 시각화 디렉토리 / GT 레이블 경로
    if vis_dir is not None:
        vis_dir = Path(vis_dir)
        ensure_dir(vis_dir)
    label_dir = Path(label_dir) if label_dir is not None else None

    for idx, frame_id in enumerate(frame_ids):
        pp_txt    = pred_dir / f"{frame_id}.txt"
        img_path  = project_root / image_dir / f"{frame_id}.png"
        save_path = save_dir / f"{frame_id}.txt"

        pp_preds = load_pp_predictions(pp_txt)

        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        t0 = time.perf_counter()
        yolo_preds = detector.predict(img_path)
        t1 = time.perf_counter()

        if len(pp_preds) == 0 and not frustum_fallback:
            save_pp_predictions([], save_path)
            if debug:
                print(f"[{idx+1}/{len(frame_ids)}] {frame_id}: no PP predictions")
            t_yolo_total += (t1 - t0)
            t_total      += (t1 - t0)
            total_frames += 1
            continue

        matches     = match_pp_with_yolo(pp_preds, yolo_preds, iou_thr=match_iou_thr)
        fused_preds = calibrate_pp_scores(
            pp_preds,
            matches,
            gamma=gamma,
            min_yolo_score=min_yolo_score,
            min_iou=min_match_iou,
            drop_unmatched=False,
            unmatched_penalty=0.8,
            debug=debug,
        )

        # ── Source 태깅 (시각화용) ────────────────────────────
        # drop_unmatched=True 이면 calibrate_pp_scores 가 일부 박스를 삭제하므로
        # matches 의 순번(i)과 fused_preds 의 인덱스가 불일치한다.
        # score_fusion 의 드롭 조건을 그대로 재현해 살아남은 match 만 추린 뒤
        # fused_preds 와 zip 으로 대응시킨다.
        matched_yolo_indices = set()
        surviving_matches = []
        for m in matches:
            if not m["matched"]:
                # unmatched → drop_unmatched=True 이므로 삭제됨
                continue
            yolo_score = float(m["yolo_obj"]["score"])
            iou = float(m["iou"])
            if yolo_score < min_yolo_score or iou < min_match_iou:
                # weak match → drop_unmatched=True 이므로 삭제됨
                continue
            surviving_matches.append(m)

        # fused_preds 의 앞부분(fallback 추가 전)은 surviving_matches 와 1:1 대응
        for fused_obj, m in zip(fused_preds[:len(surviving_matches)], surviving_matches):
            fused_obj["source"] = "pp_matched"
            if m["yolo_idx"] >= 0:
                matched_yolo_indices.add(m["yolo_idx"])

        # ── Frustum Fallback ──────────────────────────────────
        n_fallback = 0
        if frustum_fallback:
            unmatched_yolo = get_unmatched_yolo(yolo_preds, matches)

            unmatched_yolo = [
                (yi, yolo) for yi, yolo in unmatched_yolo
                if float(yolo["score"]) >= min_yolo_score_fallback
            ]

            if unmatched_yolo:
                bin_path   = velodyne_dir / f"{frame_id}.bin"
                calib_path = calib_dir / f"{frame_id}.txt"

                if bin_path.exists() and calib_path.exists():
                    points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
                    calib  = kitti_util.Calibration(str(calib_path))

                    raw_fallbacks = []
                    for yi, yolo_det in unmatched_yolo:
                        box = generate_frustum_box(
                            points, calib, yolo_det,
                            near=frustum_near,
                            far=frustum_far,
                            dbscan_eps=dbscan_eps,
                            dbscan_min_samples=dbscan_min_samples,
                            heatmap_grid_size=heatmap_grid_size,
                            heatmap_yaw_step=heatmap_yaw_step,
                            depth_bin_size=frustum_bin_size,
                            debug=debug,
                        )
                        if box is not None:
                            box["source"] = "fallback"
                            raw_fallbacks.append(box)

                    # BEV 중복 제거: PP와 겹치는 fallback 제거
                    before_dedup = len(raw_fallbacks)
                    fallbacks = filter_overlapping_fallbacks(
                        raw_fallbacks, fused_preds, debug=debug
                    )
                    fb_stats["dedup"] += (before_dedup - len(fallbacks))

                    fused_preds.extend(fallbacks)
                    n_fallback = len(fallbacks)
                elif debug:
                    print(f"[FRUSTUM] {frame_id}: bin or calib not found, skip")

            # 통계 갱신
            if n_fallback > 0:
                fb_stats["frames"] += 1
                fb_stats["total"]  += n_fallback
                for fb in fallbacks:
                    cls = fb["cls_name"]
                    fb_stats["by_cls"][cls] = fb_stats["by_cls"].get(cls, 0) + 1

        t2 = time.perf_counter()
        save_pp_predictions(fused_preds, save_path)

        # ── 시각화 저장 ───────────────────────────────────────
        if vis_dir is not None and (n_fallback > 0 or vis_all):
            vis_path = vis_dir / f"{frame_id}.png"
            gt_boxes = None
            if label_dir is not None:
                gt_boxes = load_gt_boxes(label_dir / f"{frame_id}.txt")
            save_fusion_vis(
                img_path=img_path,
                yolo_preds=yolo_preds,
                fused_preds=fused_preds,
                matched_yolo_indices=matched_yolo_indices,
                save_path=vis_path,
                frame_id=frame_id,
                bev_x_range=bev_x_range,
                bev_z_range=bev_z_range,
                gt_boxes=gt_boxes,
            )

        t_yolo_total  += (t1 - t0)
        t_match_total += (t2 - t1)
        t_total       += (t2 - t0)
        total_frames  += 1

        if idx % 50 == 0 or debug:
            print(
                f"[{idx+1}/{len(frame_ids)}] frame={frame_id} "
                f"PP={len(pp_preds)} YOLO={len(yolo_preds)} "
                f"fallback={n_fallback} saved={save_path.name}"
            )

    # ── Frustum Fallback 통계 출력 ────────────────────────────
    if frustum_fallback and fb_stats["total"] > 0:
        print(f"\n[FRUSTUM STATS]")
        print(f"  Frames with fallback : {fb_stats['frames']}/{total_frames}")
        print(f"  Total fallback boxes : {fb_stats['total']}")
        cls_str = ", ".join(f"{k}={v}" for k, v in sorted(fb_stats["by_cls"].items()))
        print(f"  By class             : {cls_str}")
        if fb_stats["frames"] > 0:
            print(f"  Avg per frame (where applied): "
                  f"{fb_stats['total']/fb_stats['frames']:.2f}")
        print(f"  BEV dedup removed    : {fb_stats['dedup']}")
    elif frustum_fallback:
        print(f"\n[FRUSTUM STATS] No fallback boxes generated.")

    if vis_dir is not None:
        n_vis = fb_stats["frames"] if not vis_all else total_frames
        print(f"[VIS] {n_vis} frames saved to {vis_dir}")

    if total_frames > 0:
        avg_yolo  = t_yolo_total  / total_frames * 1000
        avg_match = t_match_total / total_frames * 1000
        avg_total = t_total       / total_frames * 1000
        fps       = total_frames  / t_total
        print(f"\n[TIMING] YOLO+Fusion  frames={total_frames}")
        print(f"  YOLO inference : {avg_yolo:.2f} ms/frame")
        print(f"  Match + Fusion : {avg_match:.2f} ms/frame")
        print(f"  YOLO+Fusion    : {avg_total:.2f} ms/frame")
        print(f"  FPS (YOLO+Fusion only): {fps:.1f}")
        return avg_total, total_frames
    return None, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root",       type=str, default=".")
    parser.add_argument("--yolo_engine",        type=str, required=True)
    parser.add_argument("--image_dir",          type=str, default="data/kitti/training/image_2")
    parser.add_argument("--split_file",         type=str, default="tool/eval/val_occ0.txt")
    parser.add_argument("--save_dir",           type=str, default="data/kitti/fused_pred")

    parser.add_argument("--skip_pp",            action="store_true")
    parser.add_argument("--skip_baseline_eval", action="store_true")
    parser.add_argument("--skip_fused_eval",    action="store_true")

    parser.add_argument("--yolo_device",    type=int,   default=0)
    parser.add_argument("--yolo_conf",      type=float, default=0.001)
    parser.add_argument("--yolo_iou",       type=float, default=0.7)
    parser.add_argument("--yolo_imgsz",     type=int,   default=640)

    parser.add_argument("--match_iou_thr",  type=float, default=0.5)
    parser.add_argument("--gamma",          type=float, default=0.25)
    parser.add_argument("--min_yolo_score", type=float, default=0.5)
    parser.add_argument("--min_match_iou",  type=float, default=0.5)

    parser.add_argument("--max_frames",     type=int,   default=-1)
    parser.add_argument("--debug",          action="store_true")

    parser.add_argument(
        "--velodyne_dir",
        type=str,
        default="/home/a/OpenPCDet_my/data/kitti/training/velodyne",
    )
    parser.add_argument(
        "--pp_pred_dir",
        type=str,
        default="data/kitti/pred",
    )
    parser.add_argument(
        "--calib_dir",
        type=str,
        default="data/kitti/training/calib",
    )
    parser.add_argument(
        "--fov_filter",
        action="store_true",
        help="Filter LiDAR points to camera FOV before PP inference (matches OpenPCDet FOV_POINTS_ONLY=True)",
    )
    parser.add_argument(
        "--max_dist",
        type=float,
        default=-1,
        help="Evaluate only objects within this distance (camera z, meters). -1 = no limit.",
    )

    # ── Frustum Fallback ──────────────────────────────────────
    parser.add_argument(
        "--frustum_fallback",
        action="store_true",
        help="YOLO-only 검출에 대해 frustum 기반 BEV box를 생성해 예측에 추가",
    )
    parser.add_argument(
        "--frustum_near",
        type=float,
        default=0.5,
        help="Frustum near plane 깊이 (m)",
    )
    parser.add_argument(
        "--frustum_far",
        type=float,
        default=60.0,
        help="Frustum far plane 깊이 (m)",
    )
    parser.add_argument(
        "--frustum_bin_size",
        type=float,
        default=1.0,
        help="Depth 히스토그램 bin 크기 (m)",
    )
    parser.add_argument(
        "--min_yolo_score_fallback",
        type=float,
        default=0.3,
        help="Frustum fallback을 적용할 YOLO confidence 최소값 (기존 --min_yolo_score와 별개)",
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=0.7,
        help="DBSCAN 이웃 반경 [m] (frustum 클러스터링)",
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=3,
        help="DBSCAN 코어 포인트 최소 이웃 수",
    )
    parser.add_argument(
        "--heatmap_grid_size",
        type=float,
        default=0.15,
        help="Heatmap BEV grid 해상도 [m]",
    )
    parser.add_argument(
        "--heatmap_yaw_step",
        type=int,
        default=5,
        help="Heatmap yaw 탐색 간격 [deg]",
    )

    # ── 시각화 ────────────────────────────────────────────────
    parser.add_argument(
        "--vis_dir",
        type=str,
        default=None,
        help="시각화 이미지 저장 디렉토리. 지정 시 fallback 프레임에 대해 이미지+BEV PNG 생성",
    )
    parser.add_argument(
        "--vis_all",
        action="store_true",
        help="fallback이 없는 프레임도 시각화 저장 (기본: fallback 있는 프레임만)",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default="/home/a/OpenPCDet_my/data/kitti/training/label_2",
        help="GT label 디렉토리 (BEV IoU 표시용). 미지정 시 IoU 표시 안 함.",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    class_map = {
        0: "Car",
        3: "Pedestrian",
        5: "Cyclist",
    }

    print("[INFO] Step 1: Run PointPillars inference")
    pp_ms_per_frame, pp_num_frames = run_pointpillars(
        project_root,
        args.velodyne_dir,
        args.pp_pred_dir,
        skip_pp=args.skip_pp,
        calib_dir=args.calib_dir,
        image_dir=args.image_dir,
        fov_filter=args.fov_filter,
    )

    if not args.skip_pp:
        print("[INFO] Step 2: Convert PP outputs to KITTI format")
        run_kitti_format(project_root, split_file=args.split_file)
    else:
        print("[INFO] Step 2: Skipped kitti_format (--skip_pp)")

    print("[INFO] Step 3: Backup baseline predictions")
    copy_baseline_pred(project_root)

    if not args.skip_baseline_eval:
        print("[INFO] Step 4: Evaluate baseline PP")
        run_evaluate(project_root, "data/kitti/pred_baseline", max_dist=args.max_dist,
                     split_file=args.split_file)

    # BEV 시각화 범위: max_dist 기반 자동 설정
    if args.max_dist > 0:
        bev_z = (0, args.max_dist * 1.25)
        bev_x = (-args.max_dist * 0.75, args.max_dist * 0.75)
    else:
        bev_z = (0, 50)
        bev_x = (-30, 30)

    print("[INFO] Step 5: Run YOLO + Fusion")
    yolo_ms_per_frame, yolo_num_frames = run_fusion(
        project_root=project_root,
        yolo_engine=args.yolo_engine,
        image_dir=args.image_dir,
        split_file=args.split_file,
        save_dir=args.save_dir,
        class_map=class_map,
        yolo_device=args.yolo_device,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_imgsz=args.yolo_imgsz,
        match_iou_thr=args.match_iou_thr,
        gamma=args.gamma,
        min_yolo_score=args.min_yolo_score,
        min_match_iou=args.min_match_iou,
        max_frames=args.max_frames,
        debug=args.debug,
        frustum_fallback=args.frustum_fallback,
        velodyne_dir=args.velodyne_dir,
        calib_dir=args.calib_dir,
        frustum_near=args.frustum_near,
        frustum_far=args.frustum_far,
        frustum_bin_size=args.frustum_bin_size,
        min_yolo_score_fallback=args.min_yolo_score_fallback,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        heatmap_grid_size=args.heatmap_grid_size,
        heatmap_yaw_step=args.heatmap_yaw_step,
        vis_dir=args.vis_dir,
        vis_all=args.vis_all,
        bev_x_range=bev_x,
        bev_z_range=bev_z,
        label_dir=args.label_dir,
    )

    if not args.skip_fused_eval:
        print("[INFO] Step 6: Evaluate fused PP")
        pr_save = str(project_root / args.save_dir / "pr_curves")
        run_evaluate(project_root, args.save_dir, max_dist=args.max_dist,
                     pr_save_dir=pr_save, split_file=args.split_file)

    if pp_ms_per_frame is not None and yolo_ms_per_frame is not None:
        total_ms = pp_ms_per_frame + yolo_ms_per_frame
        print(f"\n{'='*50}")
        print(f"[TIMING] Full pipeline summary")
        print(f"  PointPillars   : {pp_ms_per_frame:.2f} ms/frame")
        print(f"  YOLO + Fusion  : {yolo_ms_per_frame:.2f} ms/frame")
        print(f"  Total          : {total_ms:.2f} ms/frame")
        print(f"  Pipeline FPS   : {1000/total_ms:.1f}")
        print(f"{'='*50}")

    print("[INFO] Pipeline finished successfully.")


if __name__ == "__main__":
    main()