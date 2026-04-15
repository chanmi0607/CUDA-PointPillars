# tool/fusion/matcher.py

def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def match_pp_with_yolo(pp_preds, yolo_preds, iou_thr=0.5):
    """
    Each PP bbox selects one best YOLO bbox of the same class.
    """
    match_results = []

    for pp_idx, pp in enumerate(pp_preds):
        best_iou = -1.0
        best_yolo_idx = -1
        best_yolo_obj = None

        for yolo_idx, yolo in enumerate(yolo_preds):
            if pp["cls_name"] != yolo["cls_name"]:
                continue

            iou = compute_iou(pp["bbox"], yolo["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_yolo_idx = yolo_idx
                best_yolo_obj = yolo

        matched = best_yolo_obj is not None and best_iou >= iou_thr

        match_results.append({
            "pp_idx": pp_idx,
            "matched": matched,
            "iou": best_iou if best_iou > 0 else 0.0,
            "yolo_idx": best_yolo_idx if matched else -1,
            "yolo_obj": best_yolo_obj if matched else None,
        })

    return match_results