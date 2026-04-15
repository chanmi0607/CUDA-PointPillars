# tool/fusion/score_fusion.py

from copy import deepcopy


def calibrate_pp_scores(
    pp_preds,
    match_results,
    gamma=0.25, # YOLO 영향력
    min_yolo_score=0.5, # YOLO confidence score threshold
    min_iou=0.5,
    debug=False,
):
    fused_preds = deepcopy(pp_preds)

    for match in match_results:
        pp_idx = match["pp_idx"]
        pp_obj = fused_preds[pp_idx]
        old_score = pp_obj["score"]

        if not match["matched"]:
            if debug:
                print(f"[NO MATCH] idx={pp_idx} cls={pp_obj['cls_name']} pp={old_score:.4f}")
            continue

        yolo_obj = match["yolo_obj"]
        yolo_score = float(yolo_obj["score"])
        iou = float(match["iou"])

        if yolo_score >= min_yolo_score and iou >= min_iou:
            new_score = min(1.0, old_score + gamma * yolo_score * iou)
            pp_obj["score"] = new_score
            if debug:
                print(
                    f"[BOOST] idx={pp_idx} cls={pp_obj['cls_name']} "
                    f"pp={old_score:.4f} yolo={yolo_score:.4f} iou={iou:.4f} -> fused={new_score:.4f}"
                )
        else:
            if debug:
                print(
                    f"[KEEP] idx={pp_idx} cls={pp_obj['cls_name']} "
                    f"pp={old_score:.4f} yolo={yolo_score:.4f} iou={iou:.4f}"
                )

    return fused_preds