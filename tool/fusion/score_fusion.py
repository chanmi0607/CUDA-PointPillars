# tool/fusion/score_fusion.py

from copy import deepcopy


def calibrate_pp_scores(
    pp_preds,
    match_results,
    gamma=0.5, # YOLO 영향력
    min_yolo_score=0.5, # YOLO confidence score threshold
    min_iou=0.5,
    unmatched_penalty=0.8,
    drop_unmatched=False,
    debug=False,
):
    fused_preds = deepcopy(pp_preds)

    indices_to_drop = []

    for match in match_results:
        pp_idx = match["pp_idx"]
        pp_obj = fused_preds[pp_idx]
        old_score = pp_obj["score"]

        if not match["matched"]:
            if drop_unmatched:
                indices_to_drop.append(pp_idx)
                if debug:
                    print(f"[DROP] idx={pp_idx} cls={pp_obj['cls_name']} pp={old_score:.4f}")
            else:
                # 점수 대폭 삭감 (예: 0.15점 -> 0.03점)
                new_score = old_score * (1.0 - unmatched_penalty)
                pp_obj["score"] = new_score
                if debug:
                    print(f"[PENALTY] idx={pp_idx} cls={pp_obj['cls_name']} pp={old_score:.4f} -> {new_score:.4f}")
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
                
        # 3. 매칭은 됐지만 커트라인 미달로 인정받지 못한 경우 (마찬가지로 페널티)
        else:
            if drop_unmatched:
                indices_to_drop.append(pp_idx)
                if debug:
                    print(f"[DROP-WEAK MATCH] idx={pp_idx} pp={old_score:.4f} yolo={yolo_score:.4f} iou={iou:.4f}")
            else:
                new_score = old_score * (1.0 - unmatched_penalty)
                pp_obj["score"] = new_score
                if debug:
                    print(
                        f"[PENALTY-WEAK MATCH] idx={pp_idx} cls={pp_obj['cls_name']} "
                        f"pp={old_score:.4f} -> {new_score:.4f}"
                    )
            
    for idx in sorted(indices_to_drop, reverse=True):
        del fused_preds[idx]

    return fused_preds