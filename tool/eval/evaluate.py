import time
import fire
import numpy as np
import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def filter_annos_by_distance(annos, max_dist):
    """location[:, 2]는 카메라 좌표계 z (전방 거리)."""
    filtered = []
    for anno in annos:
        if len(anno['name']) == 0:
            filtered.append(anno)
            continue
        loc = anno['location']  # (N, 3)
        mask = loc[:, 2] <= max_dist
        filtered.append({k: v[mask] if isinstance(v, np.ndarray) else v
                         for k, v in anno.items()})
    return filtered


def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             coco=False,
             score_thresh=-1,
             max_dist=-1):
    val_image_ids = _read_imageset_file(label_split_file)
    dt_annos = kitti.get_label_annos(result_path, val_image_ids)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if max_dist > 0:
        dt_annos = filter_annos_by_distance(dt_annos, max_dist)
        gt_annos = filter_annos_by_distance(gt_annos, max_dist)
    if coco:
        print(get_coco_eval_result(gt_annos, dt_annos, current_class))
    else:
        print(get_official_eval_result(gt_annos, dt_annos, current_class))


if __name__ == '__main__':
    fire.Fire()
