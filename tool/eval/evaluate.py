import time
import fire
import numpy as np
import kitti_common as kitti
import eval as eval_mod
from eval import get_official_eval_result, get_coco_eval_result


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def filter_annos_by_distance(annos, max_dist, min_dist=0.0):
    """radial 거리 r = sqrt(x^2 + z^2) (카메라 좌표 x, z) 기준으로 [min_dist, max_dist) 필터."""
    filtered = []
    for anno in annos:
        if len(anno['name']) == 0:
            filtered.append(anno)
            continue
        loc = anno['location']  # (N, 3): camera x, y, z
        r = np.sqrt(loc[:, 0] ** 2 + loc[:, 2] ** 2)
        mask = (r >= min_dist) & (r < max_dist)
        filtered.append({k: v[mask] if isinstance(v, np.ndarray) else v
                         for k, v in anno.items()})
    return filtered


def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             coco=False,
             score_thresh=-1,
             max_dist=-1,
             dist_only=False,
             dist_bins=None,
             pr_save_dir=None):
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
    elif dist_only:
        # occlusion/truncation/min-height 난이도 게이팅을 끄고 거리 제한만 적용.
        # 난이도 3열이 동일해지므로 단일 열(difficultys=[0])만 출력.
        eval_mod.DISTANCE_ONLY = True
        print(get_official_eval_result(gt_annos, dt_annos, current_class,
                                       difficultys=[0],
                                       pr_save_dir=pr_save_dir))
    else:
        print(get_official_eval_result(gt_annos, dt_annos, current_class,
                                       pr_save_dir=pr_save_dir))

    # --- 거리 구간별 표준 KITTI 평가 (radial 거리) 추가 ---
    # 전체거리 결과 뒤에 각 구간(예: 0-10,10-20,20-30,0-30 m)을 표준 Easy/Mod/Hard로 출력.
    if dist_bins and not coco:
        # fire가 "0-10,10-20"을 튜플로 넘길 수 있으므로 str/list 모두 처리
        if isinstance(dist_bins, (list, tuple)):
            bin_specs = [str(b) for b in dist_bins]
        else:
            bin_specs = str(dist_bins).split(",")
        for spec in bin_specs:
            spec = spec.strip()
            if not spec:
                continue
            lo, hi = (float(x) for x in spec.split("-"))
            gt_b = filter_annos_by_distance(gt_annos, hi, min_dist=lo)
            dt_b = filter_annos_by_distance(dt_annos, hi, min_dist=lo)
            print(f"\n================ Distance {lo:g}-{hi:g} m (radial) ================")
            print(get_official_eval_result(gt_b, dt_b, current_class))


if __name__ == '__main__':
    fire.Fire()
