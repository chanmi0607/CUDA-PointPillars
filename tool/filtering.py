#!/usr/bin/env python3
"""
KITTI split 파일에서 프레임 내 모든 GT 객체가 KITTI Easy 난이도인 프레임만 필터링.

KITTI Easy 기준 (per object):
  - 2D bbox height >= 40px
  - occlusion == 0
  - truncation <= 0.15

사용법:
    python tool/filtering.py \
        --label_dir  data/kitti/training/label_2 \
        --split_file tool/eval/val.txt \
        --out_file   tool/eval/val_easy.txt
"""

import argparse
from pathlib import Path

KITTI_CLS_NAMES = {"Car", "Pedestrian", "Cyclist", "Van", "Person_sitting",
                   "Truck", "Tram", "Misc"}


def is_all_easy(label_path: Path) -> bool:
    """프레임의 모든 유효 GT 객체가 KITTI Easy 기준을 만족하면 True."""
    if not label_path.exists():
        return False

    has_object = False
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls_name = parts[0]
            if cls_name not in KITTI_CLS_NAMES:
                continue
            has_object = True

            truncation = float(parts[1])
            occlusion = int(float(parts[2]))
            # 2D bbox: x1 y1 x2 y2 = parts[4:8]
            y1 = float(parts[5])
            y2 = float(parts[7])
            bbox_height = y2 - y1

            # KITTI Easy: occ==0, trunc<=0.15, height>=40
            if occlusion != 0 or truncation > 0.15 or bbox_height < 40:
                return False

    return has_object  # 객체가 하나도 없는 프레임은 제외


def main():
    parser = argparse.ArgumentParser(
        description="Filter KITTI frames: keep only frames where ALL GT objects meet Easy difficulty"
    )
    parser.add_argument("--label_dir", type=str,
                        default="/home/a/OpenPCDet_my/data/kitti/training/label_2")
    parser.add_argument("--split_file", type=str,
                        default="tool/eval/val.txt")
    parser.add_argument("--out_file", type=str,
                        default="tool/eval/val_easy.txt")
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    split_file = Path(args.split_file)

    frame_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    print(f"[INFO] Total frames in split: {len(frame_ids)}")

    kept = []
    for fid in frame_ids:
        label_path = label_dir / f"{fid}.txt"
        if is_all_easy(label_path):
            kept.append(fid)

    print(f"[INFO] Frames with all-Easy objects: {len(kept)} / {len(frame_ids)}")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(kept) + "\n")
    print(f"[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()
