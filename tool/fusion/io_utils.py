# tool/fusion/io_utils.py

from pathlib import Path

# ──────────────────────────────────────────────
# CUDA-PointPillars C++ 네이티브 출력 포맷 파서
# 필드 순서 (9개):
#   z  x  y  ry  l  w  h  cls_id  score
# ──────────────────────────────────────────────
_CLS_ID_MAP = {
    0: "Car",
    1: "Pedestrian",
    2: "Cyclist",
}


def parse_cuda_pp_line(line):
    """CUDA-PointPillars C++ 바이너리가 출력하는 9-field 포맷 파싱."""
    parts = line.strip().split()
    if len(parts) < 9:
        return None

    try:
        z      = float(parts[0])
        x      = float(parts[1])
        y      = float(parts[2])
        ry     = float(parts[3])
        l      = float(parts[4])
        w      = float(parts[5])
        h      = float(parts[6])
        cls_id = int(float(parts[7]))
        score  = float(parts[8])
    except ValueError:
        return None

    cls_name = _CLS_ID_MAP.get(cls_id, f"cls_{cls_id}")

    return {
        "cls_name":   cls_name,
        "cls_id":     cls_id,
        "truncated":  0.0,
        "occluded":   0,
        "alpha":      -10.0,           # 2D bbox 없으므로 placeholder
        "bbox":       [0.0, 0.0, 0.0, 0.0],  # 3D-only 출력; 2D proj 필요 시 별도 계산
        "dimensions": [h, w, l],       # KITTI 순서: h, w, l
        "location":   [x, y, z],       # KITTI 순서: x, y, z (카메라 좌표)
        "rotation_y": ry,
        "score":      score,
    }


# ──────────────────────────────────────────────
# 기존 KITTI 16-field 포맷 파서 (호환 유지)
# ──────────────────────────────────────────────

def parse_pp_kitti_line(line):
    """표준 KITTI 16-field 포맷 파싱."""
    parts = line.strip().split()
    if len(parts) < 16:
        return None

    try:
        return {
            "cls_name":   parts[0],
            "truncated":  float(parts[1]),
            "occluded":   int(float(parts[2])),
            "alpha":      float(parts[3]),
            "bbox":       [float(parts[4]), float(parts[5]),
                           float(parts[6]), float(parts[7])],
            "dimensions": [float(parts[8]), float(parts[9]), float(parts[10])],
            "location":   [float(parts[11]), float(parts[12]), float(parts[13])],
            "rotation_y": float(parts[14]),
            "score":      float(parts[15]),
        }
    except (ValueError, IndexError):
        return None


def load_pp_predictions(txt_path):
    """
    CUDA-PointPillars txt 로드.
    9-field(네이티브) 와 16-field(KITTI) 포맷을 자동 구분.
    """
    txt_path = Path(txt_path)
    preds = []

    if not txt_path.exists():
        return preds

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 16:
                obj = parse_pp_kitti_line(line)
            elif len(parts) >= 9:
                obj = parse_cuda_pp_line(line)
            else:
                obj = None

            if obj is not None:
                preds.append(obj)

    return preds


# ──────────────────────────────────────────────
# 저장 / 포맷
# ──────────────────────────────────────────────

def format_pp_prediction_line(obj):
    x1, y1, x2, y2 = obj["bbox"]
    h, w, l = obj["dimensions"]
    x, y, z = obj["location"]
    ry    = obj["rotation_y"]
    score = obj["score"]

    return (
        f"{obj['cls_name']} "
        f"{obj['truncated']:.2f} "
        f"{obj['occluded']} "
        f"{obj['alpha']:.4f} "
        f"{x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f} "
        f"{h:.4f} {w:.4f} {l:.4f} "
        f"{x:.4f} {y:.4f} {z:.4f} "
        f"{ry:.4f} "
        f"{score:.6f}"
    )


def save_pp_predictions(preds, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        for obj in preds:
            f.write(format_pp_prediction_line(obj) + "\n")


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────

def load_frame_ids(split_file):
    split_file = Path(split_file)
    frame_ids = []

    with open(split_file, "r") as f:
        for line in f:
            frame_id = line.strip()
            if frame_id:
                frame_ids.append(frame_id)

    return frame_ids


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)