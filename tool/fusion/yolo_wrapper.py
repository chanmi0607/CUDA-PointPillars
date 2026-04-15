# tool/fusion/yolo_wrapper.py

from pathlib import Path
import cv2
from ultralytics import YOLO


class YoloTRTDetector:
    def __init__(
        self,
        engine_path,
        device=0,
        conf=0.001,
        iou=0.7,
        imgsz=640,
        class_map=None,
        verbose=False,
    ):
        self.engine_path = str(engine_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.verbose = verbose

        # YOLO class id -> KITTI class name
        # 예시: {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        if class_map is None:
            class_map = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        self.class_map = class_map

        if not Path(self.engine_path).exists():
            raise FileNotFoundError(f"YOLO engine not found: {self.engine_path}")

        self.model = YOLO(self.engine_path)

    def predict(self, image_path):
        img = cv2.imread(str(image_path))
        results = self.model.predict(
            source=img,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=self.verbose,
        )

        dets = []
        if len(results) == 0:
            return dets

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return dets

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_id in zip(xyxy, confs, clss):
            if cls_id not in self.class_map:
                continue

            x1, y1, x2, y2 = box.tolist()
            dets.append({
                "cls_name": self.class_map[cls_id],
                "score": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        return dets