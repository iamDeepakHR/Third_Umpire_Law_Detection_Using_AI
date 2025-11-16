from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    score: float

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


class BallDetector:
    def __init__(self, weights_path: str, confidence_threshold: float = 0.25, iou_threshold: float = 0.45, ball_class_id: int | None = None) -> None:
        self.model = YOLO(weights_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        # If model trained specifically on ball class, set its id; otherwise we will rank by size/score heuristics
        self.ball_class_id = ball_class_id

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            image_bgr, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False
        )
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy() if boxes.cls is not None else None
            for i, box in enumerate(xyxy):
                if classes is not None and self.ball_class_id is not None:
                    if int(classes[i]) != int(self.ball_class_id):
                        continue
                x1, y1, x2, y2 = map(int, box)
                detections.append(Detection((x1, y1, x2, y2), float(scores[i])))

        # If multiple detections, prefer the smallest bbox (ball is small) with highest score
        detections.sort(key=lambda d: ((d.bbox[2]-d.bbox[0]) * (d.bbox[3]-d.bbox[1]), -d.score))
        return detections


