from __future__ import annotations

import math
from pathlib import Path

from ultralytics import YOLO


class ByteTrackPersonTracker:
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.35):
        self.model = YOLO(model_name)
        self.confidence = float(confidence)
        self.tracker_config = str(
            Path(__file__).resolve().parent.parent / "config" / "bytetrack.yaml"
        )

    def track(self, frame):
        if frame is None or frame.size == 0:
            return []

        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence,
            classes=[0],
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None
        cls = boxes.cls.cpu().numpy() if boxes.cls is not None else None
        ids = boxes.id.cpu().numpy() if boxes.id is not None else None
        height, width = frame.shape[:2]

        detections = []
        for idx, box in enumerate(xyxy):
            if len(box) != 4 or cls is not None and (
                idx >= len(cls) or int(cls[idx]) != 0
            ):
                continue
            track_id = None
            if ids is not None and idx < len(ids):
                value = float(ids[idx])
                if value >= 0:
                    track_id = int(value)
            confidence = 0.0 if conf is None or idx >= len(conf) else float(conf[idx])
            if not math.isfinite(confidence):
                confidence = 0.0
            if not all(math.isfinite(float(value)) for value in box):
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
            y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": max(0.0, min(1.0, confidence)),
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                }
            )
        return detections
