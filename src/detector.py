from __future__ import annotations

from src.tracker import ByteTrackPersonTracker


class PersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.35):
        self.tracker = ByteTrackPersonTracker(model_name=model_name, confidence=confidence)

    def detect(self, frame):
        return self.tracker.track(frame)
