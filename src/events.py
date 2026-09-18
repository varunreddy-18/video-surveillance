from __future__ import annotations

import json
from pathlib import Path


class EventLogger:
    def __init__(self, dedup_frames: int = 4):
        self.events = []
        self.last_event = {}
        self.dedup_frames = max(0, int(dedup_frames))

    def add_event(self, frame_number, timestamp, bbox, track_id, event_type, confidence, zone_id):
        if track_id is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        key = (track_id, event_type, zone_id)
        last = self.last_event.get(key)
        if last is not None and frame_number - last <= self.dedup_frames:
            return False

        payload = {
            "frame_number": int(frame_number),
            "timestamp": round(float(timestamp), 3),
            "bbox": [round(float(v), 2) for v in bbox],
            "track_id": int(track_id),
            "event_type": event_type,
            "confidence": round(float(confidence), 3),
        }
        payload["zone_id"] = zone_id

        self.events.append(payload)
        self.last_event[key] = int(frame_number)
        return True

    def save_json(self, path: str | Path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self.events, fh, indent=2)
