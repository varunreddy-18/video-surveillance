from __future__ import annotations

import cv2
import numpy as np


COLORS = {
    "person": (0, 255, 0),
    "zone": (0, 165, 255),
    "intrusion": (0, 0, 255),
    "loitering": (255, 0, 255),
}


def draw_polygon(frame, points, color, label=None, thickness=2):
    if not points:
        return
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    if label:
        centroid = pts.mean(axis=0)
        cv2.putText(frame, label, (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_detection(frame, det, event_label=None):
    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
    track_id = det.get("track_id")
    conf = det.get("confidence", 0.0)
    color = COLORS["person"]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"ID {track_id} {conf:.2f}" if track_id is not None else f"{conf:.2f}"
    cv2.putText(frame, text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if event_label:
        cv2.putText(frame, event_label, (x1, max(0, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["intrusion"], 2)


def annotate_frame(frame, detections, zones, labels):
    frame = frame.copy()
    for zone in zones:
        draw_polygon(frame, zone["polygon"], COLORS["zone"], zone.get("label"))

    for det in detections:
        label = labels.get(det.get("track_id"))
        draw_detection(frame, det, event_label=label)
    return frame
