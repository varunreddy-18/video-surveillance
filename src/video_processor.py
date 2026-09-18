from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from src.detector import PersonDetector
from src.events import EventLogger
from src.visualizer import annotate_frame


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def normalize_polygon(points, width, height):
    return [(float(x) * width, float(y) * height) for x, y in points]


def load_zones(path: str | Path, width: int, height: int):
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid zone config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Zone config must be a JSON object: {path}")
    for key in ("min_confidence", "loitering_seconds", "event_dedup_frames"):
        if key in config:
            try:
                value = float(config[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be numeric in {path}") from exc
            if value < 0:
                raise ValueError(f"{key} must be non-negative in {path}")
    if "min_confidence" in config and float(config["min_confidence"]) > 1:
        raise ValueError(f"min_confidence must be between 0 and 1 in {path}")
    raw_zones = config.get("zones", [])
    if not isinstance(raw_zones, list) or not raw_zones:
        raise ValueError(f"Zone config must contain at least one zone: {path}")
    zones = []
    for index, zone in enumerate(raw_zones):
        if not isinstance(zone, dict):
            raise ValueError(f"Zone {index} must be an object")
        zone_id = zone.get("id")
        label = zone.get("label")
        points = zone.get("points")
        if not isinstance(zone_id, str) or not zone_id.strip():
            raise ValueError(f"Zone {index} requires a non-empty id")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Zone {zone_id} requires a non-empty label")
        if not isinstance(points, list):
            raise ValueError(f"Zone {zone_id} requires polygon points")
        if len(points) < 3:
            raise ValueError(f"Zone {zone_id} requires at least 3 points")
        normalized = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"Zone {zone_id} has an invalid point")
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Zone {zone_id} has non-numeric coordinates") from exc
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise ValueError(f"Zone {zone_id} coordinates must be normalized")
            normalized.append((x, y))
        zones.append(
            {
                "id": zone_id,
                "label": label,
                "polygon": normalize_polygon(normalized, width, height),
            }
        )
    return zones, config


def discover_videos(input_dir: str | Path):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        return []
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )


def matching_zone_config(video_path, config_dir):
    stem = video_path.stem.lower()
    candidates = [Path(config_dir) / f"zones_{stem}.json"]
    if stem.startswith("virat"):
        candidates.insert(0, Path(config_dir) / "zones_virat.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def process_video(video_path, zones_path, output_dir, model_name="yolov8n.pt"):
    started = time.perf_counter()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"Unsupported video extension: {video_path.suffix}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    fps = source_fps if source_fps > 0 else 30.0
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video metadata: {video_path}")

    zones, config = load_zones(zones_path, width, height)
    detector = PersonDetector(
        model_name=config.get("model", model_name),
        confidence=float(config.get("min_confidence", 0.35)),
    )
    logger = EventLogger(config.get("event_dedup_frames", 4))
    output_video = output_dir / f"{video_path.stem}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise OSError(f"Unable to create output video: {output_video}")

    track_state = {}
    stale_limit = max(1, int(float(config.get("track_state_timeout_seconds", 2.0)) * fps))
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                continue

            frame_count += 1
            timestamp = (frame_count - 1) / fps
            detections = detector.detect(frame)
            labels = {}
            active_ids = set()

            for detection in detections:
                track_id = detection.get("track_id")
                if track_id is None:
                    continue
                active_ids.add(track_id)
                center = detection["center"]
                zone_ids = {
                    zone["id"]
                    for zone in zones
                    if point_in_polygon(center, zone["polygon"])
                }
                detection["zone_ids"] = list(zone_ids)
                state = track_state.setdefault(
                    track_id,
                    {
                        "last_center": None,
                        "stationary_seconds": 0.0,
                        "inside": set(),
                        "loitered": set(),
                        "missing_frames": 0,
                    },
                )
                state["missing_frames"] = 0
                if state["last_center"] is not None:
                    dx = center[0] - state["last_center"][0]
                    dy = center[1] - state["last_center"][1]
                    if (dx * dx + dy * dy) ** 0.5 <= 3.0:
                        state["stationary_seconds"] += 1.0 / fps
                    else:
                        state["stationary_seconds"] = 0.0
                state["last_center"] = center

                for zone in zones:
                    zone_id = zone["id"]
                    inside = zone_id in zone_ids
                    was_inside = zone_id in state["inside"]
                    if config.get("intrusion_enabled", True) and inside and not was_inside:
                        logger.add_event(
                            frame_count, timestamp, detection["bbox"], track_id,
                            "zone_intrusion", detection["confidence"], zone_id,
                        )
                        labels[track_id] = "INTRUSION"
                    if inside and state["stationary_seconds"] >= float(
                        config.get("loitering_seconds", 5.0)
                    ):
                        if zone_id not in state["loitered"]:
                            logger.add_event(
                                frame_count, timestamp, detection["bbox"], track_id,
                                "loitering", detection["confidence"], zone_id,
                            )
                            state["loitered"].add(zone_id)
                        labels[track_id] = "LOITERING"
                    elif not inside:
                        state["loitered"].discard(zone_id)
                state["inside"] = zone_ids

            for track_id in set(track_state) - active_ids:
                state = track_state[track_id]
                state["missing_frames"] += 1
                if state["missing_frames"] > stale_limit:
                    del track_state[track_id]

            writer.write(annotate_frame(frame, detections, zones, labels))
    finally:
        cap.release()
        writer.release()

    if frame_count == 0:
        raise ValueError(f"Video produced no readable frames: {video_path}")

    events_path = output_dir / f"{video_path.stem}_events.json"
    logger.save_json(events_path)
    return {
        "frame_count": frame_count,
        "source_frames": source_frames,
        "fps": fps,
        "duration": frame_count / fps,
        "processing_seconds": time.perf_counter() - started,
        "event_count": len(logger.events),
        "output_video": str(output_video),
        "output_json": str(events_path),
    }
