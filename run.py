import argparse
import json
from pathlib import Path

from src.video_processor import (
    discover_videos,
    matching_zone_config,
    process_video,
)


def main():
    parser = argparse.ArgumentParser(
        description="Detect and track people, zones, and events in MP4 video."
    )
    parser.add_argument("--video", help="Input MP4 path")
    parser.add_argument("--all", action="store_true", help="Process all input MP4 files")
    parser.add_argument("--zones", help="Zone JSON for --video")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model weights")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    input_dir = root / "input"
    config_dir = root / "config"
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        video = Path(args.video)
        if not video.is_absolute():
            video = (root / video).resolve()
        if not video.is_file():
            print(f"Video not found: {video}")
            return 1
        if video.suffix.lower() != ".mp4":
            print(f"Unsupported video extension: {video.suffix}")
            return 1
        videos = [video]
    elif args.all:
        videos = discover_videos(input_dir)
    else:
        parser.error("provide --video or --all")

    if not videos:
        print("No MP4 files found in input/")
        return 1

    failures = 0
    for video in videos:
        zones = Path(args.zones) if args.zones else matching_zone_config(video, config_dir)
        if not zones.is_absolute():
            zones = (root / zones).resolve()
        if not zones.is_file():
            print(f"{video.name}: missing zone config {zones}")
            failures += 1
            continue
        try:
            stats = process_video(video, zones, output_dir, args.model)
            print(
                f"{video.name}: frames={stats['frame_count']}/{stats['source_frames']} "
                f"fps={stats['fps']:.2f} duration={stats['duration']:.2f}s "
                f"processing={stats['processing_seconds']:.2f}s "
                f"processed_fps={stats['processed_fps']:.2f} "
                f"tracks={stats['unique_track_count']} events={stats['event_count']}"
            )
            metrics_path = output_dir / f"{video.stem}_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "video": video.name,
                        "input_fps": stats["fps"],
                        "processed_fps": stats["processed_fps"],
                        "total_frames": stats["source_frames"],
                        "processed_frames": stats["frame_count"],
                        "processing_seconds": stats["processing_seconds"],
                        "unique_tracked_people": stats["unique_track_count"],
                        "intrusion_events": stats["intrusion_event_count"],
                        "loitering_events": stats["loitering_event_count"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if len(videos) == 1:
                (output_dir / "metrics.json").write_text(
                    metrics_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
        except Exception as exc:
            failures += 1
            print(f"{video.name}: FAILED: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
