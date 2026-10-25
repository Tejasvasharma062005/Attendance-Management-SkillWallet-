"""CLI entry point for the Multimodal Attendance & Engagement Agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.session_processor import SessionProcessor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run attendance and engagement analysis on video or webcam."
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to a recorded classroom video file.",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Capture from default webcam instead of a file.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Optional separate audio file for a --video session "
        "(otherwise the video's own audio track is used when readable).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Webcam capture duration in seconds (default: 30).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames processed (useful for quick tests).",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Optional session identifier for saved reports.",
    )
    args = parser.parse_args()

    processor = SessionProcessor(max_frames=args.max_frames)

    if args.webcam:
        report = processor.process_webcam(
            duration_seconds=args.duration,
            session_id=args.session_id,
        )
    elif args.video:
        report = processor.process_video(
            video_path=args.video,
            session_id=args.session_id,
            audio_path=args.audio,
        )
    else:
        parser.error("Provide --video <path> or --webcam")

    present = sum(1 for r in report.analysis.attendance if r.present)
    print(f"Session ID: {report.session_id}")
    print(f"Duration: {report.analysis.session_duration_seconds:.1f}s")
    print(f"Participants tracked: {len(report.analysis.attendance)}")
    print(f"Marked present: {present}")
    print(f"Session events detected: {len(report.analysis.events)}")
    print(f"Teaching suggestions: {len(report.teaching_suggestions)}")
    print(f"Report saved: {report.report_path}")


if __name__ == "__main__":
    main()
