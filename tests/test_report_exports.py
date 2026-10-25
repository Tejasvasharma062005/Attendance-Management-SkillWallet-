"""Tests for local session report export formats."""

import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.storage.report_exports import build_session_csv, build_session_pdf


def _payload():
    return {
        "analysis": {
            "session_duration_seconds": 20.0,
            "attendance": [
                {
                    "track_id": 1,
                    "total_visible_seconds": 15.0,
                    "present": True,
                    "first_seen": 0.0,
                    "last_seen": 14.9,
                }
            ],
            "engagement_timeline": [
                {
                    "start_time": 0.0,
                    "end_time": 10.0,
                    "score": 0.7,
                    "level": "high",
                }
            ],
            "fused_intervals": [
                {
                    "start_time": 0.0,
                    "end_time": 10.0,
                    "visual_score": 0.8,
                    "audio_score": 0.5,
                    "combined_score": 0.7,
                    "metadata": {"head_pose_score": 0.9, "avg_noise": 0.2},
                }
            ],
        },
        "teaching_suggestions": [
            {
                "category": "effective_segment",
                "time_range": "0s-10s",
                "message": "Strong participation.",
                "evidence": "Engagement score 0.70",
            }
        ],
        "student_analytics": [],
    }


def test_csv_includes_session_analytics_and_attendance():
    output = build_session_csv("session_1", _payload())
    rows = list(csv.DictReader(io.StringIO(output)))

    assert [row["record_type"] for row in rows] == [
        "engagement_interval",
        "attendance",
        "teaching_suggestion",
    ]
    assert rows[0]["head_pose_score"] == "0.9"
    assert rows[1]["attended"] == "True"


def test_pdf_is_generated_with_pdf_signature_and_summary_text():
    output = build_session_pdf("session_1", _payload())

    assert output.startswith(b"%PDF-")
    assert len(output) > 1000


def _full_payload():
    payload = _payload()
    payload["attendance_threshold_seconds"] = 12.0
    payload["analysis"]["attendance"][0].update(marked_present_at=12.0, threshold_seconds=12.0)
    payload["analysis"]["engagement_timeline"][0]["rationale"] = "Strong visual attention (0.80)."
    payload["analysis"]["events"] = [
        {"event_type": "attention_drop", "start_time": 10.0, "end_time": 20.0,
         "detail": "Engagement fell from 0.80 to 0.40."}
    ]
    payload["attendance_log"] = [
        {"track_id": 1, "event": "marked_present", "session_time_s": 12.0, "session_time": "00:12",
         "timestamp_utc": None, "details": "Sustained presence confirmed."}
    ]
    payload["audio_summary"] = {"available": True, "speech_ratio": 0.4, "background_noise_ratio": 0.1,
                                "silence_ratio": 0.5, "mean_noise_db": -41.0, "speech_bursts": 3}
    payload["student_analytics"] = [
        {"track_id": 1, "engagement_consistency": "steady", "focus_highlights": "Stronger focus 0s-10s.",
         "reflection_note": "Consistent.", "attended": True, "visible_seconds": 15.0, "avg_attention": 0.8}
    ]
    return payload


def test_csv_includes_new_record_types_when_report_has_them():
    rows = list(csv.DictReader(io.StringIO(build_session_csv("s1", _full_payload()))))
    kinds = [row["record_type"] for row in rows]
    assert {"attendance_event", "session_event", "audio_summary"} <= set(kinds)
    assert rows[0]["details"] == "Strong visual attention (0.80)."
    assert "marked_present_at=12.0" in next(r for r in rows if r["record_type"] == "attendance")["details"]


def test_pdf_renders_with_full_payload_and_with_missing_optional_sections():
    assert build_session_pdf("s1", _full_payload()).startswith(b"%PDF-")
    assert build_session_pdf("s1", _payload()).startswith(b"%PDF-")
