"""The saved report carries the attendance log, audio summary and threshold."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.multimodal_agent import MultimodalAgent
from src.audio.analyzer import AudioWindowMetrics
from src.pipeline.payload import build_session_payload
from src.storage.logs import _serialize
from src.vision.attendance import PresenceAttendanceEngine
from src.vision.engagement_visual import VisualEngagementSample


def _payload(audio, started_at=None):
    engine = PresenceAttendanceEngine()
    samples = []
    for i in range(130):
        engine.update([1], i / 10, 0.1)
        samples.append(VisualEngagementSample(i / 10, 1, 0.1, 0.0, 0.9, 0.8))
    attendance = engine.finalize(13.0)
    analysis = MultimodalAgent().analyze(attendance, samples, audio, 13.0)
    return build_session_payload(
        session_id="s1", source_fields={"source": "webcam"}, analysis=analysis, suggestions=[],
        student_analytics=[], audio_metrics=audio, session_duration_seconds=13.0,
        attendance_threshold_seconds=engine.effective_threshold(13.0), session_started_at=started_at,
    )


def test_payload_contains_new_report_sections_and_is_json_serialisable():
    import json
    audio = [AudioWindowMetrics(0.0, 1.0, 0.05, 0.5, 0.7, False, noise_db=-26.0, activity="speech")]
    payload = _payload(audio, "2026-10-25T10:00:00+00:00")
    data = json.loads(json.dumps(_serialize(payload)))
    assert data["attendance_threshold_seconds"] == 7.8
    assert {e["event"] for e in data["attendance_log"]} == {"first_seen", "marked_present", "last_seen"}
    assert data["attendance_log"][0]["timestamp_utc"] == "2026-10-25T10:00:00+00:00"
    assert data["audio_summary"]["available"] is True
    assert data["audio_windows"][0]["activity"] == "speech"
    assert data["analysis"]["engagement_timeline"][0]["rationale"]


def test_payload_without_audio_marks_summary_unavailable_and_no_clock_times():
    payload = _payload([])
    assert payload["audio_summary"] == {"available": False}
    assert payload["session_started_at"] is None
    assert all(e["timestamp_utc"] is None for e in payload["attendance_log"])
