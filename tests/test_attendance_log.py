"""Effective presence threshold, confirmation time, and timestamped attendance log."""

import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.storage.attendance_log import (
    attendance_log_to_csv,
    build_attendance_log,
    format_session_time,
)
from src.vision.attendance import PresenceAttendanceEngine


def _run(engine, track_id, seconds, fps=10.0, start=0.0):
    for i in range(int(seconds * fps)):
        engine.update([track_id], start + i / fps, 1.0 / fps)


def test_short_session_scales_threshold_so_attendance_is_possible():
    engine = PresenceAttendanceEngine()                    # standard 30 s rule
    _run(engine, 1, 13.0)                                  # visible for the whole 13 s session
    assert engine.finalize()[0].present is False           # fixed rule: impossible
    record = engine.finalize(session_duration=13.0)[0]
    assert record.present is True
    assert 7.0 < record.threshold_seconds < 8.0


def test_long_session_keeps_the_configured_threshold():
    engine = PresenceAttendanceEngine(threshold_seconds=30.0)
    assert engine.effective_threshold(600.0) == 30.0
    assert engine.effective_threshold(None) == 30.0


def test_brief_visit_in_short_session_is_still_not_present():
    engine = PresenceAttendanceEngine()
    _run(engine, 1, 2.0)                                   # 2 s of a 13 s session
    assert engine.finalize(session_duration=13.0)[0].present is False


def test_marked_present_at_is_when_the_threshold_was_reached():
    engine = PresenceAttendanceEngine(threshold_seconds=5.0)
    _run(engine, 1, 4.0, start=0.0)
    _run(engine, 1, 6.0, start=10.0)                       # absent 4-10 s, then returns
    record = engine.finalize()[0]
    assert record.present is True
    assert 11.0 <= record.marked_present_at <= 11.5        # 4 s + 1 s into the return


def test_marked_present_at_is_none_when_not_present():
    engine = PresenceAttendanceEngine(threshold_seconds=30.0)
    _run(engine, 1, 3.0)
    assert engine.finalize()[0].marked_present_at is None


def test_log_has_chronological_events_with_session_times():
    engine = PresenceAttendanceEngine(threshold_seconds=5.0)
    _run(engine, 1, 8.0)
    _run(engine, 2, 2.0, start=3.0)
    events = build_attendance_log(engine.finalize(), "s1")
    kinds = [(e["participant"], e["event"]) for e in events]
    assert ("Participant 1", "marked_present") in kinds
    assert ("Participant 2", "marked_present") not in kinds
    times = [e["session_time_s"] for e in events]
    assert times == sorted(times)
    assert all(e["timestamp_utc"] is None for e in events)  # no clock for recorded media


def test_log_adds_clock_times_only_when_session_start_is_known():
    engine = PresenceAttendanceEngine(threshold_seconds=5.0)
    _run(engine, 1, 8.0)
    events = build_attendance_log(engine.finalize(), "s1", "2026-10-25T10:00:00+00:00")
    first = next(e for e in events if e["event"] == "first_seen")
    assert first["timestamp_utc"] == "2026-10-25T10:00:00+00:00"


def test_log_accepts_old_report_dicts_without_inventing_confirmation_time():
    old = [{"track_id": 1, "total_visible_seconds": 40.0, "present": True,
            "first_seen": 0.0, "last_seen": 39.9}]
    events = build_attendance_log(old, "old")
    assert [e["event"] for e in events] == ["first_seen", "last_seen"]


def test_log_csv_round_trips():
    engine = PresenceAttendanceEngine(threshold_seconds=2.0)
    _run(engine, 1, 5.0)
    text = attendance_log_to_csv(build_attendance_log(engine.finalize(), "s1"))
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["session_id"] == "s1"
    assert {r["event"] for r in rows} == {"first_seen", "marked_present", "last_seen"}


def test_format_session_time():
    assert format_session_time(75) == "01:15"
    assert format_session_time(3725) == "1:02:05"
    assert format_session_time(None) == ""
