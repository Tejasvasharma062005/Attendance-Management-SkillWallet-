"""Per-participant timelines and focus highlights (no ranking)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.engagement_scorer import EngagementTimelinePoint
from src.analytics.student_analytics import generate_student_analytics
from src.vision.attendance import AttendanceRecord
from src.vision.engagement_visual import VisualEngagementSample


def _samples(track_id, spans):
    out = []
    for (start, end, attention) in spans:
        out += [VisualEngagementSample(float(t), track_id, 0.1, 0.0, 0.9, attention) for t in range(start, end)]
    return out


TIMELINE = [EngagementTimelinePoint(i * 10.0, i * 10.0 + 10.0, 0.5, "medium") for i in range(6)]


def test_timeline_has_one_point_per_interval_the_participant_was_visible():
    record = AttendanceRecord(1, 40.0, True)
    samples = _samples(1, [(0, 20, 0.8), (40, 60, 0.5)])   # absent during 20-40 s
    summary = generate_student_analytics([record], samples, TIMELINE)[0]
    starts = [p["start_time"] for p in summary.timeline]
    assert starts == [0.0, 10.0, 40.0, 50.0]
    assert summary.timeline[0]["attention"] == 0.8


def test_highlights_name_the_longest_strong_and_weak_stretches():
    record = AttendanceRecord(1, 60.0, True)
    samples = _samples(1, [(0, 30, 0.9), (30, 40, 0.5), (40, 60, 0.2)])
    text = generate_student_analytics([record], samples, TIMELINE)[0].focus_highlights
    assert "between 0s and 30s" in text and "between 40s and 60s" in text


def test_highlight_does_not_span_a_gap_between_separate_strong_periods():
    record = AttendanceRecord(1, 60.0, True)
    samples = _samples(1, [(0, 10, 0.9), (10, 50, 0.5), (50, 60, 0.9)])
    text = generate_student_analytics([record], samples, TIMELINE)[0].focus_highlights
    assert "between 0s and 60s" not in text                # old min/max logic claimed this


def test_each_participant_only_gets_their_own_samples():
    records = [AttendanceRecord(1, 30.0, True), AttendanceRecord(2, 30.0, True)]
    samples = _samples(1, [(0, 30, 0.9)]) + _samples(2, [(0, 30, 0.3)])
    one, two = generate_student_analytics(records, samples, TIMELINE)
    assert one.avg_attention == 0.9 and two.avg_attention == 0.3
