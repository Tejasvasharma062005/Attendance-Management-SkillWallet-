"""Unit tests for engagement scoring (Step 11 — scores and session trends)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.engagement_scorer import (
    EngagementTimelinePoint,
    build_engagement_timeline,
    classify_level,
)
from src.agent.fusion import FusedInterval


def _make_interval(start: float, end: float, score: float) -> FusedInterval:
    return FusedInterval(
        start_time=start,
        end_time=end,
        visual_score=score,
        audio_score=score,
        combined_score=score,
        metadata={},
    )


class TestClassifyLevel:
    def test_high_at_boundary(self):
        assert classify_level(0.65) == "high"

    def test_high_above_boundary(self):
        assert classify_level(1.0) == "high"

    def test_medium_at_lower_boundary(self):
        assert classify_level(0.35) == "medium"

    def test_medium_just_below_high(self):
        assert classify_level(0.64) == "medium"

    def test_low_below_medium(self):
        assert classify_level(0.34) == "low"

    def test_low_at_zero(self):
        assert classify_level(0.0) == "low"


class TestBuildEngagementTimeline:
    def test_empty_intervals_returns_empty_timeline(self):
        assert build_engagement_timeline([]) == []

    def test_timeline_length_matches_intervals(self):
        intervals = [_make_interval(i * 10.0, (i + 1) * 10.0, 0.5) for i in range(5)]
        timeline = build_engagement_timeline(intervals)
        assert len(timeline) == 5

    def test_timeline_scores_match_combined_scores(self):
        intervals = [
            _make_interval(0.0, 10.0, 0.8),
            _make_interval(10.0, 20.0, 0.4),
            _make_interval(20.0, 30.0, 0.1),
        ]
        timeline = build_engagement_timeline(intervals)
        assert timeline[0].score == 0.8
        assert timeline[1].score == 0.4
        assert timeline[2].score == 0.1

    def test_timeline_levels_are_correct(self):
        intervals = [
            _make_interval(0.0, 10.0, 0.9),   # high
            _make_interval(10.0, 20.0, 0.5),  # medium
            _make_interval(20.0, 30.0, 0.1),  # low
        ]
        timeline = build_engagement_timeline(intervals)
        assert timeline[0].level == "high"
        assert timeline[1].level == "medium"
        assert timeline[2].level == "low"

    def test_time_ranges_are_preserved(self):
        intervals = [_make_interval(5.0, 15.0, 0.7)]
        timeline = build_engagement_timeline(intervals)
        assert timeline[0].start_time == 5.0
        assert timeline[0].end_time == 15.0
