"""Unit tests for teaching suggestion generation (Step 12 — AI suggestions)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.engagement_scorer import EngagementTimelinePoint
from src.agent.fusion import FusedInterval
from src.analytics.teaching_suggestions import TeachingSuggestion, generate_teaching_suggestions


def _point(t: float, score: float) -> EngagementTimelinePoint:
    from src.agent.engagement_scorer import classify_level
    return EngagementTimelinePoint(
        start_time=t,
        end_time=t + 10.0,
        score=score,
        level=classify_level(score),
    )


def _interval(start: float, score: float, noise: float = 0.0) -> FusedInterval:
    return FusedInterval(
        start_time=start,
        end_time=start + 10.0,
        visual_score=score,
        audio_score=score,
        combined_score=score,
        metadata={"avg_noise": noise, "sample_count_visual": 5.0, "sample_count_audio": 2.0},
    )


class TestGenerateTeachingSuggestions:
    def test_returns_at_least_one_suggestion(self):
        timeline = [_point(0.0, 0.7), _point(10.0, 0.6)]
        fused = [_interval(0.0, 0.7), _interval(10.0, 0.6)]
        suggestions = generate_teaching_suggestions(timeline, fused)
        assert len(suggestions) >= 1

    def test_general_suggestion_when_no_issues(self):
        timeline = [_point(i * 10.0, 0.75) for i in range(5)]
        fused = [_interval(i * 10.0, 0.75) for i in range(5)]
        suggestions = generate_teaching_suggestions(timeline, fused)
        categories = [s.category for s in suggestions]
        assert "general" in categories or "effective_segment" in categories

    def test_pacing_suggestion_on_sustained_low_engagement(self):
        """Three or more consecutive low intervals should trigger a pacing suggestion."""
        timeline = [_point(i * 10.0, 0.1) for i in range(4)]  # 4 low intervals
        fused = [_interval(i * 10.0, 0.1) for i in range(4)]
        suggestions = generate_teaching_suggestions(timeline, fused)
        categories = [s.category for s in suggestions]
        assert "pacing" in categories
        pacing = next(s for s in suggestions if s.category == "pacing")
        assert pacing.time_range == "0s–40s"

    def test_effective_segment_on_high_engagement_peak(self):
        timeline = [
            _point(0.0, 0.2),
            _point(10.0, 0.9),   # peak
            _point(20.0, 0.3),
        ]
        fused = [_interval(i * 10.0, [0.2, 0.9, 0.3][i]) for i in range(3)]
        suggestions = generate_teaching_suggestions(timeline, fused)
        categories = [s.category for s in suggestions]
        assert "effective_segment" in categories

    def test_classroom_management_on_high_noise_low_engagement(self):
        timeline = [_point(0.0, 0.3)]
        fused = [_interval(0.0, 0.3, noise=0.9)]  # high noise + low engagement
        suggestions = generate_teaching_suggestions(timeline, fused)
        categories = [s.category for s in suggestions]
        assert "classroom_management" in categories

    def test_suggestion_has_required_fields(self):
        timeline = [_point(0.0, 0.5)]
        fused = [_interval(0.0, 0.5)]
        for s in generate_teaching_suggestions(timeline, fused):
            assert s.category
            assert s.message
            assert s.time_range
            assert s.evidence

    def test_empty_inputs_returns_general_suggestion(self):
        suggestions = generate_teaching_suggestions([], [])
        assert len(suggestions) >= 1
        assert suggestions[0].category == "general"
