"""Unit tests for visual engagement scoring and head-pose signals."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.engagement_visual import VisualEngagementAnalyzer


class TestVisualEngagementAnalyzer:
    def test_attention_proxy_uses_head_pose_in_expected_proportions(self):
        analyzer = VisualEngagementAnalyzer()
        sample = analyzer.update(
            track_id=1,
            center=(320.0, 240.0),
            timestamp=0.0,
            head_pose_score=0.8,
        )

        assert sample.head_pose_score == 0.8
        assert sample.attention_proxy == 0.7

    def test_interval_metrics_include_average_head_pose(self):
        analyzer = VisualEngagementAnalyzer()
        samples = [
            analyzer.update(1, (320.0, 240.0), 0.0, head_pose_score=0.4),
            analyzer.update(1, (320.0, 240.0), 1.0, head_pose_score=0.8),
        ]

        metrics = analyzer.session_interval_metrics(samples, 0.0, 2.0)

        assert abs(metrics["avg_head_pose"] - 0.6) < 1e-9
