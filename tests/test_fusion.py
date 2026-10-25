"""Unit tests for multimodal fusion (Step 10 — combine visual + audio data)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.agent.fusion import FusedInterval, fuse_intervals
from src.audio.analyzer import AudioWindowMetrics
from src.vision.engagement_visual import VisualEngagementSample


def _visual(t: float, attention: float = 0.8, track_id: int = 1) -> VisualEngagementSample:
    return VisualEngagementSample(
        timestamp=t,
        track_id=track_id,
        movement_score=0.1,
        inactive_seconds=0.0,
        head_pose_score=0.9,
        attention_proxy=attention,
    )


def _audio(start: float, end: float, speech: float = 0.6) -> AudioWindowMetrics:
    return AudioWindowMetrics(
        start_time=start,
        end_time=end,
        rms_energy=0.05,
        noise_level=0.3,
        speech_activity=speech,
        is_silent=False,
    )


class TestFuseIntervals:
    def test_empty_inputs_returns_empty(self):
        result = fuse_intervals([], [], session_duration=30.0)
        assert result == []

    def test_number_of_intervals_covers_session(self):
        duration = 30.0
        samples = [_visual(t) for t in range(0, 30, 2)]
        result = fuse_intervals(samples, [], session_duration=duration)
        total_time = sum(iv.end_time - iv.start_time for iv in result)
        assert abs(total_time - duration) < 0.01

    def test_visual_only_audio_score_is_zero(self):
        samples = [_visual(t) for t in range(0, 10)]
        result = fuse_intervals(samples, [], session_duration=10.0)
        for iv in result:
            assert iv.audio_score == 0.0

    def test_audio_only_visual_score_is_zero(self):
        metrics = [_audio(0.0, 10.0)]
        result = fuse_intervals([], metrics, session_duration=10.0)
        for iv in result:
            assert iv.visual_score == 0.0

    def test_combined_score_respects_weights(self):
        """Combined = WEIGHT_VISUAL * visual + WEIGHT_AUDIO * audio."""
        samples = [_visual(t, attention=1.0) for t in range(0, 10)]
        metrics = [_audio(0.0, 10.0, speech=1.0)]
        result = fuse_intervals(samples, metrics, session_duration=10.0)
        expected = config.WEIGHT_VISUAL * 1.0 + config.WEIGHT_AUDIO * 1.0
        for iv in result:
            assert abs(iv.combined_score - expected) < 0.05

    def test_weights_sum_to_one(self):
        assert abs(config.WEIGHT_VISUAL + config.WEIGHT_AUDIO - 1.0) < 1e-9

    def test_intervals_are_contiguous(self):
        samples = [_visual(t) for t in range(0, 20, 2)]
        result = fuse_intervals(samples, [], session_duration=20.0)
        for i in range(len(result) - 1):
            assert abs(result[i].end_time - result[i + 1].start_time) < 1e-9

    def test_metadata_keys_present(self):
        samples = [_visual(0.0)]
        result = fuse_intervals(samples, [], session_duration=10.0)
        for iv in result:
            assert "avg_noise" in iv.metadata
            assert "head_pose_score" in iv.metadata
            assert "sample_count_visual" in iv.metadata
            assert "sample_count_audio" in iv.metadata

    def test_head_pose_average_is_preserved_in_interval_metadata(self):
        samples = [_visual(0.0), _visual(1.0)]
        samples[0].head_pose_score = 0.4
        samples[1].head_pose_score = 0.8
        result = fuse_intervals(samples, [], session_duration=10.0)
        assert result[0].metadata["head_pose_score"] == 0.6
