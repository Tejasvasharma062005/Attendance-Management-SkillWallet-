"""Fusion must not penalise intervals that simply have no audio."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.agent.engagement_scorer import classify_level
from src.agent.fusion import fuse_intervals
from src.audio.analyzer import AudioWindowMetrics
from src.vision.engagement_visual import VisualEngagementSample


def _visual(t, attention=0.8):
    return VisualEngagementSample(t, 1, 0.1, 0.0, 0.9, attention)


def _audio(start, end, speech=0.6, activity="speech"):
    return AudioWindowMetrics(start, end, 0.05, 0.3, speech, False, noise_db=-26.0, activity=activity)


def test_attentive_class_without_audio_can_reach_high_engagement():
    samples = [_visual(t, 0.8) for t in range(10)]
    interval = fuse_intervals(samples, [], session_duration=10.0)[0]
    assert interval.combined_score == 0.8                  # was capped at 0.6 * 0.8 = 0.48
    assert classify_level(interval.combined_score) == "high"


def test_intermittent_audio_only_renormalises_intervals_without_audio():
    samples = [_visual(t, 0.8) for t in range(20)]
    metrics = [_audio(0.0, 10.0, speech=0.5)]              # audio drops out after 10 s
    first, second = fuse_intervals(samples, metrics, session_duration=20.0)
    assert abs(first.combined_score - (0.6 * 0.8 + 0.4 * 0.5)) < 1e-6
    assert second.combined_score == 0.8
    assert second.audio_score == 0.0


def test_silent_audio_still_counts_as_audio_evidence():
    samples = [_visual(t, 0.8) for t in range(10)]
    metrics = [_audio(0.0, 10.0, speech=0.0, activity="silence")]
    interval = fuse_intervals(samples, metrics, session_duration=10.0)[0]
    assert abs(interval.combined_score - config.WEIGHT_VISUAL * 0.8) < 1e-6


def test_short_trailing_audio_window_is_weighted_by_overlap():
    samples = [_visual(t, 0.5) for t in range(10)]
    metrics = [_audio(0.0, 9.9, speech=0.0, activity="silence"), _audio(9.9, 10.0, speech=1.0)]
    interval = fuse_intervals(samples, metrics, session_duration=10.0)[0]
    assert interval.audio_score < 0.02                     # a 0.1 s window must not count like a full one


def test_interval_metadata_exposes_noise_db_speech_fraction_and_inactivity():
    samples = [_visual(t) for t in range(10)]
    samples[3].inactive_seconds = 9.0
    metrics = [_audio(0.0, 5.0), _audio(5.0, 10.0, speech=0.0, activity="silence")]
    meta = fuse_intervals(samples, metrics, session_duration=10.0)[0].metadata
    assert meta["avg_noise_db"] == -26.0
    assert abs(meta["speech_fraction"] - 0.5) < 1e-6
    assert meta["max_inactive_seconds"] == 9.0
    assert "avg_movement" in meta


def test_no_audio_keys_in_metadata_when_no_audio():
    meta = fuse_intervals([_visual(0.0)], [], session_duration=10.0)[0].metadata
    assert "avg_noise_db" not in meta and "speech_fraction" not in meta
