"""Explainable scoring, adjustable thresholds, and session-dynamics events."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.agent.engagement_scorer import build_engagement_timeline, classify_level, explain_interval
from src.agent.events import ATTENTION_DROP, INTERACTION_PEAK, PACING_DECLINE, detect_session_events
from src.agent.fusion import FusedInterval
from src.agent.multimodal_agent import MultimodalAgent
from src.analytics.teaching_suggestions import generate_teaching_suggestions


def _interval(i, combined, audio=0.0, has_audio=True, head=0.9, noise=0.2, speech=None, inactive=0.0):
    meta = {
        "avg_noise": noise, "head_pose_score": head, "max_inactive_seconds": inactive,
        "sample_count_visual": 5.0, "sample_count_audio": 3.0 if has_audio else 0.0,
    }
    if has_audio:
        meta["speech_fraction"] = audio if speech is None else speech
    return FusedInterval(i * 10.0, i * 10.0 + 10.0, combined, audio, combined, meta)


def _timeline(scores, **kw):
    fused = [_interval(i, s, **kw) for i, s in enumerate(scores)]
    return build_engagement_timeline(fused), fused


def test_level_thresholds_are_adjustable_through_config(monkeypatch):
    assert classify_level(0.5) == "medium"
    monkeypatch.setattr(config, "ENGAGEMENT_MEDIUM_THRESHOLD", 0.6)
    assert classify_level(0.5) == "low"


def test_every_timeline_point_has_a_rationale_citing_its_signals():
    timeline, _ = _timeline([0.8, 0.3])
    assert all(point.rationale.endswith(".") for point in timeline)
    assert "Strong visual attention" in timeline[0].rationale
    assert "Weak visual attention" in timeline[1].rationale


def test_rationale_says_when_scored_on_visual_signals_only():
    text = explain_interval(_interval(0, 0.7, has_audio=False))
    assert "no audio" in text.lower() and "visual signals only" in text


def test_rationale_mentions_turned_heads_and_stillness():
    text = explain_interval(_interval(0, 0.4, head=0.3, inactive=12.0))
    assert "turned away" in text and "12s" in text


def test_loud_speech_is_not_described_as_background_noise():
    loud_speech = explain_interval(_interval(0, 0.8, audio=0.9, noise=0.95))
    loud_noise = explain_interval(_interval(0, 0.4, audio=0.0, noise=0.95))
    assert "background noise" not in loud_speech
    assert "background noise" in loud_noise


def test_attention_drop_is_detected():
    timeline, fused = _timeline([0.8, 0.75, 0.4, 0.4])
    drops = [e for e in detect_session_events(timeline, fused) if e.event_type == ATTENTION_DROP]
    assert len(drops) == 1
    assert (drops[0].start_time, drops[0].end_time) == (20.0, 30.0)


def test_small_fluctuations_are_not_drops():
    timeline, fused = _timeline([0.8, 0.7, 0.75, 0.7])
    assert not [e for e in detect_session_events(timeline, fused) if e.event_type == ATTENTION_DROP]


def test_consecutive_speech_intervals_form_one_interaction_peak():
    fused = [_interval(0, 0.5, audio=0.1), _interval(1, 0.6, audio=0.8),
             _interval(2, 0.6, audio=0.9), _interval(3, 0.5, audio=0.1)]
    timeline = build_engagement_timeline(fused)
    peaks = [e for e in detect_session_events(timeline, fused) if e.event_type == INTERACTION_PEAK]
    assert len(peaks) == 1 and (peaks[0].start_time, peaks[0].end_time) == (10.0, 30.0)
    assert peaks[0].magnitude == 0.9


def test_missing_audio_never_counts_as_an_interaction_peak():
    timeline, fused = _timeline([0.8] * 4, has_audio=False)
    assert not [e for e in detect_session_events(timeline, fused) if e.event_type == INTERACTION_PEAK]


def test_pacing_decline_needs_enough_intervals_and_a_real_fall():
    timeline, fused = _timeline([0.9, 0.9, 0.8, 0.7, 0.5, 0.5])
    assert [e for e in detect_session_events(timeline, fused) if e.event_type == PACING_DECLINE]
    short_tl, short_fused = _timeline([0.9, 0.5])
    assert not [e for e in detect_session_events(short_tl, short_fused) if e.event_type == PACING_DECLINE]
    flat_tl, flat_fused = _timeline([0.7] * 6)
    assert not [e for e in detect_session_events(flat_tl, flat_fused) if e.event_type == PACING_DECLINE]


def test_agent_attaches_events_to_the_analysis():
    from src.vision.engagement_visual import VisualEngagementSample
    samples = [VisualEngagementSample(float(t), 1, 0.1, 0.0, 0.9, 0.9 if t < 20 else 0.3) for t in range(40)]
    result = MultimodalAgent().analyze([], samples, [], 40.0)
    assert any(e.event_type == ATTENTION_DROP for e in result.events)


def test_suggestions_cover_attention_drop_and_interaction_categories():
    fused = [_interval(0, 0.8, audio=0.9), _interval(1, 0.8, audio=0.9), _interval(2, 0.3), _interval(3, 0.3)]
    timeline = build_engagement_timeline(fused)
    categories = {s.category for s in generate_teaching_suggestions(timeline, fused)}
    assert {"attention_drop", "interaction"} <= categories


def test_pacing_decline_suggestion_is_not_duplicated_after_a_low_streak():
    timeline, fused = _timeline([0.9, 0.9, 0.2, 0.2, 0.2, 0.2])
    pacing = [s for s in generate_teaching_suggestions(timeline, fused) if s.category == "pacing"]
    assert len(pacing) == 1


def test_noise_suggestion_distinguishes_non_speech_noise_from_side_conversation():
    quiet_room = [_interval(0, 0.2, noise=0.9, speech=0.0)]
    chatter = [_interval(0, 0.2, noise=0.9, speech=0.9, audio=0.1)]
    a = next(s for s in generate_teaching_suggestions(build_engagement_timeline(quiet_room), quiet_room)
             if s.category == "classroom_management")
    b = next(s for s in generate_teaching_suggestions(build_engagement_timeline(chatter), chatter)
             if s.category == "classroom_management")
    assert "not mostly speech" in a.message and "overlapping speech" in b.message


def test_consecutive_noisy_intervals_become_one_suggestion_range():
    fused = [_interval(i, 0.2, noise=0.9, speech=0.0) for i in range(3)]
    mgmt = [s for s in generate_teaching_suggestions(build_engagement_timeline(fused), fused)
            if s.category == "classroom_management"]
    assert len(mgmt) == 1 and mgmt[0].time_range == "0s–30s"
