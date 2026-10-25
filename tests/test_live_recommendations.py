"""Trend-based, group-level live suggestions."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics.live_recommendations import generate_live_recommendations


def _history(values, step=0.5):
    return [{"time": i * step, "attention": v} for i, v in enumerate(values)]


def _kinds(recs):
    return [r.kind for r in recs]


def test_sustained_low_attention_warns():
    recs = generate_live_recommendations(_history([0.3] * 60), 30.0, 3)
    assert _kinds(recs) == ["warning"] and "%" in recs[0].evidence


def test_single_low_frame_does_not_warn():
    recs = generate_live_recommendations(_history([0.8] * 59 + [0.1]), 30.0, 3)
    assert "warning" not in _kinds(recs)


def test_strong_attention_is_acknowledged():
    assert _kinds(generate_live_recommendations(_history([0.85] * 60), 30.0, 3)) == ["success"]


def test_sudden_dip_is_flagged_as_info():
    recs = generate_live_recommendations(_history([0.85] * 20 + [0.4] * 20), 20.0, 3)
    assert _kinds(recs) == ["info"] and "dipped" in recs[0].message


def test_too_little_data_does_not_trigger_judgements():
    recs = generate_live_recommendations(_history([0.2] * 6), 3.0, 3)
    assert "warning" not in _kinds(recs)


def test_no_faces_suggests_checking_the_camera():
    assert "camera" in generate_live_recommendations([], 10.0, 0)[0].message.lower()


def test_messages_never_single_out_an_individual():
    for values in ([0.3] * 60, [0.85] * 60, [0.85] * 20 + [0.4] * 20):
        for rec in generate_live_recommendations(_history(values), 20.0, 3):
            assert "participant" not in rec.message.lower() and "student" not in rec.message.lower()
