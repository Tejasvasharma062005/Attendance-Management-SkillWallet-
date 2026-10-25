"""Silence / background-noise / speech classification and session audio summary."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio import extractor as extractor_module
from src.audio.analyzer import (
    ACTIVITY_NOISE,
    ACTIVITY_SILENCE,
    ACTIVITY_SPEECH,
    AudioAnalyzer,
    AudioWindowMetrics,
    summarize_audio,
)
from src.audio.extractor import AudioExtractor, AudioSegment

SR = 22050
T = np.arange(SR) / SR


def _voice(f0=120.0, amp=0.15):
    signal = sum(np.sin(2 * np.pi * f0 * k * T) / k for k in range(1, 25))
    signal = signal * np.clip(np.sin(2 * np.pi * 4.0 * T) + 0.3, 0.0, None)
    return (amp * signal / np.abs(signal).max()).astype(np.float32)


def _classify(samples):
    return AudioAnalyzer().analyze_segment(AudioSegment(0.0, 1.0, samples, SR))


def test_quiet_input_is_silence():
    rng = np.random.default_rng(0)
    metrics = _classify((0.001 * rng.standard_normal(SR)).astype(np.float32))
    assert metrics.activity == ACTIVITY_SILENCE and metrics.is_silent
    assert metrics.speech_activity == 0.0


def test_voiced_modulated_audio_is_speech_for_low_and_high_voices():
    for f0 in (110.0, 220.0):
        metrics = _classify(_voice(f0))
        assert metrics.activity == ACTIVITY_SPEECH
        assert metrics.speech_activity > 0.5


def test_speech_with_moderate_background_noise_is_still_speech():
    rng = np.random.default_rng(1)
    noisy = _voice(150.0) + (0.03 * rng.standard_normal(SR)).astype(np.float32)
    assert _classify(noisy).activity == ACTIVITY_SPEECH


def test_loud_steady_noise_hum_tone_and_dc_are_not_speech():
    rng = np.random.default_rng(2)
    steady = {
        "white noise": (0.1 * rng.standard_normal(SR)).astype(np.float32),
        "mains hum": (0.1 * np.sin(2 * np.pi * 50 * T)).astype(np.float32),
        "steady tone": (0.1 * np.sin(2 * np.pi * 1000 * T)).astype(np.float32),
        "dc offset": np.full(SR, 0.1, dtype=np.float32),
    }
    for name, samples in steady.items():
        metrics = _classify(samples)
        assert metrics.activity == ACTIVITY_NOISE, name
        assert metrics.speech_activity == 0.0, name
        assert metrics.noise_level > 0.5, name


def test_noise_db_is_reported_in_dbfs():
    metrics = _classify(np.full(SR, 0.1, dtype=np.float32))
    assert abs(metrics.noise_db - (-20.0)) < 0.1


def test_tiny_trailing_window_does_not_crash():
    metrics = AudioAnalyzer().analyze_segment(AudioSegment(0.0, 0.0001, np.array([0.5, 0.5], np.float32), SR))
    assert metrics.activity in (ACTIVITY_NOISE, ACTIVITY_SILENCE, ACTIVITY_SPEECH)


def _window(start, activity, db=-40.0):
    return AudioWindowMetrics(start, start + 1.0, 0.01, 0.5, 1.0 if activity == "speech" else 0.0,
                              activity == "silence", noise_db=db, activity=activity)


def test_summary_is_unavailable_without_audio():
    assert summarize_audio([]) == {"available": False}


def test_summary_counts_shares_bursts_and_transitions():
    labels = ["speech"] * 3 + ["silence"] * 4 + ["speech"] * 2 + ["background_noise"]
    summary = summarize_audio([_window(float(i), a) for i, a in enumerate(labels)])
    assert summary["available"] is True
    assert summary["speech_ratio"] == 0.5
    assert summary["silence_ratio"] == 0.4
    assert summary["background_noise_ratio"] == 0.1
    assert summary["speech_bursts"] == 2
    assert summary["longest_speech_seconds"] == 3.0
    assert summary["longest_silence_seconds"] == 4.0
    assert summary["speech_transitions"] == 3


def test_ffmpeg_fallback_decodes_video_audio_when_librosa_cannot(monkeypatch, tmp_path):
    if shutil.which("ffmpeg") is None:
        return                                              # ffmpeg is optional
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-shortest",
         "-c:v", "mpeg4", "-c:a", "aac", str(video)],
        check=True,
    )

    def _cannot_open(*_a, **_k):
        raise RuntimeError("no decoder back-end")

    monkeypatch.setattr(extractor_module.librosa, "load", _cannot_open)
    loaded = AudioExtractor().from_video(video)
    assert loaded is not None
    waveform, rate = loaded
    assert rate == AudioExtractor().sample_rate
    assert 1.8 < len(waveform) / rate < 2.3
    assert np.sqrt(np.mean(waveform**2)) > 0.05
    assert list(tmp_path.glob("*.wav")) == []               # decoded in memory only


def test_video_without_audio_track_returns_none(monkeypatch, tmp_path):
    if shutil.which("ffmpeg") is None:
        return
    video = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
         "-c:v", "mpeg4", str(video)],
        check=True,
    )
    monkeypatch.setattr(extractor_module.librosa, "load",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no audio")))
    assert AudioExtractor().from_video(video) is None
