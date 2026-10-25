"""Tests for in-memory microphone capture."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio import live_capture


class _FakeInputStream:
    def __init__(self, *, callback, **_kwargs):
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True
        self.callback(
            np.array([[0.1], [0.2], [-0.1]], dtype=np.float32),
            3,
            None,
            "input overflow",
        )

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_live_audio_capture_buffers_samples_and_surfaces_device_notices(monkeypatch):
    stream = _FakeInputStream(callback=lambda *_args: None)
    monkeypatch.setattr(
        live_capture.sd,
        "InputStream",
        lambda **kwargs: _create_stream(stream, kwargs),
    )

    capture = live_capture.LiveAudioCapture(sample_rate=22050)
    waveform, notices = capture.stop()

    assert np.array_equal(waveform, np.array([0.1, 0.2, -0.1], dtype=np.float32))
    assert notices == ["input overflow"]
    assert stream.started
    assert stream.stopped
    assert stream.closed
    repeated_waveform, repeated_notices = capture.stop()
    assert np.array_equal(repeated_waveform, waveform)
    assert repeated_notices == notices


def _create_stream(stream, kwargs):
    stream.callback = kwargs["callback"]
    return stream
