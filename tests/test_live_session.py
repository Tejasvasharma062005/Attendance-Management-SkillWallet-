"""Tests for incremental webcam-session processing."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import live_session
from src.vision.face_tracker import TrackedFace


class _FakeCapture:
    def __init__(self, _camera_index):
        self.opened = True
        self.released = False
        self.frames = [
            np.zeros((120, 160, 3), dtype=np.uint8),
            np.zeros((120, 160, 3), dtype=np.uint8),
        ]

    def isOpened(self):
        return self.opened

    def get(self, _property):
        return 25.0

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self):
        self.released = True


class _FakeTracker:
    closed = False

    def process_frame(self, _frame, timestamp):
        return [TrackedFace(1, (40, 25, 40, 50), (60.0, 50.0), timestamp, 0.95)]

    def close(self):
        self.closed = True


def _speech_like_waveform(sample_rate: int = 22050) -> np.ndarray:
    """One second of voiced, syllable-modulated audio (a DC offset is not speech)."""
    t = np.arange(sample_rate) / sample_rate
    voiced = sum(np.sin(2 * np.pi * 120.0 * k * t) / k for k in range(1, 25))
    syllables = np.clip(np.sin(2 * np.pi * 4.0 * t) + 0.3, 0.0, None)
    signal = voiced * syllables
    return (0.15 * signal / np.abs(signal).max()).astype(np.float32)


class _FakeHeadPose:
    closed = False

    def estimate_frame(self, _frame, faces):
        return {face.track_id: 0.8 for face in faces}

    def close(self):
        self.closed = True


def _setup_live_session(monkeypatch, clock_values):
    capture = _FakeCapture(0)
    tracker = _FakeTracker()
    head_pose = _FakeHeadPose()
    tracker.closed = False
    head_pose.closed = False
    clock = iter(clock_values)
    monkeypatch.setattr(live_session.cv2, "VideoCapture", lambda _index: capture)
    monkeypatch.setattr(live_session, "FaceTracker", lambda: tracker)
    monkeypatch.setattr(live_session, "HeadPoseEstimator", lambda: head_pose)
    monkeypatch.setattr(live_session.time, "monotonic", lambda: next(clock))
    return capture, tracker, head_pose


def test_live_session_processes_frames_and_saves_report(monkeypatch, tmp_path):
    capture, tracker, head_pose = _setup_live_session(
        monkeypatch,
        [100.0, 100.0, 100.5, 101.0],
    )
    saved = {}
    monkeypatch.setattr(
        live_session,
        "save_session_report",
        lambda session_id, payload: (
            saved.update(session_id=session_id, payload=payload) or tmp_path / "report.json"
        ),
    )

    session = live_session.LiveWebcamSession(
        duration_seconds=1.0,
        session_id="live_test",
    )
    first_frame = session.step()
    second_frame = session.step()

    assert first_frame is not None
    assert first_frame.faces[0].track_id == 1
    assert first_frame.samples[0].head_pose_score == 0.8
    assert second_frame is not None
    assert session.current_attendance[0].total_visible_seconds == 0.54
    assert session.current_attention is not None

    report = session.finalize()

    assert report.session_id == "live_test"
    assert report.report_path == tmp_path / "report.json"
    assert saved["payload"]["source"] == "webcam"
    assert saved["payload"]["analysis"].session_duration_seconds == 1.0
    assert capture.released
    assert tracker.closed
    assert head_pose.closed
    assert session.finalize() is report


def test_live_session_rejects_unavailable_camera(monkeypatch):
    capture = _FakeCapture(0)
    capture.opened = False
    monkeypatch.setattr(live_session.cv2, "VideoCapture", lambda _index: capture)

    with pytest.raises(RuntimeError, match="Could not open webcam"):
        live_session.LiveWebcamSession(duration_seconds=1.0)

    assert capture.released


def test_live_session_releases_camera_when_microphone_cannot_start(monkeypatch):
    capture, tracker, head_pose = _setup_live_session(monkeypatch, [100.0])

    def _fail_microphone(sample_rate):
        assert sample_rate > 0
        raise RuntimeError("microphone permission denied")

    monkeypatch.setattr(live_session, "LiveAudioCapture", _fail_microphone)

    with pytest.raises(RuntimeError, match="microphone permission denied"):
        live_session.LiveWebcamSession(
            duration_seconds=1.0,
            capture_microphone=True,
        )

    assert capture.released
    assert tracker.closed
    assert head_pose.closed


def test_live_session_cannot_finalize_without_frames(monkeypatch, tmp_path):
    capture, _, _ = _setup_live_session(monkeypatch, [1.0, 1.0])
    session = live_session.LiveWebcamSession(duration_seconds=1.0)

    with pytest.raises(RuntimeError, match="without captured frames"):
        session.finalize()

    assert capture.released


def test_live_session_fuses_microphone_audio_and_saves_capture_notices(
    monkeypatch, tmp_path
):
    capture, _, _ = _setup_live_session(
        monkeypatch,
        [100.0, 100.0, 100.5, 101.0],
    )
    saved = {}

    class _FakeAudioCapture:
        def __init__(self, sample_rate):
            self.sample_rate = sample_rate
            self.closed = False

        def stop(self):
            return _speech_like_waveform(), ["input overflow"]

        def close(self):
            self.closed = True

    monkeypatch.setattr(live_session, "LiveAudioCapture", _FakeAudioCapture)
    monkeypatch.setattr(
        live_session,
        "save_session_report",
        lambda session_id, payload: (
            saved.update(session_id=session_id, payload=payload) or tmp_path / "report.json"
        ),
    )

    session = live_session.LiveWebcamSession(
        duration_seconds=1.0,
        capture_microphone=True,
        session_id="live_audio_test",
    )
    session.step()
    session.step()
    report = session.finalize()

    assert report.session_id == "live_audio_test"
    assert report.analysis.fused_intervals[0].audio_score > 0
    assert saved["payload"]["audio_capture_warnings"] == ["input overflow"]
    assert capture.released
