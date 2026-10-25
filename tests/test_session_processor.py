"""Tests for recorded-session input wiring."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import session_processor
from src.pipeline.session_processor import SessionProcessor
from src.vision.attendance import PresenceAttendanceEngine
from src.vision.engagement_visual import VisualEngagementAnalyzer


class _FakeCapture:
    def __init__(self):
        self.frames = [np.zeros((120, 160, 3), dtype=np.uint8)]

    def isOpened(self):
        return True

    def get(self, _property):
        return 25.0

    def read(self):
        if self.frames:
            return True, self.frames.pop()
        return False, None

    def release(self):
        pass


class _FakeTracker:
    def process_frame(self, _frame, _timestamp):
        return []

    def close(self):
        pass


class _FakeHeadPose:
    def estimate_frame(self, _frame, _faces):
        return {}

    def close(self):
        pass


def test_recorded_session_uses_uploaded_audio_file(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "classroom.mp4"
    audio_path = tmp_path / "classroom.wav"
    video_path.write_bytes(b"video")
    audio_path.write_bytes(b"audio")
    capture = _FakeCapture()
    audio_calls = []
    saved = {}

    class _FakeAudioExtractor:
        def from_file(self, path):
            audio_calls.append(("file", path))
            return np.zeros(22050, dtype=np.float32), 22050

        def from_video(self, path):
            audio_calls.append(("video", path))
            return None

        def to_segments(self, _waveform, _sample_rate):
            return []

    monkeypatch.setattr(session_processor.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(
        SessionProcessor,
        "_build_components",
        staticmethod(
            lambda: (
                _FakeTracker(),
                PresenceAttendanceEngine(),
                VisualEngagementAnalyzer(),
                _FakeHeadPose(),
            )
        ),
    )
    monkeypatch.setattr(session_processor, "AudioExtractor", _FakeAudioExtractor)
    monkeypatch.setattr(
        session_processor,
        "save_session_report",
        lambda session_id, payload: (
            saved.update(session_id=session_id, payload=payload)
            or tmp_path / "report.json"
        ),
    )

    report = SessionProcessor().process_video(
        video_path,
        session_id="audio_test",
        audio_path=audio_path,
    )

    assert audio_calls == [("file", audio_path)]
    assert report.report_path == tmp_path / "report.json"
    assert saved["session_id"] == "audio_test"
    assert saved["payload"]["analysis"].session_duration_seconds > 0
