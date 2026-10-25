"""Incremental webcam sessions for live preview and post-session analytics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src.agent.multimodal_agent import MultimodalAgent, SessionAnalysisResult
from src.analytics.student_analytics import StudentSessionSummary, generate_student_analytics
from src.analytics.teaching_suggestions import TeachingSuggestion, generate_teaching_suggestions
import config
from src.audio.analyzer import AudioAnalyzer
from src.audio.extractor import AudioExtractor
from src.audio.live_capture import LiveAudioCapture
from src.pipeline.payload import build_session_payload
from src.storage.logs import save_session_report
from src.vision.attendance import AttendanceRecord, PresenceAttendanceEngine
from src.vision.engagement_visual import VisualEngagementAnalyzer, VisualEngagementSample
from src.vision.face_tracker import FaceTracker, TrackedFace
from src.vision.head_pose import HeadPoseEstimator


@dataclass
class LiveFrame:
    """A processed webcam frame and the signals measured from it."""

    frame_bgr: np.ndarray
    faces: List[TrackedFace]
    pose_scores: Dict[int, float]
    samples: List[VisualEngagementSample]
    elapsed_seconds: float


@dataclass
class LiveWebcamReport:
    session_id: str
    analysis: SessionAnalysisResult
    teaching_suggestions: List[TeachingSuggestion]
    student_analytics: List[StudentSessionSummary]
    report_path: Path


class LiveWebcamSession:
    """Process webcam frames incrementally until the user ends the session."""

    def __init__(
        self,
        duration_seconds: float = 120.0,
        camera_index: int = 0,
        audio_path: str | Path | None = None,
        capture_microphone: bool = False,
        session_id: str | None = None,
    ) -> None:
        if duration_seconds <= 0:
            raise ValueError("Session duration must be greater than zero.")
        if audio_path is not None and capture_microphone:
            raise ValueError("Choose either an audio file or live microphone capture.")

        self.session_id = session_id or datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        self.duration_seconds = duration_seconds
        self.audio_path = Path(audio_path) if audio_path is not None else None
        self.capture_microphone = capture_microphone
        self._audio_capture: LiveAudioCapture | None = None
        self.elapsed_seconds = 0.0
        self.frame_count = 0
        self.finished = False
        self.ended_early = False
        self._closed = False
        self._report: LiveWebcamReport | None = None
        self._last_sample_time: float | None = None
        self._visual_samples: List[VisualEngagementSample] = []

        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError("Could not open webcam. Check camera permissions and connection.")

        self._fps = self._capture.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 25.0
        self._tracker: FaceTracker | None = None
        self._head_pose: HeadPoseEstimator | None = None
        try:
            self._tracker = FaceTracker()
            self._attendance = PresenceAttendanceEngine()
            self._visual = VisualEngagementAnalyzer()
            self._head_pose = HeadPoseEstimator()
            if capture_microphone:
                self._audio_capture = LiveAudioCapture(config.AUDIO_SAMPLE_RATE)
        except Exception:
            self._capture.release()
            if self._tracker is not None:
                self._tracker.close()
            if self._head_pose is not None:
                self._head_pose.close()
            raise
        self.started_at = time.monotonic()
        self.started_wall = datetime.now(timezone.utc)

    @property
    def attendance_threshold_seconds(self) -> float:
        """Visibility needed to count as present in this (planned) session."""
        return self._attendance.effective_threshold(self.duration_seconds)

    @property
    def current_attendance(self) -> List[AttendanceRecord]:
        """Return current visibility totals without closing the active session."""
        return self._attendance.current_records(self.duration_seconds)

    @property
    def current_attention(self) -> float | None:
        if not self._visual_samples:
            return None
        return sum(sample.attention_proxy for sample in self._visual_samples) / len(
            self._visual_samples
        )

    def step(self) -> LiveFrame | None:
        """Read and analyze one frame; return None when capture should stop."""
        if self._closed:
            raise RuntimeError("This webcam session is already closed.")
        if self.finished:
            return None

        now = time.monotonic()
        self.elapsed_seconds = min(now - self.started_at, self.duration_seconds)
        if self.elapsed_seconds >= self.duration_seconds:
            self.finished = True
            return None

        ok, frame = self._capture.read()
        if not ok:
            self.ended_early = True
            self.finished = True
            if self.frame_count == 0:
                raise RuntimeError("The webcam opened but did not provide any video frames.")
            return None

        height, width = frame.shape[:2]
        self._visual.set_frame_size(width, height)
        faces = self._tracker.process_frame(frame, self.elapsed_seconds)
        frame_delta = (
            1.0 / self._fps
            if self._last_sample_time is None
            else min(max(self.elapsed_seconds - self._last_sample_time, 0.0), 1.0)
        )
        self._attendance.update(
            [face.track_id for face in faces],
            self.elapsed_seconds,
            frame_delta,
        )
        pose_scores = self._head_pose.estimate_frame(frame, faces)
        samples = [
            self._visual.update(
                face.track_id,
                face.center,
                self.elapsed_seconds,
                pose_scores.get(face.track_id, 1.0),
            )
            for face in faces
        ]
        self._visual_samples.extend(samples)
        self.frame_count += 1
        self._last_sample_time = self.elapsed_seconds
        return LiveFrame(
            frame_bgr=frame,
            faces=faces,
            pose_scores=pose_scores,
            samples=samples,
            elapsed_seconds=self.elapsed_seconds,
        )

    def finalize(self) -> LiveWebcamReport:
        """Close capture, run optional audio analysis, and save the session report."""
        if self._report is not None:
            return self._report
        if self.frame_count == 0:
            self.close()
            raise RuntimeError("Cannot save a webcam session without captured frames.")

        self.elapsed_seconds = min(
            max(time.monotonic() - self.started_at, self.elapsed_seconds),
            self.duration_seconds,
        )
        self.finished = True

        attendance = self._attendance.finalize(self.elapsed_seconds)
        audio_metrics = []
        audio_capture_warnings: List[str] = []
        try:
            if self._audio_capture is not None:
                waveform, audio_capture_warnings = self._audio_capture.stop()
                self._audio_capture = None
                segments = AudioExtractor().to_segments(
                    waveform, config.AUDIO_SAMPLE_RATE
                )
                audio_metrics = AudioAnalyzer().analyze_all(segments)
            elif self.audio_path is not None:
                extractor = AudioExtractor()
                waveform, sample_rate = extractor.from_file(self.audio_path)
                segments = extractor.to_segments(waveform, sample_rate)
                audio_metrics = AudioAnalyzer().analyze_all(segments)
        finally:
            self.close()

        analysis = MultimodalAgent().analyze(
            attendance,
            self._visual_samples,
            audio_metrics,
            self.elapsed_seconds,
        )
        suggestions = generate_teaching_suggestions(
            analysis.engagement_timeline,
            analysis.fused_intervals,
        )
        student_analytics = generate_student_analytics(
            attendance,
            self._visual_samples,
            analysis.engagement_timeline,
        )
        payload = build_session_payload(
            session_id=self.session_id,
            source_fields={
                "source": "webcam",
                "audio_capture_warnings": audio_capture_warnings,
            },
            analysis=analysis,
            suggestions=suggestions,
            student_analytics=student_analytics,
            audio_metrics=audio_metrics,
            session_duration_seconds=self.elapsed_seconds,
            attendance_threshold_seconds=self._attendance.effective_threshold(
                self.elapsed_seconds
            ),
            session_started_at=self.started_wall.isoformat(timespec="seconds"),
        )
        report_path = save_session_report(self.session_id, payload)
        self._report = LiveWebcamReport(
            session_id=self.session_id,
            analysis=analysis,
            teaching_suggestions=suggestions,
            student_analytics=student_analytics,
            report_path=report_path,
        )
        return self._report

    def close(self) -> None:
        """Release the camera and vision models once."""
        if self._closed:
            return
        self._closed = True
        audio_capture = self._audio_capture
        self._audio_capture = None
        try:
            if audio_capture is not None:
                audio_capture.close()
        finally:
            try:
                self._capture.release()
            finally:
                try:
                    if self._tracker is not None:
                        self._tracker.close()
                finally:
                    if self._head_pose is not None:
                        self._head_pose.close()
