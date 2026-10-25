"""End-to-end session processing: video (+ optional audio) through all modules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import cv2

from src.agent.multimodal_agent import MultimodalAgent, SessionAnalysisResult
from src.analytics.student_analytics import StudentSessionSummary, generate_student_analytics
from src.analytics.teaching_suggestions import TeachingSuggestion, generate_teaching_suggestions
from src.audio.analyzer import AudioAnalyzer
from src.audio.extractor import AudioExtractor
from src.pipeline.payload import build_session_payload
from src.storage.logs import save_session_report
from src.vision.attendance import PresenceAttendanceEngine
from src.vision.engagement_visual import VisualEngagementAnalyzer, VisualEngagementSample
from src.vision.face_tracker import FaceTracker
from src.vision.head_pose import HeadPoseEstimator


@dataclass
class SessionReport:
    session_id: str
    analysis: SessionAnalysisResult
    teaching_suggestions: List[TeachingSuggestion]
    student_analytics: List[StudentSessionSummary]
    report_path: Path


class SessionProcessor:
    def __init__(self, max_frames: int | None = None) -> None:
        self.max_frames = max_frames

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_components():
        """Instantiate all vision components (shared by both process methods)."""
        tracker = FaceTracker()
        attendance_engine = PresenceAttendanceEngine()
        visual_engagement = VisualEngagementAnalyzer()
        head_pose = HeadPoseEstimator()
        return tracker, attendance_engine, visual_engagement, head_pose

    @staticmethod
    def _teardown_components(tracker, head_pose):
        tracker.close()
        head_pose.close()

    # ------------------------------------------------------------------
    # Video file processing
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str | Path,
        session_id: str | None = None,
        audio_path: str | Path | None = None,
    ) -> SessionReport:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_delta = 1.0 / fps

        tracker, attendance_engine, visual_engagement, head_pose = self._build_components()
        visual_samples: List[VisualEngagementSample] = []

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if self.max_frames is not None and frame_idx >= self.max_frames:
                break

            h, w = frame.shape[:2]
            visual_engagement.set_frame_size(w, h)
            timestamp = frame_idx / fps

            faces = tracker.process_frame(frame, timestamp)
            track_ids = [f.track_id for f in faces]
            attendance_engine.update(track_ids, timestamp, frame_delta)

            # Head pose estimation for all detected faces in this frame
            pose_scores = head_pose.estimate_frame(frame, faces)

            for face in faces:
                hp_score = pose_scores.get(face.track_id, 1.0)
                visual_samples.append(
                    visual_engagement.update(face.track_id, face.center, timestamp, hp_score)
                )
            frame_idx += 1

        cap.release()
        self._teardown_components(tracker, head_pose)

        session_duration = frame_idx / fps if frame_idx else 0.0
        attendance = attendance_engine.finalize(session_duration)

        audio_extractor = AudioExtractor()
        audio_analyzer = AudioAnalyzer()
        audio_metrics = []
        loaded = (
            audio_extractor.from_file(audio_path)
            if audio_path is not None
            else audio_extractor.from_video(video_path)
        )
        if loaded:
            waveform, sr = loaded
            segments = audio_extractor.to_segments(waveform, sr)
            audio_metrics = audio_analyzer.analyze_all(segments)

        agent = MultimodalAgent()
        analysis = agent.analyze(attendance, visual_samples, audio_metrics, session_duration)
        suggestions = generate_teaching_suggestions(
            analysis.engagement_timeline, analysis.fused_intervals
        )
        student_stats = generate_student_analytics(
            attendance, visual_samples, analysis.engagement_timeline
        )

        payload = build_session_payload(
            session_id=session_id,
            source_fields={"video": str(video_path)},
            analysis=analysis,
            suggestions=suggestions,
            student_analytics=student_stats,
            audio_metrics=audio_metrics,
            session_duration_seconds=session_duration,
            attendance_threshold_seconds=attendance_engine.effective_threshold(session_duration),
        )
        report_path = save_session_report(session_id, payload)

        return SessionReport(
            session_id=session_id,
            analysis=analysis,
            teaching_suggestions=suggestions,
            student_analytics=student_stats,
            report_path=report_path,
        )

    # ------------------------------------------------------------------
    # Webcam processing
    # ------------------------------------------------------------------

    def process_webcam(
        self,
        duration_seconds: float = 30.0,
        camera_index: int = 0,
        session_id: str | None = None,
    ) -> SessionReport:
        session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_delta = 1.0 / fps
        max_frames = int(duration_seconds * fps)

        tracker, attendance_engine, visual_engagement, head_pose = self._build_components()
        visual_samples: List[VisualEngagementSample] = []

        frame_idx = 0
        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            visual_engagement.set_frame_size(w, h)
            timestamp = frame_idx / fps
            faces = tracker.process_frame(frame, timestamp)
            attendance_engine.update([f.track_id for f in faces], timestamp, frame_delta)

            # Head pose estimation
            pose_scores = head_pose.estimate_frame(frame, faces)

            for face in faces:
                hp_score = pose_scores.get(face.track_id, 1.0)
                visual_samples.append(
                    visual_engagement.update(face.track_id, face.center, timestamp, hp_score)
                )
            frame_idx += 1

        cap.release()
        self._teardown_components(tracker, head_pose)

        session_duration = frame_idx / fps
        attendance = attendance_engine.finalize(session_duration)
        started_at = datetime.now(timezone.utc) - timedelta(seconds=session_duration)
        agent = MultimodalAgent()
        analysis = agent.analyze(attendance, visual_samples, [], session_duration)
        suggestions = generate_teaching_suggestions(
            analysis.engagement_timeline, analysis.fused_intervals
        )
        student_stats = generate_student_analytics(
            attendance, visual_samples, analysis.engagement_timeline
        )
        payload = build_session_payload(
            session_id=session_id,
            source_fields={"source": "webcam"},
            analysis=analysis,
            suggestions=suggestions,
            student_analytics=student_stats,
            audio_metrics=[],
            session_duration_seconds=session_duration,
            attendance_threshold_seconds=attendance_engine.effective_threshold(session_duration),
            session_started_at=started_at.isoformat(timespec="seconds"),
        )
        report_path = save_session_report(session_id, payload)
        return SessionReport(
            session_id=session_id,
            analysis=analysis,
            teaching_suggestions=suggestions,
            student_analytics=student_stats,
            report_path=report_path,
        )
