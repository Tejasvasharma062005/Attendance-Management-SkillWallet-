"""Core multimodal agent: fuse vision + audio and produce session engagement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.agent.engagement_scorer import EngagementTimelinePoint, build_engagement_timeline
from src.agent.events import SessionEvent, detect_session_events
from src.agent.fusion import FusedInterval, fuse_intervals
from src.audio.analyzer import AudioWindowMetrics
from src.vision.attendance import AttendanceRecord
from src.vision.engagement_visual import VisualEngagementSample


@dataclass
class SessionAnalysisResult:
    attendance: List[AttendanceRecord]
    fused_intervals: List[FusedInterval]
    engagement_timeline: List[EngagementTimelinePoint]
    session_duration_seconds: float
    events: List[SessionEvent] = field(default_factory=list)


@dataclass
class MultimodalAgent:
    """Rule-based reasoning over synchronized multimodal features."""

    def analyze(
        self,
        attendance: List[AttendanceRecord],
        visual_samples: List[VisualEngagementSample],
        audio_metrics: List[AudioWindowMetrics],
        session_duration_seconds: float,
    ) -> SessionAnalysisResult:
        fused = fuse_intervals(
            visual_samples,
            audio_metrics,
            session_duration=session_duration_seconds,
        )
        timeline = build_engagement_timeline(fused)
        return SessionAnalysisResult(
            attendance=attendance,
            fused_intervals=fused,
            engagement_timeline=timeline,
            session_duration_seconds=session_duration_seconds,
            events=detect_session_events(timeline, fused),
        )
