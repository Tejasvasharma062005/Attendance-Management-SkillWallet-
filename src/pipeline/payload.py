"""Shared construction of the saved session-report payload."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agent.multimodal_agent import SessionAnalysisResult
from src.analytics.student_analytics import StudentSessionSummary
from src.analytics.teaching_suggestions import TeachingSuggestion
from src.audio.analyzer import AudioWindowMetrics, summarize_audio
from src.storage.attendance_log import build_attendance_log


def build_session_payload(
    *,
    session_id: str,
    source_fields: Dict[str, Any],
    analysis: SessionAnalysisResult,
    suggestions: List[TeachingSuggestion],
    student_analytics: List[StudentSessionSummary],
    audio_metrics: List[AudioWindowMetrics],
    session_duration_seconds: float,
    attendance_threshold_seconds: float,
    session_started_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble everything saved in ``report.json`` for one session.

    ``session_started_at`` is an ISO-8601 UTC time for live sessions. It is
    ``None`` for recorded video, whose real recording time is unknown, so the
    attendance log carries session times only and never an invented clock time.
    """
    return {
        **source_fields,
        "session_duration_seconds": session_duration_seconds,
        "session_started_at": session_started_at,
        "attendance_threshold_seconds": round(attendance_threshold_seconds, 2),
        "analysis": analysis,
        "teaching_suggestions": suggestions,
        "student_analytics": student_analytics,
        "attendance_log": build_attendance_log(
            analysis.attendance, session_id, session_started_at
        ),
        "audio_summary": summarize_audio(audio_metrics),
        "audio_windows": audio_metrics,
    }
