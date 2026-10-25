"""Rule-based engagement scoring from fused multimodal intervals.

Scores are computed per interval (not one session average) so attention drops,
peaks and long disengaged stretches stay visible. Every interval carries a
plain-language ``rationale`` naming the signals behind its score, and the level
thresholds are configurable in ``config.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import config
from src.agent.fusion import FusedInterval


@dataclass
class EngagementTimelinePoint:
    start_time: float
    end_time: float
    score: float
    level: str
    rationale: str = ""


def classify_level(score: float) -> str:
    if score >= config.ENGAGEMENT_HIGH_THRESHOLD:
        return "high"
    if score >= config.ENGAGEMENT_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def _strength(value: float) -> str:
    if value >= config.ENGAGEMENT_HIGH_THRESHOLD:
        return "strong"
    if value >= config.ENGAGEMENT_MEDIUM_THRESHOLD:
        return "moderate"
    return "weak"


def explain_interval(interval: FusedInterval) -> str:
    """Describe, in plain language, which signals produced this interval's score."""
    meta = interval.metadata or {}
    clauses: List[str] = []

    if not meta.get("sample_count_visual", 1.0):
        clauses.append("no faces were visible, so there is no visual evidence")
    else:
        clauses.append(
            f"{_strength(interval.visual_score)} visual attention ({interval.visual_score:.2f})"
        )
        head_pose = meta.get("head_pose_score")
        if head_pose is not None:
            if head_pose >= 0.8:
                clauses.append("heads mostly facing forward")
            elif head_pose < 0.5:
                clauses.append(f"heads often turned away (head pose {head_pose:.2f})")
        inactive = meta.get("max_inactive_seconds", 0.0) or 0.0
        if inactive >= config.INACTIVITY_SECONDS:
            clauses.append(f"someone stayed still for up to {inactive:.0f}s")

    if meta.get("sample_count_audio", 0.0) > 0:
        speech = meta.get("speech_fraction", 0.0) or 0.0
        if speech >= 0.5:
            clauses.append(f"speech present about {speech:.0%} of the time")
        elif speech > 0.0:
            clauses.append(f"brief speech ({speech:.0%} of the time)")
        else:
            clauses.append("no speech detected")
        # Loudness caused by speech is interaction, not "noise".
        if (meta.get("avg_noise", 0.0) or 0.0) > 0.75 and speech < 0.5:
            clauses.append("loud background noise that is not mostly speech")
    else:
        clauses.append("no audio in this interval, so it was scored on visual signals only")

    text = "; ".join(clauses)
    return text[0].upper() + text[1:] + "."


def build_engagement_timeline(fused: List[FusedInterval]) -> List[EngagementTimelinePoint]:
    timeline: List[EngagementTimelinePoint] = []
    for interval in fused:
        timeline.append(
            EngagementTimelinePoint(
                start_time=interval.start_time,
                end_time=interval.end_time,
                score=interval.combined_score,
                level=classify_level(interval.combined_score),
                rationale=explain_interval(interval),
            )
        )
    return timeline
