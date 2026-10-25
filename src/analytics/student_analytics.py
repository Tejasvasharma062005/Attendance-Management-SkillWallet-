"""Reflective per-participant session summaries (no grading or ranking).

Each summary describes one participant's own session only: attendance, how
steady their visual engagement was, and when focus was stronger or lower. There
is deliberately no cross-participant comparison, score ranking or grade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import config
from src.agent.engagement_scorer import EngagementTimelinePoint
from src.vision.attendance import AttendanceRecord
from src.vision.engagement_visual import VisualEngagementSample


@dataclass
class StudentSessionSummary:
    track_id: int
    attended: bool
    visible_seconds: float
    avg_attention: float
    engagement_consistency: str
    focus_highlights: str
    reflection_note: str
    # One entry per session interval in which the participant was visible:
    # {"start_time", "end_time", "attention"}. Gaps mean "not visible".
    timeline: List[Dict[str, float]] = field(default_factory=list)


def _consistency_label(scores: List[float]) -> str:
    if not scores:
        return "insufficient_data"
    spread = max(scores) - min(scores)
    if spread < 0.2:
        return "steady"
    if spread < 0.45:
        return "variable"
    return "fluctuating"


def _participant_timeline(
    samples: List[VisualEngagementSample],
    timeline: List[EngagementTimelinePoint],
) -> List[Dict[str, float]]:
    """Average this participant's attention within each session interval."""
    points: List[Dict[str, float]] = []
    for interval in timeline:
        window = [s for s in samples if interval.start_time <= s.timestamp < interval.end_time]
        if not window:
            continue
        points.append(
            {
                "start_time": round(interval.start_time, 2),
                "end_time": round(interval.end_time, 2),
                "attention": round(sum(s.attention_proxy for s in window) / len(window), 4),
            }
        )
    return points


def _longest_run(
    points: List[Dict[str, float]], predicate
) -> Tuple[float, float] | None:
    """Longest stretch of back-to-back intervals whose attention satisfies ``predicate``."""
    best: Tuple[float, float] | None = None
    start: float | None = None
    end = 0.0
    previous_end: float | None = None
    for point in points:
        contiguous = previous_end is not None and abs(point["start_time"] - previous_end) < 1e-6
        if predicate(point["attention"]):
            if start is None or not contiguous:
                start = point["start_time"]
            end = point["end_time"]
            if best is None or (end - start) > (best[1] - best[0]):
                best = (start, end)
        else:
            start = None
        previous_end = point["end_time"]
    return best


def generate_student_analytics(
    attendance: List[AttendanceRecord],
    visual_samples: List[VisualEngagementSample],
    timeline: List[EngagementTimelinePoint],
) -> List[StudentSessionSummary]:
    by_track: Dict[int, List[VisualEngagementSample]] = {}
    for sample in visual_samples:
        by_track.setdefault(sample.track_id, []).append(sample)

    summaries: List[StudentSessionSummary] = []
    for record in attendance:
        samples = by_track.get(record.track_id, [])
        attentions = [s.attention_proxy for s in samples]
        avg_att = sum(attentions) / len(attentions) if attentions else 0.0
        consistency = _consistency_label(attentions)
        personal_timeline = _participant_timeline(samples, timeline)

        high = _longest_run(
            personal_timeline, lambda a: a >= config.ENGAGEMENT_HIGH_THRESHOLD
        )
        low = _longest_run(
            personal_timeline, lambda a: a < config.ENGAGEMENT_MEDIUM_THRESHOLD
        )
        parts: List[str] = []
        if high:
            parts.append(f"Stronger focus observed between {high[0]:.0f}s and {high[1]:.0f}s.")
        if low:
            parts.append(f"Lower focus signals between {low[0]:.0f}s and {low[1]:.0f}s.")
        if not parts:
            parts.append(
                "No extended high- or low-focus window detected; short movement breaks are normal."
            )
        highlights = " ".join(parts)

        if not record.present:
            note = (
                "Visibility duration was below the attendance threshold. "
                "Review camera position or session participation if this seems unexpected."
            )
        elif avg_att >= 0.6:
            note = "Participation signals were generally consistent. Keep up active engagement habits."
        else:
            note = (
                "Some periods of lower visual engagement were detected. "
                "Use the session timeline to reflect on when focus dipped."
            )

        summaries.append(
            StudentSessionSummary(
                track_id=record.track_id,
                attended=record.present,
                visible_seconds=record.total_visible_seconds,
                avg_attention=round(avg_att, 3),
                engagement_consistency=consistency,
                focus_highlights=highlights,
                reflection_note=note,
                timeline=personal_timeline,
            )
        )
    return summaries
