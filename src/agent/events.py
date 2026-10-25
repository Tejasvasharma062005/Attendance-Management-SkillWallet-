"""Session-dynamics events: attention drops, interaction peaks, pacing decline.

These are the moments an instructor wants to find on a timeline (Scenario 2).
Each event states what happened and the numbers behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import config
from src.agent.engagement_scorer import EngagementTimelinePoint
from src.agent.fusion import FusedInterval

ATTENTION_DROP = "attention_drop"
INTERACTION_PEAK = "interaction_peak"
PACING_DECLINE = "pacing_decline"


@dataclass
class SessionEvent:
    event_type: str
    start_time: float
    end_time: float
    detail: str
    magnitude: float = 0.0


def detect_session_events(
    timeline: List[EngagementTimelinePoint],
    fused: List[FusedInterval],
) -> List[SessionEvent]:
    """Find attention drops, interaction peaks and a declining session pace."""
    events: List[SessionEvent] = []

    # Attention drops: engagement falls sharply from one interval to the next.
    for previous, current in zip(timeline, timeline[1:]):
        fall = previous.score - current.score
        if fall >= config.ATTENTION_DROP_DELTA:
            events.append(
                SessionEvent(
                    event_type=ATTENTION_DROP,
                    start_time=current.start_time,
                    end_time=current.end_time,
                    detail=(
                        f"Engagement fell from {previous.score:.2f} to "
                        f"{current.score:.2f} between consecutive intervals."
                    ),
                    magnitude=round(fall, 4),
                )
            )

    # Interaction peaks: consecutive intervals with strong speech activity.
    run: List[FusedInterval] = []

    def flush() -> None:
        if not run:
            return
        peak = max(i.audio_score for i in run)
        events.append(
            SessionEvent(
                event_type=INTERACTION_PEAK,
                start_time=run[0].start_time,
                end_time=run[-1].end_time,
                detail=f"Speech activity reached {peak:.2f}, suggesting discussion or interaction.",
                magnitude=round(peak, 4),
            )
        )
        run.clear()

    for interval in fused:
        has_audio = interval.metadata.get("sample_count_audio", 0.0) > 0
        if has_audio and interval.audio_score >= config.INTERACTION_PEAK_SPEECH:
            run.append(interval)
        else:
            flush()
    flush()

    # Pacing: compare the first and last third of the session.
    if len(timeline) >= config.PACING_MIN_INTERVALS:
        third = max(1, len(timeline) // 3)
        early = sum(p.score for p in timeline[:third]) / third
        late_points = timeline[-third:]
        late = sum(p.score for p in late_points) / third
        if early - late >= config.PACING_DECLINE_DELTA:
            events.append(
                SessionEvent(
                    event_type=PACING_DECLINE,
                    start_time=late_points[0].start_time,
                    end_time=late_points[-1].end_time,
                    detail=(
                        f"Average engagement fell from {early:.2f} in the first third "
                        f"of the session to {late:.2f} in the last third."
                    ),
                    magnitude=round(early - late, 4),
                )
            )

    events.sort(key=lambda e: (e.start_time, e.event_type))
    return events
