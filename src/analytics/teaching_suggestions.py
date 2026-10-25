"""Context-aware, non-prescriptive teaching suggestions from engagement trends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.agent.engagement_scorer import EngagementTimelinePoint
from src.agent.events import ATTENTION_DROP, INTERACTION_PEAK, PACING_DECLINE, detect_session_events
from src.agent.fusion import FusedInterval


@dataclass
class TeachingSuggestion:
    category: str
    message: str
    time_range: str
    evidence: str


def generate_teaching_suggestions(
    timeline: List[EngagementTimelinePoint],
    fused: List[FusedInterval],
) -> List[TeachingSuggestion]:
    suggestions: List[TeachingSuggestion] = []

    low_streak = 0
    streak_start = 0.0
    for point in timeline:
        if point.level == "low":
            if low_streak == 0:
                streak_start = point.start_time
            low_streak += 1
        else:
            if low_streak >= 3:
                suggestions.append(
                    TeachingSuggestion(
                        category="pacing",
                        message=(
                            "Consider a short interactive activity or discussion break "
                            "to re-engage the class."
                        ),
                        time_range=f"{streak_start:.0f}s–{point.start_time:.0f}s",
                        evidence=f"Sustained low engagement for {low_streak} intervals.",
                    )
                )
            low_streak = 0

    if low_streak >= 3:
        suggestions.append(
            TeachingSuggestion(
                category="pacing",
                message=(
                    "Consider a short interactive activity or discussion break "
                    "to re-engage the class."
                ),
                time_range=f"{streak_start:.0f}s–{timeline[-1].end_time:.0f}s",
                evidence=f"Sustained low engagement for {low_streak} intervals.",
            )
        )

    peaks = [p for p in timeline if p.level == "high"]
    if peaks:
        best = max(peaks, key=lambda p: p.score)
        suggestions.append(
            TeachingSuggestion(
                category="effective_segment",
                message=(
                    "This segment showed strong engagement. Note what teaching "
                    "approach or activity was used here for future sessions."
                ),
                time_range=f"{best.start_time:.0f}s–{best.end_time:.0f}s",
                evidence=f"Peak engagement score {best.score:.2f}.",
            )
        )

    # Classroom management: loud intervals with low engagement. Consecutive
    # intervals are reported as one range, and the message depends on whether
    # the noise was speech (side conversations) or not (fan, outside noise).
    noisy: List[List[FusedInterval]] = []
    previous_noisy = False
    for interval in fused:
        is_noisy = (
            interval.metadata.get("avg_noise", 0.0) > 0.75 and interval.combined_score < 0.4
        )
        if is_noisy:
            if previous_noisy:
                noisy[-1].append(interval)
            else:
                noisy.append([interval])
        previous_noisy = is_noisy

    for group in noisy[:2]:
        noise = sum(i.metadata.get("avg_noise", 0.0) for i in group) / len(group)
        speech_values = [
            i.metadata["speech_fraction"] for i in group if "speech_fraction" in i.metadata
        ]
        speech = sum(speech_values) / len(speech_values) if speech_values else None
        if speech is not None and speech < 0.3:
            message = (
                "Loud background noise that is not mostly speech coincided with lower "
                "engagement. Consider checking room conditions such as a fan, "
                "equipment or outside noise."
            )
        elif speech is not None and speech >= 0.6:
            message = (
                "Loud overlapping speech coincided with lower engagement. Consider "
                "pausing to regain attention or giving side conversations a structure."
            )
        else:
            message = (
                "High ambient noise coincided with lower engagement. "
                "Consider pausing for attention or adjusting room dynamics."
            )
        evidence = f"Average noise proxy {noise:.2f}."
        db_values = [i.metadata["avg_noise_db"] for i in group if "avg_noise_db" in i.metadata]
        if db_values:
            evidence += f" About {sum(db_values) / len(db_values):.0f} dBFS."
        suggestions.append(
            TeachingSuggestion(
                category="classroom_management",
                message=message,
                time_range=f"{group[0].start_time:.0f}s–{group[-1].end_time:.0f}s",
                evidence=evidence,
            )
        )

    # Session dynamics from detected events (attention drops, pacing, interaction).
    events = detect_session_events(timeline, fused)

    declines = [e for e in events if e.event_type == PACING_DECLINE]
    already_pacing = any(s.category == "pacing" for s in suggestions)
    if declines and not already_pacing:
        event = declines[0]
        suggestions.append(
            TeachingSuggestion(
                category="pacing",
                message=(
                    "Engagement trended downward toward the end of the session. "
                    "Consider shorter segments or a mid-session break next time."
                ),
                time_range=f"{event.start_time:.0f}s–{event.end_time:.0f}s",
                evidence=event.detail,
            )
        )

    drops = sorted(
        (e for e in events if e.event_type == ATTENTION_DROP),
        key=lambda e: e.magnitude,
        reverse=True,
    )
    for event in drops[:2]:
        suggestions.append(
            TeachingSuggestion(
                category="attention_drop",
                message=(
                    "Engagement dropped sharply here. Consider what changed just before "
                    "this point (activity, pace, content) and whether a quick check-in "
                    "would help."
                ),
                time_range=f"{event.start_time:.0f}s–{event.end_time:.0f}s",
                evidence=event.detail,
            )
        )

    interaction = [e for e in events if e.event_type == INTERACTION_PEAK]
    if interaction:
        event = max(interaction, key=lambda e: e.magnitude)
        suggestions.append(
            TeachingSuggestion(
                category="interaction",
                message=(
                    "Strong speech activity suggests discussion or interaction here. "
                    "If this was a planned activity, note what prompted it."
                ),
                time_range=f"{event.start_time:.0f}s–{event.end_time:.0f}s",
                evidence=event.detail,
            )
        )

    if not suggestions:
        suggestions.append(
            TeachingSuggestion(
                category="general",
                message="Engagement remained stable. Continue monitoring trends across sessions.",
                time_range="full session",
                evidence="No sustained low-engagement or noise patterns detected.",
            )
        )
    return suggestions
