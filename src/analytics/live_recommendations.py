"""Real-time, group-level suggestions for a live session.

Rules look at *trends over a short window* of the group's average attention,
never at a single frame, and never single out an individual participant. They
are supportive prompts for the instructor, not evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import config

_SPAN_FOR_DROP = 10.0


@dataclass
class LiveRecommendation:
    kind: str       # "success" | "info" | "warning"
    message: str
    evidence: str


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def generate_live_recommendations(
    history: List[Dict[str, float]],
    elapsed_seconds: float,
    visible_now: int,
) -> List[LiveRecommendation]:
    """Suggest next steps from recent group attention.

    ``history`` holds ``{"time": seconds, "attention": 0..1}`` points of the
    group's average attention, oldest first.
    """
    recs: List[LiveRecommendation] = []
    if not history:
        if elapsed_seconds >= 5.0 and visible_now == 0:
            recs.append(
                LiveRecommendation(
                    kind="info",
                    message="No faces are visible yet. Check the camera angle and lighting.",
                    evidence="No participant detected in the live view.",
                )
            )
        return recs

    now = history[-1]["time"]
    window = config.LIVE_LOW_ATTENTION_SECONDS
    recent = [p["attention"] for p in history if p["time"] >= now - window]
    spans_window = history[0]["time"] <= now - window * 0.8

    if spans_window and _mean(recent) < config.LIVE_LOW_ATTENTION:
        recs.append(
            LiveRecommendation(
                kind="warning",
                message=(
                    "Group attention signals have been lower for a while. A quick question, "
                    "poll or short activity may help re-engage the class."
                ),
                evidence=(
                    f"Average attention {_mean(recent):.0%} over the last {window:.0f}s "
                    f"(guide: below {config.LIVE_LOW_ATTENTION:.0%})."
                ),
            )
        )
    else:
        last = [p["attention"] for p in history if p["time"] >= now - _SPAN_FOR_DROP]
        before = [
            p["attention"]
            for p in history
            if now - 2 * _SPAN_FOR_DROP <= p["time"] < now - _SPAN_FOR_DROP
        ]
        if last and before and _mean(before) - _mean(last) >= config.ATTENTION_DROP_DELTA:
            recs.append(
                LiveRecommendation(
                    kind="info",
                    message=(
                        "Attention just dipped. Consider whether a pause or a change of "
                        "activity would help."
                    ),
                    evidence=(
                        f"Average attention fell from {_mean(before):.0%} to {_mean(last):.0%} "
                        f"over the last {2 * _SPAN_FOR_DROP:.0f}s."
                    ),
                )
            )
        elif spans_window and _mean(recent) >= config.ENGAGEMENT_HIGH_THRESHOLD:
            recs.append(
                LiveRecommendation(
                    kind="success",
                    message="Attention signals are strong. Keep the current pace.",
                    evidence=f"Average attention {_mean(recent):.0%} over the last {window:.0f}s.",
                )
            )

    if visible_now == 0:
        recs.append(
            LiveRecommendation(
                kind="info",
                message="No faces are visible in the latest frame. Check the camera view.",
                evidence="Zero participants detected in the latest frame.",
            )
        )
    if not recs:
        recs.append(
            LiveRecommendation(
                kind="info",
                message="Collecting signals. Suggestions appear once there is enough recent data.",
                evidence=f"{elapsed_seconds:.0f}s of session so far.",
            )
        )
    return recs
