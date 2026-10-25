"""Timestamp-aligned fusion of visual and audio engagement signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import config
from src.audio.analyzer import AudioWindowMetrics
from src.vision.engagement_visual import VisualEngagementSample


@dataclass
class FusedInterval:
    start_time: float
    end_time: float
    visual_score: float
    audio_score: float
    combined_score: float
    metadata: Dict[str, float]


def fuse_intervals(
    visual_samples: List[VisualEngagementSample],
    audio_metrics: List[AudioWindowMetrics],
    interval_seconds: float = config.ENGAGEMENT_INTERVAL_SECONDS,
    session_duration: float | None = None,
) -> List[FusedInterval]:
    if not visual_samples and not audio_metrics:
        return []

    max_t = session_duration
    if max_t is None:
        candidates = [0.0]
        if visual_samples:
            candidates.append(max(s.timestamp for s in visual_samples))
        if audio_metrics:
            candidates.append(max(m.end_time for m in audio_metrics))
        max_t = max(candidates)

    fused: List[FusedInterval] = []
    t = 0.0
    while t < max_t:
        t_end = min(t + interval_seconds, max_t)
        v_window = [s for s in visual_samples if t <= s.timestamp < t_end]
        a_window = [m for m in audio_metrics if m.start_time < t_end and m.end_time > t]

        visual_score = 0.0
        head_pose_score = 0.0
        movement = 0.0
        max_inactive = 0.0
        if v_window:
            visual_score = sum(s.attention_proxy for s in v_window) / len(v_window)
            head_pose_score = sum(s.head_pose_score for s in v_window) / len(v_window)
            movement = sum(s.movement_score for s in v_window) / len(v_window)
            max_inactive = max(s.inactive_seconds for s in v_window)

        audio_score = 0.0
        noise = 0.0
        noise_db = None
        speech_fraction = None
        if a_window:
            # Weight each audio window by how much of it falls inside this
            # interval, so a short trailing window does not count like a full one.
            overlaps = [
                max(min(m.end_time, t_end) - max(m.start_time, t), 0.0) for m in a_window
            ]
            total = sum(overlaps) or 1.0
            audio_score = sum(m.speech_activity * o for m, o in zip(a_window, overlaps)) / total
            noise = sum(m.noise_level * o for m, o in zip(a_window, overlaps)) / total
            noise_db = sum(m.noise_db * o for m, o in zip(a_window, overlaps)) / total
            speech_fraction = (
                sum(o for m, o in zip(a_window, overlaps) if m.activity == "speech") / total
            )

        if a_window:
            combined = (
                config.WEIGHT_VISUAL * visual_score + config.WEIGHT_AUDIO * audio_score
            )
        else:
            # No audio in this interval (no source, or an input dropout):
            # score on visual evidence alone. Leaving audio at 0 here would cap
            # every score at WEIGHT_VISUAL and make "high" engagement unreachable.
            combined = visual_score

        metadata = {
            "avg_noise": round(noise, 4),
            "head_pose_score": round(head_pose_score, 4),
            "avg_movement": round(movement, 4),
            "max_inactive_seconds": round(max_inactive, 2),
            "sample_count_visual": float(len(v_window)),
            "sample_count_audio": float(len(a_window)),
        }
        if noise_db is not None:
            metadata["avg_noise_db"] = round(noise_db, 2)
            metadata["speech_fraction"] = round(speech_fraction or 0.0, 4)
        fused.append(
            FusedInterval(
                start_time=t,
                end_time=t_end,
                visual_score=round(visual_score, 4),
                audio_score=round(audio_score, 4),
                combined_score=round(combined, 4),
                metadata=metadata,
            )
        )
        t = t_end
    return fused
