"""Visual engagement cues: head pose, movement, inactivity, and proxy attention.

The attention proxy now incorporates three signals:
  - head_pose_score  (0 = looking away, 1 = looking forward)  weight 0.50
  - inactivity       (sustained stillness → disengagement)      weight 0.30
  - movement         (gentle movement → alertness proxy)        weight 0.20
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

import config


@dataclass
class VisualEngagementSample:
    timestamp: float
    track_id: int
    movement_score: float
    inactive_seconds: float
    head_pose_score: float      # NEW: forward-gaze score in [0, 1]
    attention_proxy: float


@dataclass
class VisualEngagementAnalyzer:
    """Per-participant movement, inactivity, and head-pose tracking."""

    inactivity_limit: float = config.INACTIVITY_SECONDS
    movement_threshold: float = config.MOVEMENT_THRESHOLD
    _prev_center: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    _last_move_time: Dict[int, float] = field(default_factory=dict)
    _frame_size: Tuple[int, int] | None = None

    def set_frame_size(self, width: int, height: int) -> None:
        self._frame_size = (width, height)

    def update(
        self,
        track_id: int,
        center: Tuple[float, float],
        timestamp: float,
        head_pose_score: float = 1.0,
    ) -> VisualEngagementSample:
        w, h = self._frame_size or (640, 480)
        diag = float(np.hypot(w, h)) or 1.0

        prev = self._prev_center.get(track_id)
        movement_score = 0.0
        if prev is not None:
            dist = np.hypot(center[0] - prev[0], center[1] - prev[1]) / diag
            movement_score = min(1.0, dist / self.movement_threshold)

        if movement_score > 0.3 or track_id not in self._last_move_time:
            self._last_move_time[track_id] = timestamp

        inactive = timestamp - self._last_move_time.get(track_id, timestamp)
        self._prev_center[track_id] = center

        # Attention proxy incorporates head pose, inactivity, and movement
        inactivity_ratio = min(1.0, inactive / self.inactivity_limit)
        attention = (
            0.50 * head_pose_score
            + 0.30 * (1.0 - inactivity_ratio)
            + 0.20 * min(1.0, movement_score)
        )
        attention = round(float(np.clip(attention, 0.0, 1.0)), 4)

        return VisualEngagementSample(
            timestamp=timestamp,
            track_id=track_id,
            movement_score=round(movement_score, 4),
            inactive_seconds=round(inactive, 2),
            head_pose_score=round(head_pose_score, 4),
            attention_proxy=attention,
        )

    def session_interval_metrics(
        self, samples: List[VisualEngagementSample], t_start: float, t_end: float
    ) -> Dict[str, float]:
        window = [s for s in samples if t_start <= s.timestamp < t_end]
        if not window:
            return {
                "avg_attention": 0.0,
                "avg_movement": 0.0,
                "max_inactivity": 0.0,
                "avg_head_pose": 0.0,
            }
        return {
            "avg_attention": float(np.mean([s.attention_proxy for s in window])),
            "avg_movement": float(np.mean([s.movement_score for s in window])),
            "max_inactivity": float(max(s.inactive_seconds for s in window)),
            "avg_head_pose": float(np.mean([s.head_pose_score for s in window])),
        }
