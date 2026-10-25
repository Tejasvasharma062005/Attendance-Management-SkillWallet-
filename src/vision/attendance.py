"""Presence-based attendance from sustained face visibility.

Attendance is never decided from a single frame. Visibility is accumulated per
participant, and a participant is marked present only once the accumulated
time reaches a threshold. The engine also remembers *when* that happened so a
timestamped attendance log can be produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config

# Cumulative-visibility checkpoints are stored every 0.25 s of accumulated
# time (not every frame) so hour-long recordings stay cheap in memory.
_CHECKPOINT_RESOLUTION = 4


@dataclass
class AttendanceRecord:
    track_id: int
    total_visible_seconds: float
    present: bool
    first_seen: float | None = None
    last_seen: float | None = None
    # Session time (seconds) at which accumulated visibility reached the
    # attendance threshold; None while the participant is not yet present.
    marked_present_at: float | None = None
    # Threshold (seconds) that was applied when ``present`` was decided.
    threshold_seconds: float | None = None


@dataclass
class PresenceAttendanceEngine:
    """Accumulate visibility duration per participant; mark present above threshold."""

    threshold_seconds: float = config.PRESENCE_THRESHOLD_SECONDS
    max_session_fraction: float = config.PRESENCE_MAX_SESSION_FRACTION
    _visible_seconds: Dict[int, float] = field(default_factory=dict)
    _last_timestamp: Dict[int, float] = field(default_factory=dict)
    _first_seen: Dict[int, float] = field(default_factory=dict)
    _last_seen: Dict[int, float] = field(default_factory=dict)
    _checkpoints: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)

    def update(self, track_ids: List[int], timestamp: float, frame_delta: float) -> None:
        seen = set(track_ids)
        for tid in seen:
            if tid not in self._first_seen:
                self._first_seen[tid] = timestamp
            self._last_seen[tid] = timestamp
            before = self._visible_seconds.get(tid, 0.0)
            after = before + frame_delta
            self._visible_seconds[tid] = after
            self._last_timestamp[tid] = timestamp
            if int(after * _CHECKPOINT_RESOLUTION) > int(before * _CHECKPOINT_RESOLUTION):
                self._checkpoints.setdefault(tid, []).append((timestamp, after))

        # Tracks not seen this frame do not accumulate time (sustained presence model)

    def effective_threshold(self, session_duration: float | None = None) -> float:
        """Threshold actually applied for a session of the given length.

        The configured threshold (30 s by default) is capped at a fraction of
        the session length, so a 20-second clip is not impossible to attend.
        With no duration the configured threshold is used unchanged.
        """
        if session_duration is None or session_duration <= 0:
            return self.threshold_seconds
        return min(self.threshold_seconds, session_duration * self.max_session_fraction)

    def _marked_present_at(self, tid: int, threshold: float) -> Optional[float]:
        """Session time at which accumulated visibility first reached ``threshold``."""
        for timestamp, cumulative in self._checkpoints.get(tid, []):
            if cumulative >= threshold:
                return timestamp
        # Threshold reached between checkpoints (e.g. a very small threshold).
        if self._visible_seconds.get(tid, 0.0) >= threshold:
            return self._last_seen.get(tid)
        return None

    def current_records(self, session_duration: float | None = None) -> List[AttendanceRecord]:
        """Return attendance records for visibility accumulated so far."""
        threshold = self.effective_threshold(session_duration)
        records: List[AttendanceRecord] = []
        for tid, seconds in sorted(self._visible_seconds.items()):
            present = seconds >= threshold
            records.append(
                AttendanceRecord(
                    track_id=tid,
                    total_visible_seconds=round(seconds, 2),
                    present=present,
                    first_seen=self._first_seen.get(tid),
                    last_seen=self._last_seen.get(tid),
                    marked_present_at=(
                        self._marked_present_at(tid, threshold) if present else None
                    ),
                    threshold_seconds=round(threshold, 2),
                )
            )
        return records

    def finalize(self, session_duration: float | None = None) -> List[AttendanceRecord]:
        return self.current_records(session_duration)
