"""Timestamped attendance logs generated from presence records.

Each participant yields up to three events: ``first_seen``, ``marked_present``
(the moment sustained presence was confirmed) and ``last_seen``. Session times
are always available. Wall-clock UTC timestamps are added only when the session
start was a real clock time (live webcam sessions); for a recorded video the
real recording time is unknown, so no clock time is invented.
"""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

ATTENDANCE_LOG_FIELDS = [
    "session_id",
    "participant",
    "track_id",
    "event",
    "session_time_s",
    "session_time",
    "timestamp_utc",
    "details",
]

_EVENT_ORDER = {"first_seen": 0, "marked_present": 1, "last_seen": 2}


def format_session_time(seconds: float | None) -> str:
    """Format session seconds as ``MM:SS`` (``H:MM:SS`` beyond an hour)."""
    if seconds is None:
        return ""
    total = max(0, int(round(float(seconds))))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _as_dict(record: Any) -> Dict[str, Any]:
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)
    return dict(record)


def _clock(session_start_utc: Optional[str], seconds: float | None) -> Optional[str]:
    if not session_start_utc or seconds is None:
        return None
    try:
        start = datetime.fromisoformat(session_start_utc)
    except ValueError:
        return None
    return (start + timedelta(seconds=float(seconds))).isoformat(timespec="seconds")


def build_attendance_log(
    attendance: Iterable[Any],
    session_id: str = "",
    session_start_utc: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return chronological attendance events for the given presence records.

    ``attendance`` may hold ``AttendanceRecord`` dataclasses or the dicts found
    in a saved report, so exports of older reports keep working. Reports saved
    before presence confirmation times were recorded simply have no
    ``marked_present`` event; a time is never guessed.
    """
    events: List[Dict[str, Any]] = []

    def add(record: Dict[str, Any], event: str, seconds: float | None, details: str) -> None:
        if seconds is None:
            return
        events.append(
            {
                "session_id": session_id,
                "participant": f"Participant {record.get('track_id')}",
                "track_id": record.get("track_id"),
                "event": event,
                "session_time_s": round(float(seconds), 2),
                "session_time": format_session_time(seconds),
                "timestamp_utc": _clock(session_start_utc, seconds),
                "details": details,
            }
        )

    for item in attendance:
        record = _as_dict(item)
        visible = record.get("total_visible_seconds", 0.0)
        threshold = record.get("threshold_seconds")
        add(record, "first_seen", record.get("first_seen"), "Face first detected.")
        if record.get("present"):
            needed = f"{threshold:g}s" if threshold is not None else "the threshold"
            add(
                record,
                "marked_present",
                record.get("marked_present_at"),
                f"Sustained presence confirmed after {needed} of visibility.",
            )
        add(
            record,
            "last_seen",
            record.get("last_seen"),
            f"Total visible time {float(visible):.1f}s; "
            f"{'present' if record.get('present') else 'below attendance threshold'}.",
        )

    events.sort(
        key=lambda e: (e["session_time_s"], _EVENT_ORDER[e["event"]], e["track_id"] or 0)
    )
    return events


def attendance_log_to_csv(events: Iterable[Dict[str, Any]]) -> str:
    """Serialise attendance events as CSV text."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=ATTENDANCE_LOG_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(events)
    return output.getvalue()
