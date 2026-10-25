"""Unit tests for PresenceAttendanceEngine (Step 6 — sustained presence logic)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.attendance import AttendanceRecord, PresenceAttendanceEngine


class TestPresenceAttendanceEngine:
    """Tests for the sustained-presence attendance model."""

    def _make_engine(self, threshold: float = 5.0) -> PresenceAttendanceEngine:
        return PresenceAttendanceEngine(threshold_seconds=threshold)

    # ------------------------------------------------------------------
    # Basic presence accumulation
    # ------------------------------------------------------------------

    def test_single_participant_accumulates_time(self):
        engine = self._make_engine(threshold=5.0)
        fps = 25.0
        frame_delta = 1.0 / fps
        for i in range(int(6 * fps)):          # 6 seconds visible
            engine.update([1], i / fps, frame_delta)
        records = engine.finalize()
        assert len(records) == 1
        assert records[0].track_id == 1
        assert records[0].total_visible_seconds > 5.9

    def test_participant_marked_present_above_threshold(self):
        engine = self._make_engine(threshold=3.0)
        frame_delta = 1.0
        for i in range(4):                     # 4 seconds visible
            engine.update([42], float(i), frame_delta)
        records = engine.finalize()
        assert records[0].present is True

    def test_participant_marked_absent_below_threshold(self):
        engine = self._make_engine(threshold=10.0)
        frame_delta = 1.0
        for i in range(3):                     # only 3 seconds visible
            engine.update([7], float(i), frame_delta)
        records = engine.finalize()
        assert records[0].present is False

    # ------------------------------------------------------------------
    # Multiple participants
    # ------------------------------------------------------------------

    def test_multiple_participants_independent_tracking(self):
        engine = self._make_engine(threshold=2.0)
        frame_delta = 1.0
        # participant 1 visible for 3 s, participant 2 for 1 s
        for i in range(3):
            engine.update([1], float(i), frame_delta)
        for i in range(1):
            engine.update([2], float(i), frame_delta)
        records = {r.track_id: r for r in engine.finalize()}
        assert records[1].present is True
        assert records[2].present is False

    # ------------------------------------------------------------------
    # Timestamp metadata
    # ------------------------------------------------------------------

    def test_first_seen_last_seen_recorded(self):
        engine = self._make_engine()
        engine.update([5], 0.0, 1.0)
        engine.update([5], 1.0, 1.0)
        engine.update([5], 2.0, 1.0)
        records = engine.finalize()
        assert records[0].first_seen == 0.0
        assert records[0].last_seen == 2.0

    def test_empty_session_returns_empty_list(self):
        engine = self._make_engine()
        assert engine.finalize() == []

    # ------------------------------------------------------------------
    # Sustained-presence model: gaps do NOT accumulate time
    # ------------------------------------------------------------------

    def test_gaps_do_not_accumulate_time(self):
        """A participant absent for frames should not gain time during those frames."""
        engine = self._make_engine(threshold=5.0)
        frame_delta = 1.0
        # visible frames 0-2, absent 3-5, visible 6-8 → total ~6 s
        for i in list(range(3)) + list(range(6, 9)):
            engine.update([1], float(i), frame_delta)
        # absent frames — update with different participant so engine runs
        for i in range(3, 6):
            engine.update([99], float(i), frame_delta)
        records = {r.track_id: r for r in engine.finalize()}
        # participant 1 should have ~6 s (not 9)
        assert abs(records[1].total_visible_seconds - 6.0) < 0.1
