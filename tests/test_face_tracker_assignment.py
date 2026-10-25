"""Tracker ID assignment: one detection per track per frame, stable across frames."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.face_tracker import FaceTracker


def _tracker(max_distance: float = 80.0) -> FaceTracker:
    """Bypass model loading; only the assignment logic is under test."""
    tracker = object.__new__(FaceTracker)
    tracker.max_distance = max_distance
    tracker._next_id = 1
    tracker._tracks = {}
    return tracker


def test_two_nearby_faces_never_share_an_id():
    tracker = _tracker()
    tracker._assign_ids([(100.0, 100.0)])                 # one track exists
    ids = tracker._assign_ids([(105.0, 100.0), (150.0, 100.0)])  # both within 80px of it
    assert len(set(ids)) == 2


def test_ids_stay_stable_while_faces_drift_slightly():
    tracker = _tracker()
    first = tracker._assign_ids([(100.0, 100.0), (400.0, 100.0)])
    second = tracker._assign_ids([(104.0, 102.0), (396.0, 99.0)])
    assert first == second


def test_ids_do_not_swap_when_detection_order_changes():
    tracker = _tracker()
    a, b = tracker._assign_ids([(100.0, 100.0), (400.0, 100.0)])
    swapped = tracker._assign_ids([(401.0, 100.0), (101.0, 100.0)])
    assert swapped == [b, a]


def test_participant_returning_to_same_seat_resumes_their_id():
    tracker = _tracker()
    first = tracker._assign_ids([(100.0, 100.0)])[0]
    tracker._assign_ids([])                               # briefly occluded / out of frame
    tracker._assign_ids([])
    assert tracker._assign_ids([(110.0, 98.0)])[0] == first


def test_far_detection_opens_a_new_track():
    tracker = _tracker()
    first = tracker._assign_ids([(100.0, 100.0)])[0]
    second = tracker._assign_ids([(500.0, 300.0)])[0]
    assert second != first
