"""Landmark-to-face matching for head pose (one-to-one, bbox fallback)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.head_pose import _forward_gaze_from_bbox, _match_landmark_scores


class _Face:
    def __init__(self, track_id, center, bbox):
        self.track_id, self.center, self.bbox = track_id, center, bbox


def test_each_face_gets_its_own_nearest_landmark_score():
    faces = [
        _Face(1, (100.0, 100.0), (60, 60, 80, 80)),
        _Face(2, (400.0, 100.0), (360, 60, 80, 80)),
    ]
    results = [((402.0, 101.0), 0.2), ((101.0, 99.0), 0.9)]
    scores = _match_landmark_scores(faces, results, 640, 480)
    assert scores == {1: 0.9, 2: 0.2}


def test_one_landmark_result_is_never_used_for_two_faces():
    faces = [
        _Face(1, (100.0, 100.0), (60, 60, 80, 80)),
        _Face(2, (130.0, 100.0), (90, 60, 80, 80)),
    ]
    results = [((110.0, 100.0), 0.3)]                      # only one mesh was found
    scores = _match_landmark_scores(faces, results, 640, 480)
    matched = [tid for tid, s in scores.items() if s == 0.3]
    assert len(matched) == 1


def test_face_without_landmarks_falls_back_to_bbox_not_perfect_gaze():
    face = _Face(7, (320.0, 240.0), (300, 190, 20, 100))   # very narrow = turned head
    scores = _match_landmark_scores([face], [], 640, 480)
    assert scores[7] == _forward_gaze_from_bbox(face.bbox, 640, 480)
    assert scores[7] < 1.0


def test_distant_landmark_result_is_not_borrowed():
    face = _Face(1, (100.0, 100.0), (60, 60, 80, 80))
    results = [((600.0, 400.0), 0.05)]                     # a different person far away
    scores = _match_landmark_scores([face], results, 640, 480)
    assert scores[1] != 0.05
