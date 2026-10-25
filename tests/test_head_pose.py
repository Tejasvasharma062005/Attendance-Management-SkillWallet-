"""Unit tests for HeadPoseEstimator (Step 8 — head pose analysis)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.head_pose import (
    _forward_gaze_from_bbox,
    _NOSE_TIP,
    _CHIN,
    _LEFT_EYE_OUTER,
    _RIGHT_EYE_OUTER,
    _LEFT_EYE_INNER,
    _RIGHT_EYE_INNER,
)


# ---------------------------------------------------------------------------
# Helper: synthetic TrackedFace
# ---------------------------------------------------------------------------

class _FakeFace:
    def __init__(self, track_id: int, center, bbox=(100, 100, 80, 100)):
        self.track_id = track_id
        self.center = center
        self.bbox = bbox


# ---------------------------------------------------------------------------
# Tests for the bbox fallback (no model needed)
# ---------------------------------------------------------------------------

class TestForwardGazeFromBbox:
    """Test the geometric bounding-box fallback heuristic."""

    def test_symmetric_centered_face_scores_high(self):
        # w=80, h=100 → ratio ~0.8 ≈ ideal; cx=320 = centre of 640-wide frame
        score = _forward_gaze_from_bbox((280, 100, 80, 100), 640, 480)
        assert score >= 0.7, f"Expected >= 0.7, got {score}"

    def test_very_narrow_face_scores_low(self):
        # Very narrow → head turned sideways
        score = _forward_gaze_from_bbox((280, 100, 20, 100), 640, 480)
        assert score < 0.6, f"Expected < 0.6, got {score}"

    def test_face_far_off_centre_scores_lower(self):
        centred = _forward_gaze_from_bbox((280, 100, 80, 100), 640, 480)
        offside = _forward_gaze_from_bbox((0, 100, 80, 100), 640, 480)
        assert centred > offside

    def test_score_within_valid_range(self):
        for x in [0, 100, 500]:
            for w in [20, 80, 200]:
                score = _forward_gaze_from_bbox((x, 50, w, 100), 640, 480)
                assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_zero_height_does_not_crash(self):
        score = _forward_gaze_from_bbox((100, 100, 80, 0), 640, 480)
        assert score == 1.0


# ---------------------------------------------------------------------------
# Integration: HeadPoseEstimator with bbox fallback (no internet needed)
# ---------------------------------------------------------------------------

class TestHeadPoseEstimatorFallback:
    """Test HeadPoseEstimator in forced-fallback mode (no model download)."""

    def _make_estimator_fallback(self):
        """Create estimator and force it to use the bbox backend."""
        from src.vision.head_pose import HeadPoseEstimator, _BBoxFallbackBackend
        est = object.__new__(HeadPoseEstimator)
        est._backend = _BBoxFallbackBackend()
        est._use_landmarks = False
        return est

    def test_returns_empty_dict_for_no_faces(self):
        est = self._make_estimator_fallback()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = est.estimate_frame(frame, [])
        assert result == {}

    def test_returns_score_for_each_face(self):
        est = self._make_estimator_fallback()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = [
            _FakeFace(1, (320.0, 240.0), (280, 190, 80, 100)),
            _FakeFace(2, (100.0, 240.0), (60, 190, 80, 100)),
        ]
        result = est.estimate_frame(frame, faces)
        assert set(result.keys()) == {1, 2}
        for score in result.values():
            assert 0.0 <= score <= 1.0

    def test_track_ids_match_faces(self):
        est = self._make_estimator_fallback()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = [_FakeFace(42, (320.0, 240.0))]
        result = est.estimate_frame(frame, faces)
        assert 42 in result

    def test_close_does_not_raise(self):
        est = self._make_estimator_fallback()
        est.close()   # should not raise
