"""Head pose estimation from facial landmarks.

Uses the MediaPipe Face Landmarker (Tasks API, mediapipe >= 0.10) to obtain
478-point face mesh landmarks.  From key anatomical points (nose tip, eye
corners, chin) we compute a *forward-gaze score* in [0, 1]:

    1.0  →  participant is looking straight at the camera
    0.0  →  head is turned / tilted away from the camera

Falls back to a bounding-box aspect-ratio heuristic when the model cannot
be downloaded (offline / no-internet).
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from src.vision.face_tracker import TrackedFace

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_LANDMARKER_PATH = config.PROJECT_ROOT / "data" / "face_landmarker.task"

# Key landmark indices in the 478-point MediaPipe face mesh
_NOSE_TIP = 1
_CHIN = 152
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_LEFT_EYE_INNER = 133
_RIGHT_EYE_INNER = 362


def _ensure_landmarker_model() -> Optional[Path]:
    if _LANDMARKER_PATH.exists():
        return _LANDMARKER_PATH
    try:
        _LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("[HeadPose] Downloading face landmarker model ...")
        urllib.request.urlretrieve(_LANDMARKER_URL, _LANDMARKER_PATH)
        print("[HeadPose] face_landmarker.task downloaded OK.")
        return _LANDMARKER_PATH
    except Exception as exc:
        print(f"[HeadPose] Model download failed ({exc}); using bbox fallback.")
        return None


# ---------------------------------------------------------------------------
# Forward-gaze computation helpers
# ---------------------------------------------------------------------------

def _forward_gaze_from_landmarks(
    landmarks,  # NormalizedLandmark list (len 478)
    frame_w: int,
    frame_h: int,
) -> float:
    """Return forward-gaze score in [0, 1] from 478-point landmarks."""
    def lm(idx: int) -> Tuple[float, float]:
        p = landmarks[idx]
        return p.x * frame_w, p.y * frame_h

    nose = lm(_NOSE_TIP)
    chin = lm(_CHIN)
    le_outer = lm(_LEFT_EYE_OUTER)
    re_outer = lm(_RIGHT_EYE_OUTER)
    le_inner = lm(_LEFT_EYE_INNER)
    re_inner = lm(_RIGHT_EYE_INNER)

    # --- Yaw proxy: nose offset from eye midpoint (horizontal) ---
    eye_mid_x = (le_outer[0] + re_outer[0]) / 2.0
    face_width = abs(le_outer[0] - re_outer[0]) + 1e-6
    yaw_offset = abs(nose[0] - eye_mid_x) / face_width  # 0 = centred

    # Asymmetry between inner and outer eye half-widths is a stronger yaw cue
    left_half = abs(le_outer[0] - le_inner[0])
    right_half = abs(re_outer[0] - re_inner[0])
    half_sum = left_half + right_half + 1e-6
    asymmetry = abs(left_half - right_half) / half_sum  # 0 = symmetric

    yaw_score = max(0.0, 1.0 - (0.5 * yaw_offset + 1.5 * asymmetry))

    # --- Pitch proxy: nose tip below or above chin-eye midpoint ---
    eye_mid_y = (le_outer[1] + re_outer[1]) / 2.0
    face_height = abs(chin[1] - eye_mid_y) + 1e-6
    nose_frac = (nose[1] - eye_mid_y) / face_height  # ~0.35 when centred
    pitch_dev = abs(nose_frac - 0.35) / 0.35
    pitch_score = max(0.0, 1.0 - pitch_dev)

    return round(float(0.6 * yaw_score + 0.4 * pitch_score), 4)


def _forward_gaze_from_bbox(
    bbox: Tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> float:
    """Geometric fallback: estimate forward gaze from face bounding box shape."""
    x, y, w, h = bbox
    if h == 0:
        return 1.0

    # A face viewed straight-on has a characteristic w/h ratio (~0.75–0.85).
    # Faces turned to the side become much narrower.
    ratio = w / h
    ideal = 0.80
    ratio_score = max(0.0, 1.0 - abs(ratio - ideal) / ideal)

    # Position score: face should be somewhere in the central 60 % of the frame.
    cx = x + w / 2.0
    centre_dev = abs(cx - frame_w / 2.0) / (frame_w / 2.0)
    pos_score = max(0.0, 1.0 - centre_dev)

    return round(float(0.7 * ratio_score + 0.3 * pos_score), 4)


def _match_landmark_scores(
    faces,
    landmark_results: List[Tuple[Tuple[float, float], float]],
    frame_w: int,
    frame_h: int,
) -> Dict[int, float]:
    """Pair each tracked face with at most one landmark result.

    A landmark result is only eligible for a face when its centre lies within
    0.75 x the face's larger bbox side, and pairs are taken closest-first so no
    result is used twice. A face the landmarker did not pick up (typically a
    strongly turned or partly occluded head) falls back to the bounding-box
    heuristic instead of being assumed to look straight ahead.
    """
    pairs: List[Tuple[float, int, int]] = []
    for face_index, face in enumerate(faces):
        cx, cy = face.center
        reach = 0.75 * max(face.bbox[2], face.bbox[3], 1)
        for result_index, ((lcx, lcy), _score) in enumerate(landmark_results):
            dist = float(np.hypot(cx - lcx, cy - lcy))
            if dist <= reach:
                pairs.append((dist, face_index, result_index))
    pairs.sort()

    scores: Dict[int, float] = {}
    used_faces: set = set()
    used_results: set = set()
    for _dist, face_index, result_index in pairs:
        if face_index in used_faces or result_index in used_results:
            continue
        used_faces.add(face_index)
        used_results.add(result_index)
        scores[faces[face_index].track_id] = landmark_results[result_index][1]

    for face in faces:
        if face.track_id not in scores:
            scores[face.track_id] = _forward_gaze_from_bbox(
                face.bbox, frame_w, frame_h
            )
    return scores


# ---------------------------------------------------------------------------
# Internal backends
# ---------------------------------------------------------------------------

class _LandmarkerBackend:
    def __init__(self, model_path: Path) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=10,
            min_face_detection_confidence=config.MIN_FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp

    def estimate_all(
        self, frame_bgr: np.ndarray
    ) -> List[Tuple[Tuple[float, float], float]]:
        """Return list of (face_centre_px, forward_gaze_score) for every face."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=frame_rgb
        )
        result = self._landmarker.detect(mp_image)
        output = []
        for face_lms in (result.face_landmarks or []):
            if len(face_lms) < 478:
                continue
            # Compute face centre from bounding box of all landmarks
            xs = [lm.x * w for lm in face_lms]
            ys = [lm.y * h for lm in face_lms]
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            score = _forward_gaze_from_landmarks(face_lms, w, h)
            output.append(((cx, cy), score))
        return output

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


class _BBoxFallbackBackend:
    """No model needed — uses bounding box geometry only."""

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Public HeadPoseEstimator
# ---------------------------------------------------------------------------

@dataclass
class HeadPoseEstimator:
    """Estimate forward-gaze score for each tracked face in a frame.

    Call ``estimate_frame(frame_bgr, tracked_faces)`` once per video frame
    to get a ``Dict[track_id, forward_gaze_score]``.
    """

    _backend: object = field(default=None, init=False)
    _use_landmarks: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        model_path = _ensure_landmarker_model()
        if model_path is not None:
            try:
                self._backend = _LandmarkerBackend(model_path)
                self._use_landmarks = True
                print("[HeadPose] Using MediaPipe Face Landmarker backend.")
                return
            except Exception as exc:
                print(f"[HeadPose] Landmarker backend failed ({exc}); using bbox fallback.")
        self._backend = _BBoxFallbackBackend()
        self._use_landmarks = False
        print("[HeadPose] Using bounding-box geometry fallback.")

    def estimate_frame(
        self,
        frame_bgr: np.ndarray,
        tracked_faces: List[TrackedFace],
    ) -> Dict[int, float]:
        """Return a mapping of track_id → forward_gaze_score for this frame."""
        if not tracked_faces:
            return {}

        h, w = frame_bgr.shape[:2]

        if self._use_landmarks:
            lm_results = self._backend.estimate_all(frame_bgr)
            return _match_landmark_scores(tracked_faces, lm_results, w, h)

        # bbox fallback
        return {
            face.track_id: _forward_gaze_from_bbox(face.bbox, w, h)
            for face in tracked_faces
        }

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
