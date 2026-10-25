"""Face detection and cross-frame tracking.

Supports the new MediaPipe Tasks API (mediapipe >= 0.10.14) with an automatic
model download on first use, and falls back to OpenCV DNN face detector if the
MediaPipe model cannot be fetched.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

# ---------------------------------------------------------------------------
# Model URLs and local paths
# ---------------------------------------------------------------------------

_MP_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)
_MP_MODEL_PATH = config.PROJECT_ROOT / "data" / "face_detector.tflite"

# OpenCV DNN face detector (Caffe model — ~2 MB, much lighter fallback)
_DNN_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
_DNN_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)
_DNN_PROTO_PATH = config.PROJECT_ROOT / "data" / "face_deploy.prototxt"
_DNN_MODEL_PATH = config.PROJECT_ROOT / "data" / "face_res10.caffemodel"


def _download(url: str, dest: Path) -> bool:
    """Download *url* to *dest*. Returns True on success."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[FaceTracker] Downloading {dest.name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[FaceTracker] {dest.name} downloaded OK.")
        return True
    except Exception as exc:
        print(f"[FaceTracker] Download failed for {dest.name}: {exc}")
        return False


def _ensure_mp_model() -> Optional[Path]:
    if _MP_MODEL_PATH.exists():
        return _MP_MODEL_PATH
    ok = _download(_MP_MODEL_URL, _MP_MODEL_PATH)
    return _MP_MODEL_PATH if ok else None


def _ensure_dnn_model() -> bool:
    ok_proto = _DNN_PROTO_PATH.exists() or _download(_DNN_PROTO_URL, _DNN_PROTO_PATH)
    ok_model = _DNN_MODEL_PATH.exists() or _download(_DNN_MODEL_URL, _DNN_MODEL_PATH)
    return ok_proto and ok_model


# ---------------------------------------------------------------------------
# Shared data type (unchanged public interface)
# ---------------------------------------------------------------------------

@dataclass
class TrackedFace:
    track_id: int
    bbox: Tuple[int, int, int, int]   # x, y, w, h in pixels
    center: Tuple[float, float]
    timestamp: float
    detection_confidence: float


# ---------------------------------------------------------------------------
# Internal backends
# ---------------------------------------------------------------------------

class _MediaPipeBackend:
    """Wraps the new mediapipe.tasks vision.FaceDetector."""

    def __init__(self, model_path: Path, min_confidence: float) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self._mp = mp

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=frame_rgb
        )
        result = self._detector.detect(mp_image)
        detections = []
        for det in (result.detections or []):
            bb = det.bounding_box
            score = det.categories[0].score if det.categories else 0.9
            detections.append((bb.origin_x, bb.origin_y, bb.width, bb.height, float(score)))
        return detections

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:
            pass


class _DNNBackend:
    """OpenCV DNN face detector fallback (ResNet-SSD Caffe model)."""

    def __init__(self, min_confidence: float) -> None:
        self._min_conf = min_confidence
        self._net = cv2.dnn.readNetFromCaffe(
            str(_DNN_PROTO_PATH), str(_DNN_MODEL_PATH)
        )

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self._net.setInput(blob)
        detections = self._net.forward()
        results = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self._min_conf:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw > 0 and bh > 0:
                results.append((x1, y1, bw, bh, conf))
        return results

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Public FaceTracker (same interface as before)
# ---------------------------------------------------------------------------

@dataclass
class FaceTracker:
    """Detect faces and assign stable IDs across frames.

    Backend priority:
      1. MediaPipe Tasks FaceDetector (downloads ~4 MB model on first use)
      2. OpenCV DNN ResNet-SSD (downloads ~2 MB model on first use)
    """

    max_distance: float = config.TRACKING_MAX_DISTANCE
    _next_id: int = field(default=1, init=False)
    _tracks: Dict[int, Tuple[float, float]] = field(default_factory=dict, init=False)
    _backend: object = field(default=None, init=False)

    def __post_init__(self) -> None:
        # --- Try MediaPipe Tasks backend first ---
        model_path = _ensure_mp_model()
        if model_path is not None:
            try:
                self._backend = _MediaPipeBackend(
                    model_path, config.MIN_FACE_DETECTION_CONFIDENCE
                )
                print("[FaceTracker] Using MediaPipe Tasks backend.")
                return
            except Exception as exc:
                print(f"[FaceTracker] MediaPipe backend init failed: {exc}")

        # --- Fall back to OpenCV DNN ---
        print("[FaceTracker] Falling back to OpenCV DNN backend.")
        if _ensure_dnn_model():
            try:
                self._backend = _DNNBackend(config.MIN_FACE_DETECTION_CONFIDENCE)
                print("[FaceTracker] Using OpenCV DNN backend.")
                return
            except Exception as exc:
                print(f"[FaceTracker] OpenCV DNN backend failed: {exc}")

        raise RuntimeError(
            "No face detection backend available. "
            "Check your internet connection for the first run."
        )

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()

    def _assign_ids(self, centers: List[Tuple[float, float]]) -> List[int]:
        """Assign a track ID to every detection in one frame, one-to-one.

        Candidate (detection, track) pairs within ``max_distance`` are matched
        closest-first, and a track can be claimed by at most one detection per
        frame. (Matching each detection to its nearest track independently let
        two neighbouring faces share one ID and let IDs swap.) Unmatched
        detections open new tracks. Tracks are kept when a face is briefly
        absent, so a participant who is occluded or leaves and returns to the
        same seat resumes their ID and keeps accumulating presence time.
        """
        pairs: List[Tuple[float, int, int]] = []
        for det_index, (cx, cy) in enumerate(centers):
            for track_id, (px, py) in self._tracks.items():
                dist = float(np.hypot(cx - px, cy - py))
                if dist <= self.max_distance:
                    pairs.append((dist, det_index, track_id))
        pairs.sort()

        assigned: List[Optional[int]] = [None] * len(centers)
        claimed: set = set()
        for _dist, det_index, track_id in pairs:
            if assigned[det_index] is None and track_id not in claimed:
                assigned[det_index] = track_id
                claimed.add(track_id)

        ids: List[int] = []
        for det_index, center in enumerate(centers):
            track_id = assigned[det_index]
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
            self._tracks[track_id] = center
            ids.append(track_id)
        return ids

    def _assign_id(self, center: Tuple[float, float]) -> int:
        """Single-detection convenience wrapper around :meth:`_assign_ids`."""
        return self._assign_ids([center])[0]

    def process_frame(
        self, frame_bgr: np.ndarray, timestamp: float
    ) -> List[TrackedFace]:
        raw = self._backend.detect(frame_bgr)
        centers = [(x + w / 2.0, y + h / 2.0) for x, y, w, h, _conf in raw]
        track_ids = self._assign_ids(centers)
        faces: List[TrackedFace] = []
        for (x, y, w, h, conf), (cx, cy), track_id in zip(raw, centers, track_ids):
            faces.append(
                TrackedFace(
                    track_id=track_id,
                    bbox=(x, y, w, h),
                    center=(cx, cy),
                    timestamp=timestamp,
                    detection_confidence=conf,
                )
            )
        return faces
