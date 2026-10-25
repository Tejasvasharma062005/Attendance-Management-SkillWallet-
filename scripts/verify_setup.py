"""Epic 1 validation: confirm dependencies, local models and the core pipeline.

Run from the project root:  python scripts/verify_setup.py
Add --download-models to fetch the vision models once (needs internet);
after that the platform runs fully offline.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _check_imports() -> list:
    checks = [
        ("opencv", lambda: __import__("cv2")),
        ("mediapipe", lambda: __import__("mediapipe")),
        ("librosa", lambda: __import__("librosa")),
        ("sounddevice", lambda: __import__("sounddevice")),
        ("streamlit", lambda: __import__("streamlit")),
        ("pandas", lambda: __import__("pandas")),
        ("plotly", lambda: __import__("plotly")),
        ("reportlab", lambda: __import__("reportlab")),
        ("numpy", lambda: __import__("numpy")),
    ]
    failed = []
    for name, fn in checks:
        try:
            fn()
            print(f"[OK] {name}")
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            failed.append(name)
    return failed


def _check_models(download: bool) -> bool:
    import config
    from src.vision import face_tracker, head_pose

    if download:
        print("Downloading vision models (one-time, needs internet) ...")
        face_tracker._ensure_mp_model()
        head_pose._ensure_landmarker_model()

    models = [
        ("face detector", config.DATA_DIR / "face_detector.tflite"),
        ("face landmarker (head pose)", config.DATA_DIR / "face_landmarker.task"),
    ]
    ready = True
    for label, path in models:
        if path.exists():
            print(f"[OK] model present: {label}")
        else:
            ready = False
            print(
                f"[WARN] model missing: {label}. It downloads automatically on first "
                "use, or run: python scripts/verify_setup.py --download-models"
            )
    return ready


def _smoke_test_pipeline() -> bool:
    """Exercise fusion, scoring, attendance and the audio classifier on synthetic data."""
    import numpy as np

    from src.agent.multimodal_agent import MultimodalAgent
    from src.audio.analyzer import AudioAnalyzer
    from src.audio.extractor import AudioExtractor
    from src.vision.attendance import PresenceAttendanceEngine
    from src.vision.engagement_visual import VisualEngagementSample

    engine = PresenceAttendanceEngine()
    samples = []
    for i in range(200):
        engine.update([1], i / 10, 0.1)
        samples.append(VisualEngagementSample(i / 10, 1, 0.1, 0.0, 0.9, 0.8))
    sr = 22050
    audio = np.zeros(sr * 20, dtype=np.float32)
    segments = AudioExtractor().to_segments(audio, sr)
    metrics = AudioAnalyzer().analyze_all(segments)
    result = MultimodalAgent().analyze(engine.finalize(20.0), samples, metrics, 20.0)
    ok = bool(result.fused_intervals) and result.attendance[0].present
    print(f"[{'OK' if ok else 'FAIL'}] core pipeline smoke test")
    return ok


def main() -> None:
    failed = _check_imports()

    import config  # noqa: F401
    from src.agent.multimodal_agent import MultimodalAgent  # noqa: F401

    print("[OK] project modules")
    models_ready = _check_models("--download-models" in sys.argv)
    smoke_ok = False if failed else _smoke_test_pipeline()

    print(
        "[OK] ffmpeg found (audio can be read from video containers)"
        if shutil.which("ffmpeg")
        else "[INFO] ffmpeg not found: audio inside .mp4/.mov files may be unreadable. "
        "Install ffmpeg or supply a separate audio file."
    )

    if failed or not smoke_ok:
        sys.exit(1)
    print(
        "Environment ready."
        if models_ready
        else "Environment ready (vision models will download on first use)."
    )


if __name__ == "__main__":
    main()
