"""Unit tests for local JSON storage (Step 14 — offline, privacy-preserving storage)."""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestSessionStorage:
    """Tests for save_session_report / load_session_report using a temp dir."""

    def _patch_sessions_dir(self, tmp_path: Path):
        """Monkey-patch config.SESSIONS_DIR to an isolated temp directory."""
        import config
        self._orig = config.SESSIONS_DIR
        config.SESSIONS_DIR = tmp_path / "sessions"
        config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def _restore_sessions_dir(self):
        import config
        config.SESSIONS_DIR = self._orig

    # ------------------------------------------------------------------
    # save_session_report
    # ------------------------------------------------------------------

    def test_report_file_is_created(self, tmp_path):
        self._patch_sessions_dir(tmp_path)
        try:
            from src.storage.logs import save_session_report
            path = save_session_report("test_session", {"foo": "bar"})
            assert path.exists()
        finally:
            self._restore_sessions_dir()

    def test_report_is_valid_json(self, tmp_path):
        self._patch_sessions_dir(tmp_path)
        try:
            from src.storage.logs import save_session_report
            path = save_session_report("test_session", {"answer": 42})
            data = json.loads(path.read_text(encoding="utf-8"))
            assert "session_id" in data
            assert "saved_at" in data
            assert "payload" in data
        finally:
            self._restore_sessions_dir()

    def test_report_session_id_matches(self, tmp_path):
        self._patch_sessions_dir(tmp_path)
        try:
            from src.storage.logs import save_session_report
            path = save_session_report("session_abc", {"x": 1})
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["session_id"] == "session_abc"
        finally:
            self._restore_sessions_dir()

    def test_payload_is_preserved(self, tmp_path):
        self._patch_sessions_dir(tmp_path)
        try:
            from src.storage.logs import save_session_report
            payload = {"key": "value", "number": 3.14, "nested": {"a": [1, 2, 3]}}
            path = save_session_report("s1", payload)
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["payload"] == payload
        finally:
            self._restore_sessions_dir()

    def test_head_pose_score_is_saved_in_report(self, tmp_path):
        self._patch_sessions_dir(tmp_path)
        try:
            from src.agent.fusion import FusedInterval
            from src.storage.logs import save_session_report

            payload = {
                "analysis": {
                    "fused_intervals": [
                        FusedInterval(
                            start_time=0.0,
                            end_time=10.0,
                            visual_score=0.7,
                            audio_score=0.0,
                            combined_score=0.42,
                            metadata={"head_pose_score": 0.8},
                        )
                    ]
                }
            }
            path = save_session_report("head_pose_session", payload)
            report = json.loads(path.read_text(encoding="utf-8"))

            assert (
                report["payload"]["analysis"]["fused_intervals"][0]["metadata"][
                    "head_pose_score"
                ]
                == 0.8
            )
        finally:
            self._restore_sessions_dir()

    def test_saved_at_is_iso_format(self, tmp_path):
        self._patch_sessions_dir(tmp_path)
        try:
            from src.storage.logs import save_session_report
            from datetime import datetime
            path = save_session_report("s2", {})
            data = json.loads(path.read_text(encoding="utf-8"))
            # Should not raise
            datetime.fromisoformat(data["saved_at"])
        finally:
            self._restore_sessions_dir()

    # ------------------------------------------------------------------
    # load_session_report
    # ------------------------------------------------------------------

    def test_load_returns_none_for_missing_session(self, tmp_path):
        self._patch_sessions_dir(tmp_path)
        try:
            from src.storage.logs import load_session_report
            result = load_session_report("nonexistent_session")
            assert result is None
        finally:
            self._restore_sessions_dir()

    def test_roundtrip_save_and_load(self, tmp_path):
        self._patch_sessions_dir(tmp_path)
        try:
            from src.storage.logs import load_session_report, save_session_report
            payload = {"data": [1, 2, 3], "meta": "test"}
            save_session_report("rt_session", payload)
            loaded = load_session_report("rt_session")
            assert loaded is not None
            assert loaded["payload"] == payload
            assert loaded["session_id"] == "rt_session"
        finally:
            self._restore_sessions_dir()

    def test_multiple_sessions_are_independent(self, tmp_path):
        self._patch_sessions_dir(tmp_path)
        try:
            from src.storage.logs import load_session_report, save_session_report
            save_session_report("s_alpha", {"label": "alpha"})
            save_session_report("s_beta", {"label": "beta"})
            alpha = load_session_report("s_alpha")
            beta = load_session_report("s_beta")
            assert alpha["payload"]["label"] == "alpha"
            assert beta["payload"]["label"] == "beta"
        finally:
            self._restore_sessions_dir()
