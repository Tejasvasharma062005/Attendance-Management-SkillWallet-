"""Local JSON persistence for session outputs (offline / privacy-preserving)."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config


def _serialize(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def save_session_report(session_id: str, payload: dict) -> Path:
    out_dir = config.SESSIONS_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.json"
    envelope = {
        "session_id": session_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "payload": _serialize(payload),
    }
    path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return path


def load_session_report(session_id: str) -> dict | None:
    path = config.SESSIONS_DIR / session_id / "report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
