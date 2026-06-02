"""Shared State + Pfade für den rubin-Server.

Zentralisiert Server-State, Pfad-Konstanten und Hilfsfunktionen,
die von allen Route-Modulen (Blueprints) verwendet werden.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path

# ── Version ──
__version__ = "1.0.0"

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="[rubin] %(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rubin")

# ── Pfade ──
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
FRONTEND = APP_DIR / "frontend"

# Arbeitsverzeichnis: RUBIN_WORK_DIR (Env) > ROOT/runs (Default)
_work_dir_env = os.environ.get("RUBIN_WORK_DIR")
WORK_DIR = Path(_work_dir_env).resolve() if _work_dir_env else ROOT / "runs"
WORK_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = WORK_DIR / "uploads"
PROGRESS_FILE = WORK_DIR / ".rubin_progress.json"
MAX_UPLOAD_MB = 500  # Upload-Limit in MB (→ server.py MAX_CONTENT_LENGTH)

# ── Server-State (Thread-safe) ──
_state = {
    "status": "idle",
    "task": None,
    "generation": 0,
    "message": "",
    "step": "",
    "step_index": 0,
    "total_steps": 0,
    "percent": 0,
    "stdout_tail": "",
    "stderr_tail": "",
    "result_files": [],
    "pid": None,
}
_state_lock = threading.RLock()  # RLock: reentrant-safe (verhindert Deadlocks bei verschachtelten Aufrufen)


def set_state(**kw):
    """Thread-safe State-Update + Disk-Persistenz."""
    with _state_lock:
        _state.update(kw)
        try:
            PROGRESS_FILE.write_text(json.dumps(_state, default=str), encoding="utf-8")
        except Exception:
            pass


def get_state() -> dict:
    """Thread-safe State-Snapshot."""
    with _state_lock:
        return dict(_state)


def increment_generation() -> int:
    """Inkrementiert die Task-Generation und gibt den neuen Wert zurück."""
    with _state_lock:
        _state["generation"] += 1
        return _state["generation"]


# ── Hilfsfunktionen ──
def safe_str(v):
    """Konvertiert einen Wert sicher zu str (NaN-safe, numpy-safe)."""
    try:
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return str(v) if not isinstance(v, (int, float, bool)) else v
    except Exception:
        return str(v)
