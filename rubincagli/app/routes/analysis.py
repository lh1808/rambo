"""Analyse-Routen: Background-Tasks, Progress, Reset."""
from __future__ import annotations

import json
import logging
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from flask import Blueprint, jsonify, request

from app.state import (
    log, ROOT, WORK_DIR, UPLOAD_DIR, PROGRESS_FILE,
    get_state, set_state, increment_generation,
    _state, _state_lock,
)

bp = Blueprint("analysis", __name__)

def _run_in_background(task_name: str, cmd: list[str], timeout: int = 3600):
    """Startet einen Subprocess im Background-Thread mit Fortschritts-Tracking."""
    log.info("Task gestartet: %s (timeout=%ds)", task_name, timeout)
    gen = increment_generation()
    set_state(
        status="running", task=task_name, message=f"{task_name} gestartet...",
        step="", step_index=0, total_steps=0, percent=0,
        stdout_tail="", stderr_tail="", result_files=[], pid=None,
    )

    _last_disk_write = [0.0]  # Timestamp des letzten Disk-Writes

    def _guarded_set(**kw):
        """Setzt State nur wenn diese Task-Generation noch aktuell ist.
        Schreibt nur alle 0.5 Sekunden auf Disk (Throttle), um I/O-Overhead
        zu vermeiden wenn LightGBM/Optuna hunderte Zeilen/Sekunde auf stdout schreibt."""
        now = time.time()
        force_write = kw.get("status") in ("done", "error")
        with _state_lock:
            if get_state()["generation"] != gen:
                return
            _state.update(kw)
            # Disk-Write nur bei Status-Änderung oder alle 0.5 Sekunden
            if force_write or (now - _last_disk_write[0]) >= 0.5:
                _last_disk_write[0] = now
                try:
                    PROGRESS_FILE.write_text(json.dumps(_state, default=str), encoding="utf-8")
                except Exception:
                    pass

    def _worker():
        try:
            import select

            # start_new_session=True → Eigene Prozessgruppe.
            # os.killpg() beendet auch alle Kind-Prozesse
            # (LightGBM n_jobs=-1, joblib Worker, OpenMP Threads).
            #
            # PERFORMANCE: KEIN PYTHONUNBUFFERED!
            # Die Pipeline nutzt print(flush=True) für [rubin]-Progress-Zeilen.
            # LightGBM-Output (~500 Zeilen/s) wird vom Python-Buffer gesammelt
            # und in Batches geschrieben → dramatisch weniger Pipe-Syscalls.
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
                cwd=str(ROOT),
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT),
                    # Performance: LightGBM C++ stdout unterdrücken
                    "LIGHTGBM_VERBOSITY": "-1",
                    # Optuna-Logging reduzieren (nur Warnungen)
                    "OPTUNA_VERBOSITY": "WARNING",
                },
                start_new_session=True,
            )
            _guarded_set(pid=proc.pid)

            stdout_lines = []
            stderr_lines = []

            def _drain_stderr():
                try:
                    for line in iter(proc.stderr.readline, ""):
                        stderr_lines.append(line.rstrip())
                        if len(stderr_lines) > 500:
                            stderr_lines[:] = stderr_lines[-300:]
                except Exception:
                    pass

            stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

            # PERFORMANCE-OPTIMIERTES Stdout-Lesen:
            # Mit OS-level fd-Redirect kommen nur noch [rubin]-Zeilen an (~10 pro Lauf).
            # Jede [rubin]-Zeile aktualisiert sofort Step/Percent/Tail.
            stdout_fd = proc.stdout.fileno()

            while True:
                ready, _, _ = select.select([stdout_fd], [], [], 2.0)
                if ready:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    stripped = line.rstrip()
                    stdout_lines.append(stripped)

                    # [rubin]-Zeilen: Progress + Tail sofort aktualisieren
                    if "[rubin]" in stripped:
                        _parse_progress(line, stdout_lines, _guarded_set)
                        _guarded_set(stdout_tail="\n".join(stdout_lines[-100:]))

                    # Batch: Weitere wartende Zeilen sofort lesen
                    while True:
                        more, _, _ = select.select([stdout_fd], [], [], 0)
                        if not more:
                            break
                        line = proc.stdout.readline()
                        if not line:
                            break
                        stripped = line.rstrip()
                        stdout_lines.append(stripped)
                        if "[rubin]" in stripped:
                            _parse_progress(line, stdout_lines, _guarded_set)
                            _guarded_set(stdout_tail="\n".join(stdout_lines[-100:]))

                    if len(stdout_lines) > 100:
                        stdout_lines[:] = stdout_lines[-50:]
                else:
                    # Timeout: Prozess-Check + stderr_tail aktualisieren
                    ret = proc.poll()
                    if ret is not None:
                        log.info("Task %s: Prozess beendet (rc=%d), breche stdout-Read ab.", task_name, ret)
                        break
                    # Zwischen Steps: stderr_tail aktualisieren (zeigt Logging-Output)
                    if stderr_lines:
                        _guarded_set(stderr_tail="\n".join(stderr_lines[-500:]))

            # Hauptprozess sauber beenden lassen, dann Kindprozesse aufräumen.
            # NICHT sofort killpg — der Prozess könnte noch MLflow-Cleanup machen.
            try:
                proc.stdout.close()
            except Exception:
                pass
            stderr_thread.join(timeout=5)
            try:
                proc.stderr.close()
            except Exception:
                pass

            # Warte auf den Hauptprozess (max 30s für MLflow-Cleanup etc.)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log.warning("Task %s: Prozess reagiert nicht nach 30s, sende SIGTERM.", task_name)
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

            rc = proc.returncode
            if rc is None:
                rc = -9

            # Verwaiste Kindprozesse aufräumen (LightGBM Worker etc.)
            # NACH dem Hauptprozess, damit dessen Exit-Code nicht überschrieben wird.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass

            if rc == 0:
                from app.routes.results import _scan_result_files
                files = _scan_result_files()
                log.info("Task %s erfolgreich abgeschlossen (%d Ergebnis-Dateien).", task_name, len(files))
                _guarded_set(
                    status="done", message="Erfolgreich abgeschlossen.",
                    percent=100, result_files=files,
                    stderr_tail="\n".join(stderr_lines[-500:]),
                )
            else:
                log.error("Task %s fehlgeschlagen (Exit %d).", task_name, rc)
                _guarded_set(
                    status="error",
                    message=f"Fehlgeschlagen (Exit {rc})",
                    stderr_tail="\n".join(stderr_lines[-500:]),
                )
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                proc.kill()
            log.error("Task %s: Timeout nach %ds.", task_name, timeout)
            _guarded_set(status="error", message=f"Timeout nach {timeout}s.")
        except Exception as e:
            log.error("Task %s: Unerwarteter Fehler: %s", task_name, e)
            _guarded_set(status="error", message=str(e), stderr_tail=traceback.format_exc())

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


# Pre-compiled regex for progress parsing (avoid re.search per call)
_RE_STEP = re.compile(r"Step\s+(\d+)/(\d+):\s*(.*)")
_RE_PERCENT = re.compile(r"Progress:\s*(\d+)%")


def _parse_progress(line: str, all_lines: list[str], state_setter=None):
    """Parst Pipeline-Fortschritt aus [rubin]-stdout-Zeilen.

    Monotonie-Garantie: Der Prozentwert darf nie sinken. Verhindert Rücksprünge
    durch Step-Wechsel oder versehentliche Matches in Warn-/Info-Zeilen."""
    _update = state_setter or set_state
    current_pct = get_state().get("percent", 0)

    # [rubin] Step 3/8: Training & Cross-Predictions
    m = _RE_STEP.search(line)
    if m:
        idx, total, step_name = m.group(1), m.group(2), m.group(3).strip()
        new_pct = int(100 * int(idx) / int(total))
        _update(
            step=step_name, step_index=int(idx), total_steps=int(total),
            percent=max(new_pct, current_pct),
            message=f"Schritt {idx}/{total}: {step_name}",
        )
        return

    # [rubin] Progress: 45%
    m = _RE_PERCENT.search(line)
    if m:
        new_pct = int(m.group(1))
        _update(percent=max(new_pct, current_pct))


def _find_analysis_python() -> str:
    """Findet den Python-Interpreter für Analyse-Subprozesse.

    Bevorzugt das pixi-Default-Environment (hat alle Analyse-Dependencies),
    fällt auf sys.executable (app-env) zurück.
    """
    # 1. pixi default-Environment (hat alle Analyse-Dependencies)
    default_python = ROOT / ".pixi" / "envs" / "default" / "bin" / "python"
    if default_python.exists():
        log.info("Analyse-Python: %s (pixi default-env)", default_python)
        return str(default_python)

    # 2. pixi ohne benanntes Environment
    pixi_python = ROOT / ".pixi" / "env" / "bin" / "python"
    if pixi_python.exists():
        log.info("Analyse-Python: %s (pixi env)", pixi_python)
        return str(pixi_python)

    # 3. Fallback: gleiches Python wie der Server
    log.info("Analyse-Python: %s (sys.executable fallback)", sys.executable)
    return sys.executable


@bp.route("/api/run-analysis", methods=["POST"])
def run_analysis():
    if get_state()["status"] == "running":
        log.warning("Analyse abgelehnt: bereits ein Task aktiv.")
        return jsonify({"status": "error", "message": "Bereits eine Analyse aktiv."}), 409

    data = request.get_json(silent=True) or {}
    yaml_text = data.get("yaml", "")
    if not yaml_text or not yaml_text.strip():
        return jsonify({"status": "error", "message": "Keine Konfiguration gesendet."}), 400
    config_dir = WORK_DIR / ".rubin_cache"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Alte Ergebnisse löschen, damit kein veralteter Report angezeigt wird
    for old_file in ["analysis_report.html", "uplift_eval_summary.json"]:
        old_path = config_dir / old_file
        if old_path.exists():
            old_path.unlink()

    config_path = config_dir / "config_ui.yml"
    config_path.write_text(yaml_text, encoding="utf-8")
    log.info("Analyse-Konfiguration geschrieben: %s", config_path)

    python = _find_analysis_python()
    cmd = [python, str(ROOT / "run_analysis.py"), "--quiet", "--config", str(config_path)]
    _run_in_background("run_analysis", cmd)

    return jsonify({"status": "started", "message": "Analyse gestartet."})


@bp.route("/api/run-dataprep", methods=["POST"])
def run_dataprep():
    if get_state()["status"] == "running":
        log.warning("Datenvorbereitung abgelehnt: bereits ein Task aktiv.")
        return jsonify({"status": "error", "message": "Bereits ein Task aktiv."}), 409

    data = request.get_json(silent=True) or {}
    yaml_text = data.get("yaml", "")
    if not yaml_text or not yaml_text.strip():
        return jsonify({"status": "error", "message": "Keine Konfiguration gesendet."}), 400
    config_dir = WORK_DIR / ".rubin_cache"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config_dataprep_ui.yml"
    config_path.write_text(yaml_text, encoding="utf-8")
    log.info("DataPrep-Konfiguration geschrieben: %s", config_path)

    python = _find_analysis_python()
    cmd = [python, str(ROOT / "run_dataprep.py"), "--quiet", "--config", str(config_path)]
    _run_in_background("run_dataprep", cmd, timeout=1800)

    return jsonify({"status": "started", "message": "Datenvorbereitung gestartet."})


@bp.route("/api/dataprep-info")
def dataprep_info():
    """Gibt MLflow-Infos aus dem DataPrep-Outputverzeichnis zurück.

    Liest .mlflow_experiment und .mlflow_run_name aus dem angegebenen
    (oder Default-) Output-Pfad. Die UI nutzt das, um den Experiment-Namen
    automatisch in die Konfigurationsseite zu übernehmen.
    """
    out_path = request.args.get("output_path", "data/processed")
    p = ROOT / out_path
    result = {}
    exp_file = p / ".mlflow_experiment"
    run_file = p / ".mlflow_run_name"
    if exp_file.is_file():
        result["experiment_name"] = exp_file.read_text(encoding="utf-8").strip()
    if run_file.is_file():
        result["run_name"] = run_file.read_text(encoding="utf-8").strip()
    return jsonify(result)


# ══════════════════════════════════════════════════════
# PROGRESS (Polling)
# ══════════════════════════════════════════════════════

@bp.route("/api/progress")
def get_progress():
    state = get_state()
    # Erkennung von OOM-gekilten oder anderweitig verschwundenen Prozessen
    if state.get("status") == "running" and state.get("pid"):
        try:
            os.kill(state["pid"], 0)  # Prüft ob Prozess existiert (sendet kein Signal)
        except ProcessLookupError:
            log.error("Prozess %d ist verschwunden (vermutlich OOM-Kill). Status wird auf error gesetzt.", state["pid"])
            set_state(
                status="error",
                message="Prozess wurde unerwartet beendet (vermutlich Out-of-Memory). "
                        "Versuche weniger Daten (df_frac), weniger Modelle oder mehr RAM.",
                pid=None,
            )
            state = get_state()
    return jsonify(state)


@bp.route("/api/reset", methods=["POST"])
def reset_state():
    # Laufenden Prozess beenden
    pid = get_state().get("pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            log.info("Laufenden Prozess beendet (PID %d).", pid)
        except (OSError, ProcessLookupError):
            pass
    # Cache des letzten Laufs löschen (damit kein alter Report angezeigt wird)
    cache_dir = WORK_DIR / ".rubin_cache"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
        log.info("Cache gelöscht: %s", cache_dir)
    set_state(status="idle", task=None, message="", step="", step_index=0,
               total_steps=0, percent=0, stdout_tail="", stderr_tail="",
               result_files=[], pid=None)
    return jsonify({"status": "idle"})


