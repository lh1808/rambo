"""Health-Check, System-Info und Prozess-Restart."""
from __future__ import annotations

import os
import signal

from flask import Blueprint, jsonify

from app.state import (
    __version__, log, WORK_DIR, MAX_UPLOAD_MB,
    get_state, set_state,
)

bp = Blueprint("health", __name__)


@bp.route("/api/work-dir")
def get_work_dir():
    """Gibt das aktuelle Arbeitsverzeichnis zurück (für UI-Anzeige)."""
    return jsonify({
        "work_dir": str(WORK_DIR),
        "source": "RUBIN_WORK_DIR" if os.environ.get("RUBIN_WORK_DIR") else "default (runs/)",
        "upload_dir": str(WORK_DIR / "uploads"),
        "cache_dir": str(WORK_DIR / ".rubin_cache"),
        "mlruns_dir": str(WORK_DIR / "mlruns"),
        "max_upload_mb": MAX_UPLOAD_MB,
    })


@bp.route("/api/health")
def health():
    """Health-Check mit System-Informationen (RAM, Prozess-Status)."""
    is_domino = bool(os.environ.get("DOMINO_PROJECT_NAME"))

    # RAM-Auslastung — Container-aware (cgroups v1/v2 + /proc/meminfo Fallback)
    ram = {}
    try:
        # 1. Host-RAM aus /proc/meminfo
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])  # kB
        host_total_kb = info.get("MemTotal", 0)
        host_avail_kb = info.get("MemAvailable", info.get("MemFree", 0))

        # 2. Container-Limit aus cgroups (überschreibt host_total wenn kleiner)
        container_limit_kb = host_total_kb
        for cg_path in [
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroups v1
            "/sys/fs/cgroup/memory.max",                     # cgroups v2
        ]:
            try:
                with open(cg_path) as cg:
                    val = cg.read().strip()
                    if val != "max" and val.isdigit():
                        limit_kb = int(val) // 1024
                        # Nur verwenden wenn kleiner als Host-RAM (= echtes Container-Limit)
                        if 0 < limit_kb < host_total_kb:
                            container_limit_kb = limit_kb
                        break
            except (FileNotFoundError, PermissionError, ValueError):
                continue

        # 3. Container-Usage aus cgroups (genauer als /proc/meminfo in Containern)
        #
        # WICHTIG: memory.usage_in_bytes (v1) und memory.current (v2) enthalten
        # den Linux Page-Cache. Da Linux freien RAM aggressiv als Disk-Cache nutzt,
        # steht dieser Wert fast immer nahe am Limit — unabhängig vom tatsächlichen
        # Verbrauch der Prozesse. Um den echten Working-Set-Verbrauch zu erhalten,
        # muss der inaktive Cache (reclaimable) abgezogen werden.
        # Quelle: memory.stat → total_inactive_file (v1) / inactive_file (v2).
        container_used_kb = host_total_kb - host_avail_kb  # Fallback (MemAvailable ist cache-aware)
        for cg_usage_path, cg_stat_path, cache_key in [
            ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
             "/sys/fs/cgroup/memory/memory.stat",
             "total_inactive_file"),  # cgroups v1
            ("/sys/fs/cgroup/memory.current",
             "/sys/fs/cgroup/memory.stat",
             "inactive_file"),        # cgroups v2
        ]:
            try:
                with open(cg_usage_path) as cg:
                    val = cg.read().strip()
                    if not val.isdigit():
                        continue
                    raw_usage_kb = int(val) // 1024

                    # Cache aus memory.stat lesen und abziehen
                    cache_kb = 0
                    try:
                        with open(cg_stat_path) as sf:
                            for stat_line in sf:
                                parts = stat_line.split()
                                if len(parts) == 2 and parts[0] == cache_key:
                                    cache_kb = int(parts[1]) // 1024
                                    break
                    except (FileNotFoundError, PermissionError, ValueError):
                        pass

                    container_used_kb = max(0, raw_usage_kb - cache_kb)
                    break
            except (FileNotFoundError, PermissionError, ValueError):
                continue

        total = container_limit_kb
        used = min(container_used_kb, total)
        avail = max(0, total - used)
        ram = {
            "total_mb": round(total / 1024),
            "used_mb": round(used / 1024),
            "available_mb": round(avail / 1024),
            "percent": round(used / total * 100, 1) if total > 0 else 0,
        }
    except Exception:
        pass

    # Prozess-Status
    state = get_state()
    proc_status = state.get("status", "idle")
    proc_pid = state.get("pid")

    # Wenn running: prüfe ob Prozess noch lebt
    if proc_status == "running" and proc_pid:
        try:
            os.kill(proc_pid, 0)
        except (OSError, ProcessLookupError):
            proc_status = "crashed"

    # CPU-Auslastung — load average als Prozentsatz der verfügbaren Kerne
    cpu = {}
    try:
        load1, load5, load15 = os.getloadavg()
        # Verfügbare CPUs (Container-aware)
        n_cpus = os.cpu_count() or 1
        try:
            n_cpus = min(n_cpus, len(os.sched_getaffinity(0)))
        except (AttributeError, OSError):
            pass
        for cg_path in ["/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"]:
            try:
                with open(cg_path) as cg:
                    val = cg.read().strip()
                    if cg_path.endswith("cpu.max"):
                        parts = val.split()
                        if parts[0] != "max":
                            n_cpus = min(n_cpus, max(1, int(int(parts[0]) / int(parts[1]))))
                    elif val != "-1":
                        period = 100000
                        try:
                            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as pf:
                                period = int(pf.read().strip())
                        except Exception:
                            pass
                        n_cpus = min(n_cpus, max(1, int(int(val) / period)))
                    break
            except (FileNotFoundError, PermissionError, ValueError):
                continue
        cpu = {
            "percent": round(min(load1 / max(n_cpus, 1) * 100, 100), 1),
            "cores": n_cpus,
        }
    except (OSError, AttributeError):
        pass

    return jsonify({
        "status": "ok",
        "version": __version__,
        "environment": "domino" if is_domino else "standalone",
        "ram": ram,
        "cpu": cpu,
        "process": {
            "status": proc_status,
            "pid": proc_pid,
            "task": state.get("task"),
            "step": state.get("step", ""),
            "percent": state.get("percent", 0),
        },
    })



@bp.route("/api/restart-process", methods=["POST"])
def restart_process():
    """Beendet einen laufenden/abgestürzten Prozess und setzt den State zurück."""
    state = get_state()
    pid = state.get("pid")
    killed = False
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            killed = True
            log.info("Prozess %d (Gruppe) beendet via restart.", pid)
        except (OSError, ProcessLookupError):
            pass
        if not killed:
            try:
                os.kill(pid, signal.SIGTERM)
                killed = True
                log.info("Prozess %d beendet via restart.", pid)
            except (OSError, ProcessLookupError):
                pass
    set_state(status="idle", task=None, message="", step="", step_index=0,
              total_steps=0, percent=0, stdout_tail="", stderr_tail="",
              result_files=[], pid=None)
    return jsonify({"status": "idle", "killed": killed})
