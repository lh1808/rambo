"""Integrationstests für den ``/api/reset``-Endpunkt.

Kernverhalten: Beim Zurücksetzen muss die GANZE Prozessgruppe des laufenden
Analyse-Subprozesses beendet werden – nicht nur der Eltern-Prozess. Der Lauf
wird mit ``start_new_session=True`` gestartet, sodass die CPU-Last in
Kindprozessen (LightGBM ``n_jobs=-1``, joblib/loky-Worker) steckt. Ein reines
``os.kill(pid)`` würde diese als Waisen weiterlaufen lassen.

Die Tests spawnen eine echte Prozessgruppe (Eltern + Kind) und prüfen nach dem
POST auf ``/api/reset``, dass BEIDE Prozesse tot sind.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

import pytest

# Die App legt bei Import ihr WORK_DIR an und schreibt die Progress-Datei —
# ohne Umleitung landete das bei jedem Suite-Lauf im Repo-Baum
# (runs/.rubin_progress.json). Muss VOR dem ersten app-Import gesetzt sein.
os.environ.setdefault("RUBIN_WORK_DIR", tempfile.mkdtemp(prefix="rubin_test_work_"))

# killpg/start_new_session sind POSIX-spezifisch.
pytestmark = pytest.mark.skipif(
    not hasattr(os, "killpg"), reason="killpg nur unter POSIX verfügbar"
)


def _pid_alive(pid: int) -> bool:
    """True, solange der Prozess existiert (Signal 0 sendet nichts)."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _wait_dead(pids, timeout: float = 5.0) -> bool:
    """Wartet bis alle PIDs verschwunden sind (oder Timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(_pid_alive(p) for p in pids):
            return True
        time.sleep(0.05)
    return not any(_pid_alive(p) for p in pids)


def _spawn_group():
    """Startet eine eigene Prozessgruppe: Eltern spawnt ein langlebiges Kind.

    Gibt ``(proc, child_pid)`` zurück. ``proc.pid`` ist Session-/Gruppenleiter
    (start_new_session=True); das Kind erbt dessen Prozessgruppe.
    """
    code = (
        "import subprocess, sys, time;"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        "print(c.pid, flush=True);"
        "time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child_pid = int(proc.stdout.readline().strip())
    return proc, child_pid


@pytest.fixture()
def client():
    from app.server import create_app

    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


def test_reset_kills_entire_process_group(client):
    """POST /api/reset beendet Eltern UND Kind der Prozessgruppe."""
    from app.state import set_state

    proc, child_pid = _spawn_group()
    parent_pid = proc.pid
    try:
        # Beide laufen anfangs.
        assert _pid_alive(parent_pid)
        assert _pid_alive(child_pid)

        set_state(status="running", pid=parent_pid)
        resp = client.post("/api/reset")
        assert resp.status_code == 200
        assert resp.get_json().get("status") == "idle"

        # Eltern-Prozess: proc.wait() reapt den (sonst als Zombie verbleibenden)
        # Prozess und bestätigt damit zugleich seinen Tod. os.kill(pid, 0) taugt
        # hier NICHT, da ein nicht-gereapter Zombie fälschlich als "lebt" gilt.
        try:
            proc.wait(timeout=5)
            parent_dead = True
        except subprocess.TimeoutExpired:
            parent_dead = False
        assert parent_dead, "Eltern-Prozess wurde nicht beendet."

        # Entscheidend: das KIND (= CPU-Worker) muss ebenfalls sterben.
        # Mit einem einfachen os.kill(pid) würde nur der Eltern-Prozess
        # ein Signal bekommen und das Kind als Waise weiterlaufen. Es wird nach
        # Reparenting an init/PID 1 dort gereapt → verschwindet aus der Tabelle.
        assert _wait_dead([child_pid]), (
            f"Kindprozess (Worker) lief weiter: child_alive={_pid_alive(child_pid)}"
        )
    finally:
        # Aufräumen, falls der Test fehlschlägt (keine geleakten Prozesse).
        for p in (child_pid, parent_pid):
            try:
                os.kill(p, 9)
            except (ProcessLookupError, OSError):
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def test_reset_without_running_process_is_idempotent(client):
    """Ohne laufenden Prozess setzt /api/reset nur den State auf idle."""
    from app.state import set_state, get_state

    set_state(status="idle", pid=None)
    resp = client.post("/api/reset")
    assert resp.status_code == 200
    assert resp.get_json().get("status") == "idle"
    assert get_state().get("pid") is None
