"""MLflow-Kompatibilität: File-Store-Opt-in für MLflow >= 3.14.

MLflow hat den Filesystem-Tracking-Backend (``./mlruns`` bzw. jede
``file://``-Tracking-URI) in den Wartungsmodus versetzt und verlangt ein
explizites Opt-in per ``MLFLOW_ALLOW_FILE_STORE=true`` — sonst wirft bereits
``mlflow.set_experiment`` eine MlflowException. Für rubin ist der lokale
File-Store ein gewollter Betriebsmodus (Single-User-Tracking, Artefakte
konsolidiert unter ``work_dir``, keine Server-Features nötig), daher wird
das Opt-in hier zentral gesetzt.

Wichtig: Das muss BEDINGUNGSLOS vor der ersten Store-Berührung geschehen —
auch (gerade!) wenn der Nutzer selbst eine ``MLFLOW_TRACKING_URI`` auf einen
File-Pfad gesetzt hat. ``setdefault`` respektiert dabei eine bewusst extern
gesetzte Abschaltung (``MLFLOW_ALLOW_FILE_STORE=false`` bleibt bestehen).
"""

from __future__ import annotations

import os


def allow_file_store() -> None:
    """Erlaubt den MLflow-File-Store (Opt-in für MLflow >= 3.14)."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
