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


def set_experiment_robust(mlflow_module, experiment_name: str, logger=None):
    """``mlflow.set_experiment`` mit Selbstheilung für GELÖSCHTE Experimente.

    MLflow verweigert ``set_experiment`` auf ein Experiment im Papierkorb
    ("Cannot set a deleted experiment ... as the active experiment") — ein in
    der MLflow-UI gelöschtes Experiment blockiert damit dauerhaft jeden
    rubin-Lauf gleichen Namens (realer Vorfall: analyze-quick auf Domino).
    Dieser Helper stellt das Experiment in dem Fall automatisch wieder her
    (restore — die alten Runs bleiben erhalten) und versucht es erneut;
    jede andere Exception wird unverändert durchgereicht.
    """
    try:
        return mlflow_module.set_experiment(experiment_name)
    except Exception as e:
        if "deleted experiment" not in str(e).lower():
            raise
        try:
            from mlflow.entities import ViewType
            client = mlflow_module.tracking.MlflowClient()
            deleted = [
                exp for exp in client.search_experiments(view_type=ViewType.DELETED_ONLY)
                if exp.name == experiment_name
            ]
            if not deleted:
                raise
            client.restore_experiment(deleted[0].experiment_id)
            if logger is not None:
                logger.warning(
                    "MLflow-Experiment '%s' lag im Papierkorb (in der UI gelöscht) und hätte "
                    "jeden Lauf blockiert — automatisch wiederhergestellt (ID %s, alte Runs "
                    "bleiben erhalten). Endgültig entfernen: in der MLflow-UI 'Permanently delete'.",
                    experiment_name, deleted[0].experiment_id,
                )
            return mlflow_module.set_experiment(experiment_name)
        except Exception:
            raise e from None
