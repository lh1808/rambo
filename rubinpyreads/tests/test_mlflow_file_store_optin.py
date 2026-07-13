"""MLflow >= 3.14 File-Store-Opt-in: muss BEDINGUNGSLOS greifen.

Regression: Das Opt-in hing am ``if not MLFLOW_TRACKING_URI``-Zweig — wer die
Tracking-URI selbst auf einen File-Pfad setzte, bekam beim ersten
``set_experiment`` die Wartungsmodus-Exception (Produktionsfehler DataPrep).
"""

from __future__ import annotations

import pytest


def test_allow_file_store_enables_explicit_file_uri(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from rubin.utils.mlflow_compat import allow_file_store

    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    allow_file_store()
    # Ohne das Opt-in würde bereits set_experiment die MlflowException
    # ("maintenance mode") werfen — mit Opt-in läuft der File-Store normal.
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    exp_id = mlflow.set_experiment("optin_regression").experiment_id
    assert exp_id is not None


def test_allow_file_store_respects_explicit_opt_out(monkeypatch):
    from rubin.utils.mlflow_compat import allow_file_store
    import os

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "false")
    allow_file_store()
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "false"
