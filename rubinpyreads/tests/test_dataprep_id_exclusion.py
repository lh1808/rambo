"""Integrationstest DataPrep: Die als deduplicate_id_column konfigurierte
Spalte darf nicht als Feature in X landen.

IDs tragen kein kausales Signal, korrelieren aber häufig mit Datei-/
Zeitreihenfolge (z.B. der TMES-Eval-Datei) und öffnen damit einen
Leakage-Kanal. Ausgeschlossen wird ausschließlich die explizit
konfigurierte Spalte — keine Namensheuristik.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from rubin.pipelines.data_prep_pipeline import DataPrepPipeline


def _write_config(tmp_path, raw_csv, out_dir, dedup_col):
    cfg = {
        "mlflow": {"experiment_name": "test_dataprep_id"},
        "constants": {"SEED": 1, "work_dir": str(tmp_path / "runs")},
        "data_files": {
            "x_file": str(out_dir / "X.parquet"),
            "t_file": str(out_dir / "T.parquet"),
            "y_file": str(out_dir / "Y.parquet"),
        },
        "data_prep": {
            "data_path": [str(raw_csv)],
            "output_path": str(out_dir),
            "target": "Y",
            "treatment": "T",
            "score_name": None,
            "deduplicate": True,
            "deduplicate_id_column": dedup_col,
            "log_to_mlflow": False,
        },
    }
    p = tmp_path / "cfg.yml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _make_raw(tmp_path, n=120):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "kunde_id": np.arange(n),
        "alter": rng.normal(40, 10, n).round(1),
        "T": rng.integers(0, 2, n),
        "Y": rng.integers(0, 2, n),
    })
    p = tmp_path / "raw.csv"
    df.to_csv(p, index=False)
    return p


def test_dedup_id_column_excluded_from_features(tmp_path):
    raw = _make_raw(tmp_path)
    out = tmp_path / "processed"
    cfg_path = _write_config(tmp_path, raw, out, dedup_col="kunde_id")

    DataPrepPipeline.from_config_path(str(cfg_path)).run()

    X = pd.read_parquet(out / "X.parquet")
    assert not any(c.upper() == "KUNDE_ID" for c in X.columns), (
        f"ID-Spalte als Feature in X gelandet: {list(X.columns)}"
    )
    assert any(c.upper() == "ALTER" for c in X.columns)


def test_without_dedup_column_all_columns_are_features(tmp_path):
    """Ohne konfigurierte ID-Spalte gibt es keinen Heuristik-Ausschluss —
    jede Nicht-Target/Treatment-Spalte ist Feature (explizit statt magisch)."""
    raw = _make_raw(tmp_path)
    out = tmp_path / "processed2"
    cfg_path = _write_config(tmp_path, raw, out, dedup_col=None)
    # deduplicate ohne Spalte: Vollzeilen-Dedup
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["data_prep"]["deduplicate"] = False
    cfg["data_prep"].pop("deduplicate_id_column")
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    DataPrepPipeline.from_config_path(str(cfg_path)).run()

    X = pd.read_parquet(out / "X.parquet")
    assert any(c.upper() == "KUNDE_ID" for c in X.columns)
