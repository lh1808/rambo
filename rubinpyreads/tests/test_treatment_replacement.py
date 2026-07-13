"""Regressionstests für das typ-robuste Treatment-/Target-Replacement und die
Fail-fast-Validierungen rund um Treatment-Arme.

Hintergrund (Bug): Die UI/YAML-Config liefert Replacement-Keys immer als
Strings (settings.py: Dict[str, Any]); numerische Spalten aus SAS/CSV
(float64/int64) wurden von Series.replace() daher NICHT gematcht — das
Mapping "2" -> 1 war ein stiller No-op, T blieb [0, 1, 2] und die Analyse
scheiterte erst tief im Tuning (econml "binary treatment only", reihenweise
gescheiterte Optuna-Trials).
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
import yaml

from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
from rubin.pipelines.analysis_pipeline import AnalysisPipeline


# ── _expand_replacement_mapping ──────────────────────────────────────────


def test_expand_mapping_adds_numeric_key_variants():
    m = DataPrepPipeline._expand_replacement_mapping({"0": 0, "1": 1, "2": 1})
    assert m["2"] == 1 and m[2] == 1 and m[2.0] == 1


def test_expand_mapping_keeps_non_numeric_keys_untouched():
    assert DataPrepPipeline._expand_replacement_mapping({"J": 1, "N": 0}) == {"J": 1, "N": 0}


def test_expand_mapping_empty_and_none():
    assert DataPrepPipeline._expand_replacement_mapping(None) is None
    assert DataPrepPipeline._expand_replacement_mapping({}) is None


@pytest.mark.parametrize("dtype", ["float64", "int64"])
def test_string_keys_now_match_numeric_columns(dtype):
    """Kernszenario des Bugs: String-Keys aus YAML auf numerischer Spalte."""
    s = pd.Series([0, 1, 2, 2]).astype(dtype)
    out = s.replace(DataPrepPipeline._expand_replacement_mapping({"0": 0, "1": 1, "2": 1}))
    assert sorted(pd.unique(out).tolist()) == [0, 1]


# ── _verify_replacement_applied (Fail-fast in DataPrep) ──────────────────


def _dp_stub():
    inst = DataPrepPipeline.__new__(DataPrepPipeline)
    return inst


def test_verify_replacement_accepts_identity_mappings():
    # "0" -> 0 und "1" -> 1 lassen die Rohwerte korrekt stehen — kein Fehler.
    _dp_stub()._verify_replacement_applied(
        pd.Series([0.0, 1.0]), {"0": 0, "1": 1, "2": 1}, "T", "x.sas7bdat")


def test_verify_replacement_raises_on_silent_noop():
    with pytest.raises(ValueError, match="Typ-Mismatch"):
        _dp_stub()._verify_replacement_applied(
            pd.Series([0.0, 1.0, 2.0]), {"0": 0, "1": 1, "2": 1}, "T", "x.sas7bdat")


def test_verify_replacement_accepts_swap_mappings():
    """Swap-Mapping ("0"→1, "1"→0): Die verbleibenden Werte sind legitime
    ZIELWERTE des jeweils anderen Eintrags — kein False-Positive-Abbruch.
    Nicht als Zielwert vorkommende Keys (wie "2"→1) bleiben streng geprüft."""
    swapped = pd.Series([0.0, 1.0]).replace(
        DataPrepPipeline._expand_replacement_mapping({"0": 1, "1": 0}))
    assert sorted(swapped.tolist()) == [0, 1]
    _dp_stub()._verify_replacement_applied(swapped, {"0": 1, "1": 0}, "T", "x.csv")


# ── DataPrep End-to-End (chunked CSV) ────────────────────────────────────


def test_dataprep_applies_string_key_mapping_on_numeric_column(tmp_path):
    raw = tmp_path / "raw.csv"
    rng = np.random.default_rng(3)
    pd.DataFrame({
        "t": rng.choice([0, 1, 2], size=240),
        "y": rng.integers(0, 2, size=240),
        "f1": rng.normal(size=240),
        "f2": rng.normal(size=240),
    }).to_csv(raw, index=False)
    out_dir = tmp_path / "out"
    cfg = {
        "mlflow": {"experiment_name": "test_treat_repl"},
        "constants": {"SEED": 1, "work_dir": str(tmp_path / "runs")},
        "data_files": {
            "x_file": str(out_dir / "X.parquet"),
            "t_file": str(out_dir / "T.parquet"),
            "y_file": str(out_dir / "Y.parquet"),
        },
        "data_prep": {
            "data_path": [str(raw)],
            "output_path": str(out_dir),
            "target": "Y",
            "treatment": "T",
            "score_name": None,
            "chunksize": 100,  # Chunk-Pfad mit abdecken
            "treatment_replacement": {"0": 0, "1": 1, "2": 1},
            "log_to_mlflow": False,
        },
    }
    p = tmp_path / "cfg.yml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = DataPrepPipeline.from_config_path(str(p)).run()
    assert sorted(np.unique(result.T).tolist()) == [0, 1]


# ── _validate_treatment_arms (Fail-fast in der Analyse) ──────────────────


def _analysis_stub(t_type):
    inst = AnalysisPipeline.__new__(AnalysisPipeline)
    inst.cfg = types.SimpleNamespace(treatment=types.SimpleNamespace(type=t_type))
    return inst


def test_validate_arms_binary_accepts_zero_one():
    _analysis_stub("binary")._validate_treatment_arms(np.array([0, 1, 1, 0]))
    _analysis_stub("binary")._validate_treatment_arms(np.array([0.0, 1.0]))


@pytest.mark.parametrize("bad", [[0, 1, 2], [1, 2], [0], [0, 2]])
def test_validate_arms_binary_rejects_non_binary(bad):
    with pytest.raises(ValueError, match="treatment_replacement"):
        _analysis_stub("binary")._validate_treatment_arms(np.array(bad))


def test_validate_arms_multi_requires_reference_group():
    _analysis_stub("multi")._validate_treatment_arms(np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="Referenzgruppe 0"):
        _analysis_stub("multi")._validate_treatment_arms(np.array([1, 2]))


# ── _collect_base_learner_info (Bundle-metadata.json) ────────────────────


def _cfg_bl(t):
    return types.SimpleNamespace(base_learner=types.SimpleNamespace(type=t))


def test_collect_learner_info_plain_type():
    info = AnalysisPipeline._collect_base_learner_info(
        _cfg_bl("catboost"), {"NonParamDML": {"model_y": {"_learner_type": "lgbm"}}})
    assert info == {"type": "catboost"}


def test_collect_learner_info_both_records_winner_per_role():
    tuned = {
        "NonParamDML": {
            "model_y": {"_learner_type": "lgbm", "n_estimators": 300},
            "model_final": {"_learner_type": "catboost", "iterations": 250},
            "default": {"_learner_type": "lgbm"},   # interner Key → gefiltert
            "forest": {"n_estimators": 400},          # ohne _learner_type → ausgelassen
        },
    }
    info = AnalysisPipeline._collect_base_learner_info(_cfg_bl("both"), tuned)
    assert info["type"] == "both"
    assert info["fallback_without_tuned_params"] == "catboost"
    assert info["chosen_per_role"]["NonParamDML"] == {
        "model_y": "lgbm", "model_final": "catboost"}


def test_collect_learner_info_both_robust_against_missing_tuning():
    assert AnalysisPipeline._collect_base_learner_info(_cfg_bl("both"), None)["chosen_per_role"] == {}
