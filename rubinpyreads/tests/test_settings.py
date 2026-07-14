"""Tests für die Konfigurationsvalidierung."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rubin.settings import load_config, AnalysisConfig, SUPPORTED_MODEL_NAMES


def _write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")
    return path


@pytest.fixture
def minimal_raw() -> dict:
    """Minimale gültige Konfiguration."""
    return {
        "data_files": {
            "x_file": "X.parquet",
            "t_file": "T.parquet",
            "y_file": "Y.parquet",
        },
        "models": {
            "models_to_train": ["SLearner"],
        },
    }


class TestLoadConfig:
    def test_minimal_config(self, tmp_path, minimal_raw):
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert isinstance(cfg, AnalysisConfig)
        assert cfg.models.models_to_train == ["SLearner"]
        assert cfg.constants.random_seed == 42

    def test_seed_alias(self, tmp_path, minimal_raw):
        minimal_raw["constants"] = {"SEED": 123}
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.constants.random_seed == 123

    def test_unknown_model_rejected(self, tmp_path, minimal_raw):
        minimal_raw["models"]["models_to_train"] = ["UnknownModel"]
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception, match="unbekannte Modelle"):
            load_config(path)

    def test_extra_keys_rejected(self, tmp_path, minimal_raw):
        minimal_raw["unknown_key"] = "should_fail"
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception):
            load_config(path)

    def test_manual_champion_must_be_in_models(self, tmp_path, minimal_raw):
        minimal_raw["selection"] = {"manual_champion": "TLearner"}
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception, match="manual_champion"):
            load_config(path)

    def test_manual_champion_valid(self, tmp_path, minimal_raw):
        minimal_raw["selection"] = {"manual_champion": "SLearner"}
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.selection.manual_champion == "SLearner"

    def test_all_model_names_supported(self):
        expected = {"SLearner", "TLearner", "XLearner", "DRLearner",
                    "NonParamDML", "ParamDML", "CausalForestDML", "CausalForest"}
        assert SUPPORTED_MODEL_NAMES == expected

    def test_source_config_path_set(self, tmp_path, minimal_raw):
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.source_config_path == str(path)

    def test_validate_on_external(self, tmp_path, minimal_raw):
        minimal_raw["data_processing"] = {"validate_on": "external"}
        minimal_raw["data_files"]["eval_x_file"] = "eval_X.parquet"
        minimal_raw["data_files"]["eval_t_file"] = "eval_T.parquet"
        minimal_raw["data_files"]["eval_y_file"] = "eval_Y.parquet"
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.data_processing.validate_on == "external"

    def test_validate_on_holdout_rejected(self, tmp_path, minimal_raw):
        """Holdout wurde entfernt – muss als ungültiger Wert abgelehnt werden."""
        minimal_raw["data_processing"] = {"validate_on": "holdout"}
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception):
            load_config(path)

    def test_validate_on_test_rejected(self, tmp_path, minimal_raw):
        minimal_raw["data_processing"] = {"validate_on": "test"}
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception):
            load_config(path)

    # --- Feature-Selection Tests ---

    def test_feature_selection_defaults(self, tmp_path, minimal_raw):
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.feature_selection.enabled is False
        assert cfg.feature_selection.methods == ["catboost_importance"]
        assert cfg.feature_selection.max_features == 77
        assert cfg.feature_selection.correlation_threshold == 0.9

    def test_feature_selection_multiple_methods(self, tmp_path, minimal_raw):
        minimal_raw["feature_selection"] = {
            "enabled": True,
            "methods": ["lgbm_importance", "causal_forest"],
            "max_features": 50,
        }
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.feature_selection.methods == ["lgbm_importance", "causal_forest"]
        assert cfg.feature_selection.max_features == 50

    def test_feature_selection_old_method_field_rejected(self, tmp_path, minimal_raw):
        """Altes Singular-Feld 'method' wird durch extra=forbid abgewiesen."""
        minimal_raw["feature_selection"] = {
            "method": "lgbm_importance",
        }
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception):
            load_config(path)

    def test_feature_selection_old_importance_threshold_rejected(self, tmp_path, minimal_raw):
        """Altes Feld 'importance_threshold' wird durch extra=forbid abgewiesen."""
        minimal_raw["feature_selection"] = {
            "importance_threshold": 2.0,
        }
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception):
            load_config(path)

    # --- Surrogate-Tree Tests ---

    def test_surrogate_tree_default_disabled(self, tmp_path, minimal_raw):
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.surrogate_tree.enabled is False
        assert cfg.surrogate_tree.min_samples_leaf == 50
        assert cfg.surrogate_tree.num_leaves == 31
        assert cfg.surrogate_tree.max_depth is None

    def test_surrogate_tree_enabled(self, tmp_path, minimal_raw):
        minimal_raw["surrogate_tree"] = {
            "enabled": True,
            "min_samples_leaf": 100,
            "num_leaves": 15,
            "max_depth": 5,
        }
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.surrogate_tree.enabled is True
        assert cfg.surrogate_tree.min_samples_leaf == 100
        assert cfg.surrogate_tree.num_leaves == 15
        assert cfg.surrogate_tree.max_depth == 5

    # --- DataPrep Deduplicate Tests ---

    def test_data_prep_deduplicate_default(self, tmp_path, minimal_raw):
        minimal_raw["data_prep"] = {
            "data_path": ["data.csv"],
            "feature_path": "features.xlsx",
            "output_path": "out",
        }
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.data_prep.deduplicate is False
        assert cfg.data_prep.deduplicate_id_column is None

    def test_data_prep_deduplicate_enabled(self, tmp_path, minimal_raw):
        minimal_raw["data_prep"] = {
            "data_path": ["data.csv"],
            "feature_path": "features.xlsx",
            "output_path": "out",
            "deduplicate": True,
            "deduplicate_id_column": "PARTNER_ID",
        }
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.data_prep.deduplicate is True
        assert cfg.data_prep.deduplicate_id_column == "PARTNER_ID"

    # --- Multi-Treatment Tests ---

    def test_treatment_config_default_binary(self, tmp_path, minimal_raw):
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.treatment.type == "binary"
        assert cfg.treatment.reference_group == 0

    def test_treatment_config_multi(self, tmp_path, minimal_raw):
        minimal_raw["treatment"] = {"type": "multi", "reference_group": 0}
        minimal_raw["selection"] = {"metric": "policy_value"}
        # MT-kompatible Modelle verwenden (NonParamDML ist seit dem
        # econml-Restriktions-Fix BT-only und hier NICHT mehr zulässig)
        minimal_raw["models"] = {"models_to_train": ["ParamDML"]}
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.treatment.type == "multi"

    def test_mt_blocks_bt_only_models(self, tmp_path, minimal_raw):
        minimal_raw["treatment"] = {"type": "multi"}
        minimal_raw["models"] = {"models_to_train": ["SLearner"]}
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception, match="nicht kompatibel"):
            load_config(path)

    def test_mt_blocks_nonparamdml(self, tmp_path, minimal_raw):
        """econml beschränkt NonParamDMLs Final-Model auf Binary Treatment —
        ohne Config-Gate würde ein MT-Lauf erst mitten im Training crashen."""
        minimal_raw["treatment"] = {"type": "multi"}
        minimal_raw["selection"] = {"metric": "policy_value"}
        minimal_raw["models"] = {"models_to_train": ["NonParamDML", "ParamDML"]}
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception, match="NonParamDML"):
            load_config(path)

    def test_mt_blocks_xlearner(self, tmp_path, minimal_raw):
        minimal_raw["treatment"] = {"type": "multi"}
        minimal_raw["models"] = {"models_to_train": ["XLearner"]}
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception, match="nicht kompatibel"):
            load_config(path)

    def test_mt_blocks_tlearner(self, tmp_path, minimal_raw):
        minimal_raw["treatment"] = {"type": "multi"}
        minimal_raw["models"] = {"models_to_train": ["TLearner"]}
        path = _write_config(tmp_path, minimal_raw)
        with pytest.raises(Exception, match="nicht kompatibel"):
            load_config(path)

    def test_mt_allows_dml(self, tmp_path, minimal_raw):
        minimal_raw["treatment"] = {"type": "multi"}
        minimal_raw["selection"] = {"metric": "policy_value"}
        # NonParamDML ist BT-only (econml-Restriktion) — MT-fähig sind
        # ParamDML, DRLearner und CausalForestDML.
        minimal_raw["models"] = {"models_to_train": ["ParamDML", "DRLearner", "CausalForestDML"]}
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.treatment.type == "multi"
        assert len(cfg.models.models_to_train) == 3

    # --- Multi-Target Tests ---

    def test_data_prep_target_single_string(self, tmp_path, minimal_raw):
        minimal_raw["data_prep"] = {
            "data_path": ["data.csv"],
            "output_path": "out",
            "target": "OUTCOME",
        }
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.data_prep.target == "OUTCOME"

    def test_data_prep_target_list(self, tmp_path, minimal_raw):
        minimal_raw["data_prep"] = {
            "data_path": ["data.csv"],
            "output_path": "out",
            "target": ["COL_A", "COL_B"],
        }
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.data_prep.target == ["COL_A", "COL_B"]

    # --- SHAP / Explainability Tests ---

    def test_shap_values_defaults(self, tmp_path, minimal_raw):
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.shap_values.calculate_shap_values is False
        assert cfg.shap_values.n_shap_values == 10_000
        assert cfg.shap_values.top_n_features == 20
        assert cfg.shap_values.num_bins == 10

    def test_shap_values_enabled(self, tmp_path, minimal_raw):
        minimal_raw["shap_values"] = {
            "calculate_shap_values": True,
            "n_shap_values": 5000,
            "top_n_features": 15,
        }
        path = _write_config(tmp_path, minimal_raw)
        cfg = load_config(path)
        assert cfg.shap_values.calculate_shap_values is True
        assert cfg.shap_values.n_shap_values == 5000
        assert cfg.shap_values.top_n_features == 15


class TestMultiTreatmentScorerResolution:
    """Regressionstests: FMT-/CFT-Scorer bei Multi-Treatment.

    Der Qini-Scorer ist binär-only (uplift_curve verlangt t in {0,1} und 1-d
    CATE). resolve_scorer_auto muss 'auto' bei treatment.type='multi' daher
    IMMER zu 'rscore' auflösen (auch bei RCT); explizit gesetztes 'qini' wird
    vom MT-Validator abgelehnt — sonst scheitern alle FMT-/CFT-Trials erst
    zur Laufzeit an '_check_binary'."""

    def _base_cfg(self, **overrides):
        cfg = {
            "mlflow": {"experiment_name": "t"},
            "constants": {"SEED": 1, "work_dir": "/tmp/t"},
            "data_files": {"x_file": "x.parquet", "t_file": "t.parquet", "y_file": "y.parquet"},
            "study_type": "rct",
            "treatment": {"type": "multi", "reference_group": 0},
            "models": {"models_to_train": ["ParamDML", "DRLearner"], "ensemble": False},
            "selection": {"metric": "policy_value", "higher_is_better": True},
            "final_model_tuning": {"enabled": True, "n_trials": 2, "scorer": "auto"},
        }
        for k, v in overrides.items():
            cfg[k] = v
        return cfg

    def test_auto_resolves_to_rscore_under_multi_even_at_rct(self, tmp_path):
        import yaml
        from rubin.settings import load_config
        p = tmp_path / "c.yml"
        p.write_text(yaml.safe_dump(self._base_cfg()), encoding="utf-8")
        cfg = load_config(str(p))
        assert cfg.final_model_tuning.scorer == "rscore"
        assert cfg.causal_forest.scorer == "rscore"

    def test_auto_still_resolves_to_qini_for_binary_rct(self, tmp_path):
        import yaml
        from rubin.settings import load_config
        c = self._base_cfg(
            treatment={"type": "binary", "reference_group": 0},
            selection={"metric": "qini", "higher_is_better": True},
        )
        p = tmp_path / "c.yml"
        p.write_text(yaml.safe_dump(c), encoding="utf-8")
        cfg = load_config(str(p))
        assert cfg.final_model_tuning.scorer == "qini"

    def test_explicit_qini_fmt_scorer_rejected_under_multi(self, tmp_path):
        import yaml
        import pytest
        from rubin.settings import load_config
        c = self._base_cfg()
        c["final_model_tuning"]["scorer"] = "qini"
        p = tmp_path / "c.yml"
        p.write_text(yaml.safe_dump(c), encoding="utf-8")
        with pytest.raises(Exception, match="binär-only"):
            load_config(str(p))

    def test_explicit_qini_cft_scorer_rejected_under_multi(self, tmp_path):
        import yaml
        import pytest
        from rubin.settings import load_config
        c = self._base_cfg()
        c["causal_forest"] = {"tune_enabled": True, "n_trials": 2, "scorer": "qini"}
        p = tmp_path / "c.yml"
        p.write_text(yaml.safe_dump(c), encoding="utf-8")
        with pytest.raises(Exception, match="binär-only"):
            load_config(str(p))

    def test_treatment_only_rejected_under_multi(self, tmp_path):
        """treatment_only filtert pro Datei hart auf T==1 — bei K>2 Armen gingen
        Zeilen mit T>=2 still verloren. Kombination wird abgelehnt."""
        import yaml
        import pytest
        from rubin.settings import load_config
        c = self._base_cfg()
        c["data_prep"] = {
            "data_path": ["a.csv", "b.csv"], "output_path": "/tmp/out",
            "target": "Y", "treatment": "T",
            "multiple_files_option": "treatment_only",
        }
        p = tmp_path / "c.yml"
        p.write_text(yaml.safe_dump(c), encoding="utf-8")
        with pytest.raises(Exception, match="still verloren"):
            load_config(str(p))

    def test_balance_treatments_rejected_under_multi(self, tmp_path):
        """balance_treatments ist binär definiert (T==1 vs. T==0, Effektiv-N p·(1−p))
        — bei K>2 Armen würde die Arm-Struktur verzerrt. Kombination abgelehnt."""
        import yaml
        import pytest
        from rubin.settings import load_config
        c = self._base_cfg()
        c["data_prep"] = {
            "data_path": ["a.csv", "b.csv"], "output_path": "/tmp/out",
            "target": "Y", "treatment": "T",
            "balance_treatments": True,
        }
        p = tmp_path / "c.yml"
        p.write_text(yaml.safe_dump(c), encoding="utf-8")
        with pytest.raises(Exception, match="Balancing ist nur für Binary Treatment"):
            load_config(str(p))
