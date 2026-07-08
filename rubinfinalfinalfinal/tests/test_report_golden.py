"""Golden-Master-Regressionstest für den HTML-Analyse-Report.

Zweck: Absicherung von Refactorings des Report-Generators (`generate_html_report`),
die den Output **byte-identisch** lassen sollen. Eine voll befüllte Fixture löst
alle Report-Sektionen aus; das gerenderte HTML wird (nach Maskierung des einzigen
nicht-deterministischen Elements — des Timestamps) gegen einen eingefrorenen
Golden-Snapshot verglichen.

Golden neu erzeugen (nach BEABSICHTIGTEN Report-Änderungen):
    python tests/test_report_golden.py --update
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

from rubin.reporting.html_report import ReportCollector, generate_html_report

GOLDEN_PATH = Path(__file__).parent / "data" / "report_golden.html"

# Feste Fake-base64-Strings → deterministischer Output ohne echte Plots/matplotlib.
_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

# Nur der Timestamp ist nicht-deterministisch (datetime.now() im Template).
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}")


def _normalize(html: str) -> str:
    return _TS_RE.sub("<TIMESTAMP>", html)


def _fixture_collector() -> ReportCollector:
    """Voll befüllter Collector, der ALLE Report-Sektionen + beide Penalty-Karten auslöst.

    Werte sind reine Test-Fixtures (keine Empfehlungen) — beide Overfit-Penaltys
    sind >0 gesetzt, um die jeweiligen Anzeige-Zweige abzudecken.
    """
    c = ReportCollector()
    c.run_name = "golden_run_001"

    c.config_summary = {
        "experiment_name": "golden_master_experiment",
        "study_type": "rct",
        "seed": 42,
        "tuning_seed": 18,
        "parallel_level": 2,
        "base_learner": "both",
        "validate_on": "cv",
        "cv_splits": 5,
        "dml_crossfit_folds": 5,
        "tuning_cv_splits": 5,
        "eval_mask_file": None,
        "treatment_type": "binary",
        "models": ["NonParamDML", "DRLearner", "CausalForestDML", "CausalForest"],
        "tuning_enabled": True,
        "tuning_trials": 50,
        "tuning_metric": "log_loss (Klassifikation) / neg_mse (Regression)",
        "tuning_skill_metric": "Skill Score (Klassifikation) / R² (Regression)",
        "tuning_single_fold": False,
        "tuning_overfit_penalty": 0.2,        # Fixture: >0, um BLT-Penalty-Karte abzudecken
        "tuning_overfit_tolerance": 0.15,
        "tuning_overfit_max_penalized_gap": 1.0,
        "final_tuning_enabled": True,
        "final_tuning_trials": 50,
        "final_tuning_models": ["NonParamDML", "DRLearner"],
        "final_tuning_single_fold": False,
        "final_tuning_cv_splits": 5,
        "final_tuning_overfit_penalty": 0.3,  # Fixture: >0, um FMT-Penalty-Karte abzudecken
        "final_tuning_overfit_tolerance": 0.05,
        "final_tuning_overfit_max_penalized_gap": 1.0,
        "final_tuning_scorer": "qini",
        "feature_selection_enabled": True,
        "feature_selection_methods": ["correlation", "importance"],
        "feature_selection_max_features": 50,
        "feature_selection_corr_threshold": 0.8,
        "surrogate_enabled": True,
        "selection_metric": "qini",
        "higher_is_better": True,
        "bundle_enabled": True,
        "causal_forest_tune": True,
        "causal_forest_scorer": "qini",
        "reduce_memory": True,
        "fill_na_method": None,
        "mc_iters": 3,
    }

    c.data_stats = {
        "n_rows": 12000, "n_features": 25, "n_numeric": 20, "n_categorical": 5,
        "n_treatment_groups": 2,
        "treatment_distribution": {"T=0": 6000, "T=1": 6000},
        "outcome_rates": {"T=0": 0.12, "T=1": 0.18},
        "ate_diff_in_means": 0.06, "outcome_rate_overall": 0.15,
        "has_historical_score": True,
    }
    c.ate_barplot = _B64

    c.heterogeneity_assessment = {
        "heterogeneity": {
            "color": "#1a7f37", "level": "strong", "label": "Starke Heterogenität",
            "details": ["Das Champion-Modell zeigt deutlich heterogene Effekte."],
            "champion": "NonParamDML", "champion_qini": 0.0421, "champion_pv": 0.0312,
            "concentration": 2.4, "n_positive_qini": 3, "n_models": 4,
        },
        "hist_comparison": {
            "color": "#1a7f37", "level": "beats_both", "label": "Übertrifft historischen Score",
            "beats_qini": True, "champion_qini": 0.0421, "hist_name": "Score_alt",
            "hist_qini": 0.0310, "qini_diff": 0.0111,
            "beats_pv": True, "champion_pv": 0.0312, "hist_pv": 0.0250, "pv_diff": 0.0062,
        },
    }

    c.dataprep_info = {
        "data_files": ["train_2024.parquet"], "eval_files": [], "target": "conversion",
        "treatment": "treated", "score_name": "Score_alt", "fill_na_method": "median",
        "binary_target": True, "deduplicate": True, "deduplicate_id_column": "kunde_id",
        "balance_treatments": False, "score_as_feature": True, "feature_path": "features.yml",
        "multiple_files_option": "merge", "delimiter": ",", "output_path": "out/",
        "log_to_mlflow": True, "mlflow_experiment_name": "dataprep_exp",
        "treatment_replacement": {}, "categorical_columns": ["region", "segment"],
    }

    c.feature_selection_info = {
        "n_before": 120, "n_after": 50, "n_removed_correlation": 40,
        "n_removed_importance": 30, "n_after_correlation": 80,
    }

    c.tuning_scores = {
        "outcome": -0.412345, "propensity": -0.531234,
        "pseudo_NonParamDML": -0.221122,
    }
    c.tuning_skill_scores = {"outcome": 0.1234, "propensity": 0.0456}
    c.tuning_plan = [
        {"task_key": "outcome", "role": "Outcome E[Y|X]", "models": ["NonParamDML", "DRLearner"],
         "signature": "clf|both", "objective": "log_loss"},
        {"task_key": "propensity", "role": "Propensity E[T|X]", "models": ["NonParamDML", "DRLearner"],
         "signature": "clf|both", "objective": "log_loss"},
        {"task_key": "pseudo_NonParamDML", "role": "Pseudo-Outcome", "models": ["NonParamDML"],
         "signature": "reg|both", "objective": "neg_mse"},
    ]
    c.best_params = {
        "outcome": {"_learner_type": "lgbm", "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31},
        "propensity": {"_learner_type": "catboost", "iterations": 600, "depth": 6},
        "pseudo_NonParamDML": {"_learner_type": "lgbm", "n_estimators": 300, "num_leaves": 15},
    }

    c.fmt_info = {"scorer": "qini"}
    c.fmt_plan = [
        {"model": "NonParamDML", "method": "cache_values", "studies": 1, "trials": 50,
         "fits_per_trial": 5, "total_fits": 250, "note": "OOF-CV"},
        {"model": "DRLearner", "method": "cache_values", "studies": 1, "trials": 50,
         "fits_per_trial": 5, "total_fits": 250, "note": "OOF-CV"},
    ]
    c.fmt_best_params = {
        "NonParamDML": {"max_depth": 5, "min_samples_leaf": 50},
        "DRLearner": {"max_depth": 4, "min_samples_leaf": 80},
    }

    c.add_cft_info({
        "model_type": "CausalForestDML", "mode": "cache_values", "n_trials": 50,
        "n_trials_completed": 48, "single_fold": False, "scorer": "qini", "best_score": 0.0398,
        "best_params": {"max_depth": 7, "criterion": "het"},
    })
    c.cft_best_params = {
        "CausalForestDML": {"max_depth": 7, "min_weight_fraction_leaf": 0.01,
                            "min_var_fraction_leaf": 0.02, "criterion": "het"},
    }

    c.champion_name = "NonParamDML"
    c.model_metrics = {
        "NonParamDML": {"qini": 0.0421, "auuc": 0.0388, "policy_value": 0.0312, "uplift_at_20pct": 0.041},
        "DRLearner": {"qini": 0.0395, "auuc": 0.0361, "policy_value": 0.0290, "uplift_at_20pct": 0.038},
        "CausalForestDML": {"qini": 0.0372, "auuc": 0.0344, "policy_value": 0.0271, "uplift_at_20pct": 0.035},
        "CausalForest": {"qini": 0.0310, "auuc": 0.0288, "policy_value": 0.0233, "uplift_at_20pct": 0.030},
    }
    c.model_plots = {
        "NonParamDML": {"cate_distribution": _B64, "uplift_qini": _B64, "qini_plot": _B64},
        "DRLearner": {"cate_distribution": _B64, "uplift_qini": _B64},
        "CausalForestDML": {"cate_distribution": _B64},
        "CausalForest": {"cate_distribution": _B64},
    }

    c.surrogate_info = {
        "champion": "NonParamDML", "depth": 3, "n_leaves": 8,
        "metrics": {"qini": 0.0361, "auuc": 0.0331},
    }

    c.explainability_info = {"model_name": "NonParamDML"}
    c.explainability_plots = {"SHAP Summary": _B64, "SHAP Bar": _B64}

    c.step_durations = {
        "Datenaufbereitung": 12.3, "Feature-Selektion": 45.6,
        "Base-Learner-Tuning": 320.1, "Training": 88.7, "Evaluation": 22.4,
    }
    c.total_elapsed = 489.1

    return c


GOLDEN_EXT_PATH = Path(__file__).parent / "data" / "report_golden_external.html"


def _fixture_collector_external() -> ReportCollector:
    """Zweite Fixture: externe Validierung + FMT-best_scores (raw + penalisiert).

    Deckt Zweige ab, die die Hauptpfad-Fixture nicht auslöst: external-Pfad
    (Train+Eval-Karten, Eval-Datentabelle, Covariate-Shift-Hinweis), die
    FMT-Best-Scores-Tabelle inkl. penalisierter Spalte sowie den external-Zweig
    in Übersicht/Modellvergleich/Modell-Details.
    """
    c = _fixture_collector()
    c.config_summary = {**c.config_summary, "validate_on": "external"}
    # FMT-Best-Scores (raw + __adjusted) → Best-FMT-Scores-Tabelle mit Penalisiert-Spalte
    c.fmt_info = {"scorer": "qini", "best_scores": {
        "NonParamDML": 0.0421, "NonParamDML__adjusted": 0.0405,
        "DRLearner": 0.0395, "DRLearner__adjusted": 0.0380,
    }}
    # Separater Eval-Datensatz (löst external-Datenkarten + Covariate-Shift-Hinweis aus)
    c.eval_data_stats = {
        "n_rows": 4000, "n_features": 25, "n_treatment_groups": 2,
        "treatment_distribution": {"T=0": 2000, "T=1": 2000},
        "outcome_rates": {"T=0": 0.20, "T=1": 0.28},
        "ate_diff_in_means": 0.085, "outcome_rate_overall": 0.24,
        "has_historical_score": True,
    }
    return c


def _render_normalized(fixture=_fixture_collector) -> str:
    c = fixture()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "report.html"
        generate_html_report(c, str(out))
        return _normalize(out.read_text(encoding="utf-8"))


def test_report_golden_master():
    """Gerenderter Report (Timestamp maskiert) muss byte-identisch zum Golden sein."""
    assert GOLDEN_PATH.is_file(), (
        f"Golden-Snapshot fehlt: {GOLDEN_PATH}. Neu erzeugen mit: "
        f"python tests/test_report_golden.py --update"
    )
    expected = GOLDEN_PATH.read_text(encoding="utf-8")
    actual = _render_normalized()
    assert actual == expected, (
        "Report-Output weicht vom Golden-Snapshot ab. Falls beabsichtigt, Golden neu "
        "erzeugen mit: python tests/test_report_golden.py --update"
    )


def test_report_golden_master_external():
    """Zweiter Golden: external-Pfad + FMT-best_scores muss byte-identisch sein."""
    assert GOLDEN_EXT_PATH.is_file(), (
        f"Golden-Snapshot fehlt: {GOLDEN_EXT_PATH}. Neu erzeugen mit: "
        f"python tests/test_report_golden.py --update"
    )
    expected = GOLDEN_EXT_PATH.read_text(encoding="utf-8")
    actual = _render_normalized(_fixture_collector_external)
    assert actual == expected, (
        "External-Report-Output weicht vom Golden-Snapshot ab. Falls beabsichtigt, Golden "
        "neu erzeugen mit: python tests/test_report_golden.py --update"
    )


def test_report_renders_all_sections():
    """Sanity: Die Fixture löst alle erwarteten Sektionen aus."""
    html = _render_normalized()
    for sid in ["overview", "heterogeneity", "data", "dataprep", "feature_sel",
                "tuning", "fmt", "cft", "comparison", "surrogate", "explainability", "timing"]:
        assert f'id="{sid}"' in html, f"Sektion fehlt: {sid}"
    assert 'id="model_NonParamDML"' in html  # Per-Modell-Sektion


if __name__ == "__main__":
    if "--update" in sys.argv:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(_render_normalized(), encoding="utf-8")
        GOLDEN_EXT_PATH.write_text(_render_normalized(_fixture_collector_external), encoding="utf-8")
        print(f"Golden aktualisiert: {GOLDEN_PATH}")
        print(f"Golden aktualisiert: {GOLDEN_EXT_PATH}")
    else:
        print("Nutze --update, um den Golden-Snapshot neu zu erzeugen.")
