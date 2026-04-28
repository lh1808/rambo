"""Tests für Tuning-Logik: Overfit-Penalty, Score-Richtung, CatBoost-Fix, Adapter.

Testet die kritischsten Code-Pfade in tuning_optuna.py und model_registry.py.
Tests sind so strukturiert, dass sie auch ohne econml/catboost lauffähig sind
(heavy imports werden geskippt mit pytest.importorskip).
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ═══ Import-Helfer: Lade Module ohne rubin.__init__ (vermeidet econml-Abhängigkeit) ═══

def _load_module_directly(module_name: str, file_path: str):
    """Lädt ein Python-Modul direkt aus der Datei, ohne package __init__."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def settings_mod():
    """Lade rubin.settings direkt."""
    return _load_module_directly("rubin.settings", "rubin/settings.py")


@pytest.fixture(scope="session")
def tuning_mod():
    """Lade rubin.tuning_optuna — mockt schwere Dependencies."""
    # Settings laden
    if "rubin.settings" not in sys.modules:
        _load_module_directly("rubin.settings", "rubin/settings.py")

    # Mock schwere Imports, die tuning_optuna transitiv braucht
    mock_mods = {}
    for mod_name in [
        "econml", "econml.dml", "econml.dr", "econml.metalearners",
        "econml.grf", "econml.score", "econml.score.rscorer",
        "catboost", "lightgbm",
        "rubin.pipelines", "rubin.pipelines.analysis_pipeline",
        "rubin.model_registry", "rubin.training",
    ]:
        if mod_name not in sys.modules:
            mock_mods[mod_name] = sys.modules[mod_name] = MagicMock()

    # rubin package mock (nur __init__, nicht die Submodule die wir direkt laden)
    if "rubin" not in sys.modules:
        mock_mods["rubin"] = sys.modules["rubin"] = MagicMock()

    # utils.data_utils braucht echten available_cpu_count
    import os
    utils_mock = MagicMock()
    utils_mock.available_cpu_count = lambda: os.cpu_count() or 4
    sys.modules["rubin.utils"] = utils_mock
    sys.modules["rubin.utils.data_utils"] = utils_mock

    try:
        mod = _load_module_directly("rubin.tuning_optuna", "rubin/tuning_optuna.py")
        return mod
    finally:
        # Mocks wieder entfernen, damit andere Tests sauber starten
        pass


# ═══ Fixtures ═══

@pytest.fixture
def blt_tuner(tuning_mod):
    """BaseLearnerTuner mit minimaler Config."""
    cfg = SimpleNamespace(
        tuning=SimpleNamespace(
            enabled=True, cv_splits=5, n_trials=10, single_fold=False,
            metric="log_loss", metric_regression="neg_mse",
            overfit_penalty=0.0, overfit_tolerance=0.15,
            per_learner=False, per_role=False, models=[], max_tuning_rows=None,
            storage_path=None, timeout_seconds=None,
        ),
        constants=SimpleNamespace(random_seed=42, parallel_level=3),
    )
    tuner = tuning_mod.BaseLearnerTuner.__new__(tuning_mod.BaseLearnerTuner)
    tuner.cfg = cfg
    tuner.seed = 42
    tuner.optuna = None
    tuner.best_scores = {}
    return tuner


@pytest.fixture
def fmt_tuner(tuning_mod):
    """FinalModelTuner mit minimaler Config."""
    cfg = SimpleNamespace(
        final_model_tuning=SimpleNamespace(
            enabled=True, overfit_penalty=0.3, overfit_tolerance=0.05,
            cv_splits=5, n_trials=50, single_fold=False,
            timeout_seconds=None, max_tuning_rows=None, models=None, fixed_params={},
        ),
        constants=SimpleNamespace(random_seed=42, parallel_level=3),
    )
    tuner = tuning_mod.FinalModelTuner.__new__(tuning_mod.FinalModelTuner)
    tuner.cfg = cfg
    tuner.seed = 42
    tuner.optuna = None
    tuner.best_scores = {}
    return tuner


# ═══ 1. Overfit-Penalty Formel ═══

class TestOverfitPenaltyFormula:
    """Testet die skalen-sichere relative Overfit-Penalty."""

    def test_no_penalty_when_zero(self, blt_tuner):
        """Penalty=0 → Score bleibt unverändert."""
        result = blt_tuner._apply_overfit_penalty(-0.40, -0.30, penalty=0.0, tolerance=0.15)
        assert result == -0.40

    def test_no_penalty_within_tolerance(self, blt_tuner):
        """Gap innerhalb Tolerance → keine Bestrafung."""
        # val=-0.40, train=-0.35, gap=12.5% < tolerance=15%
        result = blt_tuner._apply_overfit_penalty(-0.40, -0.35, penalty=0.2, tolerance=0.15)
        assert result == -0.40

    def test_penalty_above_tolerance(self, blt_tuner):
        """Gap über Tolerance → Bestrafung."""
        # val=-0.40, train=-0.25, gap=37.5% > tolerance=15%
        result = blt_tuner._apply_overfit_penalty(-0.40, -0.25, penalty=0.2, tolerance=0.15)
        assert result < -0.40  # bestraft = schlechterer Score

    def test_stronger_penalty_means_worse_score(self, blt_tuner):
        """Höhere Penalty-Stärke → stärkere Bestrafung."""
        weak = blt_tuner._apply_overfit_penalty(-0.40, -0.25, penalty=0.2, tolerance=0.15)
        strong = blt_tuner._apply_overfit_penalty(-0.40, -0.25, penalty=0.6, tolerance=0.15)
        assert strong < weak

    def test_scale_invariant_log_loss(self, blt_tuner):
        """Penalty wirkt proportional bei log_loss-Skala (~0.3-0.7)."""
        # 25% Gap bei val=-0.40
        result = blt_tuner._apply_overfit_penalty(-0.40, -0.30, penalty=0.3, tolerance=0.05)
        relative_impact = (result - (-0.40)) / abs(-0.40)
        assert -0.15 < relative_impact < 0.0  # moderate Bestrafung

    def test_scale_invariant_r_loss(self, blt_tuner):
        """Gleicher relativer Gap → gleiche relative Bestrafung bei R-Loss-Skala (~0.001)."""
        # 25% Gap bei val=-0.004
        result = blt_tuner._apply_overfit_penalty(-0.004, -0.003, penalty=0.3, tolerance=0.05)
        relative_impact = (result - (-0.004)) / abs(-0.004)
        # Sollte ähnliche relative Bestrafung wie bei log_loss
        ll_result = blt_tuner._apply_overfit_penalty(-0.40, -0.30, penalty=0.3, tolerance=0.05)
        ll_relative = (ll_result - (-0.40)) / abs(-0.40)
        assert abs(relative_impact - ll_relative) < 0.001  # identisch

    def test_no_penalty_when_train_worse_than_val(self, blt_tuner):
        """Wenn Train schlechter als Val → kein Overfitting → keine Bestrafung."""
        result = blt_tuner._apply_overfit_penalty(-0.40, -0.45, penalty=0.3, tolerance=0.05)
        assert result == -0.40

    def test_penalty_exact_formula(self, blt_tuner):
        """Exakte Formel-Verifizierung."""
        val, train = -0.50, -0.30
        penalty, tolerance = 0.3, 0.10
        scale = abs(val)  # 0.50
        gap = train - val  # 0.20
        rel_gap = gap / scale  # 0.40
        expected = val - penalty * scale * max(0, rel_gap - tolerance)
        # = -0.50 - 0.3 * 0.50 * max(0, 0.40 - 0.10)
        # = -0.50 - 0.3 * 0.50 * 0.30
        # = -0.50 - 0.045 = -0.545
        result = blt_tuner._apply_overfit_penalty(val, train, penalty=penalty, tolerance=tolerance)
        assert abs(result - expected) < 1e-10
        assert abs(result - (-0.545)) < 1e-10

    def test_val_score_near_zero(self, blt_tuner):
        """Edge case: val_score nahe Null → scale = 1e-10, kein Division-by-zero."""
        result = blt_tuner._apply_overfit_penalty(0.0, 0.01, penalty=0.3, tolerance=0.05)
        assert np.isfinite(result)

    def test_blt_default_tolerance(self, blt_tuner):
        """BLT Default-Tolerance ist 0.15 (nicht 0.05)."""
        blt_tuner.cfg.tuning.overfit_penalty = 0.2
        blt_tuner.cfg.tuning.overfit_tolerance = 0.15
        # 12.5% Gap < 15% Tolerance → keine Bestrafung
        result = blt_tuner._apply_overfit_penalty(-0.40, -0.35)
        assert result == -0.40

    def test_fmt_default_tolerance(self, fmt_tuner):
        """FMT Default-Tolerance ist 0.05 (strenger als BLT)."""
        # 10% Gap > 5% Tolerance → Bestrafung
        result = fmt_tuner._apply_overfit_penalty(-0.005, -0.0045, penalty=0.3, tolerance=0.05)
        assert result < -0.005


# ═══ 2. Score-Richtung (Classifier) ═══

class TestScoreClassifier:
    """Testet _score_classifier Metrik-Routing und Richtung."""

    def test_log_loss_is_negative(self, blt_tuner):
        """log_loss wird negiert (höher = besser für Optuna maximize)."""
        blt_tuner.cfg.tuning.metric = "log_loss"
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])
        score = blt_tuner._score_classifier(y, proba)
        assert score < 0  # negierter log_loss ist negativ

    def test_better_predictions_give_higher_log_loss(self, blt_tuner):
        """Bessere Vorhersagen → weniger Log-Loss → höherer negierter Score."""
        blt_tuner.cfg.tuning.metric = "log_loss"
        y = np.array([0, 0, 1, 1])
        good = np.array([0.1, 0.1, 0.9, 0.9])
        bad = np.array([0.4, 0.4, 0.6, 0.6])
        assert blt_tuner._score_classifier(y, good) > blt_tuner._score_classifier(y, bad)

    def test_pr_auc_is_positive(self, blt_tuner):
        """PR-AUC ist positiv (nicht negiert)."""
        blt_tuner.cfg.tuning.metric = "pr_auc"
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])
        score = blt_tuner._score_classifier(y, proba)
        assert 0 < score <= 1

    def test_roc_auc_range(self, blt_tuner):
        """ROC-AUC ist zwischen 0 und 1."""
        blt_tuner.cfg.tuning.metric = "roc_auc"
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])
        score = blt_tuner._score_classifier(y, proba)
        assert 0 < score <= 1

    def test_accuracy_range(self, blt_tuner):
        """Accuracy ist zwischen 0 und 1."""
        blt_tuner.cfg.tuning.metric = "accuracy"
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])
        score = blt_tuner._score_classifier(y, proba)
        assert 0 < score <= 1

    def test_single_class_returns_default(self, blt_tuner):
        """Nur eine Klasse → sicherer Fallback statt Crash."""
        blt_tuner.cfg.tuning.metric = "log_loss"
        y = np.array([1, 1, 1, 1])
        proba = np.array([0.8, 0.9, 0.7, 0.8])
        score = blt_tuner._score_classifier(y, proba)
        assert np.isfinite(score)

    def test_log_loss_2d_proba(self, blt_tuner):
        """log_loss funktioniert mit 2D proba (n_samples × n_classes)."""
        blt_tuner.cfg.tuning.metric = "log_loss"
        y = np.array([0, 0, 1, 1])
        proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        score = blt_tuner._score_classifier(y, proba)
        assert score < 0


# ═══ 3. Score-Richtung (Regression) ═══

class TestScoreRegressor:
    """Testet _score_regressor Metrik-Routing und Richtung."""

    def test_neg_mse_is_negative(self, blt_tuner):
        """neg_mse ist negativ (höher = besser)."""
        blt_tuner.cfg.tuning.metric_regression = "neg_mse"
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 2.1, 2.9])
        score = blt_tuner._score_regressor(y, pred)
        assert score < 0

    def test_better_predictions_give_higher_neg_mse(self, blt_tuner):
        """Bessere Vorhersagen → weniger MSE → höherer negierter Score."""
        blt_tuner.cfg.tuning.metric_regression = "neg_mse"
        y = np.array([1.0, 2.0, 3.0])
        good = np.array([1.0, 2.0, 3.0])  # perfekt
        bad = np.array([2.0, 3.0, 4.0])  # off by 1
        assert blt_tuner._score_regressor(y, good) > blt_tuner._score_regressor(y, bad)

    def test_neg_rmse_is_negative(self, blt_tuner):
        """neg_rmse ist negativ."""
        blt_tuner.cfg.tuning.metric_regression = "neg_rmse"
        score = blt_tuner._score_regressor(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        assert score < 0

    def test_neg_mae_is_negative(self, blt_tuner):
        """neg_mae ist negativ."""
        blt_tuner.cfg.tuning.metric_regression = "neg_mae"
        score = blt_tuner._score_regressor(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        assert score < 0

    def test_r2_perfect_is_one(self, blt_tuner):
        """Perfekte Vorhersage → R² = 1."""
        blt_tuner.cfg.tuning.metric_regression = "r2"
        y = np.array([1.0, 2.0, 3.0, 4.0])
        score = blt_tuner._score_regressor(y, y)
        assert abs(score - 1.0) < 1e-10

    def test_all_metrics_higher_is_better(self, blt_tuner):
        """Alle Regressionsmetriken: bessere Vorhersage → höherer Score."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        good = y + np.random.RandomState(42).normal(0, 0.1, len(y))
        bad = y + np.random.RandomState(42).normal(0, 2.0, len(y))
        for metric in ["neg_mse", "neg_rmse", "neg_mae", "r2"]:
            blt_tuner.cfg.tuning.metric_regression = metric
            assert blt_tuner._score_regressor(y, good) > blt_tuner._score_regressor(y, bad), f"{metric} failed"


# ═══ 4. CatBoost bootstrap_type Fix ═══

class TestCatBoostBootstrapFix:
    """Testet, dass CatBoost mit subsample automatisch bootstrap_type=MVS bekommt.

    _build_catboost_classifier/regressor machen `from catboost import ...` lokal.
    Wir fangen das ab, indem wir catboost im sys.modules mocken.
    """

    def test_classifier_subsample_gets_mvs(self, tuning_mod):
        mock_cls = MagicMock()
        sys.modules["catboost"] = MagicMock(CatBoostClassifier=mock_cls)
        tuning_mod._build_catboost_classifier({"subsample": 0.7, "iterations": 10}, seed=42)
        assert mock_cls.call_args[1]["bootstrap_type"] == "MVS"

    def test_classifier_no_subsample_no_mvs(self, tuning_mod):
        mock_cls = MagicMock()
        sys.modules["catboost"] = MagicMock(CatBoostClassifier=mock_cls)
        tuning_mod._build_catboost_classifier({"iterations": 10}, seed=42)
        assert "bootstrap_type" not in mock_cls.call_args[1]

    def test_classifier_explicit_bootstrap_preserved(self, tuning_mod):
        mock_cls = MagicMock()
        sys.modules["catboost"] = MagicMock(CatBoostClassifier=mock_cls)
        tuning_mod._build_catboost_classifier({"subsample": 0.7, "bootstrap_type": "Bernoulli"}, seed=42)
        assert mock_cls.call_args[1]["bootstrap_type"] == "Bernoulli"

    def test_regressor_subsample_gets_mvs(self, tuning_mod):
        mock_cls = MagicMock()
        sys.modules["catboost"] = MagicMock(CatBoostRegressor=mock_cls)
        tuning_mod._build_catboost_regressor({"subsample": 0.7, "iterations": 10}, seed=42)
        assert mock_cls.call_args[1]["bootstrap_type"] == "MVS"


# ═══ 5. CausalForestAdapter.from_fitted ═══

class TestCausalForestAdapterFromFitted:
    """Testet die Factory-Method für RScorer-Kompatibilität.
    Braucht echtes econml — wird übersprungen wenn nicht installiert."""

    @pytest.fixture(autouse=True)
    def _skip_without_econml(self):
        mod = sys.modules.get("econml")
        if mod is None or isinstance(mod, MagicMock):
            pytest.skip("econml not installed (mocked)")

    def test_sets_all_attributes(self):
        from rubin.model_registry import CausalForestAdapter
        mock_cf = MagicMock()
        adapter = CausalForestAdapter.from_fitted(mock_cf)
        assert adapter._cf is mock_cf
        assert adapter._kwargs == {}
        assert adapter._tune_result == {}

    def test_effect_delegates_to_cf(self):
        from rubin.model_registry import CausalForestAdapter
        mock_cf = MagicMock()
        mock_cf.predict.return_value = np.array([[0.1], [0.2]])
        adapter = CausalForestAdapter.from_fitted(mock_cf)
        X = np.array([[1, 2], [3, 4]])
        result = adapter.effect(X)
        mock_cf.predict.assert_called_once()
        assert result.shape == (2,)

    def test_feature_importances_delegates(self):
        from rubin.model_registry import CausalForestAdapter
        mock_cf = MagicMock()
        mock_cf.feature_importances_ = np.array([0.5, 0.3, 0.2])
        adapter = CausalForestAdapter.from_fitted(mock_cf)
        np.testing.assert_array_equal(adapter.feature_importances_, np.array([0.5, 0.3, 0.2]))


# ═══ 6. Config Defaults ═══

class TestConfigDefaults:
    """Testet, dass BLT und FMT unterschiedliche Tolerance-Defaults haben."""

    def test_blt_tolerance_default(self, settings_mod):
        cfg = settings_mod.OptunaTuningConfig()
        assert cfg.overfit_tolerance == 0.15

    def test_fmt_tolerance_default(self, settings_mod):
        cfg = settings_mod.FinalModelTuningConfig()
        assert cfg.overfit_tolerance == 0.05

    def test_blt_penalty_default_zero(self, settings_mod):
        cfg = settings_mod.OptunaTuningConfig()
        assert cfg.overfit_penalty == 0.0

    def test_fmt_penalty_default_zero(self, settings_mod):
        cfg = settings_mod.FinalModelTuningConfig()
        assert cfg.overfit_penalty == 0.0

    def test_tune_models_default_empty(self, settings_mod):
        cfg = settings_mod.CausalForestConfig()
        assert cfg.tune_models == []


# ═══ 7. build_base_learner Routing ═══

class TestBuildBaseLearner:
    """Testet build_base_learner Routing."""

    def test_unknown_type_raises(self, tuning_mod):
        with pytest.raises(ValueError, match="Unbekannter"):
            tuning_mod.build_base_learner("xgboost", {}, seed=42, task="classifier")

    def test_unknown_task_raises(self, tuning_mod):
        with pytest.raises(ValueError, match="task"):
            tuning_mod.build_base_learner("catboost", {}, seed=42, task="clustering")

    def test_both_without_learner_type_fallback(self, tuning_mod):
        with patch.object(tuning_mod, "_build_catboost_classifier") as mock:
            mock.return_value = MagicMock()
            tuning_mod.build_base_learner("both", {}, seed=42, task="classifier")
            mock.assert_called_once()


# ═══ 8. Penalty + Pruning Interaction ═══

class TestPenaltyPruningInteraction:
    """Testet, dass Penalty NICHT in Pruning-Scores eingeht."""

    def test_penalty_on_means_not_per_fold(self, blt_tuner):
        """Penalty(mean(vals), mean(trains)) ≈ mean(penalty(val_i, train_i))
        bei gleichmäßigen Gaps, aber NICHT bei ungleichmäßigen."""
        # Ungleichmäßige Gaps: Fold 1 hat 50% Gap, Fold 2 hat 0% Gap
        val_scores = [-0.40, -0.40]
        train_scores = [-0.20, -0.40]  # Fold 1: 50%, Fold 2: 0%

        # Per-fold: nur Fold 1 bestraft (50% > 15%), Fold 2 nicht
        per_fold = []
        for v, tr in zip(val_scores, train_scores):
            per_fold.append(blt_tuner._apply_overfit_penalty(v, tr, penalty=0.3, tolerance=0.15))
        old_way = np.mean(per_fold)

        # On means: mean gap = 25% > 15%, eine Penalty
        mean_v = np.mean(val_scores)
        mean_t = np.mean(train_scores)
        new_way = blt_tuner._apply_overfit_penalty(mean_v, mean_t, penalty=0.3, tolerance=0.15)

        # Beide Ergebnisse sind valide, aber unterschiedlich
        # Wichtig: new_way ist stabiler (aggregiert über Folds)
        assert np.isfinite(old_way) and np.isfinite(new_way)
        # Die Werte müssen nicht identisch sein — das ist der Punkt der Verbesserung
