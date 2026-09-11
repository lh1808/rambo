"""Regressionswächter für den realen Trial-Fit-Pfad des BLT.

Ein in _fit_model fehlplatzierter Log-Aufruf (NameError '_tlog') ließ in
Produktion JEDEN Optuna-Trial fehlschlagen — das Tuning fiel still auf
Default-Parameter zurück, weil kein Suite-Test _fit_model je real ausführte.
Diese Tests fitten echte Mini-Modelle durch exakt den Produktions-Codepfad.

Eigene Datei mit sys.modules-Purge: test_tuning.py mockt schwere Module
(build_base_learner als MagicMock) prozessweit — hier werden garantiert die
ECHTEN rubin.tuning-Module geladen, unabhängig von der Testreihenfolge."""
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


def _fresh_real_tuner():
    # Alles purgen, was test_tuning.py als MagicMock installiert hat, plus die
    # rubin.tuning-Module selbst — danach lädt der Import garantiert echt.
    from unittest import mock as _mock
    for name in list(sys.modules):
        m = sys.modules[name]
        if name.startswith("rubin.tuning") or isinstance(m, _mock.MagicMock):
            del sys.modules[name]
    from rubin.tuning.base_learner import BaseLearnerTuner
    cfg = SimpleNamespace(
        tuning=SimpleNamespace(enabled=True, cv_splits=3, n_trials=2,
            single_fold=False, metric="log_loss", metric_regression="neg_mse",
            overfit_penalty=0.0, overfit_tolerance=0.15, per_learner=False,
            per_role=False, models=[], max_tuning_rows=None,
            storage_path=None, timeout_seconds=None),
        constants=SimpleNamespace(random_seed=42, tuning_seed=18, parallel_level=3),
        base_learner=SimpleNamespace(type="lgbm", fixed_params={}),
        study_type="rct",
    )
    return BaseLearnerTuner(cfg)


def _mini_xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("abcd"))
    y = (rng.random(150) > 0.5).astype(int)
    return X, y


def test_tune_all_end_to_end_produces_tuned_params():
    """Königs-Wächter: die komplette BLT-Strecke real — tune_all über
    _build_plan, _create_study, das Study-Start-Log (dessen %-Format hier
    mitgeprüft wird), Trials durch _fit_model und die Best-Param-Übernahme.
    Schlägt fehl, wenn Trials scheitern und still Default-Parameter
    zurückkommen (exakt der Produktions-Vorfall)."""
    t = _fresh_real_tuner_full()
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    T = pd.Series((rng.random(200) > 0.5).astype(int))
    Y = pd.Series((rng.random(200) > 0.5).astype(int))
    tuned = t.tune_all(["NonParamDML"], X=X, Y=Y, T=T)
    roles = tuned.get("NonParamDML", {})
    assert roles, tuned
    assert all(len(p) > 0 for p in roles.values()), ("Fallback statt Tuning", roles)


def _fresh_real_tuner_full():
    """Wie _fresh_real_tuner, aber mit allen Feldern, die der volle
    tune_all-Pfad liest (models=None bedeutet: alle Modelle tunen)."""
    from unittest import mock as _mock
    for name in list(sys.modules):
        m = sys.modules[name]
        if name.startswith("rubin.tuning") or isinstance(m, _mock.MagicMock):
            del sys.modules[name]
    from rubin.tuning.base_learner import BaseLearnerTuner
    cfg = SimpleNamespace(
        tuning=SimpleNamespace(enabled=True, cv_splits=3, n_trials=2,
            single_fold=False, metric="log_loss", metric_regression="neg_mse",
            overfit_penalty=0.0, overfit_tolerance=0.15, per_learner=False,
            per_role=False, models=None, max_tuning_rows=None,
            storage_path=None, timeout_seconds=None, optuna_seed=7,
            search_space=None),
        constants=SimpleNamespace(random_seed=42, tuning_seed=18, parallel_level=3),
        base_learner=SimpleNamespace(type="lgbm",
            fixed_params={"n_estimators": 8, "num_leaves": 7}, search_space=None),
        data_processing=SimpleNamespace(dml_crossfit_folds=3),
        study_type="rct",
    )
    return BaseLearnerTuner(cfg)


class TestFitModelPath:
    def test_fit_model_classifier_runs(self):
        t = _fresh_real_tuner()
        X, y = _mini_xy()
        m = t._fit_model({"n_estimators": 8, "num_leaves": 7}, X, y, "classifier")
        assert m.predict_proba(X).shape == (150, 2)

    def test_objective_returns_finite_score(self):
        # Voller Objective-Pfad (fit + score je Fold), trial=None wie im Fallback
        t = _fresh_real_tuner()
        X, y = _mini_xy()
        score = t._objective_all_classification(
            {"n_estimators": 8, "num_leaves": 7}, X_mat=X, target=y,
            strata=y, train_ratio=1.0, trial=None, allow_penalty=False)
        assert np.isfinite(score)
