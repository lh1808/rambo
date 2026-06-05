"""Tests für AnalysisPipeline._log_surrogate_tree_info (Surrogate-Baum-Info-Logging).

Regression: Für LGBM enthält ``booster_.dump_model()`` keinen ``max_depth``-Key
(weder pro Baum noch top-level) — die Tiefe muss aus der ``tree_structure`` abgeleitet
werden. Vor dem Fix lieferte die Funktion für LGBM immer ``depth=None``. Außerdem darf
sie für unbekannte/kaputte Modelle nie crashen und stets ein 2-Tupel zurückgeben
(der Aufrufer entpackt ``depth, n_leaves = ...``).
"""
import warnings

import numpy as np
import pytest

from rubin.pipelines.analysis_pipeline import AnalysisPipeline

_f = AnalysisPipeline._log_surrogate_tree_info


def _xy(n=3000, p=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=n)
    return X, y


def test_lgbm_depth_not_none_and_leaves_extracted():
    lgbm = pytest.importorskip("lightgbm")
    X, y = _xy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = lgbm.LGBMRegressor(n_estimators=1, num_leaves=31, max_depth=5,
                               min_child_samples=5, verbose=-1).fit(X, y)
    depth, n_leaves = _f(m, "lgbm")
    # Kern-Regression: Tiefe darf NICHT None sein (Bug: kein max_depth-Key in dump_model).
    assert isinstance(depth, int) and depth >= 1, f"LGBM-Tiefe sollte abgeleitet werden, war {depth!r}"
    assert depth <= 5, "abgeleitete Tiefe darf max_depth nicht überschreiten"
    assert isinstance(n_leaves, int) and n_leaves >= 2


def test_catboost_depth_and_leaves():
    cb = pytest.importorskip("catboost")
    X, y = _xy()
    m = cb.CatBoostRegressor(depth=6, iterations=5, verbose=False,
                             allow_writing_files=False).fit(X, y)
    depth, n_leaves = _f(m, "catboost")
    assert depth == 6
    assert n_leaves == 2 ** 6  # symmetrischer Baum


def test_none_model_returns_tuple_no_crash():
    # Aufrufer entpackt das Ergebnis → muss immer ein 2-Tupel sein, nie None.
    assert _f(None, "lgbm") == (None, None)


def test_unknown_base_type_returns_tuple():
    assert _f(object(), "randomforest") == (None, None)


def test_exception_path_returns_tuple():
    class Broken:
        @property
        def booster_(self):
            raise RuntimeError("boom")
    assert _f(Broken(), "lgbm") == (None, None)
