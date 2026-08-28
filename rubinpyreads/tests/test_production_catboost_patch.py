"""Reproduziert den realen wg-Prod-Crash: Metalearner (T-Learner) mit
CatBoost-Basismodellen und cat_features — econml konvertiert X in
const_marginal_effect() zu float-Arrays; ohne den categorical_patch crasht
CatBoost ("'data' is numpy array of floating point …"). Die Analyse lief,
weil sie den Patch installiert; die Produktion muss die predict-Seite
spiegeln (score_dataframe wickelt die Predicts jetzt in den Kontext)."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from econml.metalearners import TLearner

from rubin.utils.categorical_patch import patch_categorical_features

_PROD = Path(__file__).resolve().parents[1] / "production"
if str(_PROD) not in sys.path:
    sys.path.insert(0, str(_PROD))


def _fit_metalearner_like_analysis():
    rng = np.random.RandomState(7)
    n = 400
    X = pd.DataFrame({
        "F0": rng.normal(size=n),
        "KAT": pd.Series(rng.randint(0, 4, size=n), dtype="int32").astype("category"),
        "F2": rng.normal(size=n),
    })
    T = rng.randint(0, 2, size=n)
    y = X["F0"].to_numpy() + 0.5 * T + rng.normal(scale=0.1, size=n)
    tl = TLearner(models=CatBoostRegressor(iterations=15, depth=3, verbose=0, allow_writing_files=False))
    with patch_categorical_features(X, base_learner_type="catboost"):
        tl.fit(y, T, X=X)  # fit-Wrapper injiziert cat_features — wie in der Analyse
    return tl, X


class TestMetalearnerCatboostPredict:
    def test_unpatched_predict_reproduces_prod_crash(self):
        tl, X = _fit_metalearner_like_analysis()
        with pytest.raises(Exception, match="floating point|cat_features"):
            tl.const_marginal_effect(X)  # ohne Kontext = alte Produktion

    def test_patched_predict_succeeds_and_is_finite(self):
        tl, X = _fit_metalearner_like_analysis()
        with patch_categorical_features(X, base_learner_type="catboost"):
            eff = np.asarray(tl.const_marginal_effect(X))
        assert eff.shape[0] == len(X) and np.isfinite(eff).all()

    def test_score_dataframe_wraps_predicts_in_patch_context(self):
        """Quelltext-Invariante: Die Produktions-Predicts (SCORE_P/B/extra)
        stehen im patch_categorical_features-Kontext mit Xp als Indexquelle."""
        src = (_PROD / "run_scoring.py").read_text(encoding="utf-8")
        m = re.search(
            r'with patch_categorical_features\(Xp, base_learner_type="catboost"\):'
            r'(.*?)_nan_scores', src, re.S)
        assert m, "Patch-Kontext um die Predicts fehlt"
        inner = m.group(1)
        for marker in ('"SCORE_P"', '"SCORE_B"', "extra_models"):
            assert marker in inner, f"{marker} nicht im Patch-Kontext"
