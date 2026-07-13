"""Tests für rubin.utils.categorical_patch.

Der Patch injiziert cat_indices KLASSENWEIT (prozessglobal) in LGBM/CatBoost.
Diese Suite pinnt die impliziten Garantien fest, auf die die Pipeline baut:

1. Restauration nach normalem Exit UND nach Exception (kein State-Leak).
2. Sequenzielle Kontexte (FS-Patch → Training-Patch mit post-FS-Indizes)
   verwenden jeweils ihre eigenen Indizes.
3. Verschachtelte Kontexte verhalten sich LIFO (innerer gewinnt, äußerer
   wird beim inneren Exit wiederhergestellt).
4. Parallele Fits DESSELBEN Schemas innerhalb eines Kontexts (Optuna-
   Threading-Szenario) funktionieren.
5. CatBoost-Roundtrip: fit/predict auf numpy-float-X mit kategorialer
   Spalte läuft durch (float→int32-Konversion + cat_features-Injektion).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partialmethod

import numpy as np
import pandas as pd
import pytest

lgbm = pytest.importorskip("lightgbm")
catboost = pytest.importorskip("catboost")

from rubin.utils.categorical_patch import patch_categorical_features  # noqa: E402


def _make_X(n=80, cat_cols=("c0",), num_cols=("f0", "f1"), seed=0):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(size=n) for c in num_cols}
    for c in cat_cols:
        data[c] = pd.Series(rng.integers(0, 4, size=n)).astype("category")
    return pd.DataFrame(data)


def _lgbm_fit_keywords():
    """Injizierte categorical_feature-Indizes des aktuellen LGBMRegressor.fit
    (None, wenn ungepatcht). Zugriff via __dict__ — partialmethod ist ein
    Descriptor, Attributzugriff würde ihn zu einem partial auflösen."""
    f = lgbm.LGBMRegressor.__dict__.get("fit")
    if isinstance(f, partialmethod):
        return f.keywords.get("categorical_feature")
    return None


class TestRestoration:
    def test_patch_applied_and_restored(self):
        orig_cls_fit = lgbm.LGBMClassifier.__dict__["fit"]
        orig_reg_fit = lgbm.LGBMRegressor.__dict__["fit"]
        X = _make_X()
        with patch_categorical_features(X, base_learner_type="lgbm") as ci:
            assert ci, "Kategoriale Spalte wurde nicht erkannt"
            assert _lgbm_fit_keywords() == ci
        assert lgbm.LGBMClassifier.__dict__["fit"] is orig_cls_fit
        assert lgbm.LGBMRegressor.__dict__["fit"] is orig_reg_fit

    def test_restore_on_exception(self):
        orig_reg_fit = lgbm.LGBMRegressor.__dict__["fit"]
        orig_cb_fit = catboost.CatBoostRegressor.fit
        X = _make_X()
        with pytest.raises(RuntimeError):
            with patch_categorical_features(X, base_learner_type="both"):
                raise RuntimeError("boom")
        assert lgbm.LGBMRegressor.__dict__["fit"] is orig_reg_fit
        assert catboost.CatBoostRegressor.fit is orig_cb_fit

    def test_no_categorical_columns_no_patch(self):
        orig = lgbm.LGBMRegressor.__dict__["fit"]
        X = _make_X(cat_cols=())
        with patch_categorical_features(X, base_learner_type="lgbm") as ci:
            assert ci == []
            assert lgbm.LGBMRegressor.__dict__["fit"] is orig  # unverändert


class TestContextComposition:
    def test_sequential_contexts_use_own_indices(self):
        """FS-Patch → Training-Patch: nach Feature-Selektion verschieben sich
        die Spaltenindizes — jeder Kontext muss seine eigenen injizieren."""
        X_pre = _make_X(cat_cols=("c0",), num_cols=("f0", "f1", "f2"))
        X_post = X_pre.drop(columns=["f0"])  # FS entfernt eine Spalte davor

        with patch_categorical_features(X_pre, base_learner_type="lgbm") as ci_pre:
            assert _lgbm_fit_keywords() == ci_pre
        with patch_categorical_features(X_post, base_learner_type="lgbm") as ci_post:
            assert _lgbm_fit_keywords() == ci_post
        assert ci_pre != ci_post, "Testaufbau: Indizes sollten sich verschieben"
        assert _lgbm_fit_keywords() is None  # restauriert

    def test_nested_contexts_lifo(self):
        orig = lgbm.LGBMRegressor.__dict__["fit"]
        X_outer = _make_X(cat_cols=("c0",), num_cols=("f0", "f1", "f2"))
        X_inner = X_outer.drop(columns=["f0"])
        with patch_categorical_features(X_outer, base_learner_type="lgbm") as ci_o:
            assert _lgbm_fit_keywords() == ci_o
            with patch_categorical_features(X_inner, base_learner_type="lgbm") as ci_i:
                assert _lgbm_fit_keywords() == ci_i  # innerer gewinnt
            assert _lgbm_fit_keywords() == ci_o  # äußerer wiederhergestellt
            # Fit muss unter dem restaurierten äußeren Patch funktionsfähig sein
            # (Sicherung via Attributzugriff statt __dict__ würde hier das
            # self-Binding brechen)
            m = lgbm.LGBMRegressor(n_estimators=5, verbose=-1)
            m.fit(X_outer.astype(float).to_numpy(), np.arange(len(X_outer), dtype=float))
        assert lgbm.LGBMRegressor.__dict__["fit"] is orig


class TestFunctionalFits:
    def test_thread_parallel_lgbm_fits_same_schema(self):
        """Optuna-Szenario: mehrere Trials fitten parallel im selben Kontext
        auf demselben Schema — EconML-Stil mit numpy-float-X."""
        X = _make_X(n=200)
        rng = np.random.default_rng(1)
        y = rng.normal(size=len(X))
        X_np = X.astype(float).to_numpy()  # EconML konvertiert zu float64

        def _fit_one(seed):
            m = lgbm.LGBMRegressor(n_estimators=10, random_state=seed, verbose=-1)
            m.fit(X_np, y)
            return m.predict(X_np)

        with patch_categorical_features(X, base_learner_type="lgbm"):
            with ThreadPoolExecutor(max_workers=4) as ex:
                preds = list(ex.map(_fit_one, range(4)))
        for p in preds:
            assert np.all(np.isfinite(p)) and len(p) == len(X)

    def test_catboost_numpy_float_roundtrip(self):
        """EconML übergibt numpy float64 — der Patch muss cat_features
        injizieren und die kategorialen Spalten float→int32 konvertieren,
        sodass fit UND predict durchlaufen."""
        X = _make_X(n=120)
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, size=len(X))
        X_np = X.astype(float).to_numpy()

        with patch_categorical_features(X, base_learner_type="catboost") as ci:
            m = catboost.CatBoostClassifier(iterations=5, verbose=0, allow_writing_files=False)
            m.fit(X_np, y)
            # CatBoost speichert die Trainings-cat_features — Roundtrip beweist Konsistenz
            proba = m.predict_proba(X_np)
            score = m.score(X_np, y)
        assert proba.shape == (len(X), 2)
        assert np.all(np.isfinite(proba))
        assert 0.0 <= score <= 1.0
        # Ohne Patch würde derselbe fit auf float-X mit cat_features scheitern:
        m2 = catboost.CatBoostClassifier(iterations=5, verbose=0, allow_writing_files=False)
        with pytest.raises(Exception):
            m2.fit(X_np, y, cat_features=ci)
