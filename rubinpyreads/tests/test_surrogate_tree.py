"""Surrogate-Tree: Wirksamkeit der Config-Parameter bei BEIDEN Base-Learnern
und Konsistenz Report ↔ Bundle.

Regression für zwei Praxis-Befunde:
1. CatBoost ignorierte min_data_in_leaf (und kannte keine Blattgrenze) beim
   Default grow_policy=SymmetricTree STILL — die Config versprach Wirkung,
   hatte aber keine. Fix: grow_policy=Lossguide (+ max_leaves-Mapping).
2. Der Bundle-Export fittete einen ZWEITEN Baum auf anderen Targets als der
   Report-Baum — Produktion scorte ein Modell, das kein Report je beschrieb.
   Fix: Bundle übernimmt models[SURROGATE_MODEL_NAME] (Quelltext-Invariante).
"""
import re

import numpy as np
import pandas as pd
import pytest

from rubin.settings import AnalysisConfig


def _mini_cfg(base_type: str, **surr):
    return AnalysisConfig.model_validate({
        "data_files": {"x_file": "x.parquet", "t_file": "t.parquet", "y_file": "y.parquet"},
        "base_learner": {"type": base_type},
        "surrogate_tree": {"enabled": True, **surr},
        "mlflow": {"experiment_name": "t"},
    })


def _leaf_sizes(tree, X):
    pred = tree.predict(X)
    return pd.Series(pred).groupby(pred).size()


@pytest.mark.parametrize("base_type", ["lgbm", "catboost"])
class TestSurrogateParamsEffective:
    def _build(self, base_type, **surr):
        from rubin.pipelines.analysis_pipeline import AnalysisPipeline
        cfg = _mini_cfg(base_type, **surr)
        return AnalysisPipeline.__new__(AnalysisPipeline)._build_surrogate_regressor(cfg)

    def test_num_leaves_caps_leaf_count(self, base_type):
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.normal(size=(3000, 5)), columns=list("ABCDE"))
        y = X["A"] * 2 + (X["B"] > 0) * 1.5 + rng.normal(0, 0.05, 3000)
        tree = self._build(base_type, num_leaves=8, min_samples_leaf=10)
        tree.fit(X, y)
        assert len(_leaf_sizes(tree, X)) <= 8

    def test_min_samples_leaf_effective(self, base_type):
        """Der Kern-Regressionstest: kleine vs. große Blatt-Mindestgröße muss
        die Baumstruktur ändern (CatBoost/SymmetricTree: identisch → Bug)."""
        rng = np.random.RandomState(1)
        X = pd.DataFrame(rng.normal(size=(2000, 4)), columns=list("ABCD"))
        y = X["A"] * 2 + rng.normal(0, 0.1, 2000)
        t_small = self._build(base_type, num_leaves=31, min_samples_leaf=10)
        t_big = self._build(base_type, num_leaves=31, min_samples_leaf=800)
        t_small.fit(X, y); t_big.fit(X, y)
        assert not np.allclose(t_small.predict(X), t_big.predict(X))
        if base_type == "lgbm":
            # LightGBM garantiert die Mindestgröße hart
            assert _leaf_sizes(t_big, X).min() >= 800
        else:
            # CatBoost/Lossguide: min_data_in_leaf steuert Splits, garantiert
            # aber keine harte Untergrenze — die Wirkung muss dennoch klar
            # messbar sein (deutlich größere Minimal-Blätter als bei min=10).
            assert _leaf_sizes(t_big, X).min() >= 2 * _leaf_sizes(t_small, X).min()

    def test_single_tree(self, base_type):
        tree = self._build(base_type, num_leaves=8, min_samples_leaf=50)
        rng = np.random.RandomState(2)
        X = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("ABC"))
        tree.fit(X, rng.normal(size=500))
        n_trees = None
        if base_type == "lgbm":
            n_trees = tree.booster_.num_trees()
        else:
            n_trees = tree.tree_count_
        assert n_trees == 1


class TestBundleReuseInvariant:
    def test_bundle_takes_analysis_surrogate(self):
        """Quelltext-Invariante: Der Bundle-Block übernimmt den Analyse-Baum
        (models[SURROGATE_MODEL_NAME]) und baut nur im Fallback neu."""
        src = open("rubin/pipelines/analysis_pipeline.py", encoding="utf-8").read()
        assert re.search(
            r"_analysis_surr = \(models or \{\}\)\.get\(SURROGATE_MODEL_NAME\)", src
        ), "Bundle-Reuse des Analyse-Surrogates entfernt? Konsistenz Report↔Produktion prüfen!"
        assert "Analyse-Baum übernommen" in src
