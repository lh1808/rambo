"""Tests für Feature-Selektion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rubin.feature_selection import (
    remove_highly_correlated_features,
    select_features_by_importance,
)


class TestRemoveHighlyCorrelatedFeatures:
    """Korrelationsfilter mit importance-gesteuerter Auswahl."""

    def _make_correlated_df(self, n=200):
        rng = np.random.RandomState(42)
        base = rng.randn(n)
        return pd.DataFrame({
            "a": base,
            "b": base + rng.randn(n) * 0.3,   # ~0.96 corr with a (moderat, klar zwischen den getesteten Schwellen 0.9 und 0.999)
            "c": rng.randn(n),                  # uncorrelated
            "d": rng.randn(n),                  # uncorrelated
        })

    def test_removes_correlated_pair(self):
        X = self._make_correlated_df()
        importances = {"m": pd.Series({"a": 10, "b": 5, "c": 3, "d": 1})}
        result, removed, absorbed = remove_highly_correlated_features(
            X, correlation_threshold=0.9, importances=importances
        )
        assert "a" in result.columns  # higher importance survives
        assert "b" not in result.columns
        assert "b" in removed

    def test_absorbed_by_mapping(self):
        X = self._make_correlated_df()
        importances = {"m": pd.Series({"a": 10, "b": 5, "c": 3, "d": 1})}
        _, _, absorbed = remove_highly_correlated_features(
            X, correlation_threshold=0.9, importances=importances
        )
        assert absorbed.get("b") == "a"

    def test_keeps_uncorrelated(self):
        X = self._make_correlated_df()
        importances = {"m": pd.Series({"a": 10, "b": 5, "c": 3, "d": 1})}
        result, removed, _ = remove_highly_correlated_features(
            X, correlation_threshold=0.9, importances=importances
        )
        assert "c" in result.columns
        assert "d" in result.columns

    def test_high_threshold_removes_nothing(self):
        X = self._make_correlated_df()
        importances = {"m": pd.Series({"a": 10, "b": 5, "c": 3, "d": 1})}
        result, removed, absorbed = remove_highly_correlated_features(
            X, correlation_threshold=0.999, importances=importances
        )
        assert len(removed) == 0
        assert len(absorbed) == 0

    def test_single_column(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        result, removed, absorbed = remove_highly_correlated_features(X, correlation_threshold=0.9)
        assert list(result.columns) == ["a"]
        assert removed == []

    def test_without_importances(self):
        """Without importances, one of correlated pair is still removed."""
        X = self._make_correlated_df()
        result, removed, absorbed = remove_highly_correlated_features(
            X, correlation_threshold=0.9
        )
        # One of a/b should be removed
        assert len(removed) == 1
        assert removed[0] in ("a", "b")

    def test_categorical_columns_skipped(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            "num1": rng.randn(100),
            "num2": rng.randn(100),
            "cat1": pd.Categorical(rng.choice(["x", "y"], 100)),
        })
        result, removed, _ = remove_highly_correlated_features(X, correlation_threshold=0.5)
        assert "cat1" in result.columns  # categorical always kept


class TestSelectFeaturesByImportance:
    def test_single_method(self):
        X = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        importances = {"lgbm": pd.Series({"a": 10, "b": 5, "c": 3, "d": 1})}
        # max_features=2 → Budget 2 pro Methode → Top-2: a, b
        result, removed, top = select_features_by_importance(X, importances, max_features=2)
        assert "a" in result.columns
        assert "b" in result.columns
        assert len(removed) == 2

    def test_union_of_two_methods(self):
        X = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        importances = {
            "method1": pd.Series({"a": 10, "b": 1, "c": 1, "d": 1}),
            "method2": pd.Series({"a": 1, "b": 1, "c": 1, "d": 10}),
        }
        # max_features=2, 2 Methoden → Budget = ceil(2/2) = 1 pro Methode
        # method1→a, method2→d → Union={a,d} → max_features=2 → behalte beide
        result, removed, top = select_features_by_importance(X, importances, max_features=2)
        assert "a" in result.columns
        assert "d" in result.columns
        assert len(result.columns) == 2

    def test_max_features_caps_union(self):
        X = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        importances = {
            "m1": pd.Series({"a": 10, "b": 8, "c": 1, "d": 1}),
            "m2": pd.Series({"a": 1, "b": 1, "c": 8, "d": 10}),
        }
        # max_features=2, 2 Methoden → Budget = ceil(2/2) = 1 pro Methode
        # m1→a, m2→d → Union={a,d} → 2 ≤ max_features=2 → OK
        result, removed, top = select_features_by_importance(
            X, importances, max_features=2
        )
        assert len(result.columns) == 2

    def test_empty_importances(self):
        X = pd.DataFrame({"a": [1], "b": [2]})
        result, removed, top = select_features_by_importance(X, {}, max_features=77)
        assert len(result.columns) == 2
        assert removed == []

    def test_preserves_column_order(self):
        X = pd.DataFrame({"z": [1], "a": [2], "m": [3]})
        importances = {"m1": pd.Series({"z": 10, "a": 5, "m": 1})}
        # max_features=2, 1 Methode → Budget=2 → Top-2: z, a
        result, _, _ = select_features_by_importance(X, importances, max_features=2)
        assert list(result.columns) == ["z", "a"]


class TestExactFeatureBudget:
    """Das Ergebnis enthält exakt max_features (sofern genug Features im Pool)."""

    def _X(self, n):
        cols = [f"f{i}" for i in range(n)]
        return pd.DataFrame(np.zeros((4, n)), columns=cols), cols

    def test_overlap_tops_up_to_exact_budget(self):
        # Zwei identische Methoden → starke Überschneidung. Ohne Auffüllen wäre
        # die Union nur ceil(10/2)=5 Features; mit Auffüllen exakt 10.
        X, cols = self._X(20)
        imp = pd.Series({c: 20 - i for i, c in enumerate(cols)})
        result, removed, _ = select_features_by_importance(
            X, {"m1": imp, "m2": imp.copy()}, max_features=10
        )
        assert len(result.columns) == 10
        assert len(removed) == 10
        # Aufgefüllt wird per Konsens-Rang → die global stärksten 10 (f0..f9)
        assert set(result.columns) == {f"f{i}" for i in range(10)}

    def test_no_overlap_exact_budget(self):
        # Drei Methoden mit unterschiedlichem Ranking → Union kann > Budget sein
        # → gekappt auf exakt 10.
        X, cols = self._X(20)
        m1 = pd.Series({c: 20 - i for i, c in enumerate(cols)})
        m2 = pd.Series({c: i for i, c in enumerate(cols)})
        m3 = pd.Series({c: (i * 7) % 20 for i, c in enumerate(cols)})
        result, _, _ = select_features_by_importance(
            X, {"m1": m1, "m2": m2, "m3": m3}, max_features=10
        )
        assert len(result.columns) == 10

    def test_pool_smaller_than_budget_keeps_all(self):
        # Weniger Features als das Budget → alle behalten (mehr gibt es nicht).
        X, cols = self._X(5)
        imp = pd.Series({c: 5 - i for i, c in enumerate(cols)})
        result, removed, _ = select_features_by_importance(X, {"m1": imp}, max_features=10)
        assert len(result.columns) == 5
        assert removed == []

    def test_three_methods_exact_budget(self):
        X, cols = self._X(30)
        rng = np.random.RandomState(0)
        imps = {f"m{k}": pd.Series({c: rng.rand() for c in cols}) for k in range(3)}
        for tgt in (7, 13, 25):
            result, _, _ = select_features_by_importance(X, imps, max_features=tgt)
            assert len(result.columns) == tgt, f"tgt={tgt} → {len(result.columns)}"

    def test_top_per_method_still_reported(self):
        # Pro-Methode-Top-Listen bleiben für den Report erhalten.
        X, cols = self._X(20)
        imp = pd.Series({c: 20 - i for i, c in enumerate(cols)})
        _, _, top = select_features_by_importance(X, {"m1": imp, "m2": imp.copy()}, max_features=10)
        assert set(top.keys()) == {"m1", "m2"}
        assert all(len(v) >= 1 for v in top.values())
