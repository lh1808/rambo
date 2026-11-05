"""Tests für Feature-Selektion."""

from __future__ import annotations

import numpy as np
import pandas as pd

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


class TestCorrelationChainNoCascade:
    """Realer Vorfall: Ein wichtiges Feature W korrelierte stark mit F — und
    BEIDE wurden entfernt. Ursache: Ketten-Kaskade V~W~F (V!~F). Die alte
    Paar-Schleife entfernte F wegen W (absorbed_by[F]=W) und danach W wegen V —
    F's Keeper war tot, F's Information komplett verloren, obwohl F mit dem
    Überlebenden V gar nicht über-korreliert. Der Greedy-Fix (absteigende
    Wichtigkeit, Keeper entfernen nur LEBENDE Partner) garantiert: Jeder
    Entfernte korreliert mit einem BEHALTENEN Feature."""

    def _chain(self, n=20000, seed=0):
        rng = np.random.default_rng(seed)
        e1, e2, e3 = rng.normal(size=(3, n))
        V = e1
        W = 0.93 * e1 + np.sqrt(1 - 0.93**2) * e2       # corr(V,W)≈0.93
        F = 0.92 * W + np.sqrt(1 - 0.92**2) * e3        # corr(W,F)≈0.92, corr(V,F)≈0.86
        return V, W, F

    def test_chain_importance_keeps_tail_feature(self):
        V, W, F = self._chain()
        X = pd.DataFrame({"F_FEAT": F, "W_WICHTIG": W, "V_STARK": V})
        imp = {"m": pd.Series({"V_STARK": 100.0, "W_WICHTIG": 90.0, "F_FEAT": 50.0})}
        Xf, removed, absorbed = remove_highly_correlated_features(
            X, correlation_threshold=0.9, correlation_methods=["pearson"], importances=imp)
        assert sorted(Xf.columns) == ["F_FEAT", "V_STARK"]
        assert removed == ["W_WICHTIG"]
        assert absorbed == {"W_WICHTIG": "V_STARK"}

    def test_chain_fallback_keeps_tail_feature(self):
        V, W, F = self._chain(seed=1)
        X = pd.DataFrame({"A": V, "B": W, "C": F})
        Xf, removed, absorbed = remove_highly_correlated_features(
            X, correlation_threshold=0.9, correlation_methods=["pearson"])
        assert sorted(Xf.columns) == ["A", "C"]
        assert removed == ["B"] and absorbed == {"B": "A"}

    def test_clique_collapses_to_most_important(self):
        """Echte Clique (alle Paare über der Schwelle): Dann ist 'beide weg'
        korrekt — beide korrelieren mit dem überlebenden Wichtigsten."""
        rng = np.random.default_rng(2)
        n = 20000
        e1, e2, e3 = rng.normal(size=(3, n))
        V = e1
        W = 0.95 * e1 + np.sqrt(1 - 0.95**2) * e2
        F = 0.97 * W + np.sqrt(1 - 0.97**2) * e3        # corr(V,F)≈0.92 → Clique
        X = pd.DataFrame({"F_FEAT": F, "W_WICHTIG": W, "V_STARK": V})
        imp = {"m": pd.Series({"V_STARK": 100.0, "W_WICHTIG": 90.0, "F_FEAT": 50.0})}
        Xf, removed, absorbed = remove_highly_correlated_features(
            X, correlation_threshold=0.9, correlation_methods=["pearson"], importances=imp)
        assert list(Xf.columns) == ["V_STARK"]
        assert absorbed == {"F_FEAT": "V_STARK", "W_WICHTIG": "V_STARK"}

    def test_invariant_every_removed_correlates_with_a_survivor(self):
        """Strukturelle Garantie auf zufälligen Korrelationsgeflechten:
        (a) jeder absorbed_by-Keeper ÜBERLEBT, (b) jedes entfernte Feature
        korreliert über der Schwelle mit seinem Keeper, (c) kein überlebendes
        Paar liegt über der Schwelle (bezogen auf die Pearson-Runde)."""
        for seed in range(5):
            rng = np.random.default_rng(seed)
            n, k = 4000, 12
            base = rng.normal(size=(n, 4))
            cols = {}
            for i in range(k):
                w = rng.dirichlet(np.ones(4)) * rng.choice([1, 1, 1, 3])
                sig = base @ w
                noise = rng.uniform(0.05, 1.0)
                cols[f"f{i}"] = sig / sig.std() + noise * rng.normal(size=n)
            X = pd.DataFrame(cols)
            imp = {"m": pd.Series(rng.uniform(1, 100, k), index=X.columns)}
            thr = 0.9
            Xf, removed, absorbed = remove_highly_correlated_features(
                X, correlation_threshold=thr, correlation_methods=["pearson"], importances=imp)
            corr = X.corr(method="pearson").abs()
            surv = set(Xf.columns)
            assert set(removed) == set(absorbed)
            for drop, keep in absorbed.items():
                assert keep in surv, f"seed {seed}: Keeper {keep} wurde selbst entfernt"
                assert corr.loc[drop, keep] > thr, f"seed {seed}: {drop}~{keep} nicht über Schwelle"
            for a in surv:
                for b in surv:
                    if a < b:
                        assert corr.loc[a, b] <= thr, f"seed {seed}: Überlebende {a}~{b} über Schwelle"

    def test_redistribution_lands_on_survivor(self):
        """Nach dem Fix zeigen alle absorbed_by auf Überlebende — die
        Importance-Umverteilung (analysis_pipeline) kann nicht mehr auf ein
        totes Feature schieben."""
        V, W, F = self._chain(seed=3)
        X = pd.DataFrame({"F_FEAT": F, "W_WICHTIG": W, "V_STARK": V})
        imp = {"m": pd.Series({"V_STARK": 100.0, "W_WICHTIG": 90.0, "F_FEAT": 50.0})}
        Xf, removed, absorbed = remove_highly_correlated_features(
            X, correlation_threshold=0.9, correlation_methods=["pearson"], importances=imp)
        # Umverteilung wie in _run_feature_selection
        for method_name, s in imp.items():
            for dropped, keeper in absorbed.items():
                if dropped in s.index and keeper in s.index:
                    s[keeper] = s[keeper] + s[dropped]
        assert imp["m"]["V_STARK"] == 190.0  # V erbt W
        assert imp["m"]["F_FEAT"] == 50.0    # F bleibt eigenständig erhalten

    def test_no_cascade_across_methods(self):
        """Kaskade über Methoden-Runden: V~W nur in SPEARMAN über der Schwelle
        (monoton, nichtlinear), W~F in Pearson. Früher: Pearson-Runde droppt F
        (Keeper W), Spearman-Runde droppt W (Keeper V) → F und W beide weg.
        Jetzt: vereinte Adjazenz → ein Greedy → F überlebt."""
        rng = np.random.default_rng(4)
        n = 20000
        e2, e3 = rng.normal(size=(2, n))
        V = rng.normal(size=n)
        # W monoton in V (Spearman ≈ 1), aber durch starke Konvexität Pearson < 0.9
        W_mono = np.exp(2.2 * V)
        W = (W_mono - W_mono.mean()) / W_mono.std()
        F = 0.95 * W + np.sqrt(1 - 0.95**2) * e3
        X = pd.DataFrame({"F_FEAT": F, "W_WICHTIG": W, "V_STARK": V})
        pear = X.corr("pearson").abs(); spear = X.corr("spearman").abs()
        assert pear.loc["V_STARK", "W_WICHTIG"] < 0.9 < spear.loc["V_STARK", "W_WICHTIG"]
        assert pear.loc["W_WICHTIG", "F_FEAT"] > 0.9
        assert max(pear.loc["V_STARK", "F_FEAT"], spear.loc["V_STARK", "F_FEAT"]) < 0.9
        imp = {"m": pd.Series({"V_STARK": 100.0, "W_WICHTIG": 90.0, "F_FEAT": 50.0})}
        Xf, removed, absorbed = remove_highly_correlated_features(
            X, correlation_threshold=0.9, importances=imp)
        assert sorted(Xf.columns) == ["F_FEAT", "V_STARK"], (removed, absorbed)
        assert absorbed == {"W_WICHTIG": "V_STARK"}


class TestScaleFreeKeeperChoice:
    """Keeper-Wahl über Methoden hinweg: je Methode auf Summe 1 normiert.
    Vorher entschied die größt-skalierte Methode (LGBM-Gain ~10^3-10^5) allein —
    ein knappes 52:48 von LGBM überstimmte ein klares 90:10 von CatBoost."""

    def test_small_scale_method_vote_counts(self):
        rng = np.random.default_rng(0)
        n = 20000
        e1, e2 = rng.normal(size=(2, n))
        X = pd.DataFrame({"A": e1, "B": 0.95 * e1 + np.sqrt(1 - 0.95**2) * e2})
        imp = {"lgbm_importance": pd.Series({"A": 4800.0, "B": 5200.0}),
               "catboost_importance": pd.Series({"A": 90.0, "B": 10.0})}
        _, removed, absorbed = remove_highly_correlated_features(
            X, correlation_threshold=0.9, correlation_methods=["pearson"], importances=imp)
        assert removed == ["B"] and absorbed == {"B": "A"}

    def test_single_method_order_unchanged_by_normalization(self):
        rng = np.random.default_rng(1)
        n = 20000
        e1, e2 = rng.normal(size=(2, n))
        X = pd.DataFrame({"A": e1, "B": 0.95 * e1 + np.sqrt(1 - 0.95**2) * e2})
        imp = {"m": pd.Series({"A": 10.0, "B": 5.0})}
        _, removed, _ = remove_highly_correlated_features(
            X, correlation_threshold=0.9, correlation_methods=["pearson"], importances=imp)
        assert removed == ["B"]


class TestFeatureReductionPipelineIntegration:
    """Durchstich der gesamten Reduction-Strecke über _run_feature_selection:
    Importances → Korrelationsfilter (Kette!) → Umverteilung → Budget →
    Buchhaltung (removed-Listen getrennt nach Grund)."""

    def _pipe(self):
        import logging as _logging
        from rubin.pipelines.analysis_pipeline import AnalysisPipeline
        pipe = AnalysisPipeline.__new__(AnalysisPipeline)
        pipe._logger = _logging.getLogger("test_fs_integration")
        return pipe

    def _cfg(self, max_features, threshold=0.9, methods=("lgbm_importance",)):
        from types import SimpleNamespace
        return SimpleNamespace(
            feature_selection=SimpleNamespace(
                enabled=True, methods=list(methods),
                correlation_threshold=threshold, max_features=max_features),
            constants=SimpleNamespace(random_seed=42, parallel_level=1),
        )

    def _data(self, n=4000, seed=5):
        """V~W~F-Kette (V!~F) + Signalfeature S + Rauschfeatures.
        Y hängt von V und F ab → F muss es durch die gesamte Strecke schaffen."""
        rng = np.random.default_rng(seed)
        e = rng.normal(size=(5, n))
        V = e[0]
        W = 0.93 * V + np.sqrt(1 - 0.93**2) * e[1]
        F = 0.92 * W + np.sqrt(1 - 0.92**2) * e[2]
        X = pd.DataFrame({
            "F_FEAT": F, "W_MITTE": W, "V_STARK": V,
            "S_SIGNAL": e[3], "N1": rng.normal(size=n), "N2": rng.normal(size=n),
        })
        T = rng.integers(0, 2, n)
        Y = 2.0 * V + 1.5 * F + 1.0 * e[3] + 0.3 * rng.normal(size=n)
        return X, T, Y

    def test_chain_survivor_passes_budget(self):
        """F überlebt Kette UND Budget; W steht als Korrelations-Entfernung in
        der Buchhaltung, nicht als Importance-Entfernung."""
        from unittest.mock import MagicMock
        X, T, Y = self._data()
        pipe = self._pipe()
        X_out, removed = pipe._run_feature_selection(self._cfg(max_features=4), X, T, Y, MagicMock())
        assert "W_MITTE" in removed["high_correlation"]
        assert "F_FEAT" not in removed["high_correlation"], "Kaskade: F fälschlich wegen Korrelation entfernt"
        assert "F_FEAT" in X_out.columns, "F muss Budget überleben (trägt Y-Signal)"
        assert "V_STARK" in X_out.columns
        # Getrennte Buchhaltung: keine Doppelnennung
        assert not (set(removed["high_correlation"]) & set(removed.get("importance", [])))

    def test_redistribution_strengthens_survivor_in_budget(self):
        """Verzahnung Umverteilung→Budget: V erbt W's Importance und muss im
        engen Budget vor den Rauschfeatures liegen."""
        from unittest.mock import MagicMock
        X, T, Y = self._data(seed=6)
        pipe = self._pipe()
        X_out, removed = pipe._run_feature_selection(self._cfg(max_features=3), X, T, Y, MagicMock())
        assert "V_STARK" in X_out.columns
        assert set(X_out.columns).issubset({"V_STARK", "F_FEAT", "S_SIGNAL"}), list(X_out.columns)

    def test_constant_and_bool_columns_do_not_break_filter(self):
        rng = np.random.default_rng(7)
        n = 1000
        X = pd.DataFrame({"a": rng.normal(size=n), "konst": np.ones(n),
                          "flag": rng.integers(0, 2, n).astype(bool)})
        imp = {"m": pd.Series({"a": 5.0, "konst": 1.0, "flag": 2.0})}
        Xf, removed, _ = remove_highly_correlated_features(
            X, correlation_threshold=0.9, importances=imp)
        # Konstante Spalte: corr=NaN → kein Paar → bleibt (Budget-Sache)
        assert removed == [] and set(Xf.columns) == {"a", "konst", "flag"}

    def test_empty_importances_fall_back_cleanly(self):
        """Alle Methoden fehlgeschlagen → Korrfilter läuft im Fallback,
        Budget-Schritt entfällt, keine Exception."""
        rng = np.random.default_rng(8)
        n = 2000
        e1, e2 = rng.normal(size=(2, n))
        X = pd.DataFrame({"A": e1, "B": 0.95 * e1 + np.sqrt(1 - 0.95**2) * e2,
                          "C": rng.normal(size=n)})
        Xf, removed, absorbed = remove_highly_correlated_features(
            X, correlation_threshold=0.9, importances=None)
        assert removed == ["B"] and "A" in Xf.columns and "C" in Xf.columns
