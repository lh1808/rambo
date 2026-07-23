"""Tests für den Explainability-Plot-Kern (_plot_binned_mean & Helfer).

Abgedeckt: Quantil- vs. Equal-Width-Binning auf schiefen Verteilungen,
kompakte Bereichs-Labels, "fehlend"-Balken, Rückbeschriftung kodierter
Kategorien (value_labels), Sortierung (numerische Codes natürlich, nominale
Strings nach Effektstärke), Scatter-Tick-Relabeling.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rubin.explainability.shap_uplift import (
    _apply_value_labels,
    _fmt_num,
    _plot_binned_mean,
    _relabel_scatter_ticks,
)


def _tick_labels(ax):
    return [t.get_text() for t in ax.get_xticklabels()]


class TestFmtNum:
    def test_compact_and_readable(self):
        assert _fmt_num(0.30000000000000004) == "0.3"
        assert _fmt_num(12345.67) == "12.3k"
        assert _fmt_num(2_500_000) == "2.5M"
        assert _fmt_num(float("nan")) == "?"


class TestApplyValueLabels:
    def test_tolerant_code_matching(self):
        labels = {1: "männlich", "2": "weiblich"}
        # '1', 1, 1.0 treffen denselben Eintrag; unbekannte Codes bleiben roh
        assert _apply_value_labels([1.0, "1", 2, 3], labels) == [
            "männlich", "männlich", "weiblich", "3",
        ]

    def test_no_labels_passthrough(self):
        assert _apply_value_labels(["a", 1], None) == ["a", "1"]


class TestNumericBinning:
    def _skewed(self, n=2000, seed=0):
        rng = np.random.RandomState(seed)
        x = pd.Series(np.exp(rng.normal(0, 1.5, n)) * 100, name="BEITRAG")
        v = rng.normal(0, 1, n)
        return x, v

    def test_quantile_bins_equally_populated(self):
        x, v = self._skewed()
        fig, ax = plt.subplots()
        _plot_binned_mean(ax, x, v, "t", "y", num_bins=10, bin_strategy="quantile")
        # n=… Annotation pro Balken; Besetzung ~gleich (Quantile)
        counts = [int(a.get_text()[2:]) for a in ax.texts]
        assert len(counts) == 10
        assert max(counts) - min(counts) <= 2
        plt.close(fig)

    def test_width_bins_legacy_behavior_skewed(self):
        x, v = self._skewed()
        fig, ax = plt.subplots()
        _plot_binned_mean(ax, x, v, "t", "y", num_bins=10, bin_strategy="width")
        counts = [int(a.get_text()[2:]) for a in ax.texts]
        # Equal-Width auf Lognormal: erster Bin dominiert massiv (das
        # dokumentierte Altverhalten, das der Quantil-Default vermeidet)
        assert max(counts) > 0.8 * sum(counts)
        plt.close(fig)

    def test_range_labels_no_float_noise(self):
        x = pd.Series(np.linspace(0.1, 0.9, 500), name="F")
        fig, ax = plt.subplots()
        _plot_binned_mean(ax, x, np.ones(500), "t", "y", num_bins=4)
        labels = _tick_labels(ax)
        assert all("\u2013" in l for l in labels)          # Bereichs-Format [a–b]
        assert all(len(l) <= 16 for l in labels), labels    # kompakt, kein Float-Rauschen
        plt.close(fig)

    def test_missing_values_get_own_bar(self):
        x = pd.Series([1.0, 2.0, 3.0, 4.0] * 50 + [np.nan] * 20, name="F")
        v = np.r_[np.zeros(200), np.ones(20) * 5.0]
        fig, ax = plt.subplots()
        _plot_binned_mean(ax, x, v, "t", "y", num_bins=4)
        labels = _tick_labels(ax)
        assert labels[-1] == "fehlend"
        # Mittelwert der fehlenden Gruppe (5.0) landet im letzten Balken
        heights = [p.get_height() for p in ax.patches]
        assert heights[-1] == pytest.approx(5.0)
        plt.close(fig)


class TestCategoricalPlotting:
    def test_numeric_codes_natural_order_and_labels(self):
        # SAS-typisch: kodierte Kategorie als category-dtype über Ints
        x = pd.Series(pd.Categorical([2, 1, 3, 1, 2, 1] * 30), name="GESCHLECHT")
        v = np.tile([0.2, -0.1, 0.4, -0.1, 0.2, -0.1], 30)
        fig, ax = plt.subplots()
        _plot_binned_mean(
            ax, x, v, "t", "y", num_bins=10,
            value_labels={1: "männlich", "2": "weiblich", 3: "divers"},
        )
        assert _tick_labels(ax) == ["männlich", "weiblich", "divers"]  # 1,2,3 natürlich sortiert
        plt.close(fig)

    def test_nominal_strings_sorted_by_effect(self):
        x = pd.Series(["B", "A", "C"] * 40, name="TARIF")
        v = np.tile([0.1, 0.5, -0.2], 40)
        fig, ax = plt.subplots()
        _plot_binned_mean(ax, x, v, "t", "y", num_bins=10)
        assert _tick_labels(ax) == ["A", "B", "C"]  # nach Mittelwert absteigend
        plt.close(fig)

    def test_unobserved_categories_dropped_and_nan_labeled(self):
        x = pd.Series(pd.Categorical(["a", "b", "a", None], categories=["a", "b", "zz"]), name="F")
        fig, ax = plt.subplots()
        _plot_binned_mean(ax, x, np.array([1.0, 2.0, 3.0, 9.0]), "t", "y", num_bins=5)
        labels = _tick_labels(ax)
        assert "zz" not in labels          # observed=True
        assert labels[-1] == "fehlend"     # NaN-Gruppe sichtbar statt verschluckt
        plt.close(fig)


class TestScatterRelabel:
    def test_relabels_coded_axis(self):
        fig, ax = plt.subplots()
        ax.scatter([1, 2, 3], [0.1, 0.2, 0.3])
        _relabel_scatter_ticks(ax, pd.Series([1, 2, 3] * 10), {1: "A", 2: "B", 3: "C"})
        assert _tick_labels(ax) == ["A", "B", "C"]
        plt.close(fig)

    def test_noop_without_labels_or_continuous(self):
        fig, ax = plt.subplots()
        ax.scatter(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
        before = _tick_labels(ax)
        _relabel_scatter_ticks(ax, pd.Series(np.linspace(0, 1, 50)), None)
        assert _tick_labels(ax) == before
        plt.close(fig)


class TestScatterKeyspaceComposition:
    def test_code_axis_labels_composed_from_user_labels(self):
        """Regression: Scatter des generischen Pfads liegt auf der Code-Achse
        (0..K-1); User-Labels sind auf Originalwerte geschlüsselt. Die
        Komposition Code→Original→User-Label muss Off-by-One verhindern."""
        from rubin.explainability.shap_uplift import _apply_value_labels
        auto = {0: "1", 1: "2", 2: "3"}                      # Code → Originalwert
        user = {1: "männlich", 2: "weiblich", 3: "divers"}    # Originalwert → Label
        composed = dict(zip(auto.keys(), _apply_value_labels(list(auto.values()), user)))
        assert composed == {0: "männlich", 1: "weiblich", 2: "divers"}


class TestGenericPathEndToEnd:
    def test_coded_categorical_full_chain_scatter_labels(self):
        """E2E durch den generischen SHAP-Pfad: kodierte Kategorie (1/2/3 als
        category-Dtype) + user value_labels → Scatter-Achse muss die korrekt
        komponierten Fachlabels tragen (Regression für den Off-by-One zwischen
        Code-Achse und Originalwert-Schlüsseln). Prüft zugleich, dass der Pfad
        mit Category-Dtypes überhaupt durchläuft (historischer Stillausfall)."""
        rng = np.random.RandomState(0)
        n = 120
        X = pd.DataFrame({
            "GESCHLECHT": pd.Categorical(rng.choice([1, 2, 3], n), categories=[1, 2, 3]),
            "F1": rng.normal(size=n),
        })

        class _Stub:
            def effect(self, x):
                g = pd.to_numeric(pd.Series(np.asarray(x["GESCHLECHT"]) if isinstance(x, pd.DataFrame) else x[:, 0]), errors="coerce").to_numpy()
                f1 = np.asarray(x["F1"], dtype=float) if isinstance(x, pd.DataFrame) else np.asarray(x[:, 1], dtype=float)
                return 0.1 * (g == 2) + 0.05 * f1

        from rubin.explainability.shap_uplift import compute_shap_for_uplift, build_generic_shap_plots
        res = compute_shap_for_uplift(model=_Stub(), X=X, max_background_rows=40, seed=1)
        assert res.auto_value_labels == {"GESCHLECHT": {0: "1", 1: "2", 2: "3"}}
        bundle = build_generic_shap_plots(
            shap_result=res, X=X, cate=_Stub().effect(X), top_n=2, num_bins=4,
            value_labels={"GESCHLECHT": {1: "männlich", 2: "weiblich", 3: "divers"}},
        )
        # Scatter-Figur: Achse des GESCHLECHT-Panels muss die Fachlabels tragen
        labels_per_ax = [[t.get_text() for t in ax.get_xticklabels()] for ax in bundle.shap_scatter.axes]
        assert ["männlich", "weiblich", "divers"] in labels_per_ax, labels_per_ax
        # CATE-Profil: kategorischer Balkenplot ebenfalls rückbeschriftet
        prof_labels = [[t.get_text() for t in ax.get_xticklabels()] for ax in bundle.cate_profiles.axes]
        assert any(l == ["männlich", "weiblich", "divers"] for l in prof_labels), prof_labels
        import matplotlib.pyplot as plt
        plt.close("all")


class TestFillMissingCategories:
    def test_nan_categoricals_get_explicit_category(self):
        """Regression für den CatBoost-Crash 'must be real number, not NoneType':
        NaN in category-/object-Spalten wird zur expliziten Kategorie 'fehlend';
        numerische NaN bleiben unangetastet (nativ von LGBM/CatBoost behandelt)."""
        from rubin.utils.data_utils import fill_missing_categories
        X = pd.DataFrame({
            "TARIF": pd.Series(["A", None, "B", "A"], dtype="object"),
            "KAT": pd.Categorical(["x", "y", None, "x"]),
            "NUM": [1.0, np.nan, 3.0, 4.0],
        })
        converted = fill_missing_categories(X)
        assert set(converted) == {"TARIF", "KAT"}
        assert X["TARIF"].isna().sum() == 0 and (X["TARIF"] == "fehlend").sum() == 1
        assert "fehlend" in X["KAT"].cat.categories and X["KAT"].isna().sum() == 0
        assert np.isnan(X["NUM"]).sum() == 1  # numerisch unberührt
        # Idempotent
        assert fill_missing_categories(X) == []
