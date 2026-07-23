from __future__ import annotations

"""SHAP-basierte Erklärungen für Uplift/CATE.
Dieses Modul bündelt zwei Ebenen:
1. eine robuste, modellagnostische Berechnung von SHAP-Werten für die
Uplift-Funktion ``f(X) = CATE(X)``;
2. einen vollständigen Plot-Satz (Beeswarm, Mean Impact, Max Impact,
CATE-Profile, SHAP-Dependence, SHAP-Scatter).

Farbpalette: Alle Plots nutzen die SHAP-Standardfarben (positiv=#ff0051,
negativ=#008bfb) statt der rubin-Palette, damit native SHAP-Plots und
Custom-Plots (CATE-Profile, Dependence-Bins) konsistent aussehen."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging as _logging
_logging.getLogger("matplotlib.category").setLevel(_logging.WARNING + 1)
import warnings
warnings.filterwarnings("ignore", message=".*NumPy global RNG was seeded.*", category=FutureWarning)

from rubin.utils.plot_theme import apply_rubin_theme
apply_rubin_theme()


def shap_available() -> bool:
    """Prüft, ob das Paket ``shap`` importierbar ist."""
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class ShapUpliftResult:
    """Ergebnisobjekt für modellagnostische SHAP-Erklärungen."""

    feature_names: list[str]
    shap_values: np.ndarray
    base_values: np.ndarray
    expected_value: float
    explanation: object = None  # shap.Explanation, optional für Plot-Suite
    # Automatisch abgeleitete Rückbeschriftung für intern kodierte kategorische
    # Features: {Spalte: {Code: Original-Label}}. Wird in der Plot-Suite mit
    # user-konfigurierten value_labels gemerged (User gewinnt spaltenweise).
    auto_value_labels: dict = None

    def mean_abs_importance(self) -> pd.Series:
        imp = np.mean(np.abs(self.shap_values), axis=0)
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)


@dataclass
class ShapPlotBundle:
    """Vollständiger SHAP-Plot-Satz für Uplift-Auswertung."""

    summary: plt.Figure
    cate_profiles: plt.Figure
    shap_dependence: plt.Figure
    shap_scatter: plt.Figure
    importance: pd.Series


def _make_uplift_predict_fn(model: object) -> Callable[[pd.DataFrame], np.ndarray]:
    """Erzeugt eine Predict-Funktion für SHAP.
    Bei BT: Rückgabe (n,); bei MT: Rückgabe (n,) als Norm über Arme,
    damit SHAP einen skalaren Output hat."""
    if hasattr(model, "const_marginal_effect"):
        def _pred(x: pd.DataFrame) -> np.ndarray:
            y = np.asarray(model.const_marginal_effect(x))
            if y.ndim == 2 and y.shape[1] == 1:
                return y.reshape(-1)
            if y.ndim == 2 and y.shape[1] > 1:
                # MT: max-Effekt als skalarer Output für SHAP
                return np.max(y, axis=1)
            return y.reshape(-1)
        return _pred

    if hasattr(model, "effect"):
        def _pred(x: pd.DataFrame) -> np.ndarray:
            y = np.asarray(model.effect(x))
            if y.ndim == 2 and y.shape[1] == 1:
                return y.reshape(-1)
            if y.ndim == 2 and y.shape[1] > 1:
                return np.max(y, axis=1)
            return y.reshape(-1)
        return _pred

    raise TypeError(
        "Das übergebene Modell unterstützt weder 'const_marginal_effect' noch 'effect'. "
        "Für Explainability wird eine dieser Methoden benötigt."
    )


def compute_shap_for_uplift(
    model: object,
    X: pd.DataFrame,
    background: Optional[pd.DataFrame] = None,
    max_background_rows: int = 200,
    seed: int = 42,
    feature_names: Optional[Sequence[str]] = None,
) -> ShapUpliftResult:
    """Berechnet modellagnostische SHAP-Werte für die Uplift-Funktion."""
    if not shap_available():
        raise ImportError(
            "SHAP ist nicht installiert. Bitte installiere das Paket 'shap'."
        )

    import shap

    rng = np.random.default_rng(seed)
    feature_names = list(feature_names) if feature_names is not None else list(X.columns)

    # ── Kategorische Features: Encoding für SHAP, Decoding für das Modell ──
    # Der Permutation-Explainer arbeitet auf numerischen Arrays; String-/
    # Category-Spalten scheiterten historisch am Float-Cast ("could not
    # convert string to float") → die gesamte SHAP-Analyse fiel still aus.
    # Lösung: Spalten für SHAP zu Integer-Codes kodieren und im Predict-
    # Wrapper vor jedem Modellaufruf zurück in den originalen Category-Dtype
    # übersetzen — das Modell sieht exakt die Trainings-Dtypes, SHAP sieht
    # Zahlen. Die Code→Label-Mappings werden als auto_value_labels für die
    # Plot-Rückbeschriftung exportiert.
    cat_cols = [
        c for c in X.columns
        if isinstance(X[c].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(X[c])
    ]
    cat_categories: dict = {}
    if cat_cols:
        X_shap = X.copy()
        for c in cat_cols:
            ser = X_shap[c] if isinstance(X_shap[c].dtype, pd.CategoricalDtype) else X_shap[c].astype("category")
            cat_categories[c] = ser.cat.categories
            codes = ser.cat.codes.astype("float64")      # -1 = fehlend
            X_shap[c] = codes.where(codes >= 0, np.nan)  # NaN bleibt sichtbar
    else:
        X_shap = X
    auto_value_labels = {
        c: {int(i): str(cat) for i, cat in enumerate(cats)}
        for c, cats in cat_categories.items()
    }

    def _decode(x_df: pd.DataFrame) -> pd.DataFrame:
        if not cat_cols:
            return x_df
        out = x_df.copy()
        for c, cats in cat_categories.items():
            codes = pd.to_numeric(out[c], errors="coerce").round()
            codes = codes.where(codes.notna(), -1).astype(int).clip(-1, len(cats) - 1)
            out[c] = pd.Categorical.from_codes(codes, categories=cats)
        return out

    if background is None:
        if len(X_shap) <= max_background_rows:
            background = X_shap
        else:
            idx = rng.choice(len(X_shap), size=max_background_rows, replace=False)
            background = X_shap.iloc[idx]

    _raw_predict = _make_uplift_predict_fn(model)

    def predict_fn(x) -> np.ndarray:
        x_df = pd.DataFrame(x, columns=list(X_shap.columns)) if not isinstance(x, pd.DataFrame) else x
        return _raw_predict(_decode(x_df))

    explainer = shap.Explainer(predict_fn, background)
    explanation = explainer(X_shap, silent=True)

    shap_vals = np.asarray(explanation.values)
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(-1, 1)

    base_vals = np.asarray(explanation.base_values)
    if base_vals.ndim == 0:
        base_vals = np.full(shape=(len(X),), fill_value=float(base_vals))

    expected_value = float(np.mean(base_vals))

    return ShapUpliftResult(
        feature_names=feature_names,
        shap_values=shap_vals,
        base_values=base_vals,
        expected_value=expected_value,
        explanation=explanation,
        auto_value_labels=auto_value_labels,
    )


def _extract_primary_explanation(raw_shap_values: object):
    """Extrahiert die primäre SHAP-Explanation aus EconML-Strukturen.
Erwartet typischerweise die Struktur ``shap_values["Y0"]["T0_1"]``.
Falls die Struktur nicht vorliegt, wird versucht, das Objekt direkt als
``shap.Explanation`` zu verwenden."""
    try:
        return raw_shap_values["Y0"]["T0_1"]
    except Exception:
        return raw_shap_values



def _is_categorical(series: pd.Series) -> bool:
    return bool(isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series))


def _fmt_num(v: float) -> str:
    """Kompakte, lesbare Zahlformatierung für Bin-Grenzen (3 signifikante
    Stellen, k/M-Suffixe): 0.30000000000000004 → '0.3', 12345.67 → '12.3k'."""
    if not np.isfinite(v):
        return "?"
    a = abs(v)
    if a >= 1_000_000:
        return f"{v/1_000_000:.3g}M"
    if a >= 10_000:
        return f"{v/1_000:.3g}k"
    return f"{v:.3g}"


def _apply_value_labels(index_values, labels: Optional[dict]) -> list[str]:
    """Rückbeschriftung kodierter Kategorien (z. B. SAS-Format-Codes 1/2/3 →
    Fachlabels). Codes werden tolerant gematcht: '1', 1, 1.0 treffen denselben
    Label-Eintrag. Ohne Treffer bleibt der Originalwert stehen."""
    def _norm(v):
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else str(f)
        except (TypeError, ValueError):
            return str(v)
    if not labels:
        return [str(v) for v in index_values]
    norm_labels = {_norm(k): str(v) for k, v in labels.items()}
    return [norm_labels.get(_norm(v), str(v)) for v in index_values]



def _plot_binned_mean(
    ax: plt.Axes,
    feature_values: pd.Series,
    values: np.ndarray,
    title: str,
    y_label: str,
    num_bins: int,
    bin_strategy: str = "quantile",
    value_labels: Optional[dict] = None,
) -> None:
    """Plottet segmentierte Mittelwerte für numerische oder kategoriale Features.

    Numerisch: Quantil-Bins als Default (gleich besetzte Segmente — bei schiefen
    Verteilungen aussagekräftiger als Equal-Width, wo Ausreißer fast leere
    Rand-Bins mit Rausch-Mittelwerten erzeugen). ``bin_strategy="width"`` stellt
    das alte Equal-Width-Verhalten bereit. Bin-Beschriftung als kompakte
    Wertebereiche ("[1.2k–3.4k)") statt Float-Bin-Zentren; Besetzung (n=…) wird
    je Balken annotiert, damit dünn besetzte Segmente erkennbar sind.
    Fehlende Werte erhalten einen eigenen "fehlend"-Balken statt still zu
    verschwinden. Kategorisch: nur beobachtete Kategorien (observed=True);
    numerische Codes und geordnete Kategorien in natürlicher Reihenfolge,
    nominale Strings nach Effektstärke sortiert. ``value_labels`` beschriftet
    kodierte Kategorien zurück (Code → Fachlabel).
    Farben: SHAP-Palette (positiv=#ff0051, negativ=#008bfb).
    """
    SHAP_POS = "#ff0051"
    SHAP_NEG = "#008bfb"
    SHAP_POS_EDGE = "#d4003f"
    SHAP_NEG_EDGE = "#0070d0"
    MISSING_LABEL = "fehlend"

    df = pd.DataFrame({"feature_value": feature_values, "value": np.asarray(values).reshape(-1)})

    def _bar_colors(vals):
        return (
            [SHAP_POS if v >= 0 else SHAP_NEG for v in vals],
            [SHAP_POS_EDGE if v >= 0 else SHAP_NEG_EDGE for v in vals],
        )

    def _draw(labels, means, counts):
        colors, edges = _bar_colors(means)
        ax.bar(range(len(labels)), means, color=colors, edgecolor=edges, linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        # Besetzung dezent annotieren: dünne Segmente sind sonst nicht von
        # gut besetzten unterscheidbar und laden zur Überinterpretation ein.
        for x_pos, (m, c) in enumerate(zip(means, counts)):
            ax.annotate(
                f"n={int(c)}", (x_pos, m), textcoords="offset points",
                xytext=(0, 3 if m >= 0 else -10), ha="center",
                fontsize=7, color="#57606a",
            )

    if _is_categorical(df["feature_value"]):
        agg = (
            df.groupby("feature_value", dropna=False, observed=True)["value"]
            .agg(["mean", "count"])
        )
        # CategoricalIndex → object: erlaubt das "fehlend"-Label und
        # positionsfreies .loc[order] ohne Kategorien-Restriktionen.
        agg.index = agg.index.astype(object)
        # NaN-Gruppe ans Ende, mit sprechendem Label
        has_nan = any(pd.isna(idx) for idx in agg.index)
        agg = agg.rename(index={idx: MISSING_LABEL for idx in agg.index if pd.isna(idx)})
        # Sortierung: natürliche Ordnung für numerische Codes und geordnete
        # Kategorien (Alter_Band etc.); nominale Strings nach Effektstärke.
        idx_no_nan = [i for i in agg.index if i != MISSING_LABEL]
        _dtype = feature_values.dtype
        _ordered_cat = isinstance(_dtype, pd.CategoricalDtype) and _dtype.ordered
        _numeric_codes = all(_is_number_like(i) for i in idx_no_nan) and len(idx_no_nan) > 0
        if _ordered_cat and isinstance(_dtype, pd.CategoricalDtype):
            order = [c for c in _dtype.categories if c in set(idx_no_nan)]
        elif _numeric_codes:
            order = sorted(idx_no_nan, key=lambda v: float(v))
        else:
            order = list(agg.loc[idx_no_nan].sort_values("mean", ascending=False).index)
        if has_nan:
            order = order + [MISSING_LABEL]
        agg = agg.loc[order]
        labels = _apply_value_labels(agg.index, value_labels)
        _draw(labels, agg["mean"].to_numpy(), agg["count"].to_numpy())
    else:
        series = pd.to_numeric(df["feature_value"], errors="coerce")
        n_missing = int(series.isna().sum())
        valid = df.loc[series.notna()].copy()
        valid["feature_value"] = series.loc[series.notna()].to_numpy()
        if len(valid) <= 1:
            ax.text(0.5, 0.5, "Zu wenige gültige Werte", ha="center", va="center", color="#57606a")
            ax.set_title(title)
            ax.set_ylabel(y_label)
            return
        n_bins = max(2, min(int(num_bins), int(valid["feature_value"].nunique())))
        if bin_strategy == "width":
            bins = np.linspace(valid["feature_value"].min(), valid["feature_value"].max(), num=n_bins + 1)
            valid["bin"] = pd.cut(valid["feature_value"], bins=bins, include_lowest=True, duplicates="drop")
        else:
            # Quantil-Bins: gleich besetzte Segmente (duplicates="drop" fängt
            # Punktmassen ab, z. B. viele Nullen — dann entstehen weniger Bins).
            valid["bin"] = pd.qcut(valid["feature_value"], q=n_bins, duplicates="drop")
        agg = valid.groupby("bin", observed=True)["value"].agg(["mean", "count"]).reset_index()
        labels = [
            f"[{_fmt_num(iv.left)}\u2013{_fmt_num(iv.right)}]" for iv in agg["bin"]
        ]
        means = agg["mean"].to_list()
        counts = agg["count"].to_list()
        if n_missing > 0:
            labels.append(MISSING_LABEL)
            means.append(float(df.loc[series.isna(), "value"].mean()))
            counts.append(n_missing)
        _draw(labels, np.asarray(means), np.asarray(counts))
    ax.set_title(title)
    ax.set_xlabel(str(feature_values.name))
    ax.set_ylabel(y_label)


def _is_number_like(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False

def _relabel_scatter_ticks(ax: plt.Axes, feature_values: pd.Series, labels: Optional[dict]) -> None:
    """Beschriftet die x-Achse eines SHAP-Scatters mit Fachlabels zurück,
    wenn das Feature kodierte Kategorien trägt (<= 25 Ausprägungen). Ohne
    Labels oder bei echt kontinuierlichen Features bleibt die Achse unverändert."""
    if not labels:
        return
    try:
        uniq = pd.Series(feature_values).dropna().unique()
        if len(uniq) > 25:
            return
        codes = sorted(uniq, key=lambda v: float(v)) if all(_is_number_like(u) for u in uniq) else sorted(map(str, uniq))
        ax.set_xticks([float(c) if _is_number_like(c) else c for c in codes])
        ax.set_xticklabels(_apply_value_labels(codes, labels), rotation=45, ha="right")
    except Exception:
        pass  # Anzeige-Komfort darf die Plot-Erzeugung nie brechen


def build_shap_plots(
    model: object,
    X: pd.DataFrame,
    data: pd.DataFrame,
    cate: Sequence[float],
    top_n: int,
    num_bins: int = 10,
    bin_strategy: str = "quantile",
    value_labels: Optional[dict] = None,
) -> ShapPlotBundle:
    """Erzeugt den vollständigen SHAP-Plot-Satz.
Voraussetzungen
---------------
* Das Modell stellt ``shap_values(X=...)`` bereit.
* Das Ergebnis enthält eine nutzbare ``shap.Explanation``.
Die Funktion erzeugt folgende Plots:
* Beeswarm
* Mean Impact (Bar Summary)
* Max Impact (Beeswarm nach maximaler absoluter Wirkung)
* CATE-Profil je Top-Feature (durchschnittlicher CATE pro Wertebereich)
* SHAP-Dependence je Top-Feature (gebinnter mittlerer SHAP-Wert)
* SHAP-Scatter je Top-Feature"""
    if not shap_available():
        raise ImportError("SHAP ist nicht installiert.")
    if not hasattr(model, "shap_values"):
        raise TypeError(
            "Das Modell stellt keine Methode 'shap_values' bereit. "
            "Für diesen Plot-Satz wird ein EconML-kompatibles Modell benötigt."
        )

    import shap

    raw_shap_values = model.shap_values(X=X)
    explanation = _extract_primary_explanation(raw_shap_values)

    if not hasattr(explanation, "values"):
        raise TypeError(
            "Das Ergebnis von 'model.shap_values' hat nicht die erwartete Struktur. "
            "Die SHAP-Plots können dafür nicht erzeugt werden."
        )

    mean_abs = explanation.abs.mean(axis=0)
    mean_abs_values = np.asarray(getattr(mean_abs, "values", mean_abs)).reshape(-1)
    feature_names = list(X.columns)
    sorted_idx = np.argsort(mean_abs_values)[::-1]
    top_idx = sorted_idx[: min(top_n, len(feature_names))]
    importance = pd.Series(mean_abs_values, index=feature_names).sort_values(ascending=False)

    fig1 = plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1)
    shap.plots.beeswarm(
        explanation,
        max_display=max(21, min(len(feature_names), top_n + 1)),
        color_bar=False,
        plot_size=None,
        show=False,
    )
    plt.title("Beeswarm")

    plt.subplot(3, 2, 3)
    shap.summary_plot(
        explanation,
        plot_type="bar",
        max_display=min(20, len(feature_names)),
        color_bar=False,
        plot_size=None,
        show=False,
    )
    plt.title("Mean Impact")

    plt.subplot(3, 2, 4)
    shap.plots.beeswarm(
        explanation.abs,
        max_display=max(21, min(len(feature_names), top_n + 1)),
        order=shap.Explanation.abs.max(0),
        color_bar=False,
        plot_size=None,
        show=False,
    )
    plt.title("Max Impact")
    plt.tight_layout()

    cate_values = np.asarray(cate).reshape(-1)
    fig2 = plt.figure(figsize=(15, max(5, 4 * len(top_idx))))
    for i, feature_idx in enumerate(top_idx, start=1):
        feature_name = feature_names[int(feature_idx)]
        ax = fig2.add_subplot(len(top_idx), 1, i)
        feature_values = data[feature_name] if feature_name in data.columns else X[feature_name]
        _plot_binned_mean(
            ax=ax,
            feature_values=feature_values,
            values=cate_values,
            title=f"CATE-Profil für {feature_name}",
            y_label="Durchschnittlicher CATE",
            num_bins=num_bins,
            bin_strategy=bin_strategy,
            value_labels=(value_labels or {}).get(feature_name),
        )
    fig2.tight_layout()

    fig3 = plt.figure(figsize=(15, max(5, 4 * len(top_idx))))
    for i, feature_idx in enumerate(top_idx, start=1):
        feature_name = feature_names[int(feature_idx)]
        ax = fig3.add_subplot(len(top_idx), 1, i)
        shap_values_for_feature = np.asarray(explanation[:, int(feature_idx)].values).reshape(-1)
        _plot_binned_mean(
            ax=ax,
            feature_values=X[feature_name],
            values=shap_values_for_feature,
            title=f"SHAP-Dependence für {feature_name}",
            y_label="Durchschnittlicher SHAP-Wert",
            num_bins=num_bins,
            bin_strategy=bin_strategy,
            value_labels=(value_labels or {}).get(feature_name),
        )
    fig3.tight_layout()

    fig4, axes = plt.subplots(len(top_idx), 1, figsize=(15, max(5, 4 * len(top_idx))))
    if len(top_idx) == 1:
        axes = [axes]
    for ax, feature_idx in zip(axes, top_idx):
        feature_name = feature_names[int(feature_idx)]
        shap.plots.scatter(explanation[:, feature_name], ax=ax, show=False)
        ax.set_title(f"SHAP-Scatter für {feature_name}")
        _relabel_scatter_ticks(ax, X[feature_name], (value_labels or {}).get(feature_name))
    fig4.tight_layout()

    return ShapPlotBundle(
        summary=fig1,
        cate_profiles=fig2,
        shap_dependence=fig3,
        shap_scatter=fig4,
        importance=importance,
    )


def build_generic_shap_plots(
    shap_result: ShapUpliftResult,
    X: pd.DataFrame,
    cate: Sequence[float],
    top_n: int,
    num_bins: int = 10,
    bin_strategy: str = "quantile",
    value_labels: Optional[dict] = None,
) -> ShapPlotBundle:
    """Erzeugt den vollständigen SHAP-Plot-Satz aus generischen SHAP-Werten.

    Funktioniert mit jedem Modell, das ``effect()`` oder ``const_marginal_effect()``
    bereitstellt — auch CausalForestDML. Nutzt die von ``compute_shap_for_uplift``
    berechneten generischen SHAP-Werte statt EconMLs ``shap_values()``-Methode.
    """
    if not shap_available():
        raise ImportError("SHAP ist nicht installiert.")

    import shap

    explanation = shap_result.explanation
    if explanation is None or not hasattr(explanation, "values"):
        raise TypeError(
            "ShapUpliftResult enthält kein gültiges Explanation-Objekt. "
            "Wurde compute_shap_for_uplift mit einer aktuellen Version aufgerufen?"
        )

    # Rückbeschriftung — zwei Keyspaces sauber trennen:
    # Die BINNED-Plots (CATE-Profil/Dependence) arbeiten auf dem originalen X
    # (Originalwerte als Schlüssel) → dort gelten die user-value_labels direkt.
    # Der SCATTER des generischen Pfads liegt dagegen auf der KODIERTEN Achse
    # (Explanation-Daten = Integer-Codes 0..K-1). Ein naives Mergen von
    # Auto-Labels (Code-Schlüssel) und User-Labels (Originalwert-Schlüssel)
    # erzeugt Off-by-One-Beschriftungen. Daher Komposition pro Spalte:
    # Code → Originalwert → User-Label (falls konfiguriert), sonst Originalwert.
    _auto = getattr(shap_result, "auto_value_labels", None) or {}
    _user = value_labels or {}
    scatter_value_labels: dict = dict(_user)
    for _col, _code_map in _auto.items():
        _origs = list(_code_map.values())
        _lbls = _apply_value_labels(_origs, _user.get(_col))
        scatter_value_labels[_col] = {code: lbl for code, lbl in zip(_code_map.keys(), _lbls)}
    value_labels = _user
    mean_abs_values = np.mean(np.abs(np.asarray(explanation.values)), axis=0).reshape(-1)
    feature_names = list(X.columns)
    sorted_idx = np.argsort(mean_abs_values)[::-1]
    top_idx = sorted_idx[: min(top_n, len(feature_names))]
    importance = pd.Series(mean_abs_values, index=feature_names).sort_values(ascending=False)

    # fig1: Summary (Beeswarm + Mean Impact + Max Impact)
    fig1 = plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1)
    shap.plots.beeswarm(
        explanation,
        max_display=max(21, min(len(feature_names), top_n + 1)),
        color_bar=False,
        plot_size=None,
        show=False,
    )
    plt.title("Beeswarm")

    plt.subplot(3, 2, 3)
    shap.summary_plot(
        explanation,
        plot_type="bar",
        max_display=min(20, len(feature_names)),
        color_bar=False,
        plot_size=None,
        show=False,
    )
    plt.title("Mean Impact")

    plt.subplot(3, 2, 4)
    shap.plots.beeswarm(
        explanation.abs,
        max_display=max(21, min(len(feature_names), top_n + 1)),
        order=shap.Explanation.abs.max(0),
        color_bar=False,
        plot_size=None,
        show=False,
    )
    plt.title("Max Impact")
    plt.tight_layout()

    # fig2: CATE-Profile (durchschnittlicher CATE pro Feature-Wertebereich)
    cate_values = np.asarray(cate).reshape(-1)
    fig2 = plt.figure(figsize=(15, max(5, 4 * len(top_idx))))
    for i, feature_idx in enumerate(top_idx, start=1):
        feature_name = feature_names[int(feature_idx)]
        ax = fig2.add_subplot(len(top_idx), 1, i)
        _plot_binned_mean(
            ax=ax,
            feature_values=X[feature_name],
            values=cate_values,
            title=f"CATE-Profil für {feature_name}",
            y_label="Durchschnittlicher CATE",
            num_bins=num_bins,
            bin_strategy=bin_strategy,
            value_labels=(value_labels or {}).get(feature_name),
        )
    fig2.tight_layout()

    # fig3: SHAP-Dependence (gebinnter mittlerer SHAP-Wert)
    fig3 = plt.figure(figsize=(15, max(5, 4 * len(top_idx))))
    for i, feature_idx in enumerate(top_idx, start=1):
        feature_name = feature_names[int(feature_idx)]
        ax = fig3.add_subplot(len(top_idx), 1, i)
        shap_values_for_feature = np.asarray(explanation[:, int(feature_idx)].values).reshape(-1)
        _plot_binned_mean(
            ax=ax,
            feature_values=X[feature_name],
            values=shap_values_for_feature,
            title=f"SHAP-Dependence für {feature_name}",
            y_label="Durchschnittlicher SHAP-Wert",
            num_bins=num_bins,
            bin_strategy=bin_strategy,
            value_labels=(value_labels or {}).get(feature_name),
        )
    fig3.tight_layout()

    # fig4: SHAP-Scatter (SHAP vs. Feature-Wert pro Beobachtung)
    fig4, axes = plt.subplots(len(top_idx), 1, figsize=(15, max(5, 4 * len(top_idx))))
    if len(top_idx) == 1:
        axes = [axes]
    for ax, feature_idx in zip(axes, top_idx):
        feature_name = feature_names[int(feature_idx)]
        shap.plots.scatter(explanation[:, feature_name], ax=ax, show=False)
        ax.set_title(f"SHAP-Scatter für {feature_name}")
        # Achse des generischen Scatters = KODIERTE Explanation-Daten (0..K-1),
        # nicht die Originalwerte aus X — Tick-Positionen daher aus der
        # Explanation ableiten, Labels aus der Code-Komposition.
        _enc_vals = pd.Series(np.asarray(explanation[:, int(feature_idx)].data).reshape(-1), name=feature_name)
        _relabel_scatter_ticks(ax, _enc_vals, scatter_value_labels.get(feature_name))
    fig4.tight_layout()

    return ShapPlotBundle(
        summary=fig1,
        cate_profiles=fig2,
        shap_dependence=fig3,
        shap_scatter=fig4,
        importance=importance,
    )
