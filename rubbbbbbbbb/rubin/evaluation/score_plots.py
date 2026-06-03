from __future__ import annotations

"""Score-Vergleichs-Plots: Qini-Kurven, Policy-Value-Vergleiche, Score-Redistribution.

Visualisiert den Vergleich zwischen Modell-Score und historischem Score,
Custom-Qini-Kurven und die Score-Redistribution-Analyse.
"""

from typing import Any, Optional, Tuple
import logging

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rubin.utils.plot_theme import (
    apply_rubin_theme, RUBIN_COLORS, RUBIN_PALETTE,
    COLOR_MODEL, COLOR_MODEL_FILL, COLOR_REFERENCE, COLOR_REFERENCE_FILL,
    COLOR_BASELINE, COLOR_DIFFERENCE, COLOR_HIGHLIGHT_BOX,
    recolor_figure,
)
apply_rubin_theme()

from rubin.evaluation.uplift_metrics import uplift_curve

def compute_qini_curve(outcomes: np.ndarray, score: np.ndarray, treatment: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Berechnet Qini-Kurvenpunkte nativ.

    Q(k) = Y_t(k) − Y_c(k) · N_t(k) / N_c(k)
    Gibt (y_model, x, y_random) als interpolierte Arrays zurück."""
    from rubin.evaluation.uplift_metrics import uplift_curve as _uc

    curve = _uc(y=outcomes, t=treatment, score=score)
    n = len(outcomes)

    # Qini-Formel
    n_c_safe = np.maximum(curve.n_control, 1)
    y_qini = curve.y_treat - curve.y_control * curve.n_treat / n_c_safe

    # Auf ganzzahliges x-Raster interpolieren
    x = np.arange(n)
    x_frac = curve.fraction * n  # fraction → absolute Indizes
    y_score_i = np.interp(x, x_frac, y_qini)
    qini_total = y_qini[-1] if len(y_qini) > 0 else 0
    y_rand_i = x * qini_total / max(n, 1)
    return y_score_i, x, y_rand_i


def plot_custom_qini_curve(
    *,
    data: pd.DataFrame,
    causal_score_label: str,
    affinity_score_label: Optional[str] = None,
    ax: Optional[Any] = None,
    relative_axes: bool = True,
):
    """Custom-Qini-Plot zur direkten Gegenüberstellung zweier Scores.

    Stellt den kausalen Score gegen einen optionalen Referenzscore (z. B.
    historischer Affinity-Score) auf derselben Qini-Kurve dar. Ermöglicht
    einen schnellen visuellen Vergleich der Sortierqualität."""

    y_causal, x, y_random = compute_qini_curve(data["Y"].to_numpy(), data[causal_score_label].to_numpy(), data["T"].to_numpy())
    y_affinity = None
    if affinity_score_label is not None:
        y_affinity, _, _ = compute_qini_curve(data["Y"].to_numpy(), data[affinity_score_label].to_numpy(), data["T"].to_numpy())

    if relative_axes:
        x = x / np.max(x)
        inc_total = np.max(y_random)
        y_causal = y_causal / inc_total
        y_random = y_random / inc_total
        if y_affinity is not None:
            y_affinity = y_affinity / inc_total

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x, y_random, color=COLOR_BASELINE, label="Random", linewidth=1.5, linestyle=":")

    ax.plot(x, y_causal, color=COLOR_MODEL, label=causal_score_label)
    ax.fill_between(x, y_causal, y_random, color=COLOR_MODEL, alpha=0.08)

    if y_affinity is not None and affinity_score_label is not None:
        ax.plot(x, y_affinity, color=COLOR_REFERENCE, label=affinity_score_label)
        ax.fill_between(x, y_affinity, y_random, color=COLOR_REFERENCE, alpha=0.08)

    if relative_axes:
        ax.set(xticks=np.linspace(0, 1, 5), yticks=np.linspace(0, 1, 5))
        ax.set_xlabel("Anteil Experimentalmenge")
        ax.set_ylabel("Anteil inkrementelles Ergebnis")
    else:
        ax.set_xlabel("Experimentalmenge")
        ax.set_ylabel("Inkrementelles Ergebnis")

    ax.set_title(f"Qini-Vergleich: {causal_score_label}")
    ax.legend(loc="upper left")

    return ax

def policy_value_comparison_plots(
    policy_values_dict: dict[str, pd.DataFrame],
    comparison_model_name: str,
) -> dict[str, plt.Figure]:
    """Erzeugt Vergleichsplots der Policy Values gegen ein Referenzmodell.

    Neben der Qini-Kurve ist der erwartete inkrementelle Nutzen (Policy Value)
    für feste Treat-Anteile (z. B. 10%, 20%, …) eine zentrale Entscheidungsgrundlage.
    Diese Funktion erstellt für jedes Modell (außer dem Referenzmodell) einen Plot
    mit drei Kurven:

    1) Policy Values des Modells inkl. Konfidenzintervall
    2) Policy Values des Referenzmodells inkl. Konfidenzintervall
    3) Differenzkurve (Modell − Referenz)

    Erwartetes DataFrame-Format je Modell:
    - treated_percentage
    - policy_value
    - lower_bound
    - upper_bound"""

    if comparison_model_name not in policy_values_dict:
        raise KeyError(
            f"Referenzmodell '{comparison_model_name}' nicht in policy_values_dict vorhanden."
        )

    plots: dict[str, plt.Figure] = {}
    ref = policy_values_dict[comparison_model_name]

    # Skip if reference has no data
    if ref is None or len(ref) == 0 or "policy_value" not in ref.columns:
        return plots

    for model_name, df in policy_values_dict.items():
        if model_name == comparison_model_name:
            continue
        if df is None or len(df) == 0 or "policy_value" not in df.columns:
            continue

        treated = df["treated_percentage"]
        model_values = df["policy_value"]
        model_lower = df["lower_bound"]
        model_upper = df["upper_bound"]

        ref_values = ref["policy_value"]
        ref_lower = ref["lower_bound"]
        ref_upper = ref["upper_bound"]

        difference = model_values.to_numpy() - ref_values.to_numpy()

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(treated, model_values, label=f"{model_name} Policy Value", marker="o", color=COLOR_MODEL)
        ax.fill_between(treated, model_lower, model_upper, alpha=0.12, color=COLOR_MODEL, label=f"{model_name} Konfidenzintervall")

        ax.plot(treated, ref_values, label=f"{comparison_model_name} Policy Value", linestyle="--", marker="o", color=COLOR_REFERENCE)
        ax.fill_between(treated, ref_lower, ref_upper, alpha=0.12, color=COLOR_REFERENCE, label=f"{comparison_model_name} Konfidenzintervall")

        ax.plot(treated, difference, label="Differenz (Modell - Referenz)", linestyle="-.", marker="x", color=COLOR_DIFFERENCE)
        ax.axhline(0, color=COLOR_BASELINE, linewidth=0.8, linestyle="--", label="Keine Differenz")

        ax.set_title(f"Policy Value Vergleich: {model_name} vs. {comparison_model_name}")
        ax.set_xlabel("Treated Percentage")
        ax.set_ylabel("Policy Value")
        ax.legend(loc="upper left")

        plots[model_name] = fig

    return plots



def plot_score_redistribution(
    cate_scores: np.ndarray,
    hist_scores: np.ndarray,
    n_bins: int = 10,
    model_name: str = "",
) -> "matplotlib.figure.Figure":
    """Score-Redistribution-Plot: Gestapeltes Balkendiagramm mit Colorbar.

    Zeigt pro Dezil des neuen CATE-Scores, welcher Anteil der Samples
    aus welchem Dezil des historischen Scores stammt.

    Parameters
    ----------
    cate_scores : (n,) array
        Neue CATE-Vorhersagen (höher = stärkerer Treatment-Effekt).
    hist_scores : (n,) array
        Historische Scores (höher = besser gemäß historical_score.higher_is_better).
    n_bins : int
        Anzahl der Perzentil-Bins (Default 10 = Dezile).
    model_name : str
        Modellname für den Titel.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.colorbar import ColorbarBase
    from scipy.stats import spearmanr
    try:
        from rubin.utils.plot_theme import RUBIN_COLORS, apply_rubin_theme
        apply_rubin_theme()
    except ImportError:
        RUBIN_COLORS = {"ruby_dark": "#6B0D15", "ruby": "#9B111E",
                        "ruby_pale": "#FDF2F3", "slate": "#57606A",
                        "text": "#24292F", "grid": "#EDE6E7"}

    cate_scores = np.asarray(cate_scores, dtype=float).ravel()
    hist_scores = np.asarray(hist_scores, dtype=float).ravel()
    assert len(cate_scores) == len(hist_scores), (
        "CATE und historische Scores müssen gleich lang sein."
    )

    # ── Rangkorrelation ──
    rho, _ = spearmanr(cate_scores, hist_scores)

    # ── Dezil-Zuordnung (0=niedrigste 10%, n_bins-1=höchste 10%) ──
    def _assign_bins(scores, n):
        edges = np.percentile(scores, np.linspace(0, 100, n + 1))
        edges[-1] += 1e-10
        bins = np.digitize(scores, edges[1:], right=False)
        return np.clip(bins, 0, n - 1)

    cate_bins = _assign_bins(cate_scores, n_bins)
    hist_bins = _assign_bins(hist_scores, n_bins)

    # ── Transitionsmatrix: (n_bins × n_bins) — row=hist, col=cate ──
    matrix = np.zeros((n_bins, n_bins), dtype=float)
    for cb, hb in zip(cate_bins, hist_bins):
        matrix[hb, cb] += 1.0

    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    matrix_pct = matrix / col_sums * 100.0

    # ── Spalten umkehren: D10 (höchstes CATE) links, D1 rechts ──
    # Konsistent mit Qini-Kurve und Uplift-by-Percentile (Top-Kunden links)
    matrix_pct = matrix_pct[:, ::-1]

    # ── Farbverlauf: D1 (niedrigstes hist.) = hell, Dn (höchstes) = dunkel ──
    cmap = LinearSegmentedColormap.from_list(
        "rubin_redistribution",
        [RUBIN_COLORS["ruby_pale"], RUBIN_COLORS["ruby"], RUBIN_COLORS["ruby_dark"]],
        N=256,
    )
    colors = [cmap(i / (n_bins - 1)) for i in range(n_bins)]

    # ── Figure ──
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[35, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    x = np.arange(1, n_bins + 1)
    bottom = np.zeros(n_bins)

    for hist_d in range(n_bins):
        heights = matrix_pct[hist_d, :]
        ax.bar(x, heights, bottom=bottom, width=0.82,
               color=colors[hist_d], edgecolor="white", linewidth=0.5)
        bottom += heights

    ax.set_xlabel("Dezil des Kausal-Scores (CATE)")
    ax.set_ylabel("Anteil der Samples (%)")

    # Kompakte Kennzahl statt Titel (wie native Plots)
    ax.set_title(f"Score-Redistribution\nSpearman \u03c1 = {rho:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"D{n_bins + 1 - i}" for i in x])
    ax.set_ylim(0, 100)
    ax.set_xlim(0.35, n_bins + 0.65)

    # ── Colorbar ──
    norm = Normalize(vmin=1, vmax=n_bins)
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label("Dezil Hist. Score", fontsize=10, labelpad=8)
    tick_labels = [f"D{i}" for i in range(1, n_bins + 1)]
    tick_labels[0] = "D1\n(niedrig)"
    tick_labels[-1] = f"D{n_bins}\n(hoch)"
    cb.set_ticks(list(range(1, n_bins + 1)))
    cb.set_ticklabels(tick_labels, fontsize=8)
    cb.ax.tick_params(size=0)

    fig.subplots_adjust(bottom=0.10, top=0.90, left=0.10, right=0.91)
    return fig
