from __future__ import annotations

"""Evaluation-Plots: CATE-Verteilungen, ATE-Barplots, native Uplift-Kurven.

Erzeugt die visuellen Evaluations-Artefakte für den HTML-Report:
Qini-Kurven, Uplift-by-Percentile, Treatment-Balance und den
DRTester-gestützten evaluate_cate_with_plots-Orchestrator.
"""

from typing import Any, Optional
import logging

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rubin.utils.plot_theme import (
    apply_rubin_theme, RUBIN_COLORS, recolor_figure,
)
apply_rubin_theme()

from rubin.evaluation.drtester_core import CustomDRTester, DrTesterPlotBundle
from rubin.evaluation.uplift_metrics import tiebreak_jitter

def generate_cate_distribution_plot(
    cate_preds_val: np.ndarray,
    cate_preds_train: Optional[np.ndarray] = None,
    model_name: str = "",
    arm_label: str = "",
) -> Optional[Any]:
    """Erzeugt ein Histogramm der CATE-Predictions (Training + Cross-Validated).

    Zeigt die Verteilung der vorhergesagten Treatment-Effekte. Dient der
    visuellen Plausibilitätsprüfung: stark konzentrierte Verteilungen nahe
    Null deuten auf wenig Heterogenität hin, breite Verteilungen auf
    differenzierte Effektvorhersagen.

    Returns
    -------
    matplotlib.figure.Figure oder None bei Fehler."""
    from rubin.utils.plot_theme import apply_rubin_theme
    apply_rubin_theme()

    val = np.asarray(cate_preds_val, dtype=float).ravel()
    val = val[~np.isnan(val)]
    if len(val) == 0:
        return None

    has_train = cate_preds_train is not None
    if has_train:
        train = np.asarray(cate_preds_train, dtype=float).ravel()
        train = train[~np.isnan(train)]
        has_train = len(train) > 0

    suffix = f" {arm_label}" if arm_label else ""
    n_cols = 2 if has_train else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]

    # Gemeinsamer Bin-Bereich für Vergleichbarkeit
    all_vals = np.concatenate([val, train]) if has_train else val
    q_lo, q_hi = np.percentile(all_vals, [0.5, 99.5])
    bins = np.linspace(q_lo, q_hi, 80)

    color = RUBIN_COLORS["ruby"]
    edge_color = RUBIN_COLORS["ruby_dark"]

    if has_train:
        ax_train = axes[0]
        ax_train.hist(train, bins=bins, color=color, edgecolor=edge_color,
                      linewidth=0.3, alpha=0.85)
        ax_train.set_title(f"Training Predictions{suffix}", fontsize=12, fontweight="bold")
        ax_train.set_xlabel(f"Train_{model_name}{suffix}", fontsize=10)
        ax_train.set_ylabel("Count", fontsize=10)
        ax_train.axvline(0, color=RUBIN_COLORS["slate"], linewidth=1, linestyle="--", alpha=0.6)

    ax_val = axes[-1]
    ax_val.hist(val, bins=bins, color=color, edgecolor=edge_color,
                linewidth=0.3, alpha=0.85)
    ax_val.set_title(f"Cross-Validated Predictions{suffix}", fontsize=12, fontweight="bold")
    ax_val.set_xlabel(f"Predictions_{model_name}{suffix}", fontsize=10)
    ax_val.set_ylabel("Count", fontsize=10)
    ax_val.axvline(0, color=RUBIN_COLORS["slate"], linewidth=1, linestyle="--", alpha=0.6)

    fig.suptitle(f"CATE Distribution — {model_name}{suffix}", fontsize=14,
                 fontweight="bold", color=RUBIN_COLORS["ruby_dark"], y=1.02)
    fig.tight_layout()
    return fig


def generate_ate_barplot(
    T: np.ndarray, Y: np.ndarray,
) -> Optional[plt.Figure]:
    """Barplot der Outcome-Rate pro Treatment-Gruppe mit ATE-Annotation.

    Zeigt Response Rates (E[Y|T=k]) pro Gruppe als Balken mit
    Standard-Error-Balken und ATE-Differenz als Annotation.
    Direkt in rubin-Farbpalette, kein recolor nötig."""
    try:
        from rubin.utils.plot_theme import RUBIN_COLORS

        t_arr = np.asarray(T).ravel()
        y_arr = np.asarray(Y).ravel().astype(float)
        groups = sorted(np.unique(t_arr).tolist())
        n_groups = len(groups)

        rates = []
        ses = []
        ns = []
        labels = []
        for g in groups:
            mask = t_arr == g
            n_g = mask.sum()
            rate = float(y_arr[mask].mean()) if n_g > 0 else 0.0
            se = float(y_arr[mask].std(ddof=1)) / np.sqrt(n_g) if n_g > 1 else 0.0
            rates.append(rate)
            ses.append(se)
            ns.append(n_g)
            labels.append(f"T={int(g)}" if n_groups <= 5 else f"{int(g)}")

        # Farben: Control = Gold, Treatment(s) = Ruby-Palette
        if n_groups == 2:
            colors = [RUBIN_COLORS["gold"], RUBIN_COLORS["ruby"]]
        else:
            palette = [RUBIN_COLORS["gold"], RUBIN_COLORS["ruby"], RUBIN_COLORS["ruby_dark"],
                       RUBIN_COLORS["ruby_light"], "#8B6914", "#E07A5F"]
            colors = [palette[i % len(palette)] for i in range(n_groups)]

        fig, ax = plt.subplots(figsize=(max(4, n_groups * 1.5 + 1), 4.5))
        x = np.arange(n_groups)
        ax.bar(x, rates, yerr=ses, width=0.5, color=colors,
                      edgecolor="black", linewidth=0.5, alpha=0.85,
                      capsize=5, error_kw={"linewidth": 1.2})

        # Werte über den Balken
        for i, (r, se, n_g) in enumerate(zip(rates, ses, ns)):
            ax.text(i, r + se + max(rates) * 0.02, f"{r:.4f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=RUBIN_COLORS["text"])
            ax.text(i, r / 2, f"n={n_g:,}",
                    ha="center", va="center", fontsize=9,
                    color="white",
                    # "bold" statt "600": DejaVu Sans (matplotlib-Default) hat
                    # nur die Weights 200/400/700 — 600 löste bei jedem Lauf
                    # eine findfont-Warnung aus und fiel ohnehin auf 700 zurück.
                    fontweight="bold")

        # ATE-Annotation
        if n_groups == 2:
            ate = rates[1] - rates[0]
            ax.annotate(
                "", xy=(1, rates[1]), xytext=(0, rates[0]),
                arrowprops=dict(arrowstyle="<->", color=RUBIN_COLORS["slate"], lw=1.5),
            )
            mid_y = (rates[0] + rates[1]) / 2
            ax.text(0.5, mid_y, f"ATE = {ate:+.4f}",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color=RUBIN_COLORS["ruby_dark"],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=RUBIN_COLORS["ruby_dark"], alpha=0.9))
        elif n_groups > 2:
            # MT: ATE pro Arm vs. Control (T=0) als kleine Labels über den Balken
            for i in range(1, n_groups):
                ate_arm = rates[i] - rates[0]
                ax.text(i, rates[i] + ses[i] + max(rates) * 0.09,
                        f"Δ{ate_arm:+.4f}",
                        ha="center", va="bottom", fontsize=8.5,
                        color=RUBIN_COLORS["slate"],
                        fontstyle="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Outcome Rate  E[Y | T]", fontsize=11)
        ax.set_title("Outcome Rate by Treatment Group", fontsize=13, fontweight="bold",
                     color=RUBIN_COLORS["ruby_dark"])
        ax.set_ylim(0, max(rates) * (1.35 if n_groups > 2 else 1.25))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig
    except Exception as e:
        _logger.warning("ATE-Barplot fehlgeschlagen: %s", e)
        return None


def _native_uplift_by_percentile(
    y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray,
    strategy: str = "overall", kind: str = "bar",
    n_bins: int = 10, string_percentiles: bool = True,
) -> Optional[plt.Figure]:
    """Uplift-by-Percentile Plot — native rubin-Implementierung.

    Vollständige Uplift-by-Percentile-Implementierung inkl. aller
    Parameter-Optionen (strategy, kind, bins, string_percentiles).

    Args:
        strategy: 'overall' (Default) — sortiert ALLE Daten nach Uplift, teilt in Bins.
                  'by_group' — sortiert Treatment/Control getrennt, nimmt Top-k% je Gruppe.
        kind: 'bar' (Default) — 2-Panel Barplot (Uplift oben, Response Rates unten).
              'line' — Einzelner Lineplot mit Fehlerbalken und Fill.
        n_bins: Anzahl Bins (Default 10).
        string_percentiles: True → X-Labels als Strings ('0-10', '10-20', ...).

    """
    try:
        from rubin.utils.plot_theme import RUBIN_COLORS, COLOR_MODEL, COLOR_REFERENCE

        y_arr = np.asarray(y_true).ravel()
        u_arr = np.asarray(uplift).ravel()
        t_arr = np.asarray(treatment).ravel()

        # ── Berechnung der Perzentil-Metriken ──
        if strategy == "by_group":
            # Getrennte Sortierung: Top-k% Treatment vs Top-k% Control
            treat_idx = np.where(t_arr == 1)[0]
            ctrl_idx = np.where(t_arr == 0)[0]
            _rng_pct = np.random.default_rng(42)
            treat_order = treat_idx[np.argsort(-(u_arr[treat_idx] + tiebreak_jitter(u_arr[treat_idx], _rng_pct)))]
            ctrl_order = ctrl_idx[np.argsort(-(u_arr[ctrl_idx] + tiebreak_jitter(u_arr[ctrl_idx], _rng_pct)))]
            treat_edges = np.linspace(0, len(treat_order), n_bins + 1, dtype=int)
            ctrl_edges = np.linspace(0, len(ctrl_order), n_bins + 1, dtype=int)
            rr_treat, rr_ctrl, std_treat, std_ctrl = [], [], [], []
            n_treat_bins = []
            for i in range(n_bins):
                t_lo, t_hi = treat_edges[i], treat_edges[i + 1]
                c_lo, c_hi = ctrl_edges[i], ctrl_edges[i + 1]
                yt = y_arr[treat_order[t_lo:t_hi]]
                yc = y_arr[ctrl_order[c_lo:c_hi]]
                nt, nc = len(yt), len(yc)
                rt = float(yt.mean()) if nt > 0 else 0.0
                rc = float(yc.mean()) if nc > 0 else 0.0
                st = float(yt.std()) if nt > 1 else 0.0
                sc = float(yc.std()) if nc > 1 else 0.0
                rr_treat.append(rt); rr_ctrl.append(rc)
                std_treat.append(st); std_ctrl.append(sc)
                n_treat_bins.append(nt)
        else:  # strategy == "overall"
            _rng_ov = np.random.default_rng(42)
            order = np.argsort(-(u_arr + tiebreak_jitter(u_arr, _rng_ov)))
            y_s = y_arr[order]
            t_s = t_arr[order]
            n = len(y_s)
            bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
            rr_treat, rr_ctrl, std_treat, std_ctrl = [], [], [], []
            n_treat_bins = []
            for i in range(n_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                y_bin, t_bin = y_s[lo:hi], t_s[lo:hi]
                treat_mask = t_bin == 1
                ctrl_mask = t_bin == 0
                nt = treat_mask.sum()
                nc = ctrl_mask.sum()
                rt = float(y_bin[treat_mask].mean()) if nt > 0 else 0.0
                rc = float(y_bin[ctrl_mask].mean()) if nc > 0 else 0.0
                st = float(y_bin[treat_mask].std()) if nt > 1 else 0.0
                sc = float(y_bin[ctrl_mask].std()) if nc > 1 else 0.0
                rr_treat.append(rt); rr_ctrl.append(rc)
                std_treat.append(st); std_ctrl.append(sc)
                n_treat_bins.append(nt)

        bin_uplifts = [rt - rc for rt, rc in zip(rr_treat, rr_ctrl)]

        # Weighted average uplift
        total_treat = sum(n_treat_bins)
        weighted_avg = (sum(u * nt for u, nt in zip(bin_uplifts, n_treat_bins)) / total_treat
                        if total_treat > 0 else 0.0)

        # Perzentil-Werte (Mittelwerte der Bin-Ränder)
        pct_values = np.array([(i + 0.5) * 100 / n_bins for i in range(n_bins)])

        if string_percentiles:
            pct_labels = [f"0-{100 / n_bins:.0f}"] + \
                         [f"{i * 100 / n_bins:.0f}-{(i + 1) * 100 / n_bins:.0f}" for i in range(1, n_bins)]
        else:
            pct_labels = [f"{v:.0f}" for v in pct_values]

        # ── Plotting ──
        if kind == "line":
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(pct_values, rr_treat, linewidth=2,
                    color=RUBIN_COLORS["ruby_dark"], label="treatment\nresponse rate", marker="o", markersize=4)
            ax.plot(pct_values, rr_ctrl, linewidth=2,
                    color=COLOR_REFERENCE, label="control\nresponse rate", marker="o", markersize=4)
            ax.plot(pct_values, bin_uplifts, linewidth=2,
                    color=COLOR_MODEL, label="uplift", marker="s", markersize=4)
            ax.fill_between(pct_values, rr_treat, rr_ctrl, alpha=0.1, color=COLOR_MODEL)
            if min(bin_uplifts) < 0:
                ax.axhline(y=0, color="black", linewidth=1)
            ax.set_xticks(pct_values)
            ax.set_xticklabels(pct_labels, rotation=45)
            ax.legend(loc="upper right")
            ax.set_title(f"Uplift by percentile\nweighted average uplift = {weighted_avg:.4f}")
            ax.set_xlabel("Percentile")
            ax.set_ylabel("Uplift = treatment response rate - control response rate")
            fig.tight_layout()
            return fig

        else:  # kind == "bar"
            delta = pct_values[0] if len(pct_values) > 0 else 5
            w = delta / 3

            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
            fig.text(0.04, 0.5, "Uplift = treatment response rate - control response rate",
                     va="center", ha="center", rotation="vertical", fontsize=9,
                     color=RUBIN_COLORS.get("slate", "#57606A"))

            # axes[0]: Uplift bars
            axes[0].bar(pct_values, bin_uplifts, width=delta / 1.5,
                        color=COLOR_MODEL, edgecolor="black", linewidth=0.5, alpha=0.7,
                        label="uplift")
            axes[0].axhline(0, color="black", linewidth=1)
            axes[0].legend(loc="upper right")
            axes[0].tick_params(axis="x", bottom=False)
            axes[0].set_title(f"Uplift by percentile\nweighted average uplift = {weighted_avg:.4f}")

            # axes[1]: Response rates
            axes[1].bar(pct_values - w / 2, rr_treat, width=w,
                        color=RUBIN_COLORS["ruby_dark"], edgecolor="black", linewidth=0.5,
                        alpha=0.7, label="treatment\nresponse rate")
            axes[1].bar(pct_values + w / 2, rr_ctrl, width=w,
                        color=COLOR_REFERENCE, edgecolor="black", linewidth=0.5,
                        alpha=0.7, label="control\nresponse rate")
            axes[1].axhline(0, color="black", linewidth=1)
            axes[1].set_xticks(pct_values)
            axes[1].set_xticklabels(pct_labels, rotation=45)
            axes[1].set_xlabel("Percentile")
            axes[1].legend(loc="upper right")
            axes[1].set_title("Response rate by percentile")

            fig.tight_layout()
            fig.subplots_adjust(left=0.12)  # Platz für rotiertes Y-Label
            return fig
    except Exception as e:
        _logger.warning("Native Uplift-by-Percentile fehlgeschlagen: %s", e)
        return None


def _native_treatment_balance(
    uplift: np.ndarray, treatment: np.ndarray,
    random: bool = True, winsize: float = 0.1,
) -> Optional[plt.Figure]:
    """Treatment-Balance-Kurve — native rubin-Implementierung.

    Vollständige Treatment-Balance-Implementierung inkl.
    aller Parameter (random, winsize).

    Args:
        random: Zeichne Random-Baseline (durchschnittliche Treatment-Rate). Default True.
        winsize: Fenstergröße als Anteil (0-1, Extrema ausgeschlossen). Default 0.1."""
    try:
        from rubin.utils.plot_theme import COLOR_MODEL, COLOR_BASELINE
        # Tie-Breaking: Bei identischen Scores ist np.argsort willkürlich —
        # die Reihenfolge hängt von der Datenreihenfolge ab (z.B. alle T=1 zuerst).
        # Random Jitter verhindert artifizielle Balance-Schwankungen bei Ties.
        rng = np.random.default_rng(42)
        order = np.argsort(-(np.asarray(uplift, dtype=float) + tiebreak_jitter(uplift, rng)))
        t_s = treatment[order].astype(float)
        n = len(t_s)
        window = max(1, int(winsize * n))

        cumsum = np.cumsum(np.insert(t_s, 0, 0))
        window_sums = cumsum[window:] - cumsum[:-window]
        treatment_rates = window_sums / window

        x_axis = np.arange(len(treatment_rates)) / n
        overall_rate = float(treatment.mean())

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(x_axis, treatment_rates, linewidth=2, color=COLOR_MODEL, label="Model")

        if random:
            y_random = overall_rate * np.ones_like(x_axis)
            ax.plot(x_axis, y_random, color=COLOR_BASELINE, linewidth=1.0,
                    linestyle="--", label="Random")
            ax.fill_between(x_axis, treatment_rates, y_random, alpha=0.12, color=COLOR_MODEL)

        ax.legend()
        ax.set_title("Treatment balance curve")
        ax.set_xlabel("Percentage targeted")
        ax.set_ylabel("Balance: treatment / (treatment + control)")
        fig.tight_layout()
        return fig
    except Exception as e:
        _logger.warning("Native Treatment-Balance fehlgeschlagen: %s", e)
        return None


def _native_qini_curve(
    y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray,
    random: bool = True, perfect: bool = False, negative_effect: bool = True,
    name: str = None,
) -> Optional[plt.Figure]:
    """Qini-Kurve — native rubin-Implementierung.

    Vollständige Qini-Kurven-Implementierung inkl. aller
    Parameter (random, perfect, negative_effect, name).

    Args:
        random: Zeichne Random-Baseline. Default True.
        perfect: Zeichne perfekte Qini-Kurve (theoretisches Optimum). Default False.
        negative_effect: Enthält die perfekte Kurve negative Effekte? Default True.
        name: Modellname für die Legende. Default None."""
    try:
        from rubin.utils.plot_theme import COLOR_MODEL, COLOR_BASELINE
        from rubin.evaluation.uplift_metrics import uplift_curve as _uc, qini_coefficient as _qc

        curve = _uc(y=y_true, t=treatment, score=uplift)
        qini = _qc(curve)

        # Qini-Formel: Q(k) = Y_t(k) − Y_c(k) · N_t(k) / N_c(k)
        n_c_safe = np.maximum(curve.n_control, 1)
        y_qini = curve.y_treat - curve.y_control * curve.n_treat / n_c_safe
        n = len(np.asarray(y_true).ravel())
        x = np.r_[0, curve.fraction * n]  # Absolute Anzahl targeted
        y_qini = np.r_[0, y_qini]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Model curve
        label = f"{name} (AUC = {qini:.4f})" if name else "Model"
        ax.plot(x, y_qini, linewidth=2, color=COLOR_MODEL, label=label)

        # Random baseline
        if random:
            y_random = x * y_qini[-1] / n if n > 0 else x * 0
            ax.plot(x, y_random, label="Random", color=COLOR_BASELINE, linewidth=1.2, linestyle="--")
            ax.fill_between(x, y_qini, y_random, alpha=0.12, color=COLOR_MODEL)

        # Perfect curve
        if perfect:
            y_arr = np.asarray(y_true).ravel()
            t_arr = np.asarray(treatment).ravel()
            n_t = (t_arr == 1).sum()
            n_c = max((t_arr == 0).sum(), 1)
            if negative_effect:
                # Perfekte Sortierung: true uplift = Y·T − Y·(1−T)
                true_uplift = y_arr * t_arr - y_arr * (1 - t_arr)
                perf_curve = _uc(y=y_arr, t=t_arr, score=true_uplift)
                n_c_perf = np.maximum(perf_curve.n_control, 1)
                y_perf = perf_curve.y_treat - perf_curve.y_control * perf_curve.n_treat / n_c_perf
                x_perf = np.r_[0, perf_curve.fraction * n]
                y_perf = np.r_[0, y_perf]
            else:
                ratio = y_arr[t_arr == 1].sum() - y_arr[t_arr == 0].sum() * n_t / n_c
                x_perf = np.array([0, n_t, n])
                y_perf = np.array([0, ratio, ratio])
            ax.plot(x_perf, y_perf, label="Perfect", color="#dc2626", linewidth=1.2, linestyle=":")

        ax.legend(loc="lower right")
        ax.set_title(f"Qini curve\nqini_auc_score={qini:.4f}")
        ax.set_xlabel("Number targeted")
        ax.set_ylabel("Number of incremental outcome")
        fig.tight_layout()
        return fig
    except Exception as e:
        _logger.warning("Native Qini-Curve fehlgeschlagen: %s", e)
        return None


def generate_uplift_plots(
    cate_preds_val: np.ndarray,
    T_val: np.ndarray,
    Y_val: np.ndarray,
) -> tuple:
    """Erzeugt Uplift-Plots: Qini-Kurve, Uplift-by-Percentile, Treatment-Balance.

    Alle Plots sind native rubin-Implementierungen mit rubin-Farbpalette.
    Native Uplift-Plots (
    numpy-Kompatibilitätsprobleme, GitHub Issue #213).

    Gibt (qini, percentile, treatment_balance) als Matplotlib-Figures zurück.
    Fehlende Plots sind None."""

    uplift = np.asarray(cate_preds_val).ravel()
    y_val_arr = np.asarray(Y_val).ravel()
    t_val_arr = np.asarray(T_val).ravel()

    sk_qini = _native_qini_curve(y_val_arr, uplift, t_val_arr)
    sk_pct = _native_uplift_by_percentile(y_val_arr, uplift, t_val_arr)
    sk_tb = _native_treatment_balance(uplift, t_val_arr)

    return sk_qini, sk_pct, sk_tb


def evaluate_cate_with_plots(
    *,
    model_regression: Any = None,
    model_propensity: Any = None,
    X_val: pd.DataFrame,
    T_val: np.ndarray,
    Y_val: np.ndarray,
    cate_preds_val: np.ndarray,
    X_train: Optional[pd.DataFrame] = None,
    T_train: Optional[np.ndarray] = None,
    Y_train: Optional[np.ndarray] = None,
    cate_preds_train: Optional[np.ndarray] = None,
    n_groups: int = 10,
    fitted_tester: Optional[CustomDRTester] = None,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> DrTesterPlotBundle:
    """Hauptfunktion: DRTester-Auswertung + Uplift-Plots.

    Wenn fitted_tester übergeben wird, werden die vorberechneten Nuisance-Ergebnisse
    (DR-Outcomes) wiederverwendet — das spart das teure Nuisance-CV-Fitting.
    Nur die CATE-Predictions werden ausgetauscht."""

    if fitted_tester is not None:
        # Nuisance bereits berechnet → frischen Tester mit gleichen DR-Outcomes erstellen.
        # KEIN copy.copy (EconML-interner State ist nicht copy-sicher).
        # Stattdessen: neuer Tester, Nuisance-State manuell injizieren.
        tester = CustomDRTester(
            model_regression=getattr(fitted_tester, '_model_regression_ref', None),
            model_propensity=getattr(fitted_tester, '_model_propensity_ref', None),
            cate=None,
            cate_preds_val=cate_preds_val,
            cate_preds_train=cate_preds_train,
        )
        # DR-Outcomes + Nuisance-State aus dem Pre-Fit übernehmen (das teure Ergebnis)
        tester.dr_val_ = CustomDRTester._sanitize_dr(fitted_tester.dr_val_)
        tester.ate_val = tester.dr_val_.mean(axis=0)
        tester.Dval = fitted_tester.Dval
        tester.treatments = fitted_tester.treatments
        tester.n_treat = fitted_tester.n_treat
        tester.fit_on_train = fitted_tester.fit_on_train
        if hasattr(fitted_tester, 'dr_train_'):
            tester.dr_train_ = fitted_tester.dr_train_
        # Hinweis: cate_preds_train_ wird vom Konstruktor oben gesetzt
        # (aus dem cate_preds_train-Argument). Der Pre-Fit-Tester aus
        # fit_drtester_nuisance trägt selbst nie Train-Preds (Dummy-CATE)
        # und ist dafür keine Quelle.
        _logger.debug(
            "DRTester Nuisance wiederverwendet: dr_val=%s, cate_preds_val=%s (min=%.4g, max=%.4g)",
            tester.dr_val_.shape, tester.cate_preds_val_.shape,
            float(np.nanmin(cate_preds_val)), float(np.nanmax(cate_preds_val)),
        )
    else:
        # Fallback: Nuisance komplett neu fitten (wenn kein Pre-Fit vorhanden)
        if model_regression is None or model_propensity is None:
            raise ValueError("Entweder fitted_tester oder model_regression + model_propensity muss angegeben werden.")
        tester = CustomDRTester(
            model_regression=model_regression,
            model_propensity=model_propensity,
            cate=None,
            cate_preds_val=cate_preds_val,
            cate_preds_train=cate_preds_train,
        )

        if X_train is not None and T_train is not None and Y_train is not None:
            tester.fit_nuisance(
                Xval=X_val.values,
                Dval=np.asarray(T_val).ravel(),
                yval=np.asarray(Y_val).ravel(),
                Xtrain=X_train.values,
                Dtrain=np.asarray(T_train).ravel(),
                ytrain=np.asarray(Y_train).ravel(),
            )
        else:
            tester.fit_nuisance(
                Xval=X_val.values,
                Dval=np.asarray(T_val).ravel(),
                yval=np.asarray(Y_val).ravel(),
            )

    # ── DRTester-Plots (Calibration, Qini, TOC) ──
    summary, cal_plot, qini_plot, toc_plot, policy_values = None, None, None, None, None
    try:
        res = tester.evaluate_all(X_val.values, X_train.values if X_train is not None else None, n_groups=n_groups, n_bootstrap=n_bootstrap, seed=seed)
        summary = res.summary()
        cal_plot = res.plot_cal(1).get_figure()
        recolor_figure(cal_plot)
        qini_plot = res.plot_qini(1).get_figure()
        recolor_figure(qini_plot)
        toc_plot = res.plot_toc(1).get_figure()
        recolor_figure(toc_plot)
        policy_values = res.get_policy_values(1)
    except Exception as e:
        _logger.warning("DRTester evaluate_all fehlgeschlagen: %s", e, exc_info=True)
        if summary is None:
            summary = pd.DataFrame()
        if policy_values is None:
            policy_values = pd.DataFrame()

    # ── Uplift-Plots: Qini, Uplift-by-Percentile, Treatment-Balance ──
    # Native rubin-Implementierungen
    uplift = np.asarray(cate_preds_val).ravel()
    y_val_arr = np.asarray(Y_val).ravel()
    t_val_arr = np.asarray(T_val).ravel()

    sk_qini = _native_qini_curve(y_val_arr, uplift, t_val_arr)
    sk_pct = _native_uplift_by_percentile(y_val_arr, uplift, t_val_arr)
    sk_tb = _native_treatment_balance(uplift, t_val_arr)

    return DrTesterPlotBundle(
        summary=summary,
        cal_plot=cal_plot,
        qini_plot=qini_plot,
        toc_plot=toc_plot,
        policy_values=policy_values,
        uplift_qini=sk_qini,
        uplift_percentile=sk_pct,
        treatment_balance=sk_tb,
    )


