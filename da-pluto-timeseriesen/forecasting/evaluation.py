"""
Evaluations- und Visualisierungshilfen.

Enthält:

- :func:`evaluate_multivariate` – Standardmetriken pro Komponente
  (MAE, MSE, RMSE, MAPE, sMAPE) auf einem Forecast-Abschnitt.
- :func:`weekly_baseline_mean_last3` – einfache saisonale Baseline über
  den Mittelwert der gleichen Kalenderwoche in den Vorjahren.
- :func:`plot_agg_schriftstueck_with_components` – Visualisierung der
  aggregierten Schriftstück-Kennzahl und der Top-Komponenten.
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries

from .config import BaselineConfig


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error in Prozent.

    Punkte mit ``|y_true| + |y_pred| == 0`` werden ausgeschlossen. Gibt
    ``nan`` zurück, wenn kein Punkt zur Auswertung übrig bleibt.
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask])) * 100.0


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error in Prozent.

    Punkte mit ``y_true == 0`` werden ausgeschlossen. Gibt ``nan``
    zurück, wenn kein Punkt zur Auswertung übrig bleibt.
    """
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100.0


def evaluate_multivariate(
    true_ts: TimeSeries,
    pred_ts: TimeSeries,
    output_chunk_length: int,
    return_df: bool = True,
) -> pd.DataFrame:
    """
    Berechnet Standardmetriken je Komponente auf dem gemeinsamen
    Zeitintervall von ``true_ts`` und ``pred_ts``.

    Parameters
    ----------
    true_ts
        Grundwahrheit (multivariate Zeitreihe).
    pred_ts
        Prognose. Muss die gleichen Komponenten wie ``true_ts`` besitzen.
    output_chunk_length
        Prognosehorizont in Zeitschritten. Wird in der ``horizon``-Spalte
        mitgeführt, damit mehrere Horizonte in einem DataFrame verglichen
        werden können.
    return_df
        Aus Kompatibilitätsgründen vorhanden. Wird derzeit in beiden
        Zweigen gleich behandelt und liefert stets das DataFrame.

    Returns
    -------
    pd.DataFrame
        Eine Zeile je Komponente mit Spalten ``component``, ``horizon``,
        ``mae``, ``mse``, ``rmse``, ``mape``, ``smape``, ``wape``.
    """
    true_aligned = true_ts.slice_intersect(pred_ts)
    pred_aligned = pred_ts.slice_intersect(true_ts)

    metrics: List[dict] = []

    for component in true_aligned.components:
        y_true = true_aligned[component].values().flatten()
        y_pred = pred_aligned[component].values().flatten()

        mae = float(np.mean(np.abs(y_true - y_pred)))
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        wape = float(
            np.sum(np.abs(y_true - y_pred))
            / (np.sum(np.abs(y_true)) + 1e-8)
            * 100.0
        )

        metrics.append(
            {
                "component": component,
                "horizon": output_chunk_length,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "mape": _mape(y_true, y_pred),
                "smape": _smape(y_true, y_pred),
                "wape": wape,
            }
        )

    return pd.DataFrame(metrics)


def weekly_baseline_mean_last3(
    ts_y: TimeSeries,
    baseline_cfg: BaselineConfig,
) -> TimeSeries:
    """
    Einfache saisonale Baseline: Durchschnitt der Werte derselben
    Kalenderwoche in den letzten Vorjahren.

    Für jede Zeile wird in den Vorjahren nach derselben Kalenderwoche
    gesucht und der Jahresmittelwert dieser Kandidaten gemittelt. Reicht
    die Historie nicht (weniger Jahre als ``min_years``), bleibt die
    Zelle NaN.

    Parameters
    ----------
    ts_y
        Zielreihe mit wöchentlichem Index.
    baseline_cfg
        Steuert, wie viele Jahre zurück gesucht wird (``years_back``) und
        wie viele Jahre mindestens gefunden sein müssen (``min_years``).

    Returns
    -------
    TimeSeries
        Baseline-Serie gleicher Form wie ``ts_y``.
    """
    df = ts_y.to_dataframe()
    idx = df.index

    df["year"] = idx.year
    df["week"] = idx.isocalendar().week.values

    value_cols = [c for c in df.columns if c not in ("year", "week")]

    baseline_df = df[value_cols].copy()
    baseline_df[:] = np.nan

    for i, (ts_year, ts_week) in enumerate(zip(df["year"], df["week"])):
        mask_candidates = (
            (df["week"] == ts_week)
            & (df["year"] < ts_year)
            & (df["year"] >= ts_year - baseline_cfg.years_back)
        )
        candidate_years = df.loc[mask_candidates, "year"].unique()
        if len(candidate_years) < baseline_cfg.min_years:
            # Fallback: letzte beobachtete Werte, wenn zu wenig Historie
            if i > 0:
                baseline_df.iloc[i] = df[value_cols].iloc[i - 1]
            continue
        baseline_df.iloc[i] = (
            df.loc[mask_candidates, value_cols]
            .groupby(df["year"][mask_candidates])
            .mean()
            .mean()
        )

    return TimeSeries.from_dataframe(baseline_df, freq=ts_y.freq)


def compare_model_vs_baseline(
    true_ts: TimeSeries,
    model_pred_ts: TimeSeries,
    baseline_ts: TimeSeries,
    output_chunk_length: int,
) -> pd.DataFrame:
    """
    Vergleicht Modell- und Baseline-Metriken pro Komponente.

    Beide Prognosen werden auf **denselben Zeitraum** beschränkt —
    nämlich den Backtest-Zeitraum der Modellprognose. Dadurch ist der
    Vergleich fair: Modell und Baseline werden auf exakt denselben
    Wochen evaluiert.

    Negative Δ-Werte bedeuten: Modell ist besser als Baseline.

    Parameters
    ----------
    true_ts
        Grundwahrheit (volle Serie).
    model_pred_ts
        Modell-Prognose (Backtest, nur Validierungszeitraum).
    baseline_ts
        Baseline-Prognose (volle Serie, wird auf den Modell-Zeitraum
        zugeschnitten).
    output_chunk_length
        Prognosehorizont.

    Returns
    -------
    pd.DataFrame
        Spalten: ``component``, ``model_mae``, ``base_mae``, ``delta_mae``,
        ``model_smape``, ``base_smape``, ``delta_smape``,
        ``model_wape``, ``base_wape``, ``delta_wape``.
    """
    # Alle drei Serien auf den gemeinsamen Zeitraum (= Backtest-Fenster)
    # beschränken, damit der Vergleich fair ist.
    pred_start = model_pred_ts.start_time()
    pred_end = model_pred_ts.end_time()

    true_aligned = true_ts.slice(pred_start, pred_end)
    base_aligned = baseline_ts.slice(pred_start, pred_end)

    model_df = evaluate_multivariate(true_aligned, model_pred_ts, output_chunk_length)
    base_df = evaluate_multivariate(true_aligned, base_aligned, output_chunk_length)

    merged = model_df[["component"]].copy()
    for metric in ("mae", "smape", "wape"):
        merged[f"model_{metric}"] = model_df[metric].values
        merged[f"base_{metric}"] = base_df[metric].values
        merged[f"delta_{metric}"] = (
            model_df[metric].values - base_df[metric].values
        )

    return merged


def plot_agg_schriftstueck_with_components(
    true_ts: TimeSeries,
    pred_ts: TimeSeries,
    keyword_pattern: str = r"(?i)TERM_EINGANG_SCHRIFTST",
    title: str = "Aggregierte Schriftstück-Kennzahl (True vs Forecast)",
    diff_label: str = "Abweichung (Forecast − True)",
    overlay_top_n: int = 5,
) -> Dict[str, object]:
    """
    Visualisiert die aggregierte Schriftstück-Kennzahl als Summe über alle
    matchenden Komponenten und zeigt die umsatzstärksten Einzelkomponenten
    als Overlay.

    Der Plot wird über ``plt.show()`` direkt angezeigt; die Funktion ist
    primär für interaktive Notebook-Nutzung gedacht. Die zurückgegebenen
    Serien erlauben die spätere programmatische Weiterverarbeitung.

    Parameters
    ----------
    true_ts, pred_ts
        Grundwahrheit und Prognose.
    keyword_pattern
        Regex zur Auswahl relevanter Komponentennamen (per
        ``str.contains``).
    title
        Titel des Summenplots.
    diff_label
        Legendenlabel der Abweichungsfläche.
    overlay_top_n
        Anzahl der im Overlay gezeigten Einzelkomponenten (nach Summe
        über den Zeitraum). ``None`` bedeutet alle.

    Returns
    -------
    dict
        Mapping mit Schlüsseln ``true_sum_ts``, ``pred_sum_ts``,
        ``diff_ts``, ``matched_components``, ``overlay_components``.

    Raises
    ------
    ValueError
        Wenn keine Komponente zum ``keyword_pattern`` passt oder nach
        NaN-Filter kein gemeinsamer Punkt zwischen ``true_ts`` und
        ``pred_ts`` verbleibt.
    """
    df_true = true_ts.to_dataframe()
    df_pred = pred_ts.to_dataframe()

    target_cols = [
        c for c in df_true.columns
        if pd.Series([c]).str.contains(keyword_pattern, regex=True).iloc[0]
    ]

    if not target_cols:
        raise ValueError(
            "Keine passenden Komponenten für Schriftstücke gefunden. "
            f"Verwendetes Pattern: {keyword_pattern}"
        )

    agg_true = df_true[target_cols].sum(axis=1)
    agg_pred = df_pred[target_cols].sum(axis=1)

    mask = np.isfinite(agg_true.values) & np.isfinite(agg_pred.values)
    if mask.sum() == 0:
        raise ValueError(
            "Nach NaN-Filter keine gemeinsamen Punkte für True und Forecast übrig."
        )

    agg_true = agg_true[mask]
    agg_pred = agg_pred[mask]
    diff = agg_pred - agg_true

    plt.figure(figsize=(12, 6))
    plt.plot(agg_true.index, agg_true.values, label="True (Summe)", linewidth=2)
    plt.plot(agg_pred.index, agg_pred.values, label="Forecast (Summe)", linewidth=2, linestyle="--")
    plt.fill_between(agg_true.index, diff.values, 0, alpha=0.2, label=diff_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    vols = df_true[target_cols].sum(axis=0).sort_values(ascending=False)
    sel_cols = vols.index.tolist()
    if overlay_top_n is not None:
        sel_cols = sel_cols[:overlay_top_n]

    plt.figure(figsize=(12, 6))
    for c in sel_cols:
        plt.plot(df_true.index, df_true[c].values, label=f"True {c}", alpha=0.6)
        if c in df_pred.columns:
            plt.plot(df_pred.index, df_pred[c].values, linestyle="--", label=f"Forecast {c}", alpha=0.6)
    plt.title("Overlay einzelner Schriftstück-Komponenten (Kennzahl TERM_EINGANG_SCHRIFTST)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "true_sum_ts": TimeSeries.from_series(agg_true),
        "pred_sum_ts": TimeSeries.from_series(agg_pred),
        "diff_ts": TimeSeries.from_series(diff),
        "matched_components": target_cols,
        "overlay_components": sel_cols,
    }
