"""
Run-Reporting und retrospektive Forecast-Evaluation.

Granularität
------------
Die Komponentennamen folgen der Struktur ``KENNZAHL__PRODUKT__STATUS``
(z. B. ``TERM_EINGANG_SCHRIFTST__KFZ_Vollkasko__Neuschaden``). Das
Reporting zerlegt diese in vier Dimensionen:

- **Kennzahl** — ``TERM_EINGANG_SCHRIFTST`` / ``TERM_EINGANG_SONST``
- **Produkt**  — ``KFZ_Vollkasko``, ``HUS_Hausrat``, …
- **Status**   — ``Neuschaden`` / ``Folgebearbeitung``
- **Sparte**   — erster Teil des Produkts (``KFZ`` / ``HUS``)

Alle Auswertungen (Backtest-Metriken, retrospektive Accuracy) liegen
sowohl als CSV auf Einzelkomponenten-Ebene als auch als Plots aggregiert
nach diesen Dimensionen vor.

Metriken
--------
MAE, RMSE, MAPE und sMAPE werden durchgängig berechnet und geplottet.

Ordnerstruktur
--------------
::

    runs/
    ├── metrics_history.csv
    ├── retrospective_accuracy.csv
    ├── retro_accuracy_by_lead_global.png
    ├── retro_accuracy_by_kennzahl.png
    ├── retro_accuracy_by_produkt.png
    ├── retro_accuracy_by_sparte.png
    ├── retro_heatmap_mae.png
    ├── 2025-04-21T08-30_h13_model/
    │   ├── run_metadata.json
    │   ├── config.yaml
    │   ├── metrics.csv
    │   ├── backtest_forecast.csv
    │   ├── future_forecast.csv
    │   ├── forecast_vs_actual_global.png
    │   ├── forecast_vs_actual_by_kennzahl.png
    │   ├── forecast_vs_actual_by_produkt.png
    │   └── components_heatmap.png
    └── …
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from darts import TimeSeries

from .config import GlobalConfig

# ------------------------------------------------------------------
# Dimension-Parsing
# ------------------------------------------------------------------

_SEP = "__"


def _parse_dimensions(component: str) -> Dict[str, str]:
    """
    Zerlegt ``KENNZAHL__PRODUKT__STATUS`` in ein Dict mit vier Schlüsseln.

    Falls das Namensformat abweicht, werden Fallback-Werte gesetzt.
    """
    parts = component.split(_SEP)
    if len(parts) == 3:
        kennzahl, produkt, status = parts
    else:
        kennzahl, produkt, status = component, "unbekannt", "unbekannt"

    # Sparte = erster Teil des Produkts vor dem Unterstrich
    sparte = produkt.split("_")[0] if "_" in produkt else produkt

    return {
        "kennzahl": kennzahl,
        "produkt": produkt,
        "status": status,
        "sparte": sparte,
    }


def _add_dimensions(df: pd.DataFrame, component_col: str = "component") -> pd.DataFrame:
    """Fügt Dimensions-Spalten basierend auf dem Komponentennamen hinzu."""
    dims = df[component_col].apply(_parse_dimensions).apply(pd.Series)
    return pd.concat([df, dims], axis=1)


# ------------------------------------------------------------------
# Metriken-Hilfsfunktionen
# ------------------------------------------------------------------

def _smape_scalar(pred: float, actual: float) -> float:
    denom = (abs(pred) + abs(actual)) / 2.0
    if denom == 0:
        return np.nan
    return abs(pred - actual) / denom * 100.0


# ------------------------------------------------------------------
# Backtest-Plots (pro Run)
# ------------------------------------------------------------------

def _backtest_metrics_with_dims(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Erweitert die Backtest-Metriken um Dimensions-Spalten."""
    return _add_dimensions(metrics_df.copy())


def _plot_backtest_global(
    true_ts: TimeSeries,
    pred_ts: TimeSeries,
    path: Path,
    title: str,
) -> None:
    """Aggregierte Summe: True vs Forecast + Abweichungsfläche."""
    true_a = true_ts.slice_intersect(pred_ts)
    pred_a = pred_ts.slice_intersect(true_ts)
    if len(true_a) == 0 or len(pred_a) == 0:
        return

    df_t = true_a.to_dataframe()
    df_p = pred_a.to_dataframe()
    common = [c for c in df_t.columns if c in df_p.columns]
    if not common:
        return

    agg_t = df_t[common].sum(axis=1)
    agg_p = df_p[common].sum(axis=1)
    diff = agg_p - agg_t

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(agg_t.index, agg_t.values, label="Actual", lw=2)
    ax.plot(agg_p.index, agg_p.values, label="Forecast", lw=2, ls="--")
    ax.fill_between(agg_t.index, diff.values, 0, alpha=0.15, label="Abweichung")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_backtest_by_dimension(
    true_ts: TimeSeries,
    pred_ts: TimeSeries,
    path: Path,
    dim: str,
    title: str,
) -> None:
    """
    Gruppiert Komponenten nach Dimension (kennzahl/produkt/sparte) und
    zeigt je Ausprägung eine Subplot-Zeile.
    """
    true_a = true_ts.slice_intersect(pred_ts)
    pred_a = pred_ts.slice_intersect(true_ts)
    if len(true_a) == 0 or len(pred_a) == 0:
        return

    df_t = true_a.to_dataframe()
    df_p = pred_a.to_dataframe()
    common = [c for c in df_t.columns if c in df_p.columns]
    if not common:
        return

    # Spalten nach Dimension gruppieren
    groups: Dict[str, List[str]] = {}
    for c in common:
        d = _parse_dimensions(c)
        key = d.get(dim, "unbekannt")
        groups.setdefault(key, []).append(c)

    n = len(groups)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (label, cols) in zip(axes, sorted(groups.items())):
        agg_t = df_t[cols].sum(axis=1)
        agg_p = df_p[cols].sum(axis=1)
        ax.plot(agg_t.index, agg_t.values, label="Actual", lw=1.5)
        ax.plot(agg_p.index, agg_p.values, label="Forecast", lw=1.5, ls="--")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_backtest_heatmap(
    metrics_df: pd.DataFrame,
    path: Path,
    metric: str = "smape",
    title: str = "Backtest sMAPE je Komponente",
) -> None:
    """Heatmap: eine Zeile je Komponente, Wert = Metrik."""
    if metric not in metrics_df.columns or "component" not in metrics_df.columns:
        return

    df = metrics_df[["component", metric]].copy()
    df = df.sort_values(metric, ascending=False)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
    colors = plt.cm.YlOrRd(df[metric].values / max(df[metric].max(), 1))
    ax.barh(range(len(df)), df[metric].values, color=colors)
    ax.set_yticks(range(len(df)))
    short_names = [c.split(_SEP)[-1] if _SEP in c else c for c in df["component"]]
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel(metric.upper())
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ======================================================================
# RunReporter
# ======================================================================

class RunReporter:
    """
    Archiviert Metriken, Forecasts, Config und Plots für einen Lauf.

    Parameters
    ----------
    output_dir
        Basisverzeichnis (Default ``"runs"``).
    run_timestamp
        Zeitstempel. ``None`` → jetzt.
    """

    def __init__(
        self,
        output_dir: str = "runs",
        run_timestamp: Optional[datetime] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_timestamp = run_timestamp or datetime.now()
        self._ts_label = self.run_timestamp.strftime("%Y-%m-%dT%H-%M")

    def report_horizon(
        self,
        horizon_name: str,
        metrics_df: pd.DataFrame,
        backtest_ts: TimeSeries,
        future_forecast_ts: TimeSeries,
        true_ts: TimeSeries,
        cfg: GlobalConfig,
        data_end_date: Optional[pd.Timestamp] = None,
    ) -> Path:
        """
        Speichert alle Artefakte eines Prognosehorizonts.

        Returns
        -------
        Path
            Pfad zum Run-Ordner.
        """
        run_dir = self.output_dir / f"{self._ts_label}_{horizon_name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if data_end_date is None:
            data_end_date = true_ts.end_time()

        # 1) Metriken mit Dimensions-Spalten
        metrics_ext = _backtest_metrics_with_dims(metrics_df)
        metrics_ext["run_timestamp"] = self._ts_label
        metrics_ext.to_csv(run_dir / "metrics.csv", index=False)

        # 2) Forecasts
        backtest_ts.to_dataframe().to_csv(run_dir / "backtest_forecast.csv")
        future_forecast_ts.to_dataframe().to_csv(run_dir / "future_forecast.csv")

        # 3) Metadaten
        metadata = {
            "run_timestamp": self._ts_label,
            "data_end_date": str(data_end_date.date()),
            "horizon_name": horizon_name,
            "horizon_weeks": int(metrics_df["horizon"].iloc[0])
                if "horizon" in metrics_df.columns else 0,
        }
        with open(run_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # 4) Config
        self._save_config_snapshot(cfg, run_dir / "config.yaml")

        # 5) Plots — global
        _plot_backtest_global(
            true_ts, backtest_ts,
            run_dir / "forecast_vs_actual_global.png",
            f"Backtest Global – {horizon_name} ({self._ts_label})",
        )

        # 6) Plots — nach Kennzahl
        _plot_backtest_by_dimension(
            true_ts, backtest_ts,
            run_dir / "forecast_vs_actual_by_kennzahl.png",
            dim="kennzahl",
            title=f"Backtest nach Kennzahl – {horizon_name}",
        )

        # 7) Plots — nach Produkt
        _plot_backtest_by_dimension(
            true_ts, backtest_ts,
            run_dir / "forecast_vs_actual_by_produkt.png",
            dim="produkt",
            title=f"Backtest nach Produkt – {horizon_name}",
        )

        # 8) Heatmap — sMAPE je Komponente
        _plot_backtest_heatmap(
            metrics_df, run_dir / "components_heatmap.png",
            metric="smape",
            title=f"sMAPE je Komponente – {horizon_name}",
        )

        # 9) Globale Metriken-Historie
        self._append_metrics_history(metrics_ext)

        return run_dir

    def _save_config_snapshot(self, cfg: GlobalConfig, path: Path) -> None:
        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {f.name: _to_dict(getattr(obj, f.name))
                        for f in dataclasses.fields(obj)}
            if isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_dict(v) for v in obj]
            return obj

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(_to_dict(cfg), f, default_flow_style=False,
                      allow_unicode=True)

    def _append_metrics_history(self, metrics_df: pd.DataFrame) -> None:
        history_path = self.output_dir / "metrics_history.csv"
        write_header = not history_path.exists()
        metrics_df.to_csv(history_path, mode="a", header=write_header,
                          index=False)


# ======================================================================
# Retrospektive Forecast-Evaluation
# ======================================================================

def evaluate_past_forecasts(
    actuals_df: pd.DataFrame,
    runs_dir: str | Path,
) -> pd.DataFrame:
    """
    Vergleicht archivierte Zukunfts-Prognosen mit eingetroffenen
    Ist-Werten.

    Für jede Komponente und jeden Vorlauf werden MAE, RMSE, MAPE und
    sMAPE berechnet. Die Komponenten werden in Dimensionen (Kennzahl,
    Produkt, Status, Sparte) zerlegt, damit beliebige Aggregationen
    möglich sind.

    Returns
    -------
    pd.DataFrame
        Spalten: ``run_timestamp``, ``horizon_model``, ``forecast_date``,
        ``component``, ``kennzahl``, ``produkt``, ``status``, ``sparte``,
        ``predicted``, ``actual``, ``error``, ``abs_error``, ``pct_error``,
        ``smape``, ``lead_weeks``.
    """
    runs_dir = Path(runs_dir)
    records: List[Dict] = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        forecast_path = run_dir / "future_forecast.csv"
        metadata_path = run_dir / "run_metadata.json"
        if not forecast_path.exists() or not metadata_path.exists():
            continue

        with open(metadata_path) as f:
            metadata = json.load(f)

        data_end = pd.Timestamp(metadata["data_end_date"])
        forecast_df = pd.read_csv(forecast_path, index_col=0,
                                  parse_dates=True)

        common_dates = forecast_df.index.intersection(actuals_df.index)
        if len(common_dates) == 0:
            continue

        for date in common_dates:
            lead_weeks = max(1, int(round((date - data_end).days / 7)))
            for col in forecast_df.columns:
                if col not in actuals_df.columns:
                    continue
                pred = float(forecast_df.loc[date, col])
                actual = float(actuals_df.loc[date, col])
                if pd.isna(pred) or pd.isna(actual):
                    continue

                abs_err = abs(pred - actual)
                denom_mape = abs(actual) if actual != 0 else np.nan
                pct_err = (abs_err / denom_mape * 100.0
                           if not np.isnan(denom_mape) else np.nan)
                smape_val = _smape_scalar(pred, actual)

                dims = _parse_dimensions(col)
                records.append({
                    "run_timestamp": metadata["run_timestamp"],
                    "horizon_model": metadata.get("horizon_name", ""),
                    "forecast_date": date,
                    "component": col,
                    **dims,
                    "predicted": pred,
                    "actual": actual,
                    "error": pred - actual,
                    "abs_error": abs_err,
                    "pct_error": pct_err,
                    "smape": smape_val,
                    "lead_weeks": lead_weeks,
                })

    return pd.DataFrame(records)


def update_retrospective_report(
    actuals_df: pd.DataFrame,
    runs_dir: str | Path,
) -> Optional[pd.DataFrame]:
    """
    Führt die retrospektive Evaluation aus und erzeugt:

    - ``retrospective_accuracy.csv`` — granulare Rohdaten.
    - ``retro_accuracy_by_lead_global.png`` — MAE/RMSE/sMAPE nach
      Vorlauf (global über alle Komponenten).
    - ``retro_accuracy_by_kennzahl.png`` — MAE nach Vorlauf, pro
      Kennzahl.
    - ``retro_accuracy_by_produkt.png`` — MAE nach Vorlauf, pro Produkt.
    - ``retro_accuracy_by_sparte.png`` — MAE nach Vorlauf, pro Sparte.
    - ``retro_heatmap_mae.png`` — Heatmap Komponente × Vorlauf.

    Returns
    -------
    pd.DataFrame | None
        Rohdaten der Evaluation, oder ``None`` wenn nichts zu evaluieren.
    """
    runs_dir = Path(runs_dir)
    retro_df = evaluate_past_forecasts(actuals_df, runs_dir)

    if retro_df.empty:
        return None

    retro_df.to_csv(runs_dir / "retrospective_accuracy.csv", index=False)

    # Global: Multi-Metrik nach Vorlauf
    _plot_retro_global(retro_df, runs_dir / "retro_accuracy_by_lead_global.png")

    # Nach Dimension
    for dim in ("kennzahl", "produkt", "sparte"):
        if dim in retro_df.columns:
            _plot_retro_by_dimension(
                retro_df, dim,
                runs_dir / f"retro_accuracy_by_{dim}.png",
            )

    # Heatmap: Komponente × Vorlauf → MAE
    _plot_retro_heatmap(retro_df, runs_dir / "retro_heatmap_mae.png")

    return retro_df


# ------------------------------------------------------------------
# Retro-Plots
# ------------------------------------------------------------------

def _plot_retro_global(retro_df: pd.DataFrame, path: Path) -> None:
    """Multi-Metrik-Plot: MAE, RMSE, sMAPE nach Vorlauf."""
    by_lead = retro_df.groupby("lead_weeks").agg(
        mae=("abs_error", "mean"),
        rmse=("abs_error", lambda x: np.sqrt(np.mean(x ** 2))),
        smape=("smape", "mean"),
        mape=("pct_error", "mean"),
    ).sort_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(by_lead.index, by_lead["mae"], marker="o", ms=4, label="MAE")
    ax.plot(by_lead.index, by_lead["rmse"], marker="s", ms=4, label="RMSE")
    ax.set_xlabel("Vorlauf (Wochen)")
    ax.set_ylabel("Absolute Fehler")
    ax.set_title("MAE / RMSE nach Vorlauf")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(by_lead.index, by_lead["smape"], marker="o", ms=4,
            color="C2", label="sMAPE %")
    ax.set_xlabel("Vorlauf (Wochen)")
    ax.set_ylabel("sMAPE %")
    ax.set_title("sMAPE nach Vorlauf")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(by_lead.index, by_lead["mape"], marker="o", ms=4,
            color="C3", label="MAPE %")
    ax.set_xlabel("Vorlauf (Wochen)")
    ax.set_ylabel("MAPE %")
    ax.set_title("MAPE nach Vorlauf")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Retrospektive Forecast-Genauigkeit (alle Komponenten)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_retro_by_dimension(
    retro_df: pd.DataFrame,
    dim: str,
    path: Path,
) -> None:
    """MAE nach Vorlauf, aufgesplittet nach einer Dimension."""
    if dim not in retro_df.columns:
        return

    groups = retro_df.groupby([dim, "lead_weeks"])["abs_error"].mean().reset_index()
    dim_values = sorted(groups[dim].unique())

    fig, ax = plt.subplots(figsize=(14, 6))
    for val in dim_values:
        sub = groups[groups[dim] == val].sort_values("lead_weeks")
        ax.plot(sub["lead_weeks"], sub["abs_error"], marker="o", ms=3,
                label=val, alpha=0.8)

    ax.set_xlabel("Vorlauf (Wochen)")
    ax.set_ylabel("MAE")
    ax.set_title(f"Retrospektive MAE nach Vorlauf – gruppiert nach {dim.title()}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_retro_heatmap(
    retro_df: pd.DataFrame,
    path: Path,
    metric: str = "abs_error",
) -> None:
    """
    Heatmap: Komponente (Y) × Vorlauf in Wochen (X) → mittlerer Fehler.
    """
    pivot = retro_df.pivot_table(
        index="component",
        columns="lead_weeks",
        values=metric,
        aggfunc="mean",
    )
    if pivot.empty:
        return

    # Sortieren nach mittlerem Fehler (höchster oben)
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]

    # Kurznamen für die Y-Achse
    short_names = []
    for c in pivot.index:
        parts = c.split(_SEP)
        short_names.append(parts[-1] if len(parts) == 3 else c)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.6),
                                     max(5, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([int(x) for x in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Vorlauf (Wochen)")
    ax.set_title("Retrospektive MAE: Komponente × Vorlauf")
    fig.colorbar(im, ax=ax, label="MAE", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
