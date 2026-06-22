"""
Run-Reporting und retrospektive Forecast-Evaluation.

Granularität
------------
Die Komponentennamen folgen der Struktur
``KENNZAHL__PRODUKT__STATUS__ORGA`` (z. B.
``TERM_EINGANG_SCHRIFTST__KFZ_Vollkasko__NEU__ORG12``). Das
Reporting zerlegt diese in fünf Dimensionen:

- **Kennzahl** — ``TERM_EINGANG_SCHRIFTST`` / ``TERM_EINGANG_SONST``
- **Produkt**  — ``KFZ_Vollkasko``, ``HUS_Hausrat``, …
- **Status**   — ``NEU`` (Neuschaden) / ``LFD`` (laufend)
- **Orga**     — Organisationseinheit (DIM_ORGA)
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
    ├── retro_accuracy_by_orga.png
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
    │   ├── forecast_vs_actual_by_orga.png
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
    Zerlegt ``KENNZAHL__PRODUKT__STATUS__ORGA`` in ein Dict mit fünf
    Schlüsseln (inkl. abgeleiteter Sparte).

    Falls das Namensformat abweicht, werden Fallback-Werte gesetzt.
    """
    parts = component.split(_SEP)
    if len(parts) == 4:
        kennzahl, produkt, status, orga = parts
    elif len(parts) == 3:
        # Abwärtskompatibel: alter 3-Teiler ohne ORGA.
        kennzahl, produkt, status = parts
        orga = "unbekannt"
    else:
        kennzahl, produkt, status, orga = component, "unbekannt", "unbekannt", "unbekannt"

    # Sparte = erster Teil des Produkts vor dem Unterstrich
    sparte = produkt.split("_")[0] if "_" in produkt else produkt

    return {
        "kennzahl": kennzahl,
        "produkt": produkt,
        "status": status,
        "orga": orga,
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
# Label-Hilfsfunktionen
# ------------------------------------------------------------------

def _readable_name(component: str) -> str:
    """
    Wandelt ``TERM_EINGANG_SCHRIFTST__KFZ_Vollkasko__NEU__ORG12``
    in ``Schriftst · KFZ Vollkasko · Neu · ORG12`` um.
    """
    parts = component.split(_SEP)
    if len(parts) != 4:
        return component

    kz = parts[0].replace("TERM_EINGANG_", "").capitalize()
    produkt = parts[1].replace("_", " ")
    # Status case-insensitiv und sowohl für Kurzcodes (NEU/LFD) als auch
    # die langen Wörter (Neuschaden/Folgebearbeitung) abbilden. Unbekannte
    # Codes werden roh angezeigt statt fälschlich auf "Folge" geworfen.
    s = parts[2].strip().upper()
    if s.startswith("NEU"):
        status = "Neu"
    elif s.startswith("LFD") or s.startswith("FOLGE"):
        status = "Folge"
    else:
        status = parts[2]
    orga = parts[3]
    return f"{kz} · {produkt} · {status} · {orga}"


def _readable_produkt(component: str) -> str:
    """Extrahiert den Produktnamen lesbar: ``KFZ Vollkasko``."""
    parts = component.split(_SEP)
    if len(parts) >= 2:
        return parts[1].replace("_", " ")
    return component


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
    """Aggregierte Summe: True vs Forecast + Abweichungsfläche + MAE."""
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
    mae = float(np.mean(np.abs(diff)))
    smape = float(np.mean(
        2 * np.abs(diff) / (np.abs(agg_t) + np.abs(agg_p) + 1e-8)
    ) * 100)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                     gridspec_kw={"height_ratios": [3, 1]},
                                     sharex=True)

    ax1.plot(agg_t.index, agg_t.values, label="Actual", lw=2, color="#2563eb")
    ax1.plot(agg_p.index, agg_p.values, label="Forecast", lw=2, ls="--",
             color="#dc2626")
    ax1.fill_between(agg_t.index, agg_t.values, agg_p.values,
                     alpha=0.1, color="#dc2626")
    ax1.set_title(f"{title}\nMAE = {mae:,.0f}  |  sMAPE = {smape:.1f}%",
                  fontsize=12)
    ax1.set_ylabel("Summe Termineingänge")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Unterer Plot: Abweichung als Balken
    colors = ["#16a34a" if d >= 0 else "#dc2626" for d in diff.values]
    ax2.bar(diff.index, diff.values, color=colors, alpha=0.7, width=5)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Forecast − Actual")
    ax2.grid(True, alpha=0.3)

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
    Gruppiert Komponenten nach Dimension und zeigt je Ausprägung eine
    Subplot-Zeile mit True/Forecast + Abweichung + MAE im Titel.
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

    groups: Dict[str, List[str]] = {}
    for c in common:
        d = _parse_dimensions(c)
        key = d.get(dim, "unbekannt")
        groups.setdefault(key, []).append(c)

    n = len(groups)
    if n == 0:
        return

    dim_labels = {
        "kennzahl": "Kennzahl",
        "produkt": "Produkt",
        "sparte": "Sparte",
        "status": "Status",
        "orga": "Orga",
    }
    dim_label = dim_labels.get(dim, dim.title())

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n + 1), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (label, cols) in zip(axes, sorted(groups.items())):
        agg_t = df_t[cols].sum(axis=1)
        agg_p = df_p[cols].sum(axis=1)
        diff = agg_p - agg_t
        mae = float(np.mean(np.abs(diff)))

        readable = label.replace("_", " ").replace("TERM EINGANG ", "")
        ax.plot(agg_t.index, agg_t.values, label="Actual", lw=1.5,
                color="#2563eb")
        ax.plot(agg_p.index, agg_p.values, label="Forecast", lw=1.5,
                ls="--", color="#dc2626")
        ax.fill_between(agg_t.index, agg_t.values, agg_p.values,
                        alpha=0.08, color="#dc2626")
        ax.set_ylabel(readable, fontsize=10, fontweight="bold")
        ax.annotate(f"MAE = {mae:,.0f}  ({len(cols)} Komp.)",
                    xy=(0.98, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="gray", alpha=0.8))
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"{title} – nach {dim_label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_backtest_heatmap(
    metrics_df: pd.DataFrame,
    path: Path,
    metric: str = "smape",
    title: str = "Backtest sMAPE je Komponente",
) -> None:
    """
    Gruppiertes Balkendiagramm: Produkt × Kennzahl/Status, Wert = Metrik.
    Volle lesbare Namen.
    """
    if metric not in metrics_df.columns or "component" not in metrics_df.columns:
        return

    df = metrics_df[["component", metric]].copy()
    df["readable"] = df["component"].apply(_readable_name)
    df["produkt"] = df["component"].apply(_readable_produkt)
    df = df.sort_values(["produkt", metric], ascending=[True, False])

    n = len(df)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.45)))

    # Farbcodierung nach Sparte
    sparte_colors = {}
    palette = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#8b5cf6",
               "#ec4899", "#06b6d4", "#84cc16"]
    for c in df["component"]:
        sp = _parse_dimensions(c)["sparte"]
        if sp not in sparte_colors:
            sparte_colors[sp] = palette[len(sparte_colors) % len(palette)]

    colors = [sparte_colors[_parse_dimensions(c)["sparte"]]
              for c in df["component"]]

    bars = ax.barh(range(n), df[metric].values, color=colors, alpha=0.85)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df["readable"].values, fontsize=8)
    ax.set_xlabel(f"{metric.upper()} (%)" if "ape" in metric.lower()
                  else metric.upper(), fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")

    # Werte an die Balken schreiben
    for bar, val in zip(bars, df[metric].values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=7, color="#555")

    # Legende: Sparte → Farbe
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=s)
                      for s, c in sparte_colors.items()]
    ax.legend(handles=legend_handles, title="Sparte", fontsize=8,
              title_fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_baseline_comparison(
    comparison_df: pd.DataFrame,
    path: Path,
    title: str = "Modell vs. Baseline",
) -> None:
    """
    Gruppiertes Balkendiagramm: Modell- vs. Baseline-MAE je Komponente,
    mit Δ-Markierung (negativ = Modell besser).
    """
    if comparison_df.empty:
        return

    df = comparison_df.copy()
    df["readable"] = df["component"].apply(_readable_name)
    df = df.sort_values("delta_mae")

    n = len(df)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n * 0.45)),
                              gridspec_kw={"width_ratios": [3, 2]})

    # Links: MAE-Vergleich
    ax = axes[0]
    y_pos = np.arange(n)
    bar_h = 0.35
    ax.barh(y_pos - bar_h / 2, df["model_mae"].values, bar_h,
            label="Modell", color="#2563eb", alpha=0.85)
    ax.barh(y_pos + bar_h / 2, df["base_mae"].values, bar_h,
            label="Baseline", color="#9ca3af", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["readable"].values, fontsize=8)
    ax.set_xlabel("MAE", fontsize=10)
    ax.set_title(f"{title} – MAE", fontsize=11)
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")

    # Rechts: Δ sMAPE (negativ = Modell besser)
    ax = axes[1]
    colors = ["#16a34a" if d < 0 else "#dc2626" for d in df["delta_smape"].values]
    ax.barh(y_pos, df["delta_smape"].values, color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["readable"].values, fontsize=8)
    ax.set_xlabel("Δ sMAPE (Modell − Baseline)", fontsize=10)
    ax.set_title("Δ sMAPE (grün = Modell besser)", fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")

    for i, (d, w) in enumerate(zip(df["delta_smape"].values, df["delta_wape"].values)):
        ax.text(d + (0.5 if d >= 0 else -0.5), i,
                f"Δw={w:+.1f}", va="center", fontsize=7,
                ha="left" if d >= 0 else "right", color="#555")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ======================================================================
# Daten-Überblick (historische DB-Daten, je Run)
# ======================================================================

def _build_dim_map(components: List[str]) -> pd.DataFrame:
    """DataFrame: Index = Komponente, Spalten = Dimensionen (kennzahl, …)."""
    return pd.DataFrame(
        [_parse_dimensions(c) for c in components],
        index=list(components),
    )


def _data_overview_summary(
    df: pd.DataFrame, dim_map: pd.DataFrame
) -> pd.DataFrame:
    """
    Tabellarischer Überblick: je Dimensionswert eine Zeile mit Anzahl
    Komponenten, Gesamtvolumen, Anteil, Aktivitätszeitraum und Anzahl
    aktiver Monate.
    """
    comp_volume = df.sum(axis=0, skipna=True)
    total = float(np.nansum(df.values))
    monthly = df.resample("MS").sum(min_count=1)

    rows: List[dict] = []
    for dim in ["kennzahl", "produkt", "status", "orga"]:
        for val, comps in dim_map.groupby(dim).groups.items():
            cols = list(comps)
            vol = float(np.nansum(comp_volume[cols].values))
            daily_sum = df[cols].sum(axis=1, skipna=True)
            active = daily_sum[daily_sum.abs() > 0]
            first = active.index.min() if len(active) else pd.NaT
            last = active.index.max() if len(active) else pd.NaT
            n_active = int(
                (monthly[cols].sum(axis=1, skipna=True).abs() > 0).sum()
            )
            rows.append({
                "dimension": dim,
                "wert": val,
                "n_komponenten": len(cols),
                "volumen": round(vol, 2),
                "anteil_pct": round(100.0 * vol / total, 2) if total else 0.0,
                "erste_aktivitaet": first.date().isoformat() if pd.notna(first) else "",
                "letzte_aktivitaet": last.date().isoformat() if pd.notna(last) else "",
                "aktive_monate": n_active,
            })
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["dimension", "volumen"], ascending=[True, False]
    ).reset_index(drop=True)


def _barh(ax, series: pd.Series, title: str, topn: int = 25) -> None:
    """Horizontale Balken, nach Volumen sortiert, optional Top-N."""
    s = series.sort_values(ascending=False)
    hidden = max(0, len(s) - topn)
    s = s.head(topn).iloc[::-1]
    ax.barh(range(len(s)), s.values, color="#4C78A8")
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([str(i) for i in s.index], fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    if hidden:
        ax.set_xlabel(f"(weitere {hidden} ausgeblendet)", fontsize=8)


def _plot_data_overview_composite(
    monthly_total: pd.Series,
    vol_by: Dict[str, pd.Series],
    meta: dict,
    path: Path,
) -> None:
    """2×2-Überblick: Zeitverlauf + Verteilung nach Produkt / ORGA / Status."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (0,0) Monatliches Gesamtvolumen über die Zeit + 12M-Mittel
    ax = axes[0, 0]
    ax.fill_between(monthly_total.index, monthly_total.values,
                    color="#4C78A8", alpha=0.35)
    ax.plot(monthly_total.index, monthly_total.values,
            color="#4C78A8", lw=1.0, label="Monatssumme")
    roll = monthly_total.rolling(12, min_periods=1).mean()
    ax.plot(roll.index, roll.values, color="#E45756", lw=2.0,
            label="12-Monats-Mittel")
    ax.set_title("Monatliches Gesamtvolumen (Termineingänge)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    _barh(axes[0, 1], vol_by["produkt"], "Volumen je Produkt")
    _barh(axes[1, 0], vol_by["orga"], "Volumen je Organisationseinheit (ORGA)")
    _barh(axes[1, 1], vol_by["status"], "Volumen je Schadenstatus")

    fig.suptitle(
        f"Daten-Überblick · {meta['date_min']} – {meta['date_max']} · "
        f"{meta['n_components']} Komponenten · "
        f"Gesamtvolumen {meta['total_volume']:,.0f}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _monthly_dim_matrix(
    df: pd.DataFrame, dim_map: pd.DataFrame, dim: str, topn: int = 30
) -> pd.DataFrame:
    """Matrix Dimensionswert (Zeile) × Monat (Spalte) mit Volumen."""
    comp_volume = df.sum(axis=0, skipna=True)
    order = comp_volume.groupby(dim_map[dim]).sum().sort_values(ascending=False)
    values = list(order.head(topn).index)
    monthly = df.resample("MS").sum(min_count=1)
    mat = pd.DataFrame(index=values, columns=monthly.index, dtype=float)
    for v in values:
        cols = list(dim_map.index[dim_map[dim] == v])
        mat.loc[v] = monthly[cols].sum(axis=1, skipna=True).values
    return mat


def _plot_dim_heatmap(ax, mat: pd.DataFrame, title: str) -> None:
    """Heatmap einer Dimensionswert×Monat-Matrix mit Jahres-Ticks."""
    data = mat.to_numpy(dtype=float)
    finite = data[np.isfinite(data) & (data > 0)]
    vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0,
                   vmax=max(vmax, 1.0))
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([str(i) for i in mat.index], fontsize=8)
    # X-Ticks nur an Jahresgrenzen (Januar)
    months = list(mat.columns)
    xticks = [i for i, m in enumerate(months) if m.month == 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels([m.year for m in months if m.month == 1], fontsize=8)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="Volumen/Monat")


def _plot_data_overview_timeline(
    df: pd.DataFrame, dim_map: pd.DataFrame, path: Path
) -> None:
    """Zwei gestapelte Heatmaps: Volumen über Zeit je Produkt bzw. ORGA."""
    mat_p = _monthly_dim_matrix(df, dim_map, "produkt")
    mat_o = _monthly_dim_matrix(df, dim_map, "orga")
    h = max(4.0, 0.4 * (len(mat_p) + len(mat_o)) + 2.0)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, h),
        gridspec_kw={"height_ratios": [len(mat_p) + 1, len(mat_o) + 1]},
    )
    _plot_dim_heatmap(ax1, mat_p, "Volumen über Zeit je Produkt")
    _plot_dim_heatmap(ax2, mat_o, "Volumen über Zeit je ORGA")
    fig.suptitle("Zeitlicher Verlauf nach Dimension (monatlich)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_data_overview(
    df_daily: pd.DataFrame,
    output_dir: "str | Path",
    run_timestamp: Optional[datetime] = None,
    logger=None,
) -> Optional[Path]:
    """
    Erstellt je Lauf einen strukturierten Überblick der historischen
    DB-Daten: Wo (Dimensionen), wann (Zeitverlauf) und wie viel Volumen
    eingegangen ist. Geschrieben werden in ein eigenes Run-Unterverzeichnis:

    - ``data_overview.png``            – Zeitverlauf + Verteilungen
    - ``data_overview_zeitverlauf.png``– Heatmaps Volumen×Zeit (Produkt/ORGA)
    - ``data_overview_summary.csv``    – tabellarischer Überblick je Dimension

    Bei ~5 Jahren Tagesdaten wird zur Übersicht durchgehend auf Monatsebene
    aggregiert; Balken/Heatmaps zeigen die Top-Werte je Dimension.
    """
    if df_daily is None or df_daily.empty:
        if logger is not None:
            logger.warning("Daten-Überblick übersprungen: keine Daten.")
        return None

    ts = run_timestamp or datetime.now()
    ts_label = ts.strftime("%Y-%m-%dT%H-%M")
    run_dir = Path(output_dir) / f"{ts_label}_data_overview"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Nur echte Komponenten (vierteilig) berücksichtigen.
    df = df_daily.sort_index()
    comp_cols = [c for c in df.columns if str(c).count(_SEP) == 3]
    if comp_cols:
        df = df[comp_cols]
    dim_map = _build_dim_map(list(df.columns))

    # Kennzahlen
    total_volume = float(np.nansum(df.values))
    meta = {
        "date_min": df.index.min().date().isoformat(),
        "date_max": df.index.max().date().isoformat(),
        "n_components": df.shape[1],
        "total_volume": total_volume,
    }

    monthly_total = df.resample("MS").sum(min_count=1).sum(axis=1)
    comp_volume = df.sum(axis=0, skipna=True)
    vol_by = {
        d: comp_volume.groupby(dim_map[d]).sum().sort_values(ascending=False)
        for d in ["kennzahl", "produkt", "status", "orga"]
    }

    # Tabellarischer Überblick
    summary = _data_overview_summary(df, dim_map)
    summary.to_csv(run_dir / "data_overview_summary.csv", index=False)

    # Plots
    _plot_data_overview_composite(
        monthly_total, vol_by, meta, run_dir / "data_overview.png"
    )
    _plot_data_overview_timeline(
        df, dim_map, run_dir / "data_overview_zeitverlauf.png"
    )

    if logger is not None:
        per_orga = ", ".join(
            f"{k}={v:,.0f}" for k, v in vol_by["orga"].items()
        )
        logger.info(
            "Daten-Überblick (%s–%s): %d Komponenten, Gesamtvolumen %s. "
            "Volumen je ORGA: %s. Artefakte unter %s",
            meta["date_min"], meta["date_max"], meta["n_components"],
            f"{total_volume:,.0f}", per_orga, run_dir,
        )

    return run_dir


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

        # 7b) Plots — nach Orga
        _plot_backtest_by_dimension(
            true_ts, backtest_ts,
            run_dir / "forecast_vs_actual_by_orga.png",
            dim="orga",
            title=f"Backtest nach Orga – {horizon_name}",
        )

        # 8) Heatmap — sMAPE je Komponente
        _plot_backtest_heatmap(
            metrics_df, run_dir / "components_heatmap.png",
            metric="smape",
            title=f"sMAPE je Komponente – {horizon_name}",
        )

        # 9) Baseline-Vergleich: Modell vs. saisonale Baseline
        try:
            from .evaluation import (
                compare_model_vs_baseline,
                weekly_baseline_mean_last3,
            )
            baseline_ts = weekly_baseline_mean_last3(true_ts, cfg.baseline)
            horizon_weeks = (int(metrics_df["horizon"].iloc[0])
                             if "horizon" in metrics_df.columns else 13)
            comparison_df = compare_model_vs_baseline(
                true_ts, backtest_ts, baseline_ts, horizon_weeks,
            )
            comparison_df = _add_dimensions(comparison_df)
            comparison_df.to_csv(run_dir / "baseline_comparison.csv",
                                 index=False)
            _plot_baseline_comparison(
                comparison_df,
                run_dir / "model_vs_baseline.png",
                title=f"Modell vs. Baseline – {horizon_name}",
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Baseline-Vergleich fehlgeschlagen: %s", e
            )

        # 10) Globale Metriken-Historie
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
    for dim in ("kennzahl", "produkt", "sparte", "orga"):
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
    ax.plot(by_lead.index, by_lead["mae"], marker="o", ms=5, lw=2,
            label="MAE", color="#2563eb")
    ax.plot(by_lead.index, by_lead["rmse"], marker="s", ms=5, lw=2,
            label="RMSE", color="#dc2626")
    ax.set_xlabel("Vorlauf (Wochen)", fontsize=10)
    ax.set_ylabel("Absoluter Fehler", fontsize=10)
    ax.set_title("MAE / RMSE nach Vorlauf", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(by_lead.index, by_lead["smape"], marker="o", ms=5, lw=2,
            color="#16a34a")
    ax.set_xlabel("Vorlauf (Wochen)", fontsize=10)
    ax.set_ylabel("sMAPE (%)", fontsize=10)
    ax.set_title("sMAPE nach Vorlauf", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(by_lead.index, by_lead["mape"], marker="o", ms=5, lw=2,
            color="#f59e0b")
    ax.set_xlabel("Vorlauf (Wochen)", fontsize=10)
    ax.set_ylabel("MAPE (%)", fontsize=10)
    ax.set_title("MAPE nach Vorlauf", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Retrospektive Forecast-Genauigkeit (alle Komponenten)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_retro_by_dimension(
    retro_df: pd.DataFrame,
    dim: str,
    path: Path,
) -> None:
    """MAE nach Vorlauf, aufgesplittet nach einer Dimension, mit lesbaren Labels."""
    if dim not in retro_df.columns:
        return

    dim_labels = {
        "kennzahl": "Kennzahl",
        "produkt": "Produkt",
        "sparte": "Sparte",
        "status": "Status",
        "orga": "Orga",
    }
    dim_label = dim_labels.get(dim, dim.title())

    groups = retro_df.groupby([dim, "lead_weeks"])["abs_error"].mean().reset_index()
    dim_values = sorted(groups[dim].unique())

    fig, ax = plt.subplots(figsize=(14, 6))
    for val in dim_values:
        sub = groups[groups[dim] == val].sort_values("lead_weeks")
        readable = val.replace("_", " ").replace("TERM EINGANG ", "")
        ax.plot(sub["lead_weeks"], sub["abs_error"], marker="o", ms=4,
                label=readable, alpha=0.8, lw=1.5)

    ax.set_xlabel("Vorlauf (Wochen)", fontsize=10)
    ax.set_ylabel("MAE", fontsize=10)
    ax.set_title(f"Retrospektive MAE nach Vorlauf – nach {dim_label}",
                 fontsize=12)
    ax.legend(fontsize=9, ncol=2 if len(dim_values) > 4 else 1)
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
    Volle lesbare Namen mit Farbcodierung nach Sparte.
    """
    pivot = retro_df.pivot_table(
        index="component",
        columns="lead_weeks",
        values=metric,
        aggfunc="mean",
    )
    if pivot.empty:
        return

    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]

    readable = [_readable_name(c) for c in pivot.index]

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.8),
                                     max(6, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([int(x) for x in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(readable, fontsize=8)

    # Werte in die Zellen schreiben
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > pivot.values[~np.isnan(pivot.values)].mean() else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    ax.set_xlabel("Vorlauf (Wochen)", fontsize=10)
    ax.set_title("Retrospektive MAE: Komponente × Vorlauf", fontsize=12)
    fig.colorbar(im, ax=ax, label="MAE", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ======================================================================
# Tuned-Parameter Persistierung
# ======================================================================

def save_tuned_params(
    runs_dir: str | Path,
    horizon_name: str,
    train_length_weeks: int,
    tft_cfg: "TftConfig",
    best_params: Optional[Dict] = None,
    best_score: Optional[float] = None,
    input_chunk_length: Optional[int] = None,
) -> Path:
    """
    Speichert die getuneten Parameter als YAML im ``runs/``-Verzeichnis.

    Die Datei heißt ``tuned_<horizon_name>.yaml`` und wird bei jedem
    Tuning-Lauf überschrieben — es zählt immer das letzte Ergebnis.

    Parameters
    ----------
    runs_dir
        ``runs/``-Verzeichnis.
    horizon_name
        Name des Horizonts (z. B. ``"h13_model"``).
    train_length_weeks
        Effektive Trainingslänge nach Tuning.
    tft_cfg
        Die effektive TftConfig nach Tuning.
    best_params
        Optuna ``best_params``-Dict (wird als Referenz mitgespeichert).
    best_score
        Bester Optuna-Metrikwert.

    Returns
    -------
    Path
        Pfad zur geschriebenen YAML-Datei.
    """
    import dataclasses as dc

    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "horizon_name": horizon_name,
        "train_length_weeks": train_length_weeks,
        "tft": {f.name: getattr(tft_cfg, f.name) for f in dc.fields(tft_cfg)},
        "tuning_timestamp": datetime.now().strftime("%Y-%m-%dT%H-%M"),
    }
    if input_chunk_length is not None:
        payload["input_chunk_length"] = int(input_chunk_length)
    if best_params is not None:
        payload["optuna_best_params"] = best_params
    if best_score is not None:
        payload["optuna_best_score"] = float(best_score)

    path = runs_dir / f"tuned_{horizon_name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, default_flow_style=False, allow_unicode=True)

    return path


def load_tuned_params(
    runs_dir: str | Path,
    horizon_name: str,
) -> Optional[Dict]:
    """
    Lädt die zuletzt gespeicherten getuneten Parameter für einen Horizont.

    Parameters
    ----------
    runs_dir
        ``runs/``-Verzeichnis.
    horizon_name
        Name des Horizonts.

    Returns
    -------
    dict | None
        Dict mit ``"train_length_weeks"`` und ``"tft"``-Unterbaum,
        oder ``None`` wenn keine gespeicherten Parameter existieren.
    """
    path = Path(runs_dir) / f"tuned_{horizon_name}.yaml"
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
