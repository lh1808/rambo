"""
Disaggregation wöchentlicher Prognosen auf Tageswerte.

Hintergrund
-----------
Das Prognosemodell arbeitet bewusst auf Wochenebene (``W-SUN``) – das ist
die statistisch robuste Granularität für Termineingänge. Für die
Schreibtabelle wird aber ein Wert *je Tag* benötigt. Dieses Modul bricht
jeden Wochenwert anhand eines historischen Wochentagsprofils auf die sieben
Tage der jeweiligen Woche herunter, **ohne den Wochenwert zu verändern**:
die Summe der sieben Tageswerte ergibt exakt wieder den Wochenwert.

Die Wochentagsanteile werden je Komponente aus der Tageshistorie geschätzt.
Für dünn besetzte Komponenten greift ein hierarchischer Fallback:

    Komponente  →  Gruppe (Default: Kennzahl)  →  global  →  uniform

So bekommt auch eine kleine ORGA-Zelle ein plausibles Profil, statt an zu
wenig eigener Historie zu scheitern.

Wochentags-Konvention: ``Timestamp.weekday()`` mit Montag=0 … Sonntag=6.

Wichtig: Dieses Modul ist bewusst frei von DB- und Modell-Abhängigkeiten,
damit es isoliert getestet werden kann. Es kennt nur pandas/numpy.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

_WEEKEND = (5, 6)  # Samstag, Sonntag
_UNIFORM = np.full(7, 1.0 / 7.0)


def _default_group_key(component: str) -> str:
    """Gruppenschlüssel für den Fallback: die Kennzahl (erster Namensteil)."""
    return component.split("__")[0]


def _normalize(weekday_sums: np.ndarray) -> Optional[np.ndarray]:
    """Normalisiert ein 7er-Array auf Summe 1, oder ``None`` wenn leer/≤0."""
    total = float(weekday_sums.sum())
    if total <= 0:
        return None
    return weekday_sums / total


def compute_weekday_weights(
    df_daily: pd.DataFrame,
    *,
    window_weeks: Optional[int] = 52,
    min_window_volume: float = 7.0,
    group_key_fn: Optional[Callable[[str], str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, str]]:
    """
    Schätzt je Komponente ein Wochentagsprofil (Anteile, Summe 1) aus der
    Tageshistorie, mit hierarchischem Fallback bei dünner Datenlage.

    Parameters
    ----------
    df_daily
        Tageshistorie: DatetimeIndex (täglich), Spalten = Komponenten.
    window_weeks
        Nur die letzten ``window_weeks`` Wochen werden für die Schätzung
        herangezogen (``None`` = gesamte Historie).
    min_window_volume
        Mindest-Summe (absolut, im Fenster), ab der das *eigene* Profil
        einer Komponente bzw. Gruppe als verlässlich gilt. Darunter wird
        auf die nächst-gröbere Ebene zurückgefallen.
    group_key_fn
        Abbildung Komponente → Gruppenschlüssel für die mittlere
        Fallback-Ebene. Default: Kennzahl (erster ``__``-Teil).
    logger
        Optionaler Logger für eine Zusammenfassung der gewählten Ebenen.

    Returns
    -------
    (weights_df, global_weights, levels)
        ``weights_df``: DataFrame, Index = Komponente, Spalten 0..6,
        Werte = Wochentagsanteile (Summe je Zeile = 1).
        ``global_weights``: 7er-Array, das globale Profil (Fallback für
        unbekannte Komponenten).
        ``levels``: Dict Komponente → genutzte Ebene
        (``"component"`` / ``"group"`` / ``"global"`` / ``"uniform"``).
    """
    if group_key_fn is None:
        group_key_fn = _default_group_key

    if df_daily is None or df_daily.empty:
        return pd.DataFrame(columns=list(range(7))), _UNIFORM.copy(), {}

    df = df_daily.sort_index()
    if window_weeks is not None:
        cutoff = df.index.max() - pd.Timedelta(weeks=window_weeks)
        df = df.loc[df.index > cutoff]

    # Negative Werte ergeben für Anteile keinen Sinn → auf 0 kappen.
    df = df.clip(lower=0)
    weekday_of_row = df.index.weekday.to_numpy()

    # Wochentags-Summen je Komponente (7er-Array).
    comp_sums: Dict[str, np.ndarray] = {}
    for c in df.columns:
        vals = df[c].to_numpy(dtype=float)
        vals = np.nan_to_num(vals, nan=0.0)
        sums = np.array([vals[weekday_of_row == wd].sum() for wd in range(7)])
        comp_sums[c] = sums

    # Globales Profil.
    if comp_sums:
        global_raw = np.sum(list(comp_sums.values()), axis=0)
    else:
        global_raw = np.zeros(7)
    global_weights = _normalize(global_raw)
    if global_weights is None:
        global_weights = _UNIFORM.copy()

    # Gruppen-Profile.
    group_raw: Dict[str, np.ndarray] = {}
    for c, sums in comp_sums.items():
        g = group_key_fn(c)
        group_raw.setdefault(g, np.zeros(7))
        group_raw[g] = group_raw[g] + sums

    # Ebene je Komponente wählen.
    rows: Dict[str, np.ndarray] = {}
    levels: Dict[str, str] = {}
    for c, sums in comp_sums.items():
        if sums.sum() >= min_window_volume:
            w = _normalize(sums)
            rows[c], levels[c] = w, "component"
            continue
        g = group_key_fn(c)
        g_sums = group_raw.get(g, np.zeros(7))
        if g_sums.sum() >= min_window_volume:
            w = _normalize(g_sums)
            if w is not None:
                rows[c], levels[c] = w, "group"
                continue
        # global vs. uniform
        if global_weights is _UNIFORM or np.allclose(global_weights, _UNIFORM):
            rows[c], levels[c] = global_weights.copy(), "uniform"
        else:
            rows[c], levels[c] = global_weights.copy(), "global"

    weights_df = pd.DataFrame(rows).T
    if not weights_df.empty:
        weights_df.columns = list(range(7))

    if logger is not None:
        from collections import Counter
        cnt = Counter(levels.values())
        logger.info(
            "Wochentagsprofile: %d Komponenten (Ebenen: %s)",
            len(levels), dict(cnt),
        )

    return weights_df, global_weights, levels


def _effective_weights(
    component: str,
    weights_df: pd.DataFrame,
    global_weights: np.ndarray,
    weekend_policy: str,
) -> np.ndarray:
    """Wochentagsgewichte einer Komponente inkl. Wochenend-Politik."""
    if component in weights_df.index:
        w = weights_df.loc[component].to_numpy(dtype=float)
    else:
        w = np.asarray(global_weights, dtype=float)
    w = w.copy()

    if weekend_policy == "zero":
        w[list(_WEEKEND)] = 0.0
        s = w.sum()
        if s > 0:
            w = w / s
        else:
            # Kein Werktagssignal → werktags gleichverteilen.
            w = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0])
    elif weekend_policy != "empirical":
        raise ValueError(
            f"Unbekannte weekend_policy: {weekend_policy!r} "
            "(erlaubt: 'empirical', 'zero')."
        )
    return w


def disaggregate_weekly_to_daily(
    weekly_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    global_weights: np.ndarray,
    *,
    weekend_policy: str = "empirical",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Bricht eine wöchentliche Prognose (``W-SUN``) auf Tageswerte herunter.

    Jeder Wochenwert am Label-Tag ``S`` (Sonntag bei ``W-SUN``) wird auf die
    sieben Tage ``S-6 … S`` verteilt. Da die Wochentagsgewichte je Komponente
    auf Summe 1 normiert sind, gilt für jede Woche und Komponente:
    Summe der sieben Tageswerte == Wochenwert.

    Aufeinanderfolgende ``W-SUN``-Label liegen 7 Tage auseinander, die
    Tagesfenster überlappen also nicht – jeder Tag gehört zu genau einer
    Woche (keine Doppelzählung).

    Parameters
    ----------
    weekly_df
        Wöchentliche Prognose: DatetimeIndex (Label-Tage), Spalten =
        Komponenten.
    weights_df, global_weights
        Ausgabe von :func:`compute_weekday_weights`.
    weekend_policy
        ``"empirical"`` (Default): historische Wochenendanteile (bei
        Termineingang i. d. R. nahe 0). ``"zero"``: Sa/So hart auf 0,
        Werktage werden renormiert (Summe bleibt der Wochenwert).
    logger
        Optionaler Logger.

    Returns
    -------
    pd.DataFrame
        Tageswerte: DatetimeIndex (täglich, lückenlos über alle Wochen),
        Spalten = Komponenten in der Reihenfolge von ``weekly_df``.
    """
    if weekly_df is None or weekly_df.empty:
        return pd.DataFrame()

    comps = list(weekly_df.columns)
    weekly_df = weekly_df.sort_index()

    # Effektive Gewichtsmatrix (n_comp, 7).
    W = np.vstack([
        _effective_weights(c, weights_df, global_weights, weekend_policy)
        for c in comps
    ])

    # Lückenlose Tagesachse über alle Wochenfenster.
    last_sunday = pd.Timestamp(weekly_df.index.max())
    first_sunday = pd.Timestamp(weekly_df.index.min())
    all_days = pd.date_range(
        start=first_sunday - pd.Timedelta(days=6),
        end=last_sunday,
        freq="D",
    )
    day_pos = {d: i for i, d in enumerate(all_days)}

    data = np.zeros((len(all_days), len(comps)), dtype=float)

    for s in weekly_df.index:
        s = pd.Timestamp(s)
        days = pd.date_range(end=s, periods=7, freq="D")
        wd = np.array([d.weekday() for d in days])
        wk_vals = weekly_df.loc[s].to_numpy(dtype=float)  # (n_comp,)
        # (n_comp, 7): Wochenwert je Komponente × Wochentagsgewicht
        block = wk_vals[:, None] * W[:, wd]
        for j, d in enumerate(days):
            data[day_pos[d]] = data[day_pos[d]] + block[:, j]

    daily = pd.DataFrame(data, index=all_days, columns=comps)

    if logger is not None:
        logger.info(
            "Disaggregation: %d Wochen × %d Komponenten → %d Tage "
            "(weekend_policy=%s).",
            len(weekly_df), len(comps), len(daily), weekend_policy,
        )

    return daily
