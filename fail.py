"""
Zyklische Datums-Features für kausale Analysestrecken (X / Effekt-Heterogenität).

Bundesweiter Use Case: Feiertage werden nicht binär kodiert, sondern als
bevölkerungsgewichteter Anteil der Bundesländer, in denen der Tag gesetzlicher
Feiertag ist. Fronleichnam ist damit ~0.64 und nicht 0 oder 1.

Designprinzipien (production-safe):
  * Stateless: reine Funktion des Datums, kein fit(), kein Trainings-State.
  * Beschränkt: jedes Feature liegt in [-1, 1] bzw. [0, 1]. Kein Trend, keine
    Jahreszahl, kein Zeitindex -> keine Extrapolation ausserhalb des
    Trainingsbereichs.
  * Deterministisch und batch-invariant: eine einzelne Zeile liefert exakt
    denselben Vektor wie dieselbe Zeile im Batch.
  * Stabile Spaltenreihenfolge für Serving.

Abhängigkeit: holidays (Version pinnen, s. _REQUIRED_HOLIDAYS_NOTE).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Union

import numpy as np
import pandas as pd

try:
    import holidays as _holidays
except ImportError:  # pragma: no cover
    _holidays = None

_REQUIRED_HOLIDAYS_NOTE = (
    "pip install 'holidays==0.102' - Version pinnen. Neue Releases korrigieren "
    "Kalender teils rückwirkend; ohne Pin driften Trainings- und Serving-Features "
    "auseinander."
)

# Bevölkerung je Bundesland in Mio., gerundet (Destatis, Stand 31.12.2023).
# Nur als Gewicht verwendet und intern normiert - die zweite Nachkommastelle
# ist für den Feature-Wert irrelevant. Bei Bedarf durch die tatsächliche
# Bestandsverteilung (Kunden je Bundesland) ersetzen, s. `weights`.
POPULATION_WEIGHTS: dict[str, float] = {
    "NW": 18.14, "BY": 13.44, "BW": 11.28, "NI": 8.14,
    "HE": 6.39,  "RP": 4.16,  "SN": 4.09,  "BE": 3.76,
    "SH": 2.95,  "BB": 2.57,  "ST": 2.17,  "TH": 2.13,
    "HH": 1.89,  "MV": 1.63,  "SL": 0.99,  "HB": 0.70,
}

# Anteil der bayerischen Bevölkerung in Gemeinden mit überwiegend katholischer
# Bevölkerung -> dort ist Mariä Himmelfahrt gesetzlicher Feiertag.
CATHOLIC_SHARE_BY = 0.56

# Präfix aller generierten Spalten. Als Konstante, damit Downstream-Code
# (Feature-Auswahl, X/W-Split, Serving-Schema) nicht auf einen String-Literal
# angewiesen ist.
TIME_FEATURE_PREFIX = "TIME_FEAT_"


def time_feature_columns(
    df: pd.DataFrame, prefix: str = TIME_FEATURE_PREFIX
) -> list[str]:
    """Spaltennamen der Zeitfeatures in `df`, in stabiler Reihenfolge."""
    return [c for c in df.columns if c.startswith(prefix)]


def make_date_features(
    dates: Union[pd.Series, pd.DatetimeIndex, np.ndarray, list],
    *,
    harmonics: int = 2,
    weekly: bool = True,
    monthly: bool = True,
    holidays_mode: str = "population_share",
    subdiv: Union[pd.Series, None] = None,
    holiday_distance: bool = False,
    horizon: int = 14,
    weights: Union[dict[str, float], None] = None,
    prefix: str = TIME_FEATURE_PREFIX,
    index: Union[pd.Index, None] = None,
) -> pd.DataFrame:
    """
    Baut zyklische + kalendarische Features aus einem Datum (yyyy-mm-dd).

    Parameters
    ----------
    dates : datum-artige Serie/Array, wird via pd.to_datetime geparst.
    harmonics : Anzahl Harmonischer der Jahresperiode (1..3 sinnvoll). k=1 ist
        eine reine Sinuswelle, k>=2 erlaubt asymmetrische Saisonalität.
    weekly, monthly : Wochen- bzw. Monatsperiode zuschalten.
    holidays_mode : Kodierung der Feiertage.
        "population_share" (Default, bundesweit): Anteil der Bevölkerung, für
            die der Tag gesetzlicher Feiertag ist. Bundeseinheitliche Feiertage
            = 1.0, Fronleichnam ~0.64, Buß- und Bettag ~0.05.
        "per_state": exakt je Bundesland, erfordert `subdiv`. Vorzuziehen,
            sobald das Bundesland (z. B. via PLZ) am Vertrag verfügbar ist.
        "nationwide_only": nur die 9 bundeseinheitlichen Feiertage, binär.
        "off": keine Feiertagsfeatures.
    subdiv : Series mit Bundesland-Kürzeln ("NW", "BY", ...), nur für
        holidays_mode="per_state". Muss zum Index von `dates` passen.
    holiday_distance : zusätzlich Abstand zum letzten/nächsten Feiertag,
        auf `horizon` Tage geclippt und normiert. Default aus, um X klein zu
        halten.
    weights : überschreibt POPULATION_WEIGHTS, z. B. mit der tatsächlichen
        Bestandsverteilung {"NW": 0.31, ...}. Wird intern normiert.
    prefix : Spaltenpräfix, default TIME_FEATURE_PREFIX ("TIME_FEAT_"),
        damit die Features in der Feature-Auswahl als Block erkennbar sind.
        Siehe time_feature_columns() zum Herausfiltern.
    index : optionaler Index für das Ergebnis (default: Index der Eingabe).

    Returns
    -------
    pd.DataFrame, float64, garantiert ohne NaN.
    """
    if harmonics < 1:
        raise ValueError("harmonics muss >= 1 sein")
    valid_modes = {"population_share", "per_state", "nationwide_only", "off"}
    if holidays_mode not in valid_modes:
        raise ValueError(f"holidays_mode muss in {sorted(valid_modes)} liegen")
    if holidays_mode == "per_state" and subdiv is None:
        raise ValueError("holidays_mode='per_state' erfordert subdiv")

    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    if index is not None:
        d.index = index
    if d.isna().any():
        raise ValueError(
            f"{int(d.isna().sum())} nicht parsebare Datumswerte (NaT). Vor dem "
            "Feature-Building explizit behandeln - NaN in X macht den CATE "
            "undefiniert."
        )

    out: dict[str, np.ndarray] = {}

    # --- Jahresperiode, schaltjahrkorrekt ------------------------------------
    days_in_year = np.where(d.dt.is_leap_year, 366.0, 365.0)
    theta_y = 2.0 * np.pi * (d.dt.dayofyear.to_numpy() - 1) / days_in_year
    for k in range(1, harmonics + 1):
        suf = "" if k == 1 else str(k)
        out[f"{prefix}year_sin{suf}"] = np.sin(k * theta_y)
        out[f"{prefix}year_cos{suf}"] = np.cos(k * theta_y)

    # --- Wochenperiode --------------------------------------------------------
    dow = d.dt.dayofweek.to_numpy()
    if weekly:
        theta_w = 2.0 * np.pi * dow / 7.0
        out[f"{prefix}dow_sin"] = np.sin(theta_w)
        out[f"{prefix}dow_cos"] = np.cos(theta_w)
        out[f"{prefix}is_weekend"] = (dow >= 5).astype(float)

    # --- Monatsposition (Abbuchungs-/Zahlungszyklen) --------------------------
    if monthly:
        dom = d.dt.day.to_numpy()
        dim = d.dt.days_in_month.to_numpy()
        theta_m = 2.0 * np.pi * (dom - 1) / dim
        out[f"{prefix}dom_sin"] = np.sin(theta_m)
        out[f"{prefix}dom_cos"] = np.cos(theta_m)
        out[f"{prefix}is_month_start"] = (dom <= 3).astype(float)
        out[f"{prefix}is_month_end"] = (dim - dom <= 2).astype(float)

    # --- Feiertage ------------------------------------------------------------
    if holidays_mode != "off":
        for name, values in _holiday_features(
            d, dow, holidays_mode, subdiv, holiday_distance, horizon, weights
        ).items():
            out[f"{prefix}{name}"] = values

    df = pd.DataFrame(out, index=d.index).astype("float64")
    assert df.abs().to_numpy().max() <= 1.0 + 1e-9, "unbeschränktes Feature erzeugt"
    return df


def add_date_features(
    df: pd.DataFrame,
    date_col: str,
    *,
    drop_date_col: bool = False,
    overwrite: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Hängt die Zeitfeatures an einen DataFrame an und gibt eine Kopie zurück.

    Der Eingabe-DataFrame wird nicht verändert. Die neuen Spalten werden
    positionsbasiert angehängt, nicht über den Index gejoint - damit
    funktioniert das auch bei doppelten oder unsortierten Index-Werten, wo ein
    join/merge still falsch alignen oder abbrechen würde.

    Parameters
    ----------
    df : Eingabe-DataFrame.
    date_col : Name der Datumsspalte (yyyy-mm-dd, str oder datetime).
    drop_date_col : Datumsspalte im Ergebnis entfernen. Default False - das
        Rohdatum ist für Splits, Diagnostik und Zeitfenster-Auswertungen nützlich,
        gehört aber nicht in X (nicht beschränkt, extrapoliert nicht).
    overwrite : bestehende TIME_FEAT_-Spalten überschreiben statt Fehler werfen.
    **kwargs : werden an make_date_features durchgereicht (harmonics, weights,
        holidays_mode, subdiv, holiday_distance, prefix, ...).

    Returns
    -------
    pd.DataFrame : alle ursprünglichen Spalten plus die neuen TIME_FEAT_-Spalten.
    """
    if date_col not in df.columns:
        raise KeyError(
            f"Spalte '{date_col}' nicht im DataFrame. Vorhanden: {list(df.columns)[:20]}"
        )

    raw = df[date_col]
    parsed = pd.to_datetime(raw, errors="coerce")
    if parsed.isna().any() and not raw.isna().all():
        bad = raw[parsed.isna() & raw.notna()].unique()[:5]
        n_bad = int((parsed.isna()).sum())
        raise ValueError(
            f"'{date_col}': {n_bad} Werte nicht als Datum parsebar bzw. leer. "
            f"Beispiele: {list(bad)}. Vor dem Feature-Building klären - NaN in X "
            "macht den CATE undefiniert."
        )

    # subdiv ggf. als Spaltenname zulassen und positionsbasiert übergeben
    subdiv = kwargs.get("subdiv")
    if isinstance(subdiv, str) and subdiv in df.columns:
        kwargs["subdiv"] = df[subdiv].reset_index(drop=True)
    elif isinstance(subdiv, pd.Series):
        kwargs["subdiv"] = subdiv.reindex(df.index).reset_index(drop=True)

    feats = make_date_features(parsed.reset_index(drop=True), **kwargs)

    collisions = [c for c in feats.columns if c in df.columns]
    if collisions and not overwrite:
        raise ValueError(
            f"Spalten existieren bereits: {collisions}. "
            "overwrite=True setzen oder ein anderes prefix= wählen."
        )

    out = df.drop(columns=collisions) if collisions else df.copy()
    # Blockweise Zuweisung: positionsbasiert, kein Index-Alignment, keine
    # Fragmentierungs-Warnung durch Einzelzuweisungen.
    out[list(feats.columns)] = feats.to_numpy()

    if drop_date_col:
        out = out.drop(columns=[date_col])
    return out


# --------------------------------------------------------------------------- #
# Feiertagslogik
# --------------------------------------------------------------------------- #

def _require_holidays() -> None:
    if _holidays is None:
        raise ImportError("Paket 'holidays' nicht installiert. " + _REQUIRED_HOLIDAYS_NOTE)


@lru_cache(maxsize=256)
def _calendar(subdiv: Union[str, None], y0: int, y1: int, catholic: bool) -> tuple:
    """Sortiertes Tupel der Feiertage einer Region. Gecacht: der Aufbau des
    Kalenders dominiert sonst die Laufzeit bei zeilenweisem Scoring."""
    _require_holidays()
    categories = ("public", "catholic") if catholic else ("public",)
    cal = _holidays.country_holidays(
        "DE", subdiv=subdiv, years=range(y0, y1 + 1), categories=categories
    )
    return tuple(sorted(cal.keys()))


def _to_day_array(dates) -> np.ndarray:
    return pd.to_datetime(pd.Series(list(dates))).to_numpy().astype("datetime64[D]")


def _regions(y0: int, y1: int, weights: Union[dict, None]) -> list[tuple[float, np.ndarray]]:
    """Zerlegt Deutschland in gewichtete Teilpopulationen mit je eigenem
    Kalender. Bayern wird gesplittet, weil Mariä Himmelfahrt nur in den
    überwiegend katholischen Gemeinden gesetzlicher Feiertag ist."""
    w = dict(weights or POPULATION_WEIGHTS)
    unknown = set(w) - set(POPULATION_WEIGHTS)
    if unknown:
        raise ValueError(f"unbekannte Bundesland-Kürzel: {sorted(unknown)}")

    regions: list[tuple[float, np.ndarray]] = []
    for state, weight in w.items():
        if weight < 0:
            raise ValueError(f"negatives Gewicht für {state}")
        if state == "BY":
            regions.append(
                (weight * (1.0 - CATHOLIC_SHARE_BY),
                 _to_day_array(_calendar("BY", y0, y1, False)))
            )
            regions.append(
                (weight * CATHOLIC_SHARE_BY,
                 _to_day_array(_calendar("BY", y0, y1, True)))
            )
        else:
            regions.append((weight, _to_day_array(_calendar(state, y0, y1, False))))

    total = sum(r[0] for r in regions)
    if total <= 0:
        raise ValueError("Summe der Gewichte ist 0")
    return [(weight / total, cal) for weight, cal in regions]


def _bridge_mask(day: np.ndarray, dow: np.ndarray, cal: np.ndarray) -> np.ndarray:
    """Brückentag: Werktag, der zwischen Feiertag und Wochenende eingeklemmt ist
    (Freitag nach Donnerstags-Feiertag, Montag vor Dienstags-Feiertag)."""
    one = np.timedelta64(1, "D")
    is_hol = np.isin(day, cal)
    return (
        ~is_hol
        & (dow < 5)
        & (
            (np.isin(day - one, cal) & (dow == 4))
            | (np.isin(day + one, cal) & (dow == 0))
        )
    )


def _holiday_features(
    d: pd.Series,
    dow: np.ndarray,
    mode: str,
    subdiv: Union[pd.Series, None],
    with_distance: bool,
    horizon: int,
    weights: Union[dict, None],
) -> dict[str, np.ndarray]:
    _require_holidays()
    day = d.to_numpy().astype("datetime64[D]")
    y0, y1 = int(d.dt.year.min()) - 1, int(d.dt.year.max()) + 1
    n = len(d)

    if mode == "population_share":
        share = np.zeros(n)
        bridge = np.zeros(n)
        for weight, cal in _regions(y0, y1, weights):
            share += weight * np.isin(day, cal)
            bridge += weight * _bridge_mask(day, dow, cal)

    elif mode == "per_state":
        sub = pd.Series(subdiv).reindex(d.index)
        if sub.isna().any():
            raise ValueError("subdiv enthält fehlende Bundesländer")
        share = np.zeros(n)
        bridge = np.zeros(n)
        for state, positions in sub.groupby(sub).groups.items():
            pos = d.index.get_indexer(positions)
            catholic = str(state) == "BY"  # Mariä Himmelfahrt konservativ mitnehmen
            cal = _to_day_array(_calendar(str(state), y0, y1, catholic))
            share[pos] = np.isin(day[pos], cal).astype(float)
            bridge[pos] = _bridge_mask(day[pos], dow[pos], cal).astype(float)

    else:  # nationwide_only
        cal = _to_day_array(_calendar(None, y0, y1, False))
        share = np.isin(day, cal).astype(float)
        bridge = _bridge_mask(day, dow, cal).astype(float)

    share = np.clip(share, 0.0, 1.0)
    bridge = np.clip(bridge, 0.0, 1.0)
    res = {
        "holiday_share": share,
        "is_holiday_nationwide": (share >= 0.999).astype(float),
        "bridge_share": bridge,
        # Anteil der Bevölkerung, für die der Tag ein regulärer Arbeitstag ist.
        # Redundant für Bäume, aber die relevante Grösse für lineare Final Stages.
        "workday_share": np.where(dow >= 5, 0.0, 1.0 - share),
    }

    if with_distance:
        nat = _to_day_array(_calendar(None, y0, y1, False))
        i = np.searchsorted(nat, day, side="left")
        nxt = np.where(i < len(nat), (nat[np.clip(i, 0, len(nat) - 1)] - day).astype(int), horizon)
        prv = np.where(i > 0, (day - nat[np.clip(i - 1, 0, len(nat) - 1)]).astype(int), horizon)
        res["days_to_holiday"] = np.clip(nxt, 0, horizon) / horizon
        res["days_since_holiday"] = np.clip(prv, 0, horizon) / horizon

    return res
