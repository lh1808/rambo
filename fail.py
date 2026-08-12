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
 
# Mapping auf die 16 ISO-Kürzel. Deckt ausgeschriebene Namen (mit und ohne
# Umlaute), Destatis-Regionalschlüssel und gängige Hausschreibweisen ab.
_SUBDIV_ALIASES: dict[str, str] = {
    "SCHLESWIGHOLSTEIN": "SH", "01": "SH", "1": "SH",
    "HAMBURG": "HH", "02": "HH", "2": "HH",
    "NIEDERSACHSEN": "NI", "03": "NI", "3": "NI", "NDS": "NI",
    "BREMEN": "HB", "04": "HB", "4": "HB", "FREIEHANSESTADTBREMEN": "HB",
    "NORDRHEINWESTFALEN": "NW", "05": "NW", "5": "NW", "NRW": "NW",
    "HESSEN": "HE", "06": "HE", "6": "HE",
    "RHEINLANDPFALZ": "RP", "07": "RP", "7": "RP", "RLP": "RP",
    "BADENWUERTTEMBERG": "BW", "BADENWURTTEMBERG": "BW", "08": "BW", "8": "BW",
    "BAYERN": "BY", "09": "BY", "9": "BY", "BAY": "BY", "FREISTAATBAYERN": "BY",
    "SAARLAND": "SL", "10": "SL", "SAAR": "SL",
    "BERLIN": "BE", "11": "BE",
    "BRANDENBURG": "BB", "12": "BB",
    "MECKLENBURGVORPOMMERN": "MV", "13": "MV", "MECKPOMM": "MV",
    "SACHSEN": "SN", "14": "SN", "FREISTAATSACHSEN": "SN",
    "SACHSENANHALT": "ST", "15": "ST",
    "THUERINGEN": "TH", "THURINGEN": "TH", "16": "TH",
}
 
 
def normalize_subdiv(values: pd.Series) -> pd.Series:
    """
    Normalisiert Bundesland-Angaben auf die 16 ISO-Kürzel ("NW", "BY", ...).
 
    Toleriert Kleinschreibung, Leerzeichen, Bindestriche, Umlaute, ausgeschriebene
    Namen und Destatis-Schlüssel ("09" -> "BY"). Nicht auflösbare Werte werden
    zu NA - sie brechen die Pipeline nicht ab, sondern laufen in den Fallback
    von make_date_features.
    """
    s = values.astype("string")
    key = (
        s.str.strip()
        .str.upper()
        .str.replace("Ä", "AE", regex=False)
        .str.replace("Ö", "OE", regex=False)
        .str.replace("Ü", "UE", regex=False)
        .str.replace("ß", "SS", regex=False)
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )
    out = key.where(key.isin(list(POPULATION_WEIGHTS)), key.map(_SUBDIV_ALIASES))
    return out.where(out.isin(list(POPULATION_WEIGHTS)), pd.NA)
 
 
def time_feature_columns(
    df: pd.DataFrame, prefix: str = TIME_FEATURE_PREFIX
) -> list[str]:
    """Spaltennamen der Zeitfeatures in `df`, in stabiler Reihenfolge."""
    return [c for c in df.columns if c.startswith(prefix)]
 
 
def plz_to_subdiv(
    plz: pd.Series,
    mapping: Union[dict, pd.Series, "pd.DataFrame", str],
    *,
    plz_key: str = "plz",
    subdiv_key: str = "bundesland",
) -> pd.Series:
    """
    Löst Postleitzahlen zu Bundesland-Kürzeln auf.
 
    Die Zuordnungstabelle wird bewusst nicht mitgeliefert: PLZ-Grenzen fallen
    nicht mit Gemeinde- oder Landesgrenzen zusammen, ~90 PLZ liegen in zwei
    Bundesländern. Eine geratene Prefix-Heuristik (erste zwei Stellen) liegt je
    nach Region zweistellig daneben. Nutze die hausinterne Tabelle - in einem
    Versicherungsbestand existiert sie durch die Regionalklassen-Tarifierung
    ohnehin - oder einen offiziellen Datensatz (OpenPLZ, Destatis-GV-ISys).
 
    Parameters
    ----------
    plz : Serie mit Postleitzahlen. int, float oder str; wird auf 5 Stellen
        mit führenden Nullen normalisiert.
    mapping : dict {"01067": "SN", ...}, Series (Index = PLZ), DataFrame mit
        den Spalten plz_key/subdiv_key, oder Pfad zu einer CSV mit diesen
        Spalten.
    plz_key, subdiv_key : Spaltennamen bei DataFrame/CSV.
 
    Returns
    -------
    pd.Series mit normalisierten Kürzeln, NA wo nicht auflösbar.
    """
    if isinstance(mapping, str):
        mapping = pd.read_csv(mapping, dtype=str)
    if isinstance(mapping, pd.DataFrame):
        missing = {plz_key, subdiv_key} - set(mapping.columns)
        if missing:
            raise KeyError(f"Mapping-Tabelle ohne Spalten {sorted(missing)}")
        mapping = mapping.set_index(plz_key)[subdiv_key]
    if isinstance(mapping, dict):
        mapping = pd.Series(mapping)
 
    lookup = pd.Series(
        normalize_subdiv(pd.Series(mapping.to_numpy())).to_numpy(),
        index=normalize_plz(pd.Series(mapping.index.astype(str))).to_numpy(),
    )
    lookup = lookup[~lookup.index.duplicated(keep="first")]
 
    return normalize_plz(plz).map(lookup).astype("string")
 
 
def normalize_plz(plz: pd.Series) -> pd.Series:
    """
    Normalisiert PLZ auf fünfstellige Strings mit führenden Nullen.
 
    Fängt den häufigsten Datenfehler ab: PLZ, die irgendwo als int gelesen
    wurde, verliert die führende Null - aus 01067 (Dresden) wird 1067, und die
    ostdeutschen PLZ-Bereiche 0xxxx fallen bei einem naiven String-Join
    geschlossen aus der Zuordnung. Nicht-fünfstellige Werte werden zu NA.
    """
    s = plz.astype("string").str.strip()
    # float-Artefakte wie "01067.0" bereinigen
    s = s.str.replace(r"\.0+$", "", regex=True)
    # nur reine Ziffernfolgen mit hoechstens 5 Stellen sind gueltige PLZ;
    # alles andere (Auslands-PLZ, Buchstaben, Muell) wird NA statt still zu "00000"
    s = s.where(s.str.fullmatch(r"\d{1,5}", na=False), pd.NA)
    return s.str.zfill(5)
 
 
def make_date_features(
    dates: Union[pd.Series, pd.DatetimeIndex, np.ndarray, list],
    *,
    harmonics: int = 2,
    weekly: bool = True,
    monthly: bool = True,
    holidays_mode: str = "population_share",
    subdiv: Union[pd.Series, None] = None,
    subdiv_fallback: str = "population_share",
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
    subdiv : Series mit Bundesland-Angaben, nur für holidays_mode="per_state".
        Wird über normalize_subdiv() aufgelöst - Kürzel, ausgeschriebene Namen
        und Destatis-Schlüssel ("09") werden erkannt.
    subdiv_fallback : Umgang mit Zeilen ohne auflösbares Bundesland.
        "population_share" (Default): bevölkerungsgewichteter Anteil als
            beste Schätzung. "nationwide": nur bundeseinheitliche Feiertage
            (konservativ). "raise": Abbruch.
        In jedem Fall markiert TIME_FEAT_subdiv_known die betroffenen Zeilen.
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
            d, dow, holidays_mode, subdiv, holiday_distance, horizon, weights,
            subdiv_fallback,
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
    subdiv_fallback: str = "population_share",
) -> dict[str, np.ndarray]:
    _require_holidays()
    day = d.to_numpy().astype("datetime64[D]")
    y0, y1 = int(d.dt.year.min()) - 1, int(d.dt.year.max()) + 1
    n = len(d)
    known = None
 
    if mode == "population_share":
        share = np.zeros(n)
        bridge = np.zeros(n)
        for weight, cal in _regions(y0, y1, weights):
            share += weight * np.isin(day, cal)
            bridge += weight * _bridge_mask(day, dow, cal)
 
    elif mode == "per_state":
        sub = normalize_subdiv(pd.Series(subdiv).reindex(d.index))
        known = sub.notna().to_numpy()
 
        if not known.all():
            if subdiv_fallback == "raise":
                raise ValueError(
                    f"{int((~known).sum())} von {n} Zeilen ohne auflösbares "
                    "Bundesland. Mit normalize_subdiv() prüfen, welche Werte "
                    "betroffen sind, oder subdiv_fallback='population_share' "
                    "bzw. 'nationwide' setzen."
                )
            if subdiv_fallback not in {"population_share", "nationwide"}:
                raise ValueError(
                    "subdiv_fallback muss 'population_share', 'nationwide' "
                    "oder 'raise' sein"
                )
 
        share = np.zeros(n)
        bridge = np.zeros(n)
 
        for state, positions in sub.dropna().groupby(sub.dropna()).groups.items():
            pos = d.index.get_indexer(positions)
            catholic = str(state) == "BY"  # Mariä Himmelfahrt konservativ mitnehmen
            cal = _to_day_array(_calendar(str(state), y0, y1, catholic))
            share[pos] = np.isin(day[pos], cal).astype(float)
            bridge[pos] = _bridge_mask(day[pos], dow[pos], cal).astype(float)
 
        # Zeilen ohne Bundesland: bestmögliche Schätzung statt Abbruch.
        if not known.all():
            miss = ~known
            if subdiv_fallback == "population_share":
                for weight, cal in _regions(y0, y1, weights):
                    share[miss] += weight * np.isin(day[miss], cal)
                    bridge[miss] += weight * _bridge_mask(day[miss], dow[miss], cal)
            else:  # nationwide
                cal = _to_day_array(_calendar(None, y0, y1, False))
                share[miss] = np.isin(day[miss], cal).astype(float)
                bridge[miss] = _bridge_mask(day[miss], dow[miss], cal).astype(float)
 
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
    if mode == "per_state":
        # Stabiles Schema: Flag existiert immer im per_state-Modus, auch wenn
        # im aktuellen Batch kein Wert fehlt. Sonst weicht das Serving-Schema
        # je nach Datenqualität vom Trainings-Schema ab.
        res["subdiv_known"] = known.astype(float)
 
    if with_distance:
        nat = _to_day_array(_calendar(None, y0, y1, False))
        i = np.searchsorted(nat, day, side="left")
        nxt = np.where(i < len(nat), (nat[np.clip(i, 0, len(nat) - 1)] - day).astype(int), horizon)
        prv = np.where(i > 0, (day - nat[np.clip(i - 1, 0, len(nat) - 1)]).astype(int), horizon)
        res["days_to_holiday"] = np.clip(nxt, 0, horizon) / horizon
        res["days_since_holiday"] = np.clip(prv, 0, horizon) / horizon
 
    return res
 
 
if __name__ == "__main__":
    demo = pd.Series([
        "2025-06-19",  # Fronleichnam
        "2025-11-19",  # Buß- und Bettag
        "2025-08-15",  # Mariä Himmelfahrt
        "2025-10-31",  # Reformationstag
        "2025-10-03",  # Tag der Deutschen Einheit
        "2025-05-30",  # Brückentag nach Christi Himmelfahrt
    ])
    feats = make_date_features(demo)
    cols = [c for c in time_feature_columns(feats) if "holiday" in c or "bridge" in c or "workday" in c]
    print(feats[cols].set_axis(demo).round(3).to_string())
 
