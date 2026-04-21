"""
Future-Kovariaten aus Bundesländer-Ferien-ICS-Dateien.

Das Modul liest alle ``.ics``-Dateien unter einem Wurzelverzeichnis, mappt
die Ferien-Events auf die betroffenen deutschen Bundesländer und erzeugt
eine wöchentliche Kovariaten-Zeitreihe (binäre Flags je Bundesland plus
einige Aggregate) für den Trainings- und Prognosezeitraum.
"""

import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from icalendar import Calendar
from darts import TimeSeries

from .config import GlobalConfig


#: Mapping Freitext → ISO-3166-2-Code (ohne ``DE-``-Präfix). Deckt die
#: üblichen Schreibweisen aus ICS-Summaries und -Descriptions ab.
NAME2STATE: Dict[str, str] = {
    "baden-wuerttemberg": "BW",
    "baden-wurttemberg": "BW",
    "baden wuerttemberg": "BW",
    "baden wurttemberg": "BW",
    "baden-württemberg": "BW",
    "württemberg": "BW",
    "wuerttemberg": "BW",
    "baden": "BW",
    "bayern": "BY",
    "berlin": "BE",
    "brandenburg": "BB",
    "bremen": "HB",
    "hamburg": "HH",
    "hessen": "HE",
    "mecklenburg-vorpommern": "MV",
    "mecklenburg vorpommern": "MV",
    "mecklenburgvorpommern": "MV",
    "vorpommern": "MV",
    "niedersachsen": "NI",
    "nordrhein-westfalen": "NW",
    "nordrhein westfalen": "NW",
    "nordrheinwestfalen": "NW",
    "nrw": "NW",
    "rheinland-pfalz": "RP",
    "rheinland pfalz": "RP",
    "rheinlandpfalz": "RP",
    "saarland": "SL",
    "sachsen-anhalt": "ST",
    "sachsen anhalt": "ST",
    "sachsenanhalt": "ST",
    "sachsen": "SN",
    "schleswig-holstein": "SH",
    "schleswig holstein": "SH",
    "schleswigholstein": "SH",
    "schleswig": "SH",
    "holstein": "SH",
    "thueringen": "TH",
    "thuringen": "TH",
    "thüringen": "TH",
}

#: Alle 16 Bundesländer in fester Reihenfolge – dient zugleich als
#: Spalten-Liste der erzeugten Kovariaten-Matrix.
GER_STATES = [
    "BW", "BY", "BE", "BB", "HB", "HH", "HE", "MV",
    "NI", "NW", "RP", "SL", "SN", "ST", "SH", "TH",
]


def _normalize(text: str) -> str:
    """Trimmt Whitespace, lowercased und kollabiert Mehrfach-Whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _state_from_text(text: str) -> List[str]:
    """Extrahiert Bundesländer-Codes aus einem freien Text."""
    n = _normalize(text)
    return sorted({code for key, code in NAME2STATE.items() if key in n})


def _find_ics_files(root: Path) -> List[Path]:
    """Liefert alle ``.ics``-Pfade unter ``root`` (rekursiv, sortiert)."""
    if root.is_file() and root.suffix.lower() == ".ics":
        return [root]
    files: List[Path] = []
    for p, _, fnames in os.walk(root):
        for fn in fnames:
            if fn.lower().endswith(".ics"):
                files.append(Path(p) / fn)
    return sorted(set(files))


def _parse_ics_file(path: Path) -> List[Tuple[date, date, List[str]]]:
    """
    Parst eine ICS-Datei und liefert Ferien-Events als
    ``(start_date, end_date, bundesland_codes)``.

    Events ohne erkennbares Bundesland werden übersprungen.
    """
    events: List[Tuple[date, date, List[str]]] = []

    with open(path, "rb") as f:
        cal = Calendar.from_ical(f.read())

    for comp in cal.walk():
        if comp.name != "VEVENT":
            continue

        summary = str(comp.get("summary", ""))
        description = str(comp.get("description", ""))
        states = _state_from_text(summary + " " + description)
        if not states:
            continue

        dtstart = comp.get("dtstart").dt
        dtend = comp.get("dtend").dt

        if isinstance(dtstart, datetime):
            dtstart = dtstart.date()
        if isinstance(dtend, datetime):
            dtend = dtend.date()

        if dtend <= dtstart:
            dtend = dtstart

        events.append((dtstart, dtend, states))

    return events


def build_covariates_all_states(
    df: pd.DataFrame,
    freq: str,
    ics_root_dir: str,
    future_weeks: int = 0,
) -> TimeSeries:
    """
    Erzeugt wöchentliche Ferien-Kovariaten für den Zeitraum von ``df``
    plus optionaler Verlängerung in die Zukunft.

    Features (pro Zeile):

    - 16 binäre Flags je Bundesland (``BW``, ``BY``, …),
    - ``holidays_any_state`` – 1, wenn irgendein Bundesland in der Woche
      Ferien hatte,
    - ``vac_DE_count`` – Anzahl Bundesländer mit Ferien in der Woche
      (Summe über Tage),
    - ``vac_BY`` – Bayern-Flag separat (häufig starkes Signal im
      PLUTO-Kontext),
    - ``holidays_workdays_BY`` – Bayern-Ferientage, die auf Mo–Fr fielen,
    - ``holidays_workdays_DE_count`` – analoge Summe über alle Bundesländer,
    - ``year_turn_any`` – Marker für die ersten bzw. letzten Tage eines
      Jahres (weniger Geschäftsvorfälle).

    Parameters
    ----------
    df
        DataFrame mit DatetimeIndex. Nur Start- und Endzeit des Index
        werden genutzt.
    freq
        Ziel-Frequenz (z. B. ``"W-SUN"``).
    ics_root_dir
        Wurzelverzeichnis, unter dem rekursiv nach ``.ics``-Dateien gesucht
        wird.
    future_weeks
        Zahl zusätzlicher ``freq``-Perioden, um die der Ausgabe-Index über
        das letzte Datum in ``df`` hinaus verlängert wird. Wird typischerweise
        auf den Prognosehorizont gesetzt, damit
        ``model.predict(n=horizon, future_covariates=...)`` ohne Fehler
        durchläuft.

    Returns
    -------
    TimeSeries
        Multivariate wöchentliche Kovariaten-Serie. Der Index reicht vom
        ersten resampleten Datum aus ``df`` bis einschließlich
        ``df.index.max() + future_weeks * freq``.

    Raises
    ------
    ValueError
        Wenn ``df`` keinen ``DatetimeIndex`` hat.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "DataFrame muss einen DatetimeIndex haben, um Covariates zu erzeugen."
        )

    start_date = df.index.min().date()
    end_date = df.index.max().date()

    freq_offset = pd.tseries.frequencies.to_offset(freq)
    future_end_ts = df.index.max() + freq_offset * max(0, int(future_weeks))
    daily_end_date = max(pd.Timestamp(end_date), future_end_ts).date()

    # Tägliche Basismatrix: 1 je Bundesland je Tag, in dem Ferien liegen.
    idx = pd.date_range(start=start_date, end=daily_end_date, freq="D")
    cov_daily = pd.DataFrame(0, index=idx, columns=GER_STATES, dtype="int8")

    idx_lo = cov_daily.index[0]
    idx_hi = cov_daily.index[-1]

    for path in _find_ics_files(Path(ics_root_dir)):
        for dstart, dend, states in _parse_ics_file(path):
            for d in pd.date_range(dstart, dend, freq="D"):
                if d < idx_lo or d > idx_hi:
                    continue
                for s in states:
                    if s in cov_daily.columns:
                        cov_daily.loc[d, s] = 1

    cov_daily["holidays_any_state"] = (cov_daily[GER_STATES].sum(axis=1) > 0).astype("int8")
    cov_daily["vac_DE_count"] = cov_daily[GER_STATES].sum(axis=1).astype("int16")
    cov_daily["vac_BY"] = cov_daily["BY"].astype("int8")

    is_workday = cov_daily.index.to_series().dt.weekday < 5
    cov_daily["holidays_workdays_BY"] = ((cov_daily["BY"] == 1) & is_workday).astype("int8")
    cov_daily["holidays_workdays_DE_count"] = (
        cov_daily[GER_STATES].multiply(is_workday, axis=0).sum(axis=1)
    ).astype("int16")

    dayofyear = cov_daily.index.to_series().dt.dayofyear
    cov_daily["year_turn_any"] = ((dayofyear <= 7) | (dayofyear >= 359)).astype("int8")

    # Aggregation auf Zielfrequenz (Wochensummen für Counts, max für Flags).
    agg_dict = {state: "max" for state in GER_STATES}
    agg_dict.update(
        {
            "holidays_any_state": "max",
            "vac_DE_count": "sum",
            "vac_BY": "max",
            "holidays_workdays_BY": "sum",
            "holidays_workdays_DE_count": "sum",
            "year_turn_any": "max",
        }
    )
    cov_weekly = cov_daily.resample(freq).agg(agg_dict)

    # Ziel-Index deckt Trainingsbereich UND Prognosezeitraum ab.
    resampled_start = df.resample(freq).asfreq().index[0]
    target_index = pd.date_range(start=resampled_start, end=future_end_ts, freq=freq)
    cov_weekly = cov_weekly.reindex(target_index).fillna(0)

    return TimeSeries.from_dataframe(cov_weekly, freq=freq)


def build_weekly_covariates(
    df: pd.DataFrame,
    cfg: GlobalConfig,
    future_weeks: int = 0,
) -> TimeSeries:
    """
    Convenience-Wrapper um :func:`build_covariates_all_states` mit
    Parametern aus der :class:`GlobalConfig`.

    Parameters
    ----------
    df
        DataFrame mit DatetimeIndex.
    cfg
        Globale Konfiguration (entnimmt ``data.freq`` und
        ``data.ics_root_dir``).
    future_weeks
        Zahl zusätzlicher Perioden in die Zukunft (typ. Prognosehorizont).

    Returns
    -------
    TimeSeries
        Siehe :func:`build_covariates_all_states`.
    """
    return build_covariates_all_states(
        df=df,
        freq=cfg.data.freq,
        ics_root_dir=cfg.data.ics_root_dir,
        future_weeks=future_weeks,
    )
