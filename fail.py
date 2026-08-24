import numpy as np
import pandas as pd


def add_versanddatum(
    df: pd.DataFrame,
    start: str = "2027-10-15",
    end: str = "2027-11-26",
    only_business_days: bool = False,
    shuffle: bool = True,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Verteilt Versanddatums gleichmäßig über den gesamten Bestand.

    Rückgabe: Kopie von df mit zusätzlicher Spalte 'versanddatum' (str, yyyy-mm-dd).
    """
    dates = pd.date_range(start, end, freq="B" if only_business_days else "D")
    if len(dates) == 0:
        raise ValueError(f"Kein gültiges Datum im Bereich {start} bis {end}.")

    date_str = dates.strftime("%Y-%m-%d").to_numpy()
    values = np.resize(date_str, len(df))  # zyklisch -> balancierte Häufigkeiten

    if shuffle:
        np.random.default_rng(random_state).shuffle(values)

    out = df.copy()
    out["versanddatum"] = values
    return out
