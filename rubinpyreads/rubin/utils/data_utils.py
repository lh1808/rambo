from __future__ import annotations

from typing import Dict
import json
import os


# String-artige Dtypes über pandas-Major-Versionen hinweg:
# pandas 2: Strings sind "object". pandas 3 (String-Migration): eigener "str"-Dtype;
# select_dtypes/Vergleiche mit "object" treffen str-Spalten dort nicht mehr (bzw. nur
# noch über einen Deprecation-Shim). Diese Helfer kapseln die Äquivalenzklasse.
_OBJECT_LIKE_NAMES = frozenset({"object", "str", "string"})


def object_like_selectors() -> list:
    """Dtype-Selektoren für string-artige Spalten, passend zur laufenden pandas-Version.

    Für select_dtypes(include=/exclude=): unter pandas>=3 zusätzlich "str",
    unter pandas 2 nur "object" ("str" ist dort kein gültiger Selektor)."""
    import pandas as pd
    sels = ["object"]
    try:
        if int(pd.__version__.split(".")[0]) >= 3:
            sels.append("str")
    except Exception:
        pass
    return sels


def is_object_like_dtype(dtype) -> bool:
    """True für object-/str-/string-Dtypes (pandas-2/3-übergreifend)."""
    return str(dtype) in _OBJECT_LIKE_NAMES


def available_cpu_count() -> int:
    """Ermittelt die tatsächlich nutzbaren CPUs für diesen Prozess.

    os.cpu_count() gibt die Gesamt-CPUs des Hosts zurück — in Containern
    (Docker, K8s, Devbox) kann das deutlich mehr sein als dem Prozess
    tatsächlich zur Verfügung steht.

    Prüft in dieser Reihenfolge:
    1. cgroup v2 cpu.max (Container CPU-Quota)
    2. cgroup v1 cpu.cfs_quota_us (ältere Container)
    3. os.sched_getaffinity (CPU-Affinität des Prozesses)
    4. os.cpu_count() als Fallback

    Gibt das Minimum aller verfügbaren Werte zurück."""
    candidates = []

    # 1. cgroup v2: /sys/fs/cgroup/cpu.max → "quota period" oder "max period"
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
            if parts[0] != "max":
                quota_cpus = int(parts[0]) / int(parts[1])
                if quota_cpus > 0:
                    candidates.append(int(quota_cpus))
    except Exception:
        pass

    # 2. cgroup v1: cpu.cfs_quota_us / cpu.cfs_period_us
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            p = int(f.read().strip())
        if q > 0 and p > 0:
            candidates.append(q // p)
    except Exception:
        pass

    # 3. sched_getaffinity: Auf welchen CPUs darf der Prozess laufen?
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass

    # 4. os.cpu_count() als Fallback
    candidates.append(os.cpu_count() or 1)

    return max(1, min(candidates))

import numpy as np
import pandas as pd


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Reduziert den Speicherbedarf eines DataFrames per best-effort Downcast.

    - Integer-Spalten → kleinstmöglicher int-Typ (int8/int16/int32/int64)
    - Float-Spalten ohne NaN, deren Werte ganzzahlig sind → int-Typ
    - Übrige Float-Spalten → float32 (nicht float16: zu geringe Präzision)
    - Kategorie-Spalten werden NICHT angetastet (category-Dtype essentiell
      für patch_categorical_features).
    """
    df = df.copy()
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if isinstance(col_type, pd.CategoricalDtype):
            continue
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(col_type):
                df[col] = _smallest_int(df[col], c_min, c_max)
            else:
                # Float-Spalte: Prüfen ob alle Werte ganzzahlig und kein NaN
                has_nan = df[col].isna().any()
                if not has_nan:
                    vals = df[col].values
                    is_int_valued = np.all(np.equal(np.mod(vals, 1), 0))
                    if is_int_valued:
                        df[col] = _smallest_int(df[col].astype(np.int64), c_min, c_max)
                        continue
                # float16 wird bewusst nicht verwendet: nur ~3 Dezimalstellen
                # Genauigkeit, was bei feinen Score-Unterschieden zu
                # Informationsverlust und Qualitätseinbußen führen kann.
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    df.attrs["memory_usage_mb_before"] = float(start_mem)
    df.attrs["memory_usage_mb_after"] = float(end_mem)
    return df


def _smallest_int(series: "pd.Series", c_min: float, c_max: float) -> "pd.Series":
    """Konvertiert eine Series zum kleinstmöglichen int-Typ."""
    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
        return series.astype(np.int8)
    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
        return series.astype(np.int16)
    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
        return series.astype(np.int32)
    return series.astype(np.int64)


def load_dtypes_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fill_missing_categories(X: "pd.DataFrame", columns=None, logger=None) -> list:
    """Repräsentiert NaN in kategorischen (category-/object-)Spalten als
    explizite Kategorie "fehlend".

    Hintergrund: CatBoost akzeptiert keine None/NaN in cat_features — der
    Pool-Aufbau crasht mit "must be real number, not NoneType" (via
    categorical_patch._wrapped_fit). Für nominale Merkmale ist "fehlend" zudem
    fachlich eine eigene Ausprägung und erscheint so konsistent in den
    Explainability-Plots. Numerische Spalten bleiben unberührt (LightGBM/
    CatBoost behandeln numerische NaN nativ).

    Parameters
    ----------
    columns : optionale Eingrenzung; Default = alle category-/object-Spalten.
    Returns: Liste der konvertierten Spaltennamen (für Logging/Tests).
    """
    import pandas as pd
    cols = list(columns) if columns is not None else list(X.columns)
    converted = []
    for col in cols:
        if col not in X.columns:
            continue
        if not (isinstance(X[col].dtype, pd.CategoricalDtype) or is_object_like_dtype(X[col].dtype)):
            continue
        if not X[col].isna().any():
            continue
        ser = X[col].astype("category")
        if "fehlend" not in ser.cat.categories:
            ser = ser.cat.add_categories(["fehlend"])
        X[col] = ser.fillna("fehlend")
        converted.append(col)
    if converted and logger is not None:
        logger.info(
            "Kategorische Spalten mit fehlenden Werten: %s → als explizite "
            "Kategorie 'fehlend' repräsentiert (CatBoost-Kompatibilität).",
            converted,
        )
    return converted

def decode_bytes_categories(X: "pd.DataFrame", columns=None, logger=None) -> list:
    """Dekodiert bytes-Werte in kategorischen (category-/object-)Spalten zu str
    (UTF-8, errors="replace").

    SAS-/pyreadstat-Importe liefern Textspalten teils als bytes (b'M\xc3\xbcnchener').
    Ohne Dekodierung erscheinen solche Ausprägungen als "b'...'"-Labels in
    Explainability-Plots und Encoding-Maps von Spalten ohne NaN; mit Umlauten
    crashte zudem der category-astype(str)-Pfad im Preprocessing. Dekodierung
    erfolgt effizient: bei category-Dtype nur über das Kategorien-Array
    (rename_categories), bei object nur elementweise für bytes-Werte.
    Returns: Liste der dekodierten Spaltennamen.
    """
    import pandas as pd
    def _dec(v):
        return v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else v
    cols = list(columns) if columns is not None else list(X.columns)
    converted = []
    for col in cols:
        if col not in X.columns:
            continue
        ser = X[col]
        if isinstance(ser.dtype, pd.CategoricalDtype):
            cats = ser.cat.categories
            if any(isinstance(c, (bytes, bytearray)) for c in cats):
                decoded = [_dec(c) for c in cats]
                if len(set(decoded)) == len(decoded):
                    X[col] = ser.cat.rename_categories(decoded)
                else:
                    # Kollisions-Randfall: bytes- UND str-Repräsentation derselben
                    # Ausprägung als getrennte Kategorien (b'Münchener' + 'Münchener')
                    # → rename_categories würde an Duplikaten scheitern. Über den
                    # object-Pfad dekodieren und Kategorien dabei vereinigen.
                    X[col] = ser.astype("object").map(_dec).astype("category")
                converted.append(col)
        elif is_object_like_dtype(ser.dtype):
            sample = ser.dropna().head(50)
            if any(isinstance(v, (bytes, bytearray)) for v in sample):
                X[col] = ser.map(_dec)
                converted.append(col)
    if converted and logger is not None:
        logger.info(
            "Bytes-Textspalten dekodiert (UTF-8): %s",
            converted[:10] if len(converted) <= 10 else f"{converted[:5]}... (+{len(converted)-5})",
        )
    return converted
