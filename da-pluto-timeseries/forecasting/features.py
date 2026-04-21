"""
Preprocessing- und Feature-Engineering-Schritte.

Enthält:

- :func:`preprocess_dataframe` – Indexaufbereitung, optionale Drops/Aggregation,
  Resampling auf Zielfrequenz.
- :func:`augment_X_with_target_past_features_safe` – Lag-, Rolling- und
  YoY-Features auf einer multivariaten Reihe.
- :func:`build_target_and_past_covariates` – Trennung in Zielreihe (nur
  "echte" ``KENNZAHL__PRODUKT__STATUS``-Komponenten) und Past-Kovariaten
  (inkl. Aggregat- und abgeleiteter Features).
- :func:`add_static_covariates` – Statische Kategorien (Kennzahl, Sparte,
  Status) als One-Hot an die Zielreihe anhängen.

Spaltenkonvention
-----------------
Target-Spalten folgen dem Namensmuster ``KENNZAHL__PRODUKT__STATUS``,
z. B. ``TERM_EINGANG_SCHRIFTST__KFZ_Vollkasko__Neuschaden``. Aus dem Namen
werden im :func:`add_static_covariates`-Schritt drei statische Merkmale
(Kennzahl, Sparte, Status) extrahiert.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from sklearn.preprocessing import OneHotEncoder

from .config import GlobalConfig, PreprocessingConfig

#: Shift-Länge für YoY-Vergleich (Wochen). Entspricht einem Jahresabstand bei
#: wöchentlich aggregierten Daten.
YOY_SHIFT_WEEKS = 52


def preprocess_dataframe(df: pd.DataFrame, cfg: GlobalConfig) -> pd.DataFrame:
    """
    Bereitet ein Roh-DataFrame für die Modellierung vor.

    Schritte (in dieser Reihenfolge):

    1. DatetimeIndex sicherstellen (ggf. aus konfigurierter Datumsspalte).
    2. Nach Index sortieren.
    3. ``object``-Spalten numerisch casten (z. B. ``Decimal`` aus DB2 →
       ``float``). Andere Dtypes bleiben unangetastet.
    4. Optional Spalten droppen oder aggregieren
       (``PreprocessingConfig.drop_contains`` / ``aggregate_map``).
    5. Optional Produkt-Aggregatspalten erzeugen
       (``PreprocessingConfig.product_aggregation``).
    6. Auf Zielfrequenz resamplen (Summe, ``min_count=1``).
    7. Verbleibende NaN mit 0 auffüllen.

    Parameters
    ----------
    df
        Roh-DataFrame. Entweder mit DatetimeIndex oder mit einer
        Datumsspalte, deren Name in ``cfg.data.date_column`` steht.
    cfg
        Globale Konfiguration.

    Returns
    -------
    pd.DataFrame
        Aufbereitetes, auf Zielfrequenz aggregiertes DataFrame.

    Raises
    ------
    ValueError
        Wenn weder DatetimeIndex noch Datumsspalte vorhanden sind.
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = cfg.data.date_column
        if date_col is not None and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        else:
            raise ValueError(
                "DataFrame muss entweder einen DatetimeIndex haben oder eine "
                "konfigurierte Datums-Spalte."
            )

    df = df.sort_index()

    # Nur object-Spalten numerisch casten. Bereits numerische Dtypes bleiben
    # erhalten; so werden potenzielle nicht-numerische Fachspalten (falls
    # vorhanden) nicht versehentlich durch das folgende ``fillna(0)`` zu 0
    # gemacht.
    object_cols = df.select_dtypes(include=["object"]).columns
    for c in object_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    prep_cfg = cfg.preprocessing

    if prep_cfg.drop_contains or prep_cfg.aggregate_map:
        df, _ = drop_or_sum(
            df,
            drop_contains=prep_cfg.drop_contains,
            aggregate_map=prep_cfg.aggregate_map,
        )

    if prep_cfg.product_aggregation:
        df = add_product_aggregates_to_df(df, prep_cfg)

    if cfg.data.freq:
        df = df.resample(cfg.data.freq).sum(min_count=1)

    return df.fillna(0)


def drop_or_sum(
    df: pd.DataFrame,
    drop_contains: Optional[Iterable[str]] = None,
    aggregate_map: Optional[Dict[str, Iterable[str]]] = None,
    case_insensitive: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Droppt und/oder aggregiert Spalten nach Namen.

    Parameters
    ----------
    df
        Eingabe-DataFrame.
    drop_contains
        Iterable von Substrings. Jede Spalte, deren Name einen dieser
        Substrings enthält, wird entfernt.
    aggregate_map
        Mapping ``neuer_spaltenname -> [quellspalten]``. Die neuen Spalten
        werden als zeilenweise Summe der existierenden Quellspalten
        erzeugt; fehlende Quellspalten werden ignoriert.
    case_insensitive
        Vergleicht Spaltennamen und Patterns in Kleinbuchstaben.

    Returns
    -------
    (pd.DataFrame, dict)
        Transformiertes DataFrame und ein Log-Dict mit ``"dropped"``
        (Liste gedroppter Spalten) und ``"aggregated"`` (Mapping der
        tatsächlich aggregierten Spalten).
    """
    df = df.copy()
    log_info: Dict = {"dropped": [], "aggregated": {}}

    cols = df.columns.tolist()

    if drop_contains:
        drops = []
        for col in cols:
            name = col.lower() if case_insensitive else col
            if any(pattern.lower() in name for pattern in drop_contains):
                drops.append(col)
        df = df.drop(columns=drops)
        log_info["dropped"] = drops

    if aggregate_map:
        for new_name, col_list in aggregate_map.items():
            existing = [c for c in col_list if c in df.columns]
            if not existing:
                continue
            df[new_name] = df[existing].sum(axis=1)
            log_info["aggregated"][new_name] = existing

    return df, log_info


def add_product_aggregates_to_df(
    df: pd.DataFrame,
    prep_cfg: PreprocessingConfig,
) -> pd.DataFrame:
    """
    Erzeugt pro Produkt eine Aggregatspalte über alle Kennzahlen und
    Statusse dieses Produkts.

    Erwartet Spaltennamen der Form ``KENNZAHL__PRODUKT__STATUS``. Für jedes
    vorkommende Produkt entsteht eine neue Spalte
    ``{product_agg_prefix}{PRODUKT}`` als zeilenweise Summe aller
    passenden numerischen Spalten.

    Parameters
    ----------
    df
        DataFrame, in dessen Spaltennamen bereits die
        ``KENNZAHL__PRODUKT__STATUS``-Struktur erkennbar ist.
    prep_cfg
        Preprocessing-Konfiguration (liefert Trennzeichen,
        Produkt-Index-Position und Prefix).

    Returns
    -------
    pd.DataFrame
        DataFrame mit zusätzlichen Aggregatspalten.
    """
    df = df.copy()
    product_to_cols: Dict[str, List[str]] = {}

    for c in df.columns:
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        parts_main = c.split(prep_cfg.component_main_sep)
        if len(parts_main) <= prep_cfg.static_product_index:
            continue
        product = parts_main[prep_cfg.static_product_index]
        product_to_cols.setdefault(product, []).append(c)

    for product, group_cols in product_to_cols.items():
        new_name = f"{prep_cfg.product_agg_prefix}{product}"
        df[new_name] = df[group_cols].sum(axis=1)

    return df


def augment_X_with_target_past_features_safe(
    X: TimeSeries,
    prep_cfg: PreprocessingConfig,
) -> Tuple[TimeSeries, int]:
    """
    Ergänzt eine multivariate Zeitreihe um Lag-, Rolling- und YoY-Features.

    Für jede Spalte, deren Name zu ``prep_cfg.include_if_name_contains``
    passt (leere Liste = alle), werden je nach Aktivierungsflag erzeugt:

    - Lag-Features: ``<col>_lag<L>`` für jedes ``L`` in ``prep_cfg.lags``.
    - Rolling-Mean: ``<col>_ma<W>``.
    - Rolling-Std: ``<col>_std<W>``.
    - YoY-Feature: ``<col>_yoy`` mit wählbarem Modus (``log``, ``diff``,
      sonst relative Änderung).

    Die ersten ``warmup`` Zeilen werden verworfen, damit die erzeugten
    Features keine NaN mehr enthalten. ``warmup`` ist das Maximum aus
    größtem Lag, größtem Rolling-Fenster, YoY-Shift (wenn aktiv) und
    ``min_periods``.

    Parameters
    ----------
    X
        Eingabezeitreihe.
    prep_cfg
        Preprocessing-Konfiguration mit Flags, Lag-/Window-Listen und
        YoY-Modus.

    Returns
    -------
    (TimeSeries, int)
        Augmentierte Zeitreihe und Anzahl der verworfenen Anfangsperioden.
    """
    include_if_name_contains = prep_cfg.include_if_name_contains
    if isinstance(include_if_name_contains, str):
        include_patterns = [include_if_name_contains]
    else:
        include_patterns = list(include_if_name_contains) if include_if_name_contains else []

    def _match(name: str) -> bool:
        if not include_patterns:
            return True
        candidate = name.lower()
        return any(p.lower() in candidate for p in include_patterns)

    df = X.to_dataframe(copy=True)
    base_cols = list(df.columns)

    lags = prep_cfg.lags
    rolling_windows = prep_cfg.rolling_windows
    if isinstance(lags, int):
        lags = [lags]
    if isinstance(rolling_windows, int):
        rolling_windows = [rolling_windows]

    max_lag = max(lags) if (prep_cfg.add_lags and lags) else 0
    max_window = (
        max(rolling_windows)
        if (prep_cfg.add_ma or prep_cfg.add_std) and rolling_windows
        else 0
    )
    # Die YoY-Verschiebung muss mit in den Warmup einfließen, damit die
    # erzeugten Features nach dem Warmup keine NaN mehr tragen.
    yoy_shift = YOY_SHIFT_WEEKS if prep_cfg.add_yoy else 0
    warmup = max(max_lag, max_window, yoy_shift, prep_cfg.min_periods)

    new_df = df.copy()

    for col in base_cols:
        if not _match(col):
            continue

        if prep_cfg.add_lags and lags:
            for L in lags:
                new_df[f"{col}_lag{L}"] = df[col].shift(L)

        if prep_cfg.add_ma and rolling_windows:
            for W in rolling_windows:
                new_df[f"{col}_ma{W}"] = (
                    df[col].rolling(window=W, min_periods=prep_cfg.min_periods).mean()
                )

        if prep_cfg.add_std and rolling_windows:
            for W in rolling_windows:
                new_df[f"{col}_std{W}"] = (
                    df[col].rolling(window=W, min_periods=prep_cfg.min_periods).std()
                )

        if prep_cfg.add_yoy:
            if prep_cfg.yoy_mode == "diff":
                new_df[f"{col}_yoy"] = df[col] - df[col].shift(YOY_SHIFT_WEEKS)
            elif prep_cfg.yoy_mode == "log":
                with np.errstate(divide="ignore", invalid="ignore"):
                    new_df[f"{col}_yoy"] = np.log1p(df[col]) - np.log1p(
                        df[col].shift(YOY_SHIFT_WEEKS)
                    )
            else:
                new_df[f"{col}_yoy"] = df[col] / df[col].shift(YOY_SHIFT_WEEKS) - 1.0

    new_df = new_df.iloc[warmup:]
    return TimeSeries.from_dataframe(new_df, freq=X.freq), warmup


def build_target_and_past_covariates(
    df: pd.DataFrame,
    cfg: GlobalConfig,
) -> Tuple[TimeSeries, TimeSeries, int]:
    """
    Trennt ein aufbereitetes DataFrame in Zielreihe und Past-Kovariaten.

    Als Target werden nur Spalten verwendet, die dem vollen
    ``KENNZAHL__PRODUKT__STATUS``-Muster entsprechen (exakt zwei Vorkommen
    des Separators). Aggregat- und abgeleitete Spalten wandern in die
    Past-Kovariaten.

    Die Past-Kovariaten werden anschließend um Lag-, Rolling- und
    YoY-Features angereichert; entsprechend wird die Zielreihe um die
    ersten ``warmup`` Zeitschritte gekürzt, damit alle Reihen den gleichen
    Start haben.

    Parameters
    ----------
    df
        Aufbereitetes DataFrame (Ausgabe von :func:`preprocess_dataframe`).
    cfg
        Globale Konfiguration.

    Returns
    -------
    (TimeSeries, TimeSeries, int)
        Zielreihe, augmentierte Past-Kovariaten, Warmup-Länge.

    Raises
    ------
    ValueError
        Wenn das DataFrame keine numerischen Spalten enthält oder keine
        Spalte dem Target-Muster entspricht.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        raise ValueError("Das DataFrame enthält keine numerischen Spalten zur Modellierung.")

    prep_cfg = cfg.preprocessing
    sep = prep_cfg.component_main_sep

    target_cols = [c for c in numeric_cols if isinstance(c, str) and c.count(sep) == 2]
    if len(target_cols) == 0:
        raise ValueError(
            "Es wurden keine gültigen Target-Spalten gefunden, die der Struktur "
            f"KENNZAHL{sep}PRODUKT{sep}STATUS entsprechen."
        )

    ts_y_base = TimeSeries.from_dataframe(df[target_cols], freq=cfg.data.freq)
    ts_X_raw = TimeSeries.from_dataframe(df[numeric_cols], freq=cfg.data.freq)

    ts_X_aug, warmup = augment_X_with_target_past_features_safe(
        ts_X_raw, prep_cfg=cfg.preprocessing
    )

    y_idx = ts_y_base.time_index
    ts_y = ts_y_base.slice(y_idx[warmup], y_idx[-1])

    return ts_y, ts_X_aug, warmup


def extract_static_for_component(
    name: str,
    prep_cfg: PreprocessingConfig,
) -> Tuple[str, str, str]:
    """
    Leitet die statischen Merkmale (Kennzahl, Sparte, Status) aus einem
    Komponenten-Namen ``KENNZAHL__PRODUKT__STATUS`` ab.

    Die "Sparte" ist der erste Teil des Produkts (aufgesplittet am
    ``product_sep``, üblicherweise ``_``). Bei einem Produkt wie
    ``KFZ_Vollkasko`` ergibt das ``KFZ``.

    Parameters
    ----------
    name
        Komponenten-Name.
    prep_cfg
        Preprocessing-Konfiguration mit Index-Positionen und Trennzeichen.

    Returns
    -------
    (str, str, str)
        ``(kennzahl, sparte, status)``.

    Raises
    ------
    ValueError
        Wenn ``name`` das erwartete Muster nicht erfüllt.
    """
    parts_main = name.split(prep_cfg.component_main_sep)
    if len(parts_main) <= max(
        prep_cfg.static_kpi_index,
        prep_cfg.static_product_index,
        prep_cfg.static_status_index,
    ):
        raise ValueError(f"Komponenten-Name passt nicht zur erwarteten Struktur: {name}")

    kpi = parts_main[prep_cfg.static_kpi_index]
    product_full = parts_main[prep_cfg.static_product_index]
    status = parts_main[prep_cfg.static_status_index]

    product_parts = product_full.split(prep_cfg.product_sep)
    if len(product_parts) <= prep_cfg.product_sparte_index:
        sparte = product_full
    else:
        sparte = product_parts[prep_cfg.product_sparte_index]

    return kpi, sparte, status


def add_static_covariates(
    ts_y: TimeSeries,
    cfg: GlobalConfig,
) -> Tuple[TimeSeries, StaticCovariatesTransformer]:
    """
    Hängt pro Komponente drei statische kategoriale Merkmale (Kennzahl,
    Sparte, Status) an und fittet einen One-Hot-Transformer darauf.

    Der gefittete Transformer wird zurückgegeben und muss später für
    Inferenz-Daten per ``.transform()`` wiederverwendet werden, damit die
    One-Hot-Kategorien konsistent zum Trainingsstand bleiben.

    Parameters
    ----------
    ts_y
        Multivariate Zielreihe.
    cfg
        Globale Konfiguration.

    Returns
    -------
    (TimeSeries, StaticCovariatesTransformer)
        Zielreihe mit angehängten One-Hot-Static-Covariates und der
        gefittete Transformer.
    """
    prep_cfg = cfg.preprocessing
    components = ts_y.components

    static_covariates = pd.DataFrame(
        [extract_static_for_component(c, prep_cfg) for c in components],
        columns=["Kennzahl", "Sparte", "Status"],
        index=pd.Index(components),
    )

    # sklearn benannte ``sparse`` in 1.2 zu ``sparse_output`` um – für beide
    # Versionen kompatibel halten.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformer = StaticCovariatesTransformer(transformer_cat=ohe)

    ts_y_with_static = ts_y.with_static_covariates(static_covariates)
    ts_y_encoded = transformer.fit_transform(ts_y_with_static)

    return ts_y_encoded, transformer
