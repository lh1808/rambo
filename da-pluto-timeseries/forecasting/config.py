"""
Konfigurations-Datenklassen für die PLUTO-Pipeline.

Alle Parameter der Pipeline sind hier als typisierte :mod:`dataclasses`
zusammengefasst. :class:`GlobalConfig` fasst die Einzelteile zusammen und
ist der Einstiegspunkt. Externe Konfiguration (YAML/Env) wird über
:func:`forecasting.config_loader.load_config` auf diese Klassen gemappt.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    """
    Globale Datenkonfiguration.

    freq:
        Frequenz der Zeitreihe (Pandas/Darts). Für wöchentliche Daten z. B. 'W-SUN'.
    date_column:
        Name der Spalte mit Datumsangaben, falls der DataFrame noch keinen DatetimeIndex hat.
        Wenn None, wird kein automatisches Setzen des Index versucht.
    ics_root_dir:
        Wurzelverzeichnis, unter dem Ferien-/Holiday-ICS-Dateien liegen.
    """
    freq: str = "W-SUN"
    date_column: Optional[str] = "date"
    ics_root_dir: str = "/mnt/ferien-api-data-main/resources/de"


@dataclass
class PreprocessingConfig:
    """
    Konfiguration der Vorverarbeitung und des Feature-Engineerings.

    drop_contains:
        Spalten, deren Name einen dieser Strings enthält, werden entfernt.
    aggregate_map:
        Mapping neuer Spaltennamen auf eine Liste existierender Spalten, die zu einer Summe
        aggregiert werden.

    Produkt-Aggregation:
    --------------------
    Die Spaltennamen folgen der Struktur

        KENNZAHL__PRODUKT__STATUS

    z. B. TERM_EINGANG_SCHRIFTST__KFZ_Vollkasko__Neuschaden

    Wenn product_aggregation=True ist, werden für jedes Produkt zusätzliche
    Aggregatspalten erzeugt, in denen über alle Kennzahlen und Statusse
    des jeweiligen Produkts summiert wird:

        AGG_PROD_KFZ_Vollkasko
        AGG_PROD_HUS_Hausrat
        ...
    """
    drop_contains: List[str] = field(default_factory=list)
    aggregate_map: Dict[str, List[str]] = field(default_factory=dict)

    product_aggregation: bool = True
    product_agg_prefix: str = "AGG_PROD_"

    include_if_name_contains: List[str] = field(default_factory=list)

    add_lags: bool = True
    add_yoy: bool = True
    add_ma: bool = True
    add_std: bool = False
    # Längen bewusst kürzer gewählt (nur kurzer Test-Zeitraum verfügbar).
    # Für produktive Läufe mit mehr Historie ggf. [4, 8, 13, 52] verwenden.
    lags: List[int] = field(default_factory=lambda: [4, 8, 13])
    rolling_windows: List[int] = field(default_factory=lambda: [4, 8, 13])
    yoy_mode: str = "log"
    min_periods: int = 4

    component_main_sep: str = "__"
    static_kpi_index: int = 0
    static_product_index: int = 1
    static_status_index: int = 2
    product_sep: str = "_"
    product_sparte_index: int = 0


@dataclass
class TftConfig:
    """
    Globale Hyperparameter für das TFT-Modell (Basiseinstellungen).

    Diese Werte werden verwendet, wenn Tuning deaktiviert ist oder als Startpunkt,
    den das Tuning überschreibt.
    """
    n_epochs: int = 400
    full_attention: bool = False
    num_attention_heads: int = 4
    batch_size: int = 12
    likelihood: Optional[str] = None  # None, 'poisson', 'nb'
    random_state: int = 0
    hidden_size: int = 64
    hidden_continuous_size: int = 16
    lstm_layers: int = 2
    force_reset: bool = True
    dropout: float = 0.3
    use_static_covariates: bool = True
    add_relative_index: bool = True
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    loss: str = "mse"


@dataclass
class HorizonConfig:
    """
    Konfiguration eines einzelnen Prognosehorizonts.

    Attributes
    ----------
    name
        Lesbarer Name (wird in Logs/Metriken mitgeführt, z. B.
        ``"h13_model"``).
    horizon_weeks
        Prognosehorizont in Wochen (Länge des erzeugten Forecasts).
    input_chunk_length
        Input-Fenster des TFT-Modells (Anzahl vorgelagerter Zeitschritte
        als Kontext).
    output_chunk_length
        Output-Fenster des TFT-Modells. Ist in dieser Pipeline typ.
        gleich ``horizon_weeks``.
    train_length_weeks
        Länge des gleitenden Trainingsfensters in Wochen (wird ggf. vom
        Tuning überschrieben).
    validation_weeks
        Anzahl Wochen am Serienende, die für den Rolling-Backtest
        reserviert werden.
    stride_weeks
        Schrittweite, mit der der Test-Start im Backtest pro Block
        verschoben wird.
    """
    name: str
    horizon_weeks: int
    input_chunk_length: int
    output_chunk_length: int
    train_length_weeks: int
    validation_weeks: int
    stride_weeks: int


@dataclass
class BaselineConfig:
    """
    Konfiguration der einfachen Jahres-Baseline.
    """
    years_back: int = 3
    min_years: int = 1


@dataclass
class TuningConfig:
    """
    Konfiguration für Optuna-Tuning eines bestimmten Horizonts.
    """
    enabled: bool = False
    n_trials: int = 20
    direction: str = "minimize"
    metric: str = "smape"

    train_length_min: int = 52
    train_length_max: int = 208
    train_length_step: int = 13

    hidden_size_min: int = 32
    hidden_size_max: int = 96
    hidden_size_step: int = 16

    hidden_continuous_size_min: int = 8
    hidden_continuous_size_max: int = 32
    hidden_continuous_size_step: int = 8

    lstm_layers_min: int = 1
    lstm_layers_max: int = 3

    dropout_min: float = 0.1
    dropout_max: float = 0.5

    learning_rate_min: float = 1e-4
    learning_rate_max: float = 3e-3

    batch_size_choices: List[int] = field(default_factory=lambda: [8, 12, 16, 24])


@dataclass
class GlobalConfig:
    """
    Gesamt-Konfiguration des Projekts.
    Achtung: alle veränderlichen Defaults werden über default_factory gesetzt.
    """
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    tft: TftConfig = field(default_factory=TftConfig)

    horizons: Dict[int, HorizonConfig] = field(
        default_factory=lambda: {
            13: HorizonConfig(
                name="h13_model",
                horizon_weeks=13,
                input_chunk_length=52,
                output_chunk_length=13,
                train_length_weeks=104,
                validation_weeks=52,
                stride_weeks=13,
            ),
            52: HorizonConfig(
                name="h52_model",
                horizon_weeks=52,
                input_chunk_length=104,
                output_chunk_length=52,
                train_length_weeks=156,
                validation_weeks=104,
                stride_weeks=52,
            ),
        }
    )

    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    tuning: Dict[int, TuningConfig] = field(
        default_factory=lambda: {
            13: TuningConfig(
                enabled=False,
                n_trials=30,
                direction="minimize",
                metric="smape",
                train_length_min=52,
                train_length_max=156,
                train_length_step=13,
                hidden_size_min=32,
                hidden_size_max=96,
                hidden_size_step=16,
                hidden_continuous_size_min=8,
                hidden_continuous_size_max=32,
                hidden_continuous_size_step=8,
                lstm_layers_min=1,
                lstm_layers_max=3,
                dropout_min=0.1,
                dropout_max=0.5,
                learning_rate_min=5e-4,
                learning_rate_max=2e-3,
                batch_size_choices=[8, 12, 16, 24],
            ),
            52: TuningConfig(
                enabled=False,
                n_trials=30,
                direction="minimize",
                metric="smape",
                train_length_min=104,
                train_length_max=260,
                train_length_step=13,
                hidden_size_min=48,
                hidden_size_max=128,
                hidden_size_step=16,
                hidden_continuous_size_min=16,
                hidden_continuous_size_max=48,
                hidden_continuous_size_step=8,
                lstm_layers_min=1,
                lstm_layers_max=3,
                dropout_min=0.1,
                dropout_max=0.5,
                learning_rate_min=3e-4,
                learning_rate_max=2e-3,
                batch_size_choices=[8, 12, 16, 24],
            ),
        }
    )
