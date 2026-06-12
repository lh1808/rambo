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
    selected_future_covariates:
        Liste der Future-Kovariaten-Spalten, die dem Modell übergeben
        werden. Die vollen Bundesland-Flags (16 Stück) sind bei wenigen
        Trainingssamples reines Rauschen — die aggregierten Features
        reichen. Leere Liste = alle Kovariaten durchreichen.
    """
    freq: str = "W-SUN"
    date_column: Optional[str] = "date"
    ics_root_dir: str = "/mnt/ferien-api-data-main/resources/de"
    selected_future_covariates: List[str] = field(default_factory=lambda: [
        "vac_DE_count",
        "vac_BY",
        "holidays_workdays_BY",
        "holidays_workdays_DE_count",
        "year_turn_any",
    ])


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

        KENNZAHL__PRODUKT__STATUS__ORGA

    z. B. TERM_EINGANG_SCHRIFTST__KFZ_Vollkasko__Neuschaden__ORG12

    Die Anzahl der Namensteile ist über ``component_n_parts`` konfigurierbar
    (4 seit Aufnahme der ORGA-Dimension). Eine Spalte gilt als Zielreihe,
    wenn sie genau ``component_n_parts - 1`` Trennzeichen enthält.

    Wenn product_aggregation=True ist, werden für jedes Produkt zusätzliche
    Aggregatspalten erzeugt, in denen über alle Kennzahlen, Statusse UND
    ORGA-Ausprägungen des jeweiligen Produkts summiert wird:

        AGG_PROD_KFZ_Vollkasko
        AGG_PROD_HUS_Hausrat
        ...

    Guardrails bei hoher Komponentenzahl (32 × N_orga):
    ---------------------------------------------------
    past_covariate_select:
        Substring-Filter, der bestimmt, welche Roh-Spalten als
        Past-Kovariaten verwendet werden (leer = alle, bisheriges
        Verhalten). Für den ORGA-Fall auf ``["AGG_PROD_"]`` setzen: dann
        fließen nur die Produkt-Aggregate (plus deren Lag/MA/YoY-Features)
        als Past-Kovariaten ein, statt aller 32 × N_orga Einzel-Zellen.
        Die Zell-Historie sieht das Modell weiterhin über die Zielreihe
        selbst (Input-Chunk). Das entkoppelt die Kovariatenbreite von der
        ORGA-Kardinalität.
    min_component_total:
        Mindest-Summe (absolut, über die gesamte Historie), die eine
        KPSO-Zielzelle erreichen muss, um modelliert zu werden. Zellen
        darunter werden verworfen (mit Log-Hinweis) — praktisch leere
        Zellen sind nicht prognostizierbar, blähen aber Breite und
        Rauschen auf. ``0`` = kein Drop (bisheriges Verhalten). Die
        Produkt-Aggregate enthalten die verworfenen Zellen weiterhin.
    """
    drop_contains: List[str] = field(default_factory=list)
    aggregate_map: Dict[str, List[str]] = field(default_factory=dict)

    product_aggregation: bool = True
    product_agg_prefix: str = "AGG_PROD_"

    include_if_name_contains: List[str] = field(default_factory=list)

    # Guardrails bei hoher Komponentenzahl (siehe Klassen-Docstring).
    past_covariate_select: List[str] = field(default_factory=list)
    min_component_total: float = 0.0

    add_lags: bool = True
    add_yoy: bool = True
    add_ma: bool = True
    add_std: bool = False
    lags: List[int] = field(default_factory=lambda: [4, 8, 13, 52])
    rolling_windows: List[int] = field(default_factory=lambda: [4, 8, 13, 52])
    yoy_mode: str = "log"
    min_periods: int = 4

    component_main_sep: str = "__"
    component_n_parts: int = 4
    static_kpi_index: int = 0
    static_product_index: int = 1
    static_status_index: int = 2
    static_orga_index: int = 3
    product_sep: str = "_"
    product_sparte_index: int = 0


@dataclass
class TftConfig:
    """
    Globale Hyperparameter für das TFT-Modell (Basiseinstellungen).

    Diese Werte werden verwendet, wenn Tuning deaktiviert ist oder als
    Startpunkt, den das Tuning überschreibt.

    Attributes
    ----------
    early_stopping_patience
        Anzahl Epochen ohne Verbesserung des Training-Loss, nach denen
        das Training abgebrochen wird. ``0`` deaktiviert EarlyStopping
        und trainiert stets volle ``n_epochs``.
    feed_forward
        Feed-Forward-Netzwerk-Typ im TFT. ``"SwiGLU"`` hat sich als
        performanter als ``"GatedResidualNetwork"`` erwiesen.
    weight_decay
        L2-Regularisierung im Optimizer. Verhindert Overfitting,
        besonders bei wenigen Trainingssamples.
    gradient_clip_val
        Gradient-Clipping-Schwellwert. Stabilisiert das Training bei
        Ausreißern in den Daten. ``0`` deaktiviert Gradient Clipping.
    use_time_encoders
        Wenn ``True``, werden dem TFT automatische Zeitfeatures
        hinzugefügt (Monat, Kalenderwoche, zyklische Encodierungen,
        relative Position). Ersetzt ``add_relative_index``. Empfohlen.
    """
    n_epochs: int = 200
    full_attention: bool = False
    num_attention_heads: int = 4
    batch_size: int = 16
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
    weight_decay: float = 5e-4
    loss: str = "mse"
    feed_forward: str = "SwiGLU"
    gradient_clip_val: float = 1.0
    use_time_encoders: bool = True
    early_stopping_patience: int = 15


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
        Länge des Trainingsfensters in Wochen. Im Modus ``"fixed"``
        wird exakt dieses Fenster verwendet; im Modus ``"expanding"``
        ist es die Obergrenze. Wird ggf. vom Tuning überschrieben.
    validation_weeks
        Anzahl Wochen am Serienende, die für den Rolling-Backtest
        reserviert werden.
    stride_weeks
        Schrittweite, mit der der Test-Start im Backtest pro Block
        verschoben wird.
    window_mode
        Steuerung des Trainingsfensters im Backtest:

        - ``"fixed"`` – jeder Block trainiert auf exakt
          ``train_length_weeks`` Wochen. Blöcke, für die nicht genug
          Historie verfügbar ist, werden übersprungen. Die Backtest-
          Blöcke sind damit direkt vergleichbar.
        - ``"expanding"`` – das Fenster beginnt am Serienstart und
          wächst mit jedem Block. ``train_length_weeks`` wirkt als
          Obergrenze. Nutzt alle verfügbaren Daten, aber die Blöcke
          haben unterschiedliche Trainingsbedingungen.
    """
    name: str
    horizon_weeks: int
    input_chunk_length: int
    output_chunk_length: int
    train_length_weeks: int
    validation_weeks: int
    stride_weeks: int
    window_mode: str = "fixed"


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

    Attributes
    ----------
    metric
        Zielmetrik für die Optimierung:

        - ``"smape"`` – Symmetric Mean Absolute Percentage Error.
          Behandelt alle Komponenten gleich, unabhängig vom Volumen.
        - ``"wape"``  – Weighted Absolute Percentage Error.
          Gewichtet nach Volumen: Fehler auf großen Produkten zählen
          mehr. Besser für die Kapazitätsplanung.
        - ``"combined"`` – gewichtete Kombination aus WAPE und sMAPE:
          ``alpha × WAPE + (1 − alpha) × sMAPE``. Priorisiert
          volumenstarke Produkte, ignoriert aber volumenschwache nicht
          komplett. **Empfohlen.**
        - ``"combined_pooled"`` – wie ``"combined"``, aber WAPE und sMAPE
          werden *volumengewichtet über die Komponenten gepoolt* statt
          ungewichtet gemittelt. Bei vielen kleinen Komponenten (z. B.
          32 × N_orga) verhindert das, dass viele rauschige Klein-Zellen
          den Score dominieren. **Empfohlen, sobald ORGA aktiv ist.**
        - ``"mae"`` / ``"rmse"`` / ``"mape"`` – weitere Standardmetriken.

    tuning_metric_alpha
        Mischungsverhältnis für ``metric="combined"``:

        - ``1.0`` = reines WAPE (nur Volumen zählt)
        - ``0.0`` = reines sMAPE (alle Komponenten gleich)
        - ``0.7`` = 70% WAPE + 30% sMAPE **(Default, empfohlen)**

        Wird bei allen anderen Metriken ignoriert.
    """
    enabled: bool = False
    n_trials: int = 20
    direction: str = "minimize"
    metric: str = "combined"
    tuning_metric_alpha: float = 0.7

    # --- Optuna-Sampler / Pruning ---
    sampler_multivariate: bool = True
    """Multivariater TPE-Sampler: modelliert Wechselwirkungen zwischen den
    Hyperparametern (z. B. learning_rate × hidden_size) statt sie
    unabhängig zu behandeln. ``True`` empfohlen."""
    sampler_seed: Optional[int] = None
    """Seed des Samplers für Reproduzierbarkeit. ``None`` → es wird
    ``TftConfig.random_state`` verwendet."""
    use_pruning: bool = True
    """MedianPruner aktiv (bricht aussichtslose Trials früh ab). Achtung:
    der Per-Epoch-Callback überwacht ``train_loss`` (kein Held-out-Signal)
    und interagiert nur grob mit dem mehrblockigen Backtest. ``False``
    lässt jeden Trial voll durchlaufen (langsamer, aber sauberer)."""
    tuning_validation_weeks: Optional[int] = None
    """Validierungsfenster für die Tuning-Objective (in Wochen).

    Das Tuning arbeitet auf einer **verkürzten Serie**: die letzten
    ``HorizonConfig.validation_weeks`` werden als Held-out abgeschnitten
    und nie von Optuna gesehen. Innerhalb der verkürzten Serie werden
    die letzten ``tuning_validation_weeks`` als Validierung verwendet.

    ::

        |---- Training ----|-- Tuning-Val --|-- Held-out --|
                            ↑ Optuna         ↑ Finale Eval
                              optimiert        (52 Wochen,
                              hierauf          Optuna-frei)

    ``None`` → automatisch 52 Wochen (1 Jahr, empfohlen).
    """

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

    weight_decay_min: float = 1e-5
    weight_decay_max: float = 1e-2

    batch_size_choices: List[int] = field(default_factory=lambda: [8, 12, 16, 24])

    num_attention_heads_choices: List[int] = field(default_factory=list)
    """Optionaler Suchraum für ``num_attention_heads``. Leer = nicht
    tunen (Default, bisheriges Verhalten). Werte müssen ``hidden_size``
    teilen; inkompatible Kombinationen werden als Trial verworfen."""

    input_chunk_length_choices: List[int] = field(default_factory=list)
    """Optionaler Suchraum für ``input_chunk_length`` (Kontextfenster,
    Horizont-Ebene). Leer = nicht tunen (Default). Wird getunet, fließt
    der Wert ins finale Modell und wird mit den getuneten Parametern
    persistiert. Größere Werte brauchen mehr Trainingshistorie."""


@dataclass
class DisaggregationConfig:
    """
    Steuerung der Tages-Disaggregation auf dem Schreibpfad.

    Das Modell prognostiziert wöchentlich; ist ``enabled=True``, wird der
    kombinierte Wochen-Forecast vor dem Schreiben über ein historisches
    Wochentagsprofil auf Tageswerte heruntergebrochen (summenerhaltend).
    Reporting und Evaluation bleiben davon unberührt (wöchentlich).

    Felder
    ------
    enabled
        ``True`` → tägliche Schreibwerte. ``False`` (Default) → bisheriges
        Verhalten (wöchentliche Schreibwerte).
    window_weeks
        Fenster (in Wochen) der Tageshistorie für die Profilschätzung.
        ``None`` = gesamte Historie.
    min_window_volume
        Mindest-Volumen im Fenster, ab dem das eigene Profil einer
        Komponente/Gruppe als verlässlich gilt (sonst hierarchischer
        Fallback Komponente → Kennzahl → global).
    weekend_policy
        ``"empirical"`` (Default): historische Wochenendanteile (bei
        Termineingang nahe 0). ``"zero"``: Sa/So hart auf 0, Werktage
        renormiert.
    """
    enabled: bool = False
    window_weeks: Optional[int] = 52
    min_window_volume: float = 7.0
    weekend_policy: str = "empirical"


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
                train_length_weeks=104,    # 2 Jahre → 40 Samples/Block
                validation_weeks=52,       # → 4 Blöcke (fixed)
                stride_weeks=13,
            ),
            52: HorizonConfig(
                name="h52_model",
                horizon_weeks=52,
                input_chunk_length=52,     # 1 Jahr Kontext (statt 2)
                output_chunk_length=52,
                train_length_weeks=130,    # 2,5 Jahre → 27 Samples
                validation_weeks=52,       # → 1 Block
                stride_weeks=52,
                window_mode="expanding",
            ),
        }
    )

    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    disaggregation: DisaggregationConfig = field(
        default_factory=DisaggregationConfig
    )

    tuning: Dict[int, TuningConfig] = field(
        default_factory=lambda: {
            13: TuningConfig(
                enabled=False,
                n_trials=40,
                direction="minimize",
                metric="combined",
                tuning_metric_alpha=0.7,
                train_length_min=78,
                train_length_max=117,  # ≤ 120 (verfügbar nach Held-out)
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
                weight_decay_min=1e-5,
                weight_decay_max=1e-2,
                batch_size_choices=[8, 16, 24, 32],
            ),
            52: TuningConfig(
                enabled=False,
                n_trials=25,
                direction="minimize",
                metric="combined",
                tuning_metric_alpha=0.7,
                train_length_min=104,
                train_length_max=117,  # ≤ 120 (verfügbar nach Held-out)
                train_length_step=13,
                hidden_size_min=32,
                hidden_size_max=128,
                hidden_size_step=16,
                hidden_continuous_size_min=8,
                hidden_continuous_size_max=48,
                hidden_continuous_size_step=8,
                lstm_layers_min=1,
                lstm_layers_max=3,
                dropout_min=0.1,
                dropout_max=0.5,
                learning_rate_min=3e-4,
                learning_rate_max=2e-3,
                weight_decay_min=1e-5,
                weight_decay_max=1e-2,
                batch_size_choices=[8, 16, 24, 32],
            ),
        }
    )

    tuned_params_dir: Optional[str] = None
    """
    Verzeichnis, in dem getunete Parameter gespeichert und geladen werden.
    Ist dieser Pfad gesetzt und Tuning deaktiviert, versucht die Pipeline
    die zuletzt für den jeweiligen Horizont gespeicherten Parameter zu
    laden. ``None`` deaktiviert das Feature.

    Beim Laden werden NUR die tatsächlich getuneten Hyperparameter (aus
    ``best_params``) plus ``train_length_weeks`` übernommen. Operative und
    architektonische Felder (``n_epochs``, ``early_stopping_patience``,
    ``optimizer``, ``loss``, ``gradient_clip_val`` …) stammen weiterhin aus
    der AKTUELLEN Config – sie werden nicht vom Tuning-Lauf überschrieben.
    Das hält das Deployment-Regime konsistent zur Config, unter der das
    Modell läuft.
    """
