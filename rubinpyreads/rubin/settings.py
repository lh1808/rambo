from __future__ import annotations

"""Konfigurationsmodell und YAML-Loader."""

from typing import Any, Dict, List, Optional, Literal, Union
from pathlib import Path
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Basis: strikte Modelle
# ---------------------------------------------------------------------------

_STRICT = ConfigDict(extra="forbid")


SUPPORTED_MODEL_NAMES = {
    "SLearner",
    "TLearner",
    "XLearner",
    "DRLearner",
    "NonParamDML",
    "ParamDML",
    "CausalForestDML",
    "CausalForest",
}

# Modelle, die Multi-Treatment *nicht* unterstützen.
# NonParamDML: econml beschränkt das nichtparametrische Final-Model auf
# "single-dimensional continuous or binary categorical treatment" — mit
# discrete_treatment=True (rubin-Standard) also faktisch Binary-only.
_BT_ONLY_MODELS = {"SLearner", "TLearner", "XLearner", "CausalForest", "NonParamDML"}


class TreatmentConfig(BaseModel):
    model_config = _STRICT

    type: Literal["binary", "multi"] = "binary"
    # Welche Gruppe als Control/Baseline dient. Aktuell wird nur reference_group=0
    # unterstützt. Eine Erweiterung auf andere Baseline-Gruppen ist vorbereitet,
    # erfordert aber Anpassungen in den Metrik-Funktionen.
    reference_group: int = 0



class MLflowConfig(BaseModel):
    model_config = _STRICT

    experiment_name: str = "rubin"


class DataFilesConfig(BaseModel):
    model_config = _STRICT

    x_file: str
    t_file: str
    y_file: str
    dtypes_file: Optional[str] = None
    s_file: Optional[str] = None

    # Externe Evaluationsdaten (optional, für validate_on="external").
    # Wenn gesetzt, wird auf diesen Daten evaluiert statt auf einem Split
    # der Trainingsdaten. Die Trainingsdaten (x/t/y_file) werden vollständig
    # zum Training verwendet. Mehrere Trainingsdateien können vorab per
    # DataPrep zusammengeführt werden.
    eval_x_file: Optional[str] = None
    eval_t_file: Optional[str] = None
    eval_y_file: Optional[str] = None
    eval_s_file: Optional[str] = None

    # Eval-Maske(n) für "Train Many, Evaluate Some" (optional).
    # Ein oder mehrere Boolean-Arrays (.npy): True = Zeile wird für Evaluation verwendet.
    # Training und Cross-Prediction laufen auf ALLEN Daten (inkl. Mask-Zeilen).
    # Nur die Evaluation (Uplift-Metriken, DRTester, Policy Values) wird auf das
    # Mask-Subset eingeschränkt.
    # Akzeptiert einen einzelnen Pfad (str) oder eine Liste von Pfaden.
    # Bei mehreren Masken werden diese per OR kombiniert.
    eval_mask_file: Optional[Union[str, List[str]]] = None


class HistoricalScoreConfig(BaseModel):
    model_config = _STRICT

    """Konfiguration für einen historischen Score (Vergleichsbasis)."""

    name: str = "historical_score"
    column: str = "S"
    higher_is_better: bool = True


class DataPrepConfig(BaseModel):
    model_config = _STRICT

    # I/O
    # Hinweis: Es werden häufig mehrere Rohdateien (CSV/SAS) verarbeitet.
    # Daher ist `data_path` eine Liste. Für einfache Fälle genügt eine Datei.
    data_path: List[str]

    # Optionaler separater Evaluationsdatensatz. Wenn gesetzt, wird der
    # Preprocessor auf data_path (Train) gefittet und nur transformiert auf
    # eval_data_path angewendet. Verhindert Data-Leakage bei externer Validierung.
    # Ausgabe: X_eval.parquet, T_eval.parquet, Y_eval.parquet im output_path.
    eval_data_path: Optional[List[str]] = None

    # Feature-/Info-Dateien sind projektspezifisch. In der Praxis liegen sie oft
    # als Excel-Dateien vor. Die Pipeline wirft in `DataPrepPipeline` eine klare
    # Fehlermeldung, wenn zwingende Pfade fehlen.
    feature_path: Optional[str] = None
    info_path: Optional[str] = None
    output_path: str

    # CSV/SAS
    delimiter: str = ","
    chunksize: Optional[int] = None
    sas_encoding: str = "utf-8"

    # Zielvariablen
    target: Union[str, List[str]] = "Y"
    treatment: str = "T"
    score_name: Optional[str] = "S"
    score_as_feature: bool = False

    # Replacement/Mapping
    target_replacement: Optional[Dict[str, Any]] = None
    treatment_replacement: Optional[Dict[str, Any]] = None

    # Mehrdatei-Logik
    # - "merge": Dateien werden untereinander gehängt.
    # - "treatment_only": es werden nur Treatment-Zeilen aus allen Dateien genutzt;
    #   aus der Control-Datei (Index: control_file_index) wird zusätzlich eine
    #   „Control-Kopie“ erzeugt.
    multiple_files_option: Literal["merge", "treatment_only"] = "merge"
    control_file_index: int = 0

    # Treatment-Balance: Bei mehreren Dateien prüfen, ob die Treatment-Verteilung
    # pro Datei ähnlich ist. Falls nicht, kann per Random-Downsampling ausgeglichen werden.
    # Verhindert Treatment-Imbalances, die in der Evaluation zu verzerrten Metriken führen.
    balance_treatments: bool = False

    # Auswahl-Guardrails für das Balancing (alle optional, Default = aus).
    # Die Ziel-Treatment-Rate wird primär nach MAXIMALER EFFEKTIVER STICHPROBE
    # N_keep · p·(1−p) gewählt (nicht nach minimalem Zeilenverlust), da die Varianz
    # eines Effekt-/Uplift-Schätzers mit 1/(N·p·(1−p)) skaliert und der kleinere Arm
    # die Präzision limitiert. Diese Schranken begrenzen zusätzlich, wie klein ein Arm
    # werden darf bzw. wie viel verworfen wird:
    balance_min_arm_abs: int = 0          # absolute Mindestanzahl je Arm (0 = aus)
    balance_min_arm_frac: float = 0.0     # Mindestanteil je Arm rel. zum kleineren Ausgangs-Arm (0 = aus)
    balance_max_loss_frac: float = 1.0    # max. erlaubter Zeilenverlust-Anteil (1.0 = aus)
    balance_target_grid_step: float = 0.0 # Gitter-Schrittweite für Zielraten Richtung 0.5 (0 = nur min/max)

    # "Train Many, Evaluate Some": Index/Indizes der Dateien (0-basiert), deren Daten
    # für die Evaluation verwendet werden sollen. Alle Dateien werden für Training
    # genutzt, aber Metriken nur auf den ausgewählten Dateien berechnet.
    # Akzeptiert einen einzelnen Index (int) oder eine Liste von Indizes.
    eval_file_index: Optional[Union[int, List[int]]] = None

    # Sonstiges
    binary_target: bool = True
    fill_na_method: Optional[Literal["zero", "median", "mean", "mode", "max"]] = None
    # Steuerung der Typ-Behandlung für Features:
    #   "auto"            → automatische Erkennung (object/category → kategorisch, Rest numerisch)

    # Explizite Feature-Auswahl (aus UI oder manuell). Wenn gesetzt, werden nur diese
    # Spalten als Features in X aufgenommen. Überschreibt feature_path falls beides gesetzt.
    features: Optional[List[str]] = None
    # Explizite kategorische Spalten. Überschreibt die automatische Erkennung aus
    # feature_path oder dtype-Heuristik.
    categorical_columns: Optional[List[str]] = None

    # Deduplizierung: Wenn Kunden mehrfach im Datensatz vorkommen, kann hier auf
    # einen Eintrag pro Kunde reduziert werden. Die Spalte (z. B. PartnerID) muss
    # im Rohdatensatz vorhanden sein, wird aber NICHT als Feature übernommen.
    # Die Deduplizierung geschieht direkt nach dem Einlesen, bevor auf Features
    # reduziert wird.
    deduplicate: bool = False
    deduplicate_id_column: Optional[str] = None

    # MLflow (DataPrep kann optional loggen)
    log_to_mlflow: bool = False
    mlflow_experiment_name: Optional[str] = None
    mlflow_run_name: Optional[str] = None


class DataProcessingConfig(BaseModel):
    model_config = _STRICT

    # Memory-Reduktion: Datentypen downcasten (float64→float32, int64→int32 etc.).
    # Wird in der Analyse-Pipeline nach dem Laden der Daten angewendet.
    # In der DataPrep-Pipeline wird reduce_mem_usage() immer aufgerufen.
    reduce_memory: bool = True

    # Optionales Downsampling in der Analyse.
    # Wird als Anteil (0..1] interpretiert.
    df_frac: Optional[float] = None

    # Validierungsmodus:
    # - "cross": Cross-Predictions (Out-of-Fold) auf dem gleichen Datensatz (Standard)
    # - "external": Training auf data_files, Evaluation auf separaten eval_*-Dateien
    validate_on: Literal["cross", "external"] = "cross"

    # Anzahl der Splits für Cross-Predictions (Out-of-Fold-Vorhersagen).
    cross_validation_splits: int = 5

    # Interne Cross-Fitting-Folds für DML-Nuisance-Residuals (Y_resid, T_resid).
    # Getrennt von cross_validation_splits (äußere Evaluation).
    # EconML-Default=2 (50% Daten pro Nuisance-Fit). Höhere Werte (3–5) liefern
    # stabilere Residuals, kosten aber linear mehr Rechenzeit.
    dml_crossfit_folds: int = 5

    # Monte-Carlo-Iterationen: Wiederholt das interne Cross-Fitting N-mal mit
    # unterschiedlichen Splits und mittelt die Residuals. Reduziert Varianz der
    # Nuisance-Schätzung linear (mc_iters=3 → ~3× niedrigere Varianz).
    # None = 1 Durchlauf (Standard). Kostet linear mehr Rechenzeit.
    mc_iters: Optional[int] = None

    # Aggregation über mc_iters: "mean" (Standard) oder "median" (robuster bei Ausreißern).
    mc_agg: Literal["mean", "median"] = "mean"


class FeatureSelectionConfig(BaseModel):
    model_config = _STRICT

    enabled: bool = False

    # Methoden zur Importance-Berechnung. Mehrere können kombiniert werden;
    # die Ergebnisse werden per Union zusammengeführt.
    # - "catboost_importance": CatBoost auf Outcome (Y), Gain-Importance (native Kategorien, weniger Overfitting)
    # - "lgbm_importance": LightGBM auf Outcome (Y), Gain-Importance
    # - "causal_forest": CausalForestDML Feature-Importances (kausale Relevanz)
    # - "none": keine Importance-Filterung
    methods: List[Literal["catboost_importance", "lgbm_importance", "causal_forest", "none"]] = Field(
        default_factory=lambda: ["catboost_importance"]
    )

    # Maximale Anzahl Features nach Feature-Selektion (Default: 77).
    # Bei mehreren Methoden: jede Methode liefert max_features / Anzahl_Methoden
    # Features (aufgerundet). Die Union aller Methoden wird auf max_features gekappt.
    max_features: int = 77

    # Korrelation: ab welchem Betrag (|corr|) eine Spalte als redundant gilt.
    correlation_threshold: float = 0.9


class ModelsConfig(BaseModel):
    model_config = _STRICT

    models_to_train: List[str] = Field(default_factory=list)
    # Gleichgewichtetes Ensemble aller trainierten Modelle.
    # Die Cross-Predictions werden gemittelt; das Ensemble nimmt
    # an der Champion-Selektion teil.
    ensemble: bool = True


class BaseLearnerConfig(BaseModel):
    model_config = _STRICT

    type: Literal["catboost", "lgbm", "both"] = "catboost"
    fixed_params: Dict[str, Any] = Field(default_factory=dict)


class CausalForestSearchSpaceParam(BaseModel):
    """Low/High Override für einen CF-Suchraum-Parameter."""
    model_config = _STRICT
    low: Optional[float] = None
    high: Optional[float] = None


class CausalForestConfig(BaseModel):
    """Konfiguration für CausalForest-Tuning (CFT).

    CFT optimiert 4 kausale Parameter (max_depth, min_weight_fraction_leaf,
    min_var_fraction_leaf, criterion) für CausalForestDML und CausalForest.
    Alle anderen Forest-Parameter werden auf EconML-Defaults fixiert.
    Bei RCT wird model_t als DummyClassifier im Nuisance-Cache verwendet."""
    model_config = _STRICT

    forest_fixed_params: Dict[str, Any] = Field(default_factory=dict)  # Fixe Forest-Params (überschreiben Defaults)
    tune_enabled: bool = False  # Optuna-basiertes CausalForest-Tuning (CFT) aktivieren
    tune_models: List[str] = Field(default_factory=list)  # Welche Modelle tunen (z.B. ["CausalForestDML"])
    tune_max_rows: Optional[int] = None  # Max. Zeilen für Tuning (RAM-Kontrolle)
    n_trials: int = 50  # Optuna-Trials für CFT
    single_fold: bool = False  # Single-Fold statt K-Fold (5× schneller, weniger robust)
    scorer: Literal["auto", "qini", "rscore", "qini_argmax"] = "auto"  # auto → qini bei RCT, rscore bei observational; bei Multi-Treatment immer rscore. qini_argmax: MT-Ranking über alle Arme (Opt-in; bei K=2 identisch zu qini)
    overfit_penalty: float = 0.0  # Train-Val-Gap-Penalty (0 = deaktiviert, empfohlen ~0.2–0.35)
    overfit_tolerance: float = 0.10  # Relativer Gap-Toleranz (10%, entschärft)
    overfit_max_penalized_gap: float = 1.0  # Deckelt den bestraften relativen Gap (Saturierung). <=0 = kein Cap (unbeschränkt). Verhindert Sign-Flip/Dominanz bei kleinen Scores (Qini/R-Score nahe 0).
    search_space: Dict[str, CausalForestSearchSpaceParam] = Field(default_factory=dict)  # Low/High Overrides
    depth_choices: Optional[List] = None  # Kategorische Auswahl für max_depth (z.B. [3, 5, 7, 10, 15, "None"])
    criterion_choices: Optional[List[str]] = None  # Kategorische Auswahl für criterion (z.B. ["mse", "het"])


class SearchSpaceParameterConfig(BaseModel):
    model_config = _STRICT

    type: Literal["int", "float", "categorical"]
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    step: Optional[float] = None
    choices: Optional[List[Any]] = None

    @model_validator(mode="after")
    def validate_definition(self) -> "SearchSpaceParameterConfig":
        if self.type == "categorical":
            if not self.choices:
                raise ValueError("Für type='categorical' muss choices gesetzt sein.")
            return self

        if self.low is None or self.high is None:
            raise ValueError("Für numerische Parameter müssen low und high gesetzt sein.")
        if self.high < self.low:
            raise ValueError("high muss größer oder gleich low sein.")
        if self.step is not None and self.step <= 0:
            raise ValueError("step muss > 0 sein.")
        return self


class SearchSpaceConfig(BaseModel):
    model_config = _STRICT

    lgbm: Dict[str, SearchSpaceParameterConfig] = Field(default_factory=dict)
    catboost: Dict[str, SearchSpaceParameterConfig] = Field(default_factory=dict)


class OptunaTuningConfig(BaseModel):
    model_config = _STRICT

    enabled: bool = False
    n_trials: int = 50
    timeout_seconds: Optional[int] = None
    cv_splits: int = 5
    single_fold: bool = False
    metric: str = "log_loss"
    metric_regression: str = "neg_mse"
    overfit_penalty: float = 0.0  # BLT-Penalty: Default 0. Wirkt NUR auf Meta-Learner-Tasks (S/T/X-Learner); DML/DR-Nuisances werden nie bestraft (Cross-Fitting + Orthogonalität, siehe BaseLearnerTuner._penalty_applies). Aktivierbar (z.B. 0.2) zur Meta-Learner-Regularisierung.
    overfit_tolerance: float = 0.20  # Relativ: 20% Gap toleriert (entschärft; greift nur bei aktivierter Meta-Learner-Penalty)
    overfit_max_penalized_gap: float = 1.0  # Deckelt den bestraften relativen Gap (Saturierung). <=0 = kein Cap (unbeschränkt). Verhindert Sign-Flip/Dominanz bei kleinen Scores.
    max_tuning_rows: Optional[int] = None
    storage_path: Optional[str] = None
    study_name_prefix: str = "baselearner"
    reuse_study_if_exists: bool = True
    optuna_seed: int = 42
    search_space: SearchSpaceConfig = Field(default_factory=SearchSpaceConfig)
    # Welche Modelle per BLT optimiert werden. null = alle in models_to_train.
    # Bei expliziter Liste werden nur die Nuisance-Tasks dieser Modelle getuned.
    # Nicht-ausgewählte Modelle nutzen base_learner.fixed_params.
    models: Optional[List[str]] = None


class FinalModelTuningConfig(BaseModel):
    model_config = _STRICT

    enabled: bool = False
    n_trials: int = 50
    timeout_seconds: Optional[int] = None
    cv_splits: int = 5
    max_tuning_rows: Optional[int] = None
    models: Optional[List[str]] = None
    single_fold: bool = False
    overfit_penalty: float = 0.0
    overfit_tolerance: float = 0.10
    overfit_max_penalized_gap: float = 1.0  # Deckelt den bestraften relativen Gap (Saturierung). <=0 = kein Cap (unbeschränkt). Verhindert Sign-Flip/Dominanz bei kleinen Scores (Qini/R-Score nahe 0).
    scorer: Literal["auto", "qini", "rscore", "qini_argmax"] = "auto"  # auto → qini bei RCT, rscore bei observational; bei Multi-Treatment immer rscore. qini_argmax: MT-Ranking über alle Arme (Opt-in; bei K=2 identisch zu qini)
    fixed_params: Dict[str, Any] = Field(default_factory=dict)
    search_space: SearchSpaceConfig = Field(default_factory=SearchSpaceConfig)


class ShapConfig(BaseModel):
    model_config = _STRICT

    # Steuert, ob SHAP-Werte während der Analyse
    # berechnet werden. Wird vom Explainability-Schritt in der Pipeline ausgewertet.
    calculate_shap_values: bool = False

    # Aktiv genutzt von der Analyse-Pipeline und run_explain.py:
    n_shap_values: int = 10_000
    top_n_features: int = 20
    num_bins: int = 10

    # Binning-Strategie für CATE-Profil-/Dependence-Plots:
    # "quantile" (Default) = gleich besetzte Segmente — robust bei schiefen
    # Verteilungen; "width" = Equal-Width (Altverhalten, Rand-Bins können bei
    # Ausreißern fast leer sein und Rausch-Mittelwerte zeigen).
    bin_strategy: Literal["quantile", "width"] = "quantile"

    # Rückbeschriftung kodierter kategorischer Features in den Explainability-
    # Plots (z. B. SAS-Format-Codes): {Spaltenname: {Code: Fachlabel}}.
    # Codes tolerant gematcht ('1', 1, 1.0 identisch). Reine Anzeige-Option —
    # hat keinerlei Einfluss auf Training oder Metriken.
    # Beispiel: value_labels: {GESCHLECHT: {1: "männlich", 2: "weiblich"}}
    value_labels: Dict[str, Dict[Union[int, float, str], str]] = Field(default_factory=dict)

    # Methode für Explainability: SHAP wird automatisch versucht; falls das
    # shap-Paket nicht verfügbar ist, wird auf Surrogate-basierte Erklärungen
    # zurückgefallen (siehe explainability/shap_uplift.py → shap_available()).


class OptionalOutputConfig(BaseModel):
    model_config = _STRICT

    output_dir: Optional[str] = None
    save_predictions: bool = False
    predictions_format: Literal["csv", "parquet"] = "csv"

    # Schutz gegen extrem große Artefakte (z. B. Millionen Zeilen).
    max_prediction_rows: Optional[int] = None




class BundleConfig(BaseModel):
    model_config = _STRICT

    enabled: bool = False
    base_dir: str = "runs/bundles"
    bundle_id: Optional[str] = None
    log_to_mlflow: bool = True


class SelectionConfig(BaseModel):
    model_config = _STRICT

    metric: str = "qini"
    higher_is_better: bool = True
    # Der Bundle-Export refittet IMMER alle trainierten Modelle (inkl. self-
    # contained Ensemble-Rebuild) auf vollen Daten; im Holdout-Modus unterbleibt
    # der Voll-Refit, damit keine Holdout-Zeilen ins Produktionsmodell fließen.
    manual_champion: Optional[str] = None


class SurrogateTreeConfig(BaseModel):
    model_config = _STRICT

    # Aktiviert den Surrogate-Einzelbaum, der das Champion-Modell nachlernt.
    # Der Baum wird mit den gleichen Features trainiert und lernt die
    # CATE-Vorhersagen des Champions als Regressionsziel.
    # Es wird ein einzelner Baum des konfigurierten Base-Learners (lgbm/catboost)
    # verwendet (n_estimators=1), was dank leaf-wise Growth (LightGBM) bzw.
    # symmetrischem Splitting (CatBoost) bessere Bäume als CART liefert.
    enabled: bool = False

    # Mindestanzahl Samples pro Blatt. Stellt sicher, dass jedes Blatt
    # statistisch belastbar ist.
    # Wird auf min_child_samples (LightGBM) bzw. min_data_in_leaf (CatBoost)
    # gemappt.
    min_samples_leaf: int = 50

    # Maximale Anzahl Blätter (nur LightGBM, leaf-wise Growth).
    # Steuert die Baumkomplexität direkt. Bei CatBoost wird stattdessen
    # max_depth verwendet.
    num_leaves: int = 31

    # Maximale Baumtiefe. None = keine Begrenzung bei LightGBM (-1),
    # bei CatBoost wird 6 als Default verwendet.
    max_depth: Optional[int] = None


class ConstantsConfig(BaseModel):
    # Kanonischer Config-Key ist "SEED" (Alias); intern wird konsequent
    # `random_seed` verwendet (populate_by_name erlaubt die Python-Konstruktion
    # über den Feldnamen).
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    random_seed: int = Field(42, alias="SEED")

    # Separater Seed für Tuning-CV-Splits. Muss sich vom random_seed unterscheiden,
    # damit Optuna die Hyperparameter auf ANDEREN Folds bewertet als die spätere
    # Cross-Prediction. Gleicher Seed → identische Folds → Val-Set-Overfitting.
    tuning_seed: int = Field(18, alias="TUNING_SEED")

    # Parallelisierungs-Level:
    #   1 = Minimal:  Base Learner nutzen 1 Kern, Folds sequentiell.
    #                 Minimaler RAM-Verbrauch, sicher auf jeder Maschine.
    #   2 = Moderat:  Base Learner nutzen alle Kerne, Folds sequentiell.
    #   3 = Empfohlen: Base Learner + Tuning-Trials parallel. Bester Kompromiss
    #                 aus Suchraum-Exploration und Geschwindigkeit. Nutzt alle
    #                 verfügbaren Kerne sowohl für parallele Optuna-Trials als
    #                 auch für parallele CV-Folds.
    #   4 = Maximum:  Wie Level 3, aber mit mehr parallelen Trials und Folds.
    #                 Höchster RAM-Verbrauch, marginal schneller als Level 3.
    parallel_level: int = Field(3, ge=1, le=4)

    # Arbeitsverzeichnis für alle erzeugten Artefakte (MLflow, Report, Cache,
    # Uploads, Bundles). Hält das Repository-Verzeichnis sauber.
    # Priorität: Env RUBIN_WORK_DIR > Config work_dir > "./runs"
    work_dir: Optional[str] = None

    @property
    def resolved_work_dir(self) -> str:
        """Aufgelöstes Arbeitsverzeichnis (Priorität: Env > Config > Default)."""
        import os
        env = os.environ.get("RUBIN_WORK_DIR")
        if env:
            return os.path.abspath(env)
        if self.work_dir:
            return os.path.abspath(self.work_dir)
        return os.path.abspath("runs")


class AnalysisConfig(BaseModel):
    model_config = _STRICT

    source_config_path: Optional[str] = None
    # "rct": Randomisiertes Experiment → BLT Propensity-Diagnose (Skill ≈ 0),
    #   Training mit DummyClassifier (P(T|X) = const) für alle Propensity-Rollen.
    # "observational": Beobachtungsdaten → volles Propensity-Tuning + -Training.
    study_type: Literal["rct", "observational"] = "rct"

    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    constants: ConstantsConfig = Field(default_factory=ConstantsConfig)
    data_files: DataFilesConfig
    historical_score: HistoricalScoreConfig = Field(default_factory=HistoricalScoreConfig)
    data_prep: Optional[DataPrepConfig] = None

    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    treatment: TreatmentConfig = Field(default_factory=TreatmentConfig)
    feature_selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    base_learner: BaseLearnerConfig = Field(default_factory=BaseLearnerConfig)
    causal_forest: CausalForestConfig = Field(default_factory=CausalForestConfig)
    tuning: OptunaTuningConfig = Field(default_factory=OptunaTuningConfig)
    final_model_tuning: FinalModelTuningConfig = Field(default_factory=FinalModelTuningConfig)
    shap_values: ShapConfig = Field(default_factory=ShapConfig)
    optional_output: OptionalOutputConfig = Field(default_factory=OptionalOutputConfig)
    bundle: BundleConfig = Field(default_factory=BundleConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    surrogate_tree: SurrogateTreeConfig = Field(default_factory=SurrogateTreeConfig)



    @model_validator(mode="after")
    def validate_models_and_manual_champion(self) -> "AnalysisConfig":
        available = list(self.models.models_to_train or [])
        invalid = [name for name in available if name not in SUPPORTED_MODEL_NAMES]
        if invalid:
            raise ValueError(
                "models.models_to_train enthält unbekannte Modelle. "
                f"Erhalten: {invalid}. Erlaubt: {sorted(SUPPORTED_MODEL_NAMES)}"
            )

        # Multi-Treatment: Modelle blockieren, die MT nicht unterstützen.
        if self.treatment.type == "multi":
            if self.treatment.reference_group != 0:
                raise ValueError(
                    f"treatment.reference_group={self.treatment.reference_group}: "
                    f"Aktuell wird nur reference_group=0 unterstützt."
                )
            bt_only = [name for name in available if name in _BT_ONLY_MODELS]
            if bt_only:
                raise ValueError(
                    f"treatment.type='multi' ist nicht kompatibel mit: {bt_only}. "
                    f"Diese Modelle unterstützen nur Binary Treatment. "
                    f"Bitte entfernen oder treatment.type='binary' setzen."
                )
            # Hinweis: Bei MT gibt es keine einfache 'qini'-Metrik mehr.
            # Stattdessen: 'policy_value' (global IPW), 'qini_T1', 'qini_T2',
            # 'policy_value_T1', etc.
            _bt_only_metrics = {"qini", "auuc", "uplift_at_10pct", "uplift_at_20pct",
                                "uplift_at_50pct"}
            if self.selection.metric in _bt_only_metrics:
                raise ValueError(
                    f"treatment.type='multi' mit selection.metric='{self.selection.metric}': "
                    f"Diese Metrik existiert bei Multi-Treatment nicht (nur per-Arm-Varianten). "
                    f"Empfohlen: 'policy_value' (global IPW), 'policy_value_T1', "
                    f"'qini_T1', 'auuc_T1', etc."
                )
            # Explizit gesetzter Qini-Scorer für FMT/CFT ist bei MT nicht möglich:
            # Der Qini-Scorer ist binär-only (t ∈ {0,1}, 1-d CATE). 'auto' wird
            # oben in resolve_scorer_auto bei MT bereits zu 'rscore' aufgelöst —
            # hier wird nur die explizite Fehlkonfiguration abgefangen.
            _mt_scorer_offenders = []
            if self.final_model_tuning.enabled and self.final_model_tuning.scorer == "qini":
                _mt_scorer_offenders.append("final_model_tuning.scorer")
            if getattr(self.causal_forest, "tune_enabled", False) and self.causal_forest.scorer == "qini":
                _mt_scorer_offenders.append("causal_forest.scorer")
            if _mt_scorer_offenders:
                raise ValueError(
                    f"treatment.type='multi' mit {' und '.join(_mt_scorer_offenders)}='qini': "
                    f"Der binäre Qini-Scorer verlangt t in {{0,1}} und 1-d CATE. "
                    f"Für Multi-Treatment stehen 'rscore' (Default via 'auto'; bewertet die "
                    f"volle Effektfläche) und 'qini_argmax' (Ranking über alle Arme via "
                    f"s(x)=max_k tau_k(x); für Personen-Priorisierung) zur Verfügung."
                )
            # treatment_only-Mehrdatei-Modus ist binär definiert: Er filtert pro
            # Datei hart auf T==1 und baut die Control-Kopie aus diesen Zeilen.
            # Bei K>2 Armen würden alle Zeilen mit T>=2 STILL verworfen und die
            # Arm-Struktur zerstört — daher harte Ablehnung statt Datenverlust.
            _dp = getattr(self, "data_prep", None)
            if _dp is not None and getattr(_dp, "multiple_files_option", "merge") == "treatment_only":
                raise ValueError(
                    "treatment.type='multi' mit data_prep.multiple_files_option='treatment_only': "
                    "Dieser Modus ist nur für Binary Treatment definiert (filtert pro Datei "
                    "auf T==1; Zeilen mit T>=2 gingen still verloren). Bitte "
                    "multiple_files_option='merge' verwenden und die Arme über "
                    "treatment_replacement abbilden."
                )
            # balance_treatments ist ebenfalls binär definiert: Ziel-Raten-Suche und
            # Downsampling arbeiten mit T==1 vs. T==0 (Effektiv-N = p·(1−p)) — bei
            # K>2 Armen würden T>=2-Zeilen ignoriert bzw. die Arm-Struktur verzerrt.
            if _dp is not None and getattr(_dp, "balance_treatments", False):
                raise ValueError(
                    "treatment.type='multi' mit data_prep.balance_treatments=true: "
                    "Das dateiübergreifende Treatment-Balancing ist nur für Binary "
                    "Treatment definiert (T==1 vs. T==0). Bitte deaktivieren — die "
                    "Raten-Warnung pro Datei bleibt aktiv (bei Multi-Treatment als "
                    "Anteil T>0 mit Arm-Verteilung im Log)."
                )

        manual = (self.selection.manual_champion or "").strip()
        if not manual:
            self.selection.manual_champion = None
            return self
        if manual not in available:
            raise ValueError(
                "selection.manual_champion muss in models.models_to_train enthalten sein. "
                f"Erhalten: {manual!r}. Verfügbar: {available}"
            )
        self.selection.manual_champion = manual

        # "both"-Modus ohne Tuning ist nicht sinnvoll: Ohne Optuna-Entscheidung
        # gibt es kein _learner_type → build_base_learner fällt auf CatBoost
        # zurück (siehe tuning/common.py, passend zum Default base_learner.type).
        # User bekommt einen klaren Hinweis.
        if (self.base_learner.type or "").lower() == "both" and not self.tuning.enabled:
            import logging
            logging.getLogger("rubin.config").warning(
                "base_learner.type='both' ist nur mit aktivem Tuning sinnvoll "
                "(Optuna wählt den Learner pro Task). Bei tuning.enabled=false "
                "wird für alle Rollen CatBoost verwendet (Fallback). "
                "Empfehlung: 'catboost' oder 'lgbm' explizit setzen, oder tuning.enabled=true aktivieren."
            )

        # "both"-Modus: fixed_params MUSS nested sein ({lgbm: {...}, catboost: {...}}).
        # Flache Params (z.B. num_leaves: 31) würden an beide Learner durchgereicht,
        # was CatBoost-Trials zum Absturz bringt (unbekannter Parameter).
        if (self.base_learner.type or "").lower() == "both":
            for name, fp in [("base_learner", self.base_learner.fixed_params),
                             ("final_model_tuning", self.final_model_tuning.fixed_params)]:
                if fp:
                    # Erlaubt: leer {} oder {"lgbm": {...}, "catboost": {...}} (nested)
                    non_nested_keys = [k for k, v in fp.items()
                                       if k not in ("lgbm", "catboost") or not isinstance(v, dict)]
                    if non_nested_keys:
                        raise ValueError(
                            f"{name}.fixed_params bei base_learner.type='both' muss verschachtelt sein: "
                            f"{{lgbm: {{...}}, catboost: {{...}}}}. "
                            f"Flache Parameter gefunden: {non_nested_keys}. "
                            f"Entweder in 'lgbm' oder 'catboost' Sub-Dict einsortieren, "
                            f"oder einen spezifischen Learner ('lgbm' / 'catboost') wählen."
                        )
        return self

    @model_validator(mode="after")
    def resolve_scorer_auto(self) -> "AnalysisConfig":
        """Löst scorer='auto' zu 'qini' (RCT) oder 'rscore' (observational) auf.

        Bei treatment.type='multi' wird IMMER 'rscore' aufgelöst — auch bei RCT:
        Der Qini-Scorer (uplift_curve/qini_coefficient) ist binär-only (verlangt
        t ∈ {0,1} und 1-d CATE); bei K>2 Armen ist die CATE (n, K-1)-dimensional.
        EconMLs RScorer unterstützt Multi-Treatment (R-Loss über <θ(X), T_res>)."""
        if self.treatment.type == "multi":
            resolved = "rscore"
        else:
            resolved = "qini" if self.study_type == "rct" else "rscore"
        if self.final_model_tuning.scorer == "auto":
            object.__setattr__(self.final_model_tuning, "scorer", resolved)
        if self.causal_forest.scorer == "auto":
            object.__setattr__(self.causal_forest, "scorer", resolved)
        return self

    @model_validator(mode="after")
    def validate_external_eval_files(self) -> "AnalysisConfig":
        if str(self.data_processing.validate_on).lower() == "external":
            missing = []
            if not self.data_files.eval_x_file:
                missing.append("eval_x_file")
            if not self.data_files.eval_t_file:
                missing.append("eval_t_file")
            if not self.data_files.eval_y_file:
                missing.append("eval_y_file")
            if missing:
                raise ValueError(
                    f"validate_on='external' erfordert Evaluationsdaten. "
                    f"Fehlend in data_files: {', '.join(missing)}. "
                    f"Bitte eval_x_file, eval_t_file und eval_y_file angeben."
                )
        return self


# Forward-Referenzen auflösen (nötig wegen `from __future__ import annotations`).
# Alle Config-Klassen müssen model_rebuild() aufrufen, damit Pydantic
# String-Annotationen in echte Typen auflöst.
for _name in list(dir()):
    _obj = eval(_name)
    if isinstance(_obj, type) and issubclass(_obj, BaseModel) and _obj is not BaseModel:
        _obj.model_rebuild()


def load_config(path: Union[str, Path]) -> AnalysisConfig:
    """Lädt und validiert eine YAML-Konfiguration.
- Unbekannte Schlüssel werden abgewiesen (extra="forbid").
- Fehlende Pflichtfelder führen zu klaren Fehlermeldungen."""
    path_str = str(path)
    with open(path_str, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Die Konfigurationsdatei muss ein YAML-Objekt (Mapping) sein.")

    raw = dict(raw)
    raw["source_config_path"] = path_str
    return AnalysisConfig.model_validate(raw)
