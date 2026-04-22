"""
Forecasting-Paket für die PLUTO Terminprognose.

Öffentliche API (Re-Exports):
-----------------------------
- GlobalConfig, HorizonConfig, TftConfig, TuningConfig, PreprocessingConfig,
  DataConfig, BaselineConfig : Konfigurations-Datenklassen
- load_config                : YAML/Env-Overlay auf die Datenklassen
- run_full_job               : Gesamtjob (Training, Evaluation, Prognose)
- train_and_evaluate_for_horizon, forecast_with_artifacts, ModelArtifacts
                              : Einzelschritte der Pipeline
- evaluate_multivariate      : Metriken auf einer Forecast-Serie
- build_transformers, build_tft, TransformArtifacts
                              : Low-Level-Bausteine
"""

from .config import (
    BaselineConfig,
    DataConfig,
    GlobalConfig,
    HorizonConfig,
    PreprocessingConfig,
    TftConfig,
    TuningConfig,
)
from .config_loader import load_config
from .evaluation import (
    compare_model_vs_baseline,
    evaluate_multivariate,
    weekly_baseline_mean_last3,
)
from .model import TransformArtifacts, build_tft, build_transformers
from .pipeline import (
    ModelArtifacts,
    forecast_with_artifacts,
    run_full_job,
    train_and_evaluate_for_horizon,
)
from .reporting import (
    RunReporter,
    evaluate_past_forecasts,
    load_tuned_params,
    save_tuned_params,
    update_retrospective_report,
)

__all__ = [
    "BaselineConfig",
    "DataConfig",
    "GlobalConfig",
    "HorizonConfig",
    "ModelArtifacts",
    "PreprocessingConfig",
    "RunReporter",
    "TftConfig",
    "TransformArtifacts",
    "TuningConfig",
    "build_tft",
    "build_transformers",
    "evaluate_multivariate",
    "evaluate_past_forecasts",
    "forecast_with_artifacts",
    "load_config",
    "load_tuned_params",
    "run_full_job",
    "save_tuned_params",
    "train_and_evaluate_for_horizon",
    "update_retrospective_report",
]
