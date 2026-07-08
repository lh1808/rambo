"""rubin – Analyse- und Production-Pipelines für Causal ML.

Die Top-Level-Symbole werden lazy aufgelöst (PEP 562): `import rubin` bzw.
`from rubin.pipelines.production_pipeline import ProductionPipeline` lädt damit
NICHT mehr die komplette Analyse-Kette (matplotlib, shap, optuna, mlflow, …).
Das hält den Production-Scoring-Pfad leichtgewichtig (→ pixi-Environment `prod`)
und beschleunigt jeden Import, der nur einen Teil des Pakets braucht.

`from rubin import AnalysisPipeline` etc. funktioniert unverändert — der Import
des jeweiligen Untermoduls passiert beim ersten Attributzugriff.
"""

from typing import TYPE_CHECKING

_LAZY = {
    "AnalysisConfig": ("rubin.settings", "AnalysisConfig"),
    "load_config": ("rubin.settings", "load_config"),
    "AnalysisPipeline": ("rubin.pipelines.analysis_pipeline", "AnalysisPipeline"),
    "AnalysisResult": ("rubin.pipelines.analysis_pipeline", "AnalysisResult"),
    "ProductionPipeline": ("rubin.pipelines.production_pipeline", "ProductionPipeline"),
    "ProductionOutputs": ("rubin.pipelines.production_pipeline", "ProductionOutputs"),
    "DataPrepPipeline": ("rubin.pipelines.data_prep_pipeline", "DataPrepPipeline"),
    "DataPrepOutputs": ("rubin.pipelines.data_prep_pipeline", "DataPrepOutputs"),
    "ModelRegistry": ("rubin.model_registry", "ModelRegistry"),
    "ModelContext": ("rubin.model_registry", "ModelContext"),
    "default_registry": ("rubin.model_registry", "default_registry"),
    "promote_champion": ("rubin.model_management", "promote_champion"),
    "read_registry": ("rubin.model_management", "read_registry"),
    "float_metrics": ("rubin.model_management", "float_metrics"),
}

__all__ = list(_LAZY)

if TYPE_CHECKING:  # nur für IDEs/Typechecker — zur Laufzeit lazy
    from rubin.settings import AnalysisConfig, load_config  # noqa: F401
    from rubin.pipelines.analysis_pipeline import AnalysisPipeline, AnalysisResult  # noqa: F401
    from rubin.pipelines.production_pipeline import ProductionPipeline, ProductionOutputs  # noqa: F401
    from rubin.pipelines.data_prep_pipeline import DataPrepPipeline, DataPrepOutputs  # noqa: F401
    from rubin.model_registry import ModelRegistry, ModelContext, default_registry  # noqa: F401
    from rubin.model_management import promote_champion, read_registry, float_metrics  # noqa: F401


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'rubin' has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # Cache: nächster Zugriff ohne __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
