# Entwicklerhandbuch – rubin

Dieses Dokument richtet sich an Entwicklerinnen und Entwickler, die **rubin** erweitern oder bestehende Teile umbauen.
Der Fokus liegt auf einer konsistenten Struktur, klaren Zuständigkeiten und stabilen Produktionsartefakten.

## Entwicklungsumgebung einrichten

[Pixi](https://pixi.sh) ist das empfohlene Tool für das Environment-Management.
Es verwaltet Python, conda-forge- und PyPI-Pakete einheitlich und erzeugt ein
reproduzierbares Lockfile (`pixi.lock`).

```bash
# Pixi installieren (einmalig)
curl -fsSL https://pixi.sh/install.sh | bash

# Dev-Environment aufbauen (Tests + Linting)
cd rubin_repo
pixi install -e dev

# Tests ausführen
pixi run test

# Tests mit Coverage
pixi run test-cov

# Linting (Ruff)
pixi run lint

# Auto-Fix für Lint-Fehler
pixi run lint-fix

# Alle verfügbaren Tasks anzeigen
pixi task list
```

**Alternativ (ohne pixi):** `pip install -e ".[dev,shap]"` in einer virtuellen Umgebung.

### Environments

| Environment | Inhalt | Typischer Einsatz |
|---|---|---|
| `default` | Core-Pipeline + SHAP | Training, Evaluation, Reporting |
| `app` | default + Flask | Web-UI starten (`pixi run app`) |
| `dev` | default + pytest + ruff | Entwicklung und CI |


> **Hinweis LAN:** Für die PyPI-Installation im Firmennetz muss `PIXI_TLS_ROOT_CERTS="all"` und `UV_NATIVE_TLS=true` gesetzt sein. Die `[pypi-options]` in `pixi.toml` verweisen auf den internen Nexus-Mirror.

## Grundprinzipien

1. **Analyse ≠ Produktion**  
   Analyse darf experimentieren; Produktion muss stabil sein.

2. **Bundles sind der Vertrag**  
   Produktion arbeitet ausschließlich mit exportierten Artefakten (Bundle-Verzeichnis).

3. **Registries statt verstreuter Sonderlogik**  
   Neue Learner und Modellvarianten werden über `ModelRegistry` angebunden, nicht über zusätzliche `if/else`-Blöcke in den Runnern.

## Projektstruktur

```text
rubin/
  pipelines/
    analysis_pipeline.py
    production_pipeline.py
    data_prep_pipeline.py
  evaluation/
    drtester_plots.py
  explainability/
    shap_uplift.py
    reporting.py
  reporting/
    html_report.py            ← HTML-Report-Generator (analysis_report.html)
  model_registry.py
  model_management.py
  tuning_optuna.py
  training.py
  preprocessing.py
  feature_selection.py
  artifacts.py
  settings.py
  utils/
    categorical_patch.py       ← EconML-Kompatibilität: kategorische Features
    data_utils.py
    io_utils.py
    schema_utils.py
    uplift_metrics.py
configs/
docs/
run_analysis.py
run_production.py
run_dataprep.py
run_explain.py
run_promote.py
```

## 1) Neuen kausalen Learner ergänzen

### Ort
`rubin/model_registry.py`

### Vorgehen
1. Implementiere eine Factory-Funktion, die eine Modellinstanz erzeugt.
2. Nutze `ModelContext`, um Base-Learner-Typ, Fixparameter und getunte Rollenparameter zu beziehen.
3. Registriere das Modell im `ModelRegistry`.
4. Ergänze in `rubin/settings.py` den Modellnamen in `SUPPORTED_MODEL_NAMES`, damit Konfigurationen strikt validiert bleiben.
5. Prüfe, ob für das Modell task-basiertes Tuning benötigt wird. Falls ja, ergänze die Rollensignaturen in `rubin/tuning_optuna.py`.
6. **Multi-Treatment-Kompatibilität:** Falls das neue Modell Multi-Treatment nicht unterstützt, ergänze den Modellnamen in `_BT_ONLY_MODELS` in `rubin/settings.py`. `_predict_effect()` erwartet, dass kompatible Modelle bei MT ein 2D-Array (n, K-1) zurückgeben. Bei BT genügt (n,) oder (n, 1).

Beispiel-Skizze:

```python
from rubin.model_registry import ModelRegistry, ModelContext
from rubin.tuning_optuna import build_base_learner

def make_my_learner(ctx: ModelContext):
    base = build_base_learner(
        ctx.base_learner_type,
        {**ctx.base_fixed_params, **ctx.params_for("overall_model")},
        seed=ctx.seed,
        task="regressor",  # Meta-Learner Outcome-Modelle sind Regressoren
    )
    return MyLearner(model=base)

registry = ModelRegistry()
registry.register("MyLearner", make_my_learner)
```

Zusätzlich in der YAML:

```yaml
models:
  models_to_train:
    - MyLearner
```

## 2) Neuen Base-Learner ergänzen

### Orte
- `rubin/tuning_optuna.py`
- `rubin/model_registry.py` nutzt denselben Builder indirekt über `build_base_learner(...)`

### Schritte
- Builder-Zweig für Klassifikation und Regression ergänzen
- sinnvolle Default-Search-Spaces hinterlegen
- Search-Space-Dokumentation in `docs/tuning_optuna.md` anpassen
- Beispiel-Konfigurationen aktualisieren

## 3) Neue Metriken ergänzen

Metriken liegen in `rubin/utils/uplift_metrics.py`.

Konventionen:
- Funktionen sind möglichst side-effect-frei
- Inputs und Rückgaben bleiben numerisch und einfach serialisierbar
- neue Metriken sollten in der Analyse-Pipeline sowohl nach MLflow als auch in die JSON-Zusammenfassung geschrieben werden

## 4) Production Pipeline erweitern

Erweiterungen wie zusätzliche Ausgabeformate, Batching oder parallele Verarbeitung gehören nach
`rubin/pipelines/production_pipeline.py` und bei Bedarf in `run_production.py`.

Wichtig:
- keine Trainingslogik in Production
- keine Feature-Selektion in Production
- keine impliziten Schemaänderungen zur Laufzeit

## 5) Task-basiertes Optuna-Tuning

Das Base-Learner-Tuning arbeitet task-basiert und nutzt Optuna (Bayesian TPE) für die Hyperparameter-Optimierung. Das bedeutet:

1. aus `models_to_train` wird ein interner Trainingsplan erzeugt
2. identische Base-Learner-Aufgaben werden dedupliziert
3. die besten Parameter werden anschließend allen passenden Rollen zugeordnet

### Overfit-Penalty (BLT + FMT)

Sowohl beim Base-Learner-Tuning als auch beim Final-Model-Tuning kann eine Train-Val-Gap-Penalty aktiviert werden (`overfit_penalty`). Die Formel: `adjusted = val_score - penalty × max(0, gap - tolerance)`. Ein großer Gap (train_score >> val_score) wird bestraft. Die Penalty ist über Presets in der UI steuerbar (Aus / Moderat / Stark) oder individuell pro Tuning-Stufe einstellbar.

### Final-Model-Tuning (FMT)

Beide FMT-Modelle (NonParamDML, DRLearner) nutzen äußere OOF-CV mit `est.score()`: model_final wird auf Train gefittet und auf komplett Out-of-Fold-Val bewertet. Dies verhindert optimistische R-Score-Schätzungen. Die Architektur ist für beide Modelle identisch.

### GRF-Tuning

CausalForestDML nutzt EconML's `.tune()` (OOB-basiert). CausalForest nutzt einen eigenen Grid-Search über Wald-Parameter (12 oder 48 Kombinationen) mit R-Loss-Evaluation auf dem ersten CV-Fold. Nicht-getunte Parameter verwenden EconML-Defaults.

### Surrogate-Einzelbaum

Der Surrogate trainiert auf den **Train-Predictions** des Champions (Full-Data-Refit), um die gelernte CATE-Funktion bestmöglich nachzulernen. Cross-Validation erzeugt OOS-Predictions für faire Vergleichbarkeit mit dem Champion. Abschließend wird der Surrogate auf allen Daten nachtrainiert.

