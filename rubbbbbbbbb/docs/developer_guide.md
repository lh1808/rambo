# Entwicklerhandbuch – rubin

Dieses Dokument richtet sich an Entwicklerinnen und Entwickler, die **rubin** erweitern oder bestehende Teile umbauen.
Der Fokus liegt auf einer konsistenten Struktur, klaren Zuständigkeiten und stabilen Produktionsartefakten.

## Entwicklungsumgebung einrichten

[Pixi](https://pixi.sh) ist das empfohlene Tool für das Environment-Management.
Es verwaltet Python, conda-forge- und PyPI-Pakete einheitlich und erzeugt ein
reproduzierbares Lockfile (`pixi.lock`).

```bash
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

### Commit-Gate (pre-commit)

Zusätzlich zum vollen `lint` gibt es ein schmales **Blocking-Gate**, das nur echte
Bugs abfängt — *undefined names* (`F821`) und Syntaxfehler (z. B. Backslash in
f-strings auf Python < 3.12). Diese Klasse hat in der Vergangenheit zu stillen
Laufzeitfehlern geführt (fehlender Import, falscher Variablenname). Das Gate ist
bewusst enger als `pixi run lint`, damit es ab Tag 1 grün ist und nicht an
Stil-Findings (unbenutzte Imports etc.) scheitert.

```bash
# Einmalig pro Klon: Hook installieren
pixi run hooks-install        # entspricht: pre-commit install

# Manuell über alle Dateien (z. B. für CI)
pixi run gate                 # ruff check --select F821 --target-version py310 ...
```

Danach läuft das Gate automatisch bei jedem `git commit`; ein Commit mit einem
undefined name oder Syntaxfehler wird abgelehnt. Der Hook ist als `repo: local`
definiert und nutzt das Ruff aus der Umgebung — es wird **kein externes
Hook-Repo** geladen (firmennetz-/Nexus-tauglich).

**CI-Anbindung (plattformunabhängig):** In der Pipeline genügt ein Schritt, der
`pixi run gate` (oder `ruff check --select F821 --target-version py310 rubin/ app/
tests/ run_*.py`) ausführt; ein Exit-Code ≠ 0 lässt den Build fehlschlagen. Beispiel
für eine GitLab-CI-Job-Definition:

```yaml
lint-gate:
  stage: test
  script:
    - pixi run gate
```

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
  tuning/                    # Paket: common, base_learner, final_model, causal_forest
  training.py
  preprocessing.py
  feature_selection.py
  artifacts.py
  settings.py
  utils/
    categorical_patch.py       ← EconML-Kompatibilität: kategorische Features
    data_utils.py
    io_utils.py
    plot_theme.py              ← Rubin-Farbpalette (Ruby, Gold, Slate)
    run_names.py               ← MLflow Run-Name-Generator
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
5. Prüfe, ob für das Modell task-basiertes Tuning benötigt wird. Falls ja, ergänze die Rollensignaturen in `rubin/tuning/ (Paket)`.
6. **Multi-Treatment-Kompatibilität:** Falls das neue Modell Multi-Treatment nicht unterstützt, ergänze den Modellnamen in `_BT_ONLY_MODELS` in `rubin/settings.py`. `_predict_effect()` erwartet, dass kompatible Modelle bei MT ein 2D-Array (n, K-1) zurückgeben. Bei BT genügt (n,) oder (n, 1).

Beispiel-Skizze:

```python
from rubin.model_registry import ModelRegistry, ModelContext
from rubin.tuning.common import build_base_learner

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
- `rubin/tuning/ (Paket)`
- `rubin/model_registry.py` nutzt denselben Builder indirekt über `build_base_learner(...)`

### Schritte
- Builder-Zweig für Klassifikation und Regression ergänzen
- sinnvolle Default-Search-Spaces hinterlegen
- Search-Space-Dokumentation in `docs/tuning_optuna.md` anpassen
- Beispiel-Konfigurationen aktualisieren

## 3) Neue Metriken ergänzen

Metriken liegen in `rubin/evaluation/uplift_metrics.py`.

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

### Evaluation-Module

`rubin/evaluation/drtester_plots.py` ist ein Backward-Kompatibilitäts-Shim, der aus 3 Modulen re-exportiert:
- `drtester_core.py`: CustomDRTester, CustomEvaluationResults, DrTesterPlotBundle, Nuisance-Fitting
- `evaluation_plots.py`: CATE-Verteilungen, ATE-Barplots, native Uplift-Kurven, evaluate_cate_with_plots
- `score_plots.py`: Qini-Vergleiche, Policy-Value-Plots, Score-Redistribution

Bestehende Imports (`from rubin.evaluation.drtester_plots import ...`) funktionieren weiterhin.

### Overfit-Penalty (BLT + FMT)

Sowohl beim Base-Learner-Tuning als auch beim Final-Model-Tuning kann eine Train-Val-Gap-Penalty aktiviert werden (`overfit_penalty`). Die Formel: `adjusted = val_score - penalty × scale × min(max(0, relative_gap - tolerance), max_penalized_gap)` mit `scale = max(|val_score|, 1e-8)` und `relative_gap = (train_score - val_score) / scale`. Ein großer Gap (train_score >> val_score) wird bestraft; der `overfit_max_penalized_gap`-Cap (Default 1.0, `<=0` = aus) saturiert den Abzug und verhindert, dass die Penalty bei kleinem Val-Score das Vorzeichen kippt. Die Penalty ist über Presets in der UI steuerbar (Aus / Moderat / Stark) oder individuell pro Tuning-Stufe einstellbar.

### Final-Model-Tuning (FMT)

Beide FMT-Modelle (NonParamDML, DRLearner) nutzen äußere OOF-CV. Der Scorer ist konfigurierbar (`scorer: auto|qini|rscore`): Bei RCT (Default) wird der **QiniScorer** verwendet — aggregierter OOF-Qini über alle Folds, direkt auf Ranking-Qualität optimiert, kein Pruning. Bei Beobachtungsdaten wird der EconML **RScorer** verwendet (unabhängige Nuisance, 2-Fold T×Y-stratifiziertes Cross-Fitting auf Val-Daten, Pruning möglich). Overfit-Penalty ist skalen-sicher (relative Tolerance).

### CausalForest-Tuning (CFT)

CausalForestDML und CausalForest werden via Optuna TPE über 4 kausale Parameter getunt (max_depth, min_weight_fraction_leaf, min_var_fraction_leaf, criterion). Scorer analog FMT (`scorer: auto|qini|rscore`). CausalForestDML: Nuisance wird einmalig gecacht (`cache_values=True`), Trials ändern Forest-Parameter via `setattr()` + `refit_final()` (CausalForestDML hat kein `set_params()`). CausalForest: Pro Trial frischer Estimator mit Params im Konstruktor. Bei RCT: DummyClassifier für model_t im Cache. Siehe `docs/tuning_optuna.md → CausalForest-Tuning (Optuna)` für Details.

### Surrogate-Einzelbaum

Der Surrogate-Einzelbaum nutzt **Fold-Aligned Predictions** — ein komplett leakage-freies Verfahren:

1. **K-Fold CV (Evaluation, komplett leakage-frei):** Während der Cross-Prediction des Champions werden K Modelle trainiert (eins pro Fold). Jedes Champion_k wurde auf allen Daten OHNE Fold k trainiert. Champion_k predictet auf dem gesamten Datensatz. Für Surrogate-Fold k (Val) wird Champion_k's Prediction auf den Train-Folds als Target verwendet — Champion_k hat Fold k NIE gesehen → kein Informationspfad von Val-Samples ins Training-Target.

2. **Final-Fit (Produktion):** Der Surrogate wird auf den **Full-Data-Refit-Predictions** (`Train_<Champion>`) des Champions trainiert. Dieses Modell wird für das Produktions-Scoring im Bundle verwendet, nicht für die Evaluation.

**Leakage-Beweis (Fold k = Val):**
```
Fold k Daten → Champion_k Training?   NEIN (exkludiert)  ✓
Fold k       → Training-Target?       NEIN (nur Champion_k-Preds)  ✓
Fold k       → Surrogate Training?    NEIN (Val, nicht Train)  ✓
```

**Ensemble als Champion:** Wenn der Champion das Ensemble ist, werden die Fold-Aligned Predictions aus den Einzelmodellen gemittelt. Jedes Einzelmodell hat eigene `fold_aligned_preds[k]` — der Durchschnitt pro Fold ist ebenfalls leakage-frei, weil jeder Champion_k Fold k nie gesehen hat.

**Fallback:** Wenn Fold-Aligned Predictions nicht verfügbar sind (z.B. bei älteren Pipeline-Runs oder wenn weniger als 2 Modelle fold_aligned_preds haben), wird auf OOF-Predictions zurückgegriffen (indirektes Leakage, aber akzeptabel).

**Kosten:** Ein zusätzlicher `predict(X_all)` pro Fold während der Cross-Prediction (~1-2 Sekunden pro Fold). Kein einziger zusätzlicher Fit-Call.

