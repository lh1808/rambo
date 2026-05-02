# Changelog

## [Unreleased]

### Hinzugefügt
- **Dual-Seed-System:** Separater `tuning_seed` (Default: 18) für Tuning-CV-Folds, verhindert Val-Set-Overfitting. Cross-Prediction nutzt `SEED` (42), Tuning nutzt `tuning_seed` (18). UI-Slider + Warnung bei gleichen Seeds.
- **Produktions-Datenregime:** BLT-Tasks für DML/DR-Nuisance subsamplen die Trainingsmenge auf (K-1)/K, um das DML-interne Cross-Fitting zu simulieren. Scopes bleiben getrennt (7 Tasks).
- **RAM-Monitoring:** `_log_ram()` an 8 Phasen-Übergängen mit Warnung ab 80% Auslastung.
- **FMT Crash-Resilienz:** try/except pro FMT-Modell — DRLearner-Crash verliert nicht mehr NonParamDML-Ergebnisse.
- **Pipeline-Log Validierungsmodus:** Expliziter Log nach Pipeline-Start (Cross-Val mit Seeds / External / TMEO).

### Geändert
- **GRF-Tuning → CausalForest-Tuning:** Konsequente Umbenennung in UI, Docs und Logs. "EconML tune()" → "RScorer-Grid-Search".
- **FMT Score-Labels:** `OOF-R-Score` → `OOF-R-Loss` (NonParamDML) / `OOF-DR-MSE` (DRLearner).
- **Cross-Fitting Log:** "DML Cross-Fitting" → "Nuisance Cross-Fitting (DML + DR)".
- **FMT nutzt volle Trainingsdaten:** 80%-Subsetting entfernt — Dual-Seed-Schutz ist ausreichend.
- **GC-Sweeps:** `gc.collect()` + `del est` / `del study` an allen Phasen-Übergängen (8× gc, 2× del est, 2× del study).

### Entfernt
- `_first_crossfit_train_indices()` (26 Zeilen Dead Code, ersetzt durch Dual-Seed-Schutz).
- `crosspred_splits` Parameter aus `tune_final_model()`.
- 68 Zeilen Dead JSX (Tag, ColumnPicker, ROLES, BT_PRESETS, MT_PRESETS, etc.).


Alle relevanten Änderungen am rubin-Framework. Neueste Einträge oben.

## 2026-04-22 (Session 11 — CV-Harmonisierung, Tuning-Metrik, Signatur-Trennung)

### Kritische Fixes (Modellqualität)

- **Alle internen CVs auf EconML-Default (2) gesetzt:** BLT `tuning.cv_splits`, FMT `final_model_tuning.cv_splits`, DML `dml_crossfit_folds` — jeweils unabhängig von `cross_validation_splits` (5, äußere Evaluation). Behebt Inkonsistenz, bei der interne CVs versehentlich auf 3–5 standen und die Tuning-/Produktions-Datenmengen nicht zusammenpassten.
- **FMT `cv_splits` als eigenes Config-Feld wiederhergestellt** (Default 2). War entfernt worden, wodurch FMT `cross_validation_splits` (5) nutzte — 2.5× langsamer pro Trial und andere Scoring-Dynamik als Produktion.
- **FMT `stability_penalty` auf 0.3 zurückgesetzt** (war 0.0). Ohne Penalty fand Optuna aggressive model_final-Params, die out-of-sample schlecht generalisierten.
- **Tuning-Metrik von `pr_auc` auf `log_loss` zurückgesetzt.** Log-Loss misst Kalibrierung — essentiell für DML-Residualisierung `Y − E[Y|X]`. PR-AUC optimiert Ranking, was gut kalibrierte Nuisance-Vorhersagen nicht garantiert. Betrifft: settings, tuning_optuna, html_report, UI, config_reference, docs.
- **FMT DRLearner: Äußere Score-Folds = `cross_validation_splits` (5),** interne DRLearner cv = `final_model_tuning.cv_splits` (2). Vorher wurden beide aus demselben Feld gelesen.
- **Pipeline: `_dml_cv` liest jetzt `dml_crossfit_folds`** (Default 2), nicht mehr `cross_validation_splits` (5).

### Task-Sharing Signatur-Trennung

- **Neue `sample_scope`-Werte:** `"all"` (DML/DR-intern, ~40% N in Produktion) vs. `"all_direct"` (Meta-Learner, ~80% N). Verhindert, dass Modelle mit unterschiedlichen effektiven Trainingsmengen dasselbe Tuning teilen.
- **Betroffene Tasks:**
  - `outcome_regression`: DRLearner.model_regression (all) ↔ SLearner.overall_model (all_direct) — jetzt getrennt.
  - `propensity`: DML/DR (all) ↔ XLearner.propensity_model (all_direct) — jetzt getrennt.
- **7 Tasks statt 5** bei vollem Modell-Set (bei per_role/per_learner=false).

### Fit-Berechnungen korrigiert

- **`estimateFits`:** DML-internes cv von `cvSplits` (5) auf `dmlCv` (2) korrigiert → `fitsPerFit` von 11 auf 5 pro DML/DR-Modell.
- **`estimateFits` BLT:** Innere CV von `cvSplits` (5) auf `bltCv` (2) korrigiert → halbiert die BLT-Fit-Schätzung.
- **`estimateFits` BLT Tasks:** Spiegelt jetzt die Signatur-Trennung wider (7 Tasks).
- **`FinalTuningPlanPreview` NonParamDML:** Korrigiert von 1 fit/trial auf `cv×2+1 = 5` fits/trial (RScorer-Setup + full NonParamDML.fit() pro Trial).
- **`FinalTuningPlanPreview` DRLearner:** Korrigiert von `nCv` (2) outer folds auf `outerCv` (5), plus interne Fits pro Fold (5) = 25 fits/trial.
- **CausalForestDML Tune Preview:** Nuisance-Fits von `cvSplits×2` (10) auf `dmlCv×2` (4) korrigiert.

### Sonstige Änderungen

- **`default`-Key aus `tuned_baselearner_params.json` und MLflow-Params gefiltert.** War ein interner Fallback-Lookup-Key, der User-facing keinen Mehrwert hatte.
- **savePreds/predsFormat aus UI entfernt** (Cross-Predictions werden immer automatisch in MLflow geloggt).
- **External Eval UI redesigned:** Kompaktere Card, klarere Input-Labels (X/Features, Y/Outcome, T/Treatment, S/Score), orphaned Closing-Tags entfernt.
- **UI Help-Texte aktualisiert:** cvSplits beschreibt jetzt nur äußere Cross-Predictions; DML-internes cv separat erklärt.
- **YAML-Emission:** `tuning.cv_splits` und `dml_crossfit_folds` werden nicht mehr emittiert (Backend-Defaults greifen). Nur bei Abweichung vom Default.
- **Config Reference:** Alle neuen Defaults dokumentiert (cv_splits=2, dml_crossfit_folds=2, stability_penalty=0.3, metric=log_loss).

## 2026-04-17 (Session 10 — Class Balance, Verzeichnis-Konsolidierung, UI-Polish)

### Evaluiert & entfernt: Class-Balance / Imbalance-Weighting
- **`class_balance`-Feature evaluiert und entfernt:** Tunbares `scale_pos_weight` (Optuna) + `CalibratedClassifierCV` wurden implementiert und getestet. Bei extremer Imbalance (1:1000) war der Effekt auf PR-AUC minimal (0.015 → 0.017), da `scale_pos_weight` primär die Kalibrierung ändert, nicht das Ranking. Die Split-Struktur von Gradient-Boosting-Modellen ist bereits ohne Gewichtung auf die seltene Klasse fokussiert (Log-Loss-Gradienten). Das Feature wurde entfernt, da es ein selbstverursachtes Kalibrierungsproblem erzeugt, das dann per `CalibratedClassifierCV` (3× Trainingszeit) repariert werden muss — ohne Netto-Mehrwert.

### Verzeichnis-Konsolidierung
- **Alle Server-Schreibpfade unter `runs/`:**
  - Feature Dictionary: `ROOT/data/` → `WORK_DIR/exports/`
  - Config Save: `ROOT/configs/` → `WORK_DIR/configs/`
  - Config Import: `ROOT/config_imported.yml` → `WORK_DIR/.rubin_cache/`
  - Edge-Case-Fix: `relative_to(ROOT)` Fallback bei externem WORK_DIR.
- **Bundle-Pfad `runs/bundles` konsistent:** `settings.py` (`BundleConfig.base_dir`), `artifacts.py` (`ArtifactBundler.__init__`), `analysis_pipeline.py` (Fallback), UI (DEFAULT_CFG, YAML-Emission), 16 Config-Templates, Docs, `run_promote.py` Docstring.
- **`.gitignore` bereinigt:** Keine Legacy-Sektion, nur `runs/` + Standard-Ignores. `configs/` (Template-Configs) wird korrekt NICHT ignoriert.

### Repo-Struktur
- **`uplift_metrics.py` verschoben:** `utils/uplift_metrics.py` → `evaluation/uplift_metrics.py`. Kein Backward-Compat-Shim (Grundversion). Imports in `analysis_pipeline.py` + `test_uplift_metrics.py` aktualisiert.
- **Toter Code entfernt:** `grouped_outcome` Classifier-Variante (Methode `_objective_grouped_outcome`, Dispatch, FLAML-Set, UI-Maps, Pipeline-Label-Map) — `_role_signature()` gibt diesen Wert nie zurück.

### UI-Optimierungen
- **Datenpfade-Seite redesigned:** 1 zusammenhängender `<Sec>`-Block statt 2. Uppercase-Sektions-Labels (TRAININGSDATEN, EXTERNAL EVAL, BENCHMARK). Eval-Felder nur bei aktivem Toggle sichtbar. Benchmark nur bei gesetzter S-Datei.
- **Ausgabe vereinfacht:** `outputDir` in Expander „Erweiterte Ausgabe-Optionen" verschoben. Info-Text klarstellt: „Ergebnisse werden automatisch in MLflow geloggt."
- **Bundle-Export vereinfacht:** `bundleDir`-Eingabefeld entfernt, nur Hinweistext `runs/bundles/<timestamp-id>/`.
- **Arbeitsverzeichnis:** Vollständige Ordnerstruktur als Tree-Darstellung. Label: „Eigenes Arbeitsverzeichnis (optional)".
- **FM-Tuning Labels:** `FMT` → `FM-Tuning`, `FMT Intensiv` → `FM-Tuning Intensiv`. User-visible Text konsequent angepasst (FM-Tuning-Trials, FM-Tuning verwendet).
- **Sprachmischung bereinigt:** `Higher is better` → `Höher = besser`, `Stability Penalty` → `Stabilitäts-Penalty`.
- **Placeholder-Pfade:** `data/raw/eval_data.csv` → `runs/data/eval_data.csv`, `data/feature_dictionary.xlsx` → `runs/exports/feature_dictionary.xlsx`.
- **DataPrep Pflicht-Spalten:** Target- und Treatment-Felder starten jetzt leer (nicht vorbefüllt mit Y/T). Placeholders: „z.B. Y, OUTCOME, ..." / „z.B. T, TREATMENT, ...".

### Config-Templates
- `config_reference_all_options.yml`: `class_balance` mit Kommentar hinzugefügt.
- `config_grf_focus.yml`: Copy-Paste-Fix `output_dir: runs/dml_focus` → `runs/grf_focus`.

## 2026-04-16 (Session 9 — UX + Tiefenprüfungen)

### MLflow-Overview-Metriken
- **Champion Top-Level-Logging:** Nach der Champion-Auswahl werden jetzt dedizierte Metriken und Tags geloggt, die in der MLflow-Experiment-Übersicht als Columns sichtbar gemacht werden können (einmalige Auswahl in der UI, wird im Browser-localStorage persistiert):
  - `champion_score` (Metrik): Wert der konfigurierten Selection-Metric für den Champion
  - `champion_qini`, `champion_auuc`, `champion_policy_value` (Metriken): Zusätzliche Performance-Werte für Schnellvergleich
  - `rubin.champion_model` (Tag): Name des Champions (z.B. "NonParamDML")
  - `rubin.selection_metric` (Tag): Konfigurierte Metrik (z.B. "qini")
  - `rubin.learner` (Tag): Verwendeter Learner ("catboost", "lgbm", "both")
  - `rubin.validation_mode` (Tag): Validierungsmodus ("cross", "external", "TMEO")

### UX-Verbesserungen
- **Benchmark ohne Preset:** Das Add-On-Preset `benchmark` war ein Phantom (`cfg: {}`, keine Wirkung) und wurde entfernt. Der Benchmark-Vergleich gegen den historischen Score läuft **automatisch**, sobald eine S-Datei in `data_files.s_file` angegeben ist. Die Benchmark-Sektion auf der Datenpfade-Seite zeigt jetzt einen grünen Success-Badge bei gesetzter S-Datei, sonst einen gelben Hinweis mit Verweis.
- **External Eval immer zugänglich:** Die Eval-Datei-Felder sind jetzt permanent auf der Datenpfade-Seite sichtbar, nicht mehr conditional versteckt hinter `validateOn==="external"`. Ein Toggle „External Eval aktivieren" direkt neben der Überschrift macht den Modus explizit. Bei Eintragen/Hochladen einer Eval-X/T/Y-Datei wird `validateOn` automatisch auf "external" gesetzt.
- **Validation-Addon-Kategorie entfernt:** Nach den obigen Änderungen war die Validation-Gruppe bei den Add-Ons doppelt redundant (Toggle auf Datenpfade + Button auf Validierungs-Seite + Auto-Aktivierung). Das einzige verbliebene Preset `external_eval` und die leere Gruppe wurden entfernt. Layout: R4 (Erklärbarkeit) und R5 (Production) stehen jetzt in einer Reihe nebeneinander.
- **Ensemble-Preset entfernt:** Das Add-On-Preset `ensemble` (setzte nur `ensembleEnabled: true`) war redundant, da der Ensemble-Toggle bereits direkt auf der Modell-Seite existiert. Production & Export enthält jetzt nur noch "Bundle & Surrogate".
- **Kollisions-Warnung:** Bei gleichzeitig gesetztem `eval_mask_file` und `validateOn="external"` zeigt die UI jetzt eine Warn-Box auf der Validierungs-Seite. Backend ignoriert die Maske in diesem Fall (External-Eval-Dateien haben Vorrang), die UI macht das explizit.

### UI-Pre-Validation erweitert
Neue Checks in der Pre-Run-Validierung, die Backend-Validators spiegeln. Damit sieht der User die Fehler bereits im Validierungs-Panel statt erst beim Pydantic-Abbruch beim Run-Start:
- **Manual Champion muss in Models-Liste sein**: Wenn `selection.manual_champion` gesetzt ist, prüft die UI jetzt, dass der Name auch in `models.models_to_train` enthalten ist.
- **`fixed_params` bei `baseLearner="both"` muss nested sein**: Flache Parameter wie `{num_leaves: 31}` würden silent an beide Learner durchgereicht und CatBoost-Trials zum Absturz bringen. Die UI prüft jetzt auf `{lgbm:{...}, catboost:{...}}`-Struktur.
- **`reference_group` bei binary nur 0 erlaubt**: Bei `treatment.type="binary"` sind andere Werte als 0 aktuell nicht unterstützt.
- **MT-incompatible Selection-Metrics**: Bei `treatment.type="multi"` sind die BT-only-Metriken (`qini`, `auuc`, `uplift_at_10pct`, `uplift_at_20pct`, `uplift_at_50pct`, `policy_value`) nicht verfügbar. Die UI empfiehlt stattdessen `policy_value_T1`, `qini_T1` etc.

### Tuning-Fairness & Logging
- **Timeout bei `both`-Modus deaktiviert:** Bei `base_learner.type="both"` wird `timeout_seconds` (BLT + FMT) automatisch ignoriert und eine Warnung geloggt. Grund: LGBM-DART ist ~3× langsamer als CatBoost — ein Timeout würde zu weniger fertig durchgelaufenen LGBM-Trials führen und damit den Vergleich verzerren. Steuerung erfolgt ausschließlich über `n_trials`. UI zeigt bei Timeout + "both" einen gelben Hinweis.
- **FMT-Parallelisierung reduziert:** `FinalModelTuner._tuning_n_jobs` nutzt jetzt `n_cpus // 8` statt `n_cpus // 4`. Jeder FMT-Trial fittet ein vollständiges Kausalmodell (model_y + model_t + model_final mit Cross-Fitting) und verbraucht ~5-10× mehr RAM als ein BLT-Trial. Bei 40 Kernen: 5 statt 10 parallele Trials, dafür je 8 statt 4 Kerne pro Fit.
- **RCT-Propensity-Cap in Tuning-Plan-UI:** `TuningPlanPreview` berücksichtigt jetzt den Backend-Cap (`min(n_trials, 20)` bei RCT + Propensity). Fits-Berechnung, Zeilen-Badge und Info-Box sind korrekt. Zuvor wurde die volle Trial-Zahl angezeigt.
- **BLT-Log bereinigt:** Der irreführende `Ø Score` (Mittelwert über PR-AUC und neg_MSE — verschiedene Skalen) und die pauschale `Trials/Task`-Angabe (bei RCT variieren die Trials pro Task) wurden entfernt. Stattdessen werden die Einzel-Scores pro Task ausgegeben.

### Tiefenprüfungen ohne Findings
- UI cfg-Keys vs YAML_TO_CFG-Mapping: alle Keys gemappt oder als UI-only dokumentiert
- Backend-Felder ohne UI-Exposition: 9 technische Defaults (`storage_path`, `study_name_prefix` etc.) bewusst Backend-only
- Pydantic strict-mode Validierung der generierten YAML: alle Felder in Schemata definiert, `extra="forbid"` hält
- `fixed_params` bei `base_learner.type="both"`: Struktur `{catboost:{...}, lgbm:{...}}` ist im Backend korrekt handled (`build_base_learner` wählt anhand `_learner_type` und verwirft den nicht-gewählten Sub-Dict)
- Feature-Selection bei TMEO: läuft auf bereits reduziertem X (nach Train/Holdout-Split) — leakage-frei
- `dml_crossfit_folds`: wird zur Laufzeit auf `cross_validation_splits` gesetzt; bewusst kein separater UI-Toggle

---

## 2026-04-16 (Session 8)

### Korrektheit
- **TMEO Data-Leakage-Fix:** Bei `eval_mask_file` (Train-Many-Evaluate-One) wurden zuvor Mask-Rows sowohl zum Training als auch zur Evaluation genutzt → optimistische Metriken. Neue Logik spaltet vor dem Training: `~mask` → X/T/Y für Training, `mask` → Holdout-Eval-Set. TMEO läuft jetzt effektiv wie External Eval. MLflow loggt `tmeo_train_n` und `tmeo_eval_n`. Betroffen: `rubin/pipelines/analysis_pipeline.py`.
- **Bundle-Export Konsistenz bei TMEO:** Der `X_full`-Snapshot für Bundle-Refit wird jetzt NACH dem TMEO-Split erstellt, damit das Bundle-Modell nur auf Trainingsdaten refittet wird (nicht auf dem gesamten Datensatz inkl. Eval-Zeilen).
- **HTML-Report TMEO-Erkennung:** Der Report zeigt jetzt alle drei Validierungsmodi korrekt an (`cross (5 Folds)`, `external (separater Datensatz)`, `TMEO (Mask-Holdout)`). Summary-Bar zeigt bei TMEO zusätzlich `Eval (TMEO): N Beob.`.

### YAML-Roundtrip-Bugs (UI ↔ Backend)
- **Search-Space bei `baseLearner="both"`:** Der UI-YAML-Emitter emittierte bei "both" gar keinen `search_space`-Block → customized Ranges gingen silent verloren. Fix: Emittiert jetzt beide Learner-Sektionen wenn customized.
- **Search-Space Import:** `parseYamlToCfg` las `search_space` gar nicht (nur 1-Ebenen-Parser). Neuer verschachtelter Parser für `tuning.search_space.{catboost,lgbm}` und `final_model_tuning.search_space.{catboost,lgbm}`.
- **Regex-Parser-Fix:** `[a-z_]+` erlaubte keine Ziffern → Parameter mit Zahlen im Namen (`l2_leaf_reg`, `reg_alpha`) wurden silent verschluckt. Fix: `[a-z0-9_]+`.
- **`log`-Flag Durchgängigkeit:** UI-Search-Space-Defs hatten kein `log: true` Flag → bei Änderung von log-skalierten Params (`learning_rate`, `l2_leaf_reg`, `random_strength`, ...) emittierte die UI die Range ohne `log: true` → Backend wechselte still von log-uniform zu linearem Sampling. Fix: UI-Defs ergänzt, Emitter gibt `log: true` aus, `low` wird automatisch auf `1e-6` geclampt falls User `low=0` setzt (Optuna-Constraint bei `log=True`). SSEditor zeigt `(log)`-Suffix am Parametername.
- **PData-Scope:** `setSp`/`setSpFmt` wurden nicht an `PData` durchgereicht → YAML-Import konnte Search-Spaces nicht in UI-State restaurieren. Fix: Props durchgereicht, Import extrahiert `__sp`/`__spFmt` und ruft Setter.

### Tuning-Qualität
- **TPESampler `group=True`:** Bei `base_learner.type="both"` fiel TPE bei CatBoost-spezifischen Parametern (`l2_leaf_reg`, `random_strength`, `rsm`, `min_data_in_leaf`, `model_size_reg`, `leaf_estimation_iterations`) auf `RandomSampler` zurück, weil der Parameter-Set zwischen Trials variierte (conditional search space). Fix: `group=True` partitioniert die Trial-Historie — TPE lernt separate KDE-Modelle für LGBM- und CatBoost-Zweig. Warnungen verschwinden, Tuning-Qualität bei "both" deutlich verbessert. Doppelter Fallback bei alten Optuna-Versionen.

### Explainability
- **SHAP-Analyse:** Zweistufiger Fallback — EconML SHAP-Plot-Satz (Summary, CATE-Profile, Dependence, Scatter) → generische SHAP-Werte (TreeExplainer). Importance-CSV und Plots als MLflow-Artefakte. Auch als CLI-Runner (`run_explain.py`).

### UX / UI
- **FLAML+both Warnung:** UI zeigt jetzt vor dem Run einen roten Hinweis bei Kombination `baseLearner="both"` + `tuningAutoml="flaml"`, dass Optuna als Fallback verwendet wird (FLAML unterstützt keine kategorische Learner-Wahl).
- **Modus-Badges in Pre-Run-Summary und während des Runs:** Cross (🔵) / External (🟣) / TMEO (🟠) farblich differenziert.
- **Progress-Steps-Label:** "Training & Cross-Predictions" → "Training & Predictions" (neutral für alle Validierungsmodi).
- **Addon-Preset "Benchmark" entfernt:** Hatte leeres `cfg: {}`, bewirkte nichts — Benchmark läuft ohnehin automatisch bei gesetztem S-File.

### Dokumentation
- `docs/evaluation.md`: TMEO-Sektion vollständig umgeschrieben (Leakage-free Routing)
- `docs/konfiguration.md`: `eval_mask_file`, `validate_on`, `shap_values.method` aktualisiert
- `docs/tuning_optuna.md`: TPE `group=True` dokumentiert, FLAML+both-Einschränkung ergänzt
- `docs/explainability.md`: `shap_values.method` ergänzt
- `docs/architektur.md`: HTML-Report TMEO-Badges dokumentiert
- `docs/app_build.md`: Sync-Workflow für Index.html und Overview.html ergänzt

---

## Ältere Änderungen

Frühere Änderungen sind nicht in einem formalen Changelog erfasst. Siehe Git-Log oder die thematisch gegliederten Docs für Details.
