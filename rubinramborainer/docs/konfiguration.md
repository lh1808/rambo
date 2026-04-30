# Globale Konfiguration (config.yml)

In **rubin** wird das Verhalten der Analyse‑ und Production‑Pipelines über eine zentrale YAML‑Datei gesteuert
(`config.yml`). Ziel ist eine Konfiguration, die

- reproduzierbar (gleicher Run → gleiche Einstellungen),
- nachvollziehbar (jede Stellschraube ist dokumentiert),

bleibt.

## Grundprinzip: Was wird wo konfiguriert?

- **In `config.yml`** stehen alle fachlich/technisch sinnvollen Pipeline‑Einstellungen (Datenpfade, Feature‑Filter,
  Modellliste, Base‑Learner, Tuning, Champion‑Auswahl, Explainability‑Voreinstellungen, optionale lokale Outputs).
- **Auf der Kommandozeile** werden nur *Run‑Parameter* gesetzt, die typischerweise von Job zu Job variieren
  (Pfad zur Konfigdatei sowie optional gezielte Overrides wie Bundle‑Export oder Bundle‑Zielordner).

### Priorität (wichtig in der Praxis)

1. **Kommandozeile** (z. B. `--config`, `--bundle-dir`)  
2. **`config.yml`** (alle inhaltlichen Einstellungen, inkl. Bundle-Block)  
3. **Voreinstellungen in `settings.py`** (wenn ein Feld in der YAML fehlt)

## Minimales Beispiel

Hinweis zur Validierung

Die YAML wird beim Laden **strikt validiert**. Unbekannte Schlüssel (z. B. Tippfehler wie `baselearner` statt `base_learner`) führen zu einer klaren Fehlermeldung.
Das ist bewusst so gewählt, um stille Fehlkonfigurationen früh zu vermeiden.


```yaml
data_files:
  x_file: "data/X.parquet"
  t_file: "data/T.parquet"
  y_file: "data/Y.parquet"

mlflow:
  experiment_name: "rubin"

constants:
  SEED: 42
  tuning_seed: 18

models:
  models_to_train: ["SLearner", "TLearner"]

base_learner:
  type: "lgbm"
  fixed_params: {}

tuning:
  enabled: false
  search_space: {}

selection:
  metric: "qini"
  higher_is_better: true
  refit_champion_on_full_data: true
  manual_champion: null
```

---

# Referenz: Alle Konfigurationsbereiche

## `study_type` – Studientyp

```yaml
study_type: "rct"   # "rct" (Default) | "observational"
```

Bestimmt, ob die Daten aus einem randomisierten Experiment (RCT) oder aus Beobachtungsdaten stammen. Dies beeinflusst:

- **Modell-Empfehlungen:** Bei Beobachtungsdaten werden nur Modelle mit Confounding-Korrektur empfohlen (NonParamDML, ParamDML, DRLearner, CausalForestDML). SLearner, TLearner und CausalForest sind weiterhin auswählbar, aber ohne Confounding-Schutz.
- **Evaluationen:** Bei Beobachtungsdaten wird der naive Policy Value nicht berechnet (durch Selektionsbias verzerrt) und die Treatment Balance Curve übersprungen (zeigt nur Confounding-Muster). Die DR-korrigierten Policy Values aus dem DRTester bleiben verfügbar.
- **Propensity-Tuning:** Bei RCT wird das Propensity-Tuning auf 20 Trials reduziert (Diagnose-Check statt volles Tuning). Eine Warnung erscheint, wenn das Propensity-Modell das Treatment vorhersagen kann — das deutet auf Probleme mit der Randomisierung hin.
- **Report-Interpretation:** Angepasste Texte erklären, wie die Metriken im jeweiligen Kontext zu interpretieren sind.

Kein Modell wird blockiert — der Studientyp steuert Defaults, Warnungen und Evaluationen, nicht die Auswahl.

### Mechanismen der Confounding-Korrektur

| Modell | Mechanismus | Confounding-Korrektur |
|---|---|---|
| **NonParamDML, ParamDML** | Orthogonalisierung (Partialling-Out) | Nuisance-Residuen bereinigen Y und T um Kovariaten-Einfluss. CATE wird auf orthogonalisierten Residuen geschätzt. Doubly robust. |
| **DRLearner** | DR-Pseudo-Outcomes (AIPW) | Kombiniert Outcome- und Propensity-Schätzung zu einem doubly-robust Pseudo-Outcome. Konsistent bei Misspecification einer Nuisance-Stufe. |
| **CausalForestDML** | DML-Orthogonalisierung + Honest Splitting | Orthogonalisierte Residuen + Causal Forest mit ehrlicher Schätzung. Valide Konfidenzintervalle. |
| **XLearner** | Propensity-Gewichtung | CATE aus zwei Gruppen, gewichtet mit Propensity. Teilweise Korrektur, weniger robust als DML/DR. |
| **SLearner, TLearner, CausalForest** | Keine | Schätzen Korrelation, nicht Kausalität. Bei Beobachtungsdaten nur als Baseline geeignet. |

Referenzen: Chernozhukov et al. (2018) — DML, Kennedy (2023) — DRLearner, Athey & Wager (2019) — CausalForestDML, Künzel et al. (2019) — XLearner.

---

## `data_files` – Eingabedateien

```yaml
data_files:
  x_file: "data/X.parquet"
  t_file: "data/T.parquet"
  y_file: "data/Y.parquet"
  # Optional: historischer Score (Vergleichsbasis). Wenn gesetzt, werden die gleichen
  # Uplift-Auswertungen zusätzlich auch für diesen Score gerechnet.
  s_file: null
  # Optional: Referenz der Ziel-Datentypen (z. B. aus DataPrep als dtypes.json).
  dtypes_file: null

  # Optional: Externe Evaluationsdaten (für validate_on: "external").
  # Wenn gesetzt, wird auf diesen Daten evaluiert, während die obigen Daten
  # vollständig zum Training verwendet werden.
  eval_x_file: null
  eval_t_file: null
  eval_y_file: null
  eval_s_file: null   # optional: historischer Score im Eval-Datensatz

  # Optional: Boolean-Maske für "Train Many, Evaluate One".
  # Wird von DataPrep bei eval_file_index erzeugt.
  eval_mask_file: null  # z. B. "data/processed/eval_mask.npy"

# Optional: Einstellungen zum historischen Score (gilt nur, wenn data_files.s_file gesetzt ist)
historical_score:
  # Name, unter dem der historische Score in Ausgaben/Artefakten geführt wird.
  name: "historical_score"
  # Spaltenname im s_file (Standard: "S")
  column: "S"
  # True: große Werte sind "gut" (Top-Scores zuerst behandeln)
  # False: kleine Werte sind "gut" (intern wird der Score invertiert)
  higher_is_better: true
```

- `x_file`: Feature‑Tabelle (Spalten = Merkmale, Zeilen = Beobachtungen)
- `t_file`: Treatment‑Vektor (0/1 bei Binary Treatment)
- `y_file`: Outcome‑Vektor (0/1 bei Binary Outcome)
- `s_file` (optional): historischer Score als Vergleichsbasis (CSV oder Parquet mit Score-Spalte, Standard "S")
- `eval_x_file`, `eval_t_file`, `eval_y_file` (optional): Separater Evaluationsdatensatz für `validate_on: "external"`. Wenn gesetzt, wird auf diesen Daten evaluiert, während `x_file`/`t_file`/`y_file` vollständig zum Training verwendet werden. Kein Data-Leakage, da das Preprocessing in der DataPrep getrennt auf den Trainingsdaten gefittet und nur transformierend auf die Eval-Daten angewendet wird.
- `eval_s_file` (optional): Historischer Score im Eval-Datensatz (für Benchmark-Vergleich auf externen Daten)
- `eval_mask_file` (optional): Boolean-Maske (.npy) für „Train Many, Evaluate One". Die Pipeline spaltet **vor dem Training**: Zeilen mit `mask == True` werden als Holdout-Eval-Set zurückgehalten, Zeilen mit `mask == False` für Training genutzt. Damit läuft TMEO leakage-frei wie External Eval. Wird automatisch von DataPrep erzeugt, wenn `eval_file_index` gesetzt ist. MLflow loggt `tmeo_train_n` und `tmeo_eval_n`.

### Historischer Score (Vergleichsbasis)

Wenn `data_files.s_file` gesetzt ist, werden Uplift-Kennzahlen (Qini, AUUC, Uplift@k, Policy Value)
Modelle quantifizieren möchte.

Wichtig ist dabei die Interpretation der Score-Richtung:

- `higher_is_better: true` bedeutet: große Score-Werte sind "gut" (die Top-Scores sind die zuerst zu behandelnden Fälle).
- `higher_is_better: false` bedeutet: kleine Score-Werte sind "gut". In diesem Fall invertiert rubin den Score intern,
  damit die Sortierung in den Uplift-Metriken korrekt ist.

**Warum konfigurierbar?**  
Dateipfade unterscheiden sich zwischen lokalen Läufen, Batch‑Jobs und CI‑Umgebungen. Die Pipeline soll dafür
nicht angepasst werden müssen.

---

## `data_prep` – Datenaufbereitung (optional)

Die DataPrepPipeline ist optional. Sie wird genutzt, wenn die Rohdaten erst in die drei
Standarddateien `X.parquet`, `T.parquet`, `Y.parquet` (und optional `S.parquet`) überführt werden sollen.

Wichtig: Die Analyse-Pipeline benötigt weiterhin `data_files`. Typischer Workflow ist daher:

1. DataPrep ausführen (schreibt `X.parquet`, `T.parquet`, `Y.parquet`, `preprocessor.pkl`, … in `data_prep.output_path`)
2. `data_files.*` auf diese Ausgabedateien zeigen lassen (entweder im selben `config.yml` oder in einem
   zweiten, identischen Analyse-Config-File)
3. Analyse-Pipeline starten

Bei **externer Validierung** (`validate_on: "external"`) zusätzlich:

1. `eval_data_path` in `data_prep` setzen → DataPrep fittet Preprocessor auf Train-Daten und transformiert Eval-Daten getrennt
2. Ausgabe: zusätzlich `X_eval.parquet`, `T_eval.parquet`, `Y_eval.parquet` im gleichen Output-Verzeichnis
3. `data_files.eval_x_file` etc. auf diese Dateien setzen

Beispiel:

```yaml
data_prep:
  data_path: ["/pfad/zur/input.sas7bdat"]
  delimiter: ","
  chunksize: 300000
  sas_encoding: "ISO-8859-1"

  feature_path: "/pfad/zum/Feature_Dictionary.xlsx"  # Optional: Wenn nicht gesetzt, werden alle Spalten (außer Target/Treatment) als Features verwendet
  info_path: null

  target: "TA_HR_ABSCHLUSS_CNT"              # Einzelne Spalte oder Liste: ["SPALTE_A", "SPALTE_B"] → werden aufsummiert
  treatment: "KONTROLLGRUPPE_FLG"
  target_replacement: {0: 0, 1: 1}
  treatment_replacement: {"J": 0, "N": 1}

  score_name: "HIST_SCORE_WERT"
  score_as_feature: true

  multiple_files_option: "treatment_only"  # "merge" | "treatment_only"
  control_file_index: 0
  balance_treatments: false               # Treatment-Raten pro Datei angleichen (Downsampling)
  eval_file_index: null                   # Index der Datei für "Train Many, Evaluate One" (0-basiert)

  binary_target: true
  fill_na_method: "median"  # "median" | "mean" | "zero" | "mode" | "max" | null

  deduplicate: true                   # Kunden auf 1 Eintrag pro ID reduzieren
  deduplicate_id_column: "PARTNER_ID" # Spalte mit der Kunden-ID

  # Optional: Separater Eval-Datensatz. Der Preprocessor wird nur auf den
  # Train-Daten (data_path) gefittet und auf die Eval-Daten nur transformierend angewendet.
  eval_data_path: null                # z. B. ["/pfad/zur/eval_data.csv"]

  output_path: "data/prep_output"

  log_to_mlflow: true
  mlflow_experiment_name: "data_prep_experiment"
  mlflow_run_name: null  # null = automatisch (z. B. "Datenaufbereitung – roter-falke")
```

**Warum konfigurierbar?**
- DataPrep enthält viele "globale" Stellschrauben (Pfadlisten, Replacement-Maps, Encoding, Output-Pfade),
  die zwischen Use Cases variieren.
- In der Praxis ist es wichtig, dass diese Parameter *nicht* als Code-"Globals" gepflegt werden,

**Deduplizierung:** Wenn `deduplicate: true`, wird der Datensatz direkt nach dem Einlesen auf einen Eintrag pro `deduplicate_id_column` reduziert (erster Eintrag wird behalten). Dies geschieht *vor* der Feature-Reduktion über das Feature-Dictionary, da die ID-Spalte typischerweise kein Feature ist. Anzahl entfernter Duplikate wird geloggt.

**MLflow-Logging:** Bei `log_to_mlflow: true` wird ein eigener MLflow-Run mit zufällig generiertem Namen erzeugt (z. B. „Datenaufbereitung – roter-falke"). Experiment-Name und Run-Name werden als `.mlflow_experiment` und `.mlflow_run_name` im Output-Verzeichnis persistiert. Die Web-UI übernimmt den Experiment-Namen automatisch auf die Konfigurationsseite.

---

## `mlflow` – Experiment‑Tracking

```yaml
mlflow:
  experiment_name: "rubin"
```

- `experiment_name`: Name des MLflow‑Experiments

**Hinweis:**  
MLflow wird in der Analyse genutzt (Training/Evaluation). Die Production‑Pipeline ist bewusst unabhängig vom
Tracking und arbeitet ausschließlich mit Bundles.

---

## `constants` – Reproduzierbarkeit, Parallelisierung & Arbeitsverzeichnis

```yaml
constants:
  SEED: 42
  tuning_seed: 18    # Separater Seed für Tuning-CV-Folds
  parallel_level: 3  # 1–4
  work_dir: null      # Arbeitsverzeichnis für alle Artefakte
```

- `SEED`: globaler Seed (Cross-Prediction-Splits, Modell-Internals, DRTester, SHAP)
- `tuning_seed`: separater Seed für Tuning-CV-Folds. Muss sich von `SEED` unterscheiden, damit Hyperparameter auf *anderen* Folds bewertet werden als die spätere Evaluation. Verhindert Val-Set-Overfitting. Default: 18.
- `parallel_level`: steuert, wie aggressiv parallelisiert wird (Default: 3)
- `work_dir`: Arbeitsverzeichnis für alle erzeugten Artefakte (MLflow, Report, Cache, Uploads). Hält das Repository-Verzeichnis sauber.

**Arbeitsverzeichnis (work_dir):**

Alle erzeugten Artefakte landen im Arbeitsverzeichnis — nicht im Repository. Die Auflösung folgt dieser Priorität:

1. Umgebungsvariable `RUBIN_WORK_DIR` (höchste Priorität, einmal pro Maschine/Container)
2. Config `constants.work_dir` (pro Projekt/Run)
3. Default: `./runs` relativ zum CWD

Typische Verzeichnisstruktur im work_dir:

```
runs/
├── mlruns/              # MLflow Tracking (Metriken, Artefakte, Modelle)
├── .rubin_cache/        # HTML-Report, eval_summary, UI-generierte Configs
├── .rubin_progress.json # App-Fortschrittsanzeige
└── uploads/             # Über die App hochgeladene Dateien
```

**Empfehlung:** Für lokale Entwicklung reicht der Default `./runs` mit `.gitignore`-Eintrag. Für Server-/Domino-Deployment: `RUBIN_WORK_DIR=/mnt/workspace` einmalig setzen.

| Level | Name | Tuning-Trials | Base Learner | CV-Folds | Evaluation | RAM-Bedarf |
|-------|------|--------------|-------------|----------|------------|------------|
| 1 | Minimal | 1 sequentiell | 1 Kern | sequentiell | alle Modelle DRTester | ~1× |
| 2 | Moderat | 1 sequentiell | alle Kerne | sequentiell | alle Modelle DRTester | ~1× |
| 3 | Hoch | 2–4 parallel | Kerne/Workers | 2–4 parallel | Champion + Challenger DRTester | ~2–3× |
| 4 | Maximum | max. parallel | Kerne/Workers | alle parallel | nur Champion DRTester | ~3–5× |

Bei Level 3–4 werden die CPU-Kerne proportional aufgeteilt — sowohl für parallele Optuna-Trials im Tuning als auch für parallele CV-Folds im Training. Die parallelen Trials sind auf max. 8 (LightGBM) bzw. 6 (CatBoost) gecappt, damit der TPE-Sampler genug sequentielle Wellen zum Lernen hat. Übrige Kerne gehen an die einzelnen Fits. Level 3 ist auf max. 4 parallele Trials begrenzt. Die Feature-Selektion läuft immer sequentiell, nutzt aber ab Level 2 alle Kerne pro Methode (`n_jobs=-1`). Bei großen Datensätzen (>100k Zeilen) subsampelt CausalForest automatisch stratifiziert.

**Evaluation:** Schnelle Metriken (Qini, AUUC, Policy Value) und CATE-Verteilungs-Histogramme werden immer für alle Modelle berechnet — unabhängig vom Level. Die DRTester-Diagnostik-Plots (Calibration, Qini/TOC mit Bootstrap-CIs) werden bei Level 3 nur für Champion + besten Challenger erzeugt, bei Level 4 nur für den Champion. scikit-uplift-Plots (Qini-Kurve, Uplift-by-Percentile, Treatment-Balance) laufen ebenfalls immer für alle Modelle. Die DRTester-Nuisance-Modelle nutzen leichtere Varianten (n_estimators≤100, cv=5) für schnelleres Fitting.

**Trade-offs bei Level 3–4:**

*Tuning — Speed vs. Hyperparameter-Qualität:*  
Optuna (TPE) lernt aus vorherigen Trial-Ergebnissen, welche Hyperparameter-Regionen vielversprechend sind. Bei parallelen Trials fehlen diese Ergebnisse teilweise. Beispiel mit 50 Trials auf 64 Kernen: Level 2 hat 100 sequentielle Runden, Level 3 hat ~6 Wellen (je 16 Trials), Level 4 hat ~6 Wellen (je 16 Trials). Optuna kompensiert mit der „Constant Liar"-Strategie, aber die Exploration ist bei Level 4 weniger gezielt. Die n_startup_trials (zufällige Trials vor TPE-Exploration) skalieren mit der Anzahl paralleler Jobs: `max(parallel_jobs, min(10, n_trials//5))`. Dadurch hat TPE nach der ersten Welle genug Datenpunkte zum Lernen.

*Training — Speed vs. RAM:*  
Die CV-Fold-Ergebnisse sind mathematisch identisch, egal ob sequentiell oder parallel. Der Trade-off ist rein RAM: Level 3 hält 2–4 EconML-Modelle gleichzeitig im Speicher (~2–3× Baseline), Level 4 hält alle Folds (~3–5× Baseline). Bei großen Datensätzen (>500k Zeilen) kann Level 4 den Kernel killen.

*CatBoost vs. LightGBM bei Level 3–4:*  
CatBoost's Symmetric-Tree-Algorithmus skaliert schlechter mit wenigen Threads pro Fit als LightGBM. Deshalb werden bei CatBoost automatisch weniger parallele Workers gestartet, dafür mit mehr Threads pro Worker. Beispiel 16 Kerne, Level 4: LightGBM 8 Trials × 2 Kerne, CatBoost 4 Trials × 4 Kerne. Die gesamte CPU-Auslastung bleibt gleich, aber CatBoost nutzt die Threads effizienter.

*CausalForestDML und CausalForest:*  
CausalForestDML und CausalForest durchlaufen externe K-Fold Cross-Validation wie alle anderen Modelle. Die CV-Folds laufen immer sequentiell — der interne GRF (Generalized Random Forest) nutzt joblib-Prozesse für die Baum-Parallelisierung, die in Threads zu Deadlocks führen würden. Jeder Fold bekommt alle CPU-Kerne.

**Empfehlung:** Level 3 bietet den besten Gesamtkompromiss. Level 4 nur bei ausreichend RAM und wenn Speed kritisch ist. Level 2 für maximale Tuning-Qualität und minimalen RAM.

**Warum?**  
Ohne festen Seed werden Ergebnisse (insb. bei Optuna und Subsampling) schwer vergleichbar.
Level 3 ist der empfohlene Default (bester Kompromiss aus Exploration und Geschwindigkeit).

---

## `data_processing` – Datenumfang & Validierungsmodus

```yaml
data_processing:
  reduce_memory: true
  df_frac: null
  validate_on: "cross"   # "cross" | "external"
  cross_validation_splits: 5  # Äußere CV für Out-of-Fold CATE-Predictions
  dml_crossfit_folds: 5       # Internes Nuisance-Cross-Fitting (EconML-Default=2)

> **Hinweis:** Alle Cross-Validations verwenden `StratifiedKFold(shuffle=True)` mit Stratifizierung auf T×Y (Treatment × Outcome). Das garantiert balancierte Grundgruppen in jedem Fold — kritisch für kausale Inferenz, da DML-Residuals sonst systematisch verzerrt werden können.

  mc_iters: null          # Monte-Carlo-Iterationen für DML/DR Nuisance (null = EconML-Default)
  mc_agg: "mean"          # Aggregation über mc_iters: "mean" oder "median"
```

- `reduce_memory`: Datentypen automatisch downcasten (float64 → float32, int64 → int16/int32 etc.). Spart ca. 40–60% Arbeitsspeicher bei minimalem Präzisionsverlust. Wird sowohl in der DataPrep- als auch in der Analyse-Pipeline angewendet.
- `df_frac` (optional): Anteil der Daten für schnelle Tests (z. B. `0.1`)
- `validate_on`:
  - `"cross"`: Cross‑Predictions (robust, Standard). Kombinierbar mit einer Eval-Maske (`data_files.eval_mask_file`) für TMEO — in diesem Fall wird die Maske vor dem Training angewendet, Mask-Rows werden als Holdout zurückgehalten, und die Pipeline läuft wie External Eval (kein äußeres CV, Training nur auf ~mask-Rows).
  - `"external"`: Training auf `data_files` (x/t/y_file), Evaluation auf separatem Datensatz (`eval_x/t/y_file`). Erfordert, dass die eval-Dateien in `data_files` angegeben sind. Kein Data-Leakage — der Preprocessor wird nur auf den Trainingsdaten gefittet.
- `cross_validation_splits`: **Äußere Fold-Anzahl** für Out-of-Fold CATE-Predictions. Alle Modelle durchlaufen K-Fold CV für echte OOF-Vorhersagen. Standard: 5.
- `dml_crossfit_folds`: **Internes Nuisance-Cross-Fitting** für DML/DR-Modelle in Produktion. Steuert wie viele Folds für die Residualisierung `Y − E[Y|X]` verwendet werden. Default: 5 (EconML-Default wäre 2). Synchronisiert mit `tuning.cv_splits` und `final_model_tuning.cv_splits` — alle inneren CVs werden über die UI als ein Wert gesteuert.
- `mc_iters` (optional): Monte-Carlo-Iterationen für die Nuisance-Schätzung der DML/DR-Modelle. Bei `mc_iters: 3` wird das gesamte Nuisance-Cross-Fitting 3× wiederholt, was die Varianz der CATE-Schätzungen um ca. Faktor 3 senkt. Kosten: ~linearer Anstieg der Laufzeit. `null` = EconML-Default (1 Iteration).
- `mc_agg`: Aggregation der Monte-Carlo-Iterationen. `"mean"` (Standard) oder `"median"` (robuster bei Ausreißern in den Nuisance-Prädiktionen).

**Warum?**  
Für Entwicklung/Iteration wird häufig mit Teilmengen gearbeitet, während finale Runs auf dem vollen Datensatz
laufen sollen. Der Validierungsmodus ist zudem entscheidend für die Stabilität der Uplift‑Kennzahlen.

---


## `treatment` – Treatment-Typ (Binary vs. Multi)

```yaml
treatment:
  type: binary        # "binary" | "multi"
  reference_group: 0  # Baseline/Control-Gruppe
```

- `type`: Steuert, ob die Pipeline für binäres Treatment (T in {0,1}) oder Multi-Treatment (T in {0,1,...,K-1}) laeuft.
  Bei `"multi"` werden SLearner, TLearner und XLearner automatisch blockiert, da diese nur Binary Treatment unterstützen.
- `reference_group`: Welche Treatment-Gruppe als Control/Baseline dient (typisch 0).

**Wichtig:** Bei `type: "multi"` ändert sich die Struktur der Predictions und Evaluationsmetriken:
- Statt einer CATE-Spalte gibt es K-1 Spalten (eine pro Treatment-Arm vs. Control).
- Statt eines skalaren Qini-Koeffizienten gibt es pro-Arm-Metriken plus einen globalen Policy-Value.
- Die Champion-Auswahl sollte bei MT auf `metric: policy_value` umgestellt werden.

---


## `bundle` – Bundle-Export für Production

```yaml
bundle:
  enabled: false
  base_dir: "bundles"
  bundle_id: null
  include_challengers: true
  log_to_mlflow: true
```

- `enabled`: Export am Ende von `run_analysis.py` aktivieren/deaktivieren
- `base_dir`: Zielordner, unter dem das Bundle-Verzeichnis angelegt wird
- `bundle_id`: optionaler fixer Name des Bundle-Verzeichnisses; `null` erzeugt einen Zeitstempel-Namen
- `include_challengers`: `true` exportiert alle trainierten Modelle, `false` nur den Champion
- `log_to_mlflow`: zusätzliches Logging des erzeugten Bundle-Verzeichnisses als MLflow-Artefakt

**CLI-Overrides:**
- `--export-bundle` erzwingt Export
- `--no-export-bundle` deaktiviert Export
- `--bundle-dir` überschreibt `base_dir`
- `--bundle-id` überschreibt `bundle_id`

## `feature_selection` – optionale Feature‑Filter

```yaml
feature_selection:
  enabled: true
  methods: [lgbm_importance, causal_forest]   # Union der Top-Features
  top_pct: 15.0                                # Top-X% pro Methode
  max_features: null                           # Absolute Obergrenze nach Union
  correlation_threshold: 0.9
```

- `enabled`: Schaltet Feature‑Selektion an/aus.
- `methods`: Liste der Importance-Methoden. Mehrere können kombiniert werden – die Ergebnisse werden per Union zusammengeführt.
  - `"lgbm_importance"`: LightGBM-Regressor auf Outcome (Y), Gain-Importance. Schnell, erfasst prädiktive Relevanz.
  - `"causal_forest"`: EconML GRF CausalForest Feature-Importances. Direkte GRF-Implementierung ohne Nuisance-Fitting; erfasst kausale Relevanz (welche Features die Heterogenität des Treatment-Effekts treiben). **Kann keine fehlenden Werte verarbeiten** – wird bei NaN in den Daten automatisch übersprungen.
  - `"none"`: Keine Importance-Filterung.
- `top_pct`: Prozent der Features, die pro Methode behalten werden. Bei Union: aus jeder Methode werden die Top-X% behalten, dann vereinigt. Beispiel: 15.0 bei 100 Features → je 15 Features pro Methode, Union kann bis zu 30 enthalten.
- `max_features` (optional): Absolute Obergrenze nach der Union. Bei Überschreitung wird nach mittlerer Rank-Position über alle Methoden sortiert.
- `correlation_threshold`: Korrelationsfilter (Pearson + Spearman). Bei |corr| > Schwellwert wird das Feature mit der **niedrigeren aggregierten Importance** entfernt. Die Importance des entfernten Features wird auf den überlebenden Partner addiert (Importance-Umverteilung), um das Splitting-Problem bei korrelierten Features zu korrigieren.

**Vierstufiger Prozess:**
1. Importances auf allen Features berechnen
2. Korrelationsfilter: bei korrelierten Paaren das weniger wichtige entfernen (importance-gesteuert)
3. Importance-Umverteilung: Importance des entfernten Features auf den Partner addieren
4. Top-X%-Threshold auf den verbleibenden Features mit korrigierten Scores (Union)

**Warum Union?**
Die prädiktive Relevanz (Outcome-Importance) und die kausale Relevanz (CATE-Heterogenität) überlappen oft nur teilweise. Ein Feature kann stark prädiktiv für das Outcome sein, aber keinen heterogenen Treatment-Effekt haben (und umgekehrt). Durch die Union werden beide Perspektiven berücksichtigt.

**Kategorische Features:** Die LightGBM-Importance-Berechnung nutzt automatisch native kategoriale Splits (via `categorical_feature`). Dadurch werden kategorische Features korrekt bewertet und nicht systematisch unterbewertet.

**Parallelisierung:** Die Importance-Methoden laufen immer sequentiell, aber jede Methode bekommt alle CPU-Kerne (`n_jobs=-1`). GRF nutzt bei großen Datensätzen (>100k Zeilen) automatisch stratifiziertes Subsampling für schnelle Berechnung. Bei Level 1 wird nur ein Kern pro Fit verwendet (`n_jobs=1`).

---

## `models` – welche kausalen Learner trainiert werden

```yaml
models:
  models_to_train:
    - "SLearner"
    - "TLearner"
    - "XLearner"
    - "DRLearner"
    - "NonParamDML"        # DML-Variante (nichtlinear, frei wählbares Final-Modell)
    - "ParamDML"           # DML-Variante (linear, nutzt EconMLs LinearDML)
    - "CausalForestDML"    # DML-Residualisierung (model_y/model_t) + Causal Forest als letzte Stufe
    - "CausalForest"                # Reiner Causal Forest (direkte Effektschätzung, ohne Nuisance-Modelle)
  ensemble: true           # Gleichgewichtetes Ensemble aller trainierten Modelle (optional)
```

Nur diese Modellnamen sind gültig. Allgemeine Kürzel wie `"DML"` sind nicht erlaubt, damit Konfiguration und Registry eindeutig bleiben.

**Hinweis zu `CausalForest`:**  
Reiner Causal Forest (`econml.grf.CausalForest`) ohne DML-Residualisierung. Schätzt den Treatment-Effekt direkt mit Honest Estimation. Keine Nuisance-Modelle nötig, kein Base-Learner-Tuning. Nur Binary Treatment, keine NaN-Toleranz. Tuning über EconMLs `tune()`-Methode (`causal_forest.use_econml_tune: true`).

**Criterion (`mse` vs `het`):** Der CausalForest Optuna-Tuning variiert auch das Split-Criterion. `"mse"` (Default) bestraft Splits mit niedriger Treatment-Varianz und liefert stabilere Schätzungen. `"het"` maximiert reine Effekt-Heterogenität (aggressiver). Der R-Loss auf dem Val-Fold entscheidet datengetrieben.

Der `CausalForestAdapter` erbt von `BaseCateEstimator` und ist dadurch kompatibel mit `EnsembleCateEstimator` — CausalForest nimmt vollwertig am Ensemble teil.

**Hinweis zu `ensemble`:**  
Nutzt EconMLs `EnsembleCateEstimator` mit gleichgewichteten (`1/N`) Vorhersagen aller trainierten Modelle. Alle Modelle liefern Out-of-Fold Cross-Predictions aus externer K-Fold CV. Das Ensemble nimmt an der Champion-Selektion teil. Wird das Ensemble Champion, werden beim Bundle-Export alle Modelle auf vollen Daten refittet.

**Hinweis zu `ParamDML`:**  
`ParamDML` nutzt intern EconMLs `LinearDML`. Das bedeutet, das Final-Modell nimmt eine **lineare CATE-Struktur** an
(CATE(X) = X · β). Für nichtlineare CATE-Schätzung eignet sich `NonParamDML` besser, da dort das Final-Modell
frei wählbar ist (z. B. LightGBM-Regressor).

**Hinweis zu Binary Treatment / Binary Outcome:**  
Alle DML-Modelle (`NonParamDML`, `ParamDML`, `CausalForestDML`) sowie `DRLearner` werden in rubin mit
`discrete_treatment=True` und `discrete_outcome=True` erstellt. Das stellt sicher, dass EconML intern
die korrekte Cross-Fitting-Logik für binäre Variablen verwendet (Klassifikatoren für die Nuisance-Modelle
`model_y`, `model_t` und `model_propensity`). Die Meta-Learner (`SLearner`, `TLearner`, `XLearner`) sowie
`DRLearner.model_regression` verwenden hingegen **Regressoren** als Outcome-Modelle, da EconML intern
`model.predict()` aufruft – ein Classifier gibt dort nur 0/1 (Klassen-Labels) zurück, ein Regressor
liefert E[Y|X] ∈ [0,1] (kontinuierliche Wahrscheinlichkeit), was für die CATE-Berechnung benötigt wird.

**Hinweis zu fehlenden Werten:**  
Alle Modelle außer `CausalForestDML` können mit fehlenden Werten in den Features umgehen, da sie
LightGBM oder CatBoost als Base Learner nutzen. `CausalForestDML` basiert intern auf einem
GRF (Generalized Random Forest), der keine NaN-Werte unterstützt. Enthält der Datensatz fehlende
Werte, wird `CausalForestDML` automatisch übersprungen und ein entsprechender Hinweis geloggt.


---

## `base_learner` – Basismodell (LightGBM oder CatBoost)

```yaml
base_learner:
  type: "lgbm"          # "lgbm" oder "catboost"
  fixed_params: {}
```

- `type`: Auswahl des Base Learners
- `fixed_params`: Parameter, die direkt für alle Nuisance-Modelle (Outcome, Propensity) gesetzt werden. Relevant wenn `tuning.enabled: false` – dann werden diese statt der getunedn Parameter verwendet. Wenn Tuning aktiv ist, werden `fixed_params` ignoriert und die Optuna-Ergebnisse genutzt.

**Praxisempfehlungen:**
- LightGBM: schnell, sehr gut für viele numerische Features
- CatBoost: robust, oft stark bei kategorischen Features

**Wichtig:** Beim Wechsel des Base Learners ändern sich die verfügbaren Parameter-Namen (z.B. `n_estimators` bei LightGBM vs. `iterations` bei CatBoost). Die `fixed_params` und `final_model_tuning.fixed_params` sollten dann ebenfalls angepasst werden.

**Kategorische Features:** EconML konvertiert X intern zu numpy-Arrays, wodurch pandas `category`-Dtypes verloren gehen. rubin löst dieses Problem automatisch: Vor dem Training werden die `fit()`-Methoden von LightGBM/CatBoost so gepatcht, dass `categorical_feature` (LightGBM) bzw. `cat_features` (CatBoost) bei jedem internen Aufruf mitgegeben wird. So nutzen die Base Learner native kategoriale Splits, selbst wenn EconML die Daten als numpy übergibt.

**Externe Cross-Validation und internes Cross-Fitting:**

Alle Modelle — Meta-Learner (SLearner, TLearner, XLearner), DML-Familie (NonParamDML, ParamDML, CausalForestDML), DRLearner und CausalForest — durchlaufen externe K-Fold Cross-Validation. So erhält jede Beobachtung eine CATE-Prediction aus einem Modell, das sie nicht gesehen hat (echte Out-of-Fold-Garantie).

DML/DR-Modelle nutzen *zusätzlich* internes Cross-Fitting für die Nuisance-Residualisierung (model_y, model_t, model_propensity). Dieses interne Cross-Fitting erzeugt OOF-Residuals für die Nuisance-Komponenten, aber model_final (das CATE-Effektmodell) trainiert innerhalb jedes äußeren Folds auf allen Trainings-Residuals. Ohne externe CV wären die CATE-Predictions daher nicht out-of-fold.

```yaml
data_processing:
  cross_validation_splits: 5   # Äußere CV für Out-of-Fold CATE-Predictions
  dml_crossfit_folds: 5        # Internes Nuisance-Cross-Fitting (EconML-Default)
  mc_iters: null               # Monte-Carlo-Iterationen: null=1 Durchlauf, 2-3 empfohlen
  mc_agg: "mean"               # Aggregation: "mean" (Standard) oder "median" (robuster)
```

- `cross_validation_splits`: **Äußere Fold-Anzahl** für Out-of-Fold CATE-Predictions aller Modelle. Standard: 5.
- `dml_crossfit_folds`: **Internes Nuisance-Cross-Fitting** für DML/DR-Modelle in Produktion. Getrennt von der äußeren CV und von den Tuning-CVs. EconML-Default: 2.
- `mc_iters`: Wiederholt das interne Cross-Fitting N-mal mit unterschiedlichen Splits und mittelt die Residuals. Reduziert Varianz der Nuisance-Schätzung linear (mc_iters=3 → ~3× niedrigere Varianz). Kostet linear mehr Rechenzeit (nur Nuisance-Fits, model_final wird nur einmal gefittet).
- `mc_agg`: `"mean"` (Standard) oder `"median"` (robuster bei Ausreißern in den Residuals).

---

## `causal_forest` – Parameter für `CausalForestDML`

```yaml
causal_forest:
  forest_fixed_params: {}
  use_econml_tune: false
  n_trials: 50
  tune_models: []
```

`CausalForestDML` kombiniert **DML‑Residualisierung** mit einem **Causal Forest** als letzter Stufe.
Damit gibt es zwei Ebenen, die man konfigurieren kann:

1) **Nuisance‑Modelle** (`model_y`, `model_t`) – das sind Base Learner wie bei anderen DML‑Verfahren.
   Diese werden über `base_learner` (und ggf. `tuning`) gesteuert.

2) **Wald‑Parameter** der finalen Forest‑Stufe – diese werden über `causal_forest` gesteuert.

Felder:

- `forest_fixed_params`: Feste Parameter, die immer an die Forest‑Stufe übergeben werden
  (z. B. `honest`, `n_jobs`, `min_samples_leaf`).
- `use_econml_tune`: Wenn `true`, wird vor dem finalen Training ein Optuna-Tuning über Wald-Parameter
  durchgeführt. CausalForestDML nutzt EconMLs `tune()` (RScorer-Evaluation intern),
  CausalForest nutzt einen eigenen Optuna-Tuning mit RScorer-Evaluation.
  Das Grid ist identisch mit EconML `tune(params='auto')`: 12 Kombinationen aus
  `min_weight_fraction_leaf × max_depth × min_var_fraction_leaf`.
- `n_trials`: Anzahl Optuna-Trials für CausalForest-Tuning (Default: 50)
  (+criterion, +max_depth-Stufen, +min_var_fraction_leaf=None). Nur wirksam bei
  `use_econml_tune: true`.
- `tune_models`: Liste der Modelle, für die CausalForest-Tuning ausgeführt wird.
  Mögliche Werte: `CausalForestDML`, `CausalForest`. Leere Liste = alle
  Forest-Modelle in der Modellauswahl. Beispiel: `[CausalForestDML]` optimiert
  nur CausalForestDML, CausalForest behält EconML-Defaults.

Wichtig:

- **Optuna** optimiert in rubin weiterhin die **Base Learner** (also `model_y`/`model_t`) – auch beim
  `CausalForestDML`.
- Die **Wald‑Parameter** werden (falls gewünscht) über **EconML `tune(...)`** bestimmt, nicht über Optuna.

---

## `tuning` – Optuna‑Tuning der Base Learner

```yaml
tuning:
  enabled: true
  n_trials: 50
  timeout_seconds: null
  cv_splits: 5
  single_fold: false
  metric: "log_loss"
  metric_regression: "neg_mse"
  per_learner: false
  per_role: false
  models: null
  max_tuning_rows: null
  optuna_seed: 42
  storage_path: null
  study_name_prefix: "baselearner"
  reuse_study_if_exists: true
```

**Kernidee:**  
Nicht die kausalen Learner selbst werden getunt, sondern die Base Learner, die intern verwendet werden
(Outcome‑Modelle, Propensity‑Modelle usw.). Optional werden getrennte Parameter‑Sets optimiert:

- Standardfall: identische Tuning-Aufgaben werden task-basiert zusammengefasst
- `per_learner=true`: separates Set je kausalem Verfahren
- `per_role=true`: separates Set je Rolle innerhalb eines Verfahrens (z. B. `model_y` vs. `model_t`)

**Modellauswahl (`models`):**  
Standardmäßig (`null`) werden die Nuisance-Modelle aller in `models.models_to_train` konfigurierten Modelle getuned. Mit einer expliziten Liste kann das Tuning auf bestimmte Modelle eingeschränkt werden. Tasks, die exklusiv für nicht-ausgewählte Modelle wären, werden übersprungen — geteilte Tasks bleiben, solange mindestens ein ausgewähltes Modell sie braucht. Nicht-ausgewählte Modelle verwenden `base_learner.fixed_params`. Das ist nützlich, wenn z.B. nur die DML-Nuisance-Modelle optimiert werden sollen und SLearner/TLearner mit festen Defaults laufen können.

**Metriken (`metric`, `metric_regression`):**  
`metric` steuert die Bewertung für Classifier-Tasks (Propensity, Outcome-Klassifikation): Standard `log_loss`, alternativ `roc_auc` oder `accuracy`. `metric_regression` steuert die Bewertung für Regressor-Tasks (Meta-Learner Outcome-Modelle wie SLearner, TLearner): Standard `neg_mse`, alternativ `neg_rmse`, `neg_mae` oder `r2`.

**CV-Folds (`cv_splits`):**  
Innere Cross-Validation pro Tuning-Trial. Default: 5. Synchronisiert mit `dml_crossfit_folds` und `final_model_tuning.cv_splits` — alle inneren CVs nutzen denselben Wert. Höhere Werte stabilisieren die Trial-Bewertung, kosten aber linear mehr Rechenzeit pro Trial.

**Single-Fold-Tuning (`single_fold`):**  
Bei `single_fold: true` wird jeder Optuna-Trial nur auf **einem** zufällig gewählten Fold evaluiert statt auf allen K Folds. Das reduziert die Modell-Fits pro Trial von K auf 1 – bei 5 Folds also 5× schneller. Optuna (TPE) ist robust gegenüber verrauschteren Metriken, daher ist der Tradeoff für explorative Analysen oder große Datensätze sinnvoll. **Voraussetzung:** Der einzelne Val-Fold muss genug Minority-Fälle enthalten — empfohlen sind ≥ 100 (Minimum: ≥ 50) Fälle der kleineren Klasse: `min(n_treated, n_positive) / K`. Unterhalb von 50 ist die Metrik-Schätzung von einzelnen Samples dominiert und die HP-Auswahl quasi-zufällig (Collins et al. 2016: min. 100 Events für stabile Validation).

**Persistenz (`storage_path`)**  
Wenn `storage_path` gesetzt ist, wird die Optuna‑Study in SQLite persistiert (Fortsetzen/Analyse möglich).

**Max. Tuning-Rows (`max_tuning_rows`):**  
Begrenzt die Datenmenge für das Nuisance-Tuning (z.B. `200000`). Bei `null` werden alle Daten verwendet. Nützlich bei sehr großen Datensätzen, um die Tuning-Zeit zu begrenzen.



Die Task-Signatur berücksichtigt nicht nur den Learner-Typ, sondern auch die tatsächliche interne Lernaufgabe, u. a.:

- Base-Learner-Familie
- Objective-Familie
- Estimator-Task (Klassifikation/Regression)
- Datengrundlage bzw. Sample-Scope
- Nutzung des Treatment-Features
- Zieltyp

Dadurch werden nur wirklich gleiche Base-Learner-Aufgaben zusammengelegt.

**Hinweis zu `CausalForestDML`:**
`CausalForestDML` nutzt *zwei* Nuisance‑Modelle (`model_y`, `model_t`), die normale Base Learner sind.
Zusätzlich besitzt es eine Forest‑Stufe mit eigenen Wald‑Parametern.

- Das Optuna‑Tuning (`tuning`) betrifft weiterhin `model_y`/`model_t`.
- Die Wald‑Parameter können optional über die EconML‑Methode `tune(...)` bestimmt werden
  (siehe Abschnitt `causal_forest`).

---

## `final_model_tuning` – Optuna‑Tuning des Final‑Modells

```yaml
final_model_tuning:
  enabled: false
  n_trials: 100
  models: null
  single_fold: false
  overfit_penalty: 0.0
  overfit_tolerance: 0.05
  max_tuning_rows: null
  fixed_params: {}
```

Wofür ist das?

- Relevant für Modelle, die ein frei wählbares Final‑Modell besitzen (z. B. **NonParamDML**, **DRLearner**).
- Beide Modelle (NonParamDML, DRLearner) nutzen äußere OOF-CV mit est.score().
- DRLearner: nutzt immer den eingebauten DR‑basierten `score()+CV`.

Wichtige Regeln in der Pipeline:

- Das Tuning findet **nur einmal pro Run** statt.
- Um eine saubere Trennung zur Cross‑Prediction zu gewährleisten, wird auf der **Trainingsmenge des ersten
  Cross‑Prediction‑Folds** getunt.
- Die gefundenen Hyperparameter werden anschließend für alle weiteren Folds **wiederverwendet** ("Locking").

Parameter:

- `enabled`: Schaltet das Final‑Model‑Tuning an/aus.
- `n_trials`: Anzahl Optuna‑Trials.
- `models`: Liste der Modelle, die per FMT optimiert werden sollen (z.B. `[NonParamDML]`). Bei `null` werden alle FMT-fähigen Modelle getuned, bei einer expliziten Liste nur die genannten. Nicht getunte Modelle verwenden die `fixed_params`.
- `single_fold`: Bei `true` wird nur 1 äußerer OOF-Fold pro Trial evaluiert statt K. Gilt für beide Modelle (NonParamDML + DRLearner). Reduziert die Fits von K×Trials auf 1×Trials. Gleiche Mindestanforderung wie BLT: min(n_treated, n_positive) / K ≥ 100 empfohlen.
- `overfit_penalty`: Train-Val-Gap-Penalty für model_final (0=deaktiviert). Bestraft Konfigurationen, deren Score auf Train deutlich besser ist als auf Val (OOF). Tolerance ist relativ: BLT Default 0.10 (10% Gap — DML kompensiert Nuisance-Overfitting), FMT Default 0.05 (5% Gap — CATE-Signal schwächer). Formel: `adjusted = val_score - penalty × max(0, gap - tolerance)`.
- `max_tuning_rows`: Begrenzt die Datenmenge für das FMT (z.B. `200000`). Bei `null` werden alle Daten des Tuning-Splits verwendet.
- `fixed_params`: Feste Hyperparameter für das Final-Modell (`model_final`). Werden verwendet, wenn FMT deaktiviert ist oder wenn ein Modell nicht in `models` steht.

**Wichtig:** Das Final-Modell (`model_final`) erhält niemals die getunten Nuisance-Parameter (z. B. aus dem Base-Learner-Tuning für `model_y`/`model_t`). Ohne FMT verwendet `model_final` ausschließlich LightGBM/CatBoost-Standardwerte (bzw. `base_learner.fixed_params`). Damit wird verhindert, dass Classifier-optimierte Parameter (hoher `min_split_gain`, viele `min_child_samples`) den CATE-Baum zu einem Intercept kollabieren lassen.


## `selection` – Champion‑Auswahl (Model Registry)

```yaml
selection:
  metric: "qini"
  higher_is_better: true
  refit_champion_on_full_data: true
  manual_champion: null
```

Beim Bundle‑Export wird eine Registry geschrieben, die alle Modelle inkl. Kennzahlen enthält und einen
**Champion** festlegt. In der Produktion wird standardmäßig der Champion verwendet.

- `metric`: Kennzahl für die automatische Champion-Auswahl.
  Bei Binary Treatment: `qini`, `auuc`, `uplift_at_10pct`, `uplift_at_20pct` oder `policy_value`.
  Bei Multi-Treatment: `policy_value` (empfohlen), `policy_value_T1`, oder arm-spezifisch `qini_T1`, `qini_T2`, etc.
- `higher_is_better`: Richtung der Kennzahl
- `refit_champion_on_full_data`: refittet das ausgewählte Champion-Modell vor dem Export auf allen im Run verfügbaren Daten
- `manual_champion`: optionaler Override; falls gesetzt, wird dieses Modell unabhängig von der Kennzahl Champion

**Warum?**  
Das erleichtert den Übergang von „Analyse mit vielen Kandidaten“ zu „stabilem Produktionsmodell“.

---

## `shap_values` – Explainability‑Voreinstellungen

```yaml
shap_values:
  calculate_shap_values: true
  shap_calculation_models: [NonParamDML]
  n_shap_values: 5000
  top_n_features: 20
  num_bins: 10
```

Diese Einstellungen steuern die SHAP-Analyse in der Analyse-Pipeline. Bei `calculate_shap_values: true` wird nach dem Bundle-Export automatisch ein SHAP-Schritt ausgeführt (zweistufiger Fallback: EconML SHAP-Plots → generische SHAP). Alle Artefakte werden als MLflow-Artefakte im selben Run geloggt.

- `calculate_shap_values`: Schaltet den Explainability-Schritt in der Analyse-Pipeline an/aus.
- `shap_calculation_models`: Liste der Modelle, für die Importance berechnet wird. Leer = nur Champion.
- `n_shap_values`: maximale Stichprobe für SHAP (Performance)
- `top_n_features`: wie viele Features im Report/Plot ausgegeben werden
- `num_bins`: Standard‑Segmentierung (z. B. 10 = Dezile) für Dependency-Plots

**Hinweis:**  
Explainability ist bewusst als separater Schritt umgesetzt (kein Pflichtbestandteil eines Trainingslaufs).

---

## `optional_output` – lokale Ausgabe (zusätzlich zu MLflow)

```yaml
optional_output:
  output_dir: null
  save_predictions: false
  predictions_format: "parquet"
  max_prediction_rows: null
```

- `output_dir`: wenn gesetzt, werden ausgewählte Artefakte lokal geschrieben
- `save_predictions`: speichert Cross‑Predictions pro Modell (kann groß werden)
- `predictions_format`: `"parquet"` oder `"csv"`
- `max_prediction_rows`: optionales Limit, um I/O zu begrenzen

---

## `surrogate_tree` – Surrogate-Einzelbaum (Teacher-Learner)

```yaml
surrogate_tree:
  enabled: false
  min_samples_leaf: 50
  num_leaves: 31
  max_depth: null
```

Aktiviert einen Einzelbaum des konfigurierten Base-Learners (LightGBM/CatBoost mit `n_estimators=1`), der die CATE-Vorhersagen des Champions nachlernt (Teacher-Learner-Prinzip).

| Surrogate | Teacher | Wann trainiert |
|---|---|---|
| `SurrogateTree` | Champion (beliebiges Modell) | Wenn `enabled: true` |

- `enabled`: Aktiviert den Surrogate-Einzelbaum auf dem Champion.
- `min_samples_leaf`: Mindestanzahl an Beobachtungen pro Blatt. Wird auf `min_child_samples` (LightGBM) bzw. `min_data_in_leaf` (CatBoost) gemappt.
- `num_leaves`: Maximale Anzahl Blätter (nur LightGBM). Steuert die Baumkomplexität direkt über leaf-wise Growth.
- `max_depth`: Maximale Baumtiefe. `null` bedeutet keine Begrenzung bei LightGBM (`-1`), bei CatBoost wird `6` als Default verwendet.

**Ablauf in der Analyse-Pipeline:** Der Surrogate wird nach der Evaluation des Champions trainiert, wenn `enabled: true`. Er trainiert auf den **Train-Predictions** des Champions (Full-Data-Refit), um die gelernte CATE-Funktion bestmöglich nachzulernen. Cross-Validation (K-Fold, identisch zur äußeren CV) erzeugt OOS-Predictions, sodass die Uplift-Metriken des Surrogates fair mit dem Champion vergleichbar sind. Abschließend wird der Surrogate auf allen Daten nachtrainiert. Im Bundle-Export wird der Surrogate mit einem eigenen Registry-Eintrag exportiert.

**Production-Scoring:** Im Bundle ist der Surrogate als `SurrogateTree` verfügbar. In der Production-Pipeline kann er über `score_surrogate(X)` oder `score(X, model_names=["SurrogateTree"])` angesprochen werden. Über die CLI: `pixi run score -- --bundle ... --x ... --use-surrogate` (oder: `python run_production.py --bundle ... --x ... --use-surrogate`).

---

# Wo wird die Konfiguration „global“ wirksam?

- **Analyse‑Pipeline** (`run_analysis.py`): nutzt *alle* oben beschriebenen Bereiche.
- **Bundle‑Export**: legt `config_snapshot.yml` ab (später für Production/Explainability nutzbar).
- **Production‑Pipeline** (`run_production.py`): liest primär Artefakte aus dem Bundle (Preprocessor/Modelle/Registry).
- **Explainability** (`run_explain.py`): nutzt CLI‑Parameter, übernimmt aber Voreinstellungen aus `config_snapshot.yml`, falls vorhanden.

Damit ist die Pipeline global über `config.yml` steuerbar, während Run‑spezifische Aspekte über CLI‑Parameter
gesetzt werden können.


## `shap_values`

```yaml
shap_values:
  calculate_shap_values: true
  shap_calculation_models: []
  n_shap_values: 10000
  top_n_features: 20
  num_bins: 10
```

- `calculate_shap_values`: Aktiviert die SHAP-Analyse.
- `shap_calculation_models`: Modelle für Importance. Leer = nur Champion, explizit z.B. `[NonParamDML, DRLearner]`.
- `num_bins` steuert die Binning-Tiefe für CATE-Profil- und SHAP-Dependence-Plots.


### Overfit-Penalty (Nuisance-Regularisierung)

Die Overfit-Penalty bestraft BLT-Hyperparameter-Konfigurationen, deren Nuisance-Modelle auf den Trainingsdaten deutlich besser performen als auf der Validierung (Train-Val-Gap). Dies verhindert, dass overfittende Nuisance-Modelle kausales Signal in den Residuals absorbieren — ein bekanntes Problem bei Double/Debiased ML (Chernozhukov et al., 2016).

Konfiguration:
- `tuning.overfit_penalty`: Stärke der Penalty (0 = deaktiviert, empfohlen: 0.2–0.5). Default: 0.
- `tuning.overfit_tolerance`: Relativer Gap-Schwellwert, unter dem keine Penalty greift. Default: 0.15 (15%).

Formel (skalen-sicher): `scale = |val_score|`, `relative_gap = (train_score - val_score) / scale`, `adjusted = val_score - penalty × scale × max(0, relative_gap - tolerance)`. Penalty wirkt nur auf den finalen Trial-Score (K-Fold-Mittelwerte), nicht auf Einzelfold-Scores.

**Wann aktivieren:** Bei Verdacht auf Nuisance-Overfitting (z.B. sehr gute BLT-Scores aber schlechte Qini/AUUC-Werte) oder bei kleinen Datensätzen mit vielen Features.
