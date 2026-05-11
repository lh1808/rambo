# Optuna-Tuning der Base Learner

## Ziel
Optimiert werden ausschließlich die **Base Learner** (LightGBM/CatBoost), die innerhalb der kausalen Verfahren verwendet werden.
Das hat zwei Vorteile:

1. Du bekommst konsistente Base Modelle über verschiedene kausale Verfahren hinweg.
2. Das Tuning bleibt vergleichbar und modular, weil die kausalen Wrapper selbst nicht "aufgerissen" werden müssen.

## Wie wird getunt?
- Optuna schlägt Hyperparameter vor.
- Es wird eine CV (Cross Validation) mit `cv_splits` durchgeführt (Default: 5). Dieser Wert steuert die innere Tuning-CV und ist getrennt von `cross_validation_splits` (äußere Evaluation, Default: 5). Alle inneren CVs (`tuning.cv_splits`, `final_model_tuning.cv_splits`, `dml_crossfit_folds`) sind auf denselben Wert synchronisiert und werden in der UI über ein einzelnes Feld gesteuert.
- Als Metrik wird für **Classifier-Tasks** (Nuisance: `model_y`, `model_t`, Propensity, sowie `model_regression` bei DRLearner mit `discrete_outcome=True`) fest `log_loss` (negierter Log-Loss) verwendet. Log-Loss misst die Kalibrierung der Wahrscheinlichkeitsvorhersagen — was für die DML-Residualisierung `Y − E[Y|X]` essentiell ist. Für **Regressor-Tasks** (Meta-Learner Outcome: `overall_model`, `models`, `cate_models`, `model_final`) wird automatisch **neg. MSE** als Metrik genutzt.
- Zusätzlich wird für jeden BLT-Task ein **Skill Score** berechnet und geloggt, der die Interpretierbarkeit verbessert:
  - **Klassifikation:** `Skill = 1 − log_loss / baseline_log_loss`, wobei die Baseline ein naives Modell ist, das immer die Klassenverteilung vorhersagt (0.0 = nicht besser als Zufall, 1.0 = perfekt).
  - **Regression:** `R² = 1 − MSE / Var(Y)` (0.0 = nicht besser als Mittelwert-Modell, 1.0 = perfekt).
  - **FMT/CFT:** Hier wird der rohe R-Loss (Residual-MSE) geloggt. Dieser ist nur für den relativen Vergleich zwischen Trials sinnvoll, nicht für absolute Interpretation. Ein normierter Skill Score (wie bei BLT) würde den `RScorer` erfordern, der zu rechenintensiv für den Tuning-Loop ist.
- Bei **Multi-Treatment** wird für die Propensity-Modelle automatisch Multiclass-Log-Loss verwendet, da das Propensity-Modell dann eine Multiclass-Klassifikation durchführt (K Treatment-Gruppen). LightGBM und CatBoost erkennen Multiclass automatisch aus den Trainingsdaten.

### Warum Log-Loss als Default?

Im DML-Framework werden Nuisance-Modelle verwendet, um Residualen `Y − E[Y|X]` und `T − E[T|X]` zu bilden. Die CATE-Schätzung arbeitet auf diesen Residualen. Wenn die Nuisance-Vorhersagen gut kalibriert sind (also tatsächlich die bedingten Erwartungen widerspiegeln), enthalten die Residualen das maximale informative Signal für die Final-Stage. Log-Loss optimiert direkt diese Kalibrierung. MSE auf binären Targets Y∈{0,1} entspricht dem Brier Score und misst ebenfalls die Kalibrierung.

### GBDT als Default-Boosting-Typ (LightGBM)

Für LightGBM wird standardmäßig `boosting_type="gbdt"` (Gradient Boosted Decision Trees) verwendet. GBDT ist der stabile Standard-Modus mit vorhersagbarer Laufzeit und guter Performance. Die Regularisierung gegen Overfitting auf rauschlastigen CATE-Targets (orthogonalisierte Residuen, Pseudo-Outcomes) erfolgt über den Optuna-Suchraum: `reg_alpha`, `reg_lambda`, `min_child_samples`, `min_split_gain`, `path_smooth` und `max_bin`.

### Beide-Modus (`base_learner.type: "both"`)

Bei `base_learner.type: "both"` wird der Learner-Typ zur zusätzlichen Hyperparameter-Dimension. Optuna zieht pro Trial zuerst kategorisch zwischen `"lgbm"` und `"catboost"`, und zieht dann den jeweiligen Hyperparameter-Suchraum. TPE (Tree-structured Parzen Estimator) lernt aus den Trial-Ergebnissen, welcher Learner für den jeweiligen Task besser abschneidet, und konzentriert die weitere Suche entsprechend.

**Konsequenzen:**

- **Pro Rolle unterschiedliche Learner möglich**: In einem Lauf kann z.B. CatBoost für `model_y` gewählt werden, LightGBM für `model_t`, CatBoost für `model_final`.
- **Automatische Trial-Verdopplung**: Da der Search-Space doppelt so groß ist, werden BL-Tuning-Trials und FMT-Trials intern verdoppelt, damit beide Learner-Familien ausreichend Budget bekommen.
- **Nested fixed_params**: `base_learner.fixed_params` kann als verschachteltes Dict angegeben werden: `{lgbm: {...}, catboost: {...}}`. Beim Modellbau wird nur der Sub-Dict des gewählten Learners angewendet.
- **Categorical Patch**: Patcht beide Libraries gleichzeitig, damit kategoriale Features in beiden Fällen korrekt behandelt werden.

**Wann abschalten?** Wenn du einen spezifischen Learner erzwingen willst (z.B. nur CatBoost wegen gut kalibrierter Probabilities) oder wenn die doppelte Laufzeit nicht tolerabel ist. In dem Fall: `base_learner.type: "lgbm"` oder `"catboost"` setzen.

### Single-Fold-Tuning
Mit `single_fold: true` wird jeder Trial nur auf **einem** zufälligen Fold evaluiert statt auf allen K. Das ist besonders nützlich bei:

- Großen Datensätzen (lange CV-Läufe)
- Explorativen Analysen (schnelle Iteration wichtiger als maximale Stabilität)
- Hohen Trial-Zahlen (Optuna/TPE gleicht verrauschtere Metriken durch mehr Samples aus)

Der Speedup ist linear: Bei 5 Folds ist Single-Fold 5× schneller.

**Mindestanforderung an die Datenbasis:** Da bei Single-Fold nur 1/K der Daten für die Metrik-Berechnung verwendet wird, muss der Validierungs-Fold genug Minority-Fälle enthalten, damit die Metrik (log_loss, R-Score) statistisch stabil ist. Die Prüfformel:

```
min(n_treated, n_positive_outcome) / K ≥ 100   (empfohlen)
min(n_treated, n_positive_outcome) / K ≥ 50    (Minimum)
```

Hintergrund: Collins et al. (2016) empfehlen mindestens 100 Events für zuverlässige Validation, idealerweise 200+. Van Smeden et al. zeigten, dass für Kalibrations-Schätzungen (SE < 0.15) je nach Modellstärke 40–280 Events benötigt werden. Da log_loss kalibrationssensitiv ist und bei Single-Fold keine Varianz-Reduktion über mehrere Folds stattfindet, liegt der empfohlene Schwellwert bei 100 Minority-Fällen pro Val-Fold. Unter 50 ist die Metrik von einzelnen Samples dominiert — die HP-Auswahl wird quasi-zufällig.

Die UI zeigt eine automatische Warnung, wenn Single-Fold aktiviert und die Datenbasis (aus der Datenvorbereitung) bekannt ist.

### Parallele Trials
Bei `constants.parallel_level` 3 oder 4 werden mehrere Optuna-Trials gleichzeitig ausgeführt (`study.optimize(n_jobs=...)`). Die CPU-Kerne werden proportional aufgeteilt:

- **Level 1–2:** 1 Trial sequentiell, alle Kerne an den einzelnen Fit
- **Level 3–4:** `cpus // 4` parallele Trials (z. B. 16 bei 64 Kernen), je 4 Kerne

Dies gilt für das **Base-Learner-Tuning**. FMT und CFT laufen seit der cache_values-Optimierung
**sequentiell** (n_jobs=1) — Nuisance-Modelle werden einmalig gecacht, Trials fitten nur model_final
via `refit_final()` mit allen Kernen (parallel_jobs=-1).

### Trade-off: Speed vs. TPE-Qualität

TPE (Tree-structured Parzen Estimator) lernt aus den Ergebnissen vorheriger Trials, welche Hyperparameter-Regionen vielversprechend sind. Bei parallelen Trials fehlen diese Ergebnisse teilweise, weil mehrere Trials gleichzeitig laufen, ohne voneinander zu wissen. Optuna kompensiert mit der „Constant Liar"-Strategie: Laufende Trials bekommen Dummy-Werte, damit TPE trotzdem informiert vorschlagen kann.

Beispiel mit 50 Trials auf 16 Kernen:

| Level | Parallele Trials | Runden | Random-Starts | Informierte Trials | Speedup |
|-------|-----------------|--------|---------------|-------------------|---------|
| 2 | 1 | 30 | ~10 | ~20 | 1× |
| 3 | 4 | ~8 | ~10 | ~20 | ~3–4× |
| 4 | 8 | ~4 | ~10 | ~20 | ~5–7× |

Bei 30+ Trials ist der Qualitätsverlust gering — TPE mit 4 Runden à 8 Trials findet typischerweise 90–95% der Performance des sequentiellen Laufs. Bei wenigen Trials (<15) ist Level 4 nicht empfehlenswert, da fast alle Trials ohne informierte Führung laufen.

**Empfehlung:** Level 3 bietet den besten Kompromiss aus Speed und Tuning-Qualität. Level 4 nur bei 30+ Trials und wenn maximaler Speed wichtiger ist als letzte Prozent Tuning-Performance.

### Trade-off: Speed vs. RAM

Jeder parallele BLT-Trial hält eigene Modellinstanzen und Daten-Slices im Speicher. FMT und CFT laufen sequentiell (n_jobs=1), halten aber die gecachten Nuisance-Estimatoren für die gesamte Study-Dauer im RAM (K äußere Folds × je cv innere Nuisance-Modelle). Der RAM-Verbrauch steigt proportional zur Anzahl äußerer Folds und Nuisance-Modellgröße.

| Level | RAM (Tuning) | RAM (Training) | Kernel-Risiko |
|-------|-------------|----------------|---------------|
| 2 | ~1× | ~1× | Minimal |
| 3 | ~3× | ~2–3× | Gering |
| 4 | ~6× | ~3–5× | Mittel bei >500k Zeilen |

## Learner-spezifische Trainingsmengen
Viele kausale Learner trainieren nicht auf der vollen Datenmenge:

- **S-Learner**: ein Modell auf allen Daten (Features + Treatment als Feature)
- **T-Learner**: zwei Modelle – eins auf Control, eins auf Treatment
- **X-Learner**: mehrere Modelle (Outcome- und CATE-Modelle) teils gruppenweise

Im Tuning wird daher vor dem CV-Lauf eine **Sampling-Strategie** angewendet:
- Für T-/X-Learner wird balanciert nach `T` gezogen (Control/Treatment), damit die Datenmenge pro Modell realistisch ist.

## Separate Hyperparameter-Sets
Je nach Konfiguration werden getrennte Parameter gesucht:


Das Ergebnis wird in einem JSON-Artefakt gespeichert und beim Modellbau angewendet.

## Modellauswahl für BL-Tuning (`models`)

Standardmäßig (`models: null`) werden die Nuisance-Tasks aller konfigurierten Modelle getuned. Mit einer expliziten Liste kann das Tuning auf bestimmte Modelle eingeschränkt werden:

```yml
tuning:
  enabled: true
  models: [NonParamDML, DRLearner]  # null = alle
```

Die Task-Sharing-Logik berücksichtigt nur die ausgewählten Modelle. Tasks, die **exklusiv** für nicht-ausgewählte Modelle benötigt würden, werden übersprungen. Geteilte Tasks bleiben, solange mindestens ein ausgewähltes Modell sie braucht. Beispiel:

- `models: [NonParamDML, DRLearner]` bei `models_to_train: [NonParamDML, DRLearner, SLearner, TLearner]`
- `model_y` (Outcome-Classifier) wird getuned — NonParamDML braucht es
- `model_t` / `model_propensity` wird getuned — beide DML-Modelle brauchen es
- `overall_model` (SLearner) wird übersprungen — SLearner ist nicht in der BLT-Auswahl
- `grouped_outcome_regression` (TLearner) wird übersprungen

SLearner und TLearner werden trotzdem normal trainiert — sie nutzen dann `base_learner.fixed_params` statt getunter Hyperparameter.

---

## Propensity-Tuning bei RCT

Bei `study_type: "rct"` wird das Propensity-Tuning automatisch auf **20 Trials** reduziert (statt der vollen Trial-Anzahl). Die Begründung: Bei einem RCT ist P(T=1|X) = const — das Propensity-Modell sollte das Treatment nicht aus den Features vorhersagen können.

Das reduzierte Tuning dient als **Diagnose-Check**: Wenn das Propensity-Modell trotzdem eine AUC deutlich über 0.5 erreicht, erscheint eine Warnung im Log:

```
RCT-Warnung: Propensity-Modell erreicht -0.42 (erwartet ~-0.69 bei zufälligem Treatment).
Prüfe ob Treatment tatsächlich randomisiert ist oder ob Post-Treatment-Variablen im Datensatz sind.
```

Mögliche Ursachen:
- Post-Treatment-Variablen im Feature-Set (z.B. Variablen, die erst nach der Treatment-Zuweisung gemessen wurden)
- Unvollständige Randomisierung (Stratifizierung nach Kovariaten, Block-Randomisierung mit kleinen Blöcken)
- Leakage: Eine Feature-Spalte enthält implizit die Treatment-Information

---

## Final-Model-Tuning (OOF-CV)

Einige Verfahren besitzen zusätzlich ein **Final-Modell**, das die heterogenen Effekte lernt
(z. B. `NonParamDML`, `DRLearner`). Dieses Final-Modell ist typischerweise ein Regressor
(LightGBM/CatBoost). Die DML-Nuisance-Modelle (`model_y`, `model_t`, `model_propensity`) sind
Klassifikatoren, ebenso `model_regression` (DRLearner, seit `discrete_outcome=True`).
Die Meta-Learner Outcome-Modelle (`overall_model`, `models`, `cate_models`) bleiben Regressoren.

Damit das Final-Modell nicht mit den Nuisance-Modellen vermischt wird, gibt es eine separate
Konfigsektion:

```yml
final_model_tuning:
  enabled: true
  n_trials: 100
  models: [NonParamDML]
  single_fold: false
  fixed_params: {}
```

Wesentliche Punkte:

- Es wird ausschließlich das Final-Modell getunt (Rolle `model_final`).
- Beide Modelle (NonParamDML, DRLearner) nutzen äußere OOF-CV: est.fit(train) + est.score(val) pro Trial.
- DRLearner: Nutzt immer den eingebauten `score() + CV` (bereits DR-basiert).
- Das Tuning läuft **nur einmal pro Run**;
  die Parameter werden anschließend für alle weiteren Folds wiederverwendet ("Locking").

### Modellauswahl (`models`)

Statt alle FMT-fähigen Modelle zu tunen, kann mit `models` gezielt festgelegt werden, welche Modelle optimiert werden:

- `models: null` → alle FMT-fähigen Modelle (NonParamDML, DRLearner) werden getuned
- `models: [NonParamDML]` → nur NonParamDML wird getuned, DRLearner nutzt `fixed_params`
- `models: [NonParamDML, DRLearner]` → beide werden getuned

Das ist nützlich, weil die Kosten pro Trial zwischen den Modellen stark unterschiedlich sind:

- **Beide Modelle** verwenden äußere CV mit cache_values: Nuisance einmalig gecacht, pro Trial K × refit_final() + est.score(val)

Man kann z.B. nur NonParamDML tunen und DRLearner mit bewährten festen Parametern laufen lassen.

### Single-Fold für DRLearner (`single_fold`)

Analog zum Base-Learner-Tuning kann mit `single_fold: true` die äußere CV auf 1 Fold reduziert werden. Das reduziert die Fits von K×Trials auf 1×Trials. Gilt für beide Modelle (NonParamDML + DRLearner). Die gleiche Mindestanforderung gilt: min(n_treated, n_positive) / K ≥ 100 empfohlen (≥ 50 Minimum).

### Overfit-Penalty (`overfit_penalty`, `overfit_tolerance`)

Beide FMT-Modelle (NonParamDML und DRLearner) nutzen **äußere Cross-Validation**: model_final wird auf Train gefittet und auf komplett Out-of-Fold-Val bewertet (`est.score()`). Dies verhindert optimistische Score-Schätzungen, die entstehen, wenn model_final auf denselben Daten evaluiert wird, auf denen es trainiert wurde.

**Score-Konvention:** Beide FMT-Modelle werden über ihre **nativen EconML est.score()-Methoden** bewertet:

- **NonParamDML:** `est.score()` = R-Loss (Nie & Wager): `E[(Y_res − τ·T_res)²]`
- **DRLearner:** `est.score()` = DR-MSE: doubly-robust Pseudo-Outcome-MSE

Beide Metriken sind kausal valide, nutzen aber unterschiedliche Ansätze.
Der Vorteil nativer Metriken: Kein Train-Eval-Mismatch (das Modell wird mit
derselben Metrik evaluiert, auf die es intern optimiert).

Da Optuna `direction="maximize"` verwendet, werden beide Scores negiert (`-MSE`).

Die Overfit-Penalty ist **skalen-sicher** durch relative Tolerance: `tolerance=0.05` bedeutet "5% relativer
Gap wird toleriert". Damit funktioniert die Penalty identisch für R-Loss (~0.001) und
DR-MSE (~0.01) ohne manuelle Skaleneinstellung.

Der reine OOF-Score kann dennoch Modelle bevorzugen, die Rauschen statt echte Heterogenität lernen — insbesondere bei kleinen Treatment-Effekten oder wenigen Beobachtungen. Die Overfit-Penalty adressiert das, indem sie den Train-Val-Gap misst:

```
scale = |val_score|
relative_gap = (train_score - val_score) / scale
adjusted = val_score - penalty × scale × max(0, relative_gap - tolerance)
```

wobei die Tolerance als relativer Anteil wirkt (z.B. 0.05 = 5% Gap toleriert). Die Formel ist skalen-sicher über alle Metriken.

**BLT- und FMT-spezifische Werte (identische Formel, unterschiedliche Presets):**

| Stufe | BLT (Nuisance) | FMT (CATE) | Begründung |
|---|---|---|---|
| **Aus** (Default) | p=0, t=0.15 | p=0, t=0.05 | Reiner Val-Score, keine Penalty |
| **Moderat** | p=0.2, t=0.15 | p=0.3, t=0.05 | Balance zwischen Performance und Stabilität |
| **Stark** | p=0.4, t=0.08 | p=0.6, t=0.03 | Für kleine Stichproben oder instabile Vorlaufe |

BLT ist toleranter (15% Gap): DML-Orthogonalität kompensiert moderates Nuisance-Overfitting. FMT ist strenger (5% Gap): CATE-Signal ist schwächer als Outcome-Signal, Overfitting ist schneller problematisch.

### Feste Parameter (`fixed_params`)

Wenn FMT deaktiviert ist oder ein Modell nicht in `models` steht, werden die `fixed_params` direkt als Hyperparameter für `model_final` verwendet. Das ermöglicht eine bewusste Kombination: Tuning für ein Modell, feste Parameter für ein anderes.

### Nuisance-Caching (`cache_values`)

FMT nutzt EconML's `cache_values=True` zur drastischen Laufzeitreduktion. Die äußeren CV-Folds verwenden
denselben `tuning_seed` für alle Trials — die Fold-Zuordnung (tr/va) ist also über alle Trials identisch.
Nur `model_final` ändert sich pro Trial. Daher werden die Nuisance-Modelle (model_y, model_t für NonParamDML;
model_propensity, model_regression für DRLearner) **einmalig pro äußerem Fold** gefittet und gecacht.

**Ablauf:**

1. **Pre-Fit (einmalig):** Pro äußerem Fold wird ein vollständiger Estimator mit `cache_values=True` gefittet.
   EconML speichert die Out-of-Fold-Residuals (Y_res, T_res) und die gefitteten Nuisance-Modelle.
2. **Pro Trial:** Nur `model_final` wird getauscht (`est.model_final = new_model`) und via `est.refit_final()`
   auf den gecachten Residuals neu gefittet. Die Nuisance-Phase entfällt komplett.
3. **Bewertung:** `est.score(Y_val, T_val, X_val)` nutzt die gespeicherten Nuisance-Modelle für Predictions
   auf dem Val-Set — kein Nuisance-Training, nur schnelle Vorhersage.

**Laufzeiteffekt (Beispiel: 5 äußere Folds × 5 innere Folds × 50 Trials):**

| Phase | Ohne Cache | Mit Cache |
|---|---|---|
| Nuisance-Fits | 50 × 5 × 10 = 2.500 | 5 × 10 = **50** (einmalig) |
| model_final Fits | 50 × 5 = 250 | 5 + 50 × 5 = **255** (5 initial + 250 refit) |
| **Gesamt** | **2.750** | **305** |

Die sequentielle Trial-Ausführung (`n_jobs=1`) ist trotzdem schneller als parallele Trials ohne Cache,
weil die Nuisance-Phase (~80% der Laufzeit) komplett entfällt.

Das gleiche Prinzip gilt für **CFT** (CausalForestDML): Die Nuisance-Modelle werden einmalig gecacht,
dann pro Trial nur die GRF-Forest-Parameter via `est.set_params()` + `est.refit_final()` geändert.

## Search Space in der Config

Die Tuning-Ranges können direkt in der YAML definiert werden. Dafür gibt es getrennte Bereiche für

- `tuning.search_space` für die Base Learner der Nuisance-Modelle
- `final_model_tuning.search_space` für das Final-Modell (`model_final`)

Innerhalb jedes Bereichs werden LightGBM und CatBoost getrennt gepflegt:

```yml
tuning:
  enabled: true
  n_trials: 50
  search_space:
    lgbm:
      n_estimators: {type: "int", low: 200, high: 600}
      learning_rate: {type: "float", low: 0.01, high: 0.15, log: true}
      num_leaves: {type: "int", low: 7, high: 40}
      max_depth: {type: "int", low: 3, high: 6}
      min_child_samples: {type: "int", low: 5, high: 200}
      min_child_weight: {type: "float", low: 0.001, high: 10.0, log: true}
      subsample: {type: "float", low: 0.6, high: 1.0}
      subsample_freq: {type: "int", low: 1, high: 7}
      colsample_bytree: {type: "float", low: 0.6, high: 1.0}
      min_split_gain: {type: "float", low: 0.0, high: 1.0}
      reg_alpha: {type: "float", low: 0.00000001, high: 10.0, log: true}
      reg_lambda: {type: "float", low: 0.00000001, high: 10.0, log: true}
    catboost:
      iterations: {type: "int", low: 200, high: 600}
      learning_rate: {type: "float", low: 0.01, high: 0.15, log: true}
      depth: {type: "int", low: 4, high: 8}
      l2_leaf_reg: {type: "float", low: 1.0, high: 30.0, log: true}
      random_strength: {type: "float", low: 0.01, high: 10.0, log: true}
      subsample: {type: "float", low: 0.5, high: 1.0}
      rsm: {type: "float", low: 0.3, high: 0.9}
      min_data_in_leaf: {type: "int", low: 10, high: 200}
      model_size_reg: {type: "float", low: 0.0, high: 10.0}
      leaf_estimation_iterations: {type: "int", low: 1, high: 10}

final_model_tuning:
  enabled: true
  n_trials: 50
  cv_splits: 5
  overfit_penalty: 0.0
  overfit_tolerance: 0.05
  search_space:
    lgbm:
      n_estimators: {type: "int", low: 100, high: 400}
      learning_rate: {type: "float", low: 0.005, high: 0.12, log: true}
      num_leaves: {type: "int", low: 7, high: 63}
      max_depth: {type: "int", low: 2, high: 6}
      min_child_samples: {type: "int", low: 20, high: 500}
      max_bin: {type: "int", low: 10, high: 63}
      subsample: {type: "float", low: 0.5, high: 0.9}
      colsample_bytree: {type: "float", low: 0.5, high: 0.9}
      min_split_gain: {type: "float", low: 0.0, high: 5.0}
      reg_alpha: {type: "float", low: 0.00000001, high: 50.0, log: true}
      reg_lambda: {type: "float", low: 0.00000001, high: 50.0, log: true}
      min_child_weight: {type: "float", low: 0.001, high: 20.0, log: true}
      path_smooth: {type: "float", low: 0.0, high: 10.0}
```

Die FMT-Suchräume sind etwas konservativer als die BL-Suchräume, aber breiter als in früheren Versionen: `min_child_samples` 20–500, `num_leaves` 7–63, `n_estimators` 100–400. Bei LightGBM wird automatisch `subsample_freq=1` injiziert, wenn `subsample` im Suchraum ist — das erzwingt Bagging in jedem Boosting-Schritt.

Unterstützte Parametertypen:

- `type: "int"`
- `type: "float"`
- `type: "categorical"`

Optionale Felder:

- `log: true` für logarithmische Suche
- `step` für lineare Raster bei numerischen Parametern
- `choices` für kategoriale Parameter

Wenn `search_space` leer bleibt, verwendet rubin weiterhin die internen Standard-Ranges.

Die YAML ist bereits generisch aufgebaut: Du kannst auch weitere gültige LightGBM- oder CatBoost-Parameter ergänzen, solange sie von der jeweiligen sklearn-API akzeptiert werden. Die oben gezeigten Parameter sind nur die vordefinierten Standardräume.

## Task-basiertes Sharing
Die Tuning-Logik arbeitet nicht modellweise, sondern task-basiert.
Dafür wird aus `models_to_train` zunächst ein interner Trainingsplan abgeleitet.
Eine Task wird über folgende Merkmale beschrieben:

- Base-Learner-Familie
- Objective-Familie (`outcome`, `outcome_regression`, `grouped_outcome`, `grouped_outcome_regression`, `propensity`, `pseudo_effect`)
- Estimator-Task (`classifier` oder `regressor`)
- Sample-Scope (`all`, `group_specific_shared_params`, ...)
- Nutzung des Treatment-Features
- Zieltyp (`Y`, `T`, `D`)

Nur wenn diese Signatur identisch ist, wird ein Tuning-Ergebnis geteilt.
Dadurch werden gleiche Aufgaben nur einmal gerechnet, ohne unterschiedliche Lernprobleme künstlich zusammenzuwerfen.

Typische geteilte Aufgaben im Repo:

- `model_y` wird über NonParamDML, ParamDML und CausalForestDML geteilt (gleiche Outcome-Klassifikation)
- `model_t` (DML) und `model_propensity` (DRLearner) teilen die gleiche Propensity-Klassifikation
- gruppenspezifische Outcome-Modelle von `TLearner` und `XLearner`
- separate Regressions-Tasks für CATE-/Final-Modelle

### Signatur-Trennung: `all` vs. `all_direct`

Modelle mit gleicher statistischer Aufgabe, aber unterschiedlicher effektiver Trainingsmenge in Produktion werden über den `sample_scope` getrennt:

- **`all`**: Modelle, die innerhalb von DML/DR-internem Cross-Fitting (cv=5) trainiert werden → sehen ~64% der Gesamtdaten in Produktion (80% äußerer Fold × 80% innerer Fold).
- **`all_direct`**: Meta-Learner-Modelle, die direkt auf dem äußeren Fold trainiert werden (kein internes CV) → sehen ~80% der Gesamtdaten.

Betroffene Trennungen:
- **outcome**: DRLearner.model_regression (`all`, ~64% N, Classifier seit discrete_outcome=True) ↔ SLearner.overall_model (`all_direct`, ~80% N) — separate Tasks.
- **propensity**: DML/DR-Propensity (`all`, ~64% N) ↔ XLearner.propensity_model (`all_direct`, ~80% N) — separate Tasks.

Ohne diese Trennung würden Modelle mit sehr unterschiedlichen Trainingsmengen dasselbe Tuning teilen, was zu sub-optimalen Hyperparametern führt.

### CV-Architektur

Alle inneren CVs sind synchronisiert (Default: 5, steuerbar über ein UI-Feld):

| Stufe | Config-Feld | Default | Beschreibung |
|---|---|---|---|
| BLT Tuning | `tuning.cv_splits` | 5 | Innere CV pro Tuning-Trial |
| FMT Tuning (intern) | `final_model_tuning.cv_splits` | 5 | Nuisance-Cross-Fitting im FMT |
| DML/DR Produktion | `dml_crossfit_folds` | 5 | Internes Nuisance-Cross-Fitting |
| FMT DRLearner (äußer) | `cross_validation_splits` | 5 | Score-Folds für DRLearner-FMT |
| Cross-Predictions | `cross_validation_splits` | 5 | Äußere OOF-Evaluation |

**Wichtig:** `model_y` und `model_t` sind **keine** geteilte Aufgabe — sie haben verschiedene Zieltypen (Y vs. T) und Objective-Familien (Outcome vs. Propensity) und werden deshalb mit eigenem Optuna-Seed separat getunt.

Nuisance-Modelle mit gleicher Signatur teilen sich automatisch ein Tuning, um Rechenzeit zu sparen.

---


## Overfit-Penalty (BLT + FMT)

Die Overfit-Penalty bestraft Hyperparameter-Konfigurationen mit großem Train-Val-Gap:

```
scale = |val_score|
relative_gap = (train_score - val_score) / scale
adjusted = val_score - penalty × scale × max(0, relative_gap - tolerance)
```

### Penalty-Timing (Pruning vs. Bewertung)

Die Penalty wirkt **ausschließlich auf den finalen Trial-Score** — nicht auf die
Zwischenergebnisse, die dem Pruner gemeldet werden:

- **Pruning** basiert auf **RAW Val-Scores** (echte Generalisierungsleistung)
- **TPE-Sampler** modelliert die tatsächliche Performance (nicht verzerrte Scores)
- **Penalty** wird erst am Ende auf **stabile K-Fold-Mittelwerte** angewendet

Warum? Per-Fold-Gaps sind verrauscht — ein einzelner Fold mit zufällig hohem
Train-Val-Gap würde sonst vielversprechende Trials voreitig abbrechen. Die
Penalty auf den Mittelwert über alle Folds ist mathematisch stabiler:

```
penalty(mean(val_1..K), mean(train_1..K))     ← NEU: stabil
≠ mean(penalty(val_1,train_1), ..., penalty(val_K,train_K))  ← ALT: verrauscht
```

### BLT vs. FMT Defaults

| Stufe | Preset Moderat | Preset Stark | Default Tolerance | Begründung |
|---|---|---|---|---|
| **BLT** | p=0.2, t=0.15 | p=0.4, t=0.08 | 0.15 (15%) | DML-Orthogonalität kompensiert Nuisance-Overfitting |
| **FMT** | p=0.3, t=0.05 | p=0.6, t=0.03 | 0.05 (5%) | CATE-Signal schwächer — stärkere Regularisierung nötig |

Die Penalty ist **skalen-sicher** durch relative Tolerance. Damit funktioniert sie
identisch für log_loss (~0.3-0.7), R-Loss (~0.001) und DR-MSE (~0.01).

## Sonderfall: `CausalForestDML`

`CausalForestDML` besteht aus zwei Teilen:

1) **DML‑Residualisierung** mit Nuisance‑Modellen (`model_y`, `model_t`)  
2) **Causal Forest** als letzte Stufe

**BLT** optimiert die Base Learner der Nuisance-Modelle (model_y, model_t) — identisch zu NonParamDML.
**CFT** optimiert die Wald-Parameter der Forest-Stufe per Optuna TPE — siehe [CausalForest-Tuning (Optuna)](#causalforest-tuning-optuna) weiter unten.

## Sonderfall: `CausalForest` (Reiner GRF)

`CausalForest` (`econml.grf.CausalForest`) hat keine DML-Residualisierung und keine Nuisance-Modelle.
Er schätzt den Treatment-Effekt direkt mit Honest Estimation.

**BLT** ist für CausalForest nicht relevant (keine Base Learner).
**CFT** optimiert die Wald-Parameter per Optuna TPE. Nuisance-Residuen werden einmalig vorberechnet (mit BLT-getunten CatBoost/LightGBM Modellen), dann pro Trial nur der Forest evaluiert — siehe [CausalForest-Tuning (Optuna)](#causalforest-tuning-optuna).


## Persistente Optuna-Studies (SQLite)

### Konfiguration
Du kannst Optuna so konfigurieren, dass eine Study in einer SQLite-Datei gespeichert wird:

```yml
tuning:
  enabled: true
  n_trials: 50
  storage_path: "runs/optuna_studies/baselearner_tuning.db"
  study_name_prefix: "baselearner"
  reuse_study_if_exists: true
  optuna_seed: 42
```

### Warum ist das sinnvoll?
- **Fortsetzen** eines Tunings (z. B. wenn ein Lauf abbricht oder du später mehr Trials hinzufügen willst).
- **Transparenz**: Trial-Historie und beste Parameter lassen sich nachträglich analysieren.
- **Vergleichbarkeit** zwischen Runs.

### Study-Namen
Der Study-Name wird aus Prefix + Kontext zusammengesetzt, z. B.:

`baselearner__lgbm__outcome_regression__regressor__all__with_t__y`

Weitere typische Study-Namen:
- `baselearner__lgbm__outcome__classifier__all__no_t__y` (DML model_y)
- `baselearner__lgbm__propensity__classifier__all__no_t__t` (Propensity, geteilt)
- `baselearner__lgbm__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y` (TLearner/XLearner)

### Seed-Behandlung pro Study

Der TPE-Sampler erhält pro Study einen **eigenen Seed**, der deterministisch aus `optuna_seed` + Study-Key abgeleitet wird: `(optuna_seed + sha256(study_key)) % 2³¹`. Damit explorieren verschiedene Tasks (z. B. `model_y` vs. `model_t`) unterschiedliche Hyperparameter-Bereiche, obwohl sie denselben Basis-Seed verwenden. Die Ergebnisse bleiben reproduzierbar, solange `optuna_seed` und die Modellkonfiguration identisch sind.

### Wie TPE funktioniert

TPE (Tree-structured Parzen Estimator) ist ein Bayesian-Optimization-Algorithmus, der aus bisherigen Trial-Ergebnissen lernt, welche Hyperparameter-Regionen vielversprechend sind.

**Startup-Phase (erste ~10 Trials):** Optuna schlägt zufällige Parameter vor — reine Exploration, um ein initiales Bild des Suchraums zu bekommen.

**Informierte Phase (ab Trial ~11):** Für jeden neuen Trial führt TPE drei Schritte durch:

1. **Trials in "gut" und "schlecht" aufteilen.** TPE teilt alle bisherigen Ergebnisse am besten 25%-Quantil. Die Trials mit den besten Scores bilden die Verteilung **l(x)** ("good"), der Rest die Verteilung **g(x)** ("bad").

2. **Zwei getrennte Wahrscheinlichkeitsverteilungen fitten.** Für jeden Parameter wird eine Kernel-Density-Schätzung (KDE) erstellt — eine geglättete Verteilung der beobachteten Werte. l(x) zeigt, wo Parameter bei guten Trials lagen; g(x) zeigt, wo Parameter bei schlechten Trials lagen.

3. **Neuen Kandidaten sampeln, der l(x)/g(x) maximiert.** TPE wählt Parameter, die bei guten Trials häufig, aber bei schlechten Trials selten vorkamen. Dieses Verhältnis l(x)/g(x) ist die Acquisition-Funktion — es balanciert Exploration (unbekannte Regionen) und Exploitation (bekannt gute Regionen).

**Warum TPE für rubin besser als Grid Search ist:** Grid Search testet ein festes Raster (z. B. 3 Werte pro Parameter × 13 Parameter = Millionen Kombinationen). TPE lernt adaptiv und konzentriert sich auf die vielversprechendsten Bereiche. Bei 50 Trials findet TPE typischerweise bessere Hyperparameter als ein Grid Search mit 1000+ Kombinationen.

### TPE-Sampler-Optimierung

Der TPE-Sampler ist mit vier Optimierungen konfiguriert:

- **`multivariate=True`:** Modelliert Abhängigkeiten zwischen Parametern (z. B. learning_rate ↔ n_estimators). Ohne dies sampelt TPE jeden Parameter unabhängig, was gute Kombinationen langsamer findet.
- **`group=True`:** Partitioniert die Trial-Historie nach Parameter-Set. Relevant für `base_learner.type: "both"` — dort wählt Optuna pro Trial kategorisch zwischen `lgbm` und `catboost`, was zu unterschiedlichen Parameter-Sets pro Trial führt. Ohne `group` fällt TPE bei den "dynamischen" Parametern (z. B. CatBoost-spezifisch `l2_leaf_reg`, `random_strength`) auf `RandomSampler` zurück und produziert Warnungen wie *"sampled independently using RandomSampler instead of TPESampler"*. Mit `group=True` lernt TPE separate KDE-Modelle für jeden Zweig, was die Tuning-Qualität bei "both" signifikant verbessert. Eingeführt in Optuna 2.8; bei älteren Versionen fällt rubin automatisch auf `multivariate=True` ohne group zurück.
- **`constant_liar=True`:** Bei parallelen Trials (n_jobs > 1) weist TPE laufenden Trials einen Schätzwert zu, damit nicht dieselben Regionen doppelt gesampelt werden.
- **Adaptive `n_startup_trials`:** BL: `max(n_jobs, min(10, max(3, n_trials // 5)))`, FMT: `min(7, max(2, n_trials // 4))`, CFT: `max(5, n_trials // 5)`. Bei wenigen Trials (z. B. 50) wird die Random-Phase verkürzt, damit TPE mehr informierte Trials durchführt.

### Fold-basiertes Pruning (MedianPruner)

Jede Optuna-Study verwendet einen `MedianPruner(n_warmup_steps=1)`. Pro Objective meldet jeder Trial sein Zwischenergebnis nach jedem CV-Fold (`trial.report(mean_score, fold_i)`). Der MedianPruner vergleicht dieses Zwischenergebnis mit dem Median aller bisherigen Trials: Trials, die nach 2 Folds deutlich unter dem Median liegen, werden frühzeitig abgebrochen (`trial.should_prune()`). Das spart typischerweise 30–50% der Rechenzeit bei 5-Fold-CV, da schlechte Parameterkombinationen nicht alle 5 Folds durchlaufen müssen.

Pruning ist für alle Objective-Typen implementiert: `_objective_all_classification`, `_objective_all_regression`, `_objective_grouped_regression`, `_objective_xlearner_cate` sowie die FMT- und CFT-Objectives.

### Fehlerresilienz

- **`catch=(Exception,)`** auf allen `study.optimize()`-Aufrufen: Ein einzelner fehlgeschlagener Trial stoppt nicht die gesamte Study. Optuna loggt den Fehler und fährt mit dem nächsten Trial fort.
- **`study.best_params` Guard:** Falls alle Trials fehlschlagen (z. B. bei extremen Daten), wird auf Default-Parameter zurückgefallen statt eine ValueError zu werfen.
- **Trial-Statistik-Logging:** Nach dem Tuning wird geloggt, wie viele Trials erfolgreich waren vs. fehlgeschlagen.

---

## Kategorische Features in EconML

EconML konvertiert X intern zu numpy-Arrays (via `sklearn.check_array`), wodurch pandas `category`-Dtypes verloren gehen. Ohne Gegenmaßnahme nutzt LightGBM/CatBoost **ordinale Splits** auf nominalen Features statt nativer kategorialer Splits — deutlich schwächere Modellierung. CatBoost crasht zusätzlich, weil es float-Werte bei `cat_features` ablehnt.

rubin löst dieses Problem automatisch über den Context-Manager `patch_categorical_features` (`rubin/utils/categorical_patch.py`). Die Spaltenindizes der kategorialen Features werden vorab aus den `category`-/`object`-Spalten von X ermittelt. Die Patching-Strategie unterscheidet sich je nach Library:

- **LightGBM**: Nur `fit()` wird via `functools.partialmethod` gepatcht — `categorical_feature=<indices>` wird bei jedem `.fit()`-Aufruf injiziert. `predict()` braucht kein Patching (LGBM akzeptiert float-Kategorien nativ).
- **CatBoost**: `fit()`, `predict()`, `predict_proba()` und `score()` werden **alle** über custom Wrapper gepatcht. Zusätzlich zur `cat_features`-Injektion wird X von numpy-float zu einem DataFrame mit int32-Kategorien konvertiert (CatBoost verweigert float-Spalten bei `cat_features`). `score()` muss separat gepatcht werden, da CatBoost intern `_predict()` aufruft und damit den öffentlichen `predict()`-Wrapper umgeht.

Der Patch wird als Context-Manager um den gesamten ML-Kern gelegt (Tuning, Training, Evaluation, Surrogate, Bundle-Export). In der Pipeline gibt es zwei separate Kontexte: (1) Feature-Selektion (Spaltenindizes vor FS) und (2) Tuning + Training + Evaluation (Spaltenindizes nach FS). Im `finally`-Block werden alle originalen Methoden immer wiederhergestellt (kein globaler State-Leak).

---

## Parameter-Isolation: model_final

Die getunten Nuisance-Parameter (aus dem Base-Learner-Tuning für `model_y`/`model_t`) werden **nicht** an `model_final` vererbt. Hintergrund: Classifier-optimierte Parameter wie `min_split_gain=0.95` oder `min_child_samples=121` können bei CATE-Regression dazu führen, dass kein einziger Split akzeptiert wird — der Baum kollabiert zu einem Intercept (konstante Vorhersage für alle Samples).

`model_final` verwendet stattdessen:
- LightGBM/CatBoost-Standardwerte (wenn FMT deaktiviert)
- Explizit getunte Parameter via OOF-Score (neg. MSE, wenn FMT aktiviert)

### Overfit-Penalty (Train-Val-Gap-Regularisierung)

**Hintergrund:** Bei Double/Debiased ML (Chernozhukov et al., 2016) dienen die BLT-Nuisance-Modelle dazu, Residuals Ỹ = Y - E[Y|X] und T̃ = T - E[T|X] zu berechnen. Wenn diese Nuisance-Modelle overfitten, absorbieren sie kausales Signal — die Residuals werden systematisch zu klein, und model_final (CATE) findet weniger Heterogenität.

**Mechanismus:** Pro BLT-Trial wird zusätzlich zum Val-Score auch der Train-Score berechnet. Die Differenz (Gap = Train-Score - Val-Score) misst direkt das Overfit-Ausmaß. Große Gaps werden bestraft:

```
scale = |val_score|
relative_gap = (train_score - val_score) / scale
adjusted_score = val_score - penalty × scale × max(0, relative_gap - tolerance)
```

**Vorteile gegenüber Residual-Varianz-basierter Regularisierung:**
- Selbst-kalibrierend: unabhängig vom Signal-Rausch-Verhältnis der Daten
- Funktioniert mit `single_fold: true`
- Kein datenabhängiger Schwellenwert nötig

**Konfiguration:**
```yaml
tuning:
  overfit_penalty: 0.3    # 0 = deaktiviert (Default)
  overfit_tolerance: 0.15 # Relativer Gap-Schwellwert (15% für BLT)
```

**BLT-Presets (Nuisance-Regularisierung):**

| Preset | penalty | tolerance | Verhalten |
|---|---|---|---|
| **Aus** (Default) | 0.0 | 0.15 | Reiner Val-Score, keine Penalty |
| **Moderat** | 0.2 | 0.15 | 15% Gap toleriert — DML-Orthogonalität kompensiert |
| **Stark** | 0.4 | 0.08 | 8% Gap toleriert — konservativ bei kleinen Datensätzen |

Die FMT-Presets (CATE-Regularisierung) sind strenger: Moderat p=0.3/t=0.05, Stark p=0.6/t=0.03. Siehe Abschnitt „Overfit-Penalty (BLT + FMT)" für die vollständige Tabelle.

**Interaktion mit anderen Settings:**
- Bei `single_fold: true` ist die Penalty besonders wertvoll, weil die Trial-Bewertung ohnehin verrauschter ist.
- Bei `cv_splits: 5` (Default) ist der Gap natürlich kleiner als bei weniger Folds (mehr Trainingsdaten pro Fold → weniger Overfitting).
- Die Penalty wird auf **alle** BLT-Objectives angewendet: Klassifikation (Outcome, Propensity), Regression (SLearner, TLearner, XLearner) und gruppierte Regression (TLearner, XLearner).

## Tuning-Metriken-Gesamtübersicht

| Stufe | Modell | Metrik | Rohwert | An Optuna | Skill / R² | Overfit-Penalty |
|---|---|---|---|---|---|---|
| BLT | Nuisance (alle) | log_loss (Default) | + (lower=better) | negiert | Skill Score ✓ | Relativ ✓ |
| BLT | Nuisance (Regression) | neg_mse (Default) | + (lower=better) | negiert | R² ✓ | Relativ ✓ |
| BLT | XLearner CATE | _score_regressor | varies | varies | R² ✓ | Relativ ✓ |
| FMT | NonParamDML | est.score() = R-Loss | + (lower=better) | negiert | — (nur relativ) | Relativ ✓ |
| FMT | DRLearner | est.score() = DR-MSE | + (lower=better) | negiert | — (nur relativ) | Relativ ✓ |
| GRF | CausalForestDML | est.score() = R-Loss | + (lower=better) | negiert | — (nur relativ) | Nicht nötig (OOB) |
| GRF | CausalForest | R-Loss (manuell) | + (lower=better) | negiert | — (nur relativ) | Nicht nötig (OOB) |

**Wichtig:** Bei FMT/CFT ist `est.score()` der **rohe R-Loss** (MSE der Residual-auf-Residual-Regression), **nicht R²**. Der R-Loss ist nur für den Vergleich zwischen Trials derselben Study sinnvoll, nicht für absolute Interpretation. Ein normierter Score (wie bei BLT) würde den `RScorer` erfordern, der eine zusätzliche Baseline-Berechnung durchführt — das wäre zu teuer im Tuning-Loop.

### Warum unterschiedliche Metriken?

- **NonParamDML:** `est.score()` gibt nativ R-Loss zurück — die optimale Tuning-Metrik,
  weil model_final auch intern auf R-Loss trainiert wird (Residual-auf-Residual-Regression).
- **DRLearner:** `est.score()` gibt DR-MSE zurück — passend, weil model_final intern
  auf DR-Pseudo-Outcomes trainiert wird. R-Loss wäre hier ein Train-Eval-Mismatch.
- **GRF:** RScorer = normalisierter R-Loss. Identische Metrik wie NonParamDML.score(),
  nur auf [0,1] skaliert (analog R²). Nuisance-Modelle = BLT-getunte Base-Learner.

### Nuisance-Modell-Konsistenz

Alle kausalen Modelle verwenden dieselben BLT-getunten Nuisance-Modelle (model_y, model_t):
- NonParamDML/DRLearner: Direkt via Modellkonstruktion
- CausalForestDML: Via Modellkonstruktion (model_y, model_t mit BLT-Params)
- CausalForest: Via Pipeline-Übergabe (build_base_learner + BLT-Params für Residuen-Vorberechnung)

### CV-Zuordnung

- **Äußere CV** (`cross_validation_splits`): OOF-CATE-Predictions, FMT äußere Evaluation
- **Innere CV** (`dml_crossfit_folds` = `tuning.cv_splits` = `final_model_tuning.cv_splits`):
  BLT Fold-Splitting, FMT est.fit() intern, DRLearner Nuisance-Residuen,
  CausalForest RScorer, CausalForestDML tune()
## Val-Set-Overfitting-Schutz (Dual-Seed-System)

### Problem

Wenn Tuning und Cross-Prediction dieselben CV-Folds nutzen, entsteht **Val-Set-Overfitting**:

1. Optuna probiert 100–160 Hyperparameter-Kombinationen
2. Jede wird auf den Val-Folds bewertet
3. Die **beste** Kombination maximiert den Score auf genau diesen Folds
4. Cross-Prediction nutzt **dieselben** Folds für OOF-Predictions
5. → Die OOF-Scores sind optimistisch verzerrt

Dies ist kein klassisches Train-Overfitting, sondern eine **Selection Bias**: Die Hyperparameter wurden *für diese spezifischen Fold-Grenzen* optimiert.

### Lösung

Rubin nutzt zwei getrennte Seeds:

| Seed | Config-Key | Default | Verwendung |
|------|-----------|---------|------------|
| Random Seed | `constants.SEED` | 42 | Cross-Prediction, Modell-Internals, DRTester, SHAP |
| Tuning Seed | `constants.tuning_seed` | 18 | Nur Tuning-CV-Fold-Zuordnung |

```yaml
constants:
  SEED: 42
  tuning_seed: 18   # MUSS sich von SEED unterscheiden!
```

### Wie es wirkt

Die CV-Folds werden per `StratifiedKFold(shuffle=True, random_state=seed)` erzeugt. Verschiedene Seeds → verschiedene Fold-Zuordnungen:

- **Tuning** bewertet Params auf Folds {P, Q, R, S, T} (seed=18)
- **Cross-Prediction** evaluiert auf Folds {A, B, C, D, E} (seed=42)
- Überlappung zwischen Folds: ~20% (Zufallsniveau bei 5-Fold-CV)

### Geschützte Pfade

| Pfad | Schutz |
|------|--------|
| BLT Tuning → Cross-Prediction | Nuisance-Params auf Tuning-Folds selektiert, OOF auf CP-Folds |
| FMT Tuning → Cross-Prediction | CATE-Params auf Tuning-Folds selektiert, OOF auf CP-Folds. Volle Daten für maximale Score-Qualität |
| GRF Tuning → Cross-Prediction | EconML-interne OOB, unabhängig von beiden Seeds |
| DRTester | Eigene Nuisance + eigene CV mit random_seed |
| Champion-Selektion | Basiert auf unverzerrten OOF-Metriken |
| TMES / External Eval | Komplett separate Eval-Daten |

### Warnung bei gleichem Seed

Wenn `tuning_seed == SEED`, loggt die Pipeline eine Warnung und die Web-UI zeigt einen gelben Hinweis. In diesem Fall sind die Tuning- und CP-Folds identisch und der Schutz ist deaktiviert.

### Theoretische Einordnung

Der Dual-Seed-Ansatz ist ein Kompromiss zwischen drei Strategien:

| Strategie | Val-Set-Overfitting | Rechenkosten | Beschreibung |
|-----------|---------------------|-------------|--------------|
| Gleicher Seed | ✗ Ja (Selection Bias) | 1× | Tuning und Evaluation auf identischen Folds → optimistischer Score |
| **Dual-Seed (rubin)** | **✓ Stark reduziert** | **1×** | **Tuning-Folds ≠ Eval-Folds → unbiased OOF-Score** |
| Volle Nested CV | ✓ Eliminiert | K× teurer | Tuning wird pro Outer-Fold wiederholt (K × n_trials Trials) |

Volle Nested CV (sklearn: `cross_val_score(GridSearchCV(..., cv=inner_cv), ..., cv=outer_cv)`) ist der Goldstandard, aber bei kausalen ML-Modellen (EconML-Estimatoren mit internem Cross-Fitting, 50+ Optuna-Trials, 3 Tuning-Wellen) rechenökonomisch nicht umsetzbar — der Mehraufwand beträgt K × (BLT + FMT + CFT Trials).

Der Dual-Seed-Ansatz erzielt ~95% des Schutzes bei 0% Mehrkosten: Die getunten Parameter werden auf komplett anderen Fold-Grenzen evaluiert als beim Tuning. Die einzige verbleibende Korrelation ist, dass dieselben Datenpunkte in beiden Split-Schemata vorkommen (unvermeidbar ohne Holdout-Set). In der Praxis ist die Überlappung zwischen Tuning-Val-Folds und CP-Val-Folds bei 5-Fold-CV ~20% (Zufallsniveau), was die Selection Bias auf ein vernachlässigbares Niveau reduziert.

### Produktions-Datenregime (Train-Subsample)

DML/DR-Nuisance-Modelle (scope="all") sehen in Produktion nur (K-1)/K der äußeren Fold-Daten durch internes Cross-Fitting. BLT simuliert das, indem die Trainingsmenge pro Fold um den Faktor `(K-1)/K` subsampelt wird.

| Scope | Subsampling | Effektive Trainingsmenge (K=5) |
|-------|-------------|-------------------------------|
| `all` (DML/DR-Nuisance) | (K-1)/K = 80% | ~64% aller Daten |
| `all_direct` (Meta-Learner) | 100% | ~80% aller Daten |
| `group_specific` (TL/XL) | 100% | ~80% aller Daten |

Dadurch bleiben die Scopes immer getrennt (7 Tasks bei 8 Modellen). Meta-Learner und DML/DR-Nuisance werden nicht zusammengelegt.



### StratifiedKFold auf T×Y (Treatment × Outcome)

Alle Cross-Validations in rubin verwenden `StratifiedKFold(shuffle=True)` mit Stratifizierung auf der Kombination `T×Y`. Das bedeutet: in jedem Fold ist die Verteilung der vier Grundgruppen (T=0/Y=0, T=0/Y=1, T=1/Y=0, T=1/Y=1) annähernd identisch zum Gesamtdatensatz.

**Warum ist das bei kausaler Inferenz kritisch?**

Bei kausalen Cross-Predictions schätzt jedes Modell den Treatment-Effekt `E[Y(1)-Y(0)|X]`. Dafür benötigt es in jedem Fold ausreichend Beobachtungen aus allen vier T×Y-Zellen:

- **Ohne Stratifizierung** (einfaches KFold): Ein Fold könnte zufällig deutlich weniger Treated-Positive (T=1, Y=1) enthalten. Die Nuisance-Modelle (E[Y|X], P(T|X)) werden dann auf einer verzerrten Verteilung trainiert, was zu systematisch falschen Residuals führt. Bei kleinen Datensätzen oder stark unbalancierten Treatment-Gruppen (z.B. 10% Treated) kann ein Fold so wenige Treated-Positive enthalten, dass das Propensity-Modell oder das Outcome-Modell in dieser Zelle degeneriert.

- **Mit T×Y-Stratifizierung**: Jeder Fold hat garantiert die gleiche T×Y-Verteilung. Die Nuisance-Modelle trainieren auf einer repräsentativen Stichprobe, und die resultierenden Residuals sind unverzerrt. Das ist besonders wichtig für DML-basierte Modelle, bei denen die CATE-Schätzung auf den Residuals `Y - E[Y|X]` und `T - P(T|X)` basiert.

**Stellen im Code:**

| Stelle | Stratifizierung | Seed |
|--------|----------------|------|
| Cross-Prediction (äußere Folds) | T×Y | random_seed (42) |
| BLT äußere CV-Folds | Target (T oder Y) | tuning_cv_seed (18) |
| FMT äußere CV-Folds | T | tuning_cv_seed (18) |
| DML/DR-internes Cross-Fitting | T (via EconML) | random_seed (42) |
| DRTester Nuisance | T | random_seed (42) |
| Surrogate Cross-Prediction | T×Y | random_seed (42) |


## CausalForest-Tuning (Optuna)

### CausalForestDML
CausalForestDML wird via Optuna über 8 Wald-Parameter getunt (n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features, max_samples, min_var_fraction_leaf, min_impurity_decrease). Jeder Trial wird auf äußeren CV-Folds (tuning_cv_seed, StratifiedKFold T×Y) via `est.score()` (R-Loss, Nie & Wager 2021) bewertet. R-Loss = E[(Y_res − τ·T_res)²] — misst die Güte der CATE-Schätzung nach Robinson-Residualisierung. Nuisance-Modelle (model_y, model_t) nutzen die BLT-getunten Hyperparameter.

### CausalForest (GRF)
Der reine CausalForest (econml.grf) hat keine eigene `score()`-Methode. Stattdessen wird der R-Loss manuell berechnet: Nuisance-Modelle (BLT-getunte CatBoost/LightGBM) werden einmalig pro Fold vorberechnet, dann für jeden Trial die Residuen Y_res = Y − E[Y|X] und T_res = T − E[T|X] zur Bewertung genutzt. 9 Wald-Parameter (inkl. criterion: mse/het).

### Gemeinsame Eigenschaften
- **Stratifizierung:** T×Y (identisch mit BLT/FMT)
- **Seed:** tuning_cv_seed (identisch mit BLT/FMT)
- **Single-Fold:** Nur 1. Fold evaluiert (wie BLT/FMT)
- **n_jobs=1:** Verhindert joblib-Deadlocks bei CatBoost/LightGBM Nuisance
- **gc.collect()** nach jedem Fold

## Feature-Selektion

Die Feature-Selektion unterstützt drei Methoden, die per Union kombiniert werden:

- **catboost_importance** (Default): CatBoost-Regressor auf Outcome (Y), PredictionValuesChange (≈Gain). Native kategorische Feature-Unterstützung via Target Statistics, Ordered Boosting für stabilere Rankings.
- **lgbm_importance**: LightGBM-Regressor auf Outcome (Y), Split-Gain. Benötigt categorical_feature Patch.
- **causal_forest**: GRF CausalForest Feature-Importances — misst kausale Relevanz (Treatment-Effekt-Heterogenität pro Feature).

Alle Methoden werden innerhalb des `patch_categorical_features(X, base_learner_type="both")` Kontexts ausgeführt, sodass sowohl CatBoost als auch LightGBM die korrekten cat_features-Indizes erhalten.

### dtypes.json Auto-Erkennung
Die Pipeline sucht automatisch nach `dtypes.json` im Verzeichnis der X-Datei. Diese Datei wird von DataPrep erzeugt und enthält die korrekten Spalten-Dtypes (inkl. category). So werden kategorische Features auch ohne explizite `dtypes_file`-Konfiguration korrekt erkannt.

## CatBoost CPU-Optimierungen

Bei Datensätzen mit vielen kategorischen Features setzt rubin `max_ctr_complexity=1`, um die Trainingszeit massiv zu reduzieren.

### `max_ctr_complexity=1`

**Default: 4** — CatBoost generiert automatisch Kombinationen von kategorischen Features (Paare, Tripel, Vierer) und berechnet für jede Kombination Ordered Target Statistics. Bei 74 kategorischen Features entstehen so über 1 Million Kombinationen — das dominiert die Trainingszeit.

Mit `max_ctr_complexity=1` werden **keine Kombinationen** generiert. Die individuellen OTS pro Feature bleiben vollständig erhalten (Ordered Target Statistics + Ordered Boosting). Nur die automatischen Interaktionen zwischen kategorischen Features entfallen.

**Referenz:** CatBoost GitHub Issue #1859: "If I raise max_ctr_complexity this can get to multiple hours" (bei nur 22 kat. Features).

### Wo gilt diese Einstellung

- Feature-Selektion (`catboost_importance`)
- BLT Tuning (CatBoostClassifier + CatBoostRegressor)
- FMT Tuning (via EconML model_y/model_t)
- CF Tuning (CausalForestDML Nuisance)
- Training + Champion-Refit
