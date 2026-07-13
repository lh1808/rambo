# Optuna-Tuning der Base Learner

> **Modul-Struktur:** Die Implementierung lebt in `rubin/tuning/`:
> - `common.py` — Builder, Search-Space, Suggest-Helpers, Fold-Utilities
> - `base_learner.py` — `BaseLearnerTuner` (BLT)
> - `final_model.py` — `FinalModelTuner` (FMT)
> - `causal_forest.py` — `CausalForestTuner` (CFT)

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
  - **FMT/CFT:** Qini (Default bei RCT) oder R-Score via externem EconML `RScorer` (Default bei obs.; unabhängige Nuisance, 2-Fold T×Y Cross-Fitting). R-Score ≈ 0 = nicht besser als konstantes Effektmodell, R-Score > 0 = echte Heterogenität, R-Score < 0 = Overfitting. Der RScorer berechnet Base-MSE und Normalisierung intern.
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

Bei `study_type: "rct"` verfolgt rubin eine **zweistufige Strategie** für die Propensity-Modellierung:

### Stufe 1: BLT-Diagnose (20 Trials)

Das Propensity-Tuning wird automatisch auf **20 Trials** reduziert. Ziel ist nicht Optimierung, sondern ein **Diagnose-Check**: Bei einem echten RCT ist P(T=1|X) = const — das Propensity-Modell sollte das Treatment nicht aus den Features vorhersagen können (Skill ≈ 0).

```
# Erwartetes Ergebnis bei korrektem RCT:
RCT-Diagnose bestätigt: Propensity-Skill = 0.0000 ≈ 0.
Training verwendet konstante Propensity P(T|X) = mean(T) = 0.505.
```

Wenn der Propensity-Skill > 0.01 ist, erscheint eine Warnung:

```
⚠ RCT-Diagnose: Propensity-Skill = 0.0234 > 0.01.
Das Propensity-Modell kann Treatment besser als Zufall vorhersagen —
die Daten sind möglicherweise NICHT randomisiert.
```

Mögliche Ursachen:
- Post-Treatment-Variablen im Feature-Set (z.B. Variablen, die erst nach der Treatment-Zuweisung gemessen wurden)
- Unvollständige Randomisierung (Stratifizierung nach Kovariaten, Block-Randomisierung mit kleinen Blöcken)
- Leakage: Eine Feature-Spalte enthält implizit die Treatment-Information

### Stufe 2: Konstante Propensity beim Training (DummyClassifier)

Unabhängig vom BLT-Ergebnis wird bei `study_type: "rct"` für das **Training** aller Modelle ein `DummyClassifier(strategy="prior")` aus scikit-learn anstelle des BLT-getunten CatBoost-Modells verwendet. Dieser gibt für alle Samples die empirische Treatment-Rate als konstante Propensity zurück:

```
P(T=1|X) = mean(T) ≈ 0.5  (für alle X, X-unabhängig)
```

**Betroffene Rollen und Modelle:**

| Modell | Rolle | Effekt |
|--------|-------|--------|
| NonParamDML | `model_t` | T_res = T − mean(T), keine Korrelation mit X |
| CausalForestDML | `model_t` | Saubere DML-Residualisierung |
| ParamDML | `model_t` | Saubere DML-Residualisierung |
| DRLearner | `model_propensity` | 1/P(T|X) = const → keine verzerrten DR-Gewichte |
| XLearner | `propensity_model` | Gleichmäßige CATC/CATT-Gewichtung |

**Begründung:** Bei einem RCT ist die Treatment-Zuweisung per Definition unabhängig von X. Jedes Propensity-Modell, das Features nutzt, kann nur Rauschen lernen → overfittete P(T|X) verzerrt die DML-Residuen und DR-Gewichte. Der DummyClassifier ist theoretisch exakt und verhindert dieses Overfitting.

**Konsistente Anwendung:** Der DummyClassifier wird nicht nur beim Training, sondern auch in den Nuisance-Caching-Phasen von FMT und CFT verwendet, sodass alle Stufen der Pipeline dieselbe konstante Propensity nutzen.

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

Beide FMT-Modelle (NonParamDML und DRLearner) nutzen **äußere Cross-Validation**: model_final wird auf Train gefittet und auf komplett Out-of-Fold-Val bewertet. Der Scorer ist konfigurierbar:

```yaml
final_model_tuning:
  scorer: auto    # auto | qini | rscore
  # auto → Qini bei study_type='rct', R-Score bei 'observational'
```

### Scorer-Option 1: QiniScorer (Default bei RCT)

Optimiert direkt die **Ranking-Qualität** (wer profitiert am meisten von der Behandlung?). Architektur:

1. Pro Trial: `refit_final()` auf allen Folds → `est.effect(X_val)` pro Fold
2. Alle Fold-Predictions aggregieren (OOF — jeder Datenpunkt genau 1× in Val)
3. **Ein** Qini-Koeffizient über die gesamte aggregierte Menge → Optuna-Objective

Vorteile:
- Kein separater RScorer-Cache nötig (weniger Speicher + Komplexität)
- Robust bei schwachem CATE-Signal: Ranking kann informativ sein auch wenn absolute Effekte klein sind — verhindert den Intercept-Kollaps (Mahajan et al. 2023: "Qini DR score ended up as a dominating metric")
- Direkt auf der Business-Metrik optimiert (Uplift-Ranking = Targeting-Qualität)

Einschränkungen:
- Kein Pruning (alle Folds müssen durchlaufen, bevor Qini berechnet werden kann)
- Nur bei RCT unverzerrt — bei Beobachtungsdaten wäre DR-Qini nötig (nicht implementiert)

### Scorer-Option 2: RScorer (Default bei Beobachtungsdaten)

**Score-Konvention (bei `scorer: rscore`):** Beide FMT-Modelle werden über den EconML **RScorer** bewertet — mit unabhängiger Nuisance (2-Fold T×Y-stratifiziertes Cross-Fitting auf Val-Daten) und R-Score als Metrik. Bei `scorer: qini` (Default bei RCT) wird stattdessen der aggregierte OOF-Qini als Metrik verwendet (siehe Scorer-Option 1 oben).

- **NonParamDML + DRLearner:** Einheitlicher R-Score via `scorer.score(est)`. Der RScorer berechnet intern: R-Score = 1 − MSE(CATE) / MSE(Intercept-Only). Beide Modelle sind direkt vergleichbar (gleiche Nuisance, gleiche Metrik).

Der R-Score verhindert, dass Optuna zu konservativen Konfigurationen (Intercept-Only) konvergiert.
Der Vorteil nativer Metriken: Kein Train-Eval-Mismatch (das Modell wird mit
derselben Metrik evaluiert, auf die es intern optimiert).

Da Optuna `direction="maximize"` verwendet, werden beide Scores negiert (`-MSE`).

Die Overfit-Penalty ist **skalen-sicher** durch relative Tolerance: `tolerance=0.10` bedeutet "10% relativer
Gap wird toleriert". Damit funktioniert die Penalty identisch für R-Score (~0-1) und
DR-MSE (~0.01) ohne manuelle Skaleneinstellung.

Der reine OOF-Score kann dennoch Modelle bevorzugen, die Rauschen statt echte Heterogenität lernen — insbesondere bei kleinen Treatment-Effekten oder wenigen Beobachtungen. Die Overfit-Penalty adressiert das, indem sie den Train-Val-Gap misst:

```
scale = max(|val_score|, 1e-8)
relative_gap = (train_score - val_score) / scale
excess = min(max(0, relative_gap - tolerance), max_penalized_gap)
adjusted = val_score - penalty × scale × excess
```

wobei die Tolerance als relativer Anteil wirkt (z.B. 0.10 = 10% Gap toleriert). Die Formel ist skalen-sicher über alle Metriken. Der Scale-Floor `1e-8` hält die Division stabil; bei val_score ≈ 0 (z.B. R-Score nahe Null bei schwachem CATE-Signal) kann der relative Gap dennoch groß werden — dagegen deckelt `overfit_max_penalized_gap` (Default 1.0) den bestraften Exzess-Gap (Saturierung) und verhindert, dass die Penalty das Vorzeichen kippt oder die Selektion dominiert.

**BLT- und FMT-spezifische Werte (identische Formel, unterschiedliche Presets):**

| Stufe | BLT (nur Meta-Learner) | FMT/CFT (CATE) | Begründung |
|---|---|---|---|
| **Aus** (Default) | p=0, t=0.20 | p=0, t=0.10 | Reiner Val-Score, keine Penalty |
| **Moderat** | p=0.2, t=0.20 | p=0.2, t=0.10 | Milde Regularisierung auf Meta-Learner-Base-Models + CATE-Finalstufe |
| **Stark** | p=0.3, t=0.20 | p=0.35, t=0.10 | Stärkere Regularisierung (vom Cap geschützt) |

Die BLT-Penalty wirkt **ausschließlich auf Meta-Learner-Tasks** (S-/T-/X-Learner): Diese bilden den CATE ohne internes Cross-Fitting/Orthogonalisierung direkt aus den Base-Models, weshalb deren Overfitting unmittelbar in die CATE durchschlägt — dort ist die Penalty der primäre Overfitting-Hebel beim Tuning. **DML/DR-Nuisances (`sample_scope == "all"`) werden nie bestraft**, unabhängig von der Konfiguration: Cross-Fitting + Orthogonalität fangen deren Overfitting bereits ab (Chernozhukov et al. 2018, Bach et al. 2024). Die Gating-Logik steckt in `BaseLearnerTuner._penalty_applies`. Die Presets setzen daher eine milde Meta-Learner-BLT-Penalty plus eine Penalty auf der CATE-Finalstufe (FMT/CFT); „Stark" erhöht die Penaltys, nicht die Toleranz.

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
dann pro Trial nur die GRF-Forest-Parameter via `setattr()` + `est.refit_final()` geändert
(`set_params()` existiert nicht bei CausalForestDML).

**RCT-Modus:** Bei `study_type: "rct"` wird model_t (NonParamDML/CausalForestDML) bzw. model_propensity
(DRLearner) im Nuisance-Cache als `DummyClassifier(strategy="prior")` eingesetzt — konstante
Propensity P(T|X) = mean(T), verhindert Overfitting auf Rauschen.

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
  overfit_tolerance: 0.10
  search_space:
    lgbm:
      n_estimators: {type: "int", low: 100, high: 400}
      learning_rate: {type: "float", low: 0.005, high: 0.12, log: true}
      num_leaves: {type: "int", low: 7, high: 63}
      max_depth: {type: "int", low: 2, high: 6}
      min_child_samples: {type: "int", low: 20, high: 200}
      max_bin: {type: "int", low: 10, high: 63}
      subsample: {type: "float", low: 0.4, high: 0.85}
      colsample_bytree: {type: "float", low: 0.3, high: 0.7}
      min_split_gain: {type: "float", low: 0.0, high: 0.5}
      reg_alpha: {type: "float", low: 0.000001, high: 10.0}
      reg_lambda: {type: "float", low: 0.000001, high: 10.0}
      min_child_weight: {type: "float", low: 0.5, high: 20.0}
      path_smooth: {type: "float", low: 0.0, high: 5.0}
```

Die FMT-Suchräume erlauben bewusst **weniger Regularisierung** als die BL-Suchräume: `reg_alpha` max 10 (BLT: 20), `reg_lambda` max 10 (BLT: 20), `min_child_samples` max 200 (BLT: 200), `path_smooth` max 5 (BLT: 10). Hintergrund: Das CATE-Signal ist schwächer als das Nuisance-Signal. Bei zu hohen Regularisierungs-Obergrenzen konvergiert Optuna auf maximale Regularisierung (Intercept-Kollaps), weil eine Konstante R-Score ≈ 0 erreicht und jeder Heterogenitäts-Versuch R-Score < 0 liefert. Bei LightGBM wird automatisch `subsample_freq=1` injiziert, wenn `subsample` im Suchraum ist — das erzwingt Bagging in jedem Boosting-Schritt.

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
scale = max(|val_score|, 1e-8)
relative_gap = (train_score - val_score) / scale
excess = min(max(0, relative_gap - tolerance), max_penalized_gap)
adjusted = val_score - penalty × scale × excess
```

**Hinweis zur Interaktion mit dem externen RScorer (bei `scorer: rscore`):** Der externe RScorer (unabhängige Nuisance, 2-Fold T×Y Cross-Fitting) gibt dem Val-Score drei Ebenen des Overfitting-Schutzes: (1) unabhängige Nuisance (kein Leak), (2) R-Score < 0 = natürliche Overfitting-Erkennung, (3) K-Fold-Mittelung. Der Train-Val-Gap ist damit ein **sauberes Signal** (nur model_final-Overfitting, keine Nuisance-Artefakte), aber typischerweise **klein**. Die Penalty ist daher wenig kritisch — EconML's eigenes `tune()` verwendet keine Penalty. Sie dient als optionales Safety-Net (Default: 0 = deaktiviert).

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
| **BLT (nur Meta-Learner)** | p=0.2, t=0.20 | p=0.3, t=0.20 | 0.20 (20%) | Wirkt nur auf S/T/X-Learner-Base-Models; DML/DR-Nuisances werden nie bestraft (Cross-Fitting + Orthogonalität) |
| **FMT/CFT** | p=0.2, t=0.10 | p=0.35, t=0.10 | 0.10 (10%) | CATE-Stufe: milde Penalty gegen echtes Overfitting, vom `overfit_max_penalized_gap`-Cap gegen Entgleisen geschützt |

Die Penalty ist **skalen-sicher** durch relative Tolerance. Damit funktioniert sie
identisch für log_loss (~0.3-0.7) und R-Score (~0-1).

## Sonderfall: `CausalForestDML`

`CausalForestDML` besteht aus zwei Teilen:

1) **DML‑Residualisierung** mit Nuisance‑Modellen (`model_y`, `model_t`)  
2) **Causal Forest** als letzte Stufe

**BLT** optimiert die Base Learner der Nuisance-Modelle (model_y, model_t) — identisch zu NonParamDML.
**CFT** optimiert die Forest-Parameter der Forest-Stufe per Optuna TPE — siehe [CausalForest-Tuning (Optuna)](#causalforest-tuning-optuna) weiter unten.

### Forest-Defaults

Die Defaults entsprechen den EconML-Defaults. CFT kann diese weiter optimieren:

| Parameter | Default | EconML | Begründung |
|-----------|---------|--------|------------|
| `max_depth` | `None` | `None` | Unbegrenzt — CFT findet die optimale Tiefe |
| `min_samples_leaf` | `5` | `5` | EconML Default |
| `min_samples_split` | `10` | `10` | EconML Default |
| `max_features` | `"auto"` | `"auto"` | Alle Features pro Split (= `n_features`) |
| `max_samples` | `0.45` | `0.45` | **Max. 0.5** bei `inference=True` (EconML-Constraint) |
| `min_var_fraction_leaf` | `None` | `None` | Keine Extra-Restriktion |

> **Hinweis:** `max_samples` ist auf EconML-Default 0.45 fixiert (nicht im Suchraum). Bei `inference=True` (Default) maximal 0.5 erlaubt.

## Sonderfall: `CausalForest` (Reiner GRF)

`CausalForest` (`econml.grf.CausalForest`) hat keine DML-Residualisierung und keine Nuisance-Modelle.
Er schätzt den Treatment-Effekt direkt mit Honest Estimation.

**BLT** ist für CausalForest nicht relevant (keine Base Learner).
**CFT** optimiert die Forest-Parameter per Optuna TPE. Nuisance-Residuen werden einmalig vorberechnet (mit BLT-getunten CatBoost/LightGBM Modellen), dann pro Trial nur der Forest evaluiert — siehe [CausalForest-Tuning (Optuna)](#causalforest-tuning-optuna).


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
- **`group=True`:** Partitioniert die Trial-Historie nach Parameter-Set. Relevant für `base_learner.type: "both"` — dort wählt Optuna pro Trial kategorisch zwischen `lgbm` und `catboost`, was zu unterschiedlichen Parameter-Sets pro Trial führt. Ohne `group` fällt TPE bei den "dynamischen" Parametern (z. B. CatBoost-spezifisch `l2_leaf_reg`, `random_strength`) auf `RandomSampler` zurück und produziert Warnungen wie *"sampled independently using RandomSampler instead of TPESampler"*. Mit `group=True` lernt TPE separate KDE-Modelle für jeden Zweig, was die Tuning-Qualität bei "both" signifikant verbessert. Benötigt Optuna ≥2.8 — durch den Pin `optuna>=3.5` immer verfügbar (kein Versions-Fallback im Code).
- **`constant_liar=True`:** Bei parallelen Trials (n_jobs > 1) weist TPE laufenden Trials einen Schätzwert zu, damit nicht dieselben Regionen doppelt gesampelt werden.
- **Adaptive `n_startup_trials`:** BL: `max(n_jobs, min(10, max(3, n_trials // 5)))`, FMT: `min(7, max(2, n_trials // 4))`, CFT: `max(5, n_trials // 5)`. Bei wenigen Trials (z. B. 50) wird die Random-Phase verkürzt, damit TPE mehr informierte Trials durchführt.

### Fold-basiertes Pruning (MedianPruner)

Jede Optuna-Study verwendet einen `MedianPruner(n_warmup_steps=1)`. Pro Objective meldet jeder Trial sein Zwischenergebnis nach jedem CV-Fold (`trial.report(mean_score, fold_i)`). Der MedianPruner vergleicht dieses Zwischenergebnis mit dem Median aller bisherigen Trials: Trials, die nach 2 Folds deutlich unter dem Median liegen, werden frühzeitig abgebrochen (`trial.should_prune()`). Das spart typischerweise 30–50% der Rechenzeit bei 5-Fold-CV, da schlechte Parameterkombinationen nicht alle 5 Folds durchlaufen müssen.

Pruning ist für alle BLT-Objective-Typen implementiert: `_objective_all_classification`, `_objective_all_regression`, `_objective_grouped_regression`, `_objective_xlearner_cate`. Bei FMT und CFT ist Pruning nur bei `scorer: rscore` aktiv (MedianPruner) — bei `scorer: qini` ist Pruning deaktiviert (NopPruner), da der aggregierte OOF-Qini erst nach allen Folds berechnet werden kann.

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

**Hintergrund:** Die BLT-Penalty wirkt **ausschließlich auf Meta-Learner-Tasks** (S-/T-/X-Learner). Diese trainieren ihre Base-Models direkt (`sample_scope` `"all_direct"`/`"group_specific"`) und bilden den CATE ohne internes Cross-Fitting/Orthogonalisierung — Overfitting der Base-Models schlägt daher unmittelbar in die CATE durch, und die Penalty ist hier der primäre Overfitting-Hebel beim Tuning. **DML/DR-Nuisances (`sample_scope == "all"`) werden hingegen nie bestraft**, unabhängig von der Konfiguration: Bei Double/Debiased ML (Chernozhukov et al., 2018) dienen sie nur dazu, Residuals Ỹ = Y − E[Y|X] und T̃ = T − E[T|X] zu berechnen, und Cross-Fitting (OOF-Vorhersagen) + `mc_iters` (Mittelung über Fold-Partitionen) + Neyman-Orthogonalität fangen ihr Overfitting bereits ab. Eine zusätzliche Penalty auf diese Stufe wäre redundant und könnte Richtung Unterfitting drücken (Bach et al. 2024). Die Gating-Logik steckt in `BaseLearnerTuner._penalty_applies` (gibt `task.sample_scope != "all"` zurück) — der `overfit_penalty`-Wert wird für DML/DR-Tasks schlicht ignoriert. Zusätzlich entfaltet die Penalty Nutzen auf der **CATE-Finalstufe** (FMT/CFT).

**Mechanismus:** Pro BLT-Trial wird zusätzlich zum Val-Score auch der Train-Score berechnet. Die Differenz (Gap = Train-Score - Val-Score) misst direkt das Overfit-Ausmaß. Große Gaps werden bestraft:

```
scale = max(|val_score|, 1e-8)
relative_gap = (train_score - val_score) / scale
excess = min(max(0, relative_gap - tolerance), max_penalized_gap)
adjusted_score = val_score - penalty × scale × excess
```

**Vorteile gegenüber Residual-Varianz-basierter Regularisierung:**
- Selbst-kalibrierend: unabhängig vom Signal-Rausch-Verhältnis der Daten
- Funktioniert mit `single_fold: true`
- Kein datenabhängiger Schwellenwert nötig

**Konfiguration:**
```yaml
tuning:
  overfit_penalty: 0.0    # BLT-Penalty: Default 0. Wirkt NUR auf Meta-Learner (S/T/X); DML/DR-Nuisances werden nie bestraft. Aktivierbar (z.B. 0.2).
  overfit_tolerance: 0.20 # Relativer Gap-Schwellwert (20% für BLT; greift nur bei Meta-Learner-Penalty)
  overfit_max_penalized_gap: 1.0  # Cap gegen Sign-Flip/Dominanz (Saturierung); <=0 = kein Cap
```

**BLT-Presets (nur Meta-Learner; DML/DR werden nie bestraft):**

| Preset | penalty | tolerance | Verhalten |
|---|---|---|---|
| **Aus** (Default) | 0.0 | 0.20 | Reiner Val-Score, keine Penalty |
| **Moderat** | 0.2 | 0.20 | Milde Penalty auf Meta-Learner-Base-Models |
| **Stark** | 0.3 | 0.20 | Stärkere Penalty auf Meta-Learner-Base-Models |

Die FMT/CFT-Presets (CATE-Regularisierung): Moderat p=0.2/t=0.10, Stark p=0.35/t=0.10 — beide via `overfit_max_penalized_gap` (Default 1.0) gegen Sign-Flip gedeckelt. Siehe Abschnitt „BLT vs. FMT Defaults" für die vollständige Tabelle.

**Interaktion mit anderen Settings:**
- Bei `single_fold: true` ist die Penalty besonders wertvoll, weil die Trial-Bewertung ohnehin verrauschter ist.
- Bei `cv_splits: 5` (Default) ist der Gap natürlich kleiner als bei weniger Folds (mehr Trainingsdaten pro Fold → weniger Overfitting).
- Die Penalty wird **nur auf Meta-Learner-BLT-Tasks** angewendet (S-Learner-Regression, T-/X-Learner gruppierte Regression, X-Learner Pseudo-Effekt + Propensity `all_direct`). DML/DR-Nuisance-Objectives (Outcome/Propensity mit `sample_scope == "all"`) werden nie bestraft — siehe `_penalty_applies`.

## Tuning-Metriken-Gesamtübersicht

| Stufe | Modell | Metrik | Rohwert | An Optuna | Skill / R² | Overfit-Penalty |
|---|---|---|---|---|---|---|
| BLT | Nuisance (alle) | log_loss (Default) | + (lower=better) | negiert | Skill Score ✓ | Relativ ✓ |
| BLT | Nuisance (Regression) | neg_mse (Default) | + (lower=better) | negiert | R² ✓ | Relativ ✓ |
| BLT | XLearner CATE | _score_regressor | varies | varies | R² ✓ | Relativ ✓ |
| FMT | NonParamDML | Qini (RCT) / R-Score (obs.) | higher=better | direkt | ✓ | Relativ ✓ |
| FMT | DRLearner | Qini (RCT) / R-Score (obs.) | higher=better | direkt | ✓ | Relativ ✓ |
| GRF | CausalForestDML | Qini (RCT) / R-Score (obs.) | higher=better | direkt | ✓ | Relativ ✓ |
| GRF | CausalForest | Qini (RCT) / R-Score (obs.+Adapter) | higher=better | direkt | ✓ | Relativ ✓ |

**R-Score via externer RScorer (bei `scorer: rscore`; Nie & Wager, 2021; Schuler et al., 2018):** FMT und CFT verwenden den EconML `RScorer` für die Bewertung. Der RScorer fittet **eigene, unabhängige** Nuisance-Modelle (model_y, model_t) mit 2-Fold T×Y-stratifiziertem Cross-Fitting auf den Validierungsdaten. Dann bewertet er den CATE-Estimator über `scorer.score(est)`, das intern `est.const_marginal_effect(X_val)` aufruft und den R-Score berechnet: R-Score = 1 − MSE(heterogen) / MSE(konstant). Vorteile: (1) Nuisance-Unabhängigkeit (kein Overfitting-Leak), (2) Einheitliche Metrik über alle Learner (NonParamDML, DRLearner, CFDML, GRF via CausalForestAdapter), (3) Konsistenz mit EconML's eigenem `tune()`-Ansatz. Scorer werden über Modelle gecacht (FMT: zwischen NonParamDML und DRLearner; CFT: zwischen CFDML und GRF).

### Warum R-Score statt rohem MSE?

- **Problem:** Ein roher MSE bestraft Heterogenität überproportional.
  Ein Intercept-Only-Modell (τ(X) = const) hat niedrigeren OOS-MSE als ein heterogenes
  Modell mit Varianz → Optuna konvergiert zur konservativen Ecke (Intercept-Kollaps).
- **Lösung:** R-Score = 1 − MSE(heterogen) / MSE(konstant). Normalisiert gegen die
  Intercept-Baseline. Intercept-Only → R-Score ≈ 0, echte Heterogenität → > 0.
- **Alle Learner:** Scoring via externem EconML `RScorer` mit unabhängiger Nuisance (2-Fold T×Y-stratifiziertes Cross-Fitting auf Val-Daten). Der RScorer berechnet Base-MSE und R-Score intern. Vorteile: Nuisance-Unabhängigkeit, einheitliche Metrik, Konsistenz mit EconML `tune()`.
- **CausalForest (GRF):** Via `CausalForestAdapter` (erbt von `BaseCateEstimator`) RScorer-kompatibel. `scorer.score(adapter)` ruft `adapter.effect(X_val)` → `cf.predict(X_val)` auf.
- Referenz: Nie & Wager (2021), Schuler et al. (2018), EconML RScorer-Konzept.

### Ranking-Erhaltung

Der R-Score ist eine **monotone Transformation** des rohen MSE: `R-Score = 1 − MSE / base_mse`, wobei `base_mse` für alle Trials desselben Folds konstant ist. Dadurch ist die Ranking-Ordnung der Trials **identisch** zum rohen MSE — der Trial mit dem niedrigsten MSE hat auch den höchsten R-Score. Die Normalisierung ändert nur die Skala, nicht die Reihenfolge. Optuna's TPE-Sampler profitiert von der normierten Skala (≈ 0 bis 1 statt beliebiger absoluter MSE-Werte).

### Kongeniality-Bias

Schuler et al. (2018) beobachten, dass Metriken wie R-Loss und DR-Loss dazu tendieren, den "kongenial" zugehörigen Learner zu bevorzugen (R-Loss → R-Learner, DR-Loss → DR-Learner). Mit dem einheitlichen externen RScorer nutzen **alle Learner dieselbe R-Score-Metrik** statt je Learner einer nativen Metrik. Dies ermöglicht direkte Vergleichbarkeit der Scores zwischen NonParamDML, DRLearner, CausalForestDML und CausalForest — ein Kongeniality-Bias ist damit ausgeschlossen.

### Base-MSE-Approximation (historisch)

Die Base-MSE wird intern vom RScorer berechnet (exakt, ohne Approximation über einen `DummyRegressor("mean") + refit_final()`-Umweg). Die Helper-Funktionen `_compute_base_mse_via_dummy` und `_compute_base_mse_from_residuals` sind reine Utilities und werden in keinem aktiven Scoring-Pfad verwendet.

### Nuisance-Modell-Konsistenz

Alle kausalen Modelle verwenden dieselben BLT-getunten Nuisance-Modelle (model_y, model_t):
- NonParamDML/DRLearner: Direkt via Modellkonstruktion
- CausalForestDML: Via Modellkonstruktion (model_y, model_t mit BLT-Params)
- CausalForest: Via Pipeline-Übergabe (build_base_learner + BLT-Params für Residuen-Vorberechnung)

### CV-Zuordnung

- **Äußere CV** (`cross_validation_splits`): OOF-CATE-Predictions, FMT äußere Evaluation
- **Innere CV** (`dml_crossfit_folds` = `tuning.cv_splits` = `final_model_tuning.cv_splits`):
  BLT Fold-Splitting, FMT est.fit() intern, DRLearner Nuisance-Residuen,
  CausalForest RScorer/QiniScorer, CausalForestDML tune()
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
| FMT Tuning (RScorer) → Cross-Prediction | CATE-Params auf Tuning-Folds selektiert, OOF auf CP-Folds. R-Score ≠ Eval-Metrik (Qini/AUUC) → kein direkter Metrik-Leak |
| FMT Tuning (QiniScorer) → Cross-Prediction | CATE-Params auf Tuning-Folds selektiert (OOF-Qini), OOF auf CP-Folds. Gleiche Metrik (Qini) → Leakage-Schutz **ausschließlich durch Dual-Seed** (unterschiedliche Fold-Grenzen) |
| GRF Tuning → Cross-Prediction | EconML-interne OOB, unabhängig von beiden Seeds |
| DRTester | Eigene Nuisance + eigene CV mit random_seed |
| Champion-Selektion | Basiert auf unverzerrten OOF-Metriken |
| TMES / External Eval | Komplett separate Eval-Daten |

#### QiniScorer Overfit-Penalty

Bei aktiviertem `overfit_penalty > 0` berechnet der QiniScorer neben dem OOF-Qini (Val) auch einen **In-Sample-Qini (Train)** pro Fold via `effect(X_train)`. Der Train-Val-Gap wird mit derselben skaleninvarianten Formel wie beim RScorer bestraft:

```
scale = max(|val_qini|, 1e-8)
relative_gap = (train_qini - val_qini) / scale
excess = min(max(0, relative_gap - tolerance), max_penalized_gap)
adjusted = val_qini - penalty × scale × excess
```

Die Formel ist metrisch-agnostisch: Ein 20%-Gap wird identisch bestraft, ob R-Score (~0.3) oder Qini (~0.0001). Der `overfit_max_penalized_gap` (Default 1.0) deckelt den bestraften relativen Gap (Saturierung): Bei kleinem Val-Score würde der relative Gap sonst explodieren und der Abzug das Vorzeichen kippen bzw. die Selektion dominieren; `<=0` deaktiviert den Cap. Bei `overfit_penalty: 0` (Default) wird kein Train-Qini berechnet — kein Laufzeit-Overhead.

### QiniScorer und Leakage

Bei `scorer: qini` optimiert Optuna direkt den Qini-Koeffizienten — dieselbe Metrik, die auch für die Champion-Auswahl verwendet wird. Anders als beim RScorer (wo die Tuning-Metrik R-Score ≠ Evaluationsmetrik Qini/AUUC) gibt es hier keinen natürlichen Metrik-Mismatch als Schutzschicht.

Der Leakage-Schutz wird stattdessen **vollständig durch das Dual-Seed-System** gewährleistet:

1. **Unterschiedliche Fold-Grenzen:** Die Tuning-Folds (tuning_seed=18) sind unabhängig von den CP-Folds (SEED=42). Optuna kann keine Konfiguration finden, die spezifisch auf die Evaluations-Folds überoptimiert ist.
2. **OOF-Qini:** Jeder Datenpunkt wird nur auf dem Fold bewertet, auf dem er *nicht* trainiert wurde. Kein In-Sample-Overfitting innerhalb des Tunings.
3. **~20% Zufallsüberlappung:** Bei 5-Fold-CV überlappen Tuning-Val und CP-Val auf Zufallsniveau (~20%) — die Selection Bias durch die Überlappung ist vernachlässigbar.

In der Praxis zeigt sich, dass der QiniScorer trotz gleicher Metrik keinen messbaren Evaluations-Leak erzeugt (Mahajan et al. 2023: "Qini DR score ended up as a dominating metric" — ohne Hinweise auf Selection Bias).

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

> **Klasse:** `CausalForestTuner` in `rubin/tuning/causal_forest.py`. Bei `scorer: rscore` wird der RScorer-Cache als Instanz-State gehalten, sodass Scorer zwischen CausalForestDML- und CausalForest-Calls wiederverwendet werden. Bei `scorer: qini` wird kein RScorer benötigt (OOF-Qini direkt aus Predictions).

### CausalForestDML
CausalForestDML wird via Optuna über 4 kausale Parameter getunt: `max_depth` (kategorisch: 3, 5, 7, 10, 15, None), `min_weight_fraction_leaf` (log-skaliert: 0.0001–0.05), `min_var_fraction_leaf` (log-skaliert: 0.0005–0.05), `criterion` (kategorisch: mse, het). Alle anderen Forest-Parameter werden auf EconML-Defaults fixiert (`n_estimators=100`, `min_samples_leaf=5`, `max_samples=0.45`, `max_features=auto`, `min_impurity_decrease=0.0`). Diese Reduktion von 9 auf 4 Parameter verhindert degenerierte Konfigurationen und verbessert die TPE-Effizienz.

Die kategorischen Auswahlen sind per YAML (`depth_choices`, `criterion_choices`) und in der App (Chip-Selektoren) konfigurierbar. Die Float-Ranges sind per `search_space` (low/high) und in der App (Slider) konfigurierbar. `forest_fixed_params` aus der YAML-Config werden als Basis-Overrides vor den Optuna-Suggestions angewendet.

Die Nuisance-Modelle (model_y, model_t) werden einmalig pro äußerem Fold gecacht (`cache_values=True`). Bei RCT wird model_t als `DummyClassifier(strategy="prior")` im Cache verwendet (konstante Propensity).

Pro Trial werden nur die Forest-Parameter via `setattr()` geändert und der Forest via `refit_final()` neu gefittet — `set_params()` existiert NICHT bei CausalForestDML (kein sklearn BaseEstimator). EconML's `_gen_model_final()` liest die aktuellen Attribute und baut bei jedem `refit_final()` einen frischen CausalForest.

Bewertung via QiniScorer (RCT, aggregierter OOF-Qini, kein Pruning) oder RScorer (obs., unabhängige Nuisance, 2-Fold T×Y-stratifiziertes Cross-Fitting auf Val-Daten, Nie & Wager 2021) auf äußeren CV-Folds (tuning_cv_seed, StratifiedKFold T×Y). Tuning mit `n_estimators=100` (EconML tune()-Konvention), Production mit 500 (konfigurierbar via `forest_fixed_params`).

### CausalForest (GRF)
Bei `scorer: rscore` wird der reine CausalForest (econml.grf) über den `CausalForestAdapter` (erbt von `BaseCateEstimator`, hat `effect()`) mit dem RScorer bewertet: Pro Fold wird ein RScorer mit unabhängiger Nuisance erstellt. Pro Trial wird ein `CausalForestAdapter(**trial_params)` gefittet und via `scorer.score(adapter)` bewertet. 4 Parameter: `max_depth`, `min_weight_fraction_leaf`, `min_var_fraction_leaf` + `criterion` (mse/het). Fixe Defaults für alle anderen Parameter. Tuning mit `n_estimators=100`, Production mit 500.

### Gemeinsame Eigenschaften
- **Stratifizierung:** T×Y (identisch mit BLT/FMT)
- **Seed:** tuning_cv_seed (identisch mit BLT/FMT)
- **Single-Fold:** Nur 1. Fold evaluiert (wie BLT/FMT)
- **n_jobs=1 (Trials):** Trials sequentiell, Forest intern n_jobs=-1 (alle CPUs)
- **n_estimators=100:** EconML-Default, fixiert (nicht im Suchraum). Muss durch subforest_size=4 teilbar sein (Safety-Net in `_sanitize_forest_params`).
- **gc.collect()** nach Cache-Aufbau und am Ende

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

---

## Optuna TPE-Sampler Optimierungen

Folgende Optimierungen sind in allen drei Tuning-Stufen (BLT, FMT, CFT) konsistent aktiviert:

### Lineare statt logarithmische Suche

Alle Hyperparameter werden **linear** gesampelt (`log=False`). Einzige Ausnahme: `learning_rate` (exponentieller Effekt → `log=True`). Bei linearem Sampling verteilt TPE die Trials gleichmäßig über den Suchraum, statt sich auf das untere Ende zu konzentrieren. Dies ist besonders wichtig für regularisierende Parameter wie `min_samples_leaf`, deren Optimum bei imbalanced Daten am oberen Rand liegen kann.

### `consider_endpoints=True`

TPE exploriert explizit die Grenzen des Suchraums. Ohne diese Option werden Randwerte systematisch unterexploriert. Relevant für Parameter, deren Optimum am Rand liegen kann (z.B. `min_samples_leaf` bei stark imbalanced Daten).

### `n_startup_trials` ≤ 50% der Trials

`n_startup_trials` (Anzahl zufälliger Trials vor TPE-Exploration) wird auf maximal die Hälfte der effektiven Trials pro Task gekappt. Ohne diese Kappung sind bei Propensity-Tasks mit 20 Trials alle 20 Trials zufällig — TPE wird nie aktiviert.

### `n_warmup_steps=2`

Der MedianPruner wartet 2 CV-Folds ab, bevor er Trials pruned (statt 1). Ein einzelner Fold ist zu verrauscht für eine zuverlässige Pruning-Entscheidung.

---

## Speicher-Optimierungen (BLT)

### `del model` in allen Objectives

Nach jedem CV-Fold wird das trainierte CatBoost/LightGBM-Modell explizit gelöscht (`del model`), um den nativen C++-Speicher sofort freizugeben. Ohne `del` hält Python die Referenz bis zum nächsten Garbage-Collection-Zyklus.

### `malloc_trim(0)` zwischen BLT-Tasks

Nach jedem BLT-Task wird `ctypes.CDLL("libc.so.6").malloc_trim(0)` aufgerufen. Dies zwingt glibc, freigegebenen C/C++-Speicher (CatBoost, LightGBM) ans Betriebssystem zurückzugeben. Ohne `malloc_trim` hält glibc den Speicher in seinem internen Pool — nach 100+ CatBoost-Trials pro Task akkumuliert das erheblich.

### `n_estimators` muss durch `subforest_size` (Default=4) teilbar sein

EconML's CausalForest und CausalForestDML nutzen Bootstrap-of-Little-Bags (BLB) für Inferenz (`inference=True`). Dafür werden die Bäume in Subforests der Größe `subforest_size=4` gruppiert. `n_estimators` muss exakt durch `subforest_size` teilbar sein, sonst wirft EconML einen `ValueError`.

rubin erzwingt dies automatisch:
- **CFT-Objective:** `n_estimators` ist auf EconML-Default 100 fixiert (nicht im Suchraum). Die 4 getunten Parameter sind: `max_depth` (kategorisch), `min_weight_fraction_leaf` (log-float), `min_var_fraction_leaf` (log-float), `criterion` (kategorisch)
- **UI-Slider:** `step=4`, Range `[100, 500]` — nur gültige Werte auswählbar
- **Defaults:** `_CFDML_DEFAULTS` und `_CF_DEFAULTS` verwenden `n_estimators=100` (÷4=25)

---

## Praxis-Tipp: Historischen Affinitätsscore als Feature nutzen

Wenn ein historischer Affinitätsscore verfügbar ist (z.B. aus einer früheren Kampagne, einem Churn-Modell oder einem Empfehlungssystem), kann dieser als **Spalte in der Feature-Matrix X** aufgenommen werden. Im DML-Framework verbessert ein guter prognostischer Score die Nuisance-Modelle (insbesondere model_y), was zu saubereren Residuals und präziserer CATE-Schätzung führt.

### Besonders wirksam bei imbalancierten Designs

Der Effekt ist besonders stark, wenn die **Control-Gruppe klein** ist (z.B. Treatment:Control = 2:1 oder 3:1). Das Outcome-Modell hat dann weniger Datenpunkte zum Lernen → verrauschtere Residuals → schwächeres CATE-Signal. Der Affinitätsscore kompensiert diesen Informationsverlust.

### Voraussetzungen

- Score aus **unabhängigen Daten** (andere Kampagne, anderer Zeitraum). Kein Leakage-Risiko.
- Score als normale Spalte in X mitführen — rubin behandelt ihn automatisch in allen Modellen.
- Der Qini-Vergleich (`data_files.s_file`) bleibt aussagekräftig: Er misst den Mehrwert des neuen Modells *über den Score hinaus*.

### Literatur

- Schuler et al. (2022): *Increasing the efficiency of randomized trial estimates via linear adjustment for a prognostic score.* PROCOVA-Grundlagenarbeit. EMA-qualifiziert 2022. Varianzreduktion ∝ R².
- Oquab et al. (2025): *Prediction-powered Inference for Clinical Trials.* Explizit für imbalancierte Designs: weniger Controls bei gleicher Power.
- Guo & Basse (2021): *Machine Learning for Variance Reduction in Online Experiments.* „Asymptotically the ML step can only increase precision."
