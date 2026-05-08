# Evaluation

In der Analyse-Pipeline werden drei Evaluationsphasen unterschieden:

## Studientyp und Evaluationen

Die Konfiguration `study_type` (Default: `"rct"`) bestimmt, welche Evaluationen berechnet werden:

| Evaluation | RCT | Beobachtungsdaten | Begründung |
|---|---|---|---|
| Qini / AUUC | ✓ | ✓ (als Ranking) | Bei RCT direkt kausal interpretierbar. Bei Beobachtungsdaten valide als Ranking zwischen Modellen. |
| Uplift@K% | ✓ | ✓ (als Ranking) | Wie Qini — relatives Ranking. |
| Policy Value (naiv) | ✓ | ✗ | Durch Selektionsbias verzerrt bei Beobachtungsdaten. |
| DR Policy Value (DRTester) | ✓ | ✓ | Doubly-Robust-korrigiert mit Bootstrap-CI — auch bei Beobachtungsdaten valide. |
| DRTester (BLP, CAL, Qini-CI, TOC-CI) | ✓ | ✓ | DR-Konstruktion korrigiert für Confounding. |
| CATE-Verteilung | ✓ | ✓ | Rein deskriptiv, immer nützlich. |
| Qini-Kurve / Percentile | ✓ | ✓ | Visuelle Uplift-Darstellung. |
| Treatment Balance | ✓ | ✗ | Bei Beobachtungsdaten systematisch nicht-flach (Confounding-Muster, nicht diagnostisch). |
| Policy-Value-Vergleich (Modell vs. Hist. Score) | ✓ | ✗ (naiv) / ✓ (DR) | Der naive Vergleich wird übersprungen, die DR-korrigierten Vergleichsplots bleiben. |

Bei `study_type: "observational"` werden im HTML-Report angepasste Interpretations-Texte angezeigt und bei Modellen ohne Confounding-Korrektur (SLearner, TLearner, CausalForest) erscheint ein Warnhinweis.

**Warum wird der naive Policy Value bei Beobachtungsdaten nicht berechnet?**
Der naive Policy Value berechnet E[Y | treated according to policy] direkt aus den beobachteten Outcomes. Bei Confounding sind Personen, die behandelt wurden und vom Modell empfohlen werden, möglicherweise besser dran weil sie ohnehin behandelt worden wären (Selektionsbias) — nicht wegen des Treatments. Die DR-korrigierten Policy Values aus dem DRTester nutzen stattdessen Doubly-Robust-Pseudo-Outcomes und sind auch bei Beobachtungsdaten valide.

**Warum wird Treatment Balance bei Beobachtungsdaten übersprungen?**
Die Treatment Balance Curve zeigt den Anteil der Treatment-Gruppe pro CATE-Dezil. Bei einem RCT sollte sie flach sein (Abweichungen deuten auf Randomisierungsprobleme). Bei Beobachtungsdaten ist sie systematisch nicht-flach, weil die Treatment-Zuweisung mit X korreliert — das zeigt nur das bekannte Confounding-Muster, liefert aber keine neue Diagnose-Information.

### Phase 1: Schnelle Metriken + CATE-Verteilung (alle Modelle)

**Sortierungsmetriken** (Qini, AUUC, Uplift@k, Policy Value) werden für **alle** Modelle berechnet — reines NumPy, <1s pro Modell. Grundlage für Champion-Selektion.

**CATE-Verteilungs-Plots** werden ebenfalls für alle Modelle erzeugt (~0.5s pro Modell). Zeigen Training-Predictions und Cross-Validated-Predictions nebeneinander als Histogramm. Dienen der visuellen Plausibilitätsprüfung: stark konzentrierte Verteilungen nahe Null deuten auf wenig Heterogenität hin, breite Verteilungen auf differenzierte Effektvorhersagen. Bei MT wird ein Plot pro Treatment-Arm erzeugt.

### Phase 2: DRTester-Diagnostik (Level-abhängig)

**Diagnose-Plots** (EconML DRTester) werden Level-abhängig erstellt: Level 1–2 alle Modelle, Level 3 Champion + Challenger, Level 4 nur Champion. Die DRTester-Nuisance-Modelle nutzen leichtere Varianten (n_estimators≤100, cv=5) für ~6-7× schnelleres Fitting bei minimaler Qualitätseinbuße. Bei Multi-Treatment werden die Nuisance-Fits pro Arm bei Level 3–4 parallel ausgeführt.

### Phase 3: Uplift-Plots (alle Modelle)

Qini-Kurve, Uplift-by-Percentile, Treatment-Balance — immer für **alle** Modelle, da schnell (~2-5s). Alle Plots sind native rubin-Implementierungen mit rubin-Farbpalette.

## DRTester (EconML)

Der `DRTester` aus EconML liefert u. a.:

- **BLP-Summary** — Best-Linear-Predictor-Test
- **Calibration Plot** — Kalibrierungs-Check
- **Qini Plot** mit Bootstrap-Konfidenzintervallen
- **TOC Plot** (Targeting Operating Characteristic)
- **Policy Values** mit Konfidenzintervallen

DRTester-Plots werden erzeugt, wenn Train-Daten (`Train_*`-Spalten) vorhanden sind. Im Cross-Modus erzeugt `train_and_crosspredict` sowohl Out-of-Fold-Predictions (`Predictions_*`) als auch In-Sample-Predictions (`Train_*`) auf dem gesamten Datensatz. Damit steht `has_train=True` und die Nuisance-Modelle werden mit `X_train=X` gefittet — alle DRTester-Tests (BLP, Calibration, Qini, TOC) laufen.

In rubin werden dafür die Cross-Predictions genutzt. Die `Train_*`-Vorhersage ist **keine** Performance-Schätzung, sondern dient für konsistente DRTester-Diagnosen.

Die Nuisance-Modelle werden **einmal** gefittet und für alle kausalen Modelle wiederverwendet (siehe „DRTester-Nuisance" weiter unten).

## Uplift-Plots (native Implementierung)

Die Pipeline erzeugt vier Uplift-Plots für alle Modelle:

- **Qini-Kurve** — `_native_qini_curve()`: Q(k) = Y_t(k) − Y_c(k) · N_t(k) / N_c(k), mit Random-Baseline, optionaler Perfect-Kurve und Qini-AUC-Score im Titel. X-Achse: "Number targeted" (absolut, wie sklift).
- **Uplift-by-Percentile** — `_native_uplift_by_percentile()`: 2-Panel-Barplot (Uplift oben, Treatment/Control Response Rates unten), inkl. Fehlerbalken (Std) und Weighted Average Uplift. Unterstützt `strategy='overall'`/`'by_group'` und `kind='bar'`/`'line'`.
- **Treatment-Balance** — `_native_treatment_balance()`: Sliding-Window Treatment-Rate über nach pred. Uplift sortierten Daten.
- **ATE-Barplot** — `generate_ate_barplot()`: Zeigt Response Rates pro Treatment-Gruppe mit SE-Fehlerbalken und n= pro Gruppe. Bei BT: ATE-Pfeil zwischen Control/Treatment. Bei MT: Δ-Labels (Treatment vs. Control) pro Arm. Im Report unter "Datengrundlage" vor Treatment-Verteilung.

Alle Plots verwenden direkt die rubin-Farbpalette (Ruby, Gold, Slate). Kein `recolor_figure()` nötig.

Diese Plots waren ursprünglich in scikit-uplift (sklift) implementiert. sklift 0.5.1 hat einen bekannten numpy >=1.24 Kompatibilitäts-Bug (GitHub Issue #213, seit Mai 2024 offen) und wurde seit 2022 nicht mehr aktualisiert. Die nativen Implementierungen sind funktional identisch und unabhängig von externen Plot-Bibliotheken.

Jeder Plot durchläuft diesen Lebenszyklus:

1. Erzeugung in `generate_sklift_plots()` (Funktionsname beibehalten für API-Kompatibilität)
2. `self._report.add_plot()` konvertiert zu Base64 für den HTML-Report
3. `mlflow.log_figure()` loggt als PNG-Artefakt
4. `plt.close(fig)` gibt den Speicher frei

Wenn ein historischer Score `S` vorliegt, werden die Plots auch für diesen
Score direkt gegen den historischen Score gegenüberstellt.

## Abhängigkeiten

Für die vollständige Plot-Ausgabe werden benötigt:

- `econml` (bereits Kernabhängigkeit)
- `scipy` (für DRTester-Nuisance)

Alle Pakete werden automatisch über `pixi install` installiert (empfohlen). Alternativ sind sie auch in `requirements.txt` enthalten.


## Policy-Value-Vergleich gegen einen Referenzscore (z. B. historischer Score)

Wenn ein historischer Score (`data_files.s_file`) vorliegt, wird dieser in der Analyse zusätzlich
wie ein weiteres „Modell“ evaluiert. Neben Qini/AUUC werden auch Policy Values (inkl.
Konfidenzintervall) über den `DRTester` erzeugt. Die Ergebnisse werden als Vergleichsplots
direkt gegen den historischen Score gestellt:

- `policy_compare__<modell>_vs_<historical_name>.png`

Inhalt des Plots:
- Policy Values des Modells (mit Konfidenzband)
- Policy Values des Referenzscores (mit Konfidenzband)
- Differenzkurve (Modell minus Referenz)

Hinweis: Die Vergleichsplots werden nur dann erzeugt, wenn die DRTester-Auswertung sowohl für
die kausalen Modelle als auch für den historischen Score erfolgreich berechnet wurde.

## Multi-Treatment-Evaluation

Bei Multi-Treatment (T ∈ {0, 1, …, K-1}) wird die Evaluation automatisch angepasst:

**Ansatz A – Pro Treatment-Arm:**
Für jeden Treatment-Arm k wird separat eine Uplift-Kurve berechnet, indem nur die Beobachtungen
mit T ∈ {0, k} betrachtet werden. Daraus ergeben sich pro-Arm-Kennzahlen:
`qini_T1`, `qini_T2`, `auuc_T1`, `auuc_T2`, etc.
Zusätzlich wird ein `policy_value_T{k}` berechnet – das ist das Analogon zu
`policy_value` bei Binary Treatment: Unter allen Beobachtungen mit positivem
geschätztem Effekt (CATE_k > 0) wird die Differenz der Outcome-Raten zwischen Arm k und
Control berechnet.

**Ansatz B – Policy Value (IPW):**
Die optimale Zuweisungspolicy π*(X) = argmax_k τ_k(X) wird über einen IPW-Schätzer bewertet.
Der resultierende `policy_value` misst den erwarteten inkrementellen Nutzen der optimalen
Zuweisungsentscheidung gegenüber der Baseline (Control).

Die Propensity-Gewichte werden standardmäßig als empirische Verteilung (d. h. Anteil pro
Gruppe) geschätzt – korrekt bei Randomisierung. Für observationale Daten kann eine geschätzte
Propensity als Parameter übergeben werden.

**Treatment-Verteilung:**
Zusätzlich wird dokumentiert, welchem Anteil der Population jedes Treatment zugewiesen wird
(`best_treatment_distribution`). Dies hilft bei der fachlichen Einordnung der Ergebnisse.

**DRTester-Plots:**
Bei MT werden DRTester-Plots pro Treatment-Arm erzeugt. Dafür werden die Daten auf
T ∈ {0, k} gefiltert, sodass der DRTester binäre Treatment-Daten sieht.

**Historischer Score-Vergleich:**
Der Vergleich gegen einen historischen Score ist nur bei Binary Treatment verfügbar, da ein
einzelner Score keine Multi-Treatment-Zuweisung abbilden kann.

**Beispiel-Ausgabe (`uplift_eval_summary.json` bei MT mit K=3):**

```json
{
  "NonParamDML": {
    "qini_T1": 0.0312,
    "auuc_T1": 0.0187,
    "uplift10_T1": 0.042,
    "uplift20_T1": 0.035,
    "uplift50_T1": 0.021,
    "policy_value_T1": 0.0089,
    "qini_T2": 0.0098,
    "auuc_T2": 0.0064,
    "uplift10_T2": 0.015,
    "uplift20_T2": 0.011,
    "uplift50_T2": 0.007,
    "policy_value_T2": 0.0043,
    "policy_value": 0.0245,
    "best_treatment_distribution": {
      "T0": 0.35,
      "T1": 0.48,
      "T2": 0.17
    }
  }
}
```

## Train Many, Evaluate Some (TMES)

Bei mehreren Eingabedateien kann über `data_prep.eval_file_index` eine einzelne Datei als Evaluationsgrundlage festgelegt werden. Die DataPrep-Pipeline erzeugt dafür eine Boolean-Maske (`eval_mask.npy`), die in der Analyse-Pipeline über `data_files.eval_mask_file` geladen wird.

**Leakage-freies Train/Eval-Routing:** Die Pipeline spaltet vor dem Training:
- **Trainingsdaten** = Zeilen mit `mask == False` (alle nicht-markierten Dateien)
- **Holdout-Eval-Daten** = Zeilen mit `mask == True` (die ausgewählte Eval-Datei)

Effektiv läuft TMES damit wie der External-Eval-Pfad (siehe unten), nur dass die Eval-Daten implizit aus der Maske abgeleitet werden statt aus separaten Dateien. Alle Modelle, Bundle-Refits, SHAP-Analysen und DRTester-Nuisance-Fits sehen ausschließlich die Trainingszeilen. Uplift-Metriken werden out-of-sample auf dem Holdout berechnet.

**MLflow-Log-Parameter:** `tmes_total_n` und `tmes_eval_n` werden beim Run geloggt, damit die Split-Größen nachvollziehbar sind.

**Die Maske überlebt** Subsampling (`df_frac`) und wird positionskonsistent angewendet. Leere oder all-True-Masken werden mit Warnung ignoriert.

**Unterschied zu External Eval:**
- External Eval: separate `eval_x/t/y_file` Parquet-Dateien werden explizit übergeben
- TMES: Eval-Daten kommen aus Mask-Indizierung derselben Trainingsdatei-Menge

Beide Pfade laufen intern identisch ab (`holdout_data`-Routing, `keep_fold=False` für SHAP, kein äußeres CV).

## Treatment-Balance bei mehreren Dateien

Wenn Trainingsdaten aus mehreren Dateien zusammengeführt werden, kann die Treatment-Rate pro Datei unterschiedlich sein. Eine Differenz von mehr als 5 Prozentpunkten wird als Warnung geloggt, da sie Uplift-Metriken verzerren kann — bestimmte Cross-Validation-Folds enthalten dann systematisch mehr oder weniger Treatment-Beobachtungen.

Mit `data_prep.balance_treatments: true` wird die überrepräsentierte Gruppe pro Datei per Random-Downsampling auf die niedrigste Treatment-Rate angeglichen. Es werden nur so viele Zeilen entfernt wie nötig.

## NaN/Inf-Behandlung und DRTester-Resilienz

Alle DR-Outcomes und CATE-Predictions werden vor der OLS-Regression (in EconML's `evaluate_blp()`) auf NaN/Inf geprüft und per Percentil-Clipping sanitisiert. Historische Scores (`S`) werden beim Laden auf NaN/Inf geprüft und durch 0 ersetzt. Die `_sanitize_dr()`-Methode erzwingt `float64`-Kopie, ersetzt NaN/Inf durch 0, clippt auf das 0.5/99.5-Perzentil und validiert abschließend nochmals.

**Sanitisierungs-Kette:** CATE-Predictions werden am Eingang von `evaluate_cate_with_plots()` per `nan_to_num` gesäubert, dann in `evaluate_all()` per `_sanitize_dr()` clippt, und als letzte Verteidigungslinie nochmals in der überschriebenen `evaluate_blp()` vor dem OLS-Aufruf.

### DRTester-Nuisance: Einmal fitten, für alle Modelle wiederverwenden

Der DRTester benötigt Nuisance-Modelle (Outcome + Propensity), um Doubly-Robust-Outcomes zu berechnen. Diese sind für alle kausalen Modelle **identisch** — nur die CATE-Predictions unterscheiden sich. Die Pipeline nutzt daher ein Pre-Fit-Pattern:

1. `fit_drtester_nuisance()` wird **einmal** aufgerufen (BT) bzw. einmal pro Treatment-Arm (MT)
2. Der gefittete Tester wird in `fitted_tester_bt` / `fitted_tester_mt[arm]` gespeichert
3. `evaluate_cate_with_plots(fitted_tester=...)` kopiert DR-Outcomes (`dr_val_`, `dr_train_`, `ate_val`, `Dval`) und tauscht nur die CATE-Predictions aus

Speedup: Bei 8 Modellen spart das 7× das teure Nuisance-CV-Fitting (~30-120s je nach Datengröße). Die Nuisance-Modelle nutzen leichtere Varianten: `n_estimators≤100` (statt potentiell 400+) und `cv=5`, was nochmals schneller ist bei minimaler AUC-Einbuße (~0.5-1%).

### Datenfluss: Gespeicherte Predictions → Evaluation

Die Evaluation nutzt **ausschließlich bereits berechnete Predictions** — es werden keine Modelle erneut gefittet oder Predictions erneut berechnet. Der Datenfluss:

```
Step 4: Training
│
│  preds["SLearner"]  = DataFrame(Y, T, Predictions_SLearner, Train_SLearner)
│  preds["TLearner"]  = DataFrame(Y, T, Predictions_TLearner, Train_TLearner)
│  preds["DRLearner"] = DataFrame(Y, T, Predictions_DRLearner, Train_DRLearner)
│  ...
│
▼
Step 5: Evaluation & Metriken
│
├── Phase 1: Schnelle Metriken ────────── preds[model]["Predictions_*"].to_numpy()
│   (Qini, AUUC, Uplift@k, Policy Value)   → reines NumPy, <1s pro Modell
│
├── Nuisance Pre-Fit (EINMAL) ─────────── fit_drtester_nuisance(X_val, T_val, Y_val,
│   fitted_tester_bt                        X_train, T_train, Y_train)
│                                           → gespeichert, für alle Modelle wiederverwendet
│
├── Phase 2: DRTester-Plots ───────────── evaluate_cate_with_plots(
│   (Level-abhängig)                        fitted_tester=fitted_tester_bt,
│                                           cate_preds_val=preds[model]["Predictions_*"],
│                                           cate_preds_train=preds[model]["Train_*"],
│                                           ...)
│                                           → BLP, Cal, Qini, TOC + Policy Values
│
├── Phase 3: Uplift-Plots ─────────────── generate_sklift_plots(
│   (alle Modelle)                          preds[model]["Predictions_*"],
│                                           T, Y)
│                                           → Qini-Kurve, Uplift-by-Percentile, Balance
│
├── Historischer Score ────────────────── S-Datei (einmal geladen)
│                                         → evaluate_cate_with_plots(cate_preds_val=S)
│                                         → policy_value_comparison_plots()
│                                         → plot_custom_qini_curve()
│
▼
Step 6: Surrogate-Tree
│
│  surrogate_df["Predictions_SurrogateTree_*"] = tree.predict(X)
│  surrogate_df["Train_SurrogateTree_*"]       = tree.predict(X)
│
├── Surrogate-Metriken ────────────────── uplift_curve(score=surrogate_df["Predictions_*"])
├── Surrogate-DRTester ────────────────── evaluate_cate_with_plots(fitted_tester=...,
│                                           cate_preds_val=surrogate_df["Predictions_*"])
├── Surrogate-Uplift-Plots ──────────────── generate_sklift_plots(surrogate_preds, T, Y)
```

Wichtige Garantie: Weder `_evaluate_bt`, `_evaluate_mt`, `_evaluate_historical_score` noch die Surrogate-Evaluation rufen `.fit()` oder `.predict()` auf kausalen Modellen auf. Sie nutzen ausschließlich die in Step 4 bzw. Step 6 gespeicherten Spalten.

### Train-Daten und `has_train`-Prüfung

DRTester benötigt Train-Daten (`Xtrain`, `Dtrain`, `ytrain`) für Calibration, Qini-CI und TOC-CI. Die Pipeline prüft, ob in irgendeinem Modell `Train_*`-Spalten mit nicht-NaN-Werten vorhanden sind:

```python
has_train = any(
    any(c.startswith("Train_") and not np.all(np.isnan(dfp[c].to_numpy(dtype=float)))
        for c in dfp.columns if c.startswith("Train_"))
    for dfp in preds.values()
)
```

Wenn `has_train=True`: `X_train=X, T_train=T, Y_train=Y` → `fit_on_train=True` → alle DRTester-Tests laufen.
Wenn `has_train=False`: `X_train=None` → Nuisance wird nur auf Val gefittet. Für die Calibration-Quantil-Cuts werden die Val-Predictions als Fallback verwendet (weniger rigoros, aber die Calibration-Plots bleiben informativ).

Im Standardfall (Cross-Validation) erzeugt `train_and_crosspredict` sowohl Out-of-Fold-Predictions (`Predictions_*`) als auch In-Sample-Predictions (`Train_*`). Damit ist `has_train=True` und alle Plots werden erzeugt.

| Szenario | `has_train` | Nuisance X_train | DRTester-Tests |
|---|---|---|---|
| **Cross-Validation** | `True` (Train_* vorhanden) | X (voller Datensatz) | Alle (BLP, Cal, Qini, TOC) |
| **External Eval** | `True` (Train_* vorhanden) | X (Train-Datensatz) | Alle |
| **Train Many Eval One** | `True` (Train_* vorhanden) | X (voller Datensatz) | Alle |

### evaluate_cate_with_plots — Architektur

Die Funktion hat **einen** try/except um den gesamten DRTester-Block. Innerhalb werden alle Aufrufe **direkt** durchgeführt — ohne individuelle try/except-Blöcke:

```python
# evaluate_cate_with_plots:
try:
    res = tester.evaluate_all(X_val, X_train, ...)
    summary = res.summary()
    cal_plot = res.plot_cal(1).get_figure()
    recolor_figure(cal_plot)                    # ← rubin-Farbschema
    qini_plot = res.plot_qini(1).get_figure()
    recolor_figure(qini_plot)
    toc_plot = res.plot_toc(1).get_figure()
    recolor_figure(toc_plot)
    policy_values = res.get_policy_values(1)
except Exception:
    # Gesamter DRTester-Block fehlgeschlagen → leere Defaults
    summary = pd.DataFrame()
    policy_values = pd.DataFrame()
```

Dieses Pattern ist **identisch mit causaluka** und funktioniert zuverlässig. Wenn `evaluate_all` erfolgreich läuft, laufen auch `summary()`, `plot_cal()`, `plot_qini()`, `plot_toc()` und `get_policy_values()` — weil die Sub-Results (blp, cal, qini, toc) alle vorhanden sind.

### `summary()`-Override (Sicherheitsnetz)

Falls einzelne Sub-Results `None` sind (was im Normalbetrieb nicht vorkommt, aber bei sehr extremen Daten passieren könnte), überschreibt `CustomEvaluationResults.summary()` die EconML-Methode und baut eine partielle Tabelle:

- BLP vorhanden → BLP-Koeffizienten + p-Werte
- Calibration vorhanden → CAL-Koeffizienten + p-Werte
- Qini/TOC → haben keine tabellarische Summary, nur Plots

## HTML-Report: Plot-Vergrößerung

Alle Diagnose-Plots im HTML-Report sind klickbar. Ein Klick öffnet eine Lightbox-Vergrößerung mit dem Plot-Titel. Schließen per Klick auf den Hintergrund, ×-Button oder Escape-Taste.
