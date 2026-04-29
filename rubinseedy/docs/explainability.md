# Explainability

Explainability ist in die **Analyse-Pipeline integriert**: Bei `shap_values.calculate_shap_values: true` werden SHAP-Werte und Importance-Plots automatisch für den Champion berechnet und als MLflow-Artefakte im selben Run geloggt. Zweistufiger Fallback: ① EconML SHAP-Plot-Satz → ② generischer SHAP-Plot-Satz (u.a. CausalForestDML). Die Plots werden außerdem in den HTML-Report eingebettet.

Zusätzlich kann Explainability **als separater CLI-Runner** auf Bundle-Basis ausgeführt werden – z. B. für nachträgliche Ad-hoc-Analysen auf neuen Daten oder mit einem anderen Modell als dem Champion.

## Integrierte Explainability (Analyse-Pipeline)

### Ensemble als Champion

Wenn das Ensemble Champion wird, kann die SHAP-Analyse nicht direkt auf dem `EnsembleCateEstimator` durchgeführt werden (kein natives SHAP, KernelSHAP wäre langsam und verrauscht). Stattdessen wird automatisch das **beste Einzelmodell** im Ensemble für die Explainability herangezogen — das Modell mit dem höchsten Score auf der Selektionsmetrik. Der HTML-Report zeigt an, welches Modell verwendet wurde.

In der `config.yml`:

```yaml
shap_values:
  calculate_shap_values: true
  n_shap_values: 10000       # Max. Stichprobe
  top_n_features: 20         # Anzahl Features in Plots
  num_bins: 10               # Binning für PDP-Plots

```

**`method`** (optional, default `"shap"`):

Die Explainability wird nach dem Bundle-Export und vor dem HTML-Report ausgeführt. Artefakte in MLflow:
- SHAP-Plots (Summary mit Beeswarm/Mean/Max Impact, CATE-Profile, Dependence, Scatter) als PNG
- Importance-CSV

**Out-of-Sample:** SHAP-Werte werden immer auf Daten berechnet, die das Modell nie gesehen hat:
- **CV-Modus:** Nutzt das letzte CV-Fold-Modell (bereits auf K-1 Folds trainiert, kein zusätzliches Training nötig) und sampelt aus dessen Out-of-Fold-Indices.
- **External-Modus:** Nutzt das auf Trainingsdaten gefittete Modell und die Eval-Daten.
- **Fallback:** Stratifizierter 80/20-Split, Fit auf 80%, Explain auf 20%.

## Separater CLI-Runner (`run_explain.py`)

```bash
pixi run explain -- --bundle runs/bundles/<bundle_id> --x new_X.parquet --out-dir explain
# oder: python run_explain.py --bundle runs/bundles/<bundle_id> --x new_X.parquet --out-dir explain
```

Optional kann ein bestimmtes Modell gewählt werden:

```bash
pixi run explain -- --bundle runs/bundles/<bundle_id> --x new_X.parquet --model-name XLearner
# oder: python run_explain.py --bundle runs/bundles/<bundle_id> --x new_X.parquet --model-name XLearner
```

## Erzeugte Artefakte

### SHAP
Wenn das Modell eine EconML-kompatible `shap_values`-Schnittstelle bereitstellt, werden die SHAP-Plots erzeugt:

- `SHAP_summary_<MODEL>.png` — Beeswarm + Mean Impact + Max Impact (3 Subplots)
- `SHAP_cate_profiles_<MODEL>.png` — CATE-Profile: durchschnittlicher CATE pro Feature-Wertebereich
- `SHAP_dependence_<MODEL>.png` — SHAP Dependence: gebinnter mittlerer SHAP-Wert pro Wertebereich
- `SHAP_scatter_<MODEL>.png` — SHAP Scatter: SHAP vs. Feature-Wert mit Interaktions-Farbcodierung
- `shap_importance_<MODEL>.csv` — Feature-Importance (mean |SHAP|)

Falls das nicht möglich ist, fällt der Runner auf modellagnostische SHAP-Werte zurück und schreibt zusätzlich `shap_values_<MODEL>.csv`.


## Konfiguration
Wenn im Bundle eine `config_snapshot.yml` vorhanden ist, übernimmt `run_explain.py` daraus Voreinstellungen:

- `shap_values.n_shap_values`
- `shap_values.top_n_features`
- `shap_values.num_bins`

CLI-Parameter überschreiben diese Voreinstellungen.

## Multi-Treatment-Besonderheiten

Bei Multi-Treatment-Modellen erzeugt `_predict_effect()` ein 2D-Array mit K-1 Effektschätzungen
(eine pro Treatment-Arm vs. Control). Für die Explainability wird daraus ein skalarer Wert
abgeleitet, damit SHAP auf einer einzigen Zielgröße arbeitet:

- **SHAP:** Verwendet `max(τ_1(X), …, τ_{K-1}(X))` als skalaren Output. Das zeigt, welche
  Features den maximalen erwarteten Treatment-Effekt beeinflussen – unabhängig davon, welcher
  Arm der beste ist.
  (`||Δτ(X)||₂`), um den Gesamteinfluss eines Features auf die Effektschätzung zu messen.
