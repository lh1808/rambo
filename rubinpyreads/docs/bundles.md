# Bundles

Der Bundle-Export wird über den Block `bundle` in der Konfiguration gesteuert.

```yaml
bundle:
  enabled: true
  base_dir: "runs/bundles"
  bundle_id: null
  log_to_mlflow: true
```

`run_analysis.py` kann diese Werte bei Bedarf per CLI überschreiben.

## Was ist ein Bundle?
Ein Bundle ist ein Ordner, der **alle** Artefakte enthält, die für ein reproduzierbares Scoring benötigt werden:

- Konfigurations-Snapshot
- Preprocessing-Artefakte (Feature-Reihenfolge, Dtypes, ggf. Encoder)
- trainierte Modelle (Pickle-Dateien)

## Warum synchroner Export?
Der Export passiert **im gleichen Prozess** wie die Analyse. Dadurch gilt:
- das Bundle passt garantiert zur Analyse-Ausführung,
- keine Race Conditions,

## Typischer Workflow

1) Analyse laufen lassen und Bundle exportieren:
```bash
pixi run analyze -- --config config.yml --export-bundle
# oder: python run_analysis.py --config config.yml --export-bundle
# Ziel-Verzeichnis = bundle.base_dir aus der Config (Default: runs/bundles);
# per --bundle-dir <pfad> überschreibbar.
```

```bash
pixi run score -- --bundle runs/bundles/<bundle_id> --x new_X.parquet --out scores.csv
# Schlankes Scoring-Environment (ohne mlflow/optuna/matplotlib/shap);
# teilt die solve-group mit default → identische econml/lightgbm/…-Versionen.
# Bundles stempeln die ML-Stack-Versionen in metadata.json (ml_package_versions);
# ProductionPipeline warnt beim Laden, falls die Laufzeit abweicht.
pixi run -e prod score -- --bundle runs/bundles/<bundle_id> --x new_X.parquet --out scores.csv
# oder: python run_production.py --bundle runs/bundles/<bundle_id> --x new_X.parquet --out scores.csv
# Standard: Champion aus model_registry.json
# Optional: --model-name XLearner oder --use-all-models
# Surrogate: --use-surrogate (interpretierbarer Einzelbaum)
```


## Zusätzliche Bundle-Inhalte (Schema & Metadaten)

### schema.json
Neben dem `preprocessor.pkl` enthält ein Bundle optional eine Datei `schema.json`.
Sie beschreibt erwartete Spalten und Datentypen des Feature-Matrix-Inputs.

**Nutzen**
- Production kann sofort melden, wenn sich das Input-Layout geändert hat.
- Typische Fehler (fehlende Spalten, falsche Typen) werden früh erkannt.

### metadata.json (automatisch angereichert)
Beim Schreiben der Metadaten werden automatisch ergänzt:
- `created_at_utc`: Erstellungszeitpunkt (UTC)
- `refit_info`: Ob der Champion auf vollen Daten refittet wurde
- `feature_columns`: Liste der verwendeten Features
- `n_train_samples`: Anzahl Trainingszeilen
- `champion_name`: Name des Champion-Modells
- `base_learner`: Konfigurierter Base-Learner-Typ. Bei `type: both` zusätzlich
  `chosen_per_role` (der von Optuna pro Modell und Rolle gewählte Learner,
  z. B. `NonParamDML.model_final: catboost`) und
  `fallback_without_tuned_params: catboost` (Rollen ohne getunte Parameter
  fallen zur Laufzeit auf CatBoost zurück). Damit ist aus dem Bundle
  rekonstruierbar, welcher Learner im "both"-Modus gewonnen hat.

Damit ist die Herkunft des Bundles schnell nachvollziehbar.


## Registry-Manifest und Promotion

### model_registry.json

Beim Bundle-Export schreibt rubin zusätzlich ein Registry-Manifest `model_registry.json`.
Dieses enthält:

- `models`: Liste aller Modelle im Bundle (Name, Artefaktpfad, Metriken)
- `champion`: Name des Standardmodells für Produktion
- `selection`: Regel, mit der der Champion initial gewählt wurde (z. B. Metrik)

### Champion wechseln (Promotion)

Wenn nach fachlichem Review ein anderes Modell produktiv gehen soll, kann der Champion
per CLI umgestellt werden:

```bash
pixi run promote -- --bundle runs/bundles/<bundle_id> --model <ModelName>
# oder: python run_promote.py --bundle runs/bundles/<bundle_id> --model <ModelName>
```

Alternativ kann in Production über `--model-name` ein anderes Modell erzwungen werden.

**Warum das bewusst so umgesetzt ist:**  
Produktionsentscheidungen sollen getrennt von Trainingsläufen getroffen werden können.
Das Manifest ist dabei der „Vertrag“ zwischen Analyse und Produktion.

### Trainingsstand der exportierten Modelle

Beim Bundle-Export werden alle Modelle so gespeichert, dass sie direkt in Production
einsetzbar sind:

- **Alle Modelle:** Beim Export werden **alle** trainierten Modelle auf allen im Run
  verfügbaren Daten refittet — nicht nur der Champion. Damit ist in Production jedes Modell
  direkt einsatzbereit und frei wählbar (die Modell-Pickles werden lazy geladen — erst beim tatsächlichen Zugriff, ungenutzte Challenger kosten keinen Speicher). Im Cross-Validation-Modus werden Modelle initial nur
  auf K-1 Folds trainiert — der Refit stellt sicher, dass alle Daten einfließen.
  Im Holdout-Modus unterbleibt der Voll-Refit (die Modelle bleiben auf Train gefittet), um die
  Evaluation nicht zu entwerten.
- **Ensemble:** Wird stets self-contained aus den refitteten Mitgliedern neu erstellt
  (`ShapeSafeEnsembleCate`), unabhängig davon, ob es Champion ist. Da die Mitglieder in
  die Ensemble-Pickle serialisiert, funktioniert das Ensemble in Production mit garantiert
  gefitteten Mitgliedern — auch ohne dass die Einzelmodelle als separate Datei vorliegen.
  **Mitgliedschafts-Invariante:** Der Rebuild liest die Zusammensetzung aus
  `member_names` (Pflichtattribut, wird bei der Ensemble-Konstruktion persistiert) —
  das exportierte Ensemble enthält damit exakt die Mitglieder des evaluierten Ensembles,
  dessen Metriken im Report stehen. Mitglied wird nur, wer vollständige
  Cross-Predictions liefert (BT: `Predictions_*`; MT: alle Arm-Spalten); bei weniger
  als 2 vollständigen Mitgliedern wird das Ensemble übersprungen.
- **Surrogate-Einzelbaum**: Bei `surrogate_tree.enabled: true` wird ein `SurrogateTree.pkl` ins Bundle exportiert — ein interpretierbarer Einzelbaum, der die CATEs des Champions nachbildet. Der Surrogate wird mit einem eigenen Registry-Eintrag in `model_registry.json` aufgenommen und ist über `score_surrogate(X)` oder `--use-surrogate` in Production nutzbar. Die Production-Pipeline prüft über `has_surrogate`, ob ein Surrogate verfügbar ist.
  Bei Multi-Treatment wird pro Arm ein eigener Baum trainiert.

### eval_mask.npy (optional)

Wenn bei der Datenaufbereitung „Train Many, Evaluate Some" aktiviert wurde (`data_prep.eval_file_index`), wird eine Boolean-Maske `eval_mask.npy` im Output-Verzeichnis gespeichert. Diese Maske markiert die Zeilen, auf denen die Evaluation durchgeführt wird. In der Analyse-Pipeline wird die Maske über `data_files.eval_mask_file` geladen — Training auf allen Daten, Evaluation (Metriken) nur auf den markierten Zeilen.

## Hinweis zu Parquet und Datentypen

Die DataPrepPipeline schreibt die standardisierten Eingabedaten als Parquet. Parquet speichert Datentypen bereits mit.
In der Praxis kommen Produktionsdaten jedoch oft aus anderen Exportwegen (und damit mit abweichenden Typen). `dtypes.json`
dient hier als robuste Referenz, um Typen konsistent auszurichten.


## Multi-Treatment-Scoring in Production

Bei Multi-Treatment (T ∈ {0, 1, …, K-1}) unterscheidet sich das Score-Ergebnis der
ProductionPipeline vom Binary-Treatment-Fall:

### Output-Spalten (Binary Treatment)

| Spalte | Beschreibung |
|---|---|
| `cate_{ModelName}` | Geschätzter CATE (ein Wert pro Beobachtung) |

### Output-Spalten (Multi-Treatment)

| Spalte | Beschreibung |
|---|---|
| `cate_{ModelName}_T1`, …, `cate_{ModelName}_T{K-1}` | Geschätzter CATE pro Treatment-Arm vs. Control |
| `optimal_treatment_{ModelName}` | Optimales Treatment (0 = Control, 1..K-1 = Treatment-Arm). Wird nur zugewiesen, wenn der beste Effekt > 0 ist. |
| `treatment_confidence_{ModelName}` | Differenz zwischen bestem und zweitbestem Treatment-Effekt. Höhere Werte = klarere Empfehlung. |

### Beispiel-Aufruf

```bash
pixi run score -- --bundle runs/bundles/<bundle_id> --x new_X.parquet --out scores.csv
# oder: python run_production.py --bundle runs/bundles/<bundle_id> --x new_X.parquet --out scores.csv
```

Die Production-Pipeline erkennt automatisch anhand der Modell-Ausgabe, ob es sich um
ein BT- oder MT-Modell handelt. Es ist keine separate Konfiguration nötig.