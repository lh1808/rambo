# da_pluto_timeseries

![Buildstatus](https://tfs.lan.huk-coburg.de/web/DefaultCollection/GIT_Projects/_apis/build/status/da-pluto-timeseries?branchName=master)

Multivariate Wochen-Prognose für PLUTO-Termineingänge. Trainiert ein
Temporal-Fusion-Transformer-Modell (via [darts][darts]) je Horizont,
evaluiert per rollendem Backtest und schreibt die Prognose zurück in
DB2.

[darts]: https://unit8co.github.io/darts/

Architektur- und Implementierungsnotizen: siehe `DESIGN.md`.

## Installation

```bash
git clone ssh://tfs.lan.huk-coburg.de/web/DefaultCollection/GIT_Projects/_git/da-pluto-timeseries
cd da-pluto-timeseries

# Conda-Umgebung (aus environment.yml) anlegen
mamba env create
conda activate da-pluto-timeseries

pre-commit install
pip install --no-build-isolation -e .

# Runtime-Dependencies, die noch nicht in environment.yml stehen:
pip install "u8darts[torch]" icalendar optuna pyyaml
```

# PLUTO – Multivariate Terminprognose (13 & 52 Wochen)

Dieses Projekt implementiert eine multivariate Forecasting-Pipeline für den Termineingang
in PLUTO. Es werden wöchentliche Prognosen für 52 Wochen erzeugt, basierend auf zwei
Modellen:

- **13-Wochen-Modell** (kurzfristiger Horizont)
- **52-Wochen-Modell** (mittelfristiger Horizont)

Die kombinierte Prognose wird wie folgt verwendet:

- Wochen **1–13**: Prognose aus dem **13-Wochen-Modell**
- Wochen **14–52**: Prognose aus dem **52-Wochen-Modell**

Die Prognose wird anschließend zurück in eine DB2-Prognosetabelle geschrieben.

---

## Architekturüberblick

Die Codebasis ist in zwei Bereiche aufgeteilt:

1. **Forecasting-Logik (generisch)**
   Ordner: `forecasting/`
   Enthält alles, was zur Modellierung gehört:
   - Vorverarbeitung und Feature-Engineering (inkl. tägliche → wöchentliche Aggregation)
   - Holiday-/Ferien-Kovariaten aus ICS-Dateien (Updates der sich darin befindlichen ICS-Dateien notwendig https://github.com/paulbrejla/ferien-api-data)
   - Modell-Definition (TFT aus Darts)
   - Optionales Hyperparameter Tuning mittels Optuna
   - Rolling-Block-Evaluation (Backtest, leak-frei via `transforms_builder`)
   - Training und Inferenz für verschiedene Horizonte (13 und 52 Wochen)

2. **PLUTO-spezifische Anbindung**
   - `pluto_multivariate_repository.py`
     - DB2-Connector (lesen/schreiben)
     - Abbildung der Fachdimensionen (Kennzahl, Produkt, Schadenstatus)
   - `pluto_forecast_job.py`
     - Produktiver Run-Job:
       - lesen aus DB2
       - Forecasting-Pipeline ausführen
       - 13-/52-Wochen-Prognosen kombinieren
       - zurück in DB2 schreiben

---

## Verwendete Kennzahlen / Dimensionen

**Kennzahlen (DIM_KENNZAHL):**

- `TERM_EINGANG_SCHRIFTST`
- `TERM_EINGANG_SONST`

**Produkte (DIM_PRODUKT):**

- `KFZ_Vollkasko`
- `KFZ_Teilkasko`
- `KFZ_Haftpflicht`
- `KFZ_Rest`
- `HUS_Haftpflicht`
- `HUS_Wohngebäude`
- `HUS_Hausrat`
- `HUS_Rest`

**Schadenstatus (DIM_SCHADENSTATUS):**

- `Neuschaden`
- `Folgebearbeitung`

Im Code werden diese Dimensionen in einem Komponenten-Namen kombiniert, z. B.:

```text
TERM_EINGANG_SCHRIFTST__KFZ_Vollkasko__Neuschaden
TERM_EINGANG_SONST__HUS_Hausrat__Folgebearbeitung
```

---

## Konfiguration

Die kanonische Form der Konfiguration sind die `@dataclass`-Strukturen in
`forecasting/config.py`. Für produktive Läufe kann man diese über zwei
Wege anpassen, ohne Python-Code zu ändern:

**1. YAML-Datei.** Die YAML überschreibt nur, was sie angibt – alle
übrigen Felder bleiben auf den Defaults. Vorlage: `config.example.yaml`.

**2. Env-Variablen mit Präfix `PLUTO__`.** Doppel-Unterstrich als
Pfadtrenner, punktuelle Overrides einzelner Werte. Env schlägt YAML
schlägt Defaults.

```bash
export PLUTO__TFT__N_EPOCHS=50
export PLUTO__TUNING__13__ENABLED=true
```

Tippfehler in Feldnamen (YAML oder Env) werfen sofort einen `ValueError`
mit einer Liste der erlaubten Felder.

### Config-Profile und Shell-Steuerung

Das Repo enthält vier vorgefertigte Configs:

| Config | Zweck | Dauer (A100) |
|---|---|---|
| `config.smoke.yaml` | Schneller Funktionstest der Pipeline | ~2–5 min |
| `config.production.yaml` | Wöchentlicher Regelbetrieb | ~20–40 min |
| `config.tuning.yaml` | Optuna-Hyperparametersuche | ~2–4 h |
| `config.tuned.yaml` | Produktion mit eingefrorenen Tuning-Werten | ~20–40 min |

**Über das Shell-Skript** wird das Profil per `PLUTO_PROFILE` gesetzt:

```bash
# Schnelltest (verifiziert DB2, ICS, Pipeline)
PLUTO_PROFILE=smoke   ./schaden_forecast_job.sh

# Hyperparametersuche (h13: 15 Trials, h52: 10 Trials)
PLUTO_PROFILE=tuning  ./schaden_forecast_job.sh

# Produktion (Default, wenn PLUTO_PROFILE nicht gesetzt)
./schaden_forecast_job.sh

# Produktion mit eingefrorenen Parametern (ohne runs/-Abhängigkeit)
PLUTO_PROFILE=tuned   ./schaden_forecast_job.sh
```

Das Shell-Skript löst `PLUTO_PROFILE` auf `config.{PROFILE}.yaml` auf.
Alternativ kann ein beliebiger Pfad direkt übergeben werden:

```bash
PLUTO_CONFIG=/pfad/zu/meine_config.yaml ./schaden_forecast_job.sh
```

`PLUTO_CONFIG` hat immer Vorrang vor `PLUTO_PROFILE`.

---

## Tuning-Workflow und Parameter-Persistierung

### Ablauf

```text
1. Smoke-Test     →  PLUTO_PROFILE=smoke   ./schaden_forecast_job.sh
2. Tuning         →  PLUTO_PROFILE=tuning  ./schaden_forecast_job.sh
3. Produktion     →  ./schaden_forecast_job.sh  (ab jetzt wöchentlich)
```

### Was beim Tuning passiert

`config.tuning.yaml` aktiviert Optuna für beide Horizonte. Pro Trial
wird ein Rolling-Backtest mit reduzierten Epochen (50, EarlyStopping
patience=5) durchgeführt. Optuna variiert dabei gleichzeitig:

- `train_length_weeks` (wie viel Historie das Modell sieht)
- `hidden_size`, `hidden_continuous_size`, `lstm_layers` (Architektur)
- `dropout`, `learning_rate`, `batch_size` (Regularisierung/Optimierung)

Nach Abschluss werden die besten Parameter automatisch gespeichert:

```text
$PLUTO_RUNS_DIR/
├── tuned_h13_model.yaml   ← beste Parameter für 13-Wochen-Horizont
└── tuned_h52_model.yaml   ← beste Parameter für 52-Wochen-Horizont
```

### Getunete Parameter verwenden

**Option A — Automatisch (empfohlen):** `config.production.yaml` lässt
`tuned_params_dir` offen. Der Job setzt es automatisch auf den Wert von
`PLUTO_RUNS_DIR`. Beim nächsten Produktionslauf erkennt die Pipeline die
`tuned_*.yaml`-Dateien und lädt die Parameter. Im Log erscheint:

```text
INFO: Getunete Parameter geladen: {'source': 'runs', 'train_length_weeks': 195, ...}
```

Wenn keine `tuned_*.yaml` existiert (z. B. beim allerersten Lauf ohne
vorheriges Tuning), werden die Defaults aus der Config verwendet.

**Option B — Manuell einfrieren:** Die Werte aus
`$PLUTO_RUNS_DIR/tuned_h13_model.yaml` in `config.tuned.yaml` übertragen
(Platzhalter sind markiert mit `← aus Tuning übernehmen`). Diese Config
ist unabhängig vom `runs/`-Verzeichnis, Git-versionierbar und auf andere
Server portierbar.

**Option C — Env-Override:** Einzelne Werte können jederzeit per
Env-Variable überschrieben werden, auch wenn Tuning-Ergebnisse geladen
werden:

```bash
export PLUTO__TFT__DROPOUT=0.4
./schaden_forecast_job.sh
```

### Wann neu tunen?

- Alle 3–6 Monate im Regelbetrieb
- Wenn die retrospektive Evaluation (siehe Reporting) eine
  Verschlechterung zeigt
- Nach strukturellen Datenänderungen (neue Produkte, geändertes
  Meldeverhalten)

---

## Reporting und Monitoring

Jeder Lauf erzeugt Artefakte im **Runs-Verzeichnis**. Das Reporting
arbeitet auf drei Ebenen.

### Runs-Verzeichnis (persistenter Speicherort)

Das Shell-Skript klont das Repository bei jedem Lauf frisch (`rm -rf` +
`git clone`). Deshalb liegt das Runs-Verzeichnis **außerhalb** des
Repo-Klons — direkt unter `/mnt/`:

```text
/mnt/                                  ← Domino-Projekt-Filesystem
├── runs/                              ← PERSISTENT (wächst über Läufe)
│   ├── metrics_history.csv
│   ├── retrospective_accuracy.csv
│   ├── tuned_h13_model.yaml
│   ├── 2025-04-21T08-30_h13_model/
│   └── …
├── da-pluto-timeseries/               ← wird bei jedem Lauf gelöscht + neu geklont
│   ├── forecasting/
│   └── …
└── ferien-api-data-main/              ← Ferien-ICS-Dateien
```

**Domino Scheduled Runs:** Alle Dateien, die während eines Scheduled
Runs unter `/mnt/` geschrieben werden, synchronisiert Domino nach
Abschluss des Jobs automatisch zurück ins Projekt. Ein explizites Sync
oder Commit ist nicht nötig — das ist ein natives Domino-Feature.

Das bedeutet: `/mnt/runs/` überlebt den Shutdown des Workspaces und
steht beim nächsten Scheduled Run wieder zur Verfügung. Die Metriken-
Historie, die retrospektive Evaluation und die getuneten Parameter
wachsen so über Wochen und Monate an.

Der Pfad ist über `PLUTO_RUNS_DIR` konfigurierbar:

```bash
# Default (in schaden_forecast_job.sh gesetzt)
export PLUTO_RUNS_DIR=/mnt/runs

# Lokale Entwicklung (wenn PLUTO_RUNS_DIR nicht gesetzt)
# → fällt auf ./runs/ zurück
```

### 1. Per-Run-Archivierung (Backtest-Diagnostik)

Jeder Horizont bekommt einen eigenen Ordner:

```text
$PLUTO_RUNS_DIR/                              (z. B. /mnt/runs/)
├── metrics_history.csv              ← pro Lauf angehängt
├── retrospective_accuracy.csv       ← wächst mit jedem Lauf
├── retro_accuracy_by_lead_global.png
├── retro_accuracy_by_kennzahl.png
├── retro_accuracy_by_produkt.png
├── retro_accuracy_by_sparte.png
├── retro_heatmap_mae.png
├── tuned_h13_model.yaml             ← letzte Tuning-Ergebnisse
├── tuned_h52_model.yaml
├── 2025-04-21T08-30_h13_model/
│   ├── run_metadata.json
│   ├── config.yaml
│   ├── metrics.csv
│   ├── backtest_forecast.csv
│   ├── future_forecast.csv
│   ├── forecast_vs_actual_global.png
│   ├── forecast_vs_actual_by_kennzahl.png
│   ├── forecast_vs_actual_by_produkt.png
│   └── components_heatmap.png
└── 2025-04-28T08-30_h13_model/
    └── …
```

### 2. Metriken-Historie (Modellstabilität über Zeit)

Die Datei `metrics_history.csv` im Runs-Verzeichnis wird bei jedem Lauf um eine Zeile
je Komponente und Horizont erweitert (Append). Damit lässt sich die
Modellqualität über Wochen und Monate beobachten:

```python
import pandas as pd
hist = pd.read_csv("/mnt/runs/metrics_history.csv")
hist.groupby(["run_timestamp", "kennzahl"])["smape"].mean().unstack().plot()
```

### 3. Retrospektive Forecast-Evaluation (Vorlauf-Analyse)

Bei jedem Lauf werden alle archivierten Zukunfts-Prognosen früherer
Läufe gegen die inzwischen eingetroffenen Ist-Werte abgeglichen. Die
Kernfrage: „Wie gut war unsere Prognose vor 1/2/…/52 Wochen?"

Die retrospektive Auswertung liegt direkt im Runs-Verzeichnis:

- `retrospective_accuracy.csv` — Rohdaten (predicted, actual, error,
  abs_error, pct_error, sMAPE, lead_weeks, Dimensions-Spalten)
- `retro_accuracy_by_lead_global.png` — MAE + RMSE + sMAPE × Vorlauf
- `retro_accuracy_by_kennzahl.png` — MAE × Vorlauf, je Kennzahl
- `retro_accuracy_by_produkt.png` — MAE × Vorlauf, je Produkt
- `retro_accuracy_by_sparte.png` — MAE × Vorlauf, je Sparte
- `retro_heatmap_mae.png` — Heatmap: Komponente × Vorlauf

Die `retrospective_accuracy.csv` wird bei jedem Lauf komplett neu
geschrieben (kein Append), weil neue Ist-Werte auch ältere Forecasts
erstmals oder besser bewertbar machen.

**Wachstumsmechanik:** Je mehr Läufe archiviert sind, desto dichter wird
die Datenbasis. Nach 13 wöchentlichen Läufen hat die Retro-Evaluation
für den h13-Horizont Vergleichspunkte bei Vorlauf 1–13 Wochen. Nach
einem Jahr sind alle Vorlauf-Stufen bis 52 Wochen abgedeckt.

### Granularität

Alle Auswertungen (Backtest-Metriken und Retro-Evaluation) enthalten
Dimensions-Spalten, die aus dem Komponentennamen abgeleitet werden:

| Dimension | Beispiel | Aggregation |
|---|---|---|
| `kennzahl` | `TERM_EINGANG_SCHRIFTST` | Schriftstück vs. Sonstige |
| `produkt` | `KFZ_Vollkasko` | Einzelprodukt |
| `status` | `Neuschaden` | Neuschaden vs. Folgebearbeitung |
| `sparte` | `KFZ` | KFZ-Gesamt vs. HUS-Gesamt |

Damit lassen sich beliebige Schnitte in Python oder Excel erstellen:

```python
retro = pd.read_csv("/mnt/runs/retrospective_accuracy.csv")

# MAE nach Sparte und Vorlauf
retro.groupby(["sparte", "lead_weeks"])["abs_error"].mean().unstack(0).plot()

# Welches Produkt hat den höchsten Fehler bei 4 Wochen Vorlauf?
retro[retro["lead_weeks"] == 4].groupby("produkt")["abs_error"].mean().sort_values()
```

---

## Prognose-Tabelle (DB2)

Das Schreiben der Prognose läuft per **Staging-Swap-Pattern**:

1. Die Prognose wird zunächst in eine Staging-Tabelle
   `{target}_STAGING` geschrieben.
2. Anschließend wird die Zieltabelle in einer einzigen Transaktion per
   `DELETE + INSERT … SELECT FROM staging` ersetzt.

Damit ist der Austausch **rollback-fähig** (ein direktes `TRUNCATE` wäre
es in DB2 nicht) und die Zieltabelle bleibt für parallele Reader bis zum
finalen `COMMIT` auf dem bisherigen Stand sichtbar.

Die Staging-Tabelle wird beim ersten Lauf automatisch per
`CREATE TABLE {target}_STAGING LIKE {target}` angelegt – sofern das
DB-Account dafür Rechte hat. **Falls nicht**, bitte vom DBA einmalig
anlegen lassen:

```sql
CREATE TABLE t7.TA_DA_PLUTO_SP_2025_PROGNOSE_STAGING
  LIKE t7.TA_DA_PLUTO_SP_2025_PROGNOSE;
```
