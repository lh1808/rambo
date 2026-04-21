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

**1. YAML-Datei.** Der Job liest automatisch eine `config.yaml` neben
`pluto_forecast_job.py`, wenn vorhanden. Alternativ per Env-Variable:

```bash
export PLUTO_CONFIG=/pfad/zu/deiner/config.yaml
```

Die YAML überschreibt nur, was sie angibt – alle übrigen Felder bleiben
auf den Defaults. Vorlage: `config.example.yaml`.

```yaml
# Beispiel
tft:
  n_epochs: 100
  learning_rate: 5.0e-4
preprocessing:
  lags: [4, 8, 13, 52]
tuning:
  13:
    enabled: true
    n_trials: 50
```

**2. Env-Variablen mit Präfix `PLUTO__`.** Doppel-Unterstrich als Pfadtrenner,
punktuelle Overrides einzelner Werte. Env schlägt YAML schlägt Defaults.

```bash
export PLUTO__TFT__N_EPOCHS=50
export PLUTO__TUNING__13__ENABLED=true
```

Tippfehler in Feldnamen (YAML oder Env) werfen sofort einen `ValueError`
mit einer Liste der erlaubten Felder – keine silently-ignored-Falle.

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
