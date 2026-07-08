# production/ — Produktiv-Scoring

Generisches Scoring gegen ein rubin-Bundle mit SAS-Rückschreibung (XPT) und
Monitoring-JSON.

**Schritt-für-Schritt-Anleitung zur Produktivsetzung** (welche Dateien
anlegen/anpassen, Pflicht- vs. Kann-Felder, Erstlauf-Checkliste, Betrieb):
[PRODUKTIVSETZUNG.md](PRODUKTIVSETZUNG.md).

## Nutzung

Zwei Einstiege, gleicher Scoring-Kern (`score_dataframe`):

| Einstieg | Input | Output | Vorlage | Beispiel-Use-Case |
|---|---|---|---|---|
| `run_scoring.py` | Datei (parquet/csv/sas7bdat) | SAS-XPT (+ optional CSV) | `scoring_template_file.yml` | `scoring_ph.yml` |
| `run_scoring_saspy.py` | SAS-Library (via saspy `sd2df`) | SAS-Library (via `df2sd` + `PROC APPEND`) | `scoring_template_saspy.yml` | — |

Pro Use-Case entsteht eine `scoring_<usecase>.yml` in diesem Ordner als
Kopie der passenden Vorlage — versioniert und reviewbar wie Code. Die
Vorlagen kommentieren alle Optionen und markieren jedes Feld als PFLICHT
oder optional (mit Default).

**Output-Spalten** (beide Flows identisch, Reihenfolge per `column_order`):
die `id_columns` (unverändert durchgereicht) · `SCORE_P` (gewähltes
Hauptmodell; bei Multi-Treatment `SCORE_P1`, `SCORE_P2`, …) · optional
`SCORE_B` (Zweitmodell, z. B. Surrogate) · optional `CATE_<Name>` je
`extra_models`-Eintrag · die `meta_columns` (konstante Werte) · `TIMESTAMP`
(automatisch, Format per `timestamp_format`).

```bash
pixi run -e prod python production/run_scoring.py --config production/scoring_ph.yml
# Pfad-Overrides ohne Config-Änderung:
python production/run_scoring.py --config production/scoring_ph.yml \
    --input neuer_datensatz.parquet --bundle runs/bundles/<id> --output ziel.xpt
```

## Domino-Job

`run_scoring.sh` ist das generische Job-Einstiegsskript (Use-Cases über `SCORING_CONFIGS`): Master versioniert hier im
Repo, die **ausgeführte Kopie liegt auf dem Domino File System** (z. B.
`/mnt/Production/<usecase>/run_scoring.sh`) — nach Repo-Änderungen
dorthin synchronisieren. Ablauf: frischer Clone (pinnbar via `GIT_REF`,
für Prod-Läufe Tag/Commit statt `main` empfohlen) → pixi-Install (mit `pixi.lock` im
Repo: `--frozen`, exakter Lockfile-Stand ohne Re-Solve; ohne Lockfile — aktueller
Stand, `pixi.lock` ist gitignored — löst pixi anhand der Version-Pins, mit
Warnung im Log; **Empfehlung: `pixi.lock` für Prod-Reproduzierbarkeit
committen**, dann greift automatisch `--frozen`; kein `pixi add` auf dem Executor) →
Scoring-Schleife (ein eigener Python-Prozess pro Config; Einstieg laut
`runner:`-Key der jeweiligen YAML). Parameter per Env: `SCORING_CONFIGS`
(Leerzeichen-Liste; `SCORING_CONFIG` singular bleibt als Alias), `GIT_REF`,
`CONTINUE_ON_ERROR`, `RUNNER_SCRIPT` (erzwingt einen Einstieg für alle),
`BUNDLE_OVERRIDE`/`INPUT_OVERRIDE`/`OUTPUT_OVERRIDE` (nur im
Ein-Score-Datei-Fall). Läuft unter einem
Service-User: nur dessen SSH-Deploy-Key wird gebraucht, keine Git-Identity
(die ist nur für Commits nötig). `set -euo pipefail` propagiert jeden Fehler
als roten Domino-Job.

## Ablage-Konvention

- **XPT**: fester Zielpfad, wird bei jedem Lauf **überschrieben** — die
  Historisierung übernimmt das aufnehmende System.
- **Monitoring**: `<xpt-Verzeichnis>/monitoring/` (per `monitoring.dir`
  umlegbar). Pro Lauf eine zeitgestempelte JSON
  (`<name>_<YYYY-DDD>_<HHMMSS>.json`, bleibt liegen → Drift-Historie) plus
  eine überschriebene `<name>_latest.json` für schnellen Zugriff.

## Monitoring-Inhalt (pro Lauf)

Laufzeit/Zeitstempel; Input (Pfad, Zeilen, Spalten); Bundle (Champion,
Erstellzeitpunkt, ML-Paketversionen, Versions-Abweichungen zur Laufzeit);
Preprocessing (fehlende erwartete Spalten, verworfene Zusatzspalten,
Inf→NaN-Zellen, **−1-Rate pro kategorialer Spalte** = Anteil im Training
unbekannter Ausprägungen/Missings, das direkteste Drift-Signal); Score-
Statistiken pro Spalte (min/p25/median/mean/p75/max/std, NaN-Anzahl);
Output-Details. Erhöhte −1-Raten (>1 %) werden zusätzlich als Warnung geloggt.

## saspy-Einstieg (SAS-Library → Bundle → SAS-Library)

Für Umgebungen, in denen Input und Output als SAS-Datasets in Libraries
liegen: `run_scoring_saspy.py` zieht die Tabelle gechunkt per `sd2df`
(`firstobs`/`obs`-Dataset-Optionen, optional `where`), scort gegen das
Bundle und schreibt gechunkt per `df2sd` + `PROC APPEND` zurück
(`write_mode: replace` löscht die Zieltabelle vorab, `append` hängt an).
Jeder `submit` wird auf `ERROR:` im SAS-Log geprüft und schlägt hart fehl.

```bash
pixi run -e prod scoring-saspy -- --config production/scoring_<usecase>.yml
# Overrides: --bundle <dir> --table-in <tabelle> --table-out <tabelle>
```

Voraussetzungen: `saspy` (im prod-Environment enthalten; pip:
`pip install -e ".[saspy]"`) und eine saspy-Verbindungskonfiguration
(`sascfg_personal.py`, siehe saspy-Doku) — der `cfgname` wird in der
Scoring-Config referenziert, Credentials gehören nicht in die Config.
Die Librefs (`input.libref`, `output.libref`) müssen in der SAS-Session
zugewiesen sein (z. B. per autoexec).

Kein XPT-Umweg: SAS-Datasets erlauben 32-Zeichen-Namen, die V5-Kürzung
des Datei-Einstiegs entfällt. Monitoring-JSON wird identisch geschrieben
(`monitoring.dir` ist hier Pflicht).


## Mehrere Kausalscores in einem Job

`run_scoring.sh` fährt beliebig viele Scores sequentiell — jeder in einem
**eigenen Python-Prozess**, sodass das Betriebssystem zwischen den Scores
sämtlichen Speicher freigibt (Modelle, Frames, Heap-Fragmentierung); ein
langer Multi-Score-Job kann dadurch nicht durch akkumulierte Zustände in
einen OOM laufen.

```bash
SCORING_CONFIGS="production/scoring_ph.yml production/scoring_kfz.yml" \
  bash production/run_scoring.sh
# Transport pro Config: Top-Level-Key `runner:` in der YAML (file | saspy) —
# gemischte Jobs (Datei- und SAS-Library-Scores) sind möglich.
# RUNNER_SCRIPT (Env) erzwingt einen Einstieg für alle Configs.
# Fehlerpolitik: CONTINUE_ON_ERROR=1 → alle versuchen, Exit ≠ 0 wenn einer scheitert
# Lokaler Trockenlauf ohne Clone/Pixi: SKIP_SETUP=1 RUN_CMD="python"
```

Unterschiedliche Feature-Teilmengen pro Score sind der Normalfall und
unproblematisch: Jeder Lauf liest über sein Bundle nur die benötigten
Spalten aus der Gesamttabelle (`input.pull_only_needed_columns`, Default an;
Datei-Flow: parquet `columns=`/csv `usecols=`, saspy-Flow: `keep=`-Option).

**Zwei Modi für mehrere Scores** — je nachdem, was teuer ist:
*Getrennte Configs* in `SCORING_CONFIGS` (ein Prozess pro Score) isolieren
Speicher maximal, laden die Quelle aber pro Score. Ist das **Laden** der
Engpass (große Datei, saspy-Transport) und die Quelle dieselbe, definiert
EINE Config eine **`scores:`-Liste**: ein Prozess lädt einmal (Spalten =
Union aller Bundle-Features + IDs) und verarbeitet pro Bundle
unterschiedlich — Preis: die Tabelle bleibt über alle Scores im Speicher.
Beide Modi sind kombinierbar (eine scores-Config ist ein Eintrag in
`SCORING_CONFIGS`). Details: Vorlagen und PRODUKTIVSETZUNG.md.
Fehlt eine erwartete Spalte im Input, taucht sie im Monitoring als
`missing_expected_columns` auf. Innerhalb eines Laufs werden die großen
Zwischenstände aktiv freigegeben (Eingabetabelle nach dem Scoren,
transformierte Feature-Matrix vor dem Output-Bau); Modell-Pickles werden
lazy geladen — nur Champion/Surrogate/extra_models, nicht das ganze Bundle.


## Modellauswahl (unabhängig vom Champion)

Welche Modelle scoren, bestimmt ausschließlich die YAML — der Champion ist
nur der Default-Alias: `scoring.score_p_model` und `scoring.score_b_model`
akzeptieren jeden Modellnamen aus dem Bundle (z. B. `Ensemble`, `DRLearner`,
`SurrogateTree`), `scoring.extra_models` ergänzt beliebig viele weitere als
`CATE_<Name>`-Spalten. Da der Bundle-Export immer alle trainierten Modelle
refittet und mitliefert, ist jeder Challenger direkt produktiv wählbar —
ohne Re-Export, ohne Champion-Wechsel. Die verfügbaren Namen stehen in
`model_registry.json`/`metadata.json` des Bundles und werden beim
Scoring-Start geloggt; ein unbekannter Name schlägt hart fehl und listet
die vorhandenen. Das Monitoring-JSON dokumentiert pro Lauf, welches Modell
hinter `SCORE_P`/`SCORE_B` stand.

## Design-Entscheidungen

Keine Batch-Skalierung (ein pro Lauf gefitteter Scaler würde Scores zwischen
Läufen unvergleichbar machen); nur Rundung (`round_decimals`). `SCORE_B` (Surrogate) ist optional. Fachliche
Sonderregeln (z. B. Einzelwert-Ersetzungen) gehören in DataPrep, nicht ins
Scoring — der Scoring-Input muss im selben Roh-Zustand ankommen wie der
DataPrep-Input beim Training.
