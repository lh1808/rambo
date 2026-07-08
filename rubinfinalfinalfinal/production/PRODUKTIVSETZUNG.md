# Produktivsetzung eines Kausalscores — Schritt für Schritt

Dieser Leitfaden beschreibt, wie ein neuer Kausalscore produktiv gesetzt wird:
welche Dateien angelegt oder angepasst werden, was Pflicht und was optional
ist, und wie der Erstlauf verifiziert wird. Kurzreferenz der einzelnen
Optionen: [README.md](README.md). Kopiervorlagen mit allen Feldern und
Pflicht/Optional-Markierung: `scoring_template_file.yml` (Datei-Flow) und
`scoring_template_saspy.yml` (saspy-Flow); `scoring_ph.yml` ist ein
konkreter, gelebter Use-Case.

## Wie hängt alles zusammen?

```
Analysis-Lauf ──▶ Bundle ─────────▶ scoring_<usecase>.yml ──▶ run_scoring.sh ──▶ Score-Output + Monitoring-JSON
(Training)        (Modelle,          (der Use-Case:            (Domino-Job:        (XPT-Datei oder SAS-
                  Preprocessor,      was, woher, womit,         ein Prozess         Tabelle — je nach runner)
                  Schema)            wohin)                     pro Config)
```

Die Analyse trainiert und exportiert ein **Bundle** (self-contained: Modelle
+ Preprocessing + Schema). Eine **Scoring-YAML** pro Use-Case beschreibt,
welches Bundle gegen welche Daten scort und wohin das Ergebnis geht. Das
**Job-Skript** führt eine oder mehrere solcher YAMLs aus; jeder Lauf
hinterlässt ein **Monitoring-JSON** als Kontrollausdruck.

---

## Überblick: Welche Datei hat welche Rolle?

| Datei | Rolle | Anpassen? |
|---|---|---|
| `production/scoring_template_*.yml` | Kopiervorlagen (Datei-/saspy-Flow): alle Felder, PFLICHT/optional markiert. | Nie — nur kopieren |
| `production/scoring_<usecase>.yml` | **Der Use-Case.** Definiert Transport, Input, Bundle, Modellauswahl, Output, Monitoring. | **Pro Use-Case anlegen** (Kopie der passenden Vorlage; Beispiel: `scoring_ph.yml`) |
| `production/run_scoring.sh` | Generisches Job-Skript: Clone → pixi-Env → Scoring-Schleife. | **Nie.** Steuerung ausschließlich über Env-Variablen des Jobs |
| `production/run_scoring.py` | Datei-Flow (parquet/csv/sas7bdat → XPT). | Nie |
| `production/run_scoring_saspy.py` | SAS-Library-Flow (`sd2df` → `df2sd`/`PROC APPEND`). | Nie |
| `sascfg_personal.py` (saspy-Konfig, außerhalb des Repos) | SAS-Verbindung (Host, Port, Auth). | Einmalig pro Umgebung; **Credentials nie ins Repo** |
| Bundle-Verzeichnis (`runs/bundles/<id>` bzw. Prod-Ablage) | Modelle + Preprocessor + Schema + Metadaten, self-contained. | Nie manuell — entsteht durch den Analysis-Export |

Grundprinzip: **Der Use-Case steckt vollständig in seiner YAML.** Skripte
bleiben unverändert; der Job wählt per `SCORING_CONFIGS`, was läuft.

---

## Schritt 1 — Bundle erzeugen

In der **Analysis-Config** des Use-Cases:

```yaml
bundle:
  enabled: true
  base_dir: runs/bundles        # Ablage; bundle_id optional (sonst Zeitstempel)
```

Der Export refittet **alle** trainierten Modelle (inkl. self-contained
Ensemble) auf vollen Daten — in Production ist damit jedes Modell frei
wählbar, ohne Re-Export. Das Bundle enthält:

| Artefakt | Zweck |
|---|---|
| `models/*.pkl` | Alle Modelle (werden beim Scoring lazy geladen) |
| `preprocessor.pkl` | Rohdaten-fähig, wenn der Run über DataPrep lief (Encoding-Maps, gelernte Missing-Behandlung gemäß Trainings-Konfiguration); sonst Schema-only — dann muss der Scoring-Input bereits DataPrep-enkodiert ankommen (das Analysis-Log warnt entsprechend) |
| `schema.json`, `dtypes.json` | Erwartete Spalten/Typen; Abweichungen werden gemeldet |
| `metadata.json`, `model_registry.json` | Champion, Versions-Stempel, verfügbare Modellnamen |
| `config_snapshot.yml` | Reproduzierbarkeit des Trainings |

Das fertige Bundle-Verzeichnis an die Prod-Ablage kopieren (z. B.
`/mnt/Production/<usecase>/bundle`). **Wichtig:** Das Scoring lehnt Bundles
ohne `metadata.json`/Versions-Stempel ab; Versions-Abweichungen zwischen
Export- und Scoring-Environment werden geloggt (gleiche pixi-solve-group
vermeidet sie).

---

## Schritt 2 — Scoring-Config anlegen

`production/scoring_<usecase>.yml`, Kopiervorlage je nach Transport:
`scoring_template_file.yml` (Datei-Flow) oder `scoring_template_saspy.yml`
(SAS-Library-Flow).

### Pflichtfelder (MUSS pro Use-Case gesetzt werden)

| Flow | Feld | Bedeutung |
|---|---|---|
| beide | `name` | Use-Case-Name — bestimmt die Monitoring-Dateinamen |
| beide | `runner` | `file` oder `saspy` — das Job-Skript wählt danach den Einstieg |
| beide | `bundle` | Pfad zum Bundle-Verzeichnis |
| beide | `id_columns` | Spalten, die unverändert in den Output durchgereicht werden (kein Modell-Feature — Überschneidung wird hart abgelehnt) |
| Datei | `input.path` | Eingabedatei (parquet/csv/sas7bdat; `format` optional aus Endung) |
| Datei | `output.xpt_path` | XPT-Zieldatei, wird pro Lauf überschrieben (`table_name` optional, Default `SCORES`) |
| saspy | `input.libref`, `input.table` | Quell-Dataset: SAS-Library + Tabelle, aus der gelesen wird |
| saspy | `output.libref`, `output.table` | Ziel-Dataset: SAS-Library + Tabelle, in die geschrieben wird |
| saspy | `monitoring.dir` | Ablage der Monitoring-JSONs (im Datei-Flow optional: Default neben der XPT) |

### Optionale Felder (KANN — sinnvolle Defaults)

| Flow | Feld | Default | Wann anpassen |
|---|---|---|---|
| beide | `input.uppercase_columns` | Datei: `false`, saspy: `true` | **Muss zur Schreibweise der Bundle-Features passen** (DataPrep-Bundles: Großschreibung → `true`); bei Mismatch bricht der Lauf mit klarer Meldung ab |
| beide | `input.pull_only_needed_columns` | `true` | Nur Bundle-Features + IDs werden gelesen (Datei: `columns=`/`usecols=`; saspy: `keep=`). Auf `false` nur zum Debuggen einer breiten Tabelle |
| beide | `scoring.batch_size` | 100 000 | Zeilen pro Modell-Predict |
| beide | `scoring.round_decimals` | 6 | Score-Rundung (keine Skalierung — bewusst) |
| beide | `output.meta_columns` / `column_order` / `timestamp_format` | — | Schnittstellen-Konventionen des Zielsystems (konstante Spalten, Reihenfolge, TIMESTAMP-Format) |
| beide | `preprocessing.replace_inf_with_nan` | `true` | Inf → NaN vor dem Transform — die im Bundle gelernte Missing-Behandlung greift |
| Datei | `input.csv_sep` / `input.csv_encoding` | `,` / `utf-8` | Deutsche CSVs: `";"` bzw. `ISO-8859-1` |
| Datei | `output.file_format_version` | 5 | XPT V5 kürzt Namen auf 8 Zeichen (Kollisionen werden hart abgelehnt; volle Namen bleiben als Labels); V8 für lange Namen |
| saspy | `saspy.cfgname` | saspy-Default | Eintrag in `sascfg_personal.py` (Verbindung) |
| saspy | `saspy.setup_code` | — | SAS-Code nach Session-Start, z. B. `libname`-Zuweisungen, falls nicht per autoexec vorhanden; wird log-geprüft |
| saspy | `input.where` | — | SAS-WHERE-Selektion beim Pull |
| saspy | `input.chunk_size` / `output.write_chunk_size` | 500 000 | An Tabellengröße/Transport anpassen; Erstlauf klein wählen (siehe Schritt 5) |
| saspy | `output.write_mode` | `replace` | `append` hängt an den Bestand an |

### Variante: mehrere Bundles gegen denselben Input (Ein-Lade-Modus)

Sollen mehrere Modelle/Bundles auf **derselben Gesamttabelle** scoren und
ist das **Laden teuer**, trägt die Config statt `bundle`/`scoring`/`output`
eine **`scores:`-Liste** (je Eintrag: `name`, `bundle`, optional `scoring`,
`output`): Der Input wird nur **einmal** gelesen — Spalten = Union aller
Bundle-Features + IDs — und dann pro Eintrag mit dessen Bundle-Preprocessor
verarbeitet und gescort. Monitoring-JSONs heißen `<name>_<eintrag>_….json`.
Beispielblock: auskommentiert in beiden Vorlagen.

Zwei Dinge sind zu wissen: **(1) Gleicher Input-Zustand.** Alle Einträge
verarbeiten dieselbe Ladung — alle Bundles müssen denselben Zustand
erwarten (alle rohdaten-fähig *oder* alle Schema-only); ein Mismatch zeigt
sich im Monitoring als hohe −1-Raten bzw. `missing_expected_columns`.
**(2) Fehler-Semantik.** Nicht ladbare Bundles brechen ab, *bevor*
irgendetwas gelesen oder geschrieben wird (alle Bundles werden zuerst
geladen). Schlägt dagegen ein späterer Eintrag zur Laufzeit fehl, bleiben
die bereits geschriebenen Ziele und Monitoring-JSONs früherer Einträge
bestehen — der Job endet rot, die Meldung nennt den Eintrag.

Abgrenzung: Für **unabhängige Quellen** bleiben getrennte Configs in
`SCORING_CONFIGS` (ein Prozess pro Score, maximale Speicher-Isolation)
der richtige Weg.

---

## Schritt 3 — Modellauswahl (unabhängig vom Champion)

```yaml
scoring:
  score_p_model: champion      # Alias auf den Registry-Champion — oder konkreter Name
  score_b_model: SurrogateTree # optional; null → keine SCORE_B-Spalte
  extra_models: [DRLearner]    # beliebig viele weitere → CATE_<Name>-Spalten
```

Jeder Modellname aus dem Bundle ist wählbar (`Ensemble`, `DRLearner`, …) —
der Champion ist nur der Default-Alias. Verfügbare Namen: `model_registry.json`
im Bundle bzw. das Log beim Scoring-Start; ein unbekannter Name schlägt hart
fehl und listet die vorhandenen. Das Monitoring dokumentiert pro Lauf, welches
Modell hinter `SCORE_P`/`SCORE_B` stand.

---

## Schritt 4 — Job einrichten

`run_scoring.sh` bleibt unverändert; die Kopie auf dem Domino File System
(z. B. `/mnt/Production/<usecase>/run_scoring.sh`) nach Repo-Änderungen
synchronisieren. Steuerung über **Env-Variablen der Job-Definition**:

| Variable | Default | Bedeutung |
|---|---|---|
| `SCORING_CONFIGS` | `production/scoring_ph.yml` | Leerzeichen-Liste der Configs — **hier** kommen neue Use-Cases dazu. Jeder Score läuft in einem eigenen Python-Prozess (Speicher wird zwischen Scores vollständig freigegeben) |
| `GIT_REF` | `main` | **Für Prod pinnen** — annotiertes Tag empfohlen (Shallow-Clone, schnell); Commit-SHAs funktionieren über den automatischen Fallback (voller Clone + Checkout, langsamer). Pinnt Code **und** `pixi.lock` als Einheit, `--frozen` installiert exakt diesen Stand |
| `CONTINUE_ON_ERROR` | `0` | `1`: alle Scores versuchen; Exit ≠ 0, wenn einer scheitert |
| `RUNNER_SCRIPT` | — | Erzwingt einen Einstieg für alle Configs (normal: `runner:`-Key pro YAML) |
| `BUNDLE/INPUT/OUTPUT_OVERRIDE` | — | Nur im Ein-Score-Datei-Fall erlaubt |
| `PIXI_ENV`, `WORKDIR`, `GIT_URL` | s. Skript | Umgebung; selten anzufassen |

Gemischte Jobs (Datei- und saspy-Scores in einem Lauf) sind möglich — der
`runner:`-Key jeder Config entscheidet. Config-Pfade dürfen keine
Leerzeichen enthalten (Leerzeichen trennt die Liste).

**Exit-Codes des Job-Skripts** (für Domino-Alerting): `0` = alle Scores OK ·
`2` = Konfigurationsfehler vor dem Scoren (unbekannter `runner:`, unzulässige
Overrides) · bei Score-Fehlern: Fail-Fast (Default) propagiert den Exit-Code
des fehlgeschlagenen Runners; mit `CONTINUE_ON_ERROR=1` laufen alle Scores
und das Skript endet mit `1`, wenn mindestens einer fehlschlug (Summary im Log).

### saspy-Voraussetzungen (einmalig pro Umgebung)

1. saspy ist im pixi-`prod`-Environment enthalten (pip: `pip install -e ".[saspy]"`).
2. `sascfg_personal.py` mit der Verbindung anlegen (IOM/SSH; siehe saspy-Doku);
   der Eintragsname kommt als `saspy.cfgname` in die Config.
3. Librefs bereitstellen: per autoexec der SAS-Umgebung **oder**
   `saspy.setup_code` in der Config.

---

## Schritt 5 — Erstlauf & Verifikation

1. **Config-Check lokal** (lädt und validiert — auch Schlüssel-Tippfehler
   werden hart abgelehnt; ohne SAS/Domino):
   ```bash
   # Datei-Flow:
   python -c "import sys; sys.path.insert(0,'production'); \
     from run_scoring import load_scoring_config; \
     load_scoring_config('production/scoring_<usecase>.yml')"
   # saspy-Flow: analog mit run_scoring_saspy.load_saspy_scoring_config
   ```
2. **Trockenlauf der Job-Schleife** ohne Clone/Pixi:
   `SKIP_SETUP=1 RUN_CMD="python" SCORING_CONFIGS="production/scoring_<usecase>.yml" bash production/run_scoring.sh`
3. **Erster SAS-Lauf konservativ:** kleine `chunk_size`, Ziel auf eine
   **Testtabelle** zeigen lassen (`--table-out`-Override bzw. Test-Config),
   Monitoring-`n_rows` gegen die erwartete Selektion zählen.
4. **Monitoring-Checkliste** (`<name>_latest.json`):
   - `input.n_rows` plausibel; `column_pruning: true` (sofern nicht deaktiviert)
   - `preprocessing.missing_expected_columns` leer (sonst fehlen Features im Input!)
   - `preprocessing.minus1_rate_per_categorical` niedrig (>1 % = Drift-Warnung im Log)
   - `scores.SCORE_P`: `nan: 0`, Verteilung plausibel (std > 0)
   - `bundle.version_mismatches` leer
   - `models.score_p` = das beabsichtigte Modell

---

## Betrieb

- **Bundle-Update:** neues Bundle an die Prod-Ablage, `bundle:`-Pfad in der
  YAML (oder gleicher Pfad, Verzeichnis tauschen). Sonst nichts.
- **Modell wechseln:** nur `scoring.score_p_model` in der YAML — kein
  Re-Export, kein Champion-Wechsel nötig.
- **Weiterer Kausalscore:** neue `scoring_<usecase>.yml` + Eintrag in
  `SCORING_CONFIGS`. Unterschiedliche Feature-Teilmengen gegen dieselbe
  breite Eingabetabelle sind der Normalfall (Spalten-Pruning pro Bundle).
- **Historisierung:** XPT/Zieltabelle wird überschrieben bzw. ersetzt —
  Historisierung übernimmt das Zielsystem; Monitoring-JSONs sind versioniert.

## Häufige Fehlermeldungen

| Meldung | Ursache | Behebung |
|---|---|---|
| `Unbekannter Config-Schlüssel '…' — meinten Sie '…'?` | Tippfehler in der Scoring-YAML (Schlüssel werden strikt validiert, nichts wird still ignoriert) | Vorschlag aus der Meldung übernehmen |
| `… deklariert runner: '…' — diese Config gehört zu …` | Config wurde direkt am falschen Runner-Skript aufgerufen | Genannten Runner verwenden — oder einfach `run_scoring.sh`, das automatisch routet |
| `metadata.json fehlt im Bundle` / `ml_package_versions fehlt` | Unvollständiges/von Hand gebautes Bundle | Bundle neu aus der Analysis exportieren |
| `score_p_model '…' nicht im Bundle. Vorhanden: […]` | Tippfehler/Modell nicht trainiert | Namen aus der Fehlermeldung bzw. dem Start-Log übernehmen |
| `ID-Spalten sind zugleich Modell-Features` | ID war im Training Feature | `id_columns` bereinigen oder DataPrep mit `deduplicate_id_column` nutzen (schließt die ID aus X aus) |
| `ID-Spalten fehlen im Input` | Spalte nicht in Quelle (saspy: vom `keep=` nicht erfasst, weil falsch geschrieben) | Schreibweise gegen die Quelle prüfen |
| `KEINE der vom Bundle erwarteten Feature-Spalten ist im Input vorhanden` | `input.uppercase_columns` passt nicht zur Schreibweise der Bundle-Features | DataPrep-Bundles erwarten Großschreibung → `uppercase_columns: true`; sonst Schreibweise von Quelle und Bundle-Schema vergleichen |
| `SAS-Fehler bei PROC APPEND …` | Zieltabelle gesperrt/Libref fehlt | Sperre lösen; Libref per autoexec oder `saspy.setup_code` zuweisen |
| `XPT-V5-Namenskollision nach 8-Zeichen-Kürzung` | Lange Spaltennamen bei V5 — bei Multi-Treatment typisch: `CATE_<Name>1/2` kürzen auf denselben 8-Zeichen-Stamm | Spalten umbenennen oder `file_format_version: 8` (bei MT mit `extra_models` praktisch Pflicht) |
| NaN in `SCORE_P` (Monitoring `scores.*.nan > 0`) | Input-Missings + Schema-only-Preprocessor | Run über DataPrep fahren (rohdaten-fähiger Preprocessor) oder Input vorab imputieren |
