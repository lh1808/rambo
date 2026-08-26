# Produktivsetzung eines Kausalscores — Schritt für Schritt

Dieser Leitfaden beschreibt, wie ein neuer Kausalscore produktiv gesetzt wird:
welche Dateien angelegt oder angepasst werden, was Pflicht und was optional
ist, und wie der Erstlauf verifiziert wird. Kurzreferenz der einzelnen
Optionen: [README.md](README.md). Kopiervorlagen mit allen Feldern und
Pflicht/Optional-Markierung: `scoring_template_file.yml` (Datei-Flow) und
`scoring_template_saspy.yml` (saspy-Flow); `scoring_ph.yml` ist ein
konkreter, gelebter Use-Case.

## Auf einen Blick: Neuen Score produktiv setzen

1. **Bundle exportieren** — im Analyse-Lauf `bundle.enabled: true` + `bundle_id`
   setzen; das Bundle-Verzeichnis nach `/mnt/Production/<usecase>/bundle/`
   legen *(Details: Schritt 1)*.
2. **Scoring-Config anlegen** — `scoring_template_file.yml` (bzw. `_saspy`)
   nach `production/scoring_<usecase>.yml` kopieren, PFLICHT-Felder füllen
   *(Schritt 2, Modellauswahl: Schritt 3)*.
3. **Lokal testen** —
   `python production/run_scoring.py --config production/scoring_<usecase>.yml`
   *(Schritt 5 zeigt die Checks)*.
4. **Job-Datei anlegen** — `production/jobs/job_ph.conf` nach
   `job_<usecase>.conf` kopieren, `SCORING_CONFIGS` (und später `GIT_REF`)
   eintragen; alles committen *(Schritt 4)*.
5. **Domino-Job anlegen** — Skript- und Job-Datei-Kopie an die FS-Ablage,
   Kommando (Pfade der Ablage):
   `bash .../run_scoring.sh .../job_<usecase>.conf`;
   Schedule unter dem Service-Account-Login *(Schritt 4, inkl. Pfad-Semantik)*.
6. **Für Prod härten** — `GIT_REF` in der Job-Datei auf ein Tag pinnen und
   `REQUIRE_PINNED_REF=1` setzen; Erstlauf gegen Testziele fahren
   *(Schritt 5, Betrieb)*.

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
| `production/run_scoring.sh` | Generisches Job-Skript: Clone → pixi-Env → Scoring-Schleife. | **Nie.** Steuerung ausschließlich über die Job-Datei |
| `production/run_scoring.py` | Datei-Flow (parquet/csv/sas7bdat → XPT). | Nie |
| `production/run_scoring_saspy.py` | SAS-Library-Flow (`sd2df` → `df2sd`/`PROC APPEND`). | Nie |
| `sascfg_personal.py` (saspy-Konfig, außerhalb des Repos) | SAS-Verbindung (Host, Port, Auth). | Einmalig pro Umgebung; **Credentials nie ins Repo** |
| Bundle-Verzeichnis (`runs/bundles/<id>` bzw. Prod-Ablage) | Modelle + Preprocessor + Schema + Metadaten, self-contained. | Nie manuell — entsteht durch den Analysis-Export |

Grundprinzip: **Der Use-Case steckt vollständig in seiner YAML.** Skripte
bleiben unverändert; der Job wählt per `SCORING_CONFIGS`, was läuft.

---

## Ablage-Landkarte: Was liegt wo — und was nur einmal

| Artefakt | Ablageort | Anzahl |
|---|---|---|
| Bundle | `/mnt/Production/<usecase>/bundle/` | **1×** (außerhalb des Repos) |
| `scoring_<usecase>.yml` | Repo (`production/`) — wird aus dem **Clone** gelesen | **1×** — **nicht** aufs FS kopieren! |
| `run_scoring.py`, `run_scoring_saspy.py` | Repo — laufen aus dem Clone | **1×** |
| `run_scoring.sh` | Master im Repo **+ Bootstrap-Kopie** am FS-Startort des Jobs | **2×** (einzige Ausnahme, s. u.) |
| `job_<usecase>.conf` | Master im Repo (`production/jobs/`) **+ Kopie** neben der Skript-Kopie | **2×** (einzige Ausnahme, s. u.) |
| XPT / Monitoring-JSON | Zielpfade aus der Scoring-YAML | **1×** |
| SSH-Deploy-Key, `sascfg_personal.py` | Home des Service-Users | **1×** (nie im Repo) |
| pixi-Environment | `WORKDIR` (Clone) | ephemer, pro Lauf |

**Warum genau zwei Dateien doppelt liegen:** Der Job braucht Startskript und
Job-Datei, **bevor** der Clone existiert (Bootstrap) — alles andere kommt aus
dem frisch geklonten, gepinnten Repo-Stand. Gegen veraltete Kopien warnt das
Skript nach dem Clone automatisch (**Drift-Check**: laufende Kopie und
Job-Datei werden gegen den Repo-Master verglichen; bei Abweichung erscheint
eine WARNUNG mit beiden Pfaden im Log).

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

`run_scoring.sh` bleibt generisch und unverändert; **Skript und Job-Datei
liegen zusammen** am Startort des Jobs — die Kopien auf dem Domino File
System (z. B. `/mnt/Production/<usecase>/run_scoring.sh` +
`.../job_<usecase>.conf`; Master beider Dateien im Repo) nach Repo-Änderungen
synchronisieren. Das Job-Kommando nutzt dann die FS-Pfade:
`bash /mnt/Production/<usecase>/run_scoring.sh /mnt/Production/<usecase>/job_<usecase>.conf`. Steuerung über die **Job-Datei** (`KEY="wert"`; Flags als
Test-Override, `--help` zeigt alles; Env wird nicht gelesen):

| Conf-Key | Default | Bedeutung |
|---|---|---|
| `SCORING_CONFIGS` | `production/scoring_ph.yml` | Leerzeichen-Liste der zu fahrenden Configs — **hier** kommen neue Use-Cases dazu. Jeder Score läuft in einem eigenen Python-Prozess (Speicher wird zwischen Scores vollständig freigegeben) |
| `GIT_REF` | `main` | **Für Prod pinnen** — annotiertes Tag empfohlen (Shallow-Clone, schnell); Commit-SHAs funktionieren über den automatischen Fallback (voller Clone + Checkout, langsamer). Pinnt Code **und** `pixi.lock` als Einheit, `--frozen` installiert exakt diesen Stand |
| `CONTINUE_ON_ERROR` | `0` | `1`: alle Scores versuchen; Exit ≠ 0, wenn einer scheitert |
| `REQUIRE_PINNED_REF` | `0` | `1` **für Prod-Jobs empfohlen**: Abbruch (Exit 2), wenn `GIT_REF` ein Branch ist — statt nur Warnung |
| `GIT_USER_EMAIL` / `GIT_USER_NAME` | — | Optional: `git config --global` (bewährtes TFS-Umgebungs-Muster; fürs Klonen nicht nötig) |
| `WORKDIR`, `GIT_URL`, `PIXI_ENV` | s. Skript | Umgebung; selten anzufassen |
| `SKIP_SETUP`, `RUN_CMD` | — | Testwerkzeug — üblicher als Flags beim Trockenlauf: `--skip-setup --run-cmd "python"` |

Gemischte Jobs (Datei- und saspy-Scores in einem Lauf) sind möglich — der
`runner:`-Key jeder Config entscheidet; das Routing übernimmt
`run_scoring.py` selbst (die Shell kennt nur diesen einen Einstieg).
Config-Pfade dürfen keine Leerzeichen enthalten (Leerzeichen trennt die Liste).

### Mentales Modell der Steuerung

Das Job-Skript ist ein **vorausgefülltes Formular**: Jede Zeile der Form
`VAR="default"` ist ein vorausgefülltes Feld; übertippt wird es durch die
**Job-Datei** (und für Tests durch Flags). Es gibt zwei Präzedenz-Ebenen,
mehr nicht:

1. **Job-Ebene (Job-Datei > Skript-Default; Test-Flags nur für den
   Trockenlauf):** *Was* läuft (`SCORING_CONFIGS`), *welcher Code-Stand*
   (`GIT_REF`), *Fehlerpolitik* (`CONTINUE_ON_ERROR`). Das sind die drei
   Angaben, die man tatsächlich setzt — der Rest ist Umgebungskonstante
   oder Testwerkzeug.
2. **Score-Ebene (CLI > YAML):** *Wie* ein einzelner Score läuft, steht
   vollständig in seiner `scoring_<usecase>.yml`. Die CLI-Flags von
   `run_scoring.py` (`--input/--bundle/--output`) übersteuern einzelne
   YAML-Werte — gedacht für lokale Tests, nicht für Job-Definitionen.

Der Use-Case lebt damit versioniert im Repo (Config-Datei), der Job kennt nur
Listen von Configs und einen gepinnten Code-Stand.

### Steuerung per Job-Datei (primär — kein Env, keine Flag-Ketten)

Das Job-Interface ist eine **Job-Datei** in flacher `KEY="wert"`-Syntax
(Vorlage: `production/jobs/job_ph.conf`) — sie liest sich wie Config, wird
aber von Bash **nativ per `source` geladen** (kein YAML-Parser vor dem Clone
nötig; genau deshalb ist es bewusst kein echtes YAML). Env-Variablen werden
nicht gelesen. Die Domino-Job-Definition ist damit eine Zeile:

```
bash production/run_scoring.sh production/jobs/job_ph.conf
```

Pro Use-Case-Job eine Datei unter `production/jobs/` — **committet und
reviewbar**: ein `GIT_REF`-Bump ist ein Ein-Zeilen-Commit mit Audit-Trail.
Ein Validierungs-Guard erlaubt in der Datei ausschließlich Zuweisungen der
bekannten Parameter (plus Kommentare) und bricht sonst ab. Die Git-Identity
des Umgebungs-Setups lässt sich dort mitgeben (`GIT_USER_EMAIL`/`GIT_USER_NAME`
→ `git config --global`) — fürs Klonen nicht erforderlich, aber kompatibel
zum bewährten TFS-Einrichtungs-Muster.

Die Job-Datei ist **Pflicht-Argument** — ohne sie startet das Skript nicht
(klare Fehlermeldung mit Aufrufbeispiel). Als Kommandozeilen-Flags existieren
ausschließlich die beiden Testwerkzeuge `--skip-setup` und `--run-cmd`
(Trockenlauf derselben Job-Datei ohne Clone/Pixi); sämtliche Job-Parameter
haben genau einen Ort: die Job-Datei.

**Pfad-Semantik (wichtig):** Die Job-Datei wird **vor** dem Clone gelesen —
ihr Pfad gilt relativ zum **Startkontext** des Jobs (dort, wo auch die
`run_scoring.sh`-Kopie liegt). Die Pfade **in** `SCORING_CONFIGS` werden
dagegen **nach** dem Clone im frisch ausgecheckten Repo (`WORKDIR`)
aufgelöst — sie bleiben daher Repo-Pfade wie
`production/scoring_<usecase>.yml`, unabhängig davon, wo Skript und
Job-Datei abgelegt sind. Absolute Pfade in `SCORING_CONFIGS` funktionieren
ebenfalls (z. B. für Test-Configs außerhalb des Repos).

Ohne „Run as"-Option gilt: Scheduled Runs laufen unter dem User, der sie
angelegt hat (bzw. im Projekt-Kontext) — das Prod-Projekt daher dem
Service-Account zuordnen und die Schedules unter dessen Login anlegen; sein
`~/.ssh`-Deploy-Key wird dann automatisch gefunden. Schnelltest im
Zielprojekt: `whoami; echo ~; ls -la ~/.ssh`.

**Exit-Codes des Job-Skripts** (für Domino-Alerting): `0` = alle Scores OK ·
`2` = Abbruch vor dem Scoren durch den Reproduzierbarkeits-Guard
(`--require-pinned` bei Branch-Ref) · bei Score-/Config-Fehlern
(inkl. unbekanntem `runner:` — das prüft jetzt `run_scoring.py` selbst):
Fail-Fast (Default) propagiert den Exit-Code des fehlgeschlagenen
Python-Prozesses; mit `CONTINUE_ON_ERROR=1` laufen alle Scores und das Skript
endet mit `1`, wenn mindestens einer fehlschlug (Summary im Log).

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
   `bash production/run_scoring.sh production/jobs/job_<usecase>.conf --skip-setup --run-cmd "python"`
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
- **Weiterer Kausalscore:** neue `scoring_<usecase>.yml` + entweder Eintrag
  in `SCORING_CONFIGS` eines bestehenden Jobs (gleicher Zeitplan) oder
  eigene `job_<usecase>.conf` + eigener Domino-Job (eigener Zeitplan).
  Unterschiedliche Feature-Teilmengen gegen dieselbe breite Eingabetabelle
  sind der Normalfall (Spalten-Pruning pro Bundle).
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
