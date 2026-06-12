# Design-Notizen

Diese Datei beschreibt Architektur- und Implementierungsentscheidungen,
die sich beim Lesen des Codes nicht unmittelbar erschließen. Sie ist
bewusst kein Changelog; der Quellcode ist die Wahrheit.

## Leakage-Vermeidung im Backtest

Der Rolling-Backtest in :func:`forecasting.utils.rolling_block_forecast`
fittet Skalierer (Winsorizer, Yeo-Johnson, Standardisierung) **pro Block
lokal**, wenn der optionale Parameter ``transforms_builder`` gesetzt ist.
Damit sehen die Skalierer je Fold ausschließlich das aktuelle
Trainingsfenster und niemals die späteren Validierungsdaten. Die
Pipeline-Module geben den Callback ``build_transformers`` genau für
diesen Zweck mit.

Konsequenz: Backtest-Metriken sind realistisch in Richtung
Produktionsverhalten. Ein Fit auf die Gesamtreihe würde leicht bessere
Scores erzeugen, aber keinen validen Qualitätsmaßstab.

## Future-Kovariaten reichen in den Prognosehorizont

``build_weekly_covariates(df, cfg, future_weeks=horizon)`` erzeugt den
Ausgabe-Index über das letzte Beobachtungsdatum hinaus, bis einschließlich
``df.index.max() + horizon * freq``. Das ist Voraussetzung dafür, dass
``model.predict(n=horizon, future_covariates=...)`` durchläuft – darts
verlangt, dass Future-Kovariaten bis mindestens ``end + n`` reichen.

## Zielreihe bei Inferenz explizit übergeben

``forecast_with_artifacts`` reicht die aktuelle, transformierte Zielreihe
explizit als ``series=`` an ``model.predict`` weiter. Ohne diesen Pfad
würde darts auf die im Modell eingefrorene Trainingsserie zurückfallen
und aktualisierte Eingangsdaten ignorieren. Der im Training gefittete
``StaticCovariatesTransformer`` wird in
:class:`forecasting.pipeline.ModelArtifacts` mitgeführt und in der
Inferenz wiederverwendet, damit das One-Hot-Schema konsistent bleibt.

## Merge von Backtest-Blöcken

Bei überlappenden Backtest-Blöcken (``stride < horizon``) ist
``TimeSeries.append`` der falsche Primitiv: darts verlangt dort einen
lückenlosen Anschluss um genau einen Zeitschritt. :func:`_merge_blocks`
in ``utils.py`` geht stattdessen über ``pd.concat`` und behandelt
Duplikate je nach gewählter Strategie (``keep_last`` / ``keep_first`` /
``stack``).

## RollingWinsorizer in der darts-Pipeline

Die Invertierbarkeit einer ``Pipeline`` wird in darts über
``isinstance(..., InvertibleDataTransformer)`` geprüft, nicht über die
``supports_inverse_transform``-Property. Der
:class:`forecasting.model.RollingWinsorizer` erbt deshalb von beiden
Basisklassen (``FittableDataTransformer`` und
``InvertibleDataTransformer``); das Inverse ist die Identität, weil
Winsorizing fachlich nicht umkehrbar ist.

## Konfiguration

Kanonisch sind die :mod:`dataclasses` in ``forecasting/config.py``. Der
Loader in ``forecasting/config_loader.py`` legt darüber ein YAML-Overlay
und anschließend Env-Var-Overrides mit Präfix ``PLUTO__``. Unbekannte
Feldnamen (Tippfehler) führen sofort zu ``ValueError`` mit einer Liste
der erlaubten Felder.

Neue Einträge in Dict-Feldern (``horizons``, ``tuning``) müssen
vollständig angegeben werden – ein Top-Level-Mapping auf einen bisher
nicht existierenden Key wird als ganze Dataclass instanziiert. Env-Vars
decken nur Primitive ab (bool / int / float / str); Listen und Dicts
gehören in die YAML.

## Prognose-Schreiben (DB2)

``PlutoMultivariateRepository.write_forecast`` führt ``DELETE FROM
<target>`` gefolgt von ``executemany INSERT`` in einer einzigen
Transaktion aus. Bei Fehler wird ein ``ROLLBACK`` ausgeführt, sodass
die Zieltabelle auf dem bisherigen Stand verbleibt. ``DELETE`` ist in
DB2 — anders als ``TRUNCATE TABLE IMMEDIATE`` — vollständig
rollback-fähig.

## PyTorch-Lightning-Defaults

``build_tft`` setzt in ``pl_trainer_kwargs`` ``enable_progress_bar``,
``enable_model_summary`` und ``logger`` auf ``False``. Ohne diese
Defaults legt jeder ``fit()``-Aufruf einen eigenen TensorBoard-Log-Ordner
unter ``./darts_logs/`` an, was in Kombination mit Backtest +
Optuna-Tuning schnell in hunderten Ordnern mündet. Wer punktuell
Feedback braucht, kann ``verbose=True`` an ``fit()`` / ``predict()``
reichen.

## EarlyStopping

Bei ``tft_cfg.early_stopping_patience > 0`` wird ein PyTorch-Lightning-
``EarlyStopping``-Callback auf den Training-Loss gesetzt. Das Training
bricht ab, wenn sich der Loss über ``patience`` Epochen nicht verbessert.
Der Default ist 15 Epochen Patience bei max. 200 Epochen – bei typischer
Konvergenz nach 80–120 Epochen spart das 40–60 % der Trainingszeit pro
Block, ohne die Modellqualität zu verschlechtern.

Ein EarlyStopping auf Validierungs-Loss (statt Training-Loss) wäre
fachlich besser, erfordert aber eine Aufteilung des Trainingsfensters in
Train/Val innerhalb jedes Backtest-Blocks. Das ist eine sinnvolle
Erweiterung, verändert aber die Architektur des ``rolling_block_forecast``
und ist deshalb derzeit nicht implementiert.

## Trainingssamples-Berechnung

Die Anzahl der (input, output)-Paare, die darts aus einer Trainingsserie
erzeugt, ist:

    samples = train_length - input_chunk_length - output_chunk_length + 1

Die ``train_length_weeks``-Defaults (104 für h13, 130 für h52) sind so
gewählt, dass bei typischer Datenlage (~224 nutzbare Wochen nach
Warmup) genügend Samples entstehen und die letzten 1–2 Jahre
(fachlich relevantester Zeitraum) den Trainingsfokus bilden:

- h13: 104 - 52 - 13 + 1 = **40 Samples** (2 Jahre Training)
- h52: 130 - 52 - 52 + 1 = **27 Samples** (2,5 Jahre Training)

Liegt ``train_length_weeks`` über der tatsächlichen Serienlänge, wird
automatisch die gesamte Serie verwendet.

## Tuning-Metrik

Bei ``metric="combined"`` optimiert Optuna auf eine gewichtete
Kombination aus WAPE und sMAPE:

    score = alpha × mean(WAPE) + (1 − alpha) × mean(sMAPE)

WAPE (Weighted Absolute Percentage Error) gewichtet Fehler nach dem
Volumen der Komponente: ein 10%-Fehler auf KFZ_Vollkasko (500
Eingänge/Woche) wiegt mehr als 10% auf HUS_Rest (15 Eingänge/Woche).
sMAPE behandelt alle Komponenten gleich. Die Kombination mit
``alpha=0.7`` sorgt dafür, dass Optuna die großen Produkte
priorisiert (70% WAPE), aber volumenschwache Komponenten nicht komplett
ignoriert (30% sMAPE).

Über ``tuning_metric_alpha`` in der ``TuningConfig`` ist das
Mischungsverhältnis per YAML konfigurierbar. ``alpha=1.0`` ist reines
WAPE, ``alpha=0.0`` ist reines sMAPE.

**Wichtig bei vielen Komponenten (ORGA):** ``metric="combined"`` mittelt
WAPE und sMAPE *ungewichtet über die Komponenten*. Mit 32 × N_orga
Zielreihen dominieren dann die vielen kleinen, rauschigen ORGA-Zellen den
Score (kleine Zellen haben strukturell hohe WAPE/sMAPE), und Optuna
optimiert am Volumen vorbei. ``metric="combined_pooled"`` poolt
stattdessen volumengewichtet über die Komponenten:

    WAPE_pooled = Σ|Ist − Prognose| / Σ|Ist|   (über alle Komponenten)
    sMAPE_w     = volumengewichteter Mittelwert der Komponenten-sMAPE

Damit folgt der Score dem Volumen, was der Kapazitätsplanung entspricht.
Empfohlen, sobald ORGA aktiv ist.

## Optuna-Sampler

``create_study`` verwendet einen ``TPESampler`` mit
``multivariate=True`` (modelliert Wechselwirkungen zwischen den
Hyperparametern, z. B. ``learning_rate`` × ``hidden_size``, statt sie
unabhängig zu behandeln), ``group=True`` (saubere Behandlung der
optionalen/kategorialen Achsen) und einem festen ``seed``
(Reproduzierbarkeit; ``sampler_seed`` bzw. ``TftConfig.random_state``).
Beides ist über ``TuningConfig`` abschaltbar/einstellbar.

## Optuna-Pruning

Über ``use_pruning`` (Default ``True``) steuerbar. Aktiv nutzt die
Pipeline Optunas ``MedianPruner`` mit einem
``PyTorchLightningPruningCallback``:

1. **Aufwärmphase:** Die ersten 5 Trials laufen immer komplett
   (``n_startup_trials=5``).
2. **Epochen-Schwelle:** Innerhalb eines Trials wird frühestens ab
   Epoche 10 geprüft (``n_warmup_steps=10``).
3. **Pruning-Regel:** Ab Epoche 10 vergleicht der MedianPruner den
   ``train_loss`` des aktuellen Trials gegen den Median der
   abgeschlossenen Trials zum selben Epochenpunkt.

**Caveats (bewusst dokumentiert):** Das Pruning-Signal ist ``train_loss``
— kein Held-out-Signal; gepruned werden eher langsam konvergierende als
schlecht generalisierende Trials. Zudem startet der Epochenzähler im
mehrblockigen Backtest pro Block neu, was die Pruner-Schritte „sägezahnt".
Schlägt der Import von ``PyTorchLightningPruningCallback`` fehl (in
neueren Optuna-Versionen in ``optuna-integration`` ausgelagert), ist das
Pruning trotz konfiguriertem Pruner faktisch inaktiv — das wird jetzt per
Warnung sichtbar gemacht statt still verschluckt. Wer ein sauberes Signal
will, setzt ``use_pruning: false`` (volle Trials).

## Tuning-Suchraum (8 Parameter, optional 9.)

Optuna sampelt pro Trial folgende Hyperparameter:

=======================  ===========  ======================
Parameter                Skala        Default-Bereich (h13)
=======================  ===========  ======================
``train_length_weeks``   diskret      78 – 130, step 13
``hidden_size``          diskret      32 – 96, step 16
``hidden_continuous``    diskret      8 – 32, step 8
``lstm_layers``          diskret      1 – 3
``dropout``              kontinuierl. 0.1 – 0.5
``learning_rate``        log-kont.    5e-4 – 3e-3
``weight_decay``         log-kont.    1e-5 – 1e-2
``batch_size``           kategorisch  [8, 16, 24, 32]
=======================  ===========  ======================

Optional (leerer Default = aus): ``num_attention_heads`` als
kategorialer Suchraum (``num_attention_heads_choices``). Werte müssen
``hidden_size`` teilen; inkompatible Kombinationen werden als Trial
verworfen (``optuna.TrialPruned``). ``input_chunk_length`` ist bewusst
noch **nicht** im Suchraum (siehe unten).

Mit 40 Trials (h13) bzw. 25 Trials (h52) und multivariatem TPE wird der
Suchraum gezielt eingeengt.

## Held-out-Trennung (Tuning vs. Evaluation)

Um optimistisch verzerrte Metriken zu vermeiden, arbeitet das Tuning
auf einer **verkürzten Serie**. Die letzten ``validation_weeks``
(default: 52 Wochen = 1 Jahr) werden als Held-out abgeschnitten und
nie von Optuna gesehen:

::

    |──── Training ────|── Tuning-Val ──|── Held-out ──|
    |   (120 Wochen)   | (52 Wochen)    | (52 Wochen)  |
                        ↑ Optuna          ↑ Finale Eval
                          optimiert         (Optuna-frei)
                          hierauf

Bei 224 nutzbaren Wochen ergibt sich:

- **Training:** 120 Wochen (Position 0–120)
- **Tuning-Validierung:** 52 Wochen (Position 120–172)
- **Held-out Evaluation:** 52 Wochen (Position 172–224)

Die finale Evaluation — also die Metriken, die im Reporting
erscheinen — wird ausschließlich auf dem Held-out-Fenster berechnet.
Optuna hat diese Daten weder zum Trainieren noch zum Bewerten
gesehen. Dadurch sind die gemeldeten Metriken ein unverzerrtes Maß
für die Generalisierungsfähigkeit der getunten Parameter.

## Tuning- vs. Evaluations-Validierung (Held-out)

Die Tuning-Objective und die finale Backtest-Evaluation nutzen
**unterschiedliche** Validierungsfenster, um optimistisch verzerrte
Metriken zu vermeiden:

- **Tuning:** ``tuning_validation_weeks`` (Default: ``None`` →
  hartkodiert 52 Wochen in ``_optuna_objective_for_horizon``). Optuna
  sieht nur dieses Fenster, das innerhalb der bereits um das Held-out
  verkürzten Serie liegt.
- **Evaluation:** ``validation_weeks`` (volle Länge, Default 52). Dieses
  Fenster wird vorab als Held-out abgeschnitten und von Optuna nie
  zur Score-Bildung herangezogen.

Beispiel h13 (``validation_weeks=52``, ``tuning_validation_weeks=52``)
bei N nutzbaren Wochen:

::

    [..Training..][..Tuning-Fenster (52w)..][..Held-out (52w)..]
                  ← Optuna optimiert hier → ← nur finale Eval →
                  [N-104 .. N-52)            [N-52 .. N)

Beide Fenster sind disjunkt. Für **h52** funktioniert das ebenso: das
52-Wochen-Tuning-Fenster erzeugt genau einen Backtest-Block, der vor dem
52-Wochen-Held-out liegt — anders als eine frühere Design-Annahme nahelegte,
ist hier sehr wohl ein Held-out möglich.


## ORGA als vierte Dimension

Die Komponenten haben die Struktur ``KENNZAHL__PRODUKT__STATUS__ORGA``
(vier Teile, konfigurierbar über ``preprocessing.component_n_parts``).
ORGA wird als vierte statische Kovariate (One-Hot) ins TFT gegeben und
bekommt damit eine eigene Modell-Dynamik – im Gegensatz zu einer reinen
Anteilsverteilung. Die Zielreihe wächst dadurch von 32 auf 32 × N_orga
Komponenten.

Zwei Guardrails fangen die daraus folgende Skalierung ab:

- ``preprocessing.past_covariate_select`` (z. B. ``["AGG_PROD_"]``)
  beschränkt die Past-Kovariaten auf die Produkt-Aggregate. Ohne das
  würden ~32 × N_orga Roh-Spalten × Lag/MA/YoY-Features die
  Variable-Selection des TFT sprengen (Größenordnung 10.000+ Kanäle).
  Die Zell-Historie sieht das Modell weiterhin über die Zielreihe selbst.
- ``preprocessing.min_component_total`` verwirft praktisch leere
  KPSO-Zellen als eigene Zielreihe (sie bleiben in den Aggregaten). Der
  Schwellwert ist datenbasiert nach Sichtung des Volumen-Profils zu setzen.

Die Inferenz gleicht die Komponentenmenge über die in ``ModelArtifacts``
gespeicherten ``target_components`` / ``past_cov_components`` an den
Trainingsstand an (fehlende → 0, überzählige verworfen), damit ``predict``
nicht an einer durch veränderte Datenlage verschobenen Reihenbreite
scheitert.


## Tages-Disaggregation (Schreibpfad)

Das Modell bleibt bewusst wöchentlich (``W-SUN``) – die robuste
Granularität für Termineingänge. Für die Schreibtabelle wird jeder
Wochenwert über ein historisches Wochentagsprofil **summenerhaltend** auf
die sieben Tage seiner Woche verteilt (``forecasting/disaggregation.py``).
Wichtig:

- Die Disaggregation passiert **ausschließlich auf dem Schreibpfad**.
  Backtest, Metriken, archivierte Forecasts und die retrospektive
  Evaluation bleiben wöchentlich und ORGA-scharf (nicht tagesscharf).
- Die Wochentagsanteile werden je Komponente geschätzt, mit
  hierarchischem Fallback Komponente → Kennzahl → global → uniform.
- ``W-SUN``-Label liegen 7 Tage auseinander → die Tagesfenster überlappen
  nicht; jeder Tag gehört zu genau einer Woche (keine Doppelzählung).
- Der aktuelle, beim Resampling unvollständige Wochenrest (Tage zwischen
  letztem Ist-Datum und dem ersten prognostizierten Wochen-Label) wird vom
  Wochenmodell nicht prognostiziert und daher auch nicht geschrieben; auf
  Tagesebene wird zusätzlich strikt ``> data_end`` gefiltert.
- Schreiblast: Zeilenzahl ≈ 7 (Tage) × Komponenten × Wochen; das
  DELETE+INSERT (``executemany``) wächst entsprechend.
