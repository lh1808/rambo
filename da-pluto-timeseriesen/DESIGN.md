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

## Optuna-Pruning

Aussichtslose Trials werden automatisch abgebrochen. Die Pipeline
nutzt Optunas ``MedianPruner`` in Kombination mit einem
``PyTorchLightningPruningCallback``:

1. **Aufwärmphase:** Die ersten 5 Trials laufen immer komplett
   (``n_startup_trials=5``). Ohne Vergleichsdaten kann der Pruner
   nicht sinnvoll entscheiden.
2. **Epochen-Schwelle:** Innerhalb eines Trials wird frühestens ab
   Epoche 10 geprüft (``n_warmup_steps=10``). Frühe Epochen sind
   zu instabil für Vergleiche.
3. **Pruning-Regel:** Ab Epoche 10 vergleicht der MedianPruner den
   ``train_loss`` des aktuellen Trials gegen den Median der
   abgeschlossenen Trials zum selben Epochenpunkt. Liegt der aktuelle
   Loss über dem Median, wird der Trial abgebrochen.

Effekt: ~30-50% Rechenzeit-Ersparnis, weil schlechte
Hyperparameter-Kombinationen nicht die vollen 50 Epochen durchlaufen.
Die Pruning-Statistik (Anzahl vollständig / gepruned) wird im
Tuning-Log ausgegeben.

## Tuning-Suchraum (8 Parameter)

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

Mit 40 Trials (h13) bzw. 25 Trials (h52) und Pruning werden ca.
25-30 vollständige Evaluationen durchgeführt — genug für Optunas
TPE-Sampler, um den Suchraum gezielt einzuengen.

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

- **Tuning:** ``tuning_validation_weeks`` (Default: ``validation_weeks
  − horizon_weeks``). Optuna sieht nur diesen kürzeren Zeitraum.
- **Evaluation:** ``validation_weeks`` (volle Länge). Die Differenz
  zum Tuning-Fenster ist das Held-out-Segment.

Beispiel h13 (``validation_weeks=52``, ``horizon_weeks=13``):

::

    [...Training...][...Tuning-Fenster (39w)...][..Held-out (13w)..]
                    ← Optuna optimiert hier →   ← nur Evaluation →

Für h52 ist kein Held-out möglich (``52 − 52 = 0``), weil das
gesamte Validierungsfenster für einen einzigen Block benötigt wird.
In diesem Fall wird automatisch auf das volle Fenster zurückgefallen.
