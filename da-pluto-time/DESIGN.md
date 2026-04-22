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

Die ``train_length_weeks``-Defaults (208 für h13, 260 für h52) sind so
gewählt, dass bei typischer Datenlage (~224 nutzbare Wochen nach
Warmup) genügend Samples für ein sinnvolles Training entstehen:

- h13: min(208, 224) - 52 - 13 + 1 = 144 Samples
- h52: min(260, 224) - 104 - 52 + 1 = 69 Samples

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
