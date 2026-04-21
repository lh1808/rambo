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

## Prognose-Schreiben per Staging-Swap

``PlutoMultivariateRepository.write_forecast`` befüllt zunächst eine
Staging-Tabelle ``<target>_STAGING`` per ``executemany`` und tauscht sie
dann in einer zweiten Transaktion (``DELETE FROM <target>;
INSERT INTO <target> SELECT … FROM <staging>``) gegen den alten Stand.
Beide Teile sind rollback-fähig; ein ``TRUNCATE`` wäre es in DB2 nicht.
Für Reader bleibt die Zieltabelle bis zum finalen Commit auf dem alten
Stand sichtbar.

Die Staging-Tabelle wird beim ersten Lauf idempotent per SYSCAT-Check
angelegt (``CREATE TABLE … LIKE``). Ohne `CREATE`-Recht muss sie der
DBA einmalig als strukturelle Kopie anlegen (SQL-Vorlage siehe README).

## PyTorch-Lightning-Defaults

``build_tft`` setzt in ``pl_trainer_kwargs`` ``enable_progress_bar``,
``enable_model_summary`` und ``logger`` auf ``False``. Ohne diese
Defaults legt jeder ``fit()``-Aufruf einen eigenen TensorBoard-Log-Ordner
unter ``./darts_logs/`` an, was in Kombination mit Backtest +
Optuna-Tuning schnell in hunderten Ordnern mündet. Wer punktuell
Feedback braucht, kann ``verbose=True`` an ``fit()`` / ``predict()``
reichen.
