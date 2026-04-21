from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import pandas as pd
from darts import TimeSeries

from forecasting import (
    load_config,
    run_full_job,
    RunReporter,
    update_retrospective_report,
)
from pluto_multivariate_repository import PlutoMultivariateRepository


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Config-Auflösung:
#   1. Env-Variable PLUTO_CONFIG (voller Pfad) – sofern gesetzt.
#   2. config.yaml neben diesem Skript.
#   3. Fallback: nur dataclass-Defaults.
# Zusätzlich überschreiben Env-Vars mit Präfix PLUTO__ punktuelle Werte.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _resolve_config_path() -> Path:
    env_path = os.environ.get("PLUTO_CONFIG")
    if env_path:
        return Path(env_path)
    return DEFAULT_CONFIG_PATH


def _combine_13_und_52_wochen(
    forecast_13: TimeSeries,
    forecast_52: TimeSeries,
    horizon_13: int = 13,
    total_horizon: int = 52,
) -> pd.DataFrame:
    """
    Kombiniert zwei Vorhersagen zu einer ``total_horizon``-Wochen-Prognose.

    Wochen 1 .. ``horizon_13`` stammen aus ``forecast_13``, die restlichen
    Wochen (``horizon_13 + 1`` .. ``total_horizon``) aus ``forecast_52``.
    Beide Prognosen müssen am gleichen Zeitstempel starten, sonst wäre
    das Zusammensetzen per Position fachlich falsch.

    Parameters
    ----------
    forecast_13, forecast_52
        Prognosen des 13- bzw. 52-Wochen-Modells. ``forecast_13`` muss
        mindestens ``horizon_13`` Punkte haben, ``forecast_52`` mindestens
        ``total_horizon``.
    horizon_13
        Anzahl Wochen, die aus ``forecast_13`` übernommen werden.
    total_horizon
        Gesamtlänge der kombinierten Prognose.

    Returns
    -------
    pd.DataFrame
        Kombinierter Forecast mit Zeitstempel-Index aus ``forecast_52``
        und den Komponenten-Spalten aus ``forecast_52``.

    Raises
    ------
    ValueError
        Wenn die Prognosen zu kurz sind oder die ersten
        ``horizon_13`` Zeitstempel nicht übereinstimmen.
    """
    df_13 = forecast_13.to_dataframe()
    df_52 = forecast_52.to_dataframe()

    if len(df_13) < horizon_13:
        raise ValueError(
            f"forecast_13 hat nur {len(df_13)} Punkte, erwartet {horizon_13}."
        )
    if len(df_52) < total_horizon:
        raise ValueError(
            f"forecast_52 hat nur {len(df_52)} Punkte, erwartet {total_horizon}."
        )

    # Beide Prognosen sollten am gleichen Zeitstempel starten; sonst ist das
    # Zusammensetzen per iloc inhaltlich falsch.
    if not df_13.index[:horizon_13].equals(df_52.index[:horizon_13]):
        raise ValueError(
            "Zeitstempel-Indizes von forecast_13 und forecast_52 stimmen in den "
            f"ersten {horizon_13} Wochen nicht überein – das Zusammenführen per "
            "Position wäre fehlerhaft."
        )

    # Spaltenmengen abgleichen (Reihenfolge aus df_52 übernehmen, fehlende Spalten
    # werden mit NaN aufgefüllt; Überschneidungen werden genutzt).
    common_cols = [c for c in df_52.columns if c in df_13.columns]
    if len(common_cols) != len(df_52.columns):
        missing = set(df_52.columns) - set(df_13.columns)
        logger.warning(
            "Folgende Komponenten fehlen in forecast_13 und werden aus "
            "forecast_52 übernommen: %s",
            sorted(missing),
        )

    idx_52 = df_52.index[:total_horizon]
    df_combined = pd.DataFrame(index=idx_52, columns=df_52.columns, dtype=float)

    df_combined.iloc[0:horizon_13] = (
        df_13.reindex(columns=df_52.columns).iloc[0:horizon_13].values
    )
    df_combined.iloc[horizon_13:total_horizon] = (
        df_52.iloc[horizon_13:total_horizon].values
    )

    return df_combined


def run_pluto_multivariate_forecast_job() -> Dict[int, Dict[str, object]]:
    """
    Führt den vollständigen multivariaten Prognose-Job für PLUTO aus.
    """
    config_path = _resolve_config_path()
    if config_path.exists():
        logger.info("Lade Konfiguration aus %s", config_path)
    else:
        logger.info("Keine Config-Datei unter %s – nutze dataclass-Defaults.", config_path)
    cfg = load_config(yaml_path=config_path)

    # Runs-Verzeichnis: persistenter Speicherort für Metriken, Forecasts,
    # Plots und getunete Parameter. Liegt bewusst AUSSERHALB des Repo-Klons,
    # damit es beim erneuten Klonen nicht gelöscht wird.
    # Default "runs" (lokale Entwicklung), in Produktion per Env-Variable
    # auf einen persistenten Pfad gesetzt (z. B. /mnt/pluto-runs).
    runs_dir = os.environ.get("PLUTO_RUNS_DIR", "runs")
    logger.info("Runs-Verzeichnis: %s", runs_dir)

    # tuned_params_dir auf das runs-Verzeichnis setzen, damit getunete
    # Parameter automatisch gespeichert/geladen werden (sofern nicht
    # bereits in der Config explizit gesetzt).
    if cfg.tuned_params_dir is None:
        cfg.tuned_params_dir = runs_dir

    repo = PlutoMultivariateRepository()

    try:
        df_daily = repo.read_timeseries()
        if df_daily.empty:
            logger.warning("Keine Daten aus DB2 geladen. Abbruch.")
            return {}

        results = run_full_job(df_daily, cfg=cfg, logger=None)

        # Reporting: Metriken, Forecasts und Diagnostic-Plots je Horizont
        # archivieren. Die Artefakte wachsen über Läufe hinweg an.
        reporter = RunReporter(output_dir=runs_dir)
        for horizon, block in results.items():
            true_ts = block["true_ts"]
            run_dir = reporter.report_horizon(
                horizon_name=block["artifacts"].hcfg.name,
                metrics_df=block["metrics"],
                backtest_ts=block["backtest"],
                future_forecast_ts=block["forecast"],
                true_ts=true_ts,
                cfg=cfg,
                data_end_date=true_ts.end_time(),
            )
            logger.info("Run-Artefakte für Horizont %s unter %s", horizon, run_dir)

        # Retrospektive Evaluation: archivierte Zukunfts-Prognosen früherer
        # Läufe gegen die jetzt vorliegenden Ist-Werte prüfen.
        first_key = next(iter(results))
        actuals_df = results[first_key]["true_ts"].to_dataframe()
        retro_df = update_retrospective_report(actuals_df, runs_dir)
        if retro_df is not None and not retro_df.empty:
            logger.info(
                "Retrospektive Evaluation: %d Vergleichspunkte über %d frühere Runs.",
                len(retro_df),
                retro_df["run_timestamp"].nunique(),
            )
        else:
            logger.info("Keine früheren Forecasts zur retrospektiven Evaluation vorhanden.")

        if 13 not in results or 52 not in results:
            raise RuntimeError(
                "Ergebnisse für 13- oder 52-Wochen-Horizont fehlen in der "
                "Forecasting-Pipeline."
            )

        forecast_13: TimeSeries = results[13]["forecast"]
        forecast_52: TimeSeries = results[52]["forecast"]

        df_combined = _combine_13_und_52_wochen(
            forecast_13=forecast_13,
            forecast_52=forecast_52,
            horizon_13=13,
            total_horizon=52,
        )

        repo.write_forecast(df_combined)

        logger.info("PLUTO multivariate forecast job completed successfully.")

        return results
    finally:
        repo.close_connection()


if __name__ == "__main__":
    run_pluto_multivariate_forecast_job()
