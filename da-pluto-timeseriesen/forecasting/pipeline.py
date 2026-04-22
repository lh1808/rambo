"""
End-to-End-Pipeline: Preprocessing → optionales Tuning → Backtest →
Final-Training → Prognose in die Zukunft.

Einstiegspunkt ist :func:`run_full_job`, das für alle in
``GlobalConfig.horizons`` definierten Prognosehorizonte
:func:`train_and_evaluate_for_horizon` und anschließend
:func:`forecast_with_artifacts` aufruft.

Design-Entscheidungen
---------------------
- **Leakage-Vermeidung im Backtest.** Transformer (Winsorizer,
  Yeo-Johnson, Skalierung) werden pro Backtest-Block lokal gefittet; sie
  sehen ausschließlich das aktuelle Trainingsfenster. Umgesetzt über den
  ``transforms_builder``-Parameter von :func:`rolling_block_forecast`.
- **Statische Kovariaten konsistent.** Der im Training gefittete
  ``StaticCovariatesTransformer`` wird in :class:`ModelArtifacts`
  mitgeführt und in der Inferenz (:func:`forecast_with_artifacts`) zum
  Transformieren der aktuellen Daten wiederverwendet – damit bleibt das
  One-Hot-Schema identisch.
- **Future-Kovariaten in die Zukunft gezogen.** Beim Bau der
  Wochen-Kovariaten wird ``future_weeks=hcfg.horizon_weeks`` gesetzt,
  damit ``model.predict(n=horizon, future_covariates=...)`` durchläuft.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import optuna
import pandas as pd
from darts import TimeSeries

from .config import GlobalConfig, HorizonConfig, TftConfig, TuningConfig
from .covariates import build_weekly_covariates
from .evaluation import evaluate_multivariate
from .features import (
    add_static_covariates,
    build_target_and_past_covariates,
    extract_static_for_component,
    preprocess_dataframe,
)
from .model import TransformArtifacts, build_tft, build_transformers
from .reporting import load_tuned_params, save_tuned_params
from .utils import rolling_block_forecast, slice_from


@dataclass
class ModelArtifacts:
    """
    Artefakte eines für einen bestimmten Horizont trainierten Modells.

    Attributes
    ----------
    horizon_weeks
        Prognosehorizont in Wochen (entspricht ``hcfg.horizon_weeks``).
    hcfg
        Horizont-Konfiguration, mit der das Modell gebaut wurde.
    model
        Das trainierte darts-Modell (typischerweise :class:`TFTModel`).
    transforms
        Die in der Finalphase gefitteten Transformer (Y-Pipeline +
        X-Scaler) in Original-Skala.
    static_transformer
        Der gefittete ``StaticCovariatesTransformer`` für konsistentes
        Encoding der statischen Kategorien in der Inferenz.
    train_length_weeks
        Effektiv verwendete Trainingsfensterlänge (ggf. nach Tuning
        angepasst).
    tft_cfg
        Effektiv verwendete TFT-Hyperparameter (ggf. vom Tuning
        überschrieben).
    """

    horizon_weeks: int
    hcfg: HorizonConfig
    model: object
    transforms: TransformArtifacts
    static_transformer: object
    train_length_weeks: int
    tft_cfg: TftConfig


def _build_tuned_tft_cfg_from_trial(
    trial: optuna.Trial,
    base_cfg: TftConfig,
    tuning_cfg: TuningConfig,
) -> TftConfig:
    """
    Leitet aus einem Optuna-Trial und einer Tuning-Konfiguration eine
    :class:`TftConfig` ab. Sampelt Hidden-Size, Continuous-Size,
    LSTM-Layers, Dropout, Lernrate, Weight-Decay und optional Batch-Size.
    """
    cfg = deepcopy(base_cfg)

    cfg.hidden_size = trial.suggest_int(
        "hidden_size",
        tuning_cfg.hidden_size_min,
        tuning_cfg.hidden_size_max,
        step=tuning_cfg.hidden_size_step,
    )
    cfg.hidden_continuous_size = trial.suggest_int(
        "hidden_continuous_size",
        tuning_cfg.hidden_continuous_size_min,
        tuning_cfg.hidden_continuous_size_max,
        step=tuning_cfg.hidden_continuous_size_step,
    )
    cfg.lstm_layers = trial.suggest_int(
        "lstm_layers",
        tuning_cfg.lstm_layers_min,
        tuning_cfg.lstm_layers_max,
    )
    cfg.dropout = trial.suggest_float(
        "dropout",
        tuning_cfg.dropout_min,
        tuning_cfg.dropout_max,
    )
    cfg.learning_rate = trial.suggest_float(
        "learning_rate",
        tuning_cfg.learning_rate_min,
        tuning_cfg.learning_rate_max,
        log=True,
    )
    cfg.weight_decay = trial.suggest_float(
        "weight_decay",
        tuning_cfg.weight_decay_min,
        tuning_cfg.weight_decay_max,
        log=True,
    )
    if tuning_cfg.batch_size_choices:
        cfg.batch_size = trial.suggest_categorical(
            "batch_size",
            tuning_cfg.batch_size_choices,
        )

    return cfg


def _optuna_objective_for_horizon(
    trial: optuna.Trial,
    ts_y_static: TimeSeries,
    ts_X: TimeSeries,
    cov_ts: TimeSeries,
    hcfg: HorizonConfig,
    cfg: GlobalConfig,
    tuning_cfg: TuningConfig,
) -> float:
    """
    Optuna-Objective für einen Prognosehorizont.

    Das Tuning arbeitet auf einer **verkürzten Serie** — die letzten
    ``hcfg.validation_weeks`` Wochen (das Held-out-Fenster für die
    finale Evaluation) werden abgeschnitten. Optuna sieht diese Daten
    nie, wodurch die finale Evaluation unverzerrte Metriken liefert.

    Innerhalb der verkürzten Serie werden die letzten
    ``tuning_validation_weeks`` als Tuning-Validierung verwendet.

    ::

        |---- Training ----|-- Tuning-Val --|-- Held-out (abgeschnitten) --|
                            ↑ Optuna sieht   ↑ Finale Eval
                              nur das hier     (nie von Optuna gesehen)

    Returns
    -------
    float
        Score des Trials. ``float("inf")`` bei zu wenigen Daten.
    """
    sampled_train = trial.suggest_int(
        "train_length_weeks",
        tuning_cfg.train_length_min,
        tuning_cfg.train_length_max,
        step=tuning_cfg.train_length_step,
    )
    train_length = max(
        sampled_train,
        hcfg.input_chunk_length + 1,
        hcfg.output_chunk_length + 1,
    )

    tuned_tft_cfg = _build_tuned_tft_cfg_from_trial(
        trial=trial,
        base_cfg=cfg.tft,
        tuning_cfg=tuning_cfg,
    )

    # Optuna-Pruning: bricht aussichtslose Trials nach wenigen Epochen ab.
    pruning_callbacks = []
    try:
        from optuna.integration import PyTorchLightningPruningCallback
        pruning_callbacks.append(
            PyTorchLightningPruningCallback(trial, monitor="train_loss")
        )
    except ImportError:
        pass

    def model_builder():
        return build_tft(hcfg, tuned_tft_cfg, extra_callbacks=pruning_callbacks)

    # --- Held-out-Trennung ---
    # Die letzten validation_weeks werden abgeschnitten. Optuna arbeitet
    # nur auf dem verbleibenden Prefix der Serie.
    held_out_weeks = hcfg.validation_weeks
    if len(ts_y_static) <= held_out_weeks:
        return float("inf")

    cutoff_idx = len(ts_y_static) - held_out_weeks
    cutoff_ts = ts_y_static.time_index[cutoff_idx]
    ts_y_tuning = ts_y_static.slice(ts_y_static.start_time(), cutoff_ts)
    ts_X_tuning = ts_X.slice(ts_X.start_time(), cutoff_ts)
    # Future-Kovariaten müssen bis zum Ende des Tuning-Horizonts reichen
    cov_tuning = cov_ts.slice(cov_ts.start_time(), cutoff_ts)

    # Tuning-Validierungsfenster innerhalb der verkürzten Serie
    tuning_val = tuning_cfg.tuning_validation_weeks
    if tuning_val is None:
        tuning_val = 52  # Default: 1 Jahr

    if len(ts_y_tuning) <= tuning_val:
        return float("inf")

    test_start = ts_y_tuning.time_index[-tuning_val]

    res = rolling_block_forecast(
        model_builder=model_builder,
        y=ts_y_tuning,
        past_cov=ts_X_tuning,
        future_cov=cov_tuning,
        test_start=test_start,
        train_length=train_length,
        horizon=hcfg.horizon_weeks,
        stride=hcfg.stride_weeks,
        fit_kwargs={"verbose": False},
        predict_kwargs={},
        merge="keep_last",
        required_input_chunk_length=hcfg.input_chunk_length + hcfg.output_chunk_length,
        verbose=False,
        transforms_builder=build_transformers,
        window_mode=hcfg.window_mode,
    )

    backtest = res["merged"]
    if backtest is None:
        return float("inf")

    metrics_df = evaluate_multivariate(
        true_ts=ts_y_static,
        pred_ts=backtest,
        output_chunk_length=hcfg.horizon_weeks,
        return_df=True,
    )

    return _compute_tuning_score(metrics_df, tuning_cfg)


def _compute_tuning_score(
    metrics_df: pd.DataFrame,
    tuning_cfg: TuningConfig,
) -> float:
    """
    Berechnet den Tuning-Score aus dem Metriken-DataFrame.

    Bei ``metric="combined"`` wird eine gewichtete Kombination aus WAPE
    und sMAPE berechnet:

        score = alpha × mean(WAPE) + (1 − alpha) × mean(sMAPE)

    Das sorgt dafür, dass Optuna volumenstarke Produkte priorisiert
    (WAPE-Anteil), aber volumenschwache Komponenten nicht komplett
    ignoriert (sMAPE-Anteil).

    Bei allen anderen Metriken wird der einfache Mittelwert über alle
    Komponenten zurückgegeben.
    """
    metric_name = tuning_cfg.metric

    if metric_name == "combined":
        alpha = tuning_cfg.tuning_metric_alpha
        if "wape" not in metrics_df.columns or "smape" not in metrics_df.columns:
            raise ValueError(
                "metric='combined' benötigt die Spalten 'wape' und 'smape' "
                "im Metriken-DataFrame."
            )
        wape_mean = float(metrics_df["wape"].mean())
        smape_mean = float(metrics_df["smape"].mean())
        return alpha * wape_mean + (1 - alpha) * smape_mean

    if metric_name not in metrics_df.columns:
        raise ValueError(
            f"Metrik '{metric_name}' nicht im Metrics-DataFrame vorhanden. "
            f"Verfügbar: {list(metrics_df.columns)}"
        )

    return float(metrics_df[metric_name].mean())


def train_and_evaluate_for_horizon(
    df_raw: pd.DataFrame,
    hcfg: HorizonConfig,
    cfg: Optional[GlobalConfig] = None,
    logger: Optional[Callable[[str, Dict], None]] = None,
) -> Tuple[ModelArtifacts, pd.DataFrame]:
    """
    Trainiert und evaluiert ein Modell für einen Prognosehorizont.

    Ablauf:

    1. Preprocessing des Roh-DataFrames.
    2. Aufbau von Ziel- und Past-Kovariaten sowie Static Covariates.
    3. Aufbau der Future-Kovariaten (mit ``future_weeks=horizon``).
    4. Optional: Optuna-Tuning von Trainingsfensterlänge und
       TFT-Hyperparametern.
    5. Rolling-Backtest mit leakage-freier Pro-Block-Transformation.
    6. Auswertung per :func:`evaluate_multivariate`.
    7. Finales Training des Modells auf dem letzten
       ``effective_train_length_weeks``-Fenster.

    Parameters
    ----------
    df_raw
        Roh-DataFrame.
    hcfg
        Horizont-Konfiguration.
    cfg
        Globale Konfiguration (Default: ``GlobalConfig()``).
    logger
        Optionaler Callback ``(name, payload_dict) -> None``, z. B. für
        MLflow-Integration.

    Returns
    -------
    (ModelArtifacts, pd.DataFrame, TimeSeries, TimeSeries)
        Trainierte Artefakte, Metriken-DataFrame des Backtests,
        Backtest-Prognose (in Originaleinheit) und Grundwahrheit
        (``ts_y_static``).

    Raises
    ------
    RuntimeError
        Wenn das Validierungsfenster zu groß für die Reihe ist oder der
        Backtest keine Blöcke produziert.
    """
    cfg = cfg or GlobalConfig()

    df = preprocess_dataframe(df_raw, cfg)

    ts_y, ts_X, _ = build_target_and_past_covariates(df, cfg)
    ts_y_static, static_transformer = add_static_covariates(ts_y, cfg)
    cov_ts = build_weekly_covariates(df, cfg, future_weeks=hcfg.horizon_weeks)

    tuning_cfg = cfg.tuning.get(hcfg.horizon_weeks, TuningConfig())

    effective_train_length_weeks = hcfg.train_length_weeks
    effective_tft_cfg = deepcopy(cfg.tft)

    if tuning_cfg.enabled:
        # MedianPruner: bricht Trials ab, deren train_loss nach n_warmup
        # Epochen schlechter als der Median der bisherigen Trials ist.
        # Spart ~30-50% Rechenzeit, weil aussichtslose Kombinationen
        # nicht die vollen Epochen durchlaufen.
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,     # erste 5 Trials nie prunen (Aufwärm)
            n_warmup_steps=10,      # erst ab Epoche 10 prunen
            interval_steps=1,       # jede Epoche prüfen
        )
        study = optuna.create_study(
            direction=tuning_cfg.direction,
            pruner=pruner,
        )
        study.optimize(
            lambda trial: _optuna_objective_for_horizon(
                trial=trial,
                ts_y_static=ts_y_static,
                ts_X=ts_X,
                cov_ts=cov_ts,
                hcfg=hcfg,
                cfg=cfg,
                tuning_cfg=tuning_cfg,
            ),
            n_trials=tuning_cfg.n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        effective_train_length_weeks = max(
            best_params["train_length_weeks"],
            hcfg.input_chunk_length + 1,
            hcfg.output_chunk_length + 1,
        )
        effective_tft_cfg = _build_tuned_tft_cfg_from_trial(
            trial=optuna.trial.FixedTrial(best_params),
            base_cfg=cfg.tft,
            tuning_cfg=tuning_cfg,
        )

        if logger is not None:
            n_pruned = len([t for t in study.trials
                           if t.state == optuna.trial.TrialState.PRUNED])
            n_complete = len([t for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE])
            logger(
                f"tuning_{hcfg.name}",
                {
                    "best_params": best_params,
                    "effective_train_length_weeks": effective_train_length_weeks,
                    "best_score": study.best_value,
                    "n_trials_complete": n_complete,
                    "n_trials_pruned": n_pruned,
                },
            )

        # Getunete Parameter persistieren, damit spätere Läufe ohne
        # erneutes Tuning darauf zurückgreifen können.
        if cfg.tuned_params_dir:
            save_tuned_params(
                runs_dir=cfg.tuned_params_dir,
                horizon_name=hcfg.name,
                train_length_weeks=effective_train_length_weeks,
                tft_cfg=effective_tft_cfg,
                best_params=best_params,
                best_score=study.best_value,
            )

    elif cfg.tuned_params_dir:
        # Tuning deaktiviert, aber getunete Parameter vorhanden → laden.
        saved = load_tuned_params(cfg.tuned_params_dir, hcfg.name)
        if saved is not None:
            effective_train_length_weeks = saved["train_length_weeks"]
            tft_dict = saved.get("tft", {})
            for key, value in tft_dict.items():
                if hasattr(effective_tft_cfg, key):
                    setattr(effective_tft_cfg, key, value)
            if logger is not None:
                logger(
                    f"loaded_tuned_{hcfg.name}",
                    {
                        "source": str(cfg.tuned_params_dir),
                        "train_length_weeks": effective_train_length_weeks,
                        "tuning_timestamp": saved.get("tuning_timestamp", "?"),
                    },
                )

    # Evaluations-Backtest mit leakage-freier Pro-Block-Transformation.
    def model_builder_eval():
        return build_tft(hcfg, effective_tft_cfg)

    if len(ts_y_static) <= hcfg.validation_weeks:
        raise RuntimeError("Zu wenige Beobachtungen für das gewählte Validierungsfenster.")

    test_start = ts_y_static.time_index[-hcfg.validation_weeks]

    res = rolling_block_forecast(
        model_builder=model_builder_eval,
        y=ts_y_static,
        past_cov=ts_X,
        future_cov=cov_ts,
        test_start=test_start,
        train_length=effective_train_length_weeks,
        horizon=hcfg.horizon_weeks,
        stride=hcfg.stride_weeks,
        fit_kwargs={"verbose": False},
        predict_kwargs={},
        merge="keep_last",
        required_input_chunk_length=hcfg.input_chunk_length + hcfg.output_chunk_length,
        verbose=True,
        transforms_builder=build_transformers,
        window_mode=hcfg.window_mode,
    )

    backtest = res["merged"]
    if backtest is None:
        raise RuntimeError("rolling_block_forecast hat keine Vorhersageblöcke erzeugt.")

    metrics_df = evaluate_multivariate(
        true_ts=ts_y_static,
        pred_ts=backtest,
        output_chunk_length=hcfg.horizon_weeks,
        return_df=True,
    )
    metrics_df["model_name"] = hcfg.name

    if logger is not None:
        logger(f"metrics_{hcfg.name}", metrics_df.to_dict(orient="list"))

    # Finales Modell: Transformer und Modell werden auf dem letzten
    # ``effective_train_length_weeks``-Fenster trainiert (kein Leakage, da
    # der Cutoff bewusst nur den Trainingsteil umfasst).
    transforms_final = build_transformers()
    y_pipeline_final = transforms_final.y_pipeline
    x_scaler_final = transforms_final.x_scaler

    if len(ts_y_static) <= effective_train_length_weeks:
        cutoff = ts_y_static.start_time()
    else:
        cutoff = ts_y_static.time_index[-effective_train_length_weeks]

    y_train = slice_from(ts_y_static, cutoff)
    X_train = slice_from(ts_X, cutoff)
    cov_train = slice_from(cov_ts, cutoff)

    min_train_length = hcfg.input_chunk_length + hcfg.output_chunk_length
    if len(y_train) < min_train_length:
        raise RuntimeError(
            f"Trainingserie für finales {hcfg.name}-Modell zu kurz: "
            f"{len(y_train)} Wochen verfügbar, aber mindestens "
            f"{min_train_length} benötigt (input_chunk_length="
            f"{hcfg.input_chunk_length} + output_chunk_length="
            f"{hcfg.output_chunk_length}). "
            f"Entweder mehr Daten bereitstellen oder "
            f"input_chunk_length/output_chunk_length reduzieren."
        )

    y_train_transformed = y_pipeline_final.fit_transform(y_train)
    X_train_transformed = x_scaler_final.fit_transform(X_train)

    final_model = build_tft(hcfg, effective_tft_cfg)
    final_model.fit(
        series=y_train_transformed,
        past_covariates=X_train_transformed,
        future_covariates=cov_train,
        verbose=False,
    )

    artifacts = ModelArtifacts(
        horizon_weeks=hcfg.horizon_weeks,
        hcfg=hcfg,
        model=final_model,
        transforms=transforms_final,
        static_transformer=static_transformer,
        train_length_weeks=effective_train_length_weeks,
        tft_cfg=effective_tft_cfg,
    )

    return artifacts, metrics_df, backtest, ts_y_static


def forecast_with_artifacts(
    df_raw: pd.DataFrame,
    artifacts: ModelArtifacts,
    cfg: Optional[GlobalConfig] = None,
) -> TimeSeries:
    """
    Erzeugt eine Vorhersage auf Basis der aktuellen Eingangsdaten und eines
    bereits trainierten Modells.

    Der Lookback entspricht dem im Training verwendeten
    ``train_length_weeks``. Die Zielreihe wird explizit als ``series=`` an
    ``model.predict`` übergeben, damit bei aktualisierten Eingangsdaten
    tatsächlich der neueste Stand zur Prognose verwendet wird und nicht
    die modellintern eingefrorene Trainingsserie.

    Parameters
    ----------
    df_raw
        Aktueller Roh-DataFrame (i. d. R. derselbe, mit dem trainiert
        wurde; kann aber neuere Wochen enthalten).
    artifacts
        Modellartefakte aus :func:`train_and_evaluate_for_horizon`.
    cfg
        Globale Konfiguration (Default: ``GlobalConfig()``).

    Returns
    -------
    TimeSeries
        Prognose in Originaleinheit, Länge = ``hcfg.horizon_weeks``.
    """
    cfg = cfg or GlobalConfig()
    hcfg = artifacts.hcfg

    df = preprocess_dataframe(df_raw, cfg)
    ts_y, ts_X, _ = build_target_and_past_covariates(df, cfg)

    # Static Covariates über den im Training gefitteten Transformer an
    # die aktuelle Zielreihe hängen – das stellt sicher, dass die
    # One-Hot-Kategorien zum Modellstand passen.
    prep_cfg = cfg.preprocessing
    static_df = pd.DataFrame(
        [extract_static_for_component(c, prep_cfg) for c in ts_y.components],
        columns=["Kennzahl", "Sparte", "Status"],
        index=pd.Index(ts_y.components),
    )
    ts_y_with_static = ts_y.with_static_covariates(static_df)
    ts_y_static = artifacts.static_transformer.transform(ts_y_with_static)

    cov_ts_full = build_weekly_covariates(df, cfg, future_weeks=hcfg.horizon_weeks)

    effective_train_length_weeks = artifacts.train_length_weeks
    cutoff_pos = max(0, len(ts_y_static) - effective_train_length_weeks)
    cutoff = ts_y_static.time_index[cutoff_pos]

    y_recent = slice_from(ts_y_static, cutoff)
    X_recent = slice_from(ts_X, cutoff)
    cov_recent = slice_from(cov_ts_full, cutoff)

    y_pipeline = artifacts.transforms.y_pipeline
    x_scaler = artifacts.transforms.x_scaler

    y_recent_transformed = y_pipeline.transform(y_recent)
    X_recent_transformed = x_scaler.transform(X_recent)

    y_pred_transformed = artifacts.model.predict(
        n=hcfg.horizon_weeks,
        series=y_recent_transformed,
        past_covariates=X_recent_transformed,
        future_covariates=cov_recent,
    )

    return y_pipeline.inverse_transform(y_pred_transformed)


def run_full_job(
    df_raw: pd.DataFrame,
    cfg: Optional[GlobalConfig] = None,
    logger: Optional[Callable[[str, Dict], None]] = None,
) -> Dict[int, Dict[str, object]]:
    """
    Führt Training, Evaluation und Prognose für alle in der globalen
    Konfiguration definierten Prognosehorizonte aus.

    Parameters
    ----------
    df_raw
        Roh-DataFrame.
    cfg
        Globale Konfiguration (Default: ``GlobalConfig()``).
    logger
        Optionaler Callback ``(name, payload_dict) -> None`` für
        Tuning-/Metriken-Logs.

    Returns
    -------
    dict
        Mapping ``horizon_weeks -> {"artifacts", "metrics", "forecast",
        "backtest", "true_ts"}``.
    """
    cfg = cfg or GlobalConfig()
    results: Dict[int, Dict[str, object]] = {}

    for horizon, hcfg in cfg.horizons.items():
        artifacts, metrics_df, backtest, true_ts = train_and_evaluate_for_horizon(
            df_raw=df_raw,
            hcfg=hcfg,
            cfg=cfg,
            logger=logger,
        )
        forecast = forecast_with_artifacts(
            df_raw=df_raw,
            artifacts=artifacts,
            cfg=cfg,
        )
        results[horizon] = {
            "artifacts": artifacts,
            "metrics": metrics_df,
            "forecast": forecast,
            "backtest": backtest,
            "true_ts": true_ts,
        }

    return results
