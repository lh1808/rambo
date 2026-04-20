ERROR:darts.timeseries:ValueError: The two timestamps provided to slice() have to be of the same type.
INFO:pluto_multivariate_repository:DB2 connection successfully closed.
Traceback (most recent call last):
  File "/mnt/da-pluto-timeseries/pluto_forecast_job.py", line 98, in <module>
    run_pluto_multivariate_forecast_job()
  File "/mnt/da-pluto-timeseries/pluto_forecast_job.py", line 73, in run_pluto_multivariate_forecast_job
    results = run_full_job(df_daily, cfg=cfg, logger=None)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 379, in run_full_job
    artifacts, metrics_df = train_and_evaluate_for_horizon(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 247, in train_and_evaluate_for_horizon
    res = rolling_block_forecast(
          ^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/utils.py", line 58, in rolling_block_forecast
    past_future = past_cov.slice(train_start, None)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/timeseries.py", line 2619, in slice
    raise_log(
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/logging.py", line 132, in raise_log
    raise exception
ValueError: The two timestamps provided to slice() have to be of the same type.




  from dataclasses import dataclass
from typing import Dict, Callable, Optional
from copy import deepcopy

import optuna
import pandas as pd
from darts import TimeSeries

from .config import GlobalConfig, HorizonConfig, TuningConfig, TftConfig
from .features import (
    preprocess_dataframe,
    build_target_and_past_covariates,
    add_static_covariates,
)
from .covariates import build_weekly_covariates
from .model import build_transformers, build_tft, TransformArtifacts
from .evaluation import evaluate_multivariate
from .utils import rolling_block_forecast


@dataclass
class ModelArtifacts:
    """
    Artefakte eines trainierten Modells für einen bestimmten Horizont.
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
    Leitet aus einem Optuna-Trial und der TuningConfig einen TftConfig ab.
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
    Objective-Funktion für Optuna.

    Es werden train_length_weeks und zentrale TFT-Hyperparameter variiert.
    Bewertet wird die Backtest-Performance im rolling_block_forecast.
    """
    transforms_eval = build_transformers()
    y_pipeline_eval = transforms_eval.y_pipeline
    x_scaler_eval = transforms_eval.x_scaler

    y_transformed = y_pipeline_eval.fit_transform(ts_y_static)
    X_transformed = x_scaler_eval.fit_transform(ts_X)

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

    def model_builder():
        return build_tft(hcfg, tuned_tft_cfg)

    if len(y_transformed) <= hcfg.validation_weeks:
        return float("inf")
    test_start = y_transformed.time_index[-hcfg.validation_weeks]

    res = rolling_block_forecast(
        model_builder=model_builder,
        y=y_transformed,
        past_cov=X_transformed,
        future_cov=cov_ts,
        test_start=test_start,
        train_length=train_length,
        horizon=hcfg.horizon_weeks,
        stride=hcfg.stride_weeks,
        fit_kwargs={"verbose": False},
        predict_kwargs={},
        merge="keep_last",
        required_input_chunk_length=hcfg.input_chunk_length,
        verbose=False,
    )

    backtest_transformed = res["merged"]
    if backtest_transformed is None:
        return float("inf")

    backtest = y_pipeline_eval.inverse_transform(backtest_transformed)

    metrics_df = evaluate_multivariate(
        true_ts=ts_y_static,
        pred_ts=backtest,
        output_chunk_length=hcfg.horizon_weeks,
        return_df=True,
    )

    metric_name = tuning_cfg.metric
    if metric_name not in metrics_df.columns:
        raise ValueError(f"Metrik '{metric_name}' nicht im Metrics-DataFrame vorhanden.")

    score = float(metrics_df[metric_name].mean())

    return score


def train_and_evaluate_for_horizon(
    df_raw: pd.DataFrame,
    hcfg: HorizonConfig,
    cfg: Optional[GlobalConfig] = None,
    logger: Optional[Callable[[str, Dict], None]] = None,
) -> (ModelArtifacts, pd.DataFrame):
    """
    Trainiert und evaluiert ein Modell für einen gegebenen Horizont.
    Optional kann über Optuna ein geeignetes train_length_weeks und eine
    Hyperparameter-Konfiguration für das TFT-Modell ermittelt werden.
    """
    cfg = cfg or GlobalConfig()

    df = preprocess_dataframe(df_raw, cfg)

    ts_y, ts_X, _ = build_target_and_past_covariates(df, cfg)

    ts_y_static, static_transformer = add_static_covariates(ts_y, cfg)

    cov_ts = build_weekly_covariates(df, cfg)

    tuning_cfg = cfg.tuning.get(hcfg.horizon_weeks, TuningConfig())

    effective_train_length_weeks = hcfg.train_length_weeks
    effective_tft_cfg = deepcopy(cfg.tft)

    if tuning_cfg.enabled:
        study = optuna.create_study(direction=tuning_cfg.direction)
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
        best_train_length = best_params["train_length_weeks"]
        effective_train_length_weeks = max(
            best_train_length,
            hcfg.input_chunk_length + 1,
            hcfg.output_chunk_length + 1,
        )

        effective_tft_cfg = _build_tuned_tft_cfg_from_trial(
            trial=optuna.trial.FixedTrial(best_params),
            base_cfg=cfg.tft,
            tuning_cfg=tuning_cfg,
        )

        if logger is not None:
            logger(
                f"tuning_{hcfg.name}",
                {
                    "best_params": best_params,
                    "effective_train_length_weeks": effective_train_length_weeks,
                },
            )

    transforms_eval = build_transformers()
    y_pipeline_eval = transforms_eval.y_pipeline
    x_scaler_eval = transforms_eval.x_scaler

    y_transformed = y_pipeline_eval.fit_transform(ts_y_static)
    X_transformed = x_scaler_eval.fit_transform(ts_X)

    def model_builder_eval():
        return build_tft(hcfg, effective_tft_cfg)

    if len(y_transformed) <= hcfg.validation_weeks:
        raise RuntimeError("Zu wenige Beobachtungen für das gewählte Validierungsfenster.")

    test_start = y_transformed.time_index[-hcfg.validation_weeks]

    res = rolling_block_forecast(
        model_builder=model_builder_eval,
        y=y_transformed,
        past_cov=X_transformed,
        future_cov=cov_ts,
        test_start=test_start,
        train_length=effective_train_length_weeks,
        horizon=hcfg.horizon_weeks,
        stride=hcfg.stride_weeks,
        fit_kwargs={"verbose": True},
        predict_kwargs={},
        merge="keep_last",
        required_input_chunk_length=hcfg.input_chunk_length,
        verbose=True,
    )

    backtest_transformed = res["merged"]

    if backtest_transformed is None:
        raise RuntimeError("rolling_block_forecast hat keine Vorhersageblöcke erzeugt.")

    backtest = y_pipeline_eval.inverse_transform(backtest_transformed)

    metrics_df = evaluate_multivariate(
        true_ts=ts_y_static,
        pred_ts=backtest,
        output_chunk_length=hcfg.horizon_weeks,
        return_df=True,
    )
    metrics_df["model_name"] = hcfg.name

    if logger is not None:
        logger(
            f"metrics_{hcfg.name}",
            metrics_df.to_dict(orient="list"),
        )

    transforms_final = build_transformers()
    y_pipeline_final = transforms_final.y_pipeline
    x_scaler_final = transforms_final.x_scaler

    if len(ts_y_static) <= effective_train_length_weeks:
        cutoff = ts_y_static.start_time()
    else:
        cutoff = ts_y_static.time_index[-effective_train_length_weeks]

    y_train = ts_y_static.slice(cutoff, None)
    X_train = ts_X.slice(cutoff, None)
    cov_train = cov_ts.slice(cutoff, None)

    y_train_transformed = y_pipeline_final.fit_transform(y_train)
    X_train_transformed = x_scaler_final.fit_transform(X_train)

    final_model = build_tft(hcfg, effective_tft_cfg)
    final_model.fit(
        series=y_train_transformed,
        past_covariates=X_train_transformed,
        future_covariates=cov_train,
        verbose=True,
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

    return artifacts, metrics_df


def forecast_with_artifacts(
    df_raw: pd.DataFrame,
    artifacts: ModelArtifacts,
    cfg: Optional[GlobalConfig] = None,
) -> TimeSeries:
    """
    Erzeugt eine Vorhersage auf Basis der aktuellsten Daten und eines bereits
    trainierten Modells. Der Lookback entspricht dem im Training verwendeten
    train_length_weeks.
    """
    cfg = cfg or GlobalConfig()
    hcfg = artifacts.hcfg

    df = preprocess_dataframe(df_raw, cfg)
    ts_y, ts_X, _ = build_target_and_past_covariates(df, cfg)

    cov_ts_full = build_weekly_covariates(df, cfg)

    effective_train_length_weeks = artifacts.train_length_weeks

    cutoff_pos = max(0, len(ts_y) - effective_train_length_weeks)
    cutoff = ts_y.time_index[cutoff_pos]

    y_recent = ts_y.slice(cutoff, None)
    X_recent = ts_X.slice(cutoff, None)
    cov_recent = cov_ts_full.slice(cutoff, None)

    y_pipeline = artifacts.transforms.y_pipeline
    x_scaler = artifacts.transforms.x_scaler

    y_recent_transformed = y_pipeline.transform(y_recent)
    X_recent_transformed = x_scaler.transform(X_recent)

    y_pred_transformed = artifacts.model.predict(
        n=hcfg.horizon_weeks,
        past_covariates=X_recent_transformed,
        future_covariates=cov_recent,
    )

    y_pred = y_pipeline.inverse_transform(y_pred_transformed)

    return y_pred


def run_full_job(
    df_raw: pd.DataFrame,
    cfg: Optional[GlobalConfig] = None,
    logger: Optional[Callable[[str, Dict], None]] = None,
) -> Dict[int, Dict[str, object]]:
    """
    Führt Training, Evaluation und Vorhersage für alle in der GlobalConfig
    definierten Horizonte aus (z. B. 13 und 52 Wochen).
    """
    cfg = cfg or GlobalConfig()

    results: Dict[int, Dict[str, object]] = {}

    for horizon, hcfg in cfg.horizons.items():
        artifacts, metrics_df = train_and_evaluate_for_horizon(
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
        }

    return results
