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



pipeline.py:
  
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


utils.py:

from typing import Any, Callable, Dict, Optional

from darts import TimeSeries


def rolling_block_forecast(
    model_builder: Callable[[], Any],
    y: TimeSeries,
    past_cov: Optional[TimeSeries],
    future_cov: Optional[TimeSeries],
    test_start,
    train_length: int,
    horizon: int,
    stride: int,
    fit_kwargs: Optional[Dict] = None,
    predict_kwargs: Optional[Dict] = None,
    merge: str = "keep_last",
    required_input_chunk_length: Optional[int] = None,
    verbose: bool = False,
):
    """
    Führt einen Block-Forecast durch, bei dem iterativ neue Trainingsfenster
    verwendet und vollständige Vorhersageblöcke erzeugt werden.

    Erwartet bereits transformierte Eingaben (z. B. skalierte/transformierte y).
    """
    fit_kwargs = fit_kwargs or {}
    predict_kwargs = predict_kwargs or {}

    def _idx(ts: TimeSeries, pos: int):
        return ts.time_index[pos]

    last_valid_start = _idx(y, len(y) - horizon)

    if isinstance(test_start, int):
        cur_start = _idx(y, test_start)
    else:
        cur_start = test_start

    blocks = []

    while True:
        cur_pos = y.get_index_at_point(cur_start)
        train_start_pos = max(0, cur_pos - train_length)
        train_end_pos = cur_pos - 1

        if train_end_pos <= train_start_pos:
            break

        train_start = _idx(y, train_start_pos)
        train_end = _idx(y, train_end_pos)

        y_train = y.slice(train_start, train_end)
        y_test_start = cur_start

        if past_cov is not None:
            past_train = past_cov.slice(train_start, train_end)
            past_future = past_cov.slice(train_start, None)
        else:
            past_train = None
            past_future = None

        if future_cov is not None:
            fut_train = future_cov.slice(train_start, train_end)
            fut_future = future_cov.slice(train_start, None)
        else:
            fut_train = None
            fut_future = None

        if required_input_chunk_length is not None:
            if len(y_train) < required_input_chunk_length:
                if verbose:
                    print(
                        f"Trainingsfenster zu kurz ({len(y_train)}) für "
                        f"required_input_chunk_length={required_input_chunk_length}. Stop."
                    )
                break

        model = model_builder()
        model.fit(
            series=y_train,
            past_covariates=past_train,
            future_covariates=fut_train,
            **fit_kwargs,
        )

        y_pred = model.predict(
            n=horizon,
            past_covariates=past_future,
            future_covariates=fut_future,
            **predict_kwargs,
        )

        pred_start = y_pred.start_time()
        if pred_start != y_test_start:
            y_pred = y_pred.slice(y_test_start, None)
        blocks.append(y_pred)

        if verbose:
            print(
                f"[Block] train [{train_start.date()} .. {train_end.date()}] "
                f"-> pred [{y_pred.start_time().date()} .. {y_pred.end_time().date()}] (len={len(y_pred)})"
            )

        next_pos = cur_pos + stride
        if next_pos >= len(y):
            break
        next_start = _idx(y, next_pos)

        if next_start > last_valid_start:
            if verbose:
                print("Stop: nächster Blockstart läge hinter dem letzten gültigen Start für vollen Horizon.")
            break

        cur_start = next_start

    if not blocks:
        return {"blocks": [], "merged": None}

    merged = blocks[0]
    for b in blocks[1:]:
        if merge == "stack":
            merged = merged.stack(b, axis=0)
        else:
            merged = merged.append(b)

    return {"blocks": blocks, "merged": merged}


model.py:

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
 
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers import Scaler, FittableDataTransformer
from sklearn.preprocessing import PowerTransformer
from darts.models import TFTModel
from darts.utils.likelihood_models import PoissonLikelihood, NegativeBinomialLikelihood
import torch
 
from .config import HorizonConfig, TftConfig
 
 
@dataclass
class TransformArtifacts:
    """
    Kapselt die für ein Modell verwendeten Transformationen.
    """
    y_pipeline: Pipeline
    x_scaler: Scaler
 
 
class RollingWinsorizer(FittableDataTransformer):
    """
    Rolling/expanding Winsorizer, der Ausreißer anhand von Quantilen begrenzt.
    """
 
    def __init__(
        self,
        lower_q: float = 0.01,
        upper_q: float = 0.99,
        mode: Literal["rolling", "expanding"] = "rolling",
        window: int = 26,
        min_periods: Optional[int] = None,
        ema_alpha: Optional[float] = None,
        integer: bool = False,
        name: str = "rolling_winsorizer",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose)
        assert 0.0 <= lower_q < upper_q <= 1.0
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.mode = mode
        self.window = window
        self.min_periods = min_periods
        self.ema_alpha = ema_alpha
        self.integer = integer
 
    @classmethod
    def ts_fit(
        cls,
        series: Optional[TimeSeries],
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if series is None or len(series) == 0:
            return {}
 
        p = params or {}
        lower_q = p.get("lower_q", 0.01)
        upper_q = p.get("upper_q", 0.99)
        mode = p.get("mode", "rolling")
        window = p.get("window", 26)
        min_per = p.get("min_periods", None)
        ema_alpha = p.get("ema_alpha", None)
        integer = p.get("integer", False)
 
        idx = series.time_index
        cols = list(series.components)
        vals = series.values(copy=False)
 
        if not np.isfinite(vals).any():
            return {}
 
        df = pd.DataFrame(vals, index=idx, columns=cols)
 
        if df.isna().all().any():
            df = df.copy()
            for c in df.columns[df.isna().all()]:
                df[c] = 0.0
 
        if mode == "rolling":
            if min_per is None:
                min_per = max(5, int(0.3 * window))
            low_df = df.rolling(window=window, min_periods=min_per).quantile(lower_q)
            high_df = df.rolling(window=window, min_periods=min_per).quantile(upper_q)
 
            low_exp = df.expanding(min_periods=5).quantile(lower_q)
            high_exp = df.expanding(min_periods=5).quantile(upper_q)
 
            low_df = low_df.combine_first(low_exp)
            high_df = high_df.combine_first(high_exp)
        elif mode == "expanding":
            low_df = df.expanding(min_periods=5).quantile(lower_q)
            high_df = df.expanding(min_periods=5).quantile(upper_q)
        else:
            return {}
 
        if ema_alpha is not None and 0 < ema_alpha < 1:
            low_df = low_df.ewm(alpha=ema_alpha, adjust=False).mean()
            high_df = high_df.ewm(alpha=ema_alpha, adjust=False).mean()
 
        low_df = np.minimum(low_df, high_df)
        low_df = (
            low_df.replace([np.inf, -np.inf], np.nan)
            .fillna(method="ffill")
            .fillna(method="bfill")
        )
        high_df = (
            high_df.replace([np.inf, -np.inf], np.nan)
            .fillna(method="ffill")
            .fillna(method="bfill")
        )
 
        if integer:
            low_df = np.floor(low_df)
            high_df = np.ceil(high_df)
 
        if low_df.isna().any().any() or high_df.isna().any().any():
            return {}
 
        return {
            "index": idx,
            "columns": cols,
            "low": low_df.values.astype(float),
            "high": high_df.values.astype(float),
            "integer": bool(integer),
        }
 
    @classmethod
    def ts_transform(
        cls,
        series: Optional[TimeSeries],
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[TimeSeries]:
        if series is None or not params or "low" not in params or "high" not in params:
            return series
 
        vals = series.values(copy=True)
        idx_series = series.time_index
        cols_series = list(series.components)
 
        idx_fit = params.get("index", idx_series)
        cols_fit = params.get("columns", cols_series)
        low_arr = params["low"]
        high_arr = params["high"]
        integer = params.get("integer", False)
 
        if idx_series.equals(idx_fit) and cols_series == cols_fit:
            low = low_arr
            high = high_arr
        else:
            df_low = pd.DataFrame(low_arr, index=idx_fit, columns=cols_fit)
            df_high = pd.DataFrame(high_arr, index=idx_fit, columns=cols_fit)
 
            common_index = idx_series.intersection(df_low.index)
            common_cols = [c for c in cols_series if c in df_low.columns]
            if len(common_index) == 0 or len(common_cols) == 0:
                return series
 
            df_low = df_low.reindex(index=idx_series, columns=cols_series)
            df_high = df_high.reindex(index=idx_series, columns=cols_series)
            low = df_low.values
            high = df_high.values
 
            mask_nan = ~np.isfinite(low) | ~np.isfinite(high)
            if mask_nan.any():
                low = np.where(np.isfinite(low), low, -np.inf)
                high = np.where(np.isfinite(high), high, np.inf)
 
        vals = np.minimum(np.maximum(vals, low), high)
        if integer:
            vals = np.rint(vals)
 
        return TimeSeries.from_times_and_values(
            idx_series,
            vals,
            columns=series.components,
            static_covariates=series.static_covariates,
        )
 
    @classmethod
    def ts_inverse_transform(
        cls,
        series: Optional[TimeSeries],
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        return series
 
    @property
    def supports_inverse_transform(self) -> bool:
        return True
 
 
def build_transformers() -> TransformArtifacts:
    """
    Erzeugt die Standard-Transformationen für Ziel- und Kovariatenzeitreihen.
    """
    winsorizer = RollingWinsorizer(
        lower_q=0.01,
        upper_q=0.90,
        mode="rolling",
        window=26,
        min_periods=None,
        ema_alpha=None,
        integer=False,
    )
    yj_transformer = Scaler(
        scaler=PowerTransformer(method="yeo-johnson", standardize=False)
    )
    std_scaler = Scaler()
 
    y_pipeline = Pipeline([winsorizer, yj_transformer, std_scaler])
    x_scaler = Scaler()
 
    return TransformArtifacts(y_pipeline=y_pipeline, x_scaler=x_scaler)
 
 
def _build_likelihood(name: Optional[str]):
    if name is None:
        return None
    name = name.lower()
    if name == "poisson":
        return PoissonLikelihood()
    if name in ("nb", "negative_binomial", "negativebinomial"):
        return NegativeBinomialLikelihood()
    raise ValueError(f"Unbekannte Likelihood-Konfiguration: {name}")
 
 
def _build_optimizer_and_loss(tft_cfg: TftConfig):
    opt_name = tft_cfg.optimizer.lower()
    if opt_name == "adamw":
        optimizer_cls = torch.optim.AdamW
    elif opt_name == "adam":
        optimizer_cls = torch.optim.Adam
    else:
        raise ValueError(f"Unbekannter Optimizer-Name: {tft_cfg.optimizer}")
 
    loss_name = tft_cfg.loss.lower()
    if loss_name == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unbekannte Loss-Konfiguration: {tft_cfg.loss}")
 
    return optimizer_cls, loss_fn
 
 
def build_tft(hcfg: HorizonConfig, tft_cfg: TftConfig) -> TFTModel:
    """
    Baut ein TFT-Modell mit den konfigurierten Hyperparametern.
    """
    likelihood_obj = _build_likelihood(tft_cfg.likelihood)
    optimizer_cls, loss_fn = _build_optimizer_and_loss(tft_cfg)
 
    model = TFTModel(
        input_chunk_length=hcfg.input_chunk_length,
        output_chunk_length=hcfg.output_chunk_length,
        n_epochs=tft_cfg.n_epochs,
        full_attention=tft_cfg.full_attention,
        num_attention_heads=tft_cfg.num_attention_heads,
        batch_size=tft_cfg.batch_size,
        likelihood=likelihood_obj,
        random_state=tft_cfg.random_state,
        hidden_size=tft_cfg.hidden_size,
        hidden_continuous_size=tft_cfg.hidden_continuous_size,
        lstm_layers=tft_cfg.lstm_layers,
        force_reset=tft_cfg.force_reset,
        dropout=tft_cfg.dropout,
        loss_fn=loss_fn,
        use_static_covariates=tft_cfg.use_static_covariates,
        add_relative_index=tft_cfg.add_relative_index,
        add_encoders=None,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs={"lr": tft_cfg.learning_rate},
    )
 
    return model



features.py:

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from sklearn.preprocessing import OneHotEncoder

from .config import GlobalConfig, PreprocessingConfig


def preprocess_dataframe(df: pd.DataFrame, cfg: GlobalConfig) -> pd.DataFrame:
    """
    Bereitet das Roh-DataFrame für die Modellierung vor.

    Schritte:
    - Datums-Spalte (falls konfiguriert) in DatetimeIndex umwandeln.
    - Sortieren nach Index.
    - Spalten-Drops und explizite Aggregationen.
    - Produkt-Aggregation über alle Kennzahlen/Statusse (optional).
    - Aggregation von täglichen Daten auf die Zielfrequenz (z. B. W-SUN).
    """
    df = df.copy()
 
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = cfg.data.date_column
        if date_col is not None and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        else:
            raise ValueError(
                "DataFrame muss entweder einen DatetimeIndex haben oder eine konfigurierte Datums-Spalte."
            )
 
    df = df.sort_index()
 
    # alle Spalten in numerisch konvertieren (Decimal → float), nicht konvertierbares → NaN
    df = df.apply(pd.to_numeric, errors="coerce")
 
    prep_cfg = cfg.preprocessing
 
    if prep_cfg.drop_contains or prep_cfg.aggregate_map:
        df, _ = drop_or_sum(
            df,
            drop_contains=prep_cfg.drop_contains,
            aggregate_map=prep_cfg.aggregate_map,
        )
 
    if prep_cfg.product_aggregation:
        df = add_product_aggregates_to_df(df, prep_cfg)
 
    if cfg.data.freq:
        df = df.resample(cfg.data.freq).sum(min_count=1)
 
    # NaNs nach der Aggregation durch 0 ersetzen
    df = df.fillna(0)
 
    return df


def drop_or_sum(
    df: pd.DataFrame,
    drop_contains: Optional[Iterable[str]] = None,
    aggregate_map: Optional[Dict[str, Iterable[str]]] = None,
    case_insensitive: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Einfacher Helper, um Spalten basierend auf Namen zu droppen oder zu aggregieren.
    """
    df = df.copy()
    log_info = {"dropped": [], "aggregated": {}}

    cols = df.columns.tolist()

    if drop_contains:
        drops = []
        for col in cols:
            name = col
            if case_insensitive:
                name = name.lower()
            if any(pattern.lower() in name for pattern in drop_contains):
                drops.append(col)
        df = df.drop(columns=drops)
        log_info["dropped"] = drops

    if aggregate_map:
        for new_name, col_list in aggregate_map.items():
            existing = [c for c in col_list if c in df.columns]
            if not existing:
                continue
            df[new_name] = df[existing].sum(axis=1)
            log_info["aggregated"][new_name] = existing

    return df, log_info


def add_product_aggregates_to_df(
    df: pd.DataFrame,
    prep_cfg: PreprocessingConfig,
) -> pd.DataFrame:
    """
    Erzeugt zusätzliche Produkt-Aggregatspalten auf Basis der Spaltennamen.

    Erwartete Struktur der Spalten:
        KENNZAHL__PRODUKT__STATUS

    Beispiel:
        TERM_EINGANG_SCHRIFTST__KFZ_Vollkasko__Neuschaden
        TERM_EINGANG_SONST__KFZ_Vollkasko__Folgebearbeitung

    Für jedes Produkt (z. B. KFZ_Vollkasko, HUS_Hausrat, ...) wird eine
    Aggregatspalte gebildet, die alle Kennzahlen und Statusse dieses
    Produkts aufsummiert:

        AGG_PROD_KFZ_Vollkasko
        AGG_PROD_HUS_Hausrat
        ...
    """
    df = df.copy()
    cols = df.columns

    product_to_cols: Dict[str, List[str]] = {}

    for c in cols:
        if not np.issubdtype(df[c].dtype, np.number):
            continue

        parts_main = c.split(prep_cfg.component_main_sep)
        if len(parts_main) <= prep_cfg.static_product_index:
            continue

        product = parts_main[prep_cfg.static_product_index]
        product_to_cols.setdefault(product, []).append(c)

    for product, group_cols in product_to_cols.items():
        new_name = f"{prep_cfg.product_agg_prefix}{product}"
        df[new_name] = df[group_cols].sum(axis=1)

    return df


def augment_X_with_target_past_features_safe(
    X: TimeSeries,
    prep_cfg: PreprocessingConfig,
) -> Tuple[TimeSeries, int]:
    """
    Erweitert eine gegebene Zeitreihe um Lag-, Rolling- und YoY-Features.
    Gibt die augmentierte Zeitreihe sowie die Anzahl verworfener Anfangsperioden zurück.
    """
    include_if_name_contains = prep_cfg.include_if_name_contains

    if isinstance(include_if_name_contains, str):
        include_patterns = [include_if_name_contains]
    else:
        include_patterns = list(include_if_name_contains) if include_if_name_contains else []

    def _match(name: str) -> bool:
        if not include_patterns:
            return True
        candidate = name.lower()
        pats = [p.lower() for p in include_patterns]
        return any(p in candidate for p in pats)

    df = X.to_dataframe(copy=True)
    base_cols = list(df.columns)

    lags = prep_cfg.lags
    rolling_windows = prep_cfg.rolling_windows

    if isinstance(lags, int):
        lags = [lags]
    if isinstance(rolling_windows, int):
        rolling_windows = [rolling_windows]

    max_lag = max(lags) if (prep_cfg.add_lags and lags) else 0
    max_window = max(rolling_windows) if (prep_cfg.add_ma or prep_cfg.add_std) and rolling_windows else 0
    warmup = max(max_lag, max_window, prep_cfg.min_periods)

    new_df = df.copy()

    for col in base_cols:
        if not _match(col):
            continue

        if prep_cfg.add_lags and lags:
            for L in lags:
                new_df[f"{col}_lag{L}"] = df[col].shift(L)

        if prep_cfg.add_ma and rolling_windows:
            for W in rolling_windows:
                new_df[f"{col}_ma{W}"] = df[col].rolling(window=W, min_periods=prep_cfg.min_periods).mean()

        if prep_cfg.add_std and rolling_windows:
            for W in rolling_windows:
                new_df[f"{col}_std{W}"] = df[col].rolling(window=W, min_periods=prep_cfg.min_periods).std()

        if prep_cfg.add_yoy:
            if prep_cfg.yoy_mode == "diff":
                new_df[f"{col}_yoy"] = df[col] - df[col].shift(52)
            elif prep_cfg.yoy_mode == "log":
                with np.errstate(divide="ignore", invalid="ignore"):
                    new_df[f"{col}_yoy"] = np.log1p(df[col]) - np.log1p(df[col].shift(52))
            else:
                new_df[f"{col}_yoy"] = df[col] / df[col].shift(52) - 1.0

    new_df = new_df.iloc[warmup:]
    X_aug = TimeSeries.from_dataframe(new_df, freq=X.freq)

    return X_aug, warmup


def build_target_and_past_covariates(
    df: pd.DataFrame,
    cfg: GlobalConfig,
) -> Tuple[TimeSeries, TimeSeries, int]:
    """
    Baut Zielzeitreihe (nur "echte" Komponenten) und zugehörige Kovariaten auf.
 
    Targets:
        Alle numerischen Spalten, deren Name der Struktur
        KENNZAHL__PRODUKT__STATUS entspricht (also genau zwei "__").
    Kovariaten:
        Alle numerischen Spalten (inkl. AGG_PROD_*, etc.).
    """
    # alle numerischen Spalten (auch Aggregatspalten etc.)
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        raise ValueError("Das DataFrame enthält keine numerischen Spalten zur Modellierung.")
 
    prep_cfg = cfg.preprocessing
    sep = prep_cfg.component_main_sep
 
    # Target-Spalten: nur solche, die exakt wie KENNZAHL__PRODUKT__STATUS aufgebaut sind
    target_cols = [
        c for c in numeric_cols
        if isinstance(c, str) and c.count(sep) == 2
    ]
 
    if len(target_cols) == 0:
        raise ValueError(
            "Es wurden keine gültigen Target-Spalten gefunden, die der Struktur "
            f'KENNZAHL{sep}PRODUKT{sep}STATUS entsprechen.'
        )
 
    # Ziel-Zeitreihe: nur die "echten" Komponenten
    ts_y_base = TimeSeries.from_dataframe(df[target_cols], freq=cfg.data.freq)
 
    # Kovariaten-Basis: alle numerischen Spalten (Targets + Zusatzfeatures)
    ts_X_raw = TimeSeries.from_dataframe(df[numeric_cols], freq=cfg.data.freq)
 
    # Lag-/Rolling-/YoY-Features für alle numerischen Spalten
    ts_X_aug, warmup = augment_X_with_target_past_features_safe(
        ts_X_raw,
        prep_cfg=cfg.preprocessing,
    )
 
    # y-Reihe auf den gleichen Warmup-Bereich zuschneiden
    y_idx = ts_y_base.time_index
    ts_y = ts_y_base.slice(y_idx[warmup], y_idx[-1])
 
    return ts_y, ts_X_aug, warmup


def extract_static_for_component(
    name: str,
    prep_cfg: PreprocessingConfig,
) -> Tuple[str, str, str]:
    """
    Leitet statische Merkmale aus einem Komponenten-Namen der Form

        KENNZAHL__PRODUKT__STATUS

    ab, z. B.:

        TERM_EINGANG_SCHRIFTST__KFZ_Vollkasko__Neuschaden

    Es werden drei Merkmale erzeugt:
    - Kennzahl  (z. B. TERM_EINGANG_SCHRIFTST)
    - Sparte    (z. B. KFZ oder HUS, aus dem Produkt abgeleitet)
    - Status    (z. B. Neuschaden / Folgebearbeitung)
    """
    parts_main = name.split(prep_cfg.component_main_sep)
    if len(parts_main) <= max(
        prep_cfg.static_kpi_index,
        prep_cfg.static_product_index,
        prep_cfg.static_status_index,
    ):
        raise ValueError(f"Komponenten-Name passt nicht zur erwarteten Struktur: {name}")

    kpi = parts_main[prep_cfg.static_kpi_index]
    product_full = parts_main[prep_cfg.static_product_index]
    status = parts_main[prep_cfg.static_status_index]

    product_parts = product_full.split(prep_cfg.product_sep)
    if len(product_parts) <= prep_cfg.product_sparte_index:
        sparte = product_full
    else:
        sparte = product_parts[prep_cfg.product_sparte_index]

    return kpi, sparte, status


def add_static_covariates(
    ts_y: TimeSeries,
    cfg: GlobalConfig,
) -> Tuple[TimeSeries, StaticCovariatesTransformer]:
    """
    Fügt statische Kovariaten zu einer multivariaten Ziel-Zeitreihe hinzu.

    Für jede Komponente werden drei kategoriale Merkmale erzeugt:
    - Kennzahl (z. B. TERM_EINGANG_SCHRIFTST / TERM_EINGANG_SONST)
    - Sparte   (z. B. KFZ / HUS)
    - Status   (z. B. Neuschaden / Folgebearbeitung).
    """
    prep_cfg = cfg.preprocessing
    components = ts_y.components

    static_covariates = pd.DataFrame(
        [extract_static_for_component(c, prep_cfg) for c in components],
        columns=["Kennzahl", "Sparte", "Status"],
        index=pd.Index(components),
    )

    transformer = StaticCovariatesTransformer(
        transformer_cat=OneHotEncoder(handle_unknown="ignore")
    )

    ts_y_with_static = ts_y.with_static_covariates(static_covariates)
    ts_y_encoded = transformer.fit_transform(ts_y_with_static)

    return ts_y_encoded, transformer



evaluation.py:

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries

from .config import BaselineConfig


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric MAPE in Prozent.
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask])) * 100.0


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE in Prozent.
    """
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100.0


def evaluate_multivariate(
    true_ts: TimeSeries,
    pred_ts: TimeSeries,
    output_chunk_length: int,
    return_df: bool = True,
) -> pd.DataFrame:
    """
    Berechnet Standardmetriken pro Komponente für einen gegebenen Forecast.
    """
    true_aligned = true_ts.slice_intersect(pred_ts)
    pred_aligned = pred_ts.slice_intersect(true_ts)

    metrics: List[dict] = []

    for component in true_aligned.components:
        y_true = true_aligned[component].values().flatten()
        y_pred = pred_aligned[component].values().flatten()

        mae = float(np.mean(np.abs(y_true - y_pred)))
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mape = _mape(y_true, y_pred)
        smape = _smape(y_true, y_pred)

        metrics.append(
            {
                "component": component,
                "horizon": output_chunk_length,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "mape": mape,
                "smape": smape,
            }
        )

    df_metrics = pd.DataFrame(metrics)

    if return_df:
        return df_metrics

    return df_metrics


def weekly_baseline_mean_last3(
    ts_y: TimeSeries,
    baseline_cfg: BaselineConfig,
) -> TimeSeries:
    """
    Einfache Baseline: Durchschnitt der letzten Jahre für gleiche Kalenderwoche.
    """
    df = ts_y.to_dataframe()
    idx = df.index

    df["year"] = idx.year
    df["week"] = idx.isocalendar().week.values

    value_cols = [c for c in df.columns if c not in ("year", "week")]

    baseline_df = df[value_cols].copy()
    baseline_df[:] = np.nan

    for i, (ts_year, ts_week) in enumerate(zip(df["year"], df["week"])):
        mask_candidates = (
            (df["week"] == ts_week)
            & (df["year"] < ts_year)
            & (df["year"] >= ts_year - baseline_cfg.years_back)
        )
        candidate_years = df.loc[mask_candidates, "year"].unique()
        if len(candidate_years) < baseline_cfg.min_years:
            continue
        baseline_df.iloc[i] = (
            df.loc[mask_candidates, value_cols]
            .groupby(df["year"][mask_candidates])
            .mean()
            .mean()
        )

    baseline_ts = TimeSeries.from_dataframe(baseline_df, freq=ts_y.freq)

    return baseline_ts


def plot_agg_schriftstueck_with_components(
    true_ts: TimeSeries,
    pred_ts: TimeSeries,
    keyword_pattern: str = r"(?i)TERM_EINGANG_SCHRIFTST",
    title: str = "Aggregierte Schriftstück-Kennzahl (True vs Forecast)",
    diff_label: str = "Abweichung (Forecast − True)",
    overlay_top_n: int = 5,
) -> dict:
    """
    Aggregiert und visualisiert alle Komponenten, deren Name zur Schriftstück-Kennzahl
    passt (Standard: TERM_EINGANG_SCHRIFTST im Komponenten-Namen).
    """
    df_true = true_ts.to_dataframe()
    df_pred = pred_ts.to_dataframe()

    cols_true = df_true.columns
    target_cols = [
        c for c in cols_true
        if pd.Series([c]).str.contains(keyword_pattern, regex=True).iloc[0]
    ]

    if not target_cols:
        raise ValueError(
            "Keine passenden Komponenten für Schriftstücke gefunden. "
            f"Verwendetes Pattern: {keyword_pattern}"
        )

    agg_true = df_true[target_cols].sum(axis=1)
    agg_pred = df_pred[target_cols].sum(axis=1)

    mask = np.isfinite(agg_true.values) & np.isfinite(agg_pred.values)
    if mask.sum() == 0:
        raise ValueError("Nach NaN-Filter keine gemeinsamen Punkte für True und Forecast übrig.")

    agg_true = agg_true[mask]
    agg_pred = agg_pred[mask]
    diff = agg_pred - agg_true

    plt.figure(figsize=(12, 6))
    plt.plot(agg_true.index, agg_true.values, label="True (Summe)", linewidth=2)
    plt.plot(agg_pred.index, agg_pred.values, label="Forecast (Summe)", linewidth=2, linestyle="--")
    plt.fill_between(agg_true.index, diff.values, 0, alpha=0.2, label=diff_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    vols = df_true[target_cols].sum(axis=0).sort_values(ascending=False)
    sel_cols = vols.index.tolist()
    if overlay_top_n is not None:
        sel_cols = sel_cols[:overlay_top_n]

    plt.figure(figsize=(12, 6))
    for c in sel_cols:
        plt.plot(df_true.index, df_true[c].values, label=f"True {c}", alpha=0.6)
        if c in df_pred.columns:
            plt.plot(df_pred.index, df_pred[c].values, linestyle="--", label=f"Forecast {c}", alpha=0.6)
    plt.title("Overlay einzelner Schriftstück-Komponenten (Kennzahl TERM_EINGANG_SCHRIFTST)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "true_sum_ts": TimeSeries.from_series(agg_true),
        "pred_sum_ts": TimeSeries.from_series(agg_pred),
        "diff_ts": TimeSeries.from_series(diff),
        "matched_components": target_cols,
        "overlay_components": sel_cols,
    }




covariates.py:

import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from icalendar import Calendar
from darts import TimeSeries

from .config import GlobalConfig

NAME2STATE: Dict[str, str] = {
    "baden-wuerttemberg": "BW",
    "baden-wurttemberg": "BW",
    "baden wuerttemberg": "BW",
    "baden wurttemberg": "BW",
    "baden-württemberg": "BW",
    "württemberg": "BW",
    "wuerttemberg": "BW",
    "baden": "BW",
    "bayern": "BY",
    "berlin": "BE",
    "brandenburg": "BB",
    "bremen": "HB",
    "hamburg": "HH",
    "hessen": "HE",
    "mecklenburg-vorpommern": "MV",
    "mecklenburg vorpommern": "MV",
    "mecklenburgvorpommern": "MV",
    "vorpommern": "MV",
    "niedersachsen": "NI",
    "nordrhein-westfalen": "NW",
    "nordrhein westfalen": "NW",
    "nordrheinwestfalen": "NW",
    "nrw": "NW",
    "rheinland-pfalz": "RP",
    "rheinland pfalz": "RP",
    "rheinlandpfalz": "RP",
    "saarland": "SL",
    "sachsen-anhalt": "ST",
    "sachsen anhalt": "ST",
    "sachsenanhalt": "ST",
    "sachsen": "SN",
    "schleswig-holstein": "SH",
    "schleswig holstein": "SH",
    "schleswigholstein": "SH",
    "schleswig": "SH",
    "holstein": "SH",
    "thueringen": "TH",
    "thuringen": "TH",
    "thüringen": "TH",
}

GER_STATES = ["BW", "BY", "BE", "BB", "HB", "HH", "HE", "MV", "NI", "NW", "RP", "SL", "SN", "ST", "SH", "TH"]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _state_from_text(text: str) -> List[str]:
    n = _normalize(text)
    states: List[str] = []

    for key, code in NAME2STATE.items():
        if key in n:
            states.append(code)

    states = sorted(set(states))
    return states


def _find_ics_files(root: Path) -> List[Path]:
    files: List[Path] = []
    if root.is_file() and root.suffix.lower() == ".ics":
        files = [root]
    else:
        for p, _, fnames in os.walk(root):
            for fn in fnames:
                if fn.lower().endswith(".ics"):
                    files.append(Path(p) / fn)
    return sorted(set(files))


def _parse_ics_file(path: Path) -> List[Tuple[date, date, List[str]]]:
    events: List[Tuple[date, date, List[str]]] = []

    with open(path, "rb") as f:
        cal = Calendar.from_ical(f.read())

    for comp in cal.walk():
        if comp.name != "VEVENT":
            continue

        summary = str(comp.get("summary", ""))
        description = str(comp.get("description", ""))
        txt = summary + " " + description

        states = _state_from_text(txt)
        if not states:
            continue

        dtstart = comp.get("dtstart").dt
        dtend = comp.get("dtend").dt

        if isinstance(dtstart, datetime):
            dtstart = dtstart.date()
        if isinstance(dtend, datetime):
            dtend = dtend.date()

        if dtend <= dtstart:
            dtend = dtstart

        events.append((dtstart, dtend, states))

    return events


def build_covariates_all_states(
    df: pd.DataFrame,
    freq: str,
    ics_root_dir: str,
) -> TimeSeries:
    """
    Erzeugt Ferien-/Holiday-Kovariaten und weitere abgeleitete Variablen.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame muss einen DatetimeIndex haben, um Covariates zu erzeugen.")

    start_date = df.index.min().date()
    end_date = df.index.max().date()

    idx = pd.date_range(start=start_date, end=end_date, freq="D")
    cov_daily = pd.DataFrame(0, index=idx, columns=GER_STATES, dtype="int8")

    root = Path(ics_root_dir)
    ics_files = _find_ics_files(root)

    for path in ics_files:
        events = _parse_ics_file(path)
        for dstart, dend, states in events:
            for d in pd.date_range(dstart, dend, freq="D"):
                if d.date() < start_date or d.date() > end_date:
                    continue
                if d.date() not in cov_daily.index:
                    continue
                for s in states:
                    if s in cov_daily.columns:
                        cov_daily.loc[d.date(), s] = 1

    cov_daily["holidays_any_state"] = (cov_daily[GER_STATES].sum(axis=1) > 0).astype("int8")

    cov_daily["vac_DE_count"] = cov_daily[GER_STATES].sum(axis=1).astype("int16")

    cov_daily["vac_BY"] = cov_daily["BY"].astype("int8")

    is_workday = cov_daily.index.to_series().dt.weekday < 5
    cov_daily["holidays_workdays_BY"] = ((cov_daily["BY"] == 1) & is_workday).astype("int8")

    cov_daily["holidays_workdays_DE_count"] = (
        cov_daily[GER_STATES].multiply(is_workday, axis=0).sum(axis=1)
    ).astype("int16")

    dayofyear = cov_daily.index.to_series().dt.dayofyear
    cov_daily["year_turn_any"] = ((dayofyear <= 7) | (dayofyear >= 359)).astype("int8")

    agg_dict = {state: "max" for state in GER_STATES}
    agg_dict.update(
        {
            "holidays_any_state": "max",
            "vac_DE_count": "sum",
            "vac_BY": "max",
            "holidays_workdays_BY": "sum",
            "holidays_workdays_DE_count": "sum",
            "year_turn_any": "max",
        }
    )

    cov_weekly = cov_daily.resample(freq).agg(agg_dict)

    target_index = df.resample(freq).asfreq().index
    cov_weekly = cov_weekly.reindex(target_index).fillna(0)

    cov_ts = TimeSeries.from_dataframe(cov_weekly, freq=freq)

    return cov_ts


def build_weekly_covariates(df: pd.DataFrame, cfg: GlobalConfig) -> TimeSeries:
    """
    Convenience-Wrapper, um Kovariaten direkt in der konfigurierten Frequenz zu erzeugen.
    """
    return build_covariates_all_states(
        df=df,
        freq=cfg.data.freq,
        ics_root_dir=cfg.data.ics_root_dir,
    )




config.py:

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    """
    Globale Datenkonfiguration.

    freq:
        Frequenz der Zeitreihe (Pandas/Darts). Für wöchentliche Daten z. B. 'W-SUN'.
    date_column:
        Name der Spalte mit Datumsangaben, falls der DataFrame noch keinen DatetimeIndex hat.
        Wenn None, wird kein automatisches Setzen des Index versucht.
    ics_root_dir:
        Wurzelverzeichnis, unter dem Ferien-/Holiday-ICS-Dateien liegen.
    """
    freq: str = "W-SUN"
    date_column: Optional[str] = "date"
    ics_root_dir: str = "/mnt/ferien-api-data-main/resources/de"


@dataclass
class PreprocessingConfig:
    """
    Konfiguration der Vorverarbeitung und des Feature-Engineerings.

    drop_contains:
        Spalten, deren Name einen dieser Strings enthält, werden entfernt.
    aggregate_map:
        Mapping neuer Spaltennamen auf eine Liste existierender Spalten, die zu einer Summe
        aggregiert werden.

    Produkt-Aggregation:
    --------------------
    Die Spaltennamen folgen der Struktur

        KENNZAHL__PRODUKT__STATUS

    z. B. TERM_EINGANG_SCHRIFTST__KFZ_Vollkasko__Neuschaden

    Wenn product_aggregation=True ist, werden für jedes Produkt zusätzliche
    Aggregatspalten erzeugt, in denen über alle Kennzahlen und Statusse
    des jeweiligen Produkts summiert wird:

        AGG_PROD_KFZ_Vollkasko
        AGG_PROD_HUS_Hausrat
        ...
    """
    drop_contains: List[str] = field(default_factory=list)
    aggregate_map: Dict[str, List[str]] = field(default_factory=dict)

    product_aggregation: bool = True
    product_agg_prefix: str = "AGG_PROD_"

    include_if_name_contains: List[str] = field(default_factory=list)

    add_lags: bool = True
    add_yoy: bool = True
    add_ma: bool = True
    add_std: bool = False
    lags: List[int] = field(default_factory=lambda: [4, 8, 13]) #field(default_factory=lambda: [4, 8, 13, 52]) - Längen geändert, da bei Test nur kürzerer Zeitraum verfügbar
    rolling_windows: List[int] = field(default_factory=lambda: [4, 8, 13]) #field(default_factory=lambda: [4, 8, 13, 52]) - Längen geändert, da bei Test nur kürzerer Zeitraum verfügbar
    yoy_mode: str = "log"
    min_periods: int = 4

    component_main_sep: str = "__"
    static_kpi_index: int = 0
    static_product_index: int = 1
    static_status_index: int = 2
    product_sep: str = "_"
    product_sparte_index: int = 0


@dataclass
class TftConfig:
    """
    Globale Hyperparameter für das TFT-Modell (Basiseinstellungen).

    Diese Werte werden verwendet, wenn Tuning deaktiviert ist oder als Startpunkt,
    den das Tuning überschreibt.
    """
    n_epochs: int = 400
    full_attention: bool = False
    num_attention_heads: int = 4
    batch_size: int = 12
    likelihood: Optional[str] = None  # None, 'poisson', 'nb'
    random_state: int = 0
    hidden_size: int = 64
    hidden_continuous_size: int = 16
    lstm_layers: int = 2
    force_reset: bool = True
    dropout: float = 0.3
    use_static_covariates: bool = True
    add_relative_index: bool = True
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    loss: str = "mse"


@dataclass
class HorizonConfig:
    """
    Konfiguration für ein einzelnes Modell mit bestimmtem Vorhersagehorizont.
    """
    name: str
    horizon_weeks: int
    input_chunk_length: int
    output_chunk_length: int
    train_length_weeks: int
    validation_weeks: int
    stride_weeks: int


@dataclass
class BaselineConfig:
    """
    Konfiguration der einfachen Jahres-Baseline.
    """
    years_back: int = 3
    min_years: int = 1


@dataclass
class TuningConfig:
    """
    Konfiguration für Optuna-Tuning eines bestimmten Horizonts.

    enabled:
        Ob Optuna-Tuning für diesen Horizont aktiviert ist.
    n_trials:
        Anzahl der Tuning-Trials.
    direction:
        Optimierungsrichtung, z. B. 'minimize' für Fehler-Metriken.
    metric:
        Name der Metrik, die optimiert werden soll (z. B. 'smape' oder 'rmse').

    train_length_min, train_length_max, train_length_step:
        Suchraum für train_length_weeks in Wochen.

    hidden_size_*:
        Suchraum für die Größe der verborgenen Repräsentation im TFT.
    hidden_continuous_size_*:
        Suchraum für die Dimension kontinuierlicher Encoder im TFT.
    lstm_layers_*:
        Suchraum für die Anzahl der LSTM-Schichten.
    dropout_*:
        Suchraum für Dropout-Werte.
    learning_rate_*:
        Suchraum für die Lernrate (logarithmische Skala).
    batch_size_choices:
        Diskrete Menge erlaubter Batch-Größen.
    """
    enabled: bool = False
    n_trials: int = 20
    direction: str = "minimize"
    metric: str = "smape"

    train_length_min: int = 52
    train_length_max: int = 208
    train_length_step: int = 13

    hidden_size_min: int = 32
    hidden_size_max: int = 96
    hidden_size_step: int = 16

    hidden_continuous_size_min: int = 8
    hidden_continuous_size_max: int = 32
    hidden_continuous_size_step: int = 8

    lstm_layers_min: int = 1
    lstm_layers_max: int = 3

    dropout_min: float = 0.1
    dropout_max: float = 0.5

    learning_rate_min: float = 1e-4
    learning_rate_max: float = 3e-3

    batch_size_choices: List[int] = field(default_factory=lambda: [8, 12, 16, 24])


@dataclass
class GlobalConfig:
    """
    Gesamt-Konfiguration des Projekts.
    Achtung: alle veränderlichen Defaults werden über default_factory gesetzt.
    """
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    tft: TftConfig = field(default_factory=TftConfig)
 
    horizons: Dict[int, HorizonConfig] = field(
        default_factory=lambda: {
            13: HorizonConfig(
                name="h13_model",
                horizon_weeks=13,
                input_chunk_length=52,
                output_chunk_length=13,
                train_length_weeks=104,
                validation_weeks=52,
                stride_weeks=13,
            ),
            52: HorizonConfig(
                name="h52_model",
                horizon_weeks=52,
                input_chunk_length=104,
                output_chunk_length=52,
                train_length_weeks=156,
                validation_weeks=104,
                stride_weeks=52,
            ),
        }
    )
 
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
 
    tuning: Dict[int, TuningConfig] = field(
        default_factory=lambda: {
            13: TuningConfig(
                enabled=False,
                n_trials=30,
                direction="minimize",
                metric="smape",
                train_length_min=52,
                train_length_max=156,
                train_length_step=13,
                hidden_size_min=32,
                hidden_size_max=96,
                hidden_size_step=16,
                hidden_continuous_size_min=8,
                hidden_continuous_size_max=32,
                hidden_continuous_size_step=8,
                lstm_layers_min=1,
                lstm_layers_max=3,
                dropout_min=0.1,
                dropout_max=0.5,
                learning_rate_min=5e-4,
                learning_rate_max=2e-3,
                batch_size_choices=[8, 12, 16, 24],
            ),
            52: TuningConfig(
                enabled=False,
                n_trials=30,
                direction="minimize",
                metric="smape",
                train_length_min=104,
                train_length_max=260,
                train_length_step=13,
                hidden_size_min=48,
                hidden_size_max=128,
                hidden_size_step=16,
                hidden_continuous_size_min=16,
                hidden_continuous_size_max=48,
                hidden_continuous_size_step=8,
                lstm_layers_min=1,
                lstm_layers_max=3,
                dropout_min=0.1,
                dropout_max=0.5,
                learning_rate_min=3e-4,
                learning_rate_max=2e-3,
                batch_size_choices=[8, 12, 16, 24],
            ),
        }
    )





pluto_forecast_job.py:

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd
from darts import TimeSeries

from forecasting import GlobalConfig, run_full_job
from pluto_multivariate_repository import PlutoMultivariateRepository


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _combine_13_und_52_wochen(
    forecast_13: TimeSeries,
    forecast_52: TimeSeries,
    horizon_13: int = 13,
    total_horizon: int = 52,
) -> pd.DataFrame:
    """
    Kombiniert zwei Vorhersagen in eine 52-Wochen-Prognose.

    Logik:
    - Wochen 1 bis 13: Werte aus forecast_13.
    - Wochen 14 bis 52: Werte aus forecast_52 (ab der 14. Woche).
    """
    df_13 = forecast_13.pd_dataframe()
    df_52 = forecast_52.pd_dataframe()

    if len(df_13) < horizon_13:
        raise ValueError(f"forecast_13 hat nur {len(df_13)} Punkte, erwartet {horizon_13}.")
    if len(df_52) < total_horizon:
        raise ValueError(f"forecast_52 hat nur {len(df_52)} Punkte, erwartet {total_horizon}.")

    idx_52 = df_52.index
    df_combined = pd.DataFrame(index=idx_52, columns=df_52.columns, dtype=float)

    df_combined.iloc[0:horizon_13, :] = df_13.iloc[0:horizon_13, :].values

    remaining = total_horizon - horizon_13
    df_combined.iloc[horizon_13:total_horizon, :] = df_52.iloc[horizon_13:horizon_13 + remaining, :].values

    df_combined = df_combined.iloc[0:total_horizon]

    return df_combined


def run_pluto_multivariate_forecast_job() -> Dict[int, Dict[str, object]]:
    """
    Führt den vollständigen multivariaten Prognose-Job für PLUTO aus:

    - Lese tägliche Daten aus DB2.
    - Aggregation auf W-SUN erfolgt im Forecasting-Preprocessing.
    - Trainiere und evaluiere Modelle für 13 und 52 Wochen.
    - Erstelle 13- und 52-Wochen-Prognose.
    - Kombiniere beide Prognosen zu einer 52-Wochen-Prognose,
      bei der Wochen 1–13 aus der 13-Wochen-Prognose stammen und
      Wochen 14–52 aus der 52-Wochen-Prognose.
    - Schreibe die kombinierte Prognose zurück nach DB2.
    """
    cfg = GlobalConfig()
    repo = PlutoMultivariateRepository()

    try:
        df_daily = repo.read_timeseries()
        if df_daily.empty:
            logger.warning("Keine Daten aus DB2 geladen. Abbruch.")
            return {}

        results = run_full_job(df_daily, cfg=cfg, logger=None)

        if 13 not in results or 52 not in results:
            raise RuntimeError("Ergebnisse für 13- oder 52-Wochen-Horizont fehlen in der Forecasting-Pipeline.")

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



pluto_multivariate_repository:

from __future__ import annotations
 
from datetime import datetime
import logging
import os
from typing import List, Tuple
 
import ibm_db
import ibm_db_dbi
import pandas as pd
 
 
class PlutoMultivariateRepository:
    """
    Repository für die multivariate Prognose der Termineingänge in PLUTO.
 
    Erwartete Tabellenstruktur laut Vorgabe:
 
    Lesetabelle (z. B. t7.TA_DA_PLUTO_SP_2025)
    ------------------------------------------------
    DIM_ZEIT        INTEGER (YYYYMMDD)
    DIM_KENNZAHL    VARCHAR
    DIM_PRODUKT     VARCHAR(50)
    DIM_SCHADENSTATUS VARCHAR(50)
    KENNZAHLWERT    DECIMAL(20,9)
 
    Schreibtabelle (z. B. t7.TA_DA_PLUTO_SP_2025_PROGNOSE)
    ------------------------------------------------------
    DIM_ZEIT        INTEGER (YYYYMMDD)
    DIM_KENNZAHL    VARCHAR
    DIM_PRODUKT     VARCHAR(50)
    DIM_SCHADENSTATUS VARCHAR(50)
    KENNZAHLWERT    DECIMAL(20,9)
    LOAD_DATE       DATE
    """
 
    def __init__(
        self,
        kpi_list: List[str] | None = None,
        product_list: List[str] | None = None,
        status_list: List[str] | None = None,
        source_table: str | None = None,
        target_table: str | None = None,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self.username = os.environ["DB2_USERNAME"]
        self.password = os.environ["DB2_PASSWORT"]
        self.host = os.environ["DB2_HOST"]
        self.db_name = os.environ["DB2_DB_NAME"]
        self.port = os.environ["DB2_PORT"]
        self.schema = os.environ["DB2_SCHEMA"] 
 
        # Kennzahlen, Produkte, Status
        self.kpi_list = kpi_list or [
            "TERM_EINGANG_SCHRIFTST",
            "TERM_EINGANG_SONST",
        ]
        self.product_list = product_list or [
            "KFZ_Vollkasko",
            "KFZ_Teilkasko",
            "KFZ_Haftpflicht",
            "KFZ_Rest",
            "HUS_Haftpflicht",
            "HUS_Wohngebäude",
            "HUS_Hausrat",
            "HUS_Rest",
        ]
        self.status_list = status_list or [
            "Neuschaden",
            "Folgebearbeitung",
        ]
 
        if source_table is None:
            self.source_table = f"{self.schema}.TA_DA_PLUTO_SP_2025"
        else:
            self.source_table = source_table
 
        if target_table is None:
            self.target_table = f"{self.schema}.TA_DA_PLUTO_SP_2025_PROGNOSE"
        else:
            self.target_table = target_table
 
        dsn = [
            f"DATABASE={self.db_name};",
            f"HOSTNAME={self.host};",
            f"PORT={self.port};",
            "PROTOCOL=TCPIP;",
            f"UID={self.username};",
            f"PWD={self.password};",
        ]
        try:
            self._connection: ibm_db.IBM_DBConnection | None = ibm_db.connect("".join(dsn), "", "")
            self._conn: ibm_db_dbi.Connection = ibm_db_dbi.Connection(self._connection)
            self._logger.info("Connection to DB2 database was successful.")
            self._logger.info(f"Using source table: {self.source_table}")
            self._logger.info(f"Using target table: {self.target_table}")
        except Exception as e:
            self._connection = None
            self._logger.error(f"Error while connecting to DB2 database: {e}")
            raise
 
    @staticmethod
    def _dim_zeit_to_datetime(dim_zeit: int) -> datetime:
        return datetime.strptime(str(dim_zeit), "%Y%m%d")
 
    def read_timeseries(self) -> pd.DataFrame:
        """
        Liest die multivariaten Termineingänge aus DB2 und liefert ein breites DataFrame.
        """
        placeholders = ", ".join(["?"] * len(self.kpi_list))
        query = f"""
            SELECT
                DIM_ZEIT,
                DIM_KENNZAHL,
                DIM_PRODUKT,
                DIM_SCHADENSTATUS,
                KENNZAHLWERT
            FROM {self.source_table}
            WHERE DIM_KENNZAHL IN ({placeholders})
        """
 
        try:
            df_long = pd.read_sql(
                sql=query,
                con=self._conn,
                params=self.kpi_list,
            )
        except Exception as e:
            self._logger.error(f"Error while reading time series from DB2: {e}")
            raise
 
        if df_long.empty:
            self._logger.warning("No data returned from source table.")
            return pd.DataFrame()
 
        # DIM_ZEIT (int YYYYMMDD) → Timestamp
        df_long["date"] = df_long["DIM_ZEIT"].astype(int).apply(self._dim_zeit_to_datetime)
        df_long["date"] = pd.to_datetime(df_long["date"])
 
        # Pivot: breite Matrix je (Kennzahl, Produkt, Status)
        df_pivot = df_long.pivot_table(
            index="date",
            columns=["DIM_KENNZAHL", "DIM_PRODUKT", "DIM_SCHADENSTATUS"],
            values="KENNZAHLWERT",
            aggfunc="sum",
        )
 
        df_pivot = df_pivot.sort_index()
 
        # Spaltennamen wie im Modell erwartet: KENNZAHL__PRODUKT__STATUS
        df_pivot.columns = [
            f"{kpi}__{produkt}__{status}"
            for (kpi, produkt, status) in df_pivot.columns.to_list()
        ]
 
        self._logger.info(
            "Loaded multivariate time series from DB2. "
            f"Rows: {len(df_pivot)}, Columns (components): {len(df_pivot.columns)}"
        )
 
        return df_pivot
 
    @staticmethod
    def _parse_component_name(col_name: str) -> Tuple[str, str, str]:
        parts = col_name.split("__")
        if len(parts) != 3:
            raise ValueError(f"Unexpected component name format: {col_name}")
        return parts[0], parts[1], parts[2]
 
    def write_forecast(self, forecast: pd.DataFrame) -> None:
        """
        Schreibt die multivariate Prognose in die Prognosetabelle.
        Erwartete Zielstruktur: DIM_ZEIT, DIM_KENNZAHL, DIM_PRODUKT,
        DIM_SCHADENSTATUS, KENNZAHLWERT, LOAD_DATE.
        """
        if forecast.empty:
            self._logger.warning("Forecast DataFrame is empty. Nothing to write.")
            return
 
        cursor: ibm_db_dbi.Cursor = self._conn.cursor()
        table_name = self.target_table
 
        try:
            truncate_sql = f"TRUNCATE TABLE {table_name} IMMEDIATE"
            cursor.execute(truncate_sql)
 
            insert_sql = f"""
                INSERT INTO {table_name} (
                    DIM_ZEIT,
                    DIM_KENNZAHL,
                    DIM_PRODUKT,
                    DIM_SCHADENSTATUS,
                    KENNZAHLWERT,
                    LOAD_DATE
                )
                VALUES (?, ?, ?, ?, ?, CURRENT DATE)
            """
 
            for idx, row in forecast.iterrows():
                ts: pd.Timestamp = pd.to_datetime(idx)
                dim_zeit = int(ts.strftime("%Y%m%d"))
 
                for col_name, value in row.items():
                    if pd.isna(value):
                        continue
 
                    dim_kennzahl, dim_produkt, dim_status = self._parse_component_name(col_name)
                    kennzahlwert = float(value)
 
                    cursor.execute(
                        operation=insert_sql,
                        parameters=(
                            dim_zeit,
                            dim_kennzahl,
                            dim_produkt,
                            dim_status,
                            kennzahlwert,
                        ),
                    )
 
            self._conn.commit()
            self._logger.info("Forecast successfully written to DB2.")
        except Exception as e:
            self._logger.error(f"Error while writing forecast to DB2: {e}")
            self._conn.rollback()
            raise
 
    def close_connection(self) -> None:
        if getattr(self, "_connection", None):
            ibm_db.close(self._connection)
            self._logger.info("DB2 connection successfully closed.")



schaden_forecast_job.sh:

#!/bin/bash
 
# Beende bei jedem Fehler
set -e
 
echo "Setup SSH..."
# Sichere Berechtigungen für SSH-Key
chmod 400 ~/.ssh/id_rsa
# Add TFS zu known_hosts, falls nicht vorhanden
if ! grep -q "tfs" ~/.ssh/known_hosts 2>/dev/null; then
  ssh-keyscan -p 22 tfs >> ~/.ssh/known_hosts
fi
 
# (Optional) Git SSL prüfen deaktivieren
git config --global http.sslVerify false || true
 
# Repository-Variablen
REPO_DIR="da-pluto-timeseries"
REPO_URL="ssh://tfs:22/web/DefaultCollection/GIT_Projects/_git/da-pluto-timeseries"
BRANCH="holidays_winsorizer"
 
echo "Lösche vorhandenes Verzeichnis und klone neu..."
# Verzeichnis löschen, falls vorhanden, und neu klonen
delete_and_clone() {
  if [ -d "$REPO_DIR" ]; then
    echo "$REPO_DIR existiert – lösche es"
    rm -rf "$REPO_DIR"
  fi
  echo "Repository wird geklont"
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
}
 
# Führe Löschung und Neu-Klonen durch
delete_and_clone
 
echo "Aktuelle Dateien im Repository:"
ls -alh .
 
# >>> conda initialize >>>
echo "Conda initialize..."
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
  eval "$__conda_setup"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  . "/opt/conda/etc/profile.d/conda.sh"
else
  export PATH="/opt/conda/bin:$PATH"
fi
unset __conda_setup
# <<< conda initialize <<<
 
echo "Activate Conda environment..."
conda activate generic
 
echo "Installiere Python-Dependencies..."
pip install u8darts[torch] icalendar
 
echo "Starte Forecast Job..."
python pluto_forecast_job.py
 
echo "Job abgeschlossen."




environment.yml:

name: da-pluto-timeseries
channels:
  - conda-forge
  - qc-huk
  - nodefaults
dependencies:
  - python>=3.8
  - make
  - numpydoc
  - pip
  - pre-commit
  - pytest>=6 # Adds --import-mode option
  - pytest-cov
  - setuptools-scm
  - sphinx
  - sphinxcontrib-apidoc
  - sphinx_rtd_theme




pyproject.toml:

[build-system]
requires = ['setuptools', 'setuptools-scm', 'wheel']

[tool.setuptools_scm]
version_scheme = "post-release"

[tool.black]
exclude = '''
/(
    \.eggs
  | \.git
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
include_trailing_comma = true
ensure_newline_before_comments = true
line_length = 88
known_first_party = "da_pluto_timeseries"
skip_glob = '\.eggs/*,\.git/*,\.venv/*,build/*,dist/*'
default_section = 'THIRDPARTY'

[tool.mypy]
python_version = '3.8'
ignore_missing_imports = true
no_implicit_optional = true
check_untyped_defs = true


[tool.docformatter]
pre-summary-newline = true
recursive = true
wrap-descriptions = 88
wrap-summaries = 88

[tool.pytest.ini_options]
addopts = "--import-mode=importlib --cov=da_pluto_timeseries --cov-report term-missing"

