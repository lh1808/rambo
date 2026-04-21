"""
Modellierungsbausteine: Transformer-Pipeline für Ziel- und Kovariatenserien
sowie Factory für das TFT-Modell.

Bestandteile:

- :class:`RollingWinsorizer` – robuster Rolling/Expanding-Winsorizer als
  erste Stufe der Ziel-Pipeline.
- :func:`build_transformers` – liefert die Standard-Pipeline
  (Winsorizer → Yeo-Johnson → Standardisierung) und einen separaten
  Skalierer für Past-Kovariaten.
- :func:`build_tft` – Factory für das darts :class:`TFTModel` mit den in
  :class:`TftConfig` hinterlegten Hyperparametern.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers import (
    FittableDataTransformer,
    InvertibleDataTransformer,
    Scaler,
)
from darts.models import TFTModel
from darts.utils.likelihood_models import NegativeBinomialLikelihood, PoissonLikelihood
from sklearn.preprocessing import PowerTransformer

from .config import HorizonConfig, TftConfig


@dataclass
class TransformArtifacts:
    """
    Container für die Transformer einer Modellinstanz.

    Attributes
    ----------
    y_pipeline
        Pipeline für die Zielreihe (Winsorizer → Yeo-Johnson →
        Standardisierung).
    x_scaler
        Einfacher MinMax-/Standard-Scaler für Past-Kovariaten.
    """

    y_pipeline: Pipeline
    x_scaler: Scaler


class RollingWinsorizer(FittableDataTransformer, InvertibleDataTransformer):
    """
    Begrenzt Ausreißer auf rollende oder expandierende Quantilgrenzen.

    Je Komponente und Zeitpunkt werden aus einem Rolling- oder
    Expanding-Fenster das untere (``lower_q``) und obere (``upper_q``)
    Quantil geschätzt und die Werte auf dieses Intervall geklammert.

    Klassenhierarchie
    -----------------
    Die Mehrfach-Vererbung ist notwendig, weil darts'
    :class:`~darts.dataprocessing.pipeline.Pipeline` ihre Invertierbarkeit
    per ``isinstance(t, InvertibleDataTransformer)`` bestimmt. Ohne
    ``InvertibleDataTransformer`` in den Basisklassen würde
    ``pipeline.inverse_transform(...)`` mit *"Not all transformers in the
    pipeline can perform inverse_transform"* abbrechen. Das Inverse ist
    bewusst die Identität – Winsorizing ist fachlich nicht umkehrbar, und
    genau diese Semantik wird hier dokumentiert.

    Parameters
    ----------
    lower_q, upper_q
        Untere/obere Quantilschwelle, ``0 <= lower_q < upper_q <= 1``.
    mode
        ``"rolling"`` oder ``"expanding"``.
    window
        Fenstergröße im Rolling-Modus.
    min_periods
        Mindest-Anzahl gültiger Beobachtungen fürs Rolling-Quantil.
        ``None`` → automatisch ``max(5, 0.3 * window)``.
    ema_alpha
        Optional: EMA-Glättung der Quantilgrenzen (0 < alpha < 1).
    integer
        Wenn ``True``, Grenzen auf nächstgültige Ganzzahlen runden.
    name, n_jobs, verbose
        Standard-Parameter von :class:`FittableDataTransformer`.
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
        """Schätzt Quantilgrenzen pro Zeitpunkt und Komponente."""
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

        # Komplett-NaN-Spalten gegen 0 tauschen, damit das Rolling-Quantil
        # nicht global NaN liefert.
        if df.isna().all().any():
            df = df.copy()
            for c in df.columns[df.isna().all()]:
                df[c] = 0.0

        if mode == "rolling":
            if min_per is None:
                min_per = max(5, int(0.3 * window))
            low_df = df.rolling(window=window, min_periods=min_per).quantile(lower_q)
            high_df = df.rolling(window=window, min_periods=min_per).quantile(upper_q)

            # Für die Aufwärmphase (weniger als ``window`` Beobachtungen)
            # auf expandierende Quantile zurückfallen.
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

        # Sicherheitsnetz: low darf nie über high liegen.
        low_df = np.minimum(low_df, high_df)
        low_df = low_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        high_df = high_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

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
        """Klammert Werte auf die beim Fit gelernten Quantilgrenzen."""
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
            # Fit- und Transform-Index weichen ab – Grenzen passend
            # reindexen. Fehlende Zeitpunkte: kein Clipping (±inf).
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
        """
        Identität: Winsorizing ist per Konstruktion nicht umkehrbar. Die
        Methode existiert, damit die umgebende Pipeline als invertierbar
        klassifiziert bleibt.
        """
        return series

    @property
    def supports_inverse_transform(self) -> bool:
        return True


def build_transformers() -> TransformArtifacts:
    """
    Baut die Standard-Transformer für Ziel- und Kovariatenzeitreihen.

    Die Ziel-Pipeline ist dreistufig:

    1. :class:`RollingWinsorizer` mit ``[0.01, 0.90]`` über ein Fenster
       von 26 Wochen.
    2. Yeo-Johnson-Transform (ohne Standardisierung), um Heteroskedastie
       und Schiefe zu reduzieren.
    3. Standardisierung (Mean/Std) als finale Skalierung.

    Die Kovariaten bekommen einen eigenen Skalierer (MinMax per Default).

    Returns
    -------
    TransformArtifacts
        Container mit ``y_pipeline`` und ``x_scaler``.
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
    """Wählt das Likelihood-Objekt anhand des Konfig-Strings."""
    if name is None:
        return None
    name = name.lower()
    if name == "poisson":
        return PoissonLikelihood()
    if name in ("nb", "negative_binomial", "negativebinomial"):
        return NegativeBinomialLikelihood()
    raise ValueError(f"Unbekannte Likelihood-Konfiguration: {name}")


def _build_optimizer_and_loss(tft_cfg: TftConfig):
    """Löst Optimizer-Klasse und Loss-Funktion aus der Konfiguration auf."""
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
    Baut ein darts :class:`TFTModel` mit den konfigurierten Hyperparametern.

    Besonderheiten
    --------------
    - Bei aktiver Likelihood (probabilistisches Modell) wird ``loss_fn``
      nicht mitgegeben, da darts eine gemeinsame Vergabe ablehnt.
    - PyTorch Lightning wird per ``pl_trainer_kwargs`` ruhiggestellt:
      kein Progress-Bar, kein Model-Summary, kein TensorBoard-Logger.
    - Ist ``tft_cfg.early_stopping_patience > 0``, wird ein
      ``EarlyStopping``-Callback auf den Training-Loss gesetzt. Das
      Training bricht ab, wenn sich der Loss über ``patience`` Epochen
      nicht mehr verbessert — spart Rechenzeit und verhindert unnötiges
      Weitertrainieren auf kleinen Datasets.

    Parameters
    ----------
    hcfg
        Horizont-Konfiguration (liefert Input-/Output-Chunk-Längen).
    tft_cfg
        TFT-Hyperparameter.

    Returns
    -------
    TFTModel
        Untrainierte Modellinstanz.
    """
    likelihood_obj = _build_likelihood(tft_cfg.likelihood)
    optimizer_cls, loss_fn = _build_optimizer_and_loss(tft_cfg)
    loss_fn_arg = None if likelihood_obj is not None else loss_fn

    callbacks = []
    if tft_cfg.early_stopping_patience > 0:
        try:
            from pytorch_lightning.callbacks import EarlyStopping
        except ImportError:
            from lightning.pytorch.callbacks import EarlyStopping

        callbacks.append(
            EarlyStopping(
                monitor="train_loss",
                patience=tft_cfg.early_stopping_patience,
                min_delta=1e-4,
                mode="min",
                verbose=False,
            )
        )

    pl_trainer_kwargs = {
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
        "callbacks": callbacks,
    }

    return TFTModel(
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
        loss_fn=loss_fn_arg,
        use_static_covariates=tft_cfg.use_static_covariates,
        add_relative_index=tft_cfg.add_relative_index,
        add_encoders=None,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs={"lr": tft_cfg.learning_rate},
        pl_trainer_kwargs=pl_trainer_kwargs,
    )
