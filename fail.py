"""Predict expected claim amount (Schadenbedarf) for HF3 rows.

The risk models are pinned per ``(product, tariff generation, segment, variant)``
in :data:`CONFIG_PATH`.

Rows without a configured model keep a null prediction (never zero-filled, see
the ``schaden`` skill), unless ``strict=True`` asks for an error instead.
"""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from pricing.model.qcm.tracking import load_model_from_mlflow

from clv_analysis.products import Product, Segment

CONFIG_PATH = Path(__file__).parent / "config.json"

#: Column the predicted pure premium is written to.
PREDICTION_COLUMN = "expected_claim_amount"

DEFAULT_TARIFF_GEN = "202608"
DEFAULT_VARIANT = "best_estimate"


@dataclass(frozen=True)
class ModelRef:
    """An MLflow run pinned for one ``(product, segment)`` combination."""

    product: Product
    segment: Segment
    tariff_gen: str
    variant: str
    run_id: str
    tracking_uri: str

    @property
    def key(self) -> tuple[Product, Segment]:
        """The ``(product, segment)`` combination this run is pinned for."""
        return self.product, self.segment


def read_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the raw model configuration JSON."""
    return json.loads(path.read_text())


def model_refs(
    *,
    tariff_gen: str = DEFAULT_TARIFF_GEN,
    variant: str = DEFAULT_VARIANT,
    config: dict[str, Any] | None = None,
) -> dict[tuple[Product, Segment], ModelRef]:
    """Collect the configured runs for one tariff generation and variant.

    Config keys that are not :class:`Product` values are ignored; missing
    ``tariff_gen`` / ``variant`` entries are skipped.
    """
    config = config if config is not None else read_config()
    tracking_uri = config["tracking_uri"]

    product_values = {p.value for p in Product}

    refs: dict[tuple[Product, Segment], ModelRef] = {}
    for product_value, per_gen in config.items():
        if product_value not in product_values:
            continue
        for segment_value, per_variant in per_gen.get(tariff_gen, {}).items():
            run_id = per_variant.get(variant)
            if run_id is None:
                continue
            ref = ModelRef(
                product=Product(product_value),
                segment=Segment(segment_value),
                tariff_gen=tariff_gen,
                variant=variant,
                run_id=run_id,
                tracking_uri=tracking_uri,
            )
            refs[ref.key] = ref
    return refs


@cache
def load_model(ref: ModelRef) -> Any:
    """Load the model of a :class:`ModelRef`."""
    _, _, model = load_model_from_mlflow(
        run_id=ref.run_id,
        tracking_uri=ref.tracking_uri,
    )
    return model


def predict(
    df: pl.DataFrame,
    *,
    tariff_gen: str = DEFAULT_TARIFF_GEN,
    variant: str = DEFAULT_VARIANT,
    config: dict[str, Any] | None = None,
    strict: bool = False,
) -> pl.DataFrame:
    """Add :data:`PREDICTION_COLUMN` with the per-row predicted pure premium.

    ``df`` must carry the ``product`` and ``segment`` columns added by
    :func:`~clv_analysis.hf3_etl.load_parquet`. One model is applied
    per ``(product, segment)`` group; groups without a configured model stay
    null, or raise when ``strict`` is set.
    """
    missing = {"product", "segment"} - set(df.columns)
    if missing:
        raise KeyError(f"df is missing the model key columns {sorted(missing)}")

    refs = model_refs(tariff_gen=tariff_gen, variant=variant, config=config)

    row_column = "__row_index"
    # Single pass split instead of one full-frame filter per group.
    groups = dict(
        sorted(
            df.with_row_index(row_column)
            .partition_by("product", "segment", as_dict=True, include_key=False)
            .items()
        )
    )

    # Validate before predicting, so ``strict`` fails before any expensive work.
    for (product_value, segment_value), group in groups.items():
        if (Product(product_value), Segment(segment_value)) in refs:
            continue
        message = (
            f"no model configured for {product_value}/{segment_value} "
            f"({tariff_gen}, {variant}): {group.height:,} rows"
        )
        if strict:
            raise KeyError(message)
        logger.warning(message)

    predictions = np.zeros(df.height)
    has_prediction = np.zeros(df.height, dtype=bool)
    for (product_value, segment_value), group in groups.items():
        key = (Product(product_value), Segment(segment_value))
        if key not in refs:
            continue

        rows = group[row_column].to_numpy()
        logger.info(
            "predicting {}/{} | {:,} rows | run_id {}",
            product_value,
            segment_value,
            group.height,
            refs[key].run_id,
        )
        features = group.drop(row_column).to_pandas()
        predictions[rows] = np.asarray(load_model(refs[key]).predict(features))
        has_prediction[rows] = True

    # Rows without a model become null (not NaN); NaNs from a model are kept.
    return df.with_columns(
        pl.when(pl.Series(has_prediction))
        .then(pl.Series(predictions))
        .alias(PREDICTION_COLUMN)
    )
