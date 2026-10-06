"""Evaluate CLV models on the same partner-year rows.

``evaluate`` computes the observed ``clv`` per partner x year, keeps the rows
with a full horizon up to the case year where the partner has a segment of the
case running on 1 Jan, and splits them train/test by partner. Each model is
fitted on the train rows (all years up to the case year) and predicts the test
rows of the case year; every ``pred_*`` column it returns is scored against
``clv``.
"""

from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
import polars as pl
from quantcore.learn.metrics import metrics_table
from scipy.stats import pearsonr, spearmanr

from clv_analysis.expressions import clv
from clv_analysis.flatfile import load_flatfile
from clv_analysis.products import Product, Segment


@dataclass(frozen=True)
class Case:
    """The prediction problem: the CLV of ``product_segments`` over
    ``years_ahead`` years from 1 Jan of ``year``, discounted by ``beta``."""

    year: int
    years_ahead: int
    beta: float
    product_segments: list[tuple[Product, Segment]]


class Model(Protocol):
    def fit(self, train: pl.DataFrame, case: Case, remainders: list[int]) -> Self:
        """Fit on ``partner_id``, ``year``, ``sample`` and ``clv`` of the train
        rows; further data can be loaded by the model from the buckets
        ``remainders``."""
        ...

    def predict(
        self, test: pl.DataFrame, case: Case, remainders: list[int]
    ) -> pl.DataFrame:
        """``partner_id``, ``year`` and one or more ``pred_*`` columns for the
        test rows (``partner_id``, ``year``, ``sample``)."""
        ...


def evaluate(models: list[Model], case: Case, remainders: list[int]) -> pl.DataFrame:
    """One row per ``pred_*`` column of the models: ``n``, ``n_nans`` and the
    metrics against ``clv`` on the test rows of ``case.year``, on the flatfile
    buckets ``remainders``. Rows with a null or NaN prediction count towards
    ``n_nans`` and are left out of that column's metrics."""
    keys = [f"{p}|{s}".upper() for p, s in case.product_segments]
    rows = (
        clv(
            load_flatfile(remainders),
            years_ahead=case.years_ahead,
            beta=case.beta,
            segments=case.product_segments,
        )
        .filter(
            pl.col("year") > pl.col("year").min(),
            pl.col("year") <= pl.col("year").max() - case.years_ahead + 1,
            pl.col("year") <= case.year,
            pl.any_horizontal(
                pl.col(f"yearly_premium_jan1_{k}").is_not_null() for k in keys
            ),
        )
        .select(
            "partner_id",
            "year",
            sample=pl.when(pl.col("partner_id").hash(seed=0) % 100 < 80)
            .then(pl.lit("train"))
            .otherwise(pl.lit("test")),
            clv="clv",
        )
        .collect(engine="streaming")
    )
    train = rows.filter(pl.col("sample") == "train")
    test = rows.filter(pl.col("sample") == "test", pl.col("year") == case.year)

    scored = test.select("partner_id", "year", "clv")
    for model in models:
        model.fit(train, case, remainders)
        scored = scored.join(
            model.predict(test.drop("clv"), case, remainders).select(
                "partner_id", "year", pl.col("^pred_.*$")
            ),
            on=["partner_id", "year"],
            how="left",
            validate="1:1",
        )

    out = []
    for prediction in [c for c in scored.columns if c.startswith("pred_")]:
        valid = pl.col(prediction).is_not_null() & pl.col(prediction).is_not_nan()
        kept = scored.filter(valid)
        table = metrics_table(
            None,
            y=kept["clv"].to_numpy(),
            mu=kept[prediction].to_numpy(),
            other_metrics={
                "pearson": lambda y, mu, w: pearsonr(y, mu).statistic,
                "spearman": lambda y, mu, w: spearmanr(y, mu).statistic,
                "bias": lambda y, mu, w: np.mean(mu - y),
                "median_outcome": lambda y, mu, w: np.median(y),
                "median_prediction": lambda y, mu, w: np.median(mu),
            },
        )
        out.append(
            {
                "prediction": prediction,
                "n": scored.height,
                "n_nans": scored.height - kept.height,
            }
            | table.to_dict()
        )
    return pl.DataFrame(out)

