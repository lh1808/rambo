"""Training data for the PKW CLV model: flatfile rows with PKW plus the ``clv`` target."""

from pathlib import Path

import polars as pl
import polars.selectors as cs
from loguru import logger

from clv_analysis.expressions import clv
from clv_analysis.flatfile import load_flatfile
from clv_analysis.products import Product, Segment


def write_train_data(
    path: Path,
    features: list[str],
    remainders: list[int] | None = None,
    years_ahead: int = 3,
    beta: float = 0.95,
    claim_cap: float = 25_000,
) -> pl.DataFrame:
    """Compute ``clv`` on the flatfile buckets ``remainders`` (all if None), then
    keep rows with a full horizon where the partner has at least one PKW segment
    running on 1 Jan of the year. Only ``features`` and the target columns are
    kept and collected.

    ``clv`` is computed before any rows are dropped, so later years still count
    towards earlier years (no censoring from the filter). ``sample`` splits
    train/test by partner, so a partner's years never end up on both sides.

    ``target`` is the model outcome: on train it is ``clv_capped`` (claim amounts
    per partner x year x segment capped at ``claim_cap``), on test it is the real
    ``clv``. So large claims do not dominate the fit, but test metrics are on CLV.
    """
    flat = load_flatfile(remainders)
    segments = [
        (Product.PKW, Segment.KH),
        (Product.PKW, Segment.VK),
        (Product.PKW, Segment.TK),
    ]
    capped = clv(
        flat.with_columns(cs.starts_with("claim_amount_").clip(upper_bound=claim_cap)),
        years_ahead=years_ahead,
        beta=beta,
        segments=segments,
    ).select("partner_id", "year", clv_capped="clv")

    out = (
        clv(flat, years_ahead=years_ahead, beta=beta, segments=segments)
        .join(capped, on=["partner_id", "year"], how="left", validate="1:1")
        .filter(
            pl.col("year") <= pl.col("year").max() - years_ahead + 1,
            pl.any_horizontal(cs.starts_with("yearly_premium_jan1_PKW|").is_not_null()),
            pl.col("year") > pl.col("year").min(),
        )
        .with_columns(
            sample=pl.when(pl.col("partner_id").hash(seed=0) % 100 < 80)
            .then(pl.lit("train"))
            .otherwise(pl.lit("test"))
        )
        .with_columns(
            target=pl.when(pl.col("sample") == "train")
            .then(pl.col("clv_capped"))
            .otherwise(pl.col("clv"))
        )
        .select(
            "partner_id", "year", "sample", "target", "clv", "clv_capped", *features
        )
        .collect(engine="streaming")
    )
    logger.info(
        "train data: {:,} rows, years {}-{}",
        out.height,
        out["year"].min(),
        out["year"].max(),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    logger.info("written -> {}", path)
    return out
