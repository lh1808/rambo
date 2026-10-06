"""Reusable polars expressions."""

import polars as pl
import polars.selectors as cs
from polars.selectors import Selector

from clv_analysis.products import Product, Segment, Source


def product() -> pl.Expr:
    """Product of an HF3 row, deduced form the Source and WKZ.

    Notes
    -----
    Unknown Wagniskennziffern give null.
    """
    enum = pl.Enum(Product)
    wkz = pl.col("wkz")
    return (
        pl.when(pl.col("source") == Source.WG)
        .then(pl.lit(Product.WG, dtype=enum))
        .when(pl.col("source") == Source.PH)
        .then(pl.lit(Product.PH, dtype=enum))
        .when(wkz == "112")
        .then(pl.lit(Product.PKW, dtype=enum))
        .when(wkz.is_in(["001", "003", "014", "024", "030", "031"]))
        .then(pl.lit(Product.KRAFTRAEDER, dtype=enum))
        .when(wkz == "127")
        .then(pl.lit(Product.CAMPINGFAHRZEUGE, dtype=enum))
    )


def key() -> pl.Expr:
    """Flatfile column key from product and segment: PKW_KH, WG_FEUER."""
    return pl.concat_str("product", "segment", separator="|").str.to_uppercase()


def jan1() -> pl.Expr:
    """1 Jan of ``year``: the date a partner x year is described at."""
    return pl.datetime(pl.col("year"), 1, 1, time_unit="us")


def valid_on_jan1(start: str, end: str) -> pl.Expr:
    """The row is valid on 1 Jan of ``year``: start <= 1 Jan < end."""
    return (pl.col(start) <= jan1()) & (jan1() < pl.col(end))


def started_by_jan1(start: str) -> pl.Expr:
    """The row started on or before 1 Jan of ``year``."""
    return pl.col(start) <= jan1()


def identifiers() -> list[str]:
    return ["partner_id", "vertragsakte_id", "vertrag_id", "deckung_id"]


def available(
    effective_from: str = "effective_from_date",
    risk_count: str = "anz_risiken",
) -> list[pl.Expr]:
    """Aggregations for group_by(identifiers()).agg(...): active_from, active_to."""
    return [
        pl.col(effective_from).min().alias("active_from"),
        pl.col(effective_from).filter(pl.col(risk_count) == 0).min().alias("active_to"),
    ]


def to_intervals(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse raw deckung rows into one [active_from, active_to) row per deckung."""
    carry = [c for c in ("segment", "product") if c in df.columns]
    return (
        df.group_by([*identifiers(), *carry])
        .agg(available())
        .with_columns(
            status=pl.when(pl.col("active_to").is_null())
            .then(pl.lit("active"))
            .otherwise(pl.lit("ended"))
        )
    )


def clv(
    flat: pl.LazyFrame,
    years_ahead: int | None = None,
    beta: float = 0.95,
    segments: list[tuple[Product, Segment]] | None = None,
) -> pl.LazyFrame:
    """Add ``clv`` per partner x year: earned premium - claim amount."""
    if segments is None:
        premium = cs.starts_with("earned_premium_")
        claims = cs.starts_with("claim_amount_")
    else:
        keys = (
            pl.DataFrame(segments, schema=["product", "segment"], orient="row")
            .select(key())
            .to_series()
        )
        premium = cs.by_name([f"earned_premium_{k}" for k in keys])
        claims = cs.by_name([f"claim_amount_{k}" for k in keys])

    net = flat.select(
        "partner_id",
        future_year="year",
        net=pl.sum_horizontal(premium) - pl.sum_horizontal(claims),
    )
    horizon = pl.col("future_year") >= pl.col("year")
    if years_ahead is not None:
        horizon = horizon & (pl.col("future_year") < pl.col("year") + years_ahead)

    value = (
        flat.select("partner_id", "year")
        .join(net, on="partner_id")
        .filter(horizon)
        .group_by("partner_id", "year")
        .agg(
            clv=(
                pl.col("net")
                * pl.lit(beta)
                ** (pl.col("future_year") - pl.col("year")).cast(pl.Int32)
            ).sum()
        )
    )
    return flat.join(value, on=["partner_id", "year"], how="left", validate="1:1")


def baseline_clv(
    flat: pl.LazyFrame,
    segments: list[tuple[Product, Segment]] | None = None,
) -> pl.LazyFrame:
    """Add ``baseline_clv`` per partner x year from the Kundenwert baseline columns.

    The baseline only predicts the year from 1 Jan of ``year``:
    ``kw_premium_<KEY>`` minus ``kw_expected_claim_amount_<KEY>``.
    """
    if segments is None:
        premium: Selector = cs.starts_with("kw_premium_")
        claims: Selector = cs.starts_with("kw_expected_claim_amount_")
    else:
        keys = [f"{p}|{s}".upper() for p, s in segments]
        premium = cs.by_name([f"kw_premium_{k}" for k in keys])
        claims = cs.by_name([f"kw_expected_claim_amount_{k}" for k in keys])

    return flat.with_columns(
        baseline_clv=pl.sum_horizontal(premium) - pl.sum_horizontal(claims)
    )

