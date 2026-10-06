"""Reading of HF3 sources into polars frames with the same columns for every source."""

import polars as pl

from clv_analysis.expressions import product
from clv_analysis.products import Division, Segment

from . import _columns as cols
from ._config import Source, dataset


def scan_parquet(
    source: Source,
    years: list[int] | None = None,
    segments: list[Segment] | None = None,
) -> pl.LazyFrame:
    """Scan one source with the same columns for every source.

    All partitions go into one scan: one scan per partition makes the streaming
    engine exceed POLARS_MAX_BLOCKING_THREAD_COUNT. Segment and year come from
    the hive partition directories.
    """
    spec = dataset(source)
    files = []
    for segment in segments or spec.segments:
        for year in years or spec.years(segment):
            partition = spec.partition(segment, year)
            partition_files = sorted(partition.glob("*.parquet"))
            if not partition_files:
                raise FileNotFoundError(f"no parquet files under {partition}")
            files.extend(partition_files)
    return (
        pl.scan_parquet(
            files,
            hive_partitioning=True,
            hive_schema={spec.segment_key: pl.String, spec.year_key: pl.String},
        )
        .select(
            source=pl.lit(source, dtype=pl.Enum(Source)),
            division=pl.lit(spec.division, dtype=pl.Enum(Division)),
            # string ops, not replace_strict: a lookup dict here costs ~6 GB per 2 years
            # on disk: "kh" / "KH" / "amts-vermoegens-..." and 2-digit "21"
            segment=pl.col(spec.segment_key)
            .str.to_lowercase()
            .str.replace_all("-", "_")
            .cast(pl.Enum(Segment)),
            year=pl.col(spec.year_key).cast(pl.Int16) + 2000,
            vertragsakte_id=pl.col(cols.AKTE).cast(pl.Int64),
            exposure=pl.col(cols.EXPOSURE[source]).cast(pl.Float64),
            yearly_premium=pl.col(cols.YEARLY_PREMIUM[source]).cast(pl.Float64),
            # timeslice start and end, cut at the statistics year
            risk_start=pl.col(cols.RISIKO_BEGINN).cast(pl.Date),
            risk_end=pl.col(cols.RISIKO_ABLAUF).cast(pl.Date),
            claim_amount=pl.col(cols.SCHADEN_HOEHE[source]).cast(pl.Float64),
            claim_count=pl.col(cols.SCHADEN_ANZAHL).cast(pl.Float64),
            # "YYYY-MM-DD" string in KFZ, timestamp in WG/PH
            birth_date=pl.col(cols.GEBURTSDATUM).str.to_date("%Y-%m-%d", strict=False)
            if source is Source.KFZ
            else pl.col(cols.GEBURTSDATUM).cast(pl.Date),
            # only KFZ has a Wagniskennziffer
            wkz=pl.col(cols.WAGNISKENNZIFFER).cast(pl.String)
            if source is Source.KFZ
            else pl.lit(None, dtype=pl.String),
        )
        .with_columns(product=product())
    )


def load_parquet(
    source: Source,
    years: list[int] | None = None,
    segments: list[Segment] | None = None,
) -> pl.DataFrame:
    """Load one source into an eager frame."""
    return scan_parquet(source, years, segments).collect(engine="streaming")
