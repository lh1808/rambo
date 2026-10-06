clv_pkw.json

{
  "estimator": "lgbm",
  "outcome": "target",
  "distribution": "normal",
  "data": {
    "path": ".data/train/clv_pkw.parquet"
  },
  "sample": "sample",
  "features": [
    "age",
    "n_deckungen",
    "n_products",
    "years_since_first_contract",
    "n_contracts",
    "share_annual_payment",
    "any_kombi_bonus",
    "familienstand",
    "anzahl_kinder",
    "online_huk24",
    "online_hukde",
    "versandweg_huk24",
    "n_dunnings",
    "max_dunning_level",
    "prev_exposure_CAMPINGFAHRZEUGE|ASSV",
    "prev_exposure_CAMPINGFAHRZEUGE|FAU",
    "prev_exposure_CAMPINGFAHRZEUGE|KH",
    "prev_exposure_CAMPINGFAHRZEUGE|KU",
    "prev_exposure_CAMPINGFAHRZEUGE|KUFU",
    "prev_exposure_CAMPINGFAHRZEUGE|SBR",
    "prev_exposure_CAMPINGFAHRZEUGE|TK",
    "prev_exposure_CAMPINGFAHRZEUGE|VK",
    "prev_exposure_KRAFTRAEDER|ASSV",
    "prev_exposure_KRAFTRAEDER|KH",
    "prev_exposure_KRAFTRAEDER|KU",
    "prev_exposure_KRAFTRAEDER|SBR",
    "prev_exposure_KRAFTRAEDER|TK",
    "prev_exposure_KRAFTRAEDER|VK",
    "prev_exposure_PH|AMTS_VERMOEGENS_HAFTPFLICHT_UND_ZUSATZ",
    "prev_exposure_PH|PLUS",
    "prev_exposure_PH|PRIVATHAFTPFLICHT",
    "prev_exposure_PKW|ASSV",
    "prev_exposure_PKW|FAU",
    "prev_exposure_PKW|KH",
    "prev_exposure_PKW|KU",
    "prev_exposure_PKW|KUFU",
    "prev_exposure_PKW|SBR",
    "prev_exposure_PKW|TK",
    "prev_exposure_PKW|UMD",
    "prev_exposure_PKW|VK",
    "prev_exposure_WG|FEUER",
    "prev_exposure_WG|LEITUNGSWASSER",
    "prev_exposure_WG|NEBENPRODUKTE",
    "prev_exposure_WG|STURM",
    "prev_earned_premium_CAMPINGFAHRZEUGE|ASSV",
    "prev_earned_premium_CAMPINGFAHRZEUGE|FAU",
    "prev_earned_premium_CAMPINGFAHRZEUGE|KH",
    "prev_earned_premium_CAMPINGFAHRZEUGE|KU",
    "prev_earned_premium_CAMPINGFAHRZEUGE|KUFU",
    "prev_earned_premium_CAMPINGFAHRZEUGE|SBR",
    "prev_earned_premium_CAMPINGFAHRZEUGE|TK",
    "prev_earned_premium_CAMPINGFAHRZEUGE|VK",
    "prev_earned_premium_KRAFTRAEDER|ASSV",
    "prev_earned_premium_KRAFTRAEDER|KH",
    "prev_earned_premium_KRAFTRAEDER|KU",
    "prev_earned_premium_KRAFTRAEDER|SBR",
    "prev_earned_premium_KRAFTRAEDER|TK",
    "prev_earned_premium_KRAFTRAEDER|VK",
    "prev_earned_premium_PH|AMTS_VERMOEGENS_HAFTPFLICHT_UND_ZUSATZ",
    "prev_earned_premium_PH|PLUS",
    "prev_earned_premium_PH|PRIVATHAFTPFLICHT",
    "prev_earned_premium_PKW|ASSV",
    "prev_earned_premium_PKW|FAU",
    "prev_earned_premium_PKW|KH",
    "prev_earned_premium_PKW|KU",
    "prev_earned_premium_PKW|KUFU",
    "prev_earned_premium_PKW|SBR",
    "prev_earned_premium_PKW|TK",
    "prev_earned_premium_PKW|UMD",
    "prev_earned_premium_PKW|VK",
    "prev_earned_premium_WG|FEUER",
    "prev_earned_premium_WG|LEITUNGSWASSER",
    "prev_earned_premium_WG|NEBENPRODUKTE",
    "prev_earned_premium_WG|STURM",
    "prev_claim_amount_CAMPINGFAHRZEUGE|ASSV",
    "prev_claim_amount_CAMPINGFAHRZEUGE|FAU",
    "prev_claim_amount_CAMPINGFAHRZEUGE|KH",
    "prev_claim_amount_CAMPINGFAHRZEUGE|KU",
    "prev_claim_amount_CAMPINGFAHRZEUGE|KUFU",
    "prev_claim_amount_CAMPINGFAHRZEUGE|SBR",
    "prev_claim_amount_CAMPINGFAHRZEUGE|TK",
    "prev_claim_amount_CAMPINGFAHRZEUGE|VK",
    "prev_claim_amount_KRAFTRAEDER|ASSV",
    "prev_claim_amount_KRAFTRAEDER|KH",
    "prev_claim_amount_KRAFTRAEDER|KU",
    "prev_claim_amount_KRAFTRAEDER|SBR",
    "prev_claim_amount_KRAFTRAEDER|TK",
    "prev_claim_amount_KRAFTRAEDER|VK",
    "prev_claim_amount_PH|AMTS_VERMOEGENS_HAFTPFLICHT_UND_ZUSATZ",
    "prev_claim_amount_PH|PLUS",
    "prev_claim_amount_PH|PRIVATHAFTPFLICHT",
    "prev_claim_amount_PKW|ASSV",
    "prev_claim_amount_PKW|FAU",
    "prev_claim_amount_PKW|KH",
    "prev_claim_amount_PKW|KU",
    "prev_claim_amount_PKW|KUFU",
    "prev_claim_amount_PKW|SBR",
    "prev_claim_amount_PKW|TK",
    "prev_claim_amount_PKW|UMD",
    "prev_claim_amount_PKW|VK",
    "prev_claim_amount_WG|FEUER",
    "prev_claim_amount_WG|LEITUNGSWASSER",
    "prev_claim_amount_WG|NEBENPRODUKTE",
    "prev_claim_amount_WG|STURM",
    "prev_claim_count_CAMPINGFAHRZEUGE|ASSV",
    "prev_claim_count_CAMPINGFAHRZEUGE|FAU",
    "prev_claim_count_CAMPINGFAHRZEUGE|KH",
    "prev_claim_count_CAMPINGFAHRZEUGE|KU",
    "prev_claim_count_CAMPINGFAHRZEUGE|KUFU",
    "prev_claim_count_CAMPINGFAHRZEUGE|SBR",
    "prev_claim_count_CAMPINGFAHRZEUGE|TK",
    "prev_claim_count_CAMPINGFAHRZEUGE|VK",
    "prev_claim_count_KRAFTRAEDER|ASSV",
    "prev_claim_count_KRAFTRAEDER|KH",
    "prev_claim_count_KRAFTRAEDER|KU",
    "prev_claim_count_KRAFTRAEDER|SBR",
    "prev_claim_count_KRAFTRAEDER|TK",
    "prev_claim_count_KRAFTRAEDER|VK",
    "prev_claim_count_PH|AMTS_VERMOEGENS_HAFTPFLICHT_UND_ZUSATZ",
    "prev_claim_count_PH|PLUS",
    "prev_claim_count_PH|PRIVATHAFTPFLICHT",
    "prev_claim_count_PKW|ASSV",
    "prev_claim_count_PKW|FAU",
    "prev_claim_count_PKW|KH",
    "prev_claim_count_PKW|KU",
    "prev_claim_count_PKW|KUFU",
    "prev_claim_count_PKW|SBR",
    "prev_claim_count_PKW|TK",
    "prev_claim_count_PKW|UMD",
    "prev_claim_count_PKW|VK",
    "prev_claim_count_WG|FEUER",
    "prev_claim_count_WG|LEITUNGSWASSER",
    "prev_claim_count_WG|NEBENPRODUKTE",
    "prev_claim_count_WG|STURM",
    "prev_claim_count_total",
    "prev_claim_amount_total",
    "prev_earned_premium_total",
    "prev_exposure_total",
    "yearly_premium_jan1_CAMPINGFAHRZEUGE|ASSV",
    "yearly_premium_jan1_CAMPINGFAHRZEUGE|FAU",
    "yearly_premium_jan1_CAMPINGFAHRZEUGE|KH",
    "yearly_premium_jan1_CAMPINGFAHRZEUGE|KU",
    "yearly_premium_jan1_CAMPINGFAHRZEUGE|KUFU",
    "yearly_premium_jan1_CAMPINGFAHRZEUGE|SBR",
    "yearly_premium_jan1_CAMPINGFAHRZEUGE|TK",
    "yearly_premium_jan1_CAMPINGFAHRZEUGE|VK",
    "yearly_premium_jan1_KRAFTRAEDER|ASSV",
    "yearly_premium_jan1_KRAFTRAEDER|KH",
    "yearly_premium_jan1_KRAFTRAEDER|KU",
    "yearly_premium_jan1_KRAFTRAEDER|SBR",
    "yearly_premium_jan1_KRAFTRAEDER|TK",
    "yearly_premium_jan1_KRAFTRAEDER|VK",
    "yearly_premium_jan1_PH|AMTS_VERMOEGENS_HAFTPFLICHT_UND_ZUSATZ",
    "yearly_premium_jan1_PH|PLUS",
    "yearly_premium_jan1_PH|PRIVATHAFTPFLICHT",
    "yearly_premium_jan1_PKW|ASSV",
    "yearly_premium_jan1_PKW|FAU",
    "yearly_premium_jan1_PKW|KH",
    "yearly_premium_jan1_PKW|KU",
    "yearly_premium_jan1_PKW|KUFU",
    "yearly_premium_jan1_PKW|SBR",
    "yearly_premium_jan1_PKW|TK",
    "yearly_premium_jan1_PKW|UMD",
    "yearly_premium_jan1_PKW|VK",
    "yearly_premium_jan1_WG|FEUER",
    "yearly_premium_jan1_WG|LEITUNGSWASSER",
    "yearly_premium_jan1_WG|NEBENPRODUKTE",
    "yearly_premium_jan1_WG|STURM",
    "yearly_premium_jan1_total"
  ],
  "fixed_parameters": {
    "learning_rate": 0.03,
    "n_estimators": 400,
    "num_leaves": 15,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8
  }
}





"""Kundenwert baseline: da_kundenwert's one-year evaluation of KFZ contracts."""

from pathlib import Path

import mlflow
import polars as pl
import pyarrow.parquet as pq
from da_kundenwert.config import load_hydra_config
from da_kundenwert.evaluation import ContractEvaluator
from da_kundenwert.models import _load_risk_models
from da_kundenwert.prediction import PurePremiumPredictorKFZ
from omegaconf import DictConfig
from tqdm import tqdm

from clv_analysis.config import settings
from clv_analysis.dwh import scan_raw
from clv_analysis.expressions import key, product, valid_on_jan1
from clv_analysis.hf3_etl import dataset
from clv_analysis.products import Product, Segment, Source


def model_features(kfz_types: DictConfig, segment: Segment) -> set[str]:
    """Features of the risk models of ``segment``, over all vehicle types."""
    features: set[str] = set()
    for kfz_type, info in kfz_types.items():
        risk_models = _load_risk_models(kfz_type)
        run_id = risk_models[info.tariff_gen]["run_id"][segment.upper()][
            "best_estimate"
        ]
        mlflow.set_tracking_uri(risk_models["tracking_uri"])
        sheet = mlflow.artifacts.load_dict(f"runs:/{run_id}/model/model_sheet.json")
        features |= set(sheet["features"])
    return features


def read_rows(
    file: Path, year: int, features: set[str], contracts: pl.LazyFrame
) -> pl.DataFrame:
    """Rows of one HF3 file valid on 1 Jan with a risk model, with partner_id."""
    columns = features | {
        "ve_id",
        "ve_sparte",
        "ve_gesellschaft",
        "ve_wagniskennziffer",
        "ve_bestandsjahresnettobeitrag",
        "ve_risiko_beginn",
        "ve_risiko_ablauf",
        # model exposure
        "ve_jahreseinheit_statistikjahr",
    }
    return (
        pl.scan_parquet(file)
        .select(sorted(columns))
        .with_columns(
            year=pl.lit(year, dtype=pl.Int16),
            source=pl.lit(Source.KFZ, dtype=pl.Enum(Source)),
            wkz=pl.col("ve_wagniskennziffer").cast(pl.String),
            vertragsakte_id=pl.col("ve_id").cast(pl.Int64),
        )
        .with_columns(product=product())
        .filter(
            valid_on_jan1("ve_risiko_beginn", "ve_risiko_ablauf"),
            pl.col("product").is_not_null(),
        )
        .join(contracts, on="vertragsakte_id")
        .collect()
    )


def evaluate(
    rows: pl.DataFrame,
    predictor: PurePremiumPredictorKFZ,
    evaluator: ContractEvaluator,
) -> pl.DataFrame:
    """Premium, expected claims, cost and profit per row, from da_kundenwert."""
    df = rows.drop("source").to_pandas()
    df["ve_wkz"] = df["wkz"]
    df["ve_sparte"] = df["ve_sparte"].astype(str)
    df["ve_gesellschaft"] = df["ve_gesellschaft"].astype(str)
    predicted = predictor.predict(df)
    # production scales to the 2026 tariff's claim level; the backtest keeps the
    # model at the row's own year (gs_statistikjahr), so the factor comes out again
    predicted["ve_expected_claim_amount"] /= predicted["claim_inflation_factor"]
    out = evaluator.evaluate_kfz(predicted)
    return pl.from_pandas(out).select(
        pl.col("partner_id").cast(pl.Int64),
        pl.col("year").cast(pl.Int16),
        pl.col("product").cast(pl.String),
        segment=pl.col("ve_sparte").str.to_lowercase(),
        premium="ve_bestandsjahresnettobeitrag",
        expected_claim_amount="ve_expected_claim_amount",
        total_cost="ve_total_cost",
        profit="ve_profit",
    )


def to_wide(
    evaluated: pl.DataFrame, values: list[str], keys: list[str]
) -> pl.DataFrame:
    """Sum per partner x year x product|segment into kw_<value>_<KEY> columns.

    Every value x key gets a column; keys no partner has in this year are null.
    """
    wide = (
        evaluated.group_by("partner_id", "year", key=key())
        .agg(pl.sum(v) for v in values)
        .pivot(on="key", index=["partner_id", "year"], values=values)
    )
    return wide.select(
        "partner_id",
        "year",
        *[
            (
                pl.col(f"{v}_{k}")
                if f"{v}_{k}" in wide.columns
                else pl.lit(None, dtype=pl.Float64)
            ).alias(f"kw_{v}_{k}")
            for v in values
            for k in keys
        ],
    )


def build_baseline(remainders: list[int], years: list[int] | None = None) -> None:
    """Write the baseline per partner x year to ``<baseline_dir>/<modulo>/<year>/<r>.parquet``.

    Parameters
    ----------
    remainders : list[int]
        Write the partners with ``partner_id % modulo`` in this list, one file each.
    years : list[int] | None
        Years to run; None = every year in HF3.
    """
    kfz_types = load_hydra_config().kfz.types
    predictor = PurePremiumPredictorKFZ(kfz_types=kfz_types, model_mapping_files={})
    evaluator = ContractEvaluator(
        beitrag_col="ve_bestandsjahresnettobeitrag",
        pure_premium_col="ve_expected_claim_amount",
        cost_col="ve_total_cost",
        kfz_types=kfz_types,
    )
    segments = [Segment.KH, Segment.VK, Segment.TK]
    products = [Product.PKW, Product.KRAFTRAEDER, Product.CAMPINGFAHRZEUGE]
    values = ["premium", "expected_claim_amount", "total_cost", "profit"]
    keys = [f"{p}|{s}".upper() for p in products for s in segments]
    features = {s: model_features(kfz_types, s) for s in segments}
    contracts = (
        scan_raw("kfz_timeslices", remainders)
        .select("vertragsakte_id", "partner_id")
        .unique()
        .collect()
        .lazy()
    )

    spec = dataset(Source.KFZ)
    schema = (
        pl.DataFrame(
            schema={"partner_id": pl.Int64, "year": pl.Int16}
            | {f"kw_{v}_{k}": pl.Float64 for v in values for k in keys}
        )
        .to_arrow()
        .schema
    )
    for year in years or spec.years(Segment.KH):
        evaluated = []
        for segment in segments:
            files = sorted(spec.partition(segment, year).glob("*.parquet"))
            for file in tqdm(files, desc=f"{segment} {year}"):
                rows = read_rows(file, year, features[segment], contracts)
                if not rows.is_empty():
                    evaluated.append(evaluate(rows, predictor, evaluator))
        wide = to_wide(pl.concat(evaluated), values, keys)
        out_dir = settings.baseline_dir / str(settings.modulo) / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in remainders:
            part = wide.filter(pl.col("partner_id") % settings.modulo == r)
            pq.write_table(part.to_arrow().cast(schema), out_dir / f"{r}.parquet")
