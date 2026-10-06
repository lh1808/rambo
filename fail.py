"""CLI (typer)."""

import typer

app = typer.Typer(help="CLV analysis data pipeline.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """CLV analysis data pipeline."""


def _remainders(value: str | None) -> list[int]:
    """Parse ``--remainders`` "0,10,20"; None = all of MOD(partner_id, modulo)."""
    from clv_analysis.config import settings

    if value is None:
        return list(range(settings.modulo))
    remainders = [int(r) for r in value.split(",")]
    if not all(0 <= r < settings.modulo for r in remainders):
        raise typer.BadParameter(f"remainders must be in 0..{settings.modulo - 1}")
    return remainders


@app.command()
def build(
    dwh: bool = typer.Option(True, help="Pull raw pieces from the DWH into .data/raw."),
    hf3: bool = typer.Option(
        True, help="Aggregate every HF3 partition once into .data/hf3_agg."
    ),
    flatfile: bool = typer.Option(
        True, help="Build one flatfile per remainder into .data/flat."
    ),
    remainders: str = typer.Option(
        None,
        help="Partner pieces MOD(partner_id, 100) = r, e.g. '0,10,20'. Default: all 100.",
    ),
    overwrite: bool = typer.Option(
        False,
        help="Redo DWH pieces and flatfile buckets already on disk (default: skip them).",
    ),
) -> None:
    """Run the pipeline: DWH pull, HF3 aggregate, flatfile.

    All steps work on ``remainders``. The HF3 step is always rebuilt; the DWH
    and flatfile steps skip existing files unless ``--overwrite``.
    """
    if not (dwh or hf3 or flatfile):
        raise typer.BadParameter("--no-dwh, --no-hf3 and --no-flatfile run nothing.")
    pieces = _remainders(remainders)
    if dwh:
        from clv_analysis.dwh.partner import load_partner_features
        from clv_analysis.dwh.timeslices import load_timeslices

        load_timeslices(pieces, overwrite=overwrite)
        load_partner_features(pieces, overwrite=overwrite)
    if hf3:
        from clv_analysis.hf3_etl import build_hf3_agg

        build_hf3_agg(pieces)
    if flatfile:
        from clv_analysis.flatfile import build_flatfile

        build_flatfile(pieces, overwrite=overwrite)


@app.command()
def train(
    years_ahead: int = typer.Option(
        3, help="CLV horizon in years, including the year itself."
    ),
    beta: float = typer.Option(0.95, help="Yearly discount factor of the CLV."),
    remainders: str = typer.Option(
        None, help="Flatfile buckets to train on, e.g. '0,10,20'. Default: all built."
    ),
) -> None:
    """Write the PKW CLV training data and fit the model dict in modelling/clv_pkw.json."""
    import json
    from pathlib import Path

    from quantcore.model.training import fit

    from clv_analysis.modelling._train_data import write_train_data

    package_dir = Path(__file__).parent
    model_dict = json.loads((package_dir / "modelling" / "clv_pkw.json").read_text())
    tracking_uri = json.loads((package_dir / "models" / "config.json").read_text())[
        "tracking_uri"
    ]

    write_train_data(
        Path(model_dict["data"]["path"]),
        features=model_dict["features"],
        remainders=None if remainders is None else _remainders(remainders),
        years_ahead=years_ahead,
        beta=beta,
    )
    fit(model_dict, tracking_uri=tracking_uri)


@app.command("baseline-predict")
def baseline_predict(
    remainders: str = typer.Option(
        None, help="Partner pieces MOD(partner_id, 100) = r, e.g. '0'. Default: all."
    ),
    years: str = typer.Option(None, help="Years, e.g. '2024'. Default: all in HF3."),
) -> None:
    """Kundenwert baseline per partner x year into .data/baseline (da-kundenwert env)."""
    from clv_analysis.baseline_prediction._predict import build_baseline

    build_baseline(
        _remainders(remainders),
        years=None if years is None else [int(y) for y in years.split(",")],
    )


if __name__ == "__main__":
    app()
