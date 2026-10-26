"""Gemeinsame Test-Fixtures.

Baut die beiden Bundle-Fixtures unter runs/bundles/ bei Bedarf selbst auf
(self-contained Suite): Vorher waren das eingecheckte, GENERIERTE Artefakte
der Arbeitskopie — ein gelöschtes runs/ ließ test_numeric_coercion und
test_production_input_parity still scheitern. Jetzt erzeugt pytest sie
deterministisch beim Start, wenn sie fehlen (einmalig, wenige Sekunden).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from unittest.mock import MagicMock

_REPO = Path(__file__).resolve().parents[1]
_BUNDLES = _REPO / "runs" / "bundles"


def _export(bundle_id: str, X, T, Y, models, eval_summary, x_dir: Path):
    import numpy as np  # noqa: F401
    from types import SimpleNamespace
    from rubin.pipelines.analysis_pipeline import AnalysisPipeline

    cfg_file = x_dir / "config_snapshot.yml"
    cfg_file.write_text("# fixture config snapshot\n", encoding="utf-8")
    cfg = SimpleNamespace(
        bundle=SimpleNamespace(enabled=True, base_dir=str(_BUNDLES), bundle_id=None, log_to_mlflow=False),
        source_config_path=str(cfg_file),
        data_files=SimpleNamespace(x_file=str(x_dir / "X.parquet")),
        selection=SimpleNamespace(metric="qini", higher_is_better=True, manual_champion=None),
        surrogate_tree=SimpleNamespace(enabled=False),
        treatment=SimpleNamespace(type="binary", reference_group=0),
        base_learner=SimpleNamespace(type="catboost"),
    )
    pipe = AnalysisPipeline.__new__(AnalysisPipeline)
    pipe._logger = logging.getLogger("conftest_fixture_bundles")
    pipe._run_bundle_export(
        cfg, models, eval_summary,
        X, T, Y, X, T, Y,
        list(X.columns), None, True, str(_BUNDLES), bundle_id, MagicMock(),
    )


def _lineardml(seed: int):
    from econml.dml import LinearDML
    from sklearn.linear_model import LinearRegression, LogisticRegression
    return LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(max_iter=200),
                     discrete_treatment=True, cv=2, random_state=seed)


def _build_surr_bt():
    """Numerisches Bundle F0..F4 mit gelernter NA-Imputation (F1) —
    für die Coercion-Tests ('V' → NaN → Impute; Crash-Klasse ohne Coercion)."""
    import numpy as np
    import pandas as pd
    from rubin.preprocessing import fit_preprocessor

    rng = np.random.default_rng(11)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"F{i}" for i in range(5)])
    X.loc[rng.choice(n, 30, replace=False), "F1"] = np.nan  # NA-Fill für F1 lernen
    T = rng.integers(0, 2, n)
    Y = 0.4 * X["F0"].to_numpy() + (0.3 + 0.3 * X["F0"].to_numpy()) * T + rng.normal(scale=0.1, size=n)

    x_dir = _BUNDLES.parent / "_fixture_build" / "surr_bt"
    x_dir.mkdir(parents=True, exist_ok=True)
    preproc = fit_preprocessor(X, categorical_columns=[], fill_na_method="mean")
    with open(x_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preproc, f)
    X_fit = preproc.transform(X.copy())
    X_fit.to_parquet(x_dir / "X.parquet")
    _export("surr_bt", X_fit, T, Y,
            {"ModelA": _lineardml(1)}, {"ModelA": {"qini": 0.05}}, x_dir)


def _build_smoke_bundle_nancat():
    """Bundle mit kategorialen Features (GESCHLECHT/TARIF inkl. 'fehlend') —
    für den prepare_input-Paritätstest (bytes→str, NaN→'fehlend', dtypes)."""
    import json
    import numpy as np
    import pandas as pd
    from rubin.preprocessing import fit_preprocessor

    rng = np.random.default_rng(12)
    n = 300
    X = pd.DataFrame({
        "BEITRAG": rng.normal(100.0, 20.0, n),
        "GESCHLECHT": rng.choice(["M", "W"], n).astype(object),
        "TARIF": rng.choice(["A", "B"], n).astype(object),
        "F3": rng.normal(size=n), "F4": rng.normal(size=n),
    })
    X.loc[rng.choice(n, 20, replace=False), "GESCHLECHT"] = np.nan
    X.loc[rng.choice(n, 20, replace=False), "TARIF"] = np.nan
    from rubin.utils.data_utils import fill_missing_categories
    fill_missing_categories(X)  # NaN → "fehlend" wie im DataPrep-Training
    T = rng.integers(0, 2, n)
    Y = 0.3 * X["F3"].to_numpy() + 0.2 * T + rng.normal(scale=0.1, size=n)

    x_dir = _BUNDLES.parent / "_fixture_build" / "smoke_bundle_nancat"
    x_dir.mkdir(parents=True, exist_ok=True)
    preproc = fit_preprocessor(X, categorical_columns=["GESCHLECHT", "TARIF"], fill_na_method="mean")
    with open(x_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preproc, f)
    X_fit = preproc.transform(X.copy())
    # category-codes → numerisch für das Modell-Fit
    for c in ("GESCHLECHT", "TARIF"):
        X_fit[c] = X_fit[c].astype("int32") if str(X_fit[c].dtype) != "category" else X_fit[c].cat.codes.astype("int32")
    X_fit.to_parquet(x_dir / "X.parquet")
    _export("smoke_bundle_nancat", X_fit, T, Y,
            {"ModelA": _lineardml(2)}, {"ModelA": {"qini": 0.04}}, x_dir)
    # dtypes.json ins Bundle: kategoriale Spalten für prepare_input markieren
    dtypes = {"BEITRAG": "float64", "GESCHLECHT": "category", "TARIF": "category",
              "F3": "float64", "F4": "float64"}
    (_BUNDLES / "smoke_bundle_nancat" / "dtypes.json").write_text(json.dumps(dtypes, indent=2), encoding="utf-8")


def pytest_configure(config):
    try:
        if not (_BUNDLES / "surr_bt" / "metadata.json").exists():
            _build_surr_bt()
        if not (_BUNDLES / "smoke_bundle_nancat" / "metadata.json").exists():
            _build_smoke_bundle_nancat()
    except Exception:  # Suite nicht am Fixture-Bau scheitern lassen — Tests melden präziser
        logging.getLogger("conftest_fixture_bundles").warning("Bundle-Fixture-Aufbau fehlgeschlagen.", exc_info=True)
