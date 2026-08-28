"""End-to-End-Score-Parität Produktion ↔ Analyse: Ein im Analyse-Stil
gefitteter T-Learner (CatBoost, cat_features via Patch) wird als echtes
Bundle exportiert und über den KOMPLETTEN Produktionsweg (run_scoring:
read_input → Coerce → prepare_input → transform → Patch-Kontext → predict →
XPT) gescort — mit realen Produktions-Störungen (bytes-Kategorien,
'V'-Sonderwert, float-gelesene int-Spalte). Erwartung: Für ungestörte Zeilen
sind die Scores EXAKT die der Analyse-Referenz; Störungen werden auf die
Trainings-Repräsentation normalisiert."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from catboost import CatBoostRegressor
from econml.metalearners import TLearner

from rubin.preprocessing import fit_preprocessor
from rubin.training import _predict_effect
from rubin.utils.categorical_patch import patch_categorical_features

_PROD = Path(__file__).resolve().parents[1] / "production"
if str(_PROD) not in sys.path:
    sys.path.insert(0, str(_PROD))

from run_scoring import load_scoring_config, run_scoring  # noqa: E402


def _versions():
    import importlib.metadata as md
    out = {}
    for p in ("econml", "lightgbm", "catboost", "scikit-learn", "pandas", "numpy"):
        try:
            out[p] = md.version(p)
        except Exception:
            pass
    return out


@pytest.fixture(scope="module")
def parity_setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("parity")
    rng = np.random.RandomState(7)
    n = 800
    X = pd.DataFrame({
        "NUM1": rng.normal(size=n),
        "NUM_INT": rng.randint(0, 9, size=n).astype("int32"),
        "KAT": pd.Series(rng.choice(["A", "B", "C"], size=n), dtype=object),
    })
    T = rng.randint(0, 2, size=n)
    y = X["NUM1"].to_numpy() + 0.4 * X["NUM_INT"].to_numpy() + 0.6 * T + rng.normal(scale=0.1, size=n)

    pre = fit_preprocessor(X, categorical_columns=["KAT"])
    Xp = pre.transform(X)
    tl = TLearner(models=CatBoostRegressor(iterations=15, depth=3, verbose=0, allow_writing_files=False))
    with patch_categorical_features(Xp, base_learner_type="catboost"):
        tl.fit(y, T, X=Xp)
        ref = np.asarray(_predict_effect(tl, Xp), dtype=float)  # Analyse-Referenz

    bundle = root / "bundle"
    (bundle / "models").mkdir(parents=True)
    import pickle
    pickle.dump(pre, open(bundle / "preprocessor.pkl", "wb"))
    pickle.dump(tl, open(bundle / "models" / "TL_CB.pkl", "wb"))
    json.dump({"NUM1": "float64", "NUM_INT": "int32", "KAT": "object"},
              open(bundle / "dtypes.json", "w"))
    json.dump({"models": ["TL_CB"], "champion": "TL_CB", "treatment_type": "binary",
               "n_treatment_arms": 2, "reference_group": 0,
               "selected_feature_columns": list(X.columns),
               "base_learner": {"type": "catboost"},
               "ml_package_versions": _versions()},
              open(bundle / "metadata.json", "w"))

    # Produktions-Input: RAW mit Störungen. Zeilen 0-2 gestört, Rest sauber.
    Xprod = X.copy()
    Xprod["KAT"] = [v.encode() for v in X["KAT"]]        # bytes wie sas7bdat
    Xprod["NUM_INT"] = X["NUM_INT"].astype(float).astype(str)  # "3.0" statt 3
    # Wie sas7bdat: die ganze Spalte kommt als Text (str-Roundtrip ist exakt)
    num1 = X["NUM1"].map(repr).astype(object)
    num1.iloc[0] = "V"                                    # Sonderwert
    num1.iloc[1] = None
    Xprod["NUM1"] = num1
    Xprod.iloc[1, Xprod.columns.get_loc("NUM_INT")] = Xprod.iloc[0]["NUM_INT"]
    Xprod.iloc[1, Xprod.columns.get_loc("KAT")] = Xprod.iloc[0]["KAT"]  # Zeile1 = Zwilling von 0 (NUM1: NaN statt 'V')
    Xprod.insert(0, "ID", np.arange(n))
    Xprod.to_parquet(root / "X_prod.parquet")

    cfg = {"name": "parity", "bundle": str(bundle),
           "input": {"path": str(root / "X_prod.parquet")}, "id_columns": ["ID"],
           "output": {"xpt_path": str(root / "out.xpt")},
           "scoring": {"round_decimals": None, "score_p_model": "TL_CB"},
           "monitoring": {"dir": str(root / "mon")}}
    (root / "cfg.yml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run_scoring(load_scoring_config(str(root / "cfg.yml")))
    import pyreadstat
    out, _ = pyreadstat.read_xport(str(root / "out.xpt"))
    out = out.sort_values("ID").reset_index(drop=True)
    mon = json.load(open(sorted((root / "mon").glob("*_latest.json"))[-1]))
    return ref, out, mon


class TestEndToEndParity:
    def test_clean_rows_match_analysis_exactly(self, parity_setup):
        ref, out, _ = parity_setup
        np.testing.assert_allclose(out["SCORE_P"].to_numpy()[3:], ref[3:], atol=1e-8)

    def test_bytes_categories_are_lossless(self, parity_setup):
        ref, out, mon = parity_setup
        # Zeile 2 ist nur durch bytes/"x.0" gestört → nach Normalisierung == Referenz
        assert out["SCORE_P"].iloc[2] == pytest.approx(ref[2], abs=1e-8)
        assert mon["preprocessing"]["minus1_rate_per_categorical"] in ({}, {"KAT": 0.0})

    def test_special_value_equals_missing_twin(self, parity_setup):
        _, out, mon = parity_setup
        assert out["SCORE_P"].iloc[0] == pytest.approx(out["SCORE_P"].iloc[1], abs=1e-9)
        assert mon["preprocessing"]["numeric_coerce_rate_per_feature"].get("NUM1", 0) > 0
