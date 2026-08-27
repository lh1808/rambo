"""Produktions-Robustheit: Sonderwerte (z. B. SAS-'V') in numerisch
trainierten Features des Gesamtbestands → to_numeric-Coercion → NaN →
gelernte Imputation. Reproduziert den realen Prod-Crash
("could not convert string to float: 'V'") und beweist die Behebung."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROD = Path(__file__).resolve().parents[1] / "production"
if str(_PROD) not in sys.path:
    sys.path.insert(0, str(_PROD))

from run_scoring import load_scoring_config, run_scoring  # noqa: E402


def _cfg(tmp_path, n=400, coerce=True):
    rng = np.random.RandomState(7)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"F{i}" for i in range(5)])
    X.insert(0, "ID", np.arange(n))
    # F1 als object mit Sonderwerten: Zeile 0 'V' (Prod-Fall), Zeile 1 echtes NaN,
    # Rest Zahlen-Strings (object-dtype wie beim sas7bdat-Load des Gesamtbestands)
    X.iloc[1, 1:] = X.iloc[0, 1:]  # Zeile 1 = Feature-Kopie von Zeile 0 …
    f1 = X["F1"].astype(str)
    f1.iloc[0] = "V"               # … nur F1 unterscheidet sich: 'V' vs. NaN.
    f1.iloc[1] = np.nan
    X["F1"] = f1.astype(object)
    inp = tmp_path / "X.parquet"
    X.to_parquet(inp)
    y = {
        "name": "coerce_test",
        "bundle": "runs/bundles/surr_bt",
        "input": {"path": str(inp)},
        "id_columns": ["ID"],
        "output": {"xpt_path": str(tmp_path / "out.xpt")},
        "preprocessing": {"coerce_numeric_strings": coerce},
        "monitoring": {"dir": str(tmp_path / "mon")},
    }
    p = tmp_path / "cfg.yml"
    import yaml
    p.write_text(yaml.safe_dump(y), encoding="utf-8")
    return p, tmp_path


class TestNumericCoercion:
    def test_v_value_scores_like_nan_and_is_monitored(self, tmp_path):
        p, root = _cfg(tmp_path, coerce=True)
        run_scoring(load_scoring_config(str(p)))
        import pyreadstat
        df, _ = pyreadstat.read_xport(str(root / "out.xpt"))
        assert len(df) == 400 and df["SCORE_P"].notna().all()
        # 'V' (Zeile 0) semantisch == fehlend (Zeile 1): identischer Median-Impute
        assert df["SCORE_P"].iloc[0] == pytest.approx(df["SCORE_P"].iloc[1], rel=1e-9)
        mon = sorted((root / "mon").glob("*.json"))
        core = json.load(open(mon[-1]))
        rates = core["preprocessing"]["numeric_coerce_rate_per_feature"]
        assert rates.get("F1", 0) > 0  # nur 'V' zählt als coerced (NaN war schon NaN)

    def test_disabled_reproduces_prod_crash(self, tmp_path):
        p, _ = _cfg(tmp_path, coerce=False)
        # Prod-Log: ValueError (CausalForest); Testbundle: CatBoostError
        # (SurrogateTree scort zuerst) — gleiche Fehlerklasse, daher breit:
        with pytest.raises(Exception, match=r"[Cc]onvert.*float|float.*'V'|'V'"):
            run_scoring(load_scoring_config(str(p)))
