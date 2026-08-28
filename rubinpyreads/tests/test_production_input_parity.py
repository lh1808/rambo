"""Eingangs-Parität Produktion ↔ Analyse: ProductionPipeline.prepare_input
(bundle_dtypes-Angleichung, bytes→str-Decode, NaN→"fehlend") ist der geteilte
Kern von score() UND dem Runner (score_dataframe). Ohne ihn liefen
sas7bdat-bytes-Kategorien (b"M" statt "M") und float-gelesene int-Spalten
("1.0" statt "1") massenhaft auf die -1-Kategorie — sichtbar als extreme
minus1-Raten im Produktions-Monitoring."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rubin.pipelines.production_pipeline import ProductionPipeline

_PROD = Path(__file__).resolve().parents[1] / "production"
if str(_PROD) not in sys.path:
    sys.path.insert(0, str(_PROD))


class TestPrepareInputParity:
    def test_bytes_and_nan_categories_are_normalized(self):
        pipe = ProductionPipeline("runs/bundles/smoke_bundle_nancat")
        df = pd.DataFrame({
            "BEITRAG": [100.0, 200.0, 300.0],
            "GESCHLECHT": pd.Series([b"M", None, b"W"], dtype=object),
            "TARIF": pd.Series(["A", "B", None], dtype=object),
            "F3": [1.0, 2.0, 3.0], "F4": [0.1, 0.2, 0.3],
        })
        out = pipe.prepare_input(df)
        g = out["GESCHLECHT"].astype(str).tolist()
        assert g[0] == "M" and g[2] == "W"          # bytes → str dekodiert
        assert "fehlend" in (g[1], out["TARIF"].astype(str).tolist()[2])  # NaN → "fehlend"
        assert not out["GESCHLECHT"].isna().any() and not out["TARIF"].isna().any()

    def test_score_uses_prepare_input(self):
        src = Path("rubin/pipelines/production_pipeline.py").read_text(encoding="utf-8")
        i = src.index("def score(self, X_raw")
        end = src.find("\n    def ", i)
        body = src[i:end if end != -1 else len(src)]
        assert "self.prepare_input(" in body

    def test_runner_calls_prepare_input_before_transform(self):
        src = (_PROD / "run_scoring.py").read_text(encoding="utf-8")
        i_prep = src.index("pipe.prepare_input(df)")
        i_transform = src.index("pipe.preprocessor.transform(df)")
        i_coerce = src.index('cfg["preprocessing"].get("coerce_numeric_strings"')
        assert i_coerce < i_prep < i_transform  # Coerce → prepare_input → transform
