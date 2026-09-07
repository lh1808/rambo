"""Eingangs-Parität Produktion ↔ Analyse: ProductionPipeline.prepare_input
(bundle_dtypes-Angleichung, bytes→str-Decode, NaN→"fehlend"). Die
Verdrahtung in score() und im Runner sichert der End-to-End-Paritätstest
(test_production_score_parity) verhaltensbasiert ab. Ohne den Kern liefen
sas7bdat-bytes-Kategorien (b"M" statt "M") und float-gelesene int-Spalten
("1.0" statt "1") massenhaft auf die -1-Kategorie — sichtbar als extreme
minus1-Raten im Produktions-Monitoring."""
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
