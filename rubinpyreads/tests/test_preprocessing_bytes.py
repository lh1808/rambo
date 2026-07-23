"""Regression für den Produktions-Crash "Cannot cast object dtype to <U0" /
UnicodeDecodeError 0xc3 in preprocessing.transform.

Kette: SAS-Import liefert bytes-Textspalten mit UTF-8-Umlauten; sobald eine
solche object-Spalte (via fill_missing_categories) category-Dtype trägt,
kastete transform() das Kategorien-Array über numpy (<U…) und crashte am
ersten Umlaut — fit_preprocessor lief dagegen durch (object-Pfad, pandas
dekodiert bytes elementweise). Fix: transform geht denselben object-Weg
(astype("object") vor astype(str)) → identische Encoding-Keys wie fit.
Zusätzlich: decode_bytes_categories dekodiert bytes-Kategorien in DataPrep
generell (lesbare Labels statt "b'...'" auch für Spalten ohne NaN).
"""
import numpy as np
import pandas as pd

from rubin.preprocessing import fit_preprocessor
from rubin.utils.data_utils import decode_bytes_categories, fill_missing_categories

BYTES_UMLAUT = b"M\xc3\xbcnchener"  # 'Münchener' UTF-8-kodiert (0xc3 an Pos. 5-Kontext)


class TestTransformBytesSymmetry:
    def test_category_with_bytes_categories_roundtrip(self):
        """Exakt die Produktions-Kette: object+bytes+NaN → fill (→ category)
        → fit → transform. Crashte vor dem Fix in transform."""
        X = pd.DataFrame({"G": pd.Series([BYTES_UMLAUT, None, b"X", BYTES_UMLAUT], dtype="object")})
        fill_missing_categories(X, columns=["G"])
        assert isinstance(X["G"].dtype, pd.CategoricalDtype)
        pre = fit_preprocessor(X, categorical_columns=["G"], fill_na_method=None)
        Xp = pre.transform(X)
        codes = Xp["G"].astype(int)
        assert (codes >= 0).all()          # fit- und transform-Keys identisch → kein -1
        pd.testing.assert_frame_equal(Xp, pre.transform(X.copy()))  # deterministisch

    def test_plain_string_categories_unchanged(self):
        """Regressionskontrolle: str-Kategorien kodieren wie zuvor."""
        X = pd.DataFrame({"T": pd.Categorical(["A", "B", "A"])})
        pre = fit_preprocessor(X, categorical_columns=["T"], fill_na_method=None)
        codes = pre.transform(X)["T"].astype(int).tolist()
        assert codes == [0, 1, 0]


class TestDecodeBytesCategories:
    def test_object_and_category_decoded_readable(self):
        X = pd.DataFrame({
            "OBJ": pd.Series([BYTES_UMLAUT, "schon_str", None], dtype="object"),
            "CAT": pd.Categorical([BYTES_UMLAUT, b"X", BYTES_UMLAUT]),
            "NUM": [1.0, 2.0, 3.0],
        })
        converted = decode_bytes_categories(X)
        assert set(converted) == {"OBJ", "CAT"}
        assert X["OBJ"].iloc[0] == "Münchener" and X["OBJ"].iloc[1] == "schon_str"
        assert "Münchener" in X["CAT"].cat.categories
        assert decode_bytes_categories(X) == []   # idempotent

    def test_invalid_utf8_replaced_not_crashing(self):
        X = pd.DataFrame({"B": pd.Series([b"\xff\xfe kaputt"], dtype="object")})
        assert decode_bytes_categories(X) == ["B"]
        assert "kaputt" in X["B"].iloc[0]          # errors="replace" statt Crash


class TestProductionCrossRepresentation:
    def test_fit_on_decoded_transform_on_raw_bytes_identical_codes(self):
        """Produktions-Kritisch: DataPrep dekodiert künftig vor dem Fit
        (Encoding-Maps mit 'Münchener'-Keys) — Scoring-Daten können aber
        weiterhin ROHE bytes liefern. Der score()-Pfad dekodiert jetzt
        ebenfalls; zusätzlich dekodiert pandas' object-astype(str) bytes
        selbst via UTF-8. Beide Repräsentationen müssen identische Codes
        ergeben — sonst würden Scoring-Zeilen still als 'unbekannt' (-1)
        kodiert."""
        from rubin.utils.data_utils import decode_bytes_categories
        X_train = pd.DataFrame({"G": pd.Series([BYTES_UMLAUT, b"X", BYTES_UMLAUT], dtype="object")})
        decode_bytes_categories(X_train)
        pre = fit_preprocessor(X_train, categorical_columns=["G"], fill_na_method=None)
        codes_train = pre.transform(X_train)["G"].astype(int).tolist()

        X_score_raw = pd.DataFrame({"G": pd.Series([b"X", BYTES_UMLAUT], dtype="object")})
        decode_bytes_categories(X_score_raw)          # wie im score()-Pfad
        codes_score = pre.transform(X_score_raw)["G"].astype(int).tolist()
        assert codes_score == [codes_train[1], codes_train[0]]
        assert -1 not in codes_score                   # keine Schein-Unbekannten

    def test_mixed_bytes_and_str_in_one_column(self):
        """Inkrementelle SAS-Exporte: dieselbe Spalte kann bytes UND str
        mischen — nach Dekodierung muss beides auf dieselbe Kategorie fallen."""
        from rubin.utils.data_utils import decode_bytes_categories, fill_missing_categories
        X = pd.DataFrame({"G": pd.Series([BYTES_UMLAUT, "Münchener", None], dtype="object")})
        decode_bytes_categories(X)
        fill_missing_categories(X, columns=["G"])
        assert list(X["G"].cat.categories) == ["Münchener", "fehlend"]
        assert (X["G"] == "Münchener").sum() == 2      # bytes und str vereinigt

    def test_category_with_colliding_bytes_and_str_categories(self):
        """Härtung: Kategorie-Spalte trägt dieselbe Ausprägung als bytes UND
        str (getrennte Kategorien) — Dekodierung muss vereinigen statt an
        rename_categories-Duplikaten zu scheitern."""
        from rubin.utils.data_utils import decode_bytes_categories
        X = pd.DataFrame({"G": pd.Categorical([BYTES_UMLAUT, "Münchener", b"X"])})
        assert decode_bytes_categories(X) == ["G"]
        assert sorted(X["G"].cat.categories) == ["Münchener", "X"]
        assert (X["G"] == "Münchener").sum() == 2
