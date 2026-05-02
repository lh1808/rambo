"""Kategorische Feature-Unterstützung für EconML-Modelle.

Problem
-------
EconML konvertiert X intern zu numpy-Arrays (via sklearn check_array), wodurch
pandas category-Dtypes verloren gehen. LightGBM und CatBoost erhalten dann nur
float64-Werte und können keine kategorialen Splits mehr nutzen. Stattdessen
werden ordinale Splits auf nominalen Features angewendet — deutlich schwächere
Modellierung.

Lösung
------
Die ``partialmethod``-Technik patcht die ``.fit()``-Methoden von LGBMClassifier,
LGBMRegressor, CatBoostClassifier und CatBoostRegressor so, dass
``categorical_feature`` (LightGBM) bzw. ``cat_features`` (CatBoost) bei jedem
``.fit()``-Aufruf automatisch übergeben wird — auch wenn EconML intern nur
``model.fit(X_numpy, y)`` aufruft.

Nutzung
-------
>>> with patch_categorical_features(X):
...     model = NonParamDML(model_y=LGBMClassifier(), ...)
...     model.fit(Y, T, X=X)
...     cate = model.const_marginal_effect(X)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import partialmethod
from typing import List, Optional

import pandas as pd

_logger = logging.getLogger("rubin.categorical")


def _detect_cat_indices(X: pd.DataFrame) -> List[int]:
    """Ermittelt die Spaltenindizes aller kategorialen Features."""
    cat_cols = X.select_dtypes(include=["category", "object"]).columns
    return [i for i, col in enumerate(X.columns) if col in cat_cols]


@contextmanager
def patch_categorical_features(X: pd.DataFrame, base_learner_type: str = "lgbm"):
    """Context-Manager: Patcht LightGBM/CatBoost für kategorische Features.

    Erkennt automatisch category-/object-Spalten in X, berechnet deren
    Spaltenindizes und patcht die .fit()-Methoden der Base Learner so, dass
    ``categorical_feature`` (LightGBM) bzw. ``cat_features`` (CatBoost)
    bei jedem Aufruf automatisch übergeben wird.

    Nach Verlassen des Kontexts werden die originalen .fit()-Methoden
    wiederhergestellt (kein globaler State-Leak).

    Parameters
    ----------
    X : pd.DataFrame
        Feature-Matrix mit den finalen Spalten (nach Feature-Selektion).
    base_learner_type : str
        "lgbm" oder "catboost". Bestimmt, welche Klassen gepatcht werden.
    """
    cat_indices = _detect_cat_indices(X)

    if not cat_indices:
        _logger.debug("Keine kategorialen Spalten erkannt — kein Patching nötig.")
        yield cat_indices
        return

    cat_col_names = [X.columns[i] for i in cat_indices]
    _logger.info(
        "Kategorische Spalten erkannt: %d von %d Features (%s). "
        "Patche %s .fit()-Methoden für EconML-Kompatibilität.",
        len(cat_indices), len(X.columns),
        cat_col_names[:5] if len(cat_col_names) <= 5 else f"{cat_col_names[:3]}... (+{len(cat_col_names)-3})",
        base_learner_type.upper(),
    )

    originals = {}
    base_type = (base_learner_type or "lgbm").lower()
    # "Both"-Modus: Beide Libraries werden gleichzeitig gepatcht, da Optuna pro
    # Trial zwischen LGBM und CatBoost wählen kann. Für LGBM-Trials wirkt der
    # LGBM-Patch, für CatBoost-Trials der CatBoost-Patch — keine Interferenz.
    patch_lgbm = base_type in ("lgbm", "both")
    patch_catboost = base_type in ("catboost", "both")

    try:
        if patch_lgbm:
            import lightgbm as lgbm

            # Originale sichern
            originals["LGBMClassifier.fit"] = lgbm.LGBMClassifier.fit
            originals["LGBMRegressor.fit"] = lgbm.LGBMRegressor.fit

            # Patchen: categorical_feature wird bei JEDEM .fit()-Aufruf übergeben,
            # auch wenn EconML intern nur model.fit(X_numpy, y) aufruft.
            lgbm.LGBMClassifier.fit = partialmethod(
                lgbm.LGBMClassifier.fit, categorical_feature=cat_indices
            )
            lgbm.LGBMRegressor.fit = partialmethod(
                lgbm.LGBMRegressor.fit, categorical_feature=cat_indices
            )

        if patch_catboost:
            from catboost import CatBoostClassifier, CatBoostRegressor
            import numpy as _np

            # Originale sichern
            originals["CatBoostClassifier.fit"] = CatBoostClassifier.fit
            originals["CatBoostRegressor.fit"] = CatBoostRegressor.fit
            originals["CatBoostClassifier.predict"] = CatBoostClassifier.predict
            originals["CatBoostRegressor.predict"] = CatBoostRegressor.predict
            originals["CatBoostClassifier.predict_proba"] = CatBoostClassifier.predict_proba

            # CatBoost-Problem: EconML konvertiert X zu numpy float64 via
            # sklearn check_array. CatBoost verweigert cat_features auf
            # float-Spalten — sowohl bei fit() als auch bei predict/predict_proba,
            # weil CatBoost die cat_features vom Training intern speichert und
            # bei Prediction erneut prüft.
            # Lösung: Alle relevanten Methoden wrappen.

            def _convert_float_to_df(X, ci, np_mod):
                """Konvertiert numpy float-Array → pandas DataFrame mit korrekten Dtypes.

                CatBoost akzeptiert DataFrames mit gemischten Typen und cat_features-Indices.
                DEUTLICH schneller als numpy object-Array (C++ statt Python-Object-Layer).
                Float64 → float32: CatBoost-Doku empfiehlt float32 als optimales Format.

                WICHTIG: pandas iloc-Assignment ändert NICHT den Spalten-Dtype!
                df.iloc[:, idx] = int_values → castet still zu float zurück.
                Lösung: Spalte komplett ersetzen via df[col] = Series(dtype=int).
                """
                if not ci or not isinstance(X, np_mod.ndarray) or X.dtype.kind != "f":
                    return X
                import pandas as _pd
                # float32 statt float64: halber Speicher, CatBoost-optimal
                df = _pd.DataFrame(X.astype(np_mod.float32))
                for idx in ci:
                    if idx < df.shape[1]:
                        col = df.iloc[:, idx].values
                        col_int = np_mod.nan_to_num(col, nan=-1).astype(np_mod.int32)
                        # Spalte komplett ersetzen — iloc würde zu float zurückcasten!
                        df[df.columns[idx]] = col_int
                return df

            _fit_logged = [False]
            _pred_logged = [False]

            def _make_fit_wrapper(original_fit):
                def _wrapped_fit(self, X, y=None, **kwargs):
                    if "cat_features" not in kwargs:
                        kwargs["cat_features"] = cat_indices
                    ci = kwargs.get("cat_features", [])
                    X_conv = _convert_float_to_df(X, ci, _np)
                    if X_conv is not X and not _fit_logged[0]:
                        _logger.info(
                            "CatBoost categorical patch (fit): %d/%d Spalten float→int konvertiert.",
                            len(ci), X.shape[1],
                        )
                        _fit_logged[0] = True
                    elif ci and not _fit_logged[0]:
                        _logger.info(
                            "CatBoost categorical patch (fit): %d cat_features injiziert (DataFrame).",
                            len(ci),
                        )
                        _fit_logged[0] = True
                    return original_fit(self, X_conv, y, **kwargs)
                return _wrapped_fit

            def _make_predict_wrapper(original_predict):
                def _wrapped_predict(self, X, *args, **kwargs):
                    # Immer cat_indices aus Patch-Closure verwenden (korrekt nach FS)
                    X_conv = _convert_float_to_df(X, cat_indices, _np)
                    if X_conv is not X and not _pred_logged[0]:
                        _logger.info(
                            "CatBoost categorical patch (predict): %d Spalten float→int konvertiert.",
                            len(cat_indices),
                        )
                        _pred_logged[0] = True
                    return original_predict(self, X_conv, *args, **kwargs)
                return _wrapped_predict

            CatBoostClassifier.fit = _make_fit_wrapper(originals["CatBoostClassifier.fit"])
            CatBoostRegressor.fit = _make_fit_wrapper(originals["CatBoostRegressor.fit"])
            CatBoostClassifier.predict = _make_predict_wrapper(originals["CatBoostClassifier.predict"])
            CatBoostRegressor.predict = _make_predict_wrapper(originals["CatBoostRegressor.predict"])
            CatBoostClassifier.predict_proba = _make_predict_wrapper(originals["CatBoostClassifier.predict_proba"])

        yield cat_indices

    finally:
        # Originale wiederherstellen — IMMER, auch bei Exceptions.
        # Im "both"-Modus werden beide Libraries restauriert.
        if "LGBMClassifier.fit" in originals:
            import lightgbm as lgbm
            lgbm.LGBMClassifier.fit = originals["LGBMClassifier.fit"]
            lgbm.LGBMRegressor.fit = originals["LGBMRegressor.fit"]
            _logger.debug("LightGBM .fit()-Methoden wiederhergestellt.")

        if "CatBoostClassifier.fit" in originals:
            from catboost import CatBoostClassifier, CatBoostRegressor
            CatBoostClassifier.fit = originals["CatBoostClassifier.fit"]
            CatBoostRegressor.fit = originals["CatBoostRegressor.fit"]
            CatBoostClassifier.predict = originals["CatBoostClassifier.predict"]
            CatBoostRegressor.predict = originals["CatBoostRegressor.predict"]
            CatBoostClassifier.predict_proba = originals["CatBoostClassifier.predict_proba"]
            _logger.debug("CatBoost .fit()/.predict()/.predict_proba() wiederhergestellt.")
