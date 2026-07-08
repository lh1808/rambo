"""Regressionstest: Eval-Mask × Train-CATE-Semantik in _evaluate_bt.

Bei "Train Many, Evaluate Some" (eval_mask aktiv) müssen:
- die Val-Seite (cate_preds_val, X/T/Y_val) auf die Eval-Zeilen gefiltert sein,
- die Train-CATEs (cate_preds_train) aber die VOLLE Spalte bleiben —
  evaluate_cal nutzt sie ausschließlich für Quantil-Cuts (np.quantile);
  eine auf Eval-Zeilen gefilterte Teilmenge verzerrt die Cuts zur
  Eval-Subpopulation und verkleinert die Stichprobe grundlos.

Der Test patcht evaluate_cate_with_plots und prüft die tatsächlich
übergebenen Argumente — ohne teure Nuisance-Fits.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from rubin.pipelines.analysis_pipeline import AnalysisPipeline


def _make_inputs(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    T = rng.integers(0, 2, size=n)
    Y = rng.integers(0, 2, size=n)
    dfp = pd.DataFrame({
        "Y": Y, "T": T,
        "Predictions_ModelA": rng.normal(size=n),
        "Train_ModelA": rng.normal(loc=5.0, size=n),  # klar unterscheidbare Verteilung
    })
    eval_mask = np.zeros(n, dtype=bool)
    eval_mask[: n // 4] = True  # 25% Eval-Zeilen
    return X, T, Y, dfp, eval_mask


def _run_evaluate_bt(X, T, Y, dfp, eval_mask):
    pipe = AnalysisPipeline.__new__(AnalysisPipeline)
    pipe._logger = logging.getLogger("test_eval_mask")
    cfg = SimpleNamespace(constants=SimpleNamespace(random_seed=42))

    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        bundle = MagicMock()
        bundle.policy_values = {"policy_value": 0.0}
        return bundle

    with patch("rubin.pipelines.analysis_pipeline.evaluate_cate_with_plots", side_effect=_spy):
        pipe._evaluate_bt(
            cfg, X, T, Y, None, "ModelA", dfp,
            dfp["Y"].to_numpy(), dfp["T"].to_numpy(),
            {}, {}, {}, MagicMock(),
            fitted_tester=None, eval_mask=eval_mask,
        )
    assert captured, "evaluate_cate_with_plots wurde nicht aufgerufen"
    return captured


def test_mask_filters_val_but_keeps_full_train_distribution():
    X, T, Y, dfp, eval_mask = _make_inputs()
    n, n_eval = len(X), int(eval_mask.sum())

    captured = _run_evaluate_bt(X, T, Y, dfp, eval_mask)

    # Val-Seite: exakt die Eval-Zeilen
    assert len(captured["cate_preds_val"]) == n_eval
    assert len(captured["X_val"]) == n_eval
    np.testing.assert_array_equal(
        captured["cate_preds_val"],
        dfp.loc[eval_mask, "Predictions_ModelA"].to_numpy(dtype=float),
    )

    # Train-Seite: VOLLE Spalte (Cuts aus der echten Train-Verteilung),
    # konsistent zur Länge von X_train/T_train/Y_train
    assert len(captured["cate_preds_train"]) == n
    np.testing.assert_array_equal(
        captured["cate_preds_train"], dfp["Train_ModelA"].to_numpy(dtype=float),
    )
    assert len(captured["X_train"]) == n
    assert len(captured["T_train"]) == n
    assert len(captured["Y_train"]) == n


def test_no_mask_behaviour_unchanged():
    X, T, Y, dfp, _ = _make_inputs()
    n = len(X)

    captured = _run_evaluate_bt(X, T, Y, dfp, eval_mask=None)

    assert len(captured["cate_preds_val"]) == n
    assert len(captured["cate_preds_train"]) == n
    np.testing.assert_array_equal(
        captured["cate_preds_train"], dfp["Train_ModelA"].to_numpy(dtype=float),
    )


def test_all_nan_train_column_yields_none():
    X, T, Y, dfp, eval_mask = _make_inputs()
    dfp["Train_ModelA"] = np.nan  # Holdout-Semantik: keine Train-CATEs

    captured = _run_evaluate_bt(X, T, Y, dfp, eval_mask)

    assert captured["cate_preds_train"] is None
    assert captured["X_train"] is None
