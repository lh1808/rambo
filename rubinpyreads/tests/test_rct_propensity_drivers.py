"""RCT-Diagnose-Ausbau: Schlägt die Propensity-Skill-Warnung an, benennt
_log_propensity_drivers die TREIBER der Treatment-Vorhersage (Top-Features,
Holdout-AUC) und die Treatment-Raten je Quelldatei — trennt die zwei
Ursachen 'gepoolte Experimente' vs. 'Selektion/Post-Treatment-Features'."""
import logging
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rubin.pipelines.analysis_pipeline import AnalysisPipeline


class TestPropensityDrivers:
    def test_leak_feature_is_named_with_auc_and_rates(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 12000
        X = pd.DataFrame({"ALTER": rng.integers(18, 80, n).astype(float),
                          "BEITRAG": rng.normal(90, 20, n),
                          "AFFIN": rng.random(n)})
        T = (rng.random(n) < (0.2 + 0.6 * (X["ALTER"] > 50))).astype(int)
        fs = np.array(["a.parquet"] * (n // 2) + ["b.parquet"] * (n - n // 2))
        pd.DataFrame({"file_source": fs}).to_parquet(tmp_path / "file_source.parquet")
        fake = SimpleNamespace(_logger=logging.getLogger("t"))
        res = AnalysisPipeline._log_propensity_drivers(
            fake, X, T, x_file=str(tmp_path / "X.parquet"))
        assert res["auc"] is not None and res["auc"] > 0.7
        assert res["top_features"][0][0] == "ALTER"
        assert set(res["rates_per_file"]) == {"a.parquet", "b.parquet"}

    def test_mlflow_logging_metric_param_artifact(self, tmp_path):
        import os
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        import mlflow
        old_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        try:
            mlflow.set_experiment("rct_diag_test")
            rng = np.random.default_rng(0)
            n = 8000
            X = pd.DataFrame({"ALTER": rng.integers(18, 80, n).astype(float),
                              "AFFIN": rng.random(n)})
            T = (rng.random(n) < (0.2 + 0.6 * (X["ALTER"] > 50))).astype(int)
            fake = SimpleNamespace(_logger=logging.getLogger("t"))
            with mlflow.start_run() as run:
                AnalysisPipeline._log_propensity_drivers(fake, X, T, mlflow_module=mlflow)
            client = mlflow.tracking.MlflowClient()
            r = client.get_run(run.info.run_id)
            assert r.data.metrics.get("rct_propensity_auc", 0) > 0.6
            assert "ALTER" in r.data.params.get("rct_propensity_top_treiber", "")
            arts = [a.path for a in client.list_artifacts(run.info.run_id)]
            assert "rct_propensity_diagnose.json" in arts
        finally:
            mlflow.set_tracking_uri(old_uri)

    def test_true_rct_shows_no_strong_driver(self):
        rng = np.random.default_rng(1)
        n = 12000
        X = pd.DataFrame({"ALTER": rng.integers(18, 80, n).astype(float),
                          "AFFIN": rng.random(n)})
        T = rng.integers(0, 2, n)  # echtes RCT
        fake = SimpleNamespace(_logger=logging.getLogger("t"))
        res = AnalysisPipeline._log_propensity_drivers(fake, X, T)
        assert res["auc"] is not None and abs(res["auc"] - 0.5) < 0.05
