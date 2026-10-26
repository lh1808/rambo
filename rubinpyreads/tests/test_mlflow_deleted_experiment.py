"""set_experiment_robust: Ein in der MLflow-UI GELÖSCHTES Experiment
(Papierkorb) blockierte jeden rubin-Lauf gleichen Namens — realer Vorfall:
analyze-quick auf Domino crashte mit "Cannot set a deleted experiment
'rubin_quickstart' as the active experiment". Der Helper restored und
wiederholt; fremde Fehler werden unverändert durchgereicht."""
import logging
import os

import pytest


@pytest.fixture()
def _mlflow(tmp_path):
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    import mlflow
    old = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    yield mlflow
    mlflow.set_tracking_uri(old)


class TestSetExperimentRobust:
    def test_deleted_experiment_is_restored_and_set(self, _mlflow):
        from rubin.utils.mlflow_compat import set_experiment_robust
        exp_id = _mlflow.create_experiment("robust_test")
        _mlflow.tracking.MlflowClient().delete_experiment(exp_id)
        with pytest.raises(Exception, match="deleted experiment"):
            _mlflow.set_experiment("robust_test")  # Repro des Vorfalls
        exp = set_experiment_robust(_mlflow, "robust_test", logger=logging.getLogger("t"))
        assert exp.experiment_id == exp_id and exp.lifecycle_stage == "active"

    def test_fresh_name_passes_through(self, _mlflow):
        from rubin.utils.mlflow_compat import set_experiment_robust
        exp = set_experiment_robust(_mlflow, "neu_angelegt")
        assert exp.lifecycle_stage == "active"

    def test_foreign_errors_are_reraised(self):
        from rubin.utils.mlflow_compat import set_experiment_robust
        class _Boom:
            @staticmethod
            def set_experiment(name):
                raise RuntimeError("ganz anderer Fehler")
        with pytest.raises(RuntimeError, match="ganz anderer"):
            set_experiment_robust(_Boom, "x")
