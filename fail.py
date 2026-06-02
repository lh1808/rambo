=================================================== FAILURES ====================================================
_________________________ TestBuildBaseLearner.test_both_without_learner_type_fallback __________________________
tests/test_tuning.py:398: in test_both_without_learner_type_fallback
    mock.assert_called_once()
.pixi/envs/dev/lib/python3.12/unittest/mock.py:928: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected '_build_catboost_classifier' to have been called once. Called 0 times.
----------------------------------------------- Captured log call -----------------------------------------------
WARNING  rubin.tuning:common.py:430 base_type='both' aber kein '_learner_type' in params — Fallback auf 'catboost' (globaler Default). Das deutet auf ein fehlendes Tuning oder eine Rolle ohne getunte Params hin.
_________________________ TestFinalModelTunerInit.test_fmt_overfit_penalty_scale_floor __________________________
tests/test_tuning.py:489: in test_fmt_overfit_penalty_scale_floor
    assert result == pytest.approx(0.005, abs=1e-6), \
E   AssertionError: Bei val=0.005, gap=0.005 und tolerance=0.05 sollte keine Penalty greifen (got 0.0002500000000000002)
E   assert 0.0002500000000000002 == 0.005 ± 1.0e-06
E     
E     comparison failed
E     Obtained: 0.0002500000000000002
E     Expected: 0.005 ± 1.0e-06
============================================ short test summary info ============================================
FAILED tests/test_tuning.py::TestBuildBaseLearner::test_both_without_learner_type_fallback - AssertionError: Expected '_build_catboost_classifier' to have been called once. Called 0 times.
FAILED tests/test_tuning.py::TestFinalModelTunerInit::test_fmt_overfit_penalty_scale_floor - AssertionError: Bei val=0.005, gap=0.005 und tolerance=0.05 sollte keine Penalty greifen (got 0.000250000000...
======================================== 2 failed, 154 passed in 46.41s =========================================
(generic) ubuntu@192.168.5.237 ~/rubin $ 
