16:58:06 INFO [rubin.analysis] Starte Tuning: X=(299988, 128), Y=(299988,) (unique=[0, 1]), T=(299988,) (unique=[0, 1])
16:58:06 INFO [rubin.tuning] tune_all gestartet: models=['NonParamDML', 'DRLearner', 'SLearner', 'TLearner', 'ParamDML', 'CausalForestDML', 'CausalForest'], X=(299988, 128), Y=(299988,) (unique=[0, 1]), T=(299988,) (unique=[0, 1]), cv_splits=5, n_trials=160, parallel_trials=10
16:58:06 INFO [rubin.tuning] BLT-Modellfilter: ['XLearner'] übersprungen (nutzen fixed_params). Aktiv: ['NonParamDML', 'DRLearner', 'SLearner', 'TLearner', 'ParamDML', 'CausalForestDML', 'CausalForest']
16:58:06 INFO [rubin.tuning] Tuning-Task 'catboost__outcome__classifier__all__no_t__y': X_input=299988 rows, indices=299988, X_task=(299988, 128), target=(299988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome
17:12:12 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__no_t__y': 160/160 Trials abgeschlossen (parallel=10, Wellen≈16).
17:12:12 INFO [rubin.tuning] Tuning-Task 'catboost__outcome_regression__regressor__all__with_t__y': X_input=299988 rows, indices=299988, X_task=(299988, 129), target=(299988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome_regression
17:25:25 INFO [rubin.tuning] BLT 'catboost__outcome_regression__regressor__all__with_t__y': 160/160 Trials abgeschlossen (parallel=10, Wellen≈16).
17:25:25 INFO [rubin.tuning] Tuning-Task 'catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': X_input=299988 rows, indices=299988, X_task=(299988, 128), target=(299988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=grouped_outcome_regression
17:34:31 INFO [rubin.tuning] BLT 'catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': 160/160 Trials abgeschlossen (parallel=10, Wellen≈16).
17:34:32 INFO [rubin.tuning] Tuning-Task 'catboost__propensity__classifier__all__no_t__t': X_input=299988 rows, indices=299988, X_task=(299988, 128), target=(299988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=T, objective=propensity
17:43:25 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all__no_t__t': 160/160 Trials abgeschlossen (parallel=10, Wellen≈16).
17:43:25 INFO [rubin.analysis] Base-Learner-Tuning: 4 Tasks abgeschlossen (160 Trials/Task). Ø Score: -0.17727
Traceback (most recent call last):
  File "/mnt/rubin/run_analysis.py", line 132, in <module>
    main()
  File "/mnt/rubin/run_analysis.py", line 128, in main
    pipe.run(export_bundle=args.export_bundle, bundle_dir=args.bundle_dir, bundle_id=args.bundle_id)
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 2915, in run
    tuned_params_by_model = self._run_tuning(cfg, X, T, Y, mlflow)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 516, in _run_tuning
    add = final_tuner.tune_final_model(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1376, in tune_final_model
    _tlog = logging.getLogger("rubin.tuning")
            ^^^^^^^
UnboundLocalError: cannot access local variable 'logging' where it is not associated with a value
