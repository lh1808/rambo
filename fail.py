15:02:06 INFO [rubin.analysis] Starte Tuning: X=(299988, 128), Y=(299988,) (unique=[0, 1]), T=(299988,) (unique=[0, 1])
15:02:06 INFO [rubin.tuning] tune_all gestartet: models=['NonParamDML', 'DRLearner', 'SLearner', 'TLearner', 'ParamDML', 'CausalForestDML', 'CausalForest'], X=(299988, 128), Y=(299988,) (unique=[0, 1]), T=(299988,) (unique=[0, 1]), cv_splits=5, n_trials=160, parallel_trials=10
15:02:06 INFO [rubin.tuning] BLT-Modellfilter: ['XLearner'] übersprungen (nutzen fixed_params). Aktiv: ['NonParamDML', 'DRLearner', 'SLearner', 'TLearner', 'ParamDML', 'CausalForestDML', 'CausalForest']
15:02:07 INFO [rubin.tuning] Tuning-Task 'catboost__outcome__classifier__all__no_t__y': X_input=299988 rows, indices=299988, X_task=(299988, 128), target=(299988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome
Traceback (most recent call last):
  File "/mnt/rubin/run_analysis.py", line 132, in <module>
    main()
  File "/mnt/rubin/run_analysis.py", line 128, in main
    pipe.run(export_bundle=args.export_bundle, bundle_dir=args.bundle_dir, bundle_id=args.bundle_id)
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 2915, in run
    tuned_params_by_model = self._run_tuning(cfg, X, T, Y, mlflow)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 414, in _run_tuning
    tuned_params_by_model = tuner.tune_all(cfg.models.models_to_train, X=X, Y=Y, T=T)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1231, in tune_all
    best = self._tune_task(task, X=X, Y=Y, T=T, shared_params=tuned_by_task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 924, in _tune_task
    self._logger.info(
    ^^^^^^^^^^^^
AttributeError: 'BaseLearnerTuner' object has no attribute '_logger'
