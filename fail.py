22:10:38 INFO [rubin.tuning] tune_all gestartet: models=['NonParamDML', 'DRLearner', 'SLearner', 'TLearner', 'XLearner', 'ParamDML', 'CausalForestDML', 'CausalForest'], X=(389984, 88), Y=(389984,) (unique=[0, 1]), T=(389984,) (unique=[0, 1]), cv_splits=5, n_trials=100, parallel_trials=5, parallel_level=3, cores_per_fit=8, CPU=[effective=40]
22:10:38 INFO [rubin.tuning] K-Fold aktiv (cv=5): Parallele Trials reduziert 10 → 5 (Speicherschutz: jeder Trial hält 5 Fold-Fits gleichzeitig).
22:10:39 INFO [rubin.tuning] Tuning-Task 'catboost__outcome__classifier__all__no_t__y': X_input=389984 rows, indices=389984, X_task=(389984, 88), target=(389984,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome
Traceback (most recent call last):
  File "/mnt/rubin/run_analysis.py", line 132, in <module>
    main()
  File "/mnt/rubin/run_analysis.py", line 128, in main
    pipe.run(export_bundle=args.export_bundle, bundle_dir=args.bundle_dir, bundle_id=args.bundle_id)
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 3019, in run
    tuned_params_by_model = self._run_tuning(cfg, X, T, Y, mlflow, _progress_cb=_progress)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 414, in _run_tuning
    tuned_params_by_model = tuner.tune_all(cfg.models.models_to_train, X=X, Y=Y, T=T)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1197, in tune_all
    best = self._tune_task(task, X=X, Y=Y, T=T, shared_params=tuned_by_task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1058, in _tune_task
    n_jobs=self._tuning_n_jobs(_blt_cv_folds), catch=(Exception,))
                               ^^^^^^^^^^^^^
NameError: name '_blt_cv_folds' is not defined
[W 2026-04-24 20:51:12,734] Trial 2 failed with value None.
