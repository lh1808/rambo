13:12:00 INFO [rubin.tuning] CFT 'CausalForestDML': Starte 30 Trials (cache_values, 5-Fold, parallel_jobs=-1).
13:12:52 INFO [rubin.tuning] CFT Diagnose (Trial 0, Fold 0): set={'n_estimators': 360, 'max_depth': 17, 'min_samples_leaf': 177, 'min_samples_split': 99, 'max_features': 0.8670097618071374, 'max_samples': 0.41254770013710373, 'min_var_fraction_leaf': 0.03330541727259865, 'min_impurity_decrease': 0.009878954482759623}, actual={'n_estimators': 360, 'max_depth': 17, 'min_samples_leaf': 177, 'min_samples_split': 99, 'max_features': 0.8670097618071374, 'max_samples': 0.41254770013710373, 'min_var_fraction_leaf': 0.03330541727259865, 'min_impurity_decrease': 0.009878954482759623}, score=-0.00267966
15:21:30 INFO [rubin.tuning] CFT 'CausalForestDML': 30/30 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned, parallel=1).
15:21:30 INFO [rubin.tuning] CFT 'CausalForestDML': Study + Cache freigegeben, gc.collect() durchgeführt.
15:21:30 INFO [rubin.analysis]   CausalForestDML: Starte 5-Fold Cross-Prediction (Seed=42).
15:21:31 INFO [rubin.training] CausalForestDML: Erzwinge sequentielle Folds (GRF nutzt intern joblib-Prozesse, die in Threads zu Deadlocks führen).
15:21:31 INFO [rubin.training] CausalForestDML: 5 Folds sequentiell.
Traceback (most recent call last):
  File "/mnt/rubin/run_analysis.py", line 132, in <module>
    main()
  File "/mnt/rubin/run_analysis.py", line 128, in main
    pipe.run(export_bundle=args.export_bundle, bundle_dir=args.bundle_dir, bundle_id=args.bundle_id)
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 3226, in run
    models, preds, fold_models = self._run_training(cfg, X, T, Y, tuned_params_by_model, holdout_data, mlflow, _progress_cb=_progress)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 949, in _run_training
    result = train_and_crosspredict_bt_bo(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/training.py", line 336, in train_and_crosspredict_bt_bo
    preds, last_fold_model, last_fold_val_idx, fold_full_preds = _run_folds_sequential(
                                                                 ^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/training.py", line 176, in _run_folds_sequential
    m.fit(Y[tr_idx], T[tr_idx], X=X.iloc[tr_idx])
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/causal_forest.py", line 964, in fit
    return super().fit(Y, T, X=X, W=W,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 501, in fit
    return super().fit(Y, T, X=X, W=W,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_cate_estimator.py", line 134, in call
    m(self, Y, T, *args, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 876, in fit
    self._fit_final(Y=Y,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 979, in _fit_final
    self._ortho_learner_model_final.fit(Y, T, **filter_none_kwargs(X=X, W=W, Z=Z,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 101, in fit
    self._model_final.fit(X, T, T_res, Y_res, sample_weight=sample_weight,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/causal_forest.py", line 66, in fit
    self._model.fit(fts, T_res, Y_res, sample_weight=sample_weight)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/grf/classes.py", line 37, in fit
    [estimator.fit(X, T, y[:, [it]], sample_weight=sample_weight, **kwargs)
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/grf/classes.py", line 395, in fit
    return super().fit(X, T, y, sample_weight=sample_weight)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/grf/_base_grf.py", line 331, in fit
    raise ValueError("The number of estimators to be constructed must be divisible "
ValueError: The number of estimators to be constructed must be divisible the `subforest_size` parameter. Asked to build `n_estimators=466` with `subforest_size=4`.
