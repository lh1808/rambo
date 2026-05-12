Analyse fehlgeschlagen: Fehlgeschlagen (Exit 1)

Details:
09:54:59 INFO [rubin.analysis] Arbeitsverzeichnis: /mnt/rubin/runs
09:55:00 INFO [rubin.analysis] MLflow-Experiment 'rubin_WG_WG' (identisch mit DataPrep).
09:55:00 INFO [rubin.analysis] Run-Name-Suffix 'scharfer-wolf' aus DataPrep übernommen.
09:55:00 INFO [rubin.analysis] DataPrep-Config nach MLflow geloggt: /mnt/rubin/runs/data/dataprep_config.yml
09:55:00 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
09:55:00 INFO [rubin.analysis] rubin Pipeline Start
09:55:00 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
09:55:00 INFO [rubin.analysis] Config: 8 Modelle (NonParamDML, DRLearner, SLearner, TLearner, XLearner, ParamDML, CausalForestDML, CausalForest), CATBOOST, 3-Fold CV, Parallel-Level 3
09:55:00 INFO [rubin.analysis] Nuisance Cross-Fitting (DML + DR): 5 interne Folds
09:55:00 INFO [rubin.analysis] Aktiv: BL-Tuning (100 Trials) | FMT (50 Trials) | CFT (50 Trials) | Ensemble
09:55:00 INFO [rubin.analysis] Validierungsmodus: Cross-Validation (3 Folds, Seed=42, Tuning-Seed=18)
09:55:01 INFO [rubin.analysis] Historischer Score: 606 NaN-Werte durch 0 ersetzt.
09:55:01 INFO [rubin.analysis] Daten: 38,998 Zeilen, 745 Features | T: 19303 Control (49.5%), 19695 Treatment (50.5%) | Y: 0.3% positiv
09:55:01 INFO [rubin.analysis] Historischer Score (S): vorhanden (38998 Werte)
09:55:01 INFO [rubin.analysis] dtypes.json auto-erkannt (runs/data/dtypes.json): 254 Spalten-Dtypes wiederhergestellt (254 kategorial).
09:55:02 INFO [rubin.analysis] Memory-Reduktion: 31.8 MB → 31.8 MB (0% gespart).
09:55:02 INFO [rubin.analysis] Daten geladen: X=(38998, 745), T=(38998,) (unique=[0, 1]), Y=(38998,) (unique=[0, 1]), S=(38998,)
09:55:02 INFO [rubin.categorical] Kategorische Spalten erkannt: 254 von 745 Features (['D_FRAU', 'GESELLSCHAFT_MM', 'BEAMTER_FLG']... (+251)). Patche BOTH Methoden (fit/predict/predict_proba/score) für EconML-Kompatibilität.
09:55:03 INFO [rubin.feature_selection] Feature-Selektion: 2 Methoden sequentiell (alle Kerne pro Methode).
09:55:03 INFO [rubin.categorical] CatBoost categorical patch (fit): 254 cat_features injiziert (DataFrame).
09:55:12 INFO [rubin.feature_selection] CausalForest FS: X=(38998, 745) (dtypes: 491 numeric, 254 category), T=(38998,) (unique=2), Y=(38998,), n_jobs=-1, in_thread=False
09:55:12 INFO [rubin.feature_selection] CausalForest FS: fit(38998×745, T unique=2, n_estimators=100, n_jobs=-1)...
09:56:16 INFO [rubin.feature_selection] Korrelationsfilter (|r| > 0.90, importance-gesteuert): 221 Features entfernt, 270 verbleiben.
09:56:16 INFO [rubin.analysis] Importance-Umverteilung: 221 entfernte Features → Importance auf Partner übertragen.
09:56:16 INFO [rubin.feature_selection] Feature-Selection 'catboost_importance': Top-50 = 50 / 524 Features (Budget: 50 pro Methode).
09:56:16 INFO [rubin.feature_selection] Feature-Selection 'causal_forest': Top-50 = 50 / 524 Features (Budget: 50 pro Methode).
09:56:16 INFO [rubin.feature_selection] Feature-Selection: 82 / 524 Features behalten (max_features=100, 2 Methoden × 50 pro Methode).
09:56:17 INFO [rubin.analysis] Feature-Selektion gesamt: 745 → 82 Features (Korrelation: −221, Importance: −442)
09:56:17 INFO [rubin.analysis] Feature-Selektion: 745 → 82 Features (-663 entfernt)
09:56:17 INFO [rubin.categorical] Kategorische Spalten erkannt: 29 von 82 Features (['GESELLSCHAFT_MM', 'PLZ1', 'PLZ_1A']... (+26)). Patche CATBOOST Methoden (fit/predict/predict_proba/score) für EconML-Kompatibilität.
09:56:17 INFO [rubin.analysis] Starte Tuning: X=(38998, 82), Y=(38998,) (unique=[0, 1]), T=(38998,) (unique=[0, 1])
09:56:17 INFO [rubin.tuning] tune_all gestartet: models=['NonParamDML', 'DRLearner', 'SLearner', 'TLearner', 'XLearner', 'ParamDML', 'CausalForestDML', 'CausalForest'], X=(38998, 82), Y=(38998,) (unique=[0, 1]), T=(38998,) (unique=[0, 1]), cv_splits=5, n_trials=100, parallel_trials=20, parallel_level=3, cores_per_fit=4, CPU=[effective=80]
09:56:17 INFO [rubin.tuning] Tuning-Plan: 7 Tasks für 8 Modelle.
09:56:17 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__no_t__y': X=(38998, 82), target=(38998,) (unique=[0, 1]), subsample=80%, cv=5, objective=outcome
09:56:17 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__no_t__y': Starte 100 Trials (parallel=20, 5-Fold, n_jobs=4).
09:56:17 INFO [rubin.categorical] CatBoost categorical patch (fit): 29 cat_features injiziert (DataFrame).
10:00:24 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__no_t__y': 53/100 Trials abgeschlossen (0 fehlgeschlagen, 47 gepruned, parallel=20).
10:00:24 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__no_t__y': Study freigegeben, gc.collect() + malloc_trim durchgeführt.
10:00:24 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__with_t__y': X=(38998, 83), target=(38998,) (unique=[0, 1]), subsample=80%, cv=5, objective=outcome
10:00:24 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__with_t__y': Starte 100 Trials (parallel=20, 5-Fold, n_jobs=4).
10:04:47 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__with_t__y': 69/100 Trials abgeschlossen (0 fehlgeschlagen, 31 gepruned, parallel=20).
10:04:47 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__with_t__y': Study freigegeben, gc.collect() + malloc_trim durchgeführt.
10:04:47 INFO [rubin.tuning] BLT 'catboost__outcome_regression__regressor__all_direct__with_t__y': X=(38998, 83), target=(38998,) (unique=[0, 1]), subsample=100%, cv=5, objective=outcome_regression
10:04:47 INFO [rubin.tuning] BLT 'catboost__outcome_regression__regressor__all_direct__with_t__y': Starte 100 Trials (parallel=20, 5-Fold, n_jobs=4).
10:09:49 INFO [rubin.tuning] BLT 'catboost__outcome_regression__regressor__all_direct__with_t__y': 65/100 Trials abgeschlossen (0 fehlgeschlagen, 35 gepruned, parallel=20).
10:09:49 INFO [rubin.tuning] BLT 'catboost__outcome_regression__regressor__all_direct__with_t__y': Study freigegeben, gc.collect() + malloc_trim durchgeführt.
10:09:49 INFO [rubin.tuning] BLT 'catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': X=(38998, 82), target=(38998,) (unique=[0, 1]), subsample=100%, cv=5, objective=grouped_outcome_regression
10:09:49 INFO [rubin.tuning] BLT 'catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': Starte 100 Trials (parallel=20, 5-Fold, n_jobs=4).
10:15:41 INFO [rubin.tuning] BLT 'catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': 73/100 Trials abgeschlossen (0 fehlgeschlagen, 27 gepruned, parallel=20).
10:15:41 INFO [rubin.tuning] BLT 'catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': Study freigegeben, gc.collect() + malloc_trim durchgeführt.
10:15:41 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all__no_t__t': X=(38998, 82), target=(38998,) (unique=[0, 1]), subsample=80%, cv=5, objective=propensity
10:15:41 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all__no_t__t': Starte 20 Trials (parallel=20, 5-Fold, n_jobs=4).
10:17:10 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all__no_t__t': 20/20 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned, parallel=20).
10:17:10 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all__no_t__t': Study freigegeben, gc.collect() + malloc_trim durchgeführt.
10:17:10 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all_direct__no_t__t': X=(38998, 82), target=(38998,) (unique=[0, 1]), subsample=100%, cv=5, objective=propensity
10:17:10 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all_direct__no_t__t': Starte 20 Trials (parallel=20, 5-Fold, n_jobs=4).
10:18:33 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all_direct__no_t__t': 20/20 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned, parallel=20).
10:18:33 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all_direct__no_t__t': Study freigegeben, gc.collect() + malloc_trim durchgeführt.
10:18:34 INFO [rubin.tuning] BLT 'catboost__pseudo_effect__regressor__group_specific_shared_params__no_t__d': X=(38998, 82), target=(38998,) (unique=[0, 1]), subsample=100%, cv=5, objective=pseudo_effect
10:18:34 INFO [rubin.tuning] BLT 'catboost__pseudo_effect__regressor__group_specific_shared_params__no_t__d': Starte 100 Trials (parallel=20, 5-Fold, n_jobs=4).
10:27:19 INFO [rubin.tuning] BLT 'catboost__pseudo_effect__regressor__group_specific_shared_params__no_t__d': 84/100 Trials abgeschlossen (0 fehlgeschlagen, 16 gepruned, parallel=20).
10:27:20 INFO [rubin.tuning] BLT 'catboost__pseudo_effect__regressor__group_specific_shared_params__no_t__d': Study freigegeben, gc.collect() + malloc_trim durchgeführt.
10:27:20 INFO [rubin.analysis] Base-Learner-Tuning: 7 Tasks abgeschlossen.
10:27:20 INFO [rubin.analysis]   catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y best=-0.00290438  skill=0.0035
10:27:20 INFO [rubin.analysis]   catboost__outcome__classifier__all__no_t__y best=-0.0193073  skill=0.0335
10:27:20 INFO [rubin.analysis]   catboost__outcome__classifier__all__with_t__y best=-0.0191988  skill=0.0389
10:27:20 INFO [rubin.analysis]   catboost__outcome_regression__regressor__all_direct__with_t__y best=-0.00291023  skill=0.0015
10:27:20 INFO [rubin.analysis]   catboost__propensity__classifier__all__no_t__t best=-0.693101  skill=-0.0000
10:27:20 INFO [rubin.analysis]   catboost__propensity__classifier__all_direct__no_t__t best=-0.693027  skill=0.0001
10:27:20 INFO [rubin.analysis]   catboost__pseudo_effect__regressor__group_specific_shared_params__no_t__d best=-0.00290451  skill=0.0035
10:27:39 INFO [rubin.categorical] CatBoost categorical patch (predict): 29 Spalten float→int konvertiert.
10:32:01 INFO [rubin.tuning] FMT 'NonParamDML': Nuisance gecacht für 3 äußere Folds (je 5 innere × model_y + model_t). Trials fitten nur noch model_final via refit_final().
10:32:01 INFO [rubin.tuning] FMT 'NonParamDML': Starte 50 Trials (cache_values, 3-Fold, parallel_jobs=-1).
10:40:33 INFO [rubin.tuning] FMT 'NonParamDML': 23/50 Trials abgeschlossen (0 fehlgeschlagen, 27 gepruned, parallel=1).
10:40:34 INFO [rubin.tuning] FMT 'NonParamDML': Study + Cache freigegeben, gc.collect() durchgeführt.
10:43:57 INFO [rubin.tuning] FMT 'DRLearner': Nuisance gecacht für 3 äußere Folds (je 5 innere × model_propensity + model_regression). Trials fitten nur noch model_final via refit_final().
10:43:57 INFO [rubin.tuning] FMT 'DRLearner': Starte 50 Trials (cache_values, 3-Fold, parallel_jobs=-1).
10:53:35 INFO [rubin.tuning] FMT 'DRLearner': 27/50 Trials abgeschlossen (0 fehlgeschlagen, 23 gepruned, parallel=1).
10:53:35 INFO [rubin.tuning] FMT 'DRLearner': Study + Cache freigegeben, gc.collect() durchgeführt.
10:53:35 INFO [rubin.analysis] Final-Model-Tuning: 2 Modelle abgeschlossen.
10:53:35 INFO [rubin.analysis] FMT 'NonParamDML': OOF-R-Loss → -0.00291148
10:53:35 INFO [rubin.analysis] FMT 'DRLearner': OOF-DR-MSE → -0.0115864
11:02:33 INFO [rubin.tuning] CFT 'CausalForestDML': Nuisance gecacht für 5 äußere Folds (je 5 innere × model_y + model_t). Trials fitten nur noch den CausalForest via refit_final().
11:02:33 INFO [rubin.tuning] CFT 'CausalForestDML': Starte 50 Trials (cache_values, 5-Fold, parallel_jobs=-1).
11:02:34 INFO [rubin.tuning] CFT 'CausalForestDML': 50/50 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned, parallel=1).
11:02:34 INFO [rubin.tuning] CFT 'CausalForestDML': Study + Cache freigegeben, gc.collect() durchgeführt.
11:02:34 INFO [rubin.analysis]   CausalForestDML: Starte 3-Fold Cross-Prediction (Seed=42).
11:02:34 INFO [rubin.training] CausalForestDML: Erzwinge sequentielle Folds (GRF nutzt intern joblib-Prozesse, die in Threads zu Deadlocks führen).
11:02:34 INFO [rubin.training] CausalForestDML: 3 Folds sequentiell.
Traceback (most recent call last):
  File "/mnt/rubin/run_analysis.py", line 132, in <module>
    main()
  File "/mnt/rubin/run_analysis.py", line 128, in main
    pipe.run(export_bundle=args.export_bundle, bundle_dir=args.bundle_dir, bundle_id=args.bundle_id)
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 3097, in run
    models, preds, fold_models = self._run_training(cfg, X, T, Y, tuned_params_by_model, holdout_data, mlflow, _progress_cb=_progress)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 921, in _run_training
    result = train_and_crosspredict_bt_bo(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/training.py", line 312, in train_and_crosspredict_bt_bo
    preds, last_fold_model, last_fold_val_idx = _run_folds_sequential(
                                                ^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/training.py", line 165, in _run_folds_sequential
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/grf/_base_grf.py", line 340, in fit
    raise ValueError("Parameter `max_samples` must be in (0, .5], if `inference=True`. "
ValueError: Parameter `max_samples` must be in (0, .5], if `inference=True`. Got value 0.7500817716636065
