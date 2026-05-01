12:28:18 INFO [rubin.categorical] Kategorische Spalten erkannt: 42 von 159 Features (['D_FRAU', 'PLZ1', 'PLZ_2A']... (+39)). Patche CATBOOST .fit()-Methoden für EconML-Kompatibilität.
12:28:18 INFO [rubin.analysis] Starte Tuning: X=(299988, 159), Y=(299988,) (unique=[0, 1]), T=(299988,) (unique=[0, 1])
12:28:18 INFO [rubin.tuning] tune_all gestartet: models=['NonParamDML', 'DRLearner', 'SLearner', 'TLearner', 'XLearner', 'ParamDML', 'CausalForestDML', 'CausalForest'], X=(299988, 159), Y=(299988,) (unique=[0, 1]), T=(299988,) (unique=[0, 1]), cv_splits=5, n_trials=100, parallel_trials=10, parallel_level=3, cores_per_fit=4, CPU=[effective=40]
12:28:18 INFO [rubin.tuning] Tuning-Plan: 7 Tasks für 8 Modelle.
12:28:18 INFO [rubin.tuning] Tuning-Task 'catboost__outcome__classifier__all__no_t__y': X_input=299988 rows, indices=299988, X_task=(299988, 159), target=(299988,) (unique=[0, 1]), train_subsample=80%, T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome
12:28:39 INFO [rubin.categorical] CatBoost categorical patch (fit): 42/159 Spalten float→int konvertiert.
12:34:18 INFO [rubin.categorical] CatBoost categorical patch (predict): 42 Spalten float→int konvertiert.
