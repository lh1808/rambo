[rubin] Step 1/11: Daten laden & Preprocessing
[rubin] dtypes.json auto-erkannt: 254 Spalten-Dtypes wiederhergestellt (254 kategorial).
[rubin] Step 2/11: Feature-Selektion
[rubin] Step 3/11: Base-Learner-Tuning
[rubin] Kategorische Features: 74 von 165 Spalten → CATBOOST erhält cat_feature-Indizes.
23:08:46 INFO [rubin.analysis] Arbeitsverzeichnis: /mnt/rubin/runs
23:08:46 INFO [rubin.analysis] MLflow-Experiment 'rubin_GRP_PBV' (identisch mit DataPrep).
23:08:46 INFO [rubin.analysis] Run-Name-Suffix 'goldener-otter' aus DataPrep übernommen.
23:08:46 INFO [rubin.analysis] DataPrep-Config nach MLflow geloggt: /mnt/rubin/runs/data/dataprep_config.yml
23:08:46 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
23:08:46 INFO [rubin.analysis] rubin Pipeline Start
23:08:46 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
23:08:46 INFO [rubin.analysis] Config: 8 Modelle (NonParamDML, DRLearner, SLearner, TLearner, XLearner, ParamDML, CausalForestDML, CausalForest), CATBOOST, 5-Fold CV, Parallel-Level 3
23:08:46 INFO [rubin.analysis] Nuisance Cross-Fitting (DML + DR): 5 interne Folds
23:08:46 INFO [rubin.analysis] Aktiv: BL-Tuning (100 Trials) | FMT (100 Trials) | CausalForest-Tuning | Ensemble
23:08:46 INFO [rubin.analysis] Validierungsmodus: Cross-Validation (5 Folds, Seed=42, Tuning-Seed=18)
23:08:46 INFO [rubin.analysis] Historischer Score: 3 NaN-Werte durch 0 ersetzt.
23:08:47 INFO [rubin.analysis] dtypes.json auto-erkannt (runs/data/dtypes.json): 254 Spalten-Dtypes wiederhergestellt (254 kategorial).
23:08:48 INFO [rubin.analysis] Memory-Reduktion: 240.7 MB → 240.7 MB (0% gespart).
23:08:48 INFO [rubin.analysis] Daten geladen: X=(299988, 745), T=(299988,) (unique=[0, 1]), Y=(299988,) (unique=[0, 1]), S=(299988,)
23:08:48 INFO [rubin.categorical] Kategorische Spalten erkannt: 254 von 745 Features (['D_FRAU', 'GESELLSCHAFT_MM', 'BEAMTER_FLG']... (+251)). Patche BOTH .fit()-Methoden für EconML-Kompatibilität.
23:08:49 INFO [rubin.feature_selection] Feature-Selektion: 2 Methoden sequentiell (alle Kerne pro Methode).
23:08:50 INFO [rubin.gpu] CatBoost GPU erkannt: 1 NVIDIA GPU(s) verfügbar → task_type='GPU'. Inkompatible Parameter (colsample_bylevel, rsm) werden automatisch entfernt.
23:08:50 INFO [rubin.categorical] CatBoost categorical patch (fit): 254 cat_features injiziert (DataFrame).
23:09:32 INFO [rubin.feature_selection] CausalForest FS: X=(299988, 745) (dtypes: 491 numeric, 254 category), T=(299988,) (unique=2), Y=(299988,), n_jobs=-1, in_thread=False
23:09:34 INFO [rubin.feature_selection] CausalForest FS: Subsampling 299988 → 99999 Zeilen (stratifiziert nach T).
23:09:34 INFO [rubin.feature_selection] CausalForest FS: fit(99999×745, T unique=2, n_estimators=100, n_jobs=-1)...
23:21:53 INFO [rubin.feature_selection] Korrelationsfilter (|r| > 0.90, importance-gesteuert): 87 Features entfernt, 404 verbleiben.
23:21:54 INFO [rubin.analysis] Importance-Umverteilung: 87 entfernte Features → Importance auf Partner übertragen.
23:21:54 INFO [rubin.feature_selection] Feature-Selection 'catboost_importance': Top-15% = 99 / 658 Features.
23:21:54 INFO [rubin.feature_selection] Feature-Selection 'causal_forest': Top-15% = 99 / 658 Features.
23:21:54 INFO [rubin.feature_selection] Feature-Selection Union: 165 / 658 Features behalten, 493 entfernt.
23:21:54 INFO [rubin.analysis] Feature-Selektion gesamt: 745 → 165 Features (Korrelation: −87, Importance: −493)
23:21:54 INFO [rubin.analysis] Feature-Selektion: 745 → 165 Features (-580 entfernt)
23:21:54 INFO [rubin.categorical] Kategorische Spalten erkannt: 74 von 165 Features (['GESELLSCHAFT_MM', 'PLZ1', 'PLZ_1A']... (+71)). Patche CATBOOST .fit()-Methoden für EconML-Kompatibilität.
23:21:54 INFO [rubin.analysis] Starte Tuning: X=(299988, 165), Y=(299988,) (unique=[0, 1]), T=(299988,) (unique=[0, 1])
23:21:54 INFO [rubin.tuning] tune_all gestartet: models=['NonParamDML', 'DRLearner', 'SLearner', 'TLearner', 'XLearner', 'ParamDML', 'CausalForestDML', 'CausalForest'], X=(299988, 165), Y=(299988,) (unique=[0, 1]), T=(299988,) (unique=[0, 1]), cv_splits=5, n_trials=100, parallel_trials=20, parallel_level=3, cores_per_fit=4, CPU=[effective=80]
23:21:54 INFO [rubin.tuning] Tuning-Plan: 7 Tasks für 8 Modelle.
23:21:54 INFO [rubin.tuning] Tuning-Task 'catboost__outcome__classifier__all__no_t__y': X_input=299988 rows, indices=299988, X_task=(299988, 165), target=(299988,) (unique=[0, 1]), train_subsample=80%, T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome
23:21:54 INFO [rubin.categorical] CatBoost categorical patch (fit): 74 cat_features injiziert (DataFrame).
03:39:09 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__no_t__y': 72/100 Trials abgeschlossen (0 fehlgeschlagen, 28 gepruned, parallel=1).
03:39:09 INFO [rubin.tuning] Tuning-Task 'catboost__outcome_regression__regressor__all__with_t__y': X_input=299988 rows, indices=299988, X_task=(299988, 166), target=(299988,) (unique=[0, 1]), train_subsample=80%, T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome_regression
06:54:34 INFO [rubin.tuning] BLT 'catboost__outcome_regression__regressor__all__with_t__y': 74/100 Trials abgeschlossen (0 fehlgeschlagen, 26 gepruned, parallel=1).
06:54:35 INFO [rubin.tuning] Tuning-Task 'catboost__outcome_regression__regressor__all_direct__with_t__y': X_input=299988 rows, indices=299988, X_task=(299988, 166), target=(299988,) (unique=[0, 1]), train_subsample=100%, T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome_regression
10:28:42 INFO [rubin.tuning] BLT 'catboost__outcome_regression__regressor__all_direct__with_t__y': 74/100 Trials abgeschlossen (0 fehlgeschlagen, 26 gepruned, parallel=1).
10:28:42 INFO [rubin.tuning] Tuning-Task 'catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': X_input=299988 rows, indices=299988, X_task=(299988, 165), target=(299988,) (unique=[0, 1]), train_subsample=100%, T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=grouped_outcome_regression
