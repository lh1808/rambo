Analyse fehlgeschlagen: Fehlgeschlagen (Exit -11)

Details:
21:01:10 INFO [rubin.analysis] Arbeitsverzeichnis: /mnt/rubin/runs
21:01:10 INFO [rubin.analysis] MLflow-Experiment 'rubin_WG_WG' (identisch mit DataPrep).
21:01:10 INFO [rubin.analysis] Run-Name-Suffix 'kühler-otter' aus DataPrep übernommen.
21:01:10 INFO [rubin.analysis] DataPrep-Config nach MLflow geloggt: /mnt/rubin/runs/data/dataprep_config.yml
21:01:10 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
21:01:10 INFO [rubin.analysis] rubin Pipeline Start
21:01:10 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
21:01:10 INFO [rubin.analysis] Config: 8 Modelle (NonParamDML, DRLearner, SLearner, TLearner, XLearner, ParamDML, CausalForestDML, CausalForest), CATBOOST, 5-Fold CV, Parallel-Level 3
21:01:10 INFO [rubin.analysis] Nuisance Cross-Fitting (DML + DR): 5 interne Folds
21:01:10 INFO [rubin.analysis] Aktiv: BL-Tuning (100 Trials) | FMT (50 Trials) | CFT (50 Trials) | Ensemble
21:01:10 INFO [rubin.analysis] Validierungsmodus: Cross-Validation (5 Folds, Seed=42, Tuning-Seed=18)
21:01:11 INFO [rubin.analysis] Historischer Score: 606 NaN-Werte durch 0 ersetzt.
21:01:12 INFO [rubin.analysis] dtypes.json auto-erkannt (runs/data/dtypes.json): 254 Spalten-Dtypes wiederhergestellt (254 kategorial).
21:01:13 INFO [rubin.analysis] Memory-Reduktion: 312.7 MB → 312.7 MB (0% gespart).
21:01:13 INFO [rubin.analysis] Daten geladen: X=(389983, 745), T=(389983,) (unique=[0, 1]), Y=(389983,) (unique=[0, 1]), S=(389983,)
21:01:14 INFO [rubin.categorical] Kategorische Spalten erkannt: 254 von 745 Features (['D_FRAU', 'GESELLSCHAFT_MM', 'BEAMTER_FLG']... (+251)). Patche BOTH Methoden (fit/predict/predict_proba/score) für EconML-Kompatibilität.
21:01:15 INFO [rubin.feature_selection] Feature-Selektion: 2 Methoden sequentiell (alle Kerne pro Methode).
21:01:15 INFO [rubin.categorical] CatBoost categorical patch (fit): 254 cat_features injiziert (DataFrame).
21:01:46 INFO [rubin.feature_selection] CausalForest FS: X=(389983, 745) (dtypes: 491 numeric, 254 category), T=(389983,) (unique=2), Y=(389983,), n_jobs=-1, in_thread=False
21:01:49 INFO [rubin.feature_selection] CausalForest FS: Subsampling 389983 → 99999 Zeilen (stratifiziert nach T).
21:01:49 INFO [rubin.feature_selection] CausalForest FS: fit(99999×745, T unique=2, n_estimators=100, n_jobs=-1)...
21:12:56 INFO [rubin.feature_selection] Korrelationsfilter (|r| > 0.90, importance-gesteuert): 223 Features entfernt, 268 verbleiben.
21:12:56 INFO [rubin.analysis] Importance-Umverteilung: 223 entfernte Features → Importance auf Partner übertragen.
21:12:56 INFO [rubin.feature_selection] Feature-Selection 'catboost_importance': Top-50 = 50 / 522 Features (Budget: 50 pro Methode).
21:12:56 INFO [rubin.feature_selection] Feature-Selection 'causal_forest': Top-50 = 50 / 522 Features (Budget: 50 pro Methode).
21:12:56 INFO [rubin.feature_selection] Feature-Selection: 80 / 522 Features behalten (max_features=100, 2 Methoden × 50 pro Methode).
21:12:56 INFO [rubin.analysis] Feature-Selektion gesamt: 745 → 80 Features (Korrelation: −223, Importance: −442)
21:12:56 INFO [rubin.analysis] Feature-Selektion: 745 → 80 Features (-665 entfernt)
21:12:56 INFO [rubin.categorical] Kategorische Spalten erkannt: 29 von 80 Features (['MOBIL_PRIVAT', 'PLZ1', 'PLZ_2A']... (+26)). Patche CATBOOST Methoden (fit/predict/predict_proba/score) für EconML-Kompatibilität.
21:12:56 INFO [rubin.analysis] Starte Tuning: X=(389983, 80), Y=(389983,) (unique=[0, 1]), T=(389983,) (unique=[0, 1])
21:12:57 INFO [rubin.tuning] tune_all gestartet: models=['NonParamDML', 'DRLearner', 'SLearner', 'TLearner', 'XLearner', 'ParamDML', 'CausalForestDML', 'CausalForest'], X=(389983, 80), Y=(389983,) (unique=[0, 1]), T=(389983,) (unique=[0, 1]), cv_splits=5, n_trials=100, parallel_trials=20, parallel_level=3, cores_per_fit=4, CPU=[effective=80]
21:12:57 INFO [rubin.tuning] Tuning-Plan: 7 Tasks für 8 Modelle.
21:12:57 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__no_t__y': X=(389983, 80), target=(389983,) (unique=[0, 1]), subsample=80%, cv=5, objective=outcome
21:12:57 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__no_t__y': Starte 100 Trials (parallel=20, 5-Fold, n_jobs=4).
21:12:57 INFO [rubin.categorical] CatBoost categorical patch (fit): 29 cat_features injiziert (DataFrame).
22:05:06 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__no_t__y': 77/100 Trials abgeschlossen (0 fehlgeschlagen, 23 gepruned, parallel=20).
22:05:07 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__no_t__y': Study freigegeben, gc.collect() durchgeführt.
22:05:07 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__with_t__y': X=(389983, 81), target=(389983,) (unique=[0, 1]), subsample=80%, cv=5, objective=outcome
22:05:07 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__with_t__y': Starte 100 Trials (parallel=20, 5-Fold, n_jobs=4).
22:56:58 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__with_t__y': 89/100 Trials abgeschlossen (0 fehlgeschlagen, 11 gepruned, parallel=20).
22:56:58 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__with_t__y': Study freigegeben, gc.collect() durchgeführt.
22:56:58 INFO [rubin.tuning] BLT 'catboost__outcome_regression__regressor__all_direct__with_t__y': X=(389983, 81), target=(389983,) (unique=[0, 1]), subsample=100%, cv=5, objective=outcome_regression
22:56:58 INFO [rubin.tuning] BLT 'catboost__outcome_regression__regressor__all_direct__with_t__y': Starte 100 Trials (parallel=20, 5-Fold, n_jobs=4).
23:54:21 INFO [rubin.tuning] BLT 'catboost__outcome_regression__regressor__all_direct__with_t__y': 99/100 Trials abgeschlossen (0 fehlgeschlagen, 1 gepruned, parallel=20).
23:54:21 INFO [rubin.tuning] BLT 'catboost__outcome_regression__regressor__all_direct__with_t__y': Study freigegeben, gc.collect() durchgeführt.
23:54:21 INFO [rubin.tuning] BLT 'catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': X=(389983, 80), target=(389983,) (unique=[0, 1]), subsample=100%, cv=5, objective=grouped_outcome_regression
23:54:21 INFO [rubin.tuning] BLT 'catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': Starte 100 Trials (parallel=20, 5-Fold, n_jobs=4).
