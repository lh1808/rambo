14:17:36 INFO [rubin.feature_selection] Feature-Selection 'causal_forest': Top-39 = 39 / 530 Features (Budget: 39 pro Methode).
14:17:36 INFO [rubin.feature_selection] Feature-Selection: 63 / 530 Features behalten (max_features=77, 2 Methoden × 39 pro Methode).
14:17:36 INFO [rubin.analysis] Feature-Selektion gesamt: 745 → 63 Features (Korrelation: −215, Importance: −467)
14:17:36 INFO [rubin.analysis] Feature-Selektion: 745 → 63 Features (-682 entfernt)
14:17:36 INFO [rubin.categorical] Kategorische Spalten erkannt: 24 von 63 Features (['GESELLSCHAFT_MM', 'PLZ1', 'PLZ_2A']... (+21)). Patche LGBM Methoden (fit/predict/predict_proba/score) für EconML-Kompatibilität.
14:17:36 INFO [rubin.analysis] Starte Tuning: X=(416805, 63), Y=(416805,) (unique=[0, 1]), T=(416805,) (unique=[0, 1])
14:17:36 INFO [rubin.tuning] tune_all gestartet: models=['DRLearner', 'CausalForestDML'], X=(416805, 63), Y=(416805,) (unique=[0, 1]), T=(416805,) (unique=[0, 1]), cv_splits=5, n_trials=100, parallel_trials=20, parallel_level=3, cores_per_fit=4, CPU=[effective=80]
14:17:36 INFO [rubin.tuning] Tuning-Plan: 3 Tasks für 2 Modelle.
14:17:36 INFO [rubin.tuning] BLT 'lgbm__outcome__classifier__all__with_t__y': X=(416805, 64), target=(416805,) (unique=[0, 1]), subsample=80%, cv=5, objective=outcome
14:17:36 INFO [rubin.tuning] BLT 'lgbm__outcome__classifier__all__with_t__y': Starte 100 Trials (parallel=20, 5-Fold, n_jobs=4).
14:24:10 INFO [rubin.tuning] BLT 'lgbm__outcome__classifier__all__with_t__y': 100/100 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned, parallel=20).
14:24:11 INFO [rubin.tuning] BLT 'lgbm__outcome__classifier__all__with_t__y': Study freigegeben, gc.collect() + malloc_trim durchgeführt.
14:24:11 INFO [rubin.tuning] BLT 'lgbm__outcome__classifier__all__no_t__y': X=(416805, 63), target=(416805,) (unique=[0, 1]), subsample=80%, cv=5, objective=outcome
14:24:11 INFO [rubin.tuning] BLT 'lgbm__outcome__classifier__all__no_t__y': Starte 100 Trials (parallel=20, 5-Fold, n_jobs=4).
14:30:54 INFO [rubin.tuning] BLT 'lgbm__outcome__classifier__all__no_t__y': 100/100 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned, parallel=20).
14:30:55 INFO [rubin.tuning] BLT 'lgbm__outcome__classifier__all__no_t__y': Study freigegeben, gc.collect() + malloc_trim durchgeführt.
14:30:55 INFO [rubin.tuning] BLT 'lgbm__propensity__classifier__all__no_t__t': X=(416805, 63), target=(416805,) (unique=[0, 1]), subsample=80%, cv=5, objective=propensity
14:30:55 INFO [rubin.tuning] BLT 'lgbm__propensity__classifier__all__no_t__t': Starte 20 Trials (parallel=20, 5-Fold, n_jobs=4).
14:32:40 INFO [rubin.tuning] BLT 'lgbm__propensity__classifier__all__no_t__t': 20/20 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned, parallel=20).
14:32:40 WARNING [rubin.tuning] RCT-Warnung: Propensity-Modell erreicht neg_log_loss=0.0000 (bei randomisiertem Treatment sollte das Modell nicht besser als Zufall sein). Prüfe ob Treatment tatsächlich randomisiert ist oder ob Post-Treatment-Variablen im Datensatz sind.
14:32:41 INFO [rubin.tuning] BLT 'lgbm__propensity__classifier__all__no_t__t': Study freigegeben, gc.collect() + malloc_trim durchgeführt.
14:32:41 INFO [rubin.analysis] Base-Learner-Tuning: 3 Tasks abgeschlossen.
14:32:41 INFO [rubin.analysis]   lgbm__outcome__classifier__all__no_t__y  best=0  skill=1.0000
14:32:41 INFO [rubin.analysis]   lgbm__outcome__classifier__all__with_t__y best=0  skill=1.0000
14:32:41 INFO [rubin.analysis]   lgbm__propensity__classifier__all__no_t__t best=0  skill=1.0000
14:32:41 WARNING [rubin.analysis] RCT-Diagnose: Propensity-Skill = 1.0000 > 0.01. Das Propensity-Modell kann Treatment besser als Zufall vorhersagen — die Daten sind möglicherweise NICHT randomisiert. Prüfe die Treatment-Zuweisung. Training verwendet trotzdem konstante Propensity (DummyClassifier), wie für RCT konfiguriert.
14:42:35 INFO [rubin.tuning] FMT 'DRLearner': Nuisance gecacht für 5 äußere Folds (je 5 innere × model_propensity + model_regression). Trials fitten nur noch model_final via refit_final(). model_propensity = DummyClassifier (RCT: konstante Propensity).
14:51:38 INFO [rubin.tuning] FMT 'DRLearner': RScorer (unabhängige Nuisance, 2-Fold T×Y) erstellt + gecacht (5 Val + 5 Train).
14:51:38 INFO [rubin.tuning] FMT 'DRLearner': Starte 50 Trials (cache_values, R-Score, 5-Fold, parallel_jobs=-1).
