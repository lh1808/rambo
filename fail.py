[rubin] Step 1/11: Daten laden & Preprocessing
[rubin] dtypes.json auto-erkannt: 254 Spalten-Dtypes wiederhergestellt (254 kategorial).
[rubin] Step 2/11: Feature-Selektion
[rubin] Step 3/11: Base-Learner-Tuning
[rubin] Kategorische Features: 42 von 159 Spalten → CATBOOST erhält cat_feature-Indizes.
14:57:36 INFO [rubin.analysis] Arbeitsverzeichnis: /mnt/rubin/runs
14:57:37 INFO [rubin.analysis] MLflow-Experiment 'rubin_GRP_PBV' (identisch mit DataPrep).
14:57:37 INFO [rubin.analysis] Run-Name-Suffix 'roter-kranich' aus DataPrep übernommen.
14:57:38 INFO [rubin.analysis] DataPrep-Config nach MLflow geloggt: /mnt/rubin/runs/data/dataprep_config.yml
14:57:38 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
14:57:38 INFO [rubin.analysis] rubin Pipeline Start
14:57:38 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
14:57:38 INFO [rubin.analysis] Config: 8 Modelle (NonParamDML, DRLearner, SLearner, TLearner, XLearner, ParamDML, CausalForestDML, CausalForest), CATBOOST, 5-Fold CV, Parallel-Level 3
14:57:38 INFO [rubin.analysis] Nuisance Cross-Fitting (DML + DR): 5 interne Folds
14:57:38 INFO [rubin.analysis] Aktiv: BL-Tuning (100 Trials) | FMT (100 Trials) | CausalForest-Tuning | Ensemble
14:57:38 INFO [rubin.analysis] Validierungsmodus: Cross-Validation (5 Folds, Seed=42, Tuning-Seed=18)
14:57:39 INFO [rubin.analysis] Historischer Score: 3 NaN-Werte durch 0 ersetzt.
14:57:41 INFO [rubin.analysis] dtypes.json auto-erkannt (runs/data/dtypes.json): 254 Spalten-Dtypes wiederhergestellt (254 kategorial).
14:57:42 INFO [rubin.analysis] Memory-Reduktion: 240.7 MB → 240.7 MB (0% gespart).
14:57:42 INFO [rubin.analysis] Daten geladen: X=(299988, 745), T=(299988,) (unique=[0, 1]), Y=(299988,) (unique=[0, 1]), S=(299988,)
14:57:43 INFO [rubin.categorical] Kategorische Spalten erkannt: 254 von 745 Features (['D_FRAU', 'GESELLSCHAFT_MM', 'BEAMTER_FLG']... (+251)). Patche BOTH .fit()-Methoden für EconML-Kompatibilität.
14:57:45 INFO [rubin.feature_selection] Feature-Selektion: 2 Methoden sequentiell (alle Kerne pro Methode).
14:57:45 INFO [rubin.categorical] CatBoost categorical patch (fit): 254 cat_features injiziert (DataFrame).
14:59:23 INFO [rubin.feature_selection] CausalForest FS: X=(299988, 745) (dtypes: 491 numeric, 254 category), T=(299988,) (unique=2), Y=(299988,), n_jobs=-1, in_thread=False
14:59:25 INFO [rubin.feature_selection] CausalForest FS: Subsampling 299988 → 99999 Zeilen (stratifiziert nach T).
14:59:25 INFO [rubin.feature_selection] CausalForest FS: fit(99999×745, T unique=2, n_estimators=100, n_jobs=-1)...
15:16:30 INFO [rubin.feature_selection] Korrelationsfilter (|r| > 0.90, importance-gesteuert): 88 Features entfernt, 403 verbleiben.
15:16:31 INFO [rubin.analysis] Importance-Umverteilung: 88 entfernte Features → Importance auf Partner übertragen.
15:16:31 INFO [rubin.feature_selection] Feature-Selection 'catboost_importance': Top-15% = 99 / 657 Features.
15:16:31 INFO [rubin.feature_selection] Feature-Selection 'causal_forest': Top-15% = 99 / 657 Features.
15:16:31 INFO [rubin.feature_selection] Feature-Selection Union: 159 / 657 Features behalten, 498 entfernt.
15:16:31 INFO [rubin.analysis] Feature-Selektion gesamt: 745 → 159 Features (Korrelation: −88, Importance: −498)
15:16:31 INFO [rubin.analysis] Feature-Selektion: 745 → 159 Features (-586 entfernt)
15:16:31 INFO [rubin.categorical] Kategorische Spalten erkannt: 42 von 159 Features (['D_FRAU', 'PLZ1', 'PLZ_2A']... (+39)). Patche CATBOOST .fit()-Methoden für EconML-Kompatibilität.
15:16:31 INFO [rubin.analysis] Starte Tuning: X=(299988, 159), Y=(299988,) (unique=[0, 1]), T=(299988,) (unique=[0, 1])
15:16:31 INFO [rubin.tuning] tune_all gestartet: models=['NonParamDML', 'DRLearner', 'SLearner', 'TLearner', 'XLearner', 'ParamDML', 'CausalForestDML', 'CausalForest'], X=(299988, 159), Y=(299988,) (unique=[0, 1]), T=(299988,) (unique=[0, 1]), cv_splits=5, n_trials=100, parallel_trials=10, parallel_level=3, cores_per_fit=4, CPU=[effective=40]
15:16:31 INFO [rubin.tuning] Tuning-Plan: 7 Tasks für 8 Modelle.
15:16:33 INFO [rubin.tuning] Tuning-Task 'catboost__outcome__classifier__all__no_t__y': X_input=299988 rows, indices=299988, X_task=(299988, 159), target=(299988,) (unique=[0, 1]), train_subsample=80%, T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome
15:16:35 INFO [rubin.categorical] CatBoost categorical patch (fit): 42/159 Spalten float→int konvertiert.
15:21:05 INFO [rubin.categorical] CatBoost categorical patch (predict): 42 Spalten float→int konvertiert.
