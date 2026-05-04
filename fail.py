Analyse fehlgeschlagen: Fehlgeschlagen (Exit 1)

Details:
13:13:57 INFO [rubin.analysis] Arbeitsverzeichnis: /mnt/rubin/runs
13:13:57 INFO [rubin.analysis] MLflow-Experiment 'rubin_GRP_PBV' (identisch mit DataPrep).
13:13:57 INFO [rubin.analysis] Run-Name-Suffix 'heller-löwe' aus DataPrep übernommen.
13:13:58 INFO [rubin.analysis] DataPrep-Config nach MLflow geloggt: /mnt/rubin/runs/data/dataprep_config.yml
13:13:58 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
13:13:58 INFO [rubin.analysis] rubin Pipeline Start
13:13:58 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
13:13:58 INFO [rubin.analysis] Config: 3 Modelle (NonParamDML, DRLearner, CausalForestDML), CATBOOST, 5-Fold CV, Parallel-Level 3
13:13:58 INFO [rubin.analysis] Nuisance Cross-Fitting (DML + DR): 5 interne Folds
13:13:58 INFO [rubin.analysis] Aktiv: BL-Tuning (50 Trials) | FMT (50 Trials) | CFT (50 Trials) | Ensemble
13:13:58 INFO [rubin.analysis] Validierungsmodus: Cross-Validation (5 Folds, Seed=42, Tuning-Seed=18)
13:13:58 INFO [rubin.analysis] Historischer Score: 3 NaN-Werte durch 0 ersetzt.
13:13:58 INFO [rubin.analysis] dtypes.json auto-erkannt (runs/data/dtypes.json): 254 Spalten-Dtypes wiederhergestellt (254 kategorial).
13:13:59 INFO [rubin.analysis] Memory-Reduktion: 240.7 MB → 240.7 MB (0% gespart).
13:13:59 INFO [rubin.analysis] Daten geladen: X=(299988, 745), T=(299988,) (unique=[0, 1]), Y=(299988,) (unique=[0, 1]), S=(299988,)
13:13:59 INFO [rubin.categorical] Kategorische Spalten erkannt: 254 von 745 Features (['D_FRAU', 'GESELLSCHAFT_MM', 'BEAMTER_FLG']... (+251)). Patche BOTH Methoden (fit/predict/predict_proba/score) für EconML-Kompatibilität.
13:14:00 INFO [rubin.feature_selection] Feature-Selektion: 2 Methoden sequentiell (alle Kerne pro Methode).
13:14:00 INFO [rubin.categorical] CatBoost categorical patch (fit): 254 cat_features injiziert (DataFrame).
13:14:12 INFO [rubin.feature_selection] CausalForest FS: X=(299988, 745) (dtypes: 491 numeric, 254 category), T=(299988,) (unique=2), Y=(299988,), n_jobs=-1, in_thread=False
13:14:14 INFO [rubin.feature_selection] CausalForest FS: Subsampling 299988 → 99999 Zeilen (stratifiziert nach T).
13:14:14 INFO [rubin.feature_selection] CausalForest FS: fit(99999×745, T unique=2, n_estimators=100, n_jobs=-1)...
13:27:36 INFO [rubin.feature_selection] Korrelationsfilter (|r| > 0.90, importance-gesteuert): 87 Features entfernt, 404 verbleiben.
13:27:37 INFO [rubin.analysis] Importance-Umverteilung: 87 entfernte Features → Importance auf Partner übertragen.
13:27:37 INFO [rubin.feature_selection] Feature-Selection 'catboost_importance': Top-30 = 30 / 658 Features (Budget: 30 pro Methode).
13:27:37 INFO [rubin.feature_selection] Feature-Selection 'causal_forest': Top-30 = 30 / 658 Features (Budget: 30 pro Methode).
13:27:37 INFO [rubin.feature_selection] Feature-Selection: 52 / 658 Features behalten (max_features=60, 2 Methoden × 30 pro Methode).
13:27:37 INFO [rubin.analysis] Feature-Selektion gesamt: 745 → 52 Features (Korrelation: −87, Importance: −606)
13:27:37 INFO [rubin.analysis] Feature-Selektion: 745 → 52 Features (-693 entfernt)
13:27:37 INFO [rubin.categorical] Kategorische Spalten erkannt: 11 von 52 Features (['PLZ1', 'PLZ_3A', 'RS_ONL_ANGEBOT_06']... (+8)). Patche CATBOOST Methoden (fit/predict/predict_proba/score) für EconML-Kompatibilität.
13:27:37 INFO [rubin.analysis] Starte Tuning: X=(299988, 52), Y=(299988,) (unique=[0, 1]), T=(299988,) (unique=[0, 1])
13:27:37 INFO [rubin.tuning] tune_all gestartet: models=['NonParamDML', 'DRLearner', 'CausalForestDML'], X=(299988, 52), Y=(299988,) (unique=[0, 1]), T=(299988,) (unique=[0, 1]), cv_splits=5, n_trials=50, parallel_trials=20, parallel_level=3, cores_per_fit=4, CPU=[effective=80]
13:27:37 INFO [rubin.tuning] Tuning-Plan: 3 Tasks für 3 Modelle.
13:27:37 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__no_t__y': X=(299988, 52), target=(299988,) (unique=[0, 1]), subsample=80%, cv=5, objective=outcome
13:27:37 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__no_t__y': Starte 50 Trials (parallel=20, cv=5).
13:27:37 INFO [rubin.categorical] CatBoost categorical patch (fit): 11 cat_features injiziert (DataFrame).
13:30:20 INFO [rubin.tuning] BLT 'catboost__outcome__classifier__all__no_t__y': 50/50 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned, parallel=20).
Traceback (most recent call last):
  File "/mnt/rubin/run_analysis.py", line 132, in <module>
    main()
  File "/mnt/rubin/run_analysis.py", line 128, in main
    pipe.run(export_bundle=args.export_bundle, bundle_dir=args.bundle_dir, bundle_id=args.bundle_id)
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 3071, in run
    tuned_params_by_model = self._run_tuning(cfg, X, T, Y, mlflow, _progress_cb=_progress)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 485, in _run_tuning
    tuned_params_by_model = tuner.tune_all(cfg.models.models_to_train, X=X, Y=Y, T=T)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1212, in tune_all
    best = self._tune_task(task, X=X, Y=Y, T=T, shared_params=tuned_by_task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1139, in _tune_task
    gc.collect()
    ^^
NameError: name 'gc' is not defined. Did you forget to import 'gc'?
