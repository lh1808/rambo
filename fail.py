Analyse fehlgeschlagen: Fehlgeschlagen (Exit -11)

Details:
08:50:40 INFO [rubin.analysis] Arbeitsverzeichnis: /home/ubuntu/da-hf1-rubin/runs
08:50:40 INFO [rubin.analysis] MLflow-Experiment 'rubin_h24_voucher' (identisch mit DataPrep).
08:50:40 INFO [rubin.analysis] Run-Name-Suffix 'blauer-rabe' aus DataPrep übernommen.
08:50:40 INFO [rubin.analysis] DataPrep-Config nach MLflow geloggt: /home/ubuntu/da-hf1-rubin/runs/data/dataprep_config.yml
08:50:40 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
08:50:40 INFO [rubin.analysis] rubin Pipeline Start
08:50:40 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
08:50:40 INFO [rubin.analysis] Config: 5 Modelle (NonParamDML, DRLearner, XLearner, CausalForestDML, CausalForest), BOTH, 5-Fold CV, Parallel-Level 3
08:50:40 INFO [rubin.analysis] Nuisance Cross-Fitting (DML + DR): 5 interne Folds, mc_iters=3
08:50:40 INFO [rubin.analysis] Aktiv: BL-Tuning (160 Trials) | FMT (100 Trials) | CFT (100 Trials) | Ensemble
08:50:40 INFO [rubin.analysis] Validierungsmodus: Cross-Validation (5 Folds, Seed=42, Tuning-Seed=18)
08:50:40 INFO [rubin.analysis] dtypes.json auto-erkannt (runs/data/dtypes.json): 5 Spalten-Dtypes wiederhergestellt (5 kategorial).
08:50:40 INFO [rubin.analysis] Memory-Reduktion: 1.3 MB → 1.3 MB (0% gespart).
08:50:40 INFO [rubin.analysis] Daten geladen: X=(39246, 14), T=(39246,) (unique=[0, 1, 2]), Y=(39246,) (unique=[0, 1]), S=None
08:50:41 WARNING [matplotlib.font_manager] findfont: Failed to find font weight 600, now using 700.
08:50:41 INFO [rubin.categorical] Kategorische Spalten erkannt: 5 von 14 Features (['FAMILIENSTAND', 'BERUFSGRUPPE', 'PRODUKTLINIE', 'SELBSTBETEILIGUNG', 'ZAHLUNGSWEISE']). Patche BOTH Methoden (fit/predict/predict_proba/score) für EconML-Kompatibilität.
08:50:41 INFO [rubin.categorical] Kategorische Spalten erkannt: 5 von 14 Features (['FAMILIENSTAND', 'BERUFSGRUPPE', 'PRODUKTLINIE', 'SELBSTBETEILIGUNG', 'ZAHLUNGSWEISE']). Patche BOTH Methoden (fit/predict/predict_proba/score) für EconML-Kompatibilität.
08:50:41 INFO [rubin.analysis] Starte Tuning: X=(39246, 14), Y=(39246,) (unique=[0, 1]), T=(39246,) (unique=[0, 1, 2])
08:50:41 INFO [rubin.tuning] tune_all gestartet: models=['NonParamDML', 'DRLearner', 'XLearner', 'CausalForestDML', 'CausalForest'], X=(39246, 14), Y=(39246,) (unique=[0, 1]), T=(39246,) (unique=[0, 1, 2]), cv_splits=5, n_trials=160, parallel_trials=20, parallel_level=3, cores_per_fit=4, CPU=[effective=80]
08:50:41 INFO [rubin.tuning] Tuning-Plan: 6 Tasks für 5 Modelle.
08:50:41 INFO [rubin.tuning] BLT 'both__outcome__classifier__all__no_t__y': X=(39246, 14), target=(39246,) (unique=[0, 1]), subsample=80%, cv=5, objective=outcome
08:50:41 INFO [rubin.tuning] BLT 'both__outcome__classifier__all__no_t__y': Starte 320 Trials (parallel=20, 5-Fold, n_jobs=4).
08:50:41 INFO [rubin.categorical] CatBoost categorical patch (fit): 5 cat_features injiziert (DataFrame).
