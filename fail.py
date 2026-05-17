19:09:25 INFO [rubin.analysis]   NonParamDML: Starte 5-Fold Cross-Prediction (Seed=42).
19:09:26 INFO [rubin.training] NonParamDML: 5 Folds parallel (n_jobs=5, threads) auf 80 Kernen.
19:10:15 INFO [rubin.analysis]   NonParamDML: Training + Cross-Predictions in 50.2s
19:10:17 WARNING [rubin.analysis] WARNUNG: Predictions_NonParamDML hat nur 5 distinkte Werte bei 5 Folds (Range=1.71e-05, Mean=9.55e-04). Das CATE-Modell ist wahrscheinlich zu einem Intercept kollabiert. Empfehlungen: (1) final_model_tuning.enabled=true aktivieren, (2) Prüfen ob base_fixed_params zu restriktiv sind (min_child_samples, num_leaves, max_depth), (3) Mehr Features oder Feature-Engineering.
19:10:17 INFO [rubin.analysis]   DRLearner: RCT-Modus → konstante Propensity P(T|X) = mean(T) (DummyClassifier).
19:10:17 INFO [rubin.analysis] DRLearner model_final effektive Params: {'n_estimators': 164, 'num_leaves': 23, 'max_depth': 5, 'min_child_samples': 317, 'min_child_weight': 43.630737646789385, 'colsample_bytree': 0.2711590190849684, 'reg_alpha': 45.938714555585065, 'reg_lambda': 94.17625924384025, 'path_smooth': 11.891856273457003} (explicit_tuned=ja, fmt_fixed=False)
19:10:17 INFO [rubin.analysis]   DRLearner: Starte 5-Fold Cross-Prediction (Seed=42).
19:10:18 INFO [rubin.training] DRLearner: 5 Folds parallel (n_jobs=5, threads) auf 80 Kernen.
19:11:26 INFO [rubin.analysis]   DRLearner: Training + Cross-Predictions in 68.6s
19:11:27 WARNING [rubin.analysis] HINWEIS: Predictions_DRLearner hat nur 19 distinkte Werte bei 416805 Samples. Das Modell differenziert wenig zwischen Individuen.
19:11:28 INFO [rubin.analysis] Ensemble: Fold-Aligned Predictions aus 3 Einzelmodellen gemittelt (leakage-frei).
19:11:29 INFO [rubin.analysis] Ensemble erstellt (EconML EnsembleCateEstimator): 3 Modelle gleichgewichtet (CausalForestDML, NonParamDML, DRLearner).
19:14:05 INFO [rubin.analysis] DRTester Nuisance einmalig gefittet (BT, cv=5, n_est≤100). Wird für alle Modelle wiederverwendet.
