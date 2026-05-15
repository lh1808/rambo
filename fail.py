11:34:27 INFO [rubin.tuning] CFT 'CausalForestDML': Starte 100 Trials (cache_values, 5-Fold, parallel_jobs=-1).
19:29:32 INFO [rubin.tuning] CFT 'CausalForestDML': 100/100 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned, parallel=1).
19:29:33 INFO [rubin.tuning] CFT 'CausalForestDML': Study + Cache freigegeben, gc.collect() durchgeführt.
19:29:33 INFO [rubin.analysis]   CausalForestDML: Starte 5-Fold Cross-Prediction (Seed=42).
19:29:33 INFO [rubin.training] CausalForestDML: Erzwinge sequentielle Folds (GRF nutzt intern joblib-Prozesse, die in Threads zu Deadlocks führen).
19:29:33 INFO [rubin.training] CausalForestDML: 5 Folds sequentiell.
19:47:04 INFO [rubin.analysis]   CausalForestDML: Training + Cross-Predictions in 29894.3s
19:47:06 WARNING [rubin.analysis] WARNUNG: Predictions_CausalForestDML hat nur 5 distinkte Werte bei 5 Folds (Range=4.10e-05, Mean=9.64e-04). Das CATE-Modell ist wahrscheinlich zu einem Intercept kollabiert. Empfehlungen: (1) final_model_tuning.enabled=true aktivieren, (2) Prüfen ob base_fixed_params zu restriktiv sind (min_child_samples, num_leaves, max_depth), (3) Mehr Features oder Feature-Engineering.
19:48:11 INFO [rubin.tuning] GRF R-Loss: Nuisance-Residuen vorberechnet (5 Folds, CATBOOST Base-Learner). model_t = DummyClassifier (RCT: konstante Propensity).
19:48:11 INFO [rubin.tuning] CFT 'CausalForest': Starte 100 Trials (cache_values, 5-Fold, parallel_jobs=-1).
