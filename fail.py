10:00:52 INFO [rubin.analysis]   CausalForestDML: RCT-Modus → konstante Propensity P(T|X) = mean(T) (DummyClassifier).
10:07:15 INFO [rubin.tuning] CFT 'CausalForestDML': Nuisance gecacht für 5 äußere Folds (je 5 innere × model_y + model_t). Trials fitten nur noch den CausalForest via refit_final(). model_t = DummyClassifier (RCT: konstante Propensity).
10:07:15 INFO [rubin.tuning] CFT 'CausalForestDML': Starte 100 Trials (cache_values, 5-Fold, parallel_jobs=-1).
10:07:16 INFO [rubin.tuning] CFT 'CausalForestDML': 100/100 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned, parallel=1).
10:07:16 INFO [rubin.tuning] CFT 'CausalForestDML': Study + Cache freigegeben, gc.collect() durchgeführt.
10:07:16 INFO [rubin.analysis]   CausalForestDML: Starte 5-Fold Cross-Prediction (Seed=42).
10:07:17 INFO [rubin.training] CausalForestDML: Erzwinge sequentielle Folds (GRF nutzt intern joblib-Prozesse, die in Threads zu Deadlocks führen).
10:07:17 INFO [rubin.training] CausalForestDML: 5 Folds sequentiell.
10:26:07 INFO [rubin.analysis]   CausalForestDML: Training + Cross-Predictions in 1515.5s
10:26:09 WARNING [rubin.analysis] WARNUNG: Predictions_CausalForestDML hat nur 5 distinkte Werte bei 5 Folds (Range=2.37e-05, Mean=9.81e-04). Das CATE-Modell ist wahrscheinlich zu einem Intercept kollabiert. Empfehlungen: (1) final_model_tuning.enabled=true aktivieren, (2) Prüfen ob base_fixed_params zu restriktiv sind (min_child_samples, num_leaves, max_depth), (3) Mehr Features oder Feature-Engineering.
