20:17:50 INFO [rubin.tuning] GRF: 5 Fold-Splits vorberechnet (CATBOOST Base-Learner für RScorer). model_t = DummyClassifier (RCT: konstante Propensity).
20:18:55 INFO [rubin.tuning] CFT 'CausalForest': RScorer (unabhängige Nuisance, 2-Fold T×Y) pro Fold erstellt (5 Scorers).
20:18:55 INFO [rubin.tuning] CFT 'CausalForest': Starte 10 Trials (CausalForestAdapter + RScorer, R-Score, 5-Fold, parallel_jobs=-1).
20:18:55 INFO [rubin.tuning] CFT 'CausalForest': 10/10 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned, parallel=1).
20:18:56 INFO [rubin.tuning] CFT 'CausalForest': Study + Cache freigegeben, gc.collect() durchgeführt.
20:18:56 INFO [rubin.analysis] CFT 'CausalForest': R-Score → -1e+12 (10 Trials abgeschlossen).
20:18:56 INFO [rubin.analysis]   CausalForest: Starte 5-Fold Cross-Prediction (Seed=42).
20:18:56 INFO [rubin.training] CausalForest: Erzwinge sequentielle Folds (GRF nutzt intern joblib-Prozesse, die in Threads zu Deadlocks führen).
20:18:56 INFO [rubin.training] CausalForest: 5 Folds sequentiell.
