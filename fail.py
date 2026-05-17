15:56:52 INFO [rubin.tuning] FMT 'NonParamDML': Starte 50 Trials (cache_values, R-Score, 5-Fold, parallel_jobs=-1).
16:25:13 INFO [rubin.tuning] FMT 'NonParamDML': 49/50 Trials abgeschlossen (0 fehlgeschlagen, 1 gepruned, parallel=1).
16:25:13 INFO [rubin.tuning] FMT 'NonParamDML': Study + Cache + Scorers freigegeben, gc.collect() durchgeführt.
16:32:14 INFO [rubin.tuning] FMT 'DRLearner': Nuisance gecacht für 5 äußere Folds (je 5 innere × model_propensity + model_regression). Trials fitten nur noch model_final via refit_final(). model_propensity = DummyClassifier (RCT: konstante Propensity).
16:32:14 INFO [rubin.tuning] FMT 'DRLearner': Gecachte RScorer wiederverwendet (5 Val + 5 Train).
16:32:14 INFO [rubin.tuning] FMT 'DRLearner': Starte 50 Trials (cache_values, R-Score, 5-Fold, parallel_jobs=-1).
17:06:13 INFO [rubin.tuning] FMT 'DRLearner': 41/50 Trials abgeschlossen (0 fehlgeschlagen, 9 gepruned, parallel=1).
17:06:13 INFO [rubin.tuning] FMT 'DRLearner': Study + Cache + Scorers freigegeben, gc.collect() durchgeführt.
17:06:13 INFO [rubin.analysis] Final-Model-Tuning: 2 Modelle abgeschlossen.
17:06:13 INFO [rubin.analysis] FMT 'NonParamDML': OOF-R-Score → -1.79661e-08
17:06:13 INFO [rubin.analysis] FMT 'DRLearner': OOF-R-Score → -3.73101e-06
17:06:13 INFO [rubin.analysis]   CausalForestDML: RCT-Modus → konstante Propensity P(T|X) = mean(T) (DummyClassifier).
17:11:17 INFO [rubin.tuning] CFT 'CausalForestDML': Nuisance gecacht für 5 Folds. Trials fitten nur noch Forest via refit_final().
