18:02:54 INFO [rubin.analysis] RCT-Diagnose bestätigt: Propensity-Skill = -0.0000 ≈ 0. Training verwendet konstante Propensity P(T|X) = mean(T) = 0.667.
18:03:12 INFO [rubin.categorical] CatBoost categorical patch (predict): 24 Spalten float→int konvertiert.
18:12:08 INFO [rubin.tuning] FMT 'NonParamDML': Nuisance gecacht für 5 äußere Folds (je 5 innere × model_y + model_t). Trials fitten nur noch model_final via refit_final(). model_t = DummyClassifier (RCT: konstante Propensity).
18:12:08 INFO [rubin.tuning] FMT 'NonParamDML': Scorer='qini' — Pruning deaktiviert (QiniScorer benötigt alle Folds).
18:12:08 INFO [rubin.tuning] FMT 'NonParamDML': Starte 50 Trials (cache_values, OOF-Qini, 5-Fold).
18:57:31 INFO [rubin.tuning] FMT 'NonParamDML': 50/50 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned, parallel=1).
18:57:32 INFO [rubin.tuning] FMT 'NonParamDML': Study + Cache freigegeben, gc.collect() durchgeführt.
19:11:24 INFO [rubin.tuning] FMT 'DRLearner': Nuisance gecacht für 5 äußere Folds (je 5 innere × model_propensity + model_regression). Trials fitten nur noch model_final via refit_final(). model_propensity = DummyClassifier (RCT: konstante Propensity).
19:11:24 INFO [rubin.tuning] FMT 'DRLearner': Scorer='qini' — Pruning deaktiviert (QiniScorer benötigt alle Folds).
19:11:24 INFO [rubin.tuning] FMT 'DRLearner': Starte 50 Trials (cache_values, OOF-Qini, 5-Fold).
19:53:12 INFO [rubin.tuning] FMT 'DRLearner': 50/50 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned, parallel=1).
19:53:13 INFO [rubin.tuning] FMT 'DRLearner': Study + Cache freigegeben, gc.collect() durchgeführt.
19:53:13 INFO [rubin.analysis] Final-Model-Tuning: 2 Modelle abgeschlossen.
19:53:13 INFO [rubin.analysis] FMT 'NonParamDML': OOF-R-Score → 0.000114532
19:53:13 INFO [rubin.analysis] FMT 'DRLearner': OOF-R-Score → 0.000131113
