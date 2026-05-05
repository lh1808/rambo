04:25:15 INFO [rubin.tuning] GRF R-Loss: Nuisance-Residuen vorberechnet (1 Folds, CATBOOST Base-Learner).
04:25:15 INFO [rubin.tuning] CFT 'CausalForest': Starte 50 Trials (parallel=1, cv=1, Seed=18).
04:26:31 INFO [rubin.tuning] CFT 'CausalForest': 50/50 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned, parallel=1).
04:26:31 INFO [rubin.tuning] CFT 'CausalForest': Study freigegeben, gc.collect() durchgeführt.
04:26:31 WARNING [rubin.analysis] CFT 'CausalForest' fehlgeschlagen.
Traceback (most recent call last):
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 887, in _run_training
    result = tune_causal_forest(
             ^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1860, in tune_causal_forest
    return {"best_params": best, "best_score": best_score, "n_trials_completed": n_complete}
                                                                                 ^^^^^^^^^^
NameError: name 'n_complete' is not defined
