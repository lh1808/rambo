17:21:44 WARNING [rubin.analysis] CFT 'CausalForestDML' fehlgeschlagen.
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/rubin/pipelines/analysis_pipeline.py", line 1028, in _run_training
    result = _cft_tuner.tune(
             ^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/causal_forest.py", line 231, in tune
    model_y=build_base_learner(base_type, nuisance_params_y or {}, seed=seed, task="classifier", parallel_jobs=-1),
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/common.py", line 453, in build_base_learner
    return _build_catboost_classifier(params, seed, parallel_jobs) if task == "classifier" else _build_catboost_regressor(params, seed, parallel_jobs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/common.py", line 387, in _build_catboost_classifier
    return CatBoostClassifier(**fixed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: CatBoostClassifier.__init__() got an unexpected keyword argument '_learner_type'
17:21:44 INFO [rubin.analysis] CausalForestDML model_final (Forest) effektive Params: {} (cft_tuned=nein, forest_fixed=True)
17:21:44 INFO [rubin.analysis]   CausalForestDML: Starte 5-Fold Cross-Prediction (Seed=42).
17:21:44 INFO [rubin.training] CausalForestDML: Erzwinge sequentielle Folds (GRF nutzt intern joblib-Prozesse, die in Threads zu Deadlocks führen).
17:21:44 INFO [rubin.training] CausalForestDML: 5 Folds sequentiell.
17:23:25 INFO [rubin.analysis]   CausalForestDML: Training + Cross-Predictions in 100.8s
17:23:25 INFO [rubin.analysis] Predictions_CausalForestDML: CATE min=-0.160749, median=0.0213471, max=0.205847, std=0.0400149, unique=37626/39246, non-zero=39246/39246
