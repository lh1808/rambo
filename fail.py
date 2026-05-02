20:43:40 INFO [rubin.categorical] CatBoost categorical patch (fit): 254 cat_features injiziert (DataFrame).
20:43:40 WARNING [rubin.feature_selection] Feature-Importance 'catboost_importance' fehlgeschlagen.
Traceback (most recent call last):
  File "/mnt/rubin/rubin/feature_selection.py", line 398, in _run_method
    return method, _catboost_gain_importance(X, Y, seed, n_jobs=method_n_jobs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/feature_selection.py", line 177, in _catboost_gain_importance
    model.fit(X, Y)
  File "/mnt/rubin/rubin/utils/categorical_patch.py", line 162, in _wrapped_fit
    return original_fit(self, X_conv, y, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 6178, in fit
    return self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2701, in _fit
    train_params = self._prepare_train_params(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2627, in _prepare_train_params
    _check_train_params(params)
  File "_catboost.pyx", line 7029, in _catboost._check_train_params
  File "_catboost.pyx", line 7051, in _catboost._check_train_params
_catboost.CatBoostError: catboost/private/libs/options/catboost_options.cpp:638: Error: rsm on GPU is supported for pairwise modes only
20:43:40 INFO [rubin.feature_selection] CausalForest FS: X=(299988, 745) (dtypes: 491 numeric, 254 category), T=(299988,) (unique=2), Y=(299988,), n_jobs=-1, in_thread=False
20:43:42 INFO [rubin.feature_selection] CausalForest FS: Subsampling 299988 → 99999 Zeilen (stratifiziert nach T).
20:43:42 INFO [rubin.feature_selection] CausalForest FS: fit(99999×745, T unique=2, n_estimators=100, n_jobs=-1)...
