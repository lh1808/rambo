[rubin] Step 5/11: CausalForest-Tuning
    return original_fit(self, X_conv, y, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5547, in fit
    self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline, use_best_model,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2701, in _fit
    train_params = self._prepare_train_params(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2581, in _prepare_train_params
    train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, graph,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 1540, in _build_train_pool
    train_pool = Pool(X, y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs, graph=graph, weight=sample_weight, group_id=group_id,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 863, in __init__
    self._init(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 1518, in _init
    self._init_pool(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
  File "_catboost.pyx", line 4695, in _catboost._PoolBase._init_pool
  File "_catboost.pyx", line 4753, in _catboost._PoolBase._init_pool
  File "_catboost.pyx", line 4541, in _catboost._PoolBase._init_features_order_layout_pool
  File "_catboost.pyx", line 3170, in _catboost._set_features_order_data_pd_data_frame
  File "_catboost.pyx", line 3070, in _catboost._set_features_order_data_frame_generic_categorical_column
  File "_catboost.pyx", line 2694, in _catboost.get_cat_factor_bytes_representation
_catboost.CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=0]=0.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.
[W 2026-05-01 16:03:30,507] Trial 96 failed with value None.
[W 2026-05-01 16:03:31,143] Trial 94 failed with parameters: {'iterations': 163, 'learning_rate': 0.013097731052500623, 'depth': 6, 'l2_leaf_reg': 40.10121725091117, 'random_strength': 1.9153225680388837, 'subsample': 0.7306952011663961, 'rsm': 0.4460911688516921, 'min_data_in_leaf': 466, 'model_size_reg': 0.9143142237553132, 'leaf_estimation_iterations': 2} because of the following error: CatBoostError('Invalid type for cat_feature[non-default value idx=0,feature_idx=0]=0.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.').
Traceback (most recent call last):
  File "_catboost.pyx", line 2687, in _catboost.get_cat_factor_bytes_representation
  File "_catboost.pyx", line 2180, in _catboost.get_id_object_bytes_string_representation
_catboost.CatBoostError: bad object for id: 0.0

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1543, in objective
    est.fit(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr])
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 1703, in fit
    return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 501, in fit
    return super().fit(Y, T, X=X, W=W,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_cate_estimator.py", line 134, in call
    m(self, Y, T, *args, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 821, in fit
    nuisances, fitted_models, new_inds, scores = self._fit_nuisances(
                                                 ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 971, in _fit_nuisances
    nuisances, fitted_models, fitted_inds, scores = _crossfit(self._ortho_learner_model_nuisance, folds,
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 278, in _crossfit
    nuisance_temp, model_out, score_temp = _fit_fold(model, train_idxs, test_idxs,
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 97, in _fit_fold
    model.train(False, None, *args_train, **kwargs_train)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 55, in train
    self._model_t.train(is_selecting, folds, X, W, T, **
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 110, in train
    self._model.train(is_selecting, folds, _combine(X, W, Target.shape[0]), Target,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 405, in train
    _fit_with_groups(self.model, X, y, groups=groups, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 378, in _fit_with_groups
    return model.fit(X, y, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/utils/categorical_patch.py", line 163, in _wrapped_fit
    return original_fit(self, X_conv, y, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5547, in fit
    self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline, use_best_model,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2701, in _fit
    train_params = self._prepare_train_params(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2581, in _prepare_train_params
    train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, graph,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 1540, in _build_train_pool
    train_pool = Pool(X, y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs, graph=graph, weight=sample_weight, group_id=group_id,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 863, in __init__
    self._init(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 1518, in _init
    self._init_pool(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
  File "_catboost.pyx", line 4695, in _catboost._PoolBase._init_pool
  File "_catboost.pyx", line 4753, in _catboost._PoolBase._init_pool
  File "_catboost.pyx", line 4541, in _catboost._PoolBase._init_features_order_layout_pool
  File "_catboost.pyx", line 3170, in _catboost._set_features_order_data_pd_data_frame
  File "_catboost.pyx", line 3070, in _catboost._set_features_order_data_frame_generic_categorical_column
  File "_catboost.pyx", line 2694, in _catboost.get_cat_factor_bytes_representation
_catboost.CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=0]=0.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.
[W 2026-05-01 16:03:31,158] Trial 95 failed with parameters: {'iterations': 133, 'learning_rate': 0.009423641237465766, 'depth': 2, 'l2_leaf_reg': 16.436588539927804, 'random_strength': 1.220320686501668, 'subsample': 0.7381229632226767, 'rsm': 0.3959722392824687, 'min_data_in_leaf': 285, 'model_size_reg': 0.47873534079507746, 'leaf_estimation_iterations': 5} because of the following error: CatBoostError('Invalid type for cat_feature[non-default value idx=0,feature_idx=0]=0.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.').
Traceback (most recent call last):
  File "_catboost.pyx", line 2687, in _catboost.get_cat_factor_bytes_representation
  File "_catboost.pyx", line 2180, in _catboost.get_id_object_bytes_string_representation
_catboost.CatBoostError: bad object for id: 0.0

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1543, in objective
    est.fit(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr])
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 1703, in fit
    return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 501, in fit
    return super().fit(Y, T, X=X, W=W,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_cate_estimator.py", line 134, in call
    m(self, Y, T, *args, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 821, in fit
    nuisances, fitted_models, new_inds, scores = self._fit_nuisances(
                                                 ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 971, in _fit_nuisances
    nuisances, fitted_models, fitted_inds, scores = _crossfit(self._ortho_learner_model_nuisance, folds,
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 278, in _crossfit
    nuisance_temp, model_out, score_temp = _fit_fold(model, train_idxs, test_idxs,
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 97, in _fit_fold
    model.train(False, None, *args_train, **kwargs_train)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 55, in train
    self._model_t.train(is_selecting, folds, X, W, T, **
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 110, in train
    self._model.train(is_selecting, folds, _combine(X, W, Target.shape[0]), Target,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 405, in train
    _fit_with_groups(self.model, X, y, groups=groups, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 378, in _fit_with_groups
    return model.fit(X, y, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/utils/categorical_patch.py", line 163, in _wrapped_fit
    return original_fit(self, X_conv, y, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5547, in fit
    self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline, use_best_model,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2701, in _fit
    train_params = self._prepare_train_params(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2581, in _prepare_train_params
    train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, graph,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 1540, in _build_train_pool
    train_pool = Pool(X, y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs, graph=graph, weight=sample_weight, group_id=group_id,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 863, in __init__
    self._init(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 1518, in _init
    self._init_pool(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
  File "_catboost.pyx", line 4695, in _catboost._PoolBase._init_pool
  File "_catboost.pyx", line 4753, in _catboost._PoolBase._init_pool
  File "_catboost.pyx", line 4541, in _catboost._PoolBase._init_features_order_layout_pool
  File "_catboost.pyx", line 3170, in _catboost._set_features_order_data_pd_data_frame
  File "_catboost.pyx", line 3070, in _catboost._set_features_order_data_frame_generic_categorical_column
  File "_catboost.pyx", line 2694, in _catboost.get_cat_factor_bytes_representation
_catboost.CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=0]=0.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.
[W 2026-05-01 16:03:31,187] Trial 95 failed with value None.
[W 2026-05-01 16:03:31,177] Trial 94 failed with value None.
[W 2026-05-01 16:03:34,448] Trial 97 failed with parameters: {'iterations': 363, 'learning_rate': 0.03471969412687759, 'depth': 5, 'l2_leaf_reg': 6.2667039451070465, 'random_strength': 1.3029671106692755, 'subsample': 0.4625853285214029, 'rsm': 0.44138125683941726, 'min_data_in_leaf': 427, 'model_size_reg': 2.071418223591099, 'leaf_estimation_iterations': 5} because of the following error: CatBoostError('Invalid type for cat_feature[non-default value idx=0,feature_idx=0]=0.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.').
Traceback (most recent call last):
  File "_catboost.pyx", line 2687, in _catboost.get_cat_factor_bytes_representation
  File "_catboost.pyx", line 2180, in _catboost.get_id_object_bytes_string_representation
_catboost.CatBoostError: bad object for id: 0.0

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1543, in objective
    est.fit(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr])
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 1703, in fit
    return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 501, in fit
    return super().fit(Y, T, X=X, W=W,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_cate_estimator.py", line 134, in call
    m(self, Y, T, *args, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 821, in fit
    nuisances, fitted_models, new_inds, scores = self._fit_nuisances(
                                                 ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 971, in _fit_nuisances
    nuisances, fitted_models, fitted_inds, scores = _crossfit(self._ortho_learner_model_nuisance, folds,
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 278, in _crossfit
    nuisance_temp, model_out, score_temp = _fit_fold(model, train_idxs, test_idxs,
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 97, in _fit_fold
    model.train(False, None, *args_train, **kwargs_train)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 55, in train
    self._model_t.train(is_selecting, folds, X, W, T, **
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 110, in train
    self._model.train(is_selecting, folds, _combine(X, W, Target.shape[0]), Target,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 405, in train
    _fit_with_groups(self.model, X, y, groups=groups, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 378, in _fit_with_groups
    return model.fit(X, y, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/utils/categorical_patch.py", line 163, in _wrapped_fit
    return original_fit(self, X_conv, y, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5547, in fit
    self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline, use_best_model,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2701, in _fit
    train_params = self._prepare_train_params(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2581, in _prepare_train_params
    train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, graph,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 1540, in _build_train_pool
    train_pool = Pool(X, y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs, graph=graph, weight=sample_weight, group_id=group_id,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 863, in __init__
    self._init(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 1518, in _init
    self._init_pool(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
  File "_catboost.pyx", line 4695, in _catboost._PoolBase._init_pool
  File "_catboost.pyx", line 4753, in _catboost._PoolBase._init_pool
  File "_catboost.pyx", line 4541, in _catboost._PoolBase._init_features_order_layout_pool
  File "_catboost.pyx", line 3170, in _catboost._set_features_order_data_pd_data_frame
  File "_catboost.pyx", line 3070, in _catboost._set_features_order_data_frame_generic_categorical_column
  File "_catboost.pyx", line 2694, in _catboost.get_cat_factor_bytes_representation
_catboost.CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=0]=0.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.
[W 2026-05-01 16:03:34,451] Trial 97 failed with value None.
[W 2026-05-01 16:03:34,517] Trial 98 failed with parameters: {'iterations': 366, 'learning_rate': 0.048269569205517525, 'depth': 5, 'l2_leaf_reg': 12.173787370997111, 'random_strength': 1.4349128412565515, 'subsample': 0.5475574186492825, 'rsm': 0.4292604690721167, 'min_data_in_leaf': 172, 'model_size_reg': 2.131182751161153, 'leaf_estimation_iterations': 1} because of the following error: CatBoostError('Invalid type for cat_feature[non-default value idx=0,feature_idx=0]=0.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.').
Traceback (most recent call last):
  File "_catboost.pyx", line 2687, in _catboost.get_cat_factor_bytes_representation
  File "_catboost.pyx", line 2180, in _catboost.get_id_object_bytes_string_representation
_catboost.CatBoostError: bad object for id: 0.0

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1543, in objective
    est.fit(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr])
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 1703, in fit
    return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 501, in fit
    return super().fit(Y, T, X=X, W=W,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_cate_estimator.py", line 134, in call
    m(self, Y, T, *args, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 821, in fit
    nuisances, fitted_models, new_inds, scores = self._fit_nuisances(
                                                 ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 971, in _fit_nuisances
    nuisances, fitted_models, fitted_inds, scores = _crossfit(self._ortho_learner_model_nuisance, folds,
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 278, in _crossfit
    nuisance_temp, model_out, score_temp = _fit_fold(model, train_idxs, test_idxs,
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 97, in _fit_fold
    model.train(False, None, *args_train, **kwargs_train)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 55, in train
    self._model_t.train(is_selecting, folds, X, W, T, **
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 110, in train
    self._model.train(is_selecting, folds, _combine(X, W, Target.shape[0]), Target,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 405, in train
    _fit_with_groups(self.model, X, y, groups=groups, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 378, in _fit_with_groups
    return model.fit(X, y, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/utils/categorical_patch.py", line 163, in _wrapped_fit
    return original_fit(self, X_conv, y, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5547, in fit
    self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline, use_best_model,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2701, in _fit
    train_params = self._prepare_train_params(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2581, in _prepare_train_params
    train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, graph,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 1540, in _build_train_pool
    train_pool = Pool(X, y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs, graph=graph, weight=sample_weight, group_id=group_id,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 863, in __init__
    self._init(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 1518, in _init
    self._init_pool(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
  File "_catboost.pyx", line 4695, in _catboost._PoolBase._init_pool
  File "_catboost.pyx", line 4753, in _catboost._PoolBase._init_pool
  File "_catboost.pyx", line 4541, in _catboost._PoolBase._init_features_order_layout_pool
  File "_catboost.pyx", line 3170, in _catboost._set_features_order_data_pd_data_frame
  File "_catboost.pyx", line 3070, in _catboost._set_features_order_data_frame_generic_categorical_column
  File "_catboost.pyx", line 2694, in _catboost.get_cat_factor_bytes_representation
_catboost.CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=0]=0.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.
[W 2026-05-01 16:03:34,521] Trial 98 failed with value None.
[W 2026-05-01 16:03:36,340] Trial 99 failed with parameters: {'iterations': 353, 'learning_rate': 0.0289873192400066, 'depth': 6, 'l2_leaf_reg': 11.418036125662335, 'random_strength': 2.2910568901627832, 'subsample': 0.8079160431339756, 'rsm': 0.3259428435229126, 'min_data_in_leaf': 287, 'model_size_reg': 0.1212847082067221, 'leaf_estimation_iterations': 5} because of the following error: CatBoostError('Invalid type for cat_feature[non-default value idx=0,feature_idx=0]=0.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.').
Traceback (most recent call last):
  File "_catboost.pyx", line 2687, in _catboost.get_cat_factor_bytes_representation
  File "_catboost.pyx", line 2180, in _catboost.get_id_object_bytes_string_representation
_catboost.CatBoostError: bad object for id: 0.0

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1543, in objective
    est.fit(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr])
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 1703, in fit
    return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 501, in fit
    return super().fit(Y, T, X=X, W=W,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_cate_estimator.py", line 134, in call
    m(self, Y, T, *args, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 821, in fit
    nuisances, fitted_models, new_inds, scores = self._fit_nuisances(
                                                 ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 971, in _fit_nuisances
    nuisances, fitted_models, fitted_inds, scores = _crossfit(self._ortho_learner_model_nuisance, folds,
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 278, in _crossfit
    nuisance_temp, model_out, score_temp = _fit_fold(model, train_idxs, test_idxs,
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 97, in _fit_fold
    model.train(False, None, *args_train, **kwargs_train)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 55, in train
    self._model_t.train(is_selecting, folds, X, W, T, **
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 110, in train
    self._model.train(is_selecting, folds, _combine(X, W, Target.shape[0]), Target,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 405, in train
    _fit_with_groups(self.model, X, y, groups=groups, **kwargs)
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 378, in _fit_with_groups
    return model.fit(X, y, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/utils/categorical_patch.py", line 163, in _wrapped_fit
    return original_fit(self, X_conv, y, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5547, in fit
    self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline, use_best_model,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2701, in _fit
    train_params = self._prepare_train_params(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2581, in _prepare_train_params
    train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, graph,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 1540, in _build_train_pool
    train_pool = Pool(X, y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs, graph=graph, weight=sample_weight, group_id=group_id,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 863, in __init__
    self._init(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 1518, in _init
    self._init_pool(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
  File "_catboost.pyx", line 4695, in _catboost._PoolBase._init_pool
  File "_catboost.pyx", line 4753, in _catboost._PoolBase._init_pool
  File "_catboost.pyx", line 4541, in _catboost._PoolBase._init_features_order_layout_pool
  File "_catboost.pyx", line 3170, in _catboost._set_features_order_data_pd_data_frame
  File "_catboost.pyx", line 3070, in _catboost._set_features_order_data_frame_generic_categorical_column
  File "_catboost.pyx", line 2694, in _catboost.get_cat_factor_bytes_representation
_catboost.CatBoostError: Invalid type for cat_feature[non-default value idx=0,feature_idx=0]=0.0 : cat_features must be integer or string, real number values and NaN values should be converted to string.
[W 2026-05-01 16:03:36,344] Trial 99 failed with value None.
16:03:36 INFO [rubin.tuning] FMT 'NonParamDML': 0/100 Trials abgeschlossen (100 fehlgeschlagen, 0 gepruned).
16:03:36 WARNING [rubin.tuning] FMT 'NonParamDML': 100/100 Trials FEHLGESCHLAGEN. Fehlertypen:
16:03:36 WARNING [rubin.tuning]   [100×] Unbekannt
16:03:36 WARNING [rubin.tuning] FMT 'NonParamDML': Häufigster Fehler — vollständiger Traceback:
Unbekannt
16:03:36 WARNING [rubin.tuning] FMT 'NonParamDML': Keine abgeschlossenen Trials. Verwende Default-Parameter.
16:03:37 INFO [rubin.tuning] FMT 'NonParamDML': Study freigegeben, gc.collect() durchgeführt.
16:03:39 INFO [rubin.tuning] CausalForest-Tuning (CausalForestDML): Starte 100 Optuna-Trials, 5 äußere Folds (Seed=18).
