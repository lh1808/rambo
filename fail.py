Analyse fehlgeschlagen: Fehlgeschlagen (Exit 1)

Details:
[W 2026-04-30 22:17:57,839] Trial 94 failed with parameters: {'iterations': 361, 'learning_rate': 0.017713907971923296, 'depth': 5, 'l2_leaf_reg': 44.70471844788231, 'random_strength': 0.9938457243592219, 'subsample': 0.640005265078105, 'rsm': 0.5978230184952011, 'min_data_in_leaf': 132, 'model_size_reg': 6.74382600519477, 'leaf_estimation_iterations': 1} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-04-30 22:17:57,855] Trial 94 failed with value None.
[W 2026-04-30 22:17:58,214] Trial 96 failed with parameters: {'iterations': 290, 'learning_rate': 0.007737927913712262, 'depth': 4, 'l2_leaf_reg': 29.212164790880657, 'random_strength': 5.424582743129767, 'subsample': 0.4837419326629647, 'rsm': 0.5978447097732702, 'min_data_in_leaf': 439, 'model_size_reg': 3.546740752168005, 'leaf_estimation_iterations': 3} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-04-30 22:17:58,217] Trial 96 failed with value None.
[W 2026-04-30 22:17:58,244] Trial 95 failed with parameters: {'iterations': 339, 'learning_rate': 0.07781972247315681, 'depth': 5, 'l2_leaf_reg': 5.095049487459031, 'random_strength': 2.2416063375240625, 'subsample': 0.7904656607477165, 'rsm': 0.6318845637033661, 'min_data_in_leaf': 107, 'model_size_reg': 15.885841011023349, 'leaf_estimation_iterations': 1} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-04-30 22:17:58,248] Trial 95 failed with value None.
[W 2026-04-30 22:18:00,128] Trial 98 failed with parameters: {'iterations': 110, 'learning_rate': 0.07612827600939066, 'depth': 3, 'l2_leaf_reg': 13.925265693730244, 'random_strength': 9.32973515446159, 'subsample': 0.5681439259878602, 'rsm': 0.5808773043925161, 'min_data_in_leaf': 399, 'model_size_reg': 1.2036190118674648, 'leaf_estimation_iterations': 5} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-04-30 22:18:00,132] Trial 98 failed with value None.
[W 2026-04-30 22:18:00,167] Trial 97 failed with parameters: {'iterations': 350, 'learning_rate': 0.03115042606008819, 'depth': 5, 'l2_leaf_reg': 15.097006001401082, 'random_strength': 3.0777520136202923, 'subsample': 0.8499082231558004, 'rsm': 0.25523192311155896, 'min_data_in_leaf': 242, 'model_size_reg': 0.12889988890181975, 'leaf_estimation_iterations': 2} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-04-30 22:18:00,169] Trial 97 failed with value None.
[W 2026-04-30 22:18:01,483] Trial 99 failed with parameters: {'iterations': 231, 'learning_rate': 0.03850507301677706, 'depth': 6, 'l2_leaf_reg': 36.212478893585626, 'random_strength': 7.122077954456384, 'subsample': 0.518786729624477, 'rsm': 0.5665228899111685, 'min_data_in_leaf': 23, 'model_size_reg': 1.72316637007211, 'leaf_estimation_iterations': 2} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-04-30 22:18:01,487] Trial 99 failed with value None.
22:18:01 INFO [rubin.tuning] FMT 'NonParamDML': 0/100 Trials abgeschlossen (100 fehlgeschlagen, 0 gepruned).
22:18:01 WARNING [rubin.tuning] FMT 'NonParamDML': 100/100 Trials FEHLGESCHLAGEN. Fehlertypen:
22:18:01 WARNING [rubin.tuning]   [100×] Unbekannt
22:18:01 WARNING [rubin.tuning] FMT 'NonParamDML': Häufigster Fehler — vollständiger Traceback:
Unbekannt
22:18:01 WARNING [rubin.tuning] FMT 'NonParamDML': Keine abgeschlossenen Trials. Verwende Default-Parameter.
22:18:02 INFO [rubin.tuning] FMT 'NonParamDML': Study freigegeben, gc.collect() durchgeführt.
22:18:03 WARNING [rubin.analysis] CausalForest-Tuning (CausalForestDML) fehlgeschlagen.
Traceback (most recent call last):
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 870, in _run_training
    result = tune_causal_forest(
             ^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1856, in tune_causal_forest
    model_type, n_trials, n_eval_folds, tuning_cv_seed,
                          ^^^^^^^^^^^^
NameError: name 'n_eval_folds' is not defined
22:18:03 INFO [rubin.analysis]   CausalForestDML: Starte 5-Fold Cross-Prediction (Seed=42).
22:18:04 INFO [rubin.training] CausalForestDML: Erzwinge sequentielle Folds (GRF nutzt intern joblib-Prozesse, die in Threads zu Deadlocks führen).
22:18:04 INFO [rubin.training] CausalForestDML: 5 Folds sequentiell.
Traceback (most recent call last):
  File "/mnt/rubin/run_analysis.py", line 132, in <module>
    main()
  File "/mnt/rubin/run_analysis.py", line 128, in main
    pipe.run(export_bundle=args.export_bundle, bundle_dir=args.bundle_dir, bundle_id=args.bundle_id)
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 3096, in run
    models, preds, fold_models = self._run_training(cfg, X, T, Y, tuned_params_by_model, holdout_data, mlflow, _progress_cb=_progress)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 914, in _run_training
    result = train_and_crosspredict_bt_bo(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/training.py", line 312, in train_and_crosspredict_bt_bo
    preds, last_fold_model, last_fold_val_idx = _run_folds_sequential(
                                                ^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/training.py", line 165, in _run_folds_sequential
    m.fit(Y[tr_idx], T[tr_idx], X=X.iloc[tr_idx])
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/causal_forest.py", line 964, in fit
    return super().fit(Y, T, X=X, W=W,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
