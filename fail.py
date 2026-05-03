[rubin] Step 4/11: Final-Model-Tuning
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 104, in _fit_fold
    score_temp = model.score(*args_test, **kwargs_test)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 64, in score
    T_score = self._model_t.score(X, W, T, **filter_none_kwargs(sample_weight=sample_weight),
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 338, in score
    return self.best_model.score(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 82, in score
    return self._model.score(XW_combined, Target)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5869, in score
    predicted_classes = self._predict(
                        ^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-03 12:50:26,253] Trial 6 failed with value None.
[W 2026-05-03 12:50:28,042] Trial 8 failed with parameters: {'iterations': 350, 'learning_rate': 0.018809408110099568, 'depth': 2, 'l2_leaf_reg': 44.45070767588796, 'random_strength': 1.1416200500933449, 'subsample': 0.7060707365337817, 'rsm': 0.6419783554953077, 'min_data_in_leaf': 439, 'model_size_reg': 0.10970122512795853, 'leaf_estimation_iterations': 5} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 104, in _fit_fold
    score_temp = model.score(*args_test, **kwargs_test)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 64, in score
    T_score = self._model_t.score(X, W, T, **filter_none_kwargs(sample_weight=sample_weight),
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 338, in score
    return self.best_model.score(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 82, in score
    return self._model.score(XW_combined, Target)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5869, in score
    predicted_classes = self._predict(
                        ^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-03 12:50:28,045] Trial 8 failed with value None.
[W 2026-05-03 12:50:29,064] Trial 5 failed with parameters: {'iterations': 317, 'learning_rate': 0.014358561674786177, 'depth': 6, 'l2_leaf_reg': 8.338679796984108, 'random_strength': 8.460108481087596, 'subsample': 0.5977418910424116, 'rsm': 0.6837322920291478, 'min_data_in_leaf': 351, 'model_size_reg': 3.3146950732053293, 'leaf_estimation_iterations': 3} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 104, in _fit_fold
    score_temp = model.score(*args_test, **kwargs_test)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 64, in score
    T_score = self._model_t.score(X, W, T, **filter_none_kwargs(sample_weight=sample_weight),
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 338, in score
    return self.best_model.score(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 82, in score
    return self._model.score(XW_combined, Target)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5869, in score
    predicted_classes = self._predict(
                        ^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-03 12:50:29,068] Trial 5 failed with value None.
[W 2026-05-03 12:59:49,216] Trial 10 failed with parameters: {'iterations': 242, 'learning_rate': 0.08644533267163283, 'depth': 3, 'l2_leaf_reg': 50.71187444649126, 'random_strength': 1.3009042852293657, 'subsample': 0.8402981710085186, 'rsm': 0.37784408910639194, 'min_data_in_leaf': 460, 'model_size_reg': 4.68921667931623, 'leaf_estimation_iterations': 1} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 104, in _fit_fold
    score_temp = model.score(*args_test, **kwargs_test)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 64, in score
    T_score = self._model_t.score(X, W, T, **filter_none_kwargs(sample_weight=sample_weight),
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 338, in score
    return self.best_model.score(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 82, in score
    return self._model.score(XW_combined, Target)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5869, in score
    predicted_classes = self._predict(
                        ^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-03 12:59:49,629] Trial 10 failed with value None.
[W 2026-05-03 12:59:51,935] Trial 11 failed with parameters: {'iterations': 295, 'learning_rate': 0.01018575473773546, 'depth': 4, 'l2_leaf_reg': 9.71923788416116, 'random_strength': 9.93645608782917, 'subsample': 0.5843447032046994, 'rsm': 0.45727123010283954, 'min_data_in_leaf': 66, 'model_size_reg': 14.017314879770142, 'leaf_estimation_iterations': 4} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 104, in _fit_fold
    score_temp = model.score(*args_test, **kwargs_test)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 64, in score
    T_score = self._model_t.score(X, W, T, **filter_none_kwargs(sample_weight=sample_weight),
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 338, in score
    return self.best_model.score(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 82, in score
    return self._model.score(XW_combined, Target)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5869, in score
    predicted_classes = self._predict(
                        ^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-03 12:59:51,942] Trial 11 failed with value None.
[W 2026-05-03 12:59:54,707] Trial 12 failed with parameters: {'iterations': 140, 'learning_rate': 0.03621294076757047, 'depth': 6, 'l2_leaf_reg': 8.74877409613634, 'random_strength': 1.634887521263846, 'subsample': 0.6482941777370633, 'rsm': 0.4487276079436804, 'min_data_in_leaf': 280, 'model_size_reg': 0.8619936846381854, 'leaf_estimation_iterations': 1} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 104, in _fit_fold
    score_temp = model.score(*args_test, **kwargs_test)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 64, in score
    T_score = self._model_t.score(X, W, T, **filter_none_kwargs(sample_weight=sample_weight),
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 338, in score
    return self.best_model.score(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 82, in score
    return self._model.score(XW_combined, Target)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5869, in score
    predicted_classes = self._predict(
                        ^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-03 12:59:54,982] Trial 12 failed with value None.
[W 2026-05-03 12:59:57,825] Trial 13 failed with parameters: {'iterations': 106, 'learning_rate': 0.07064514050690418, 'depth': 2, 'l2_leaf_reg': 3.3385993928608273, 'random_strength': 0.9663692858557871, 'subsample': 0.6698850038579695, 'rsm': 0.6222056311702404, 'min_data_in_leaf': 160, 'model_size_reg': 1.2277752731172051, 'leaf_estimation_iterations': 5} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 104, in _fit_fold
    score_temp = model.score(*args_test, **kwargs_test)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 64, in score
    T_score = self._model_t.score(X, W, T, **filter_none_kwargs(sample_weight=sample_weight),
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 338, in score
    return self.best_model.score(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 82, in score
    return self._model.score(XW_combined, Target)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5869, in score
    predicted_classes = self._predict(
                        ^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-03 12:59:57,845] Trial 13 failed with value None.
[W 2026-05-03 12:59:58,895] Trial 14 failed with parameters: {'iterations': 252, 'learning_rate': 0.020503112411709917, 'depth': 6, 'l2_leaf_reg': 53.623948433435466, 'random_strength': 0.5943204004452838, 'subsample': 0.6442314945192923, 'rsm': 0.2665709757565696, 'min_data_in_leaf': 402, 'model_size_reg': 3.230727124293863, 'leaf_estimation_iterations': 1} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
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
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 104, in _fit_fold
    score_temp = model.score(*args_test, **kwargs_test)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 64, in score
    T_score = self._model_t.score(X, W, T, **filter_none_kwargs(sample_weight=sample_weight),
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/sklearn_extensions/model_selection.py", line 338, in score
    return self.best_model.score(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 82, in score
    return self._model.score(XW_combined, Target)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5869, in score
    predicted_classes = self._predict(
                        ^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-03 12:59:58,898] Trial 14 failed with value None.
