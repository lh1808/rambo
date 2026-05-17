
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 673, in objective
    return self._objective_xlearner_cate(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 564, in _objective_xlearner_cate
    pseudo = self._build_xlearner_pseudo_outcomes(X_mat, Y, T, nuisance_params=nuisance_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 533, in _build_xlearner_pseudo_outcomes
    m0 = self._fit_model(nuisance_params, X_mat.iloc[control_train], control_y, "classifier")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 354, in _fit_model
    model.fit(X_train, y_train)
  File "/mnt/rubin/rubin/utils/categorical_patch.py", line 179, in _wrapped_fit
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
  File "_catboost.pyx", line 3189, in _catboost._set_features_order_data_pd_data_frame
  File "_catboost.pyx", line 2673, in _catboost.create_num_factor_data
  File "_catboost.pyx", line 2635, in _catboost._set_features_order_data_frame_generic_num_column
  File "_catboost.pyx", line 2608, in _catboost.get_float_feature
_catboost.CatBoostError: Bad value for num_feature[non_default_doc_idx=0,feature_idx=2]="2025-10-08T00:00:00": Cannot convert obj 2025-10-08T00:00:00 to float
[W 2026-05-17 22:21:51,482] Trial 60 failed with value None.
[W 2026-05-17 22:21:51,488] Trial 61 failed with parameters: {'iterations': 227, 'learning_rate': 0.02880615887479098, 'depth': 8, 'l2_leaf_reg': 23.40580076283386, 'random_strength': 2.587809105163032, 'subsample': 0.8101055300656175, 'rsm': 0.822450536083132, 'min_data_in_leaf': 64, 'model_size_reg': 0.33490759971921413, 'leaf_estimation_iterations': 9} because of the following error: CatBoostError('Bad value for num_feature[non_default_doc_idx=0,feature_idx=2]="2025-10-08T00:00:00": Cannot convert obj 2025-10-08T00:00:00 to float').
Traceback (most recent call last):
  File "_catboost.pyx", line 1292, in _catboost._FloatOrNan
TypeError: float() argument must be a string or a real number, not 'datetime.datetime'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "_catboost.pyx", line 2606, in _catboost.get_float_feature
  File "_catboost.pyx", line 1294, in _catboost._FloatOrNan
TypeError: Cannot convert obj 2025-10-08T00:00:00 to float

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 673, in objective
    return self._objective_xlearner_cate(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 564, in _objective_xlearner_cate
    pseudo = self._build_xlearner_pseudo_outcomes(X_mat, Y, T, nuisance_params=nuisance_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 533, in _build_xlearner_pseudo_outcomes
    m0 = self._fit_model(nuisance_params, X_mat.iloc[control_train], control_y, "classifier")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 354, in _fit_model
    model.fit(X_train, y_train)
  File "/mnt/rubin/rubin/utils/categorical_patch.py", line 179, in _wrapped_fit
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
  File "_catboost.pyx", line 3189, in _catboost._set_features_order_data_pd_data_frame
  File "_catboost.pyx", line 2673, in _catboost.create_num_factor_data
  File "_catboost.pyx", line 2635, in _catboost._set_features_order_data_frame_generic_num_column
  File "_catboost.pyx", line 2608, in _catboost.get_float_feature
_catboost.CatBoostError: Bad value for num_feature[non_default_doc_idx=0,feature_idx=2]="2025-10-08T00:00:00": Cannot convert obj 2025-10-08T00:00:00 to float
[W 2026-05-17 22:21:51,492] Trial 61 failed with value None.
[W 2026-05-17 22:21:51,533] Trial 64 failed with parameters: {'iterations': 295, 'learning_rate': 0.10582958549359817, 'depth': 7, 'l2_leaf_reg': 27.71528223685406, 'random_strength': 4.255075176329455, 'subsample': 0.8132152845939966, 'rsm': 0.6364613522246645, 'min_data_in_leaf': 194, 'model_size_reg': 9.09343262325799, 'leaf_estimation_iterations': 3} because of the following error: CatBoostError('Bad value for num_feature[non_default_doc_idx=0,feature_idx=2]="2025-10-08T00:00:00": Cannot convert obj 2025-10-08T00:00:00 to float').
Traceback (most recent call last):
  File "_catboost.pyx", line 1292, in _catboost._FloatOrNan
TypeError: float() argument must be a string or a real number, not 'datetime.datetime'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "_catboost.pyx", line 2606, in _catboost.get_float_feature
  File "_catboost.pyx", line 1294, in _catboost._FloatOrNan
TypeError: Cannot convert obj 2025-10-08T00:00:00 to float

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 673, in objective
    return self._objective_xlearner_cate(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 564, in _objective_xlearner_cate
    pseudo = self._build_xlearner_pseudo_outcomes(X_mat, Y, T, nuisance_params=nuisance_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 533, in _build_xlearner_pseudo_outcomes
    m0 = self._fit_model(nuisance_params, X_mat.iloc[control_train], control_y, "classifier")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 354, in _fit_model
    model.fit(X_train, y_train)
  File "/mnt/rubin/rubin/utils/categorical_patch.py", line 179, in _wrapped_fit
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
  File "_catboost.pyx", line 3189, in _catboost._set_features_order_data_pd_data_frame
  File "_catboost.pyx", line 2673, in _catboost.create_num_factor_data
  File "_catboost.pyx", line 2635, in _catboost._set_features_order_data_frame_generic_num_column
  File "_catboost.pyx", line 2608, in _catboost.get_float_feature
_catboost.CatBoostError: Bad value for num_feature[non_default_doc_idx=0,feature_idx=2]="2025-10-08T00:00:00": Cannot convert obj 2025-10-08T00:00:00 to float
[W 2026-05-17 22:21:51,537] Trial 65 failed with parameters: {'iterations': 597, 'learning_rate': 0.08680861694325329, 'depth': 6, 'l2_leaf_reg': 6.657886599760105, 'random_strength': 3.806586566138453, 'subsample': 0.5748592749244026, 'rsm': 0.6386746071319551, 'min_data_in_leaf': 137, 'model_size_reg': 0.9191976327429574, 'leaf_estimation_iterations': 2} because of the following error: CatBoostError('Bad value for num_feature[non_default_doc_idx=0,feature_idx=2]="2025-10-08T00:00:00": Cannot convert obj 2025-10-08T00:00:00 to float').
Traceback (most recent call last):
  File "_catboost.pyx", line 1292, in _catboost._FloatOrNan
TypeError: float() argument must be a string or a real number, not 'datetime.datetime'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "_catboost.pyx", line 2606, in _catboost.get_float_feature
  File "_catboost.pyx", line 1294, in _catboost._FloatOrNan
TypeError: Cannot convert obj 2025-10-08T00:00:00 to float

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 673, in objective
    return self._objective_xlearner_cate(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 564, in _objective_xlearner_cate
    pseudo = self._build_xlearner_pseudo_outcomes(X_mat, Y, T, nuisance_params=nuisance_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 533, in _build_xlearner_pseudo_outcomes
    m0 = self._fit_model(nuisance_params, X_mat.iloc[control_train], control_y, "classifier")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 354, in _fit_model
    model.fit(X_train, y_train)
  File "/mnt/rubin/rubin/utils/categorical_patch.py", line 179, in _wrapped_fit
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
  File "_catboost.pyx", line 3189, in _catboost._set_features_order_data_pd_data_frame
  File "_catboost.pyx", line 2673, in _catboost.create_num_factor_data
  File "_catboost.pyx", line 2635, in _catboost._set_features_order_data_frame_generic_num_column
  File "_catboost.pyx", line 2608, in _catboost.get_float_feature
_catboost.CatBoostError: Bad value for num_feature[non_default_doc_idx=0,feature_idx=2]="2025-10-08T00:00:00": Cannot convert obj 2025-10-08T00:00:00 to float
[W 2026-05-17 22:21:51,539] Trial 64 failed with value None.
[W 2026-05-17 22:21:51,542] Trial 65 failed with value None.
[W 2026-05-17 22:21:51,557] Trial 62 failed with parameters: {'iterations': 555, 'learning_rate': 0.13415425169923145, 'depth': 5, 'l2_leaf_reg': 27.083401397903668, 'random_strength': 7.338707073424967, 'subsample': 0.669718875483712, 'rsm': 0.8047846746953156, 'min_data_in_leaf': 64, 'model_size_reg': 7.230825121861901, 'leaf_estimation_iterations': 10} because of the following error: CatBoostError('Bad value for num_feature[non_default_doc_idx=0,feature_idx=2]="2025-10-08T00:00:00": Cannot convert obj 2025-10-08T00:00:00 to float').
Traceback (most recent call last):
  File "_catboost.pyx", line 1292, in _catboost._FloatOrNan
TypeError: float() argument must be a string or a real number, not 'datetime.datetime'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "_catboost.pyx", line 2606, in _catboost.get_float_feature
  File "_catboost.pyx", line 1294, in _catboost._FloatOrNan
TypeError: Cannot convert obj 2025-10-08T00:00:00 to float

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 673, in objective
    return self._objective_xlearner_cate(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 564, in _objective_xlearner_cate
    pseudo = self._build_xlearner_pseudo_outcomes(X_mat, Y, T, nuisance_params=nuisance_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 533, in _build_xlearner_pseudo_outcomes
    m0 = self._fit_model(nuisance_params, X_mat.iloc[control_train], control_y, "classifier")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 354, in _fit_model
    model.fit(X_train, y_train)
  File "/mnt/rubin/rubin/utils/categorical_patch.py", line 179, in _wrapped_fit
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
  File "_catboost.pyx", line 3189, in _catboost._set_features_order_data_pd_data_frame
  File "_catboost.pyx", line 2673, in _catboost.create_num_factor_data
  File "_catboost.pyx", line 2635, in _catboost._set_features_order_data_frame_generic_num_column
  File "_catboost.pyx", line 2608, in _catboost.get_float_feature
_catboost.CatBoostError: Bad value for num_feature[non_default_doc_idx=0,feature_idx=2]="2025-10-08T00:00:00": Cannot convert obj 2025-10-08T00:00:00 to float
[W 2026-05-17 22:21:51,561] Trial 62 failed with value None.
[W 2026-05-17 22:21:51,597] Trial 63 failed with parameters: {'iterations': 227, 'learning_rate': 0.010201866040808928, 'depth': 5, 'l2_leaf_reg': 14.511383396404023, 'random_strength': 2.120372082661333, 'subsample': 0.692568459143134, 'rsm': 0.3006002888840829, 'min_data_in_leaf': 199, 'model_size_reg': 7.7105825509575885, 'leaf_estimation_iterations': 8} because of the following error: CatBoostError('Bad value for num_feature[non_default_doc_idx=0,feature_idx=2]="2025-10-08T00:00:00": Cannot convert obj 2025-10-08T00:00:00 to float').
Traceback (most recent call last):
  File "_catboost.pyx", line 1292, in _catboost._FloatOrNan
TypeError: float() argument must be a string or a real number, not 'datetime.datetime'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "_catboost.pyx", line 2606, in _catboost.get_float_feature
  File "_catboost.pyx", line 1294, in _catboost._FloatOrNan
TypeError: Cannot convert obj 2025-10-08T00:00:00 to float

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 673, in objective
    return self._objective_xlearner_cate(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 564, in _objective_xlearner_cate
    pseudo = self._build_xlearner_pseudo_outcomes(X_mat, Y, T, nuisance_params=nuisance_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 533, in _build_xlearner_pseudo_outcomes
    m0 = self._fit_model(nuisance_params, X_mat.iloc[control_train], control_y, "classifier")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning/base_learner.py", line 354, in _fit_
