[rubin] Step 1/11: Daten laden & Preprocessing
[rubin] dtypes.json auto-erkannt: 254 Spalten-Dtypes wiederhergestellt (254 kategorial).
[rubin] Step 2/11: Feature-Selektion
[rubin] Step 3/11: Base-Learner-Tuning
[rubin] Kategorische Features: 42 von 159 Spalten → CATBOOST erhält cat_feature-Indizes.
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:35:20,034] Trial 17 failed with value None.
[W 2026-05-01 10:35:20,214] Trial 13 failed with parameters: {'iterations': 479, 'learning_rate': 0.12610359877128502, 'depth': 6, 'l2_leaf_reg': 12.619213251882993, 'random_strength': 0.02973746432233828, 'subsample': 0.5657872603312378, 'rsm': 0.5925726116692638, 'min_data_in_leaf': 131, 'model_size_reg': 2.726277022736494, 'leaf_estimation_iterations': 9} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:35:20,216] Trial 13 failed with value None.
[W 2026-05-01 10:35:46,525] Trial 16 failed with parameters: {'iterations': 544, 'learning_rate': 0.011999355844942697, 'depth': 7, 'l2_leaf_reg': 11.975704255814831, 'random_strength': 3.800375953523501, 'subsample': 0.6211840815289911, 'rsm': 0.4690309540552533, 'min_data_in_leaf': 82, 'model_size_reg': 9.347513023417026, 'leaf_estimation_iterations': 4} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:35:46,609] Trial 16 failed with value None.
[W 2026-05-01 10:36:23,321] Trial 22 failed with parameters: {'iterations': 272, 'learning_rate': 0.03196642710846895, 'depth': 6, 'l2_leaf_reg': 15.332015986352602, 'random_strength': 2.295750293480815, 'subsample': 0.942597919890354, 'rsm': 0.30108781726195716, 'min_data_in_leaf': 146, 'model_size_reg': 8.316976610816905, 'leaf_estimation_iterations': 1} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:36:23,323] Trial 22 failed with value None.
[W 2026-05-01 10:37:26,332] Trial 19 failed with parameters: {'iterations': 450, 'learning_rate': 0.0307432884046234, 'depth': 4, 'l2_leaf_reg': 13.026926421885948, 'random_strength': 0.015015741181791484, 'subsample': 0.8144432533451116, 'rsm': 0.7792979451493576, 'min_data_in_leaf': 125, 'model_size_reg': 9.679231663463941, 'leaf_estimation_iterations': 10} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:37:26,335] Trial 19 failed with value None.
[W 2026-05-01 10:37:51,178] Trial 26 failed with parameters: {'iterations': 305, 'learning_rate': 0.02300105047975867, 'depth': 5, 'l2_leaf_reg': 1.6213972061301418, 'random_strength': 1.0921070773095911, 'subsample': 0.6826671150332972, 'rsm': 0.4410640213192266, 'min_data_in_leaf': 107, 'model_size_reg': 2.92313217648434, 'leaf_estimation_iterations': 8} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:37:51,181] Trial 26 failed with value None.
[W 2026-05-01 10:38:38,638] Trial 15 failed with parameters: {'iterations': 581, 'learning_rate': 0.022818033825979217, 'depth': 6, 'l2_leaf_reg': 25.45826716286499, 'random_strength': 0.013885142609872152, 'subsample': 0.7988751051070264, 'rsm': 0.7627876416679646, 'min_data_in_leaf': 114, 'model_size_reg': 0.29819581601468137, 'leaf_estimation_iterations': 10} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:38:38,640] Trial 15 failed with value None.
[W 2026-05-01 10:39:29,196] Trial 23 failed with parameters: {'iterations': 545, 'learning_rate': 0.03647352740430349, 'depth': 6, 'l2_leaf_reg': 1.3911077228496058, 'random_strength': 0.026784978448683532, 'subsample': 0.9569841955447176, 'rsm': 0.37628701372100004, 'min_data_in_leaf': 163, 'model_size_reg': 5.1303448534869105, 'leaf_estimation_iterations': 1} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:39:29,198] Trial 23 failed with value None.
[W 2026-05-01 10:39:54,996] Trial 24 failed with parameters: {'iterations': 513, 'learning_rate': 0.02067788502180789, 'depth': 4, 'l2_leaf_reg': 1.6860842001387524, 'random_strength': 0.29123918123177217, 'subsample': 0.9266233652156453, 'rsm': 0.8472846364701343, 'min_data_in_leaf': 66, 'model_size_reg': 0.13909750428004353, 'leaf_estimation_iterations': 2} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:39:54,998] Trial 24 failed with value None.
[W 2026-05-01 10:40:42,779] Trial 30 failed with parameters: {'iterations': 260, 'learning_rate': 0.012401289120025489, 'depth': 6, 'l2_leaf_reg': 1.5490109708209168, 'random_strength': 0.141233529553505, 'subsample': 0.9254076903565691, 'rsm': 0.4470717293521606, 'min_data_in_leaf': 10, 'model_size_reg': 9.297539931840124, 'leaf_estimation_iterations': 5} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:40:42,785] Trial 30 failed with value None.
[W 2026-05-01 10:40:47,489] Trial 20 failed with parameters: {'iterations': 461, 'learning_rate': 0.13794706392728637, 'depth': 6, 'l2_leaf_reg': 8.694596919746308, 'random_strength': 0.018085593382159586, 'subsample': 0.8536632480976123, 'rsm': 0.7217275390684672, 'min_data_in_leaf': 88, 'model_size_reg': 3.7500701347013896, 'leaf_estimation_iterations': 6} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:40:47,492] Trial 20 failed with value None.
[W 2026-05-01 10:40:51,871] Trial 25 failed with parameters: {'iterations': 437, 'learning_rate': 0.04574188469963848, 'depth': 6, 'l2_leaf_reg': 2.558242345679237, 'random_strength': 0.6394983747161817, 'subsample': 0.7393337497140815, 'rsm': 0.8941738945914561, 'min_data_in_leaf': 183, 'model_size_reg': 3.4048925648214534, 'leaf_estimation_iterations': 1} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:40:51,873] Trial 25 failed with value None.
[W 2026-05-01 10:41:05,920] Trial 27 failed with parameters: {'iterations': 377, 'learning_rate': 0.055660690841442546, 'depth': 8, 'l2_leaf_reg': 2.321926278832008, 'random_strength': 4.004929841632015, 'subsample': 0.9214985626160811, 'rsm': 0.4805758509792273, 'min_data_in_leaf': 186, 'model_size_reg': 0.26979721221187414, 'leaf_estimation_iterations': 9} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:41:05,922] Trial 27 failed with value None.
[W 2026-05-01 10:41:06,086] Trial 21 failed with parameters: {'iterations': 531, 'learning_rate': 0.011685401315765495, 'depth': 7, 'l2_leaf_reg': 12.512755563090044, 'random_strength': 3.5354492607779493, 'subsample': 0.7503092198188994, 'rsm': 0.8751498294974636, 'min_data_in_leaf': 128, 'model_size_reg': 9.565477769825918, 'leaf_estimation_iterations': 9} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:41:06,088] Trial 21 failed with value None.
[W 2026-05-01 10:42:02,195] Trial 31 failed with parameters: {'iterations': 283, 'learning_rate': 0.04296875468238988, 'depth': 7, 'l2_leaf_reg': 4.597813836446807, 'random_strength': 0.06233930003931802, 'subsample': 0.6403670110647992, 'rsm': 0.4579349512668147, 'min_data_in_leaf': 64, 'model_size_reg': 6.368393584318443, 'leaf_estimation_iterations': 6} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:42:02,199] Trial 31 failed with value None.
[W 2026-05-01 10:43:20,395] Trial 29 failed with parameters: {'iterations': 327, 'learning_rate': 0.12881413988157084, 'depth': 7, 'l2_leaf_reg': 6.288004932116509, 'random_strength': 0.01161301498128327, 'subsample': 0.7837075382568499, 'rsm': 0.7453836913219698, 'min_data_in_leaf': 108, 'model_size_reg': 4.165824501883826, 'leaf_estimation_iterations': 8} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:43:20,399] Trial 29 failed with value None.
[W 2026-05-01 10:43:30,050] Trial 33 failed with parameters: {'iterations': 256, 'learning_rate': 0.050284249923873064, 'depth': 7, 'l2_leaf_reg': 5.026418257474655, 'random_strength': 0.32679761161992676, 'subsample': 0.6211417089628929, 'rsm': 0.7192613695060931, 'min_data_in_leaf': 25, 'model_size_reg': 5.260252636102923, 'leaf_estimation_iterations': 9} because of the following error: CatBoostError("'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 782, in _objective_all_classification
    proba_val = model.predict_proba(X_mat[va])[:, 1]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
    data = Pool(
           ^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
    raise CatBoostError(
_catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
[W 2026-05-01 10:43:30,053] Trial 33 failed with value None.
