12:55:48 INFO [rubin.analysis]   both__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y best=-1.37788e-07  skill=1.0000
12:55:48 INFO [rubin.analysis]   both__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y__adjusted best=-1.58982e-07  skill=0.0000
12:55:48 INFO [rubin.analysis]   both__outcome__classifier__all__no_t__y  best=-4.56809e-06  skill=1.0000
12:55:48 INFO [rubin.analysis]   both__outcome__classifier__all__with_t__y best=-3.13619e-05  skill=0.9999
12:55:48 INFO [rubin.analysis]   both__propensity__classifier__all__no_t__t best=-0.849901  skill=0.2264
12:55:48 INFO [rubin.analysis]   both__propensity__classifier__all_direct__no_t__t best=-0.853308  skill=0.2233
12:55:48 INFO [rubin.analysis]   both__pseudo_effect__regressor__group_specific_shared_params__no_t__d best=-3.89162e-07  skill=1.0000
12:55:48 INFO [rubin.analysis]   both__pseudo_effect__regressor__group_specific_shared_params__no_t__d__adjusted best=-4.02828e-07  skill=0.0000
12:55:48 WARNING [rubin.analysis] RCT-Diagnose: Propensity-Skill = 0.2264 > 0.01. Das Propensity-Modell kann Treatment besser als Zufall vorhersagen — die Daten sind möglicherweise NICHT randomisiert. Prüfe die Treatment-Zuweisung. Training verwendet trotzdem konstante Propensity (DummyClassifier), wie für RCT konfiguriert.
12:55:50 WARNING [rubin.tuning] base_type='both' aber kein '_learner_type' in params — Fallback auf 'catboost' (globaler Default). Das deutet auf ein fehlendes Tuning oder eine Rolle ohne getunte Params hin.
12:55:51 INFO [rubin.categorical] CatBoost categorical patch (predict): 5 Spalten float→int konvertiert.
12:56:16 ERROR [rubin.analysis] FMT 'NonParamDML' fehlgeschlagen — bisherige Ergebnisse bleiben erhalten. Modell wird mit Default-/BLT-Parametern trainiert.
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/rubin/pipelines/analysis_pipeline.py", line 645, in _run_tuning
    add = final_tuner.tune_final_model(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 449, in tune_final_model
    est.fit(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr], cache_values=True)
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 1703, in fit
    return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 501, in fit
    return super().fit(Y, T, X=X, W=W,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_cate_estimator.py", line 134, in call
    m(self, Y, T, *args, **kwargs)
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 876, in fit
    self._fit_final(Y=Y,
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 979, in _fit_final
    self._ortho_learner_model_final.fit(Y, T, **filter_none_kwargs(X=X, W=W, Z=Z,
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 101, in fit
    self._model_final.fit(X, T, T_res, Y_res, sample_weight=sample_weight,
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 198, in fit
    raise AttributeError("This method can only be used with single-dimensional continuous treatment "
AttributeError: This method can only be used with single-dimensional continuous treatment or binary categorical treatment.
12:56:16 WARNING [rubin.tuning] base_type='both' aber kein '_learner_type' in params — Fallback auf 'catboost' (globaler Default). Das deutet auf ein fehlendes Tuning oder eine Rolle ohne getunte Params hin.
12:57:07 WARNING [rubin.tuning] base_type='both' aber kein '_learner_type' in params — Fallback auf 'catboost' (globaler Default). Das deutet auf ein fehlendes Tuning oder eine Rolle ohne getunte Params hin.
12:57:59 WARNING [rubin.tuning] base_type='both' aber kein '_learner_type' in params — Fallback auf 'catboost' (globaler Default). Das deutet auf ein fehlendes Tuning oder eine Rolle ohne getunte Params hin.
12:58:51 WARNING [rubin.tuning] base_type='both' aber kein '_learner_type' in params — Fallback auf 'catboost' (globaler Default). Das deutet auf ein fehlendes Tuning oder eine Rolle ohne getunte Params hin.
12:59:43 WARNING [rubin.tuning] base_type='both' aber kein '_learner_type' in params — Fallback auf 'catboost' (globaler Default). Das deutet auf ein fehlendes Tuning oder eine Rolle ohne getunte Params hin.


[rubin] Step 1/11: Daten laden & Preprocessing
[rubin] Step 2/11: Feature-Selektion
[rubin] Step 3/11: Base-Learner-Tuning
[rubin] Step 4/11: Final-Model-Tuning
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:12:35,785] Trial 94 failed with value None.
[W 2026-07-13 13:12:39,587] Trial 95 failed with parameters: {'_learner_type': 'catboost', 'iterations': 317, 'learning_rate': 0.05559355000878084, 'depth': 4, 'l2_leaf_reg': 7.445751156258468, 'random_strength': 4.331671089948085, 'subsample': 0.4528229176780879, 'rsm': 0.5501161602792778, 'min_data_in_leaf': 25, 'model_size_reg': 1.5727805029615334, 'leaf_estimation_iterations': 3} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:12:39,588] Trial 95 failed with value None.
[W 2026-07-13 13:12:41,015] Trial 96 failed with parameters: {'_learner_type': 'catboost', 'iterations': 166, 'learning_rate': 0.00799296492481218, 'depth': 3, 'l2_leaf_reg': 25.93882465031478, 'random_strength': 4.5796270141037265, 'subsample': 0.5731625219917647, 'rsm': 0.627311415495287, 'min_data_in_leaf': 124, 'model_size_reg': 1.8968614144931888, 'leaf_estimation_iterations': 1} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:12:41,016] Trial 96 failed with value None.
[W 2026-07-13 13:13:09,710] Trial 97 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 344, 'learning_rate': 0.020710554071391572, 'num_leaves': 30, 'max_depth': 6, 'min_child_samples': 180, 'min_child_weight': 0.723604706739454, 'subsample': 0.4776531404044284, 'colsample_bytree': 0.3398766218232308, 'max_bin': 80, 'min_split_gain': 0.059552854430397095, 'reg_alpha': 4.6032042776284365, 'reg_lambda': 9.476559429416746, 'path_smooth': 0.8578785575770825} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:13:09,712] Trial 97 failed with value None.
[W 2026-07-13 13:13:34,897] Trial 98 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 389, 'learning_rate': 0.005700182741157331, 'num_leaves': 47, 'max_depth': 6, 'min_child_samples': 82, 'min_child_weight': 12.808062152593235, 'subsample': 0.792048004949595, 'colsample_bytree': 0.45846201717051205, 'max_bin': 79, 'min_split_gain': 0.48263277837365137, 'reg_alpha': 0.48195158664109505, 'reg_lambda': 7.211913675304831, 'path_smooth': 3.3482347421588576} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:13:34,898] Trial 98 failed with value None.
[W 2026-07-13 13:13:36,902] Trial 99 failed with parameters: {'_learner_type': 'catboost', 'iterations': 199, 'learning_rate': 0.007419139317491683, 'depth': 5, 'l2_leaf_reg': 5.772750983819799, 'random_strength': 8.405295636699531, 'subsample': 0.8070112824425808, 'rsm': 0.5033978761270418, 'min_data_in_leaf': 51, 'model_size_reg': 2.890880216121692, 'leaf_estimation_iterations': 1} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:13:36,903] Trial 99 failed with value None.
[W 2026-07-13 13:13:41,328] Trial 100 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 215, 'learning_rate': 0.03761539568335726, 'num_leaves': 42, 'max_depth': 3, 'min_child_samples': 113, 'min_child_weight': 18.73409639672268, 'subsample': 0.7416319966755629, 'colsample_bytree': 0.6369202501441523, 'max_bin': 18, 'min_split_gain': 0.4998035107296122, 'reg_alpha': 4.581349682748593, 'reg_lambda': 9.988343395328558, 'path_smooth': 3.568847206150354} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:13:41,328] Trial 100 failed with value None.
[W 2026-07-13 13:13:44,240] Trial 101 failed with parameters: {'_learner_type': 'catboost', 'iterations': 217, 'learning_rate': 0.058838665323908586, 'depth': 2, 'l2_leaf_reg': 22.21778970613603, 'random_strength': 8.36839628861016, 'subsample': 0.731811205829186, 'rsm': 0.3965084121789939, 'min_data_in_leaf': 151, 'model_size_reg': 6.362609904078935, 'leaf_estimation_iterations': 5} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:13:44,241] Trial 101 failed with value None.
[W 2026-07-13 13:13:47,517] Trial 102 failed with parameters: {'_learner_type': 'catboost', 'iterations': 227, 'learning_rate': 0.012548934809742571, 'depth': 4, 'l2_leaf_reg': 12.16349324356715, 'random_strength': 4.241988364171785, 'subsample': 0.5651141060834183, 'rsm': 0.6799120581841838, 'min_data_in_leaf': 60, 'model_size_reg': 1.7046876965672757, 'leaf_estimation_iterations': 4} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:13:47,518] Trial 102 failed with value None.
[W 2026-07-13 13:13:52,105] Trial 103 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 189, 'learning_rate': 0.018674744942135866, 'num_leaves': 38, 'max_depth': 4, 'min_child_samples': 54, 'min_child_weight': 13.379840778541006, 'subsample': 0.401292416367342, 'colsample_bytree': 0.48752864662664597, 'max_bin': 16, 'min_split_gain': 0.46016489960638146, 'reg_alpha': 6.3114195533630095, 'reg_lambda': 3.778835253925518, 'path_smooth': 3.0667742962919915} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:13:52,106] Trial 103 failed with value None.
[W 2026-07-13 13:13:56,448] Trial 104 failed with parameters: {'_learner_type': 'catboost', 'iterations': 259, 'learning_rate': 0.06445732238874878, 'depth': 6, 'l2_leaf_reg': 13.524707936565772, 'random_strength': 3.720200131245407, 'subsample': 0.648673290360912, 'rsm': 0.356704871555437, 'min_data_in_leaf': 142, 'model_size_reg': 5.540273066496275, 'leaf_estimation_iterations': 3} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:13:56,449] Trial 104 failed with value None.
[W 2026-07-13 13:13:59,510] Trial 105 failed with parameters: {'_learner_type': 'catboost', 'iterations': 263, 'learning_rate': 0.049851410636535996, 'depth': 2, 'l2_leaf_reg': 17.78629100621972, 'random_strength': 4.992848068917379, 'subsample': 0.612695883656602, 'rsm': 0.47268254398066334, 'min_data_in_leaf': 157, 'model_size_reg': 0.16682991137908249, 'leaf_estimation_iterations': 4} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:13:59,510] Trial 105 failed with value None.
[W 2026-07-13 13:14:03,680] Trial 106 failed with parameters: {'_learner_type': 'catboost', 'iterations': 209, 'learning_rate': 0.05865493015637415, 'depth': 6, 'l2_leaf_reg': 14.549583718172546, 'random_strength': 2.09168892693398, 'subsample': 0.7240103534690832, 'rsm': 0.3769994941605117, 'min_data_in_leaf': 111, 'model_size_reg': 9.010445246698605, 'leaf_estimation_iterations': 4} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:14:03,681] Trial 106 failed with value None.
[W 2026-07-13 13:14:08,054] Trial 107 failed with parameters: {'_learner_type': 'catboost', 'iterations': 264, 'learning_rate': 0.02245294169868934, 'depth': 5, 'l2_leaf_reg': 28.75728313504965, 'random_strength': 3.73486152879878, 'subsample': 0.7155845150857176, 'rsm': 0.534493969430692, 'min_data_in_leaf': 33, 'model_size_reg': 0.8480841108977315, 'leaf_estimation_iterations': 5} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:14:08,055] Trial 107 failed with value None.
[W 2026-07-13 13:14:10,721] Trial 108 failed with parameters: {'_learner_type': 'catboost', 'iterations': 304, 'learning_rate': 0.01989430935272381, 'depth': 5, 'l2_leaf_reg': 23.411672637275835, 'random_strength': 9.070942636474529, 'subsample': 0.800770180343948, 'rsm': 0.3545289416189952, 'min_data_in_leaf': 147, 'model_size_reg': 9.12181353371468, 'leaf_estimation_iterations': 1} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:14:10,722] Trial 108 failed with value None.
[W 2026-07-13 13:14:25,632] Trial 109 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 384, 'learning_rate': 0.007581029573432681, 'num_leaves': 23, 'max_depth': 3, 'min_child_samples': 78, 'min_child_weight': 16.278485013384028, 'subsample': 0.4693634451444456, 'colsample_bytree': 0.4649898058786237, 'max_bin': 48, 'min_split_gain': 0.27475763672707526, 'reg_alpha': 1.855247395043319, 'reg_lambda': 5.371468456287239, 'path_smooth': 2.704897848925145} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:14:25,633] Trial 109 failed with value None.
[W 2026-07-13 13:14:29,134] Trial 110 failed with parameters: {'_learner_type': 'catboost', 'iterations': 369, 'learning_rate': 0.029575999258994862, 'depth': 3, 'l2_leaf_reg': 11.375217609351717, 'random_strength': 4.062619195770715, 'subsample': 0.4816523465204922, 'rsm': 0.6359154799183567, 'min_data_in_leaf': 29, 'model_size_reg': 9.282896850563635, 'leaf_estimation_iterations': 2} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:14:29,135] Trial 110 failed with value None.
[W 2026-07-13 13:14:34,627] Trial 111 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 257, 'learning_rate': 0.10424540659577057, 'num_leaves': 15, 'max_depth': 6, 'min_child_samples': 132, 'min_child_weight': 13.875669965714811, 'subsample': 0.7639967008046415, 'colsample_bytree': 0.6109861498422853, 'max_bin': 91, 'min_split_gain': 0.2603937404511799, 'reg_alpha': 0.896223812993745, 'reg_lambda': 2.3498519185775715, 'path_smooth': 1.6378572099651134} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:14:34,628] Trial 111 failed with value None.
[W 2026-07-13 13:14:37,471] Trial 112 failed with parameters: {'_learner_type': 'catboost', 'iterations': 176, 'learning_rate': 0.02093925160498015, 'depth': 5, 'l2_leaf_reg': 6.276549112038619, 'random_strength': 5.839460226119051, 'subsample': 0.4729637289435739, 'rsm': 0.6541617118237643, 'min_data_in_leaf': 114, 'model_size_reg': 7.284927337340933, 'leaf_estimation_iterations': 4} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:14:37,472] Trial 112 failed with value None.
[W 2026-07-13 13:14:38,583] Trial 113 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 123, 'learning_rate': 0.030014295810306003, 'num_leaves': 39, 'max_depth': 5, 'min_child_samples': 53, 'min_child_weight': 8.299141446912717, 'subsample': 0.5446687435001214, 'colsample_bytree': 0.6471573617417903, 'max_bin': 54, 'min_split_gain': 0.3899660813083193, 'reg_alpha': 7.240338566453039, 'reg_lambda': 4.9961653144425435, 'path_smooth': 0.7686820757833263} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:14:38,584] Trial 113 failed with value None.
[W 2026-07-13 13:14:40,185] Trial 114 failed with parameters: {'_learner_type': 'catboost', 'iterations': 104, 'learning_rate': 0.0492575641282782, 'depth': 2, 'l2_leaf_reg': 12.726091555617087, 'random_strength': 7.615992021834252, 'subsample': 0.6282811548849007, 'rsm': 0.4443889064672703, 'min_data_in_leaf': 54, 'model_size_reg': 6.369416120476384, 'leaf_estimation_iterations': 5} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:14:40,186] Trial 114 failed with value None.
[W 2026-07-13 13:14:43,586] Trial 115 failed with parameters: {'_learner_type': 'catboost', 'iterations': 265, 'learning_rate': 0.07982913147133792, 'depth': 3, 'l2_leaf_reg': 19.391331821509475, 'random_strength': 9.641878034168005, 'subsample': 0.6857121792485654, 'rsm': 0.46022708572352933, 'min_data_in_leaf': 153, 'model_size_reg': 5.546667526085232, 'leaf_estimation_iterations': 4} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:14:43,587] Trial 115 failed with value None.
[W 2026-07-13 13:14:47,730] Trial 116 failed with parameters: {'_learner_type': 'catboost', 'iterations': 319, 'learning_rate': 0.058315653573220584, 'depth': 4, 'l2_leaf_reg': 24.74395600879494, 'random_strength': 6.5606215640024, 'subsample': 0.40019090956777165, 'rsm': 0.32622997533490256, 'min_data_in_leaf': 113, 'model_size_reg': 6.823974501023309, 'leaf_estimation_iterations': 4} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 230, in objective
    fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 78, in uplift_curve
    _check_binary(t, "t")
  File "/home/ubuntu/da-hf1-rubin/rubin/evaluation/uplift_metrics.py", line 38, in _check_binary
    raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")
ValueError: t muss binär (0/1) sein, gefunden: [0 1 2]
[W 2026-07-13 13:14:47,731] Trial 116 failed with value None.
[W 2026-07-13 13:14:54,608] Trial 117 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 306, 'learning_rate': 0.06955069919279144, 'num_leaves': 11, 'max_depth': 3, 'min_child_samples': 37, 'min_child_weight': 19.738258144846267, 'subsample': 0.5966767827428744, 'colsample_bytree': 0.4475821086474504, 'max_bin': 25, 'min_split_gain': 0.16850886626890288, 'reg_alpha': 9.701500901784822, 'reg_lamb




