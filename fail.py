[rubin] Step 1/11: Daten laden & Preprocessing
[rubin] Step 2/11: Feature-Selektion
[rubin] Step 3/11: Base-Learner-Tuning
[rubin] Step 4/11: Final-Model-Tuning
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
[W 2026-07-14 09:41:28,872] Trial 83 failed with value None.
[W 2026-07-14 09:41:39,383] Trial 84 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 258, 'learning_rate': 0.005342424611958841, 'num_leaves': 57, 'max_depth': 3, 'min_child_samples': 150, 'min_child_weight': 11.63821337182118, 'subsample': 0.684559622002304, 'colsample_bytree': 0.576438365738404, 'max_bin': 21, 'min_split_gain': 0.05143736973449803, 'reg_alpha': 1.831672222459123, 'reg_lambda': 0.557113455714402, 'path_smooth': 4.865434439174881} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:41:39,384] Trial 84 failed with value None.
[W 2026-07-14 09:41:40,164] Trial 85 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 152, 'learning_rate': 0.00832644828473533, 'num_leaves': 42, 'max_depth': 2, 'min_child_samples': 92, 'min_child_weight': 11.43060062396676, 'subsample': 0.633460241031909, 'colsample_bytree': 0.6357001713719101, 'max_bin': 112, 'min_split_gain': 0.14928431277213278, 'reg_alpha': 1.7835328138030238, 'reg_lambda': 4.394989037595645, 'path_smooth': 4.113778460978618} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:41:40,165] Trial 85 failed with value None.
[W 2026-07-14 09:41:44,012] Trial 86 failed with parameters: {'_learner_type': 'catboost', 'iterations': 218, 'learning_rate': 0.07154938573838597, 'depth': 6, 'l2_leaf_reg': 12.261709186771043, 'random_strength': 2.2170764437420036, 'subsample': 0.5974510208905548, 'rsm': 0.4868112087472867, 'min_data_in_leaf': 71, 'model_size_reg': 7.031927889479901, 'leaf_estimation_iterations': 3} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:41:44,013] Trial 86 failed with value None.
[W 2026-07-14 09:41:50,140] Trial 87 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 351, 'learning_rate': 0.05902993322580122, 'num_leaves': 10, 'max_depth': 4, 'min_child_samples': 77, 'min_child_weight': 0.7865347602389063, 'subsample': 0.5551645022953958, 'colsample_bytree': 0.4990159252604238, 'max_bin': 71, 'min_split_gain': 0.23464578766609984, 'reg_alpha': 7.53176159598538, 'reg_lambda': 2.2341502029742166, 'path_smooth': 1.0165807278894157} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:41:50,141] Trial 87 failed with value None.
[W 2026-07-14 09:41:52,820] Trial 88 failed with parameters: {'_learner_type': 'catboost', 'iterations': 165, 'learning_rate': 0.03744648140923126, 'depth': 6, 'l2_leaf_reg': 18.600019149038975, 'random_strength': 7.755001024738309, 'subsample': 0.5441439881367471, 'rsm': 0.41888033604429364, 'min_data_in_leaf': 37, 'model_size_reg': 9.618602867588182, 'leaf_estimation_iterations': 3} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:41:52,821] Trial 88 failed with value None.
[W 2026-07-14 09:41:56,708] Trial 89 failed with parameters: {'_learner_type': 'catboost', 'iterations': 251, 'learning_rate': 0.03051203244792356, 'depth': 5, 'l2_leaf_reg': 16.195193546039466, 'random_strength': 1.5786838754177324, 'subsample': 0.6696236793525625, 'rsm': 0.33142610491878033, 'min_data_in_leaf': 38, 'model_size_reg': 6.553857268775055, 'leaf_estimation_iterations': 4} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:41:56,710] Trial 89 failed with value None.
[W 2026-07-14 09:41:59,632] Trial 90 failed with parameters: {'_learner_type': 'catboost', 'iterations': 177, 'learning_rate': 0.024344249491385188, 'depth': 6, 'l2_leaf_reg': 29.254961136037007, 'random_strength': 7.931022666659377, 'subsample': 0.6673618509047163, 'rsm': 0.6122754675899579, 'min_data_in_leaf': 136, 'model_size_reg': 4.542160501959256, 'leaf_estimation_iterations': 3} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:41:59,633] Trial 90 failed with value None.
[W 2026-07-14 09:42:03,108] Trial 91 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 209, 'learning_rate': 0.009729000822629135, 'num_leaves': 43, 'max_depth': 3, 'min_child_samples': 110, 'min_child_weight': 15.400574354167029, 'subsample': 0.5658013759037179, 'colsample_bytree': 0.6903196796748221, 'max_bin': 20, 'min_split_gain': 0.12860820852153287, 'reg_alpha': 3.1356889638330387, 'reg_lambda': 0.18378472962317982, 'path_smooth': 3.1819271490678407} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:42:03,109] Trial 91 failed with value None.
[W 2026-07-14 09:42:05,807] Trial 92 failed with parameters: {'_learner_type': 'catboost', 'iterations': 275, 'learning_rate': 0.012099642485062217, 'depth': 2, 'l2_leaf_reg': 15.14641937277781, 'random_strength': 9.324594169534159, 'subsample': 0.7726117534356054, 'rsm': 0.6791993681593416, 'min_data_in_leaf': 157, 'model_size_reg': 4.447843730322917, 'leaf_estimation_iterations': 2} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:42:05,807] Trial 92 failed with value None.
[W 2026-07-14 09:42:07,725] Trial 93 failed with parameters: {'_learner_type': 'catboost', 'iterations': 311, 'learning_rate': 0.007862926294718475, 'depth': 2, 'l2_leaf_reg': 11.25024466580608, 'random_strength': 4.768998368460428, 'subsample': 0.656818928581058, 'rsm': 0.3071288447734311, 'min_data_in_leaf': 69, 'model_size_reg': 4.743930594572974, 'leaf_estimation_iterations': 1} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:42:07,726] Trial 93 failed with value None.
[W 2026-07-14 09:42:21,524] Trial 94 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 344, 'learning_rate': 0.03485112440891032, 'num_leaves': 51, 'max_depth': 6, 'min_child_samples': 22, 'min_child_weight': 4.141301079571376, 'subsample': 0.8281702217911888, 'colsample_bytree': 0.627239725236987, 'max_bin': 83, 'min_split_gain': 0.40248664491477104, 'reg_alpha': 7.005488143702979, 'reg_lambda': 0.8805809673955023, 'path_smooth': 3.5943524371057283} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:42:21,525] Trial 94 failed with value None.
[W 2026-07-14 09:42:25,404] Trial 95 failed with parameters: {'_learner_type': 'catboost', 'iterations': 317, 'learning_rate': 0.05559355000878084, 'depth': 4, 'l2_leaf_reg': 7.445751156258468, 'random_strength': 4.331671089948085, 'subsample': 0.4528229176780879, 'rsm': 0.5501161602792778, 'min_data_in_leaf': 25, 'model_size_reg': 1.5727805029615334, 'leaf_estimation_iterations': 3} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:42:25,405] Trial 95 failed with value None.
[W 2026-07-14 09:42:26,769] Trial 96 failed with parameters: {'_learner_type': 'catboost', 'iterations': 166, 'learning_rate': 0.00799296492481218, 'depth': 3, 'l2_leaf_reg': 25.93882465031478, 'random_strength': 4.5796270141037265, 'subsample': 0.5731625219917647, 'rsm': 0.627311415495287, 'min_data_in_leaf': 124, 'model_size_reg': 1.8968614144931888, 'leaf_estimation_iterations': 1} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:42:26,770] Trial 96 failed with value None.
[W 2026-07-14 09:42:32,240] Trial 97 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 344, 'learning_rate': 0.020710554071391572, 'num_leaves': 30, 'max_depth': 6, 'min_child_samples': 180, 'min_child_weight': 0.723604706739454, 'subsample': 0.4776531404044284, 'colsample_bytree': 0.3398766218232308, 'max_bin': 80, 'min_split_gain': 0.059552854430397095, 'reg_alpha': 4.6032042776284365, 'reg_lambda': 9.476559429416746, 'path_smooth': 0.8578785575770825} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:42:32,241] Trial 97 failed with value None.
[W 2026-07-14 09:42:42,732] Trial 98 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 389, 'learning_rate': 0.005700182741157331, 'num_leaves': 47, 'max_depth': 6, 'min_child_samples': 82, 'min_child_weight': 12.808062152593235, 'subsample': 0.792048004949595, 'colsample_bytree': 0.45846201717051205, 'max_bin': 79, 'min_split_gain': 0.48263277837365137, 'reg_alpha': 0.48195158664109505, 'reg_lambda': 7.211913675304831, 'path_smooth': 3.3482347421588576} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:42:42,733] Trial 98 failed with value None.
[W 2026-07-14 09:42:44,618] Trial 99 failed with parameters: {'_learner_type': 'catboost', 'iterations': 199, 'learning_rate': 0.007419139317491683, 'depth': 5, 'l2_leaf_reg': 5.772750983819799, 'random_strength': 8.405295636699531, 'subsample': 0.8070112824425808, 'rsm': 0.5033978761270418, 'min_data_in_leaf': 51, 'model_size_reg': 2.890880216121692, 'leaf_estimation_iterations': 1} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:42:44,618] Trial 99 failed with value None.
[W 2026-07-14 09:42:48,124] Trial 100 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 215, 'learning_rate': 0.03761539568335726, 'num_leaves': 42, 'max_depth': 3, 'min_child_samples': 113, 'min_child_weight': 18.73409639672268, 'subsample': 0.7416319966755629, 'colsample_bytree': 0.6369202501441523, 'max_bin': 18, 'min_split_gain': 0.4998035107296122, 'reg_alpha': 4.581349682748593, 'reg_lambda': 9.988343395328558, 'path_smooth': 3.568847206150354} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:42:48,125] Trial 100 failed with value None.
[W 2026-07-14 09:42:51,247] Trial 101 failed with parameters: {'_learner_type': 'catboost', 'iterations': 217, 'learning_rate': 0.058838665323908586, 'depth': 2, 'l2_leaf_reg': 22.21778970613603, 'random_strength': 8.36839628861016, 'subsample': 0.731811205829186, 'rsm': 0.3965084121789939, 'min_data_in_leaf': 151, 'model_size_reg': 6.362609904078935, 'leaf_estimation_iterations': 5} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:42:51,248] Trial 101 failed with value None.
[W 2026-07-14 09:42:54,591] Trial 102 failed with parameters: {'_learner_type': 'catboost', 'iterations': 227, 'learning_rate': 0.012548934809742571, 'depth': 4, 'l2_leaf_reg': 12.16349324356715, 'random_strength': 4.241988364171785, 'subsample': 0.5651141060834183, 'rsm': 0.6799120581841838, 'min_data_in_leaf': 60, 'model_size_reg': 1.7046876965672757, 'leaf_estimation_iterations': 4} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:42:54,592] Trial 102 failed with value None.
[W 2026-07-14 09:42:59,181] Trial 103 failed with parameters: {'_learner_type': 'lgbm', 'n_estimators': 189, 'learning_rate': 0.018674744942135866, 'num_leaves': 38, 'max_depth': 4, 'min_child_samples': 54, 'min_child_weight': 13.379840778541006, 'subsample': 0.401292416367342, 'colsample_bytree': 0.48752864662664597, 'max_bin': 16, 'min_split_gain': 0.46016489960638146, 'reg_alpha': 6.3114195533630095, 'reg_lambda': 3.778835253925518, 'path_smooth': 3.0667742962919915} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:42:59,182] Trial 103 failed with value None.
[W 2026-07-14 09:43:03,444] Trial 104 failed with parameters: {'_learner_type': 'catboost', 'iterations': 259, 'learning_rate': 0.06445732238874878, 'depth': 6, 'l2_leaf_reg': 13.524707936565772, 'random_strength': 3.720200131245407, 'subsample': 0.648673290360912, 'rsm': 0.356704871555437, 'min_data_in_leaf': 142, 'model_size_reg': 5.540273066496275, 'leaf_estimation_iterations': 3} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:43:03,445] Trial 104 failed with value None.
[W 2026-07-14 09:43:06,524] Trial 105 failed with parameters: {'_learner_type': 'catboost', 'iterations': 263, 'learning_rate': 0.049851410636535996, 'depth': 2, 'l2_leaf_reg': 17.78629100621972, 'random_strength': 4.992848068917379, 'subsample': 0.612695883656602, 'rsm': 0.47268254398066334, 'min_data_in_leaf': 157, 'model_size_reg': 0.16682991137908249, 'leaf_estimation_iterations': 4} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:43:06,525] Trial 105 failed with value None.
[W 2026-07-14 09:43:10,442] Trial 106 failed with parameters: {'_learner_type': 'catboost', 'iterations': 209, 'learning_rate': 0.05865493015637415, 'depth': 6, 'l2_leaf_reg': 14.549583718172546, 'random_strength': 2.09168892693398, 'subsample': 0.7240103534690832, 'rsm': 0.3769994941605117, 'min_data_in_leaf': 111, 'model_size_reg': 9.010445246698605, 'leaf_estimation_iterations': 4} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:43:10,442] Trial 106 failed with value None.
[W 2026-07-14 09:43:14,674] Trial 107 failed with parameters: {'_learner_type': 'catboost', 'iterations': 264, 'learning_rate': 0.02245294169868934, 'depth': 5, 'l2_leaf_reg': 28.75728313504965, 'random_strength': 3.73486152879878, 'subsample': 0.7155845150857176, 'rsm': 0.534493969430692, 'min_data_in_leaf': 33, 'model_size_reg': 0.8480841108977315, 'leaf_estimation_iterations': 5} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:43:14,675] Trial 107 failed with value None.
[W 2026-07-14 09:43:17,199] Trial 108 failed with parameters: {'_learner_type': 'catboost', 'iterations': 304, 'learning_rate': 0.01989430935272381, 'depth': 5, 'l2_leaf_reg': 23.411672637275835, 'random_strength': 9.070942636474529, 'subsample': 0.800770180343948, 'rsm': 0.3545289416189952, 'min_data_in_leaf': 147, 'model_size_reg': 9.12181353371468, 'leaf_estimation_iterations': 1} because of the following error: ValueError('t muss binär (0/1) sein, gefunden: [0 1 2]').
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
[W 2026-07-14 09:43:17,200] Trial 108 failed with value None.
