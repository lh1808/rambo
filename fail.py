Analyse fehlgeschlagen: Fehlgeschlagen (Exit 1)

Details:
[W 2026-09-11 11:46:42,418] Trial 17 failed with value None.
[W 2026-09-11 11:46:42,426] Trial 18 failed with parameters: {'iterations': 525, 'learning_rate': 0.01326983928862889, 'depth': 8, 'l2_leaf_reg': 13.811880945559109, 'random_strength': 1.4587053224071307, 'subsample': 0.9486585545311248, 'rsm': 0.39716538435125737, 'min_data_in_leaf': 93, 'model_size_reg': 6.251389534079896, 'leaf_estimation_iterations': 1} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:42,428] Trial 18 failed with value None.
[W 2026-09-11 11:46:42,432] Trial 19 failed with parameters: {'iterations': 453, 'learning_rate': 0.04207966278963679, 'depth': 7, 'l2_leaf_reg': 23.41446329570275, 'random_strength': 2.6117489948690955, 'subsample': 0.7160130010382724, 'rsm': 0.5782021245133502, 'min_data_in_leaf': 33, 'model_size_reg': 3.7299113874557754, 'leaf_estimation_iterations': 7} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:42,435] Trial 19 failed with value None.
11:46:42 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all__no_t__t': 0/20 Trials abgeschlossen (20 fehlgeschlagen, 0 gepruned, parallel=4).
11:46:42 WARNING [rubin.tuning] BLT 'catboost__propensity__classifier__all__no_t__t': 20/20 Trials FEHLGESCHLAGEN. Fehlertypen:
11:46:42 WARNING [rubin.tuning]   [20×] Unbekannt
11:46:42 WARNING [rubin.tuning] BLT 'catboost__propensity__classifier__all__no_t__t': Häufigster Fehler — vollständiger Traceback:
Unbekannt
11:46:42 WARNING [rubin.tuning] BLT 'catboost__propensity__classifier__all__no_t__t': Keine abgeschlossenen Trials. Verwende Default-Parameter.
11:46:42 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all__no_t__t': Study freigegeben, gc.collect() + malloc_trim durchgeführt.
11:46:42 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all_direct__no_t__t': X=(163502, 77), target=(163502,) (unique=[0, 1]), subsample=100%, cv=5, objective=propensity
11:46:42 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all_direct__no_t__t': Starte 20 Trials (parallel=4, 5-Fold, n_jobs=64).
[W 2026-09-11 11:46:42,867] Trial 3 failed with parameters: {'iterations': 347, 'learning_rate': 0.02507984888404308, 'depth': 4, 'l2_leaf_reg': 29.666571802672102, 'random_strength': 9.685349794223269, 'subsample': 0.5519990798842773, 'rsm': 0.7043370672001181, 'min_data_in_leaf': 134, 'model_size_reg': 8.733803310880102, 'leaf_estimation_iterations': 8} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:42,869] Trial 3 failed with value None.
[W 2026-09-11 11:46:42,886] Trial 2 failed with parameters: {'iterations': 538, 'learning_rate': 0.013146680662227093, 'depth': 8, 'l2_leaf_reg': 12.806407496088607, 'random_strength': 5.60155609326646, 'subsample': 0.6596595764644638, 'rsm': 0.4132868175089994, 'min_data_in_leaf': 174, 'model_size_reg': 8.98058465834096, 'leaf_estimation_iterations': 2} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:42,888] Trial 2 failed with value None.
[W 2026-09-11 11:46:42,892] Trial 0 failed with parameters: {'iterations': 308, 'learning_rate': 0.03280317606891467, 'depth': 8, 'l2_leaf_reg': 1.050186939928115, 'random_strength': 4.618748795451214, 'subsample': 0.9558921085099645, 'rsm': 0.5736411800534043, 'min_data_in_leaf': 89, 'model_size_reg': 0.8608322481601871, 'leaf_estimation_iterations': 8} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:42,895] Trial 0 failed with value None.
[W 2026-09-11 11:46:42,903] Trial 1 failed with parameters: {'iterations': 514, 'learning_rate': 0.01564072316000063, 'depth': 5, 'l2_leaf_reg': 2.8095216967915895, 'random_strength': 8.567373742946769, 'subsample': 0.8541840190373537, 'rsm': 0.8829243018958843, 'min_data_in_leaf': 197, 'model_size_reg': 1.6250910012403574, 'leaf_estimation_iterations': 10} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:42,905] Trial 1 failed with value None.
[W 2026-09-11 11:46:42,955] Trial 4 failed with parameters: {'iterations': 449, 'learning_rate': 0.04265873497280987, 'depth': 4, 'l2_leaf_reg': 11.55644694378277, 'random_strength': 0.025004615757199842, 'subsample': 0.7276459352173769, 'rsm': 0.5530556458548213, 'min_data_in_leaf': 111, 'model_size_reg': 1.4387094154276647, 'leaf_estimation_iterations': 2} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:42,957] Trial 4 failed with value None.
[W 2026-09-11 11:46:42,965] Trial 5 failed with parameters: {'iterations': 420, 'learning_rate': 0.04940002147891178, 'depth': 7, 'l2_leaf_reg': 13.574298877254318, 'random_strength': 2.2158577548327307, 'subsample': 0.8010419308422948, 'rsm': 0.42184448513075157, 'min_data_in_leaf': 162, 'model_size_reg': 8.546097272637533, 'leaf_estimation_iterations': 9} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:42,968] Trial 5 failed with value None.
[W 2026-09-11 11:46:42,978] Trial 6 failed with parameters: {'iterations': 275, 'learning_rate': 0.01277437958082484, 'depth': 8, 'l2_leaf_reg': 4.6048425167886275, 'random_strength': 8.97071039149292, 'subsample': 0.7876892445445035, 'rsm': 0.31845890731332466, 'min_data_in_leaf': 172, 'model_size_reg': 4.470218546949862, 'leaf_estimation_iterations': 1} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:42,980] Trial 6 failed with value None.
[W 2026-09-11 11:46:42,991] Trial 7 failed with parameters: {'iterations': 244, 'learning_rate': 0.02656356242831727, 'depth': 7, 'l2_leaf_reg': 15.543987818582936, 'random_strength': 9.286027900725442, 'subsample': 0.5369701726898877, 'rsm': 0.8333562539490424, 'min_data_in_leaf': 148, 'model_size_reg': 1.3575179791512382, 'leaf_estimation_iterations': 5} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:42,993] Trial 7 failed with value None.
[W 2026-09-11 11:46:43,042] Trial 8 failed with parameters: {'iterations': 336, 'learning_rate': 0.05075995511391504, 'depth': 5, 'l2_leaf_reg': 20.743020378630977, 'random_strength': 8.174658194687272, 'subsample': 0.936241812836776, 'rsm': 0.6705641760202133, 'min_data_in_leaf': 61, 'model_size_reg': 1.025197995413123, 'leaf_estimation_iterations': 2} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:43,045] Trial 8 failed with value None.
[W 2026-09-11 11:46:43,052] Trial 9 failed with parameters: {'iterations': 573, 'learning_rate': 0.02580301756983372, 'depth': 7, 'l2_leaf_reg': 27.319090699238654, 'random_strength': 8.945581505096994, 'subsample': 0.7513567822993907, 'rsm': 0.42467263779097697, 'min_data_in_leaf': 199, 'model_size_reg': 8.246492361688983, 'leaf_estimation_iterations': 8} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:43,055] Trial 9 failed with value None.
[W 2026-09-11 11:46:43,066] Trial 10 failed with parameters: {'iterations': 274, 'learning_rate': 0.023812760892932012, 'depth': 8, 'l2_leaf_reg': 17.06141727655842, 'random_strength': 0.26597724689025837, 'subsample': 0.7210343453848762, 'rsm': 0.645799613104703, 'min_data_in_leaf': 176, 'model_size_reg': 2.7223154667326286, 'leaf_estimation_iterations': 3} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:43,068] Trial 10 failed with value None.
[W 2026-09-11 11:46:43,080] Trial 11 failed with parameters: {'iterations': 485, 'learning_rate': 0.05275038522358857, 'depth': 8, 'l2_leaf_reg': 25.71917820770998, 'random_strength': 6.96022527975204, 'subsample': 0.6747084479862637, 'rsm': 0.6828496763805529, 'min_data_in_leaf': 42, 'model_size_reg': 2.8068770586694445, 'leaf_estimation_iterations': 8} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:43,089] Trial 11 failed with value None.
[W 2026-09-11 11:46:43,130] Trial 12 failed with parameters: {'iterations': 575, 'learning_rate': 0.015211937537161685, 'depth': 4, 'l2_leaf_reg': 17.682878458594285, 'random_strength': 5.574406809731343, 'subsample': 0.7524093085077606, 'rsm': 0.6175399332710407, 'min_data_in_leaf': 138, 'model_size_reg': 0.8183766984805374, 'leaf_estimation_iterations': 4} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:43,132] Trial 12 failed with value None.
[W 2026-09-11 11:46:43,141] Trial 13 failed with parameters: {'iterations': 226, 'learning_rate': 0.0393844144127825, 'depth': 8, 'l2_leaf_reg': 22.86812837824838, 'random_strength': 4.3412906644577145, 'subsample': 0.584078437895198, 'rsm': 0.6090019885557402, 'min_data_in_leaf': 154, 'model_size_reg': 5.673709412157775, 'leaf_estimation_iterations': 10} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:43,143] Trial 13 failed with value None.
[W 2026-09-11 11:46:43,155] Trial 14 failed with parameters: {'iterations': 267, 'learning_rate': 0.04972416514117063, 'depth': 5, 'l2_leaf_reg': 28.517345321445504, 'random_strength': 9.17483267458061, 'subsample': 0.9486841368299928, 'rsm': 0.8464891474260807, 'min_data_in_leaf': 115, 'model_size_reg': 8.788669840856214, 'leaf_estimation_iterations': 9} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:43,160] Trial 14 failed with value None.
[W 2026-09-11 11:46:43,167] Trial 15 failed with parameters: {'iterations': 392, 'learning_rate': 0.014120864522131396, 'depth': 7, 'l2_leaf_reg': 22.992874350987094, 'random_strength': 9.253077012652337, 'subsample': 0.9489943695298146, 'rsm': 0.79795554392624, 'min_data_in_leaf': 127, 'model_size_reg': 0.6304699637938804, 'leaf_estimation_iterations': 9} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:43,169] Trial 15 failed with value None.
[W 2026-09-11 11:46:43,218] Trial 16 failed with parameters: {'iterations': 557, 'learning_rate': 0.028151065828119815, 'depth': 5, 'l2_leaf_reg': 16.00851692157867, 'random_strength': 9.69671731280842, 'subsample': 0.8039444044912718, 'rsm': 0.5795245187888893, 'min_data_in_leaf': 85, 'model_size_reg': 7.337991032619362, 'leaf_estimation_iterations': 9} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:43,220] Trial 16 failed with value None.
[W 2026-09-11 11:46:43,226] Trial 17 failed with parameters: {'iterations': 562, 'learning_rate': 0.043523284444352044, 'depth': 6, 'l2_leaf_reg': 24.633247855692645, 'random_strength': 4.082039160635304, 'subsample': 0.7822125043355921, 'rsm': 0.37065441713541747, 'min_data_in_leaf': 199, 'model_size_reg': 6.939630276176391, 'leaf_estimation_iterations': 10} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:43,228] Trial 17 failed with value None.
[W 2026-09-11 11:46:43,234] Trial 18 failed with parameters: {'iterations': 357, 'learning_rate': 0.012211340425660574, 'depth': 4, 'l2_leaf_reg': 18.806770797470318, 'random_strength': 9.961647672457406, 'subsample': 0.6078231486833996, 'rsm': 0.3634274530607547, 'min_data_in_leaf': 57, 'model_size_reg': 9.003872242136504, 'leaf_estimation_iterations': 6} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:43,237] Trial 18 failed with value None.
[W 2026-09-11 11:46:43,241] Trial 19 failed with parameters: {'iterations': 473, 'learning_rate': 0.04379584069960197, 'depth': 7, 'l2_leaf_reg': 14.906201526985955, 'random_strength': 4.512629358773994, 'subsample': 0.5238239404992135, 'rsm': 0.5876947540434616, 'min_data_in_leaf': 22, 'model_size_reg': 6.492609060412984, 'leaf_estimation_iterations': 3} because of the following error: NameError("name '_tlog' is not defined").
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 732, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial, allow_penalty=_allow_pen)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 394, in _objective_all_classification
    model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
[W 2026-09-11 11:46:43,243] Trial 19 failed with value None.
11:46:43 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all_direct__no_t__t': 0/20 Trials abgeschlossen (20 fehlgeschlagen, 0 gepruned, parallel=4).
11:46:43 WARNING [rubin.tuning] BLT 'catboost__propensity__classifier__all_direct__no_t__t': 20/20 Trials FEHLGESCHLAGEN. Fehlertypen:
11:46:43 WARNING [rubin.tuning]   [20×] Unbekannt
11:46:43 WARNING [rubin.tuning] BLT 'catboost__propensity__classifier__all_direct__no_t__t': Häufigster Fehler — vollständiger Traceback:
Unbekannt
11:46:43 WARNING [rubin.tuning] BLT 'catboost__propensity__classifier__all_direct__no_t__t': Keine abgeschlossenen Trials. Verwende Default-Parameter.
11:46:43 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all_direct__no_t__t': Study freigegeben, gc.collect() + malloc_trim durchgeführt.
11:46:43 INFO [rubin.tuning] BLT 'catboost__pseudo_effect__regressor__group_specific_shared_params__no_t__d': X=(163502, 77), target=(163502,) (unique=[0, 1]), subsample=100%, cv=5, objective=pseudo_effect
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/run_analysis.py", line 132, in <module>
    main()
  File "/home/ubuntu/da-hf1-rubin/run_analysis.py", line 128, in main
    pipe.run(export_bundle=args.export_bundle, bundle_dir=args.bundle_dir, bundle_id=args.bundle_id)
  File "/home/ubuntu/da-hf1-rubin/rubin/pipelines/analysis_pipeline.py", line 3720, in run
    tuned_params_by_model = self._run_tuning(cfg, X, T, Y, mlflow, _progress_cb=_progress)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/pipelines/analysis_pipeline.py", line 562, in _run_tuning
    tuned_params_by_model = tuner.tune_all(cfg.models.models_to_train, X=X, Y=Y, T=T)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 915, in tune_all
    best = self._tune_task(task, X=X, Y=Y, T=T, shared_params=tuned_by_task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 707, in _tune_task
    _xl_pseudo = self._build_xlearner_pseudo_outcomes(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 579, in _build_xlearner_pseudo_outcomes
    m0 = self._fit_model(nuisance_params, X_mat.iloc[control_train], control_y, "classifier")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/base_learner.py", line 361, in _fit_model
    _tlog.info(
    ^^^^^
NameError: name '_tlog' is not defined
