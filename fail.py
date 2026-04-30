[rubin] Keine kategorialen Features erkannt → Standard-Encoding (numerisch).
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,379] Trial 5 failed with value None.
[W 2026-04-30 20:02:53,370] Trial 2 failed with value None.
[W 2026-04-30 20:02:53,371] Trial 7 failed with parameters: {'iterations': 227, 'learning_rate': 0.012860410306343608, 'depth': 4, 'l2_leaf_reg': 8.829146123148167, 'random_strength': 4.852551360730613, 'subsample': 0.754167880273547, 'rsm': 0.42963744344436416, 'min_data_in_leaf': 70, 'model_size_reg': 2.935722539962401, 'leaf_estimation_iterations': 10} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,385] Trial 7 failed with value None.
[W 2026-04-30 20:02:53,363] Trial 6 failed with parameters: {'iterations': 359, 'learning_rate': 0.04167117708549746, 'depth': 8, 'l2_leaf_reg': 10.06497738742153, 'random_strength': 0.8614622530875888, 'subsample': 0.5555288857225984, 'rsm': 0.5091886158588492, 'min_data_in_leaf': 148, 'model_size_reg': 1.0385867300007456, 'leaf_estimation_iterations': 3} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,389] Trial 6 failed with value None.
[W 2026-04-30 20:02:53,367] Trial 8 failed with parameters: {'iterations': 515, 'learning_rate': 0.04877078831744909, 'depth': 4, 'l2_leaf_reg': 1.6501589290837895, 'random_strength': 0.6934931072162503, 'subsample': 0.5937769987992714, 'rsm': 0.7395270786806087, 'min_data_in_leaf': 15, 'model_size_reg': 0.6150947647717264, 'leaf_estimation_iterations': 7} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,381] Trial 11 failed with parameters: {'iterations': 279, 'learning_rate': 0.04003753191927439, 'depth': 8, 'l2_leaf_reg': 7.749038181230141, 'random_strength': 0.011528607054661926, 'subsample': 0.8068272439266039, 'rsm': 0.8390433011589693, 'min_data_in_leaf': 196, 'model_size_reg': 6.993496996223413, 'leaf_estimation_iterations': 3} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,371] Trial 9 failed with parameters: {'iterations': 466, 'learning_rate': 0.01361351579188636, 'depth': 6, 'l2_leaf_reg': 3.1042084882822665, 'random_strength': 0.6550563588482357, 'subsample': 0.6288293959467453, 'rsm': 0.8379013108467874, 'min_data_in_leaf': 34, 'model_size_reg': 2.412321030231437, 'leaf_estimation_iterations': 2} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,388] Trial 12 failed with parameters: {'iterations': 541, 'learning_rate': 0.03954494465629728, 'depth': 6, 'l2_leaf_reg': 3.2520838468974174, 'random_strength': 0.10694571467313421, 'subsample': 0.8787611539615836, 'rsm': 0.6033891160086071, 'min_data_in_leaf': 14, 'model_size_reg': 4.671862209989507, 'leaf_estimation_iterations': 7} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,402] Trial 12 failed with value None.
[W 2026-04-30 20:02:53,390] Trial 13 failed with parameters: {'iterations': 389, 'learning_rate': 0.06734462104183508, 'depth': 5, 'l2_leaf_reg': 7.224581904696664, 'random_strength': 0.5617577709401106, 'subsample': 0.9247174049892225, 'rsm': 0.6414138118774879, 'min_data_in_leaf': 197, 'model_size_reg': 3.958458047357344, 'leaf_estimation_iterations': 10} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,407] Trial 13 failed with value None.
[W 2026-04-30 20:02:53,395] Trial 14 failed with parameters: {'iterations': 303, 'learning_rate': 0.037826914868236305, 'depth': 4, 'l2_leaf_reg': 2.2059147645702213, 'random_strength': 0.06275950736546387, 'subsample': 0.8182272801648622, 'rsm': 0.4063462236017179, 'min_data_in_leaf': 139, 'model_size_reg': 1.8380482811845988, 'leaf_estimation_iterations': 5} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,397] Trial 11 failed with value None.
[W 2026-04-30 20:02:53,397] Trial 15 failed with parameters: {'iterations': 514, 'learning_rate': 0.07376215494286625, 'depth': 7, 'l2_leaf_reg': 7.242131135763712, 'random_strength': 0.968310999584299, 'subsample': 0.6146343984834831, 'rsm': 0.38619715188007514, 'min_data_in_leaf': 12, 'model_size_reg': 1.6176246674200823, 'leaf_estimation_iterations': 5} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,414] Trial 15 failed with value None.
[W 2026-04-30 20:02:53,377] Trial 10 failed with parameters: {'iterations': 326, 'learning_rate': 0.014864113605340668, 'depth': 4, 'l2_leaf_reg': 14.889462847258923, 'random_strength': 0.30282619928199256, 'subsample': 0.8305690079761883, 'rsm': 0.4520997363019404, 'min_data_in_leaf': 192, 'model_size_reg': 1.384137850655337, 'leaf_estimation_iterations': 3} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,417] Trial 10 failed with value None.
[W 2026-04-30 20:02:53,395] Trial 8 failed with value None.
[W 2026-04-30 20:02:53,412] Trial 14 failed with value None.
[W 2026-04-30 20:02:53,413] Trial 17 failed with parameters: {'iterations': 552, 'learning_rate': 0.01635109751872954, 'depth': 5, 'l2_leaf_reg': 3.5676768836621, 'random_strength': 0.4823493181666334, 'subsample': 0.8350318574211139, 'rsm': 0.47641944049574436, 'min_data_in_leaf': 126, 'model_size_reg': 7.515814671879842, 'leaf_estimation_iterations': 5} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,420] Trial 17 failed with value None.
[W 2026-04-30 20:02:53,402] Trial 16 failed with parameters: {'iterations': 499, 'learning_rate': 0.1382989995718745, 'depth': 7, 'l2_leaf_reg': 1.022533637230956, 'random_strength': 8.939927892780826, 'subsample': 0.5166089501997364, 'rsm': 0.73235866886135, 'min_data_in_leaf': 17, 'model_size_reg': 0.6175361342256214, 'leaf_estimation_iterations': 5} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,422] Trial 16 failed with value None.
[W 2026-04-30 20:02:53,399] Trial 9 failed with value None.
[W 2026-04-30 20:02:53,422] Trial 18 failed with parameters: {'iterations': 596, 'learning_rate': 0.03457870579712981, 'depth': 5, 'l2_leaf_reg': 1.9979842472851057, 'random_strength': 0.39647943042731076, 'subsample': 0.625605236101181, 'rsm': 0.7289799928387433, 'min_data_in_leaf': 120, 'model_size_reg': 1.4362527585969453, 'leaf_estimation_iterations': 10} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,423] Trial 19 failed with parameters: {'iterations': 414, 'learning_rate': 0.1410592131989539, 'depth': 6, 'l2_leaf_reg': 11.380578788368881, 'random_strength': 6.912320949325617, 'subsample': 0.725287855005641, 'rsm': 0.8265684120140635, 'min_data_in_leaf': 182, 'model_size_reg': 0.9402250564947257, 'leaf_estimation_iterations': 10} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,425] Trial 18 failed with value None.
[W 2026-04-30 20:02:53,427] Trial 19 failed with value None.
20:02:53 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all__no_t__t': 0/20 Trials abgeschlossen (20 fehlgeschlagen, 0 gepruned, parallel=10).
20:02:53 WARNING [rubin.tuning] BLT 'catboost__propensity__classifier__all__no_t__t': 20/20 Trials FEHLGESCHLAGEN. Fehlertypen:
20:02:53 WARNING [rubin.tuning]   [20×] Unbekannt
20:02:53 WARNING [rubin.tuning] BLT 'catboost__propensity__classifier__all__no_t__t': Häufigster Fehler — vollständiger Traceback:
Unbekannt
20:02:53 WARNING [rubin.tuning] Tuning-Task 'catboost__propensity__classifier__all__no_t__t': Keine abgeschlossenen Trials. Verwende Default-Parameter.
20:02:53 INFO [rubin.tuning] Tuning-Task 'catboost__propensity__classifier__all_direct__no_t__t': X_input=299988 rows, indices=299988, X_task=(299988, 129), target=(299988,) (unique=[0, 1]), train_subsample=100%, T_task unique=[0, 1], cv_splits=5, target_name=T, objective=propensity
[W 2026-04-30 20:02:53,751] Trial 0 failed with parameters: {'iterations': 248, 'learning_rate': 0.05674111744373595, 'depth': 4, 'l2_leaf_reg': 1.5663287603842078, 'random_strength': 0.3068602700396115, 'subsample': 0.5243107086425953, 'rsm': 0.5093159521457838, 'min_data_in_leaf': 169, 'model_size_reg': 9.821379953959253, 'leaf_estimation_iterations': 5} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,753] Trial 1 failed with parameters: {'iterations': 282, 'learning_rate': 0.024974157059265284, 'depth': 4, 'l2_leaf_reg': 8.139994062189691, 'random_strength': 0.015652121698617683, 'subsample': 0.9832394064207723, 'rsm': 0.7089860081220187, 'min_data_in_leaf': 105, 'model_size_reg': 4.318951857643246, 'leaf_estimation_iterations': 7} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,757] Trial 2 failed with parameters: {'iterations': 359, 'learning_rate': 0.018385564808449023, 'depth': 8, 'l2_leaf_reg': 2.054754227913476, 'random_strength': 0.10224509115459142, 'subsample': 0.665648314796385, 'rsm': 0.3226204284608945, 'min_data_in_leaf': 153, 'model_size_reg': 3.8426909390795183, 'leaf_estimation_iterations': 10} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,758] Trial 3 failed with parameters: {'iterations': 460, 'learning_rate': 0.045719248070627536, 'depth': 5, 'l2_leaf_reg': 7.33398234036339, 'random_strength': 0.2835452704376963, 'subsample': 0.87206154117517, 'rsm': 0.6339868579874567, 'min_data_in_leaf': 35, 'model_size_reg': 3.0057119105493246, 'leaf_estimation_iterations': 9} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,758] Trial 4 failed with parameters: {'iterations': 448, 'learning_rate': 0.14279585241459486, 'depth': 6, 'l2_leaf_reg': 5.341390455853453, 'random_strength': 1.1386444089083716, 'subsample': 0.6289142489358146, 'rsm': 0.5182693559097238, 'min_data_in_leaf': 72, 'model_size_reg': 1.2337318251557416, 'leaf_estimation_iterations': 9} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,758] Trial 0 failed with value None.
[W 2026-04-30 20:02:53,761] Trial 1 failed with value None.
[W 2026-04-30 20:02:53,762] Trial 5 failed with parameters: {'iterations': 465, 'learning_rate': 0.010930550092292608, 'depth': 8, 'l2_leaf_reg': 5.473847922280424, 'random_strength': 0.011636556906873107, 'subsample': 0.5212052383055461, 'rsm': 0.7129307774583707, 'min_data_in_leaf': 63, 'model_size_reg': 6.529496719674636, 'leaf_estimation_iterations': 4} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,777] Trial 5 failed with value None.
[W 2026-04-30 20:02:53,766] Trial 2 failed with value None.
[W 2026-04-30 20:02:53,766] Trial 7 failed with parameters: {'iterations': 401, 'learning_rate': 0.06319219396061641, 'depth': 7, 'l2_leaf_reg': 2.265939730912071, 'random_strength': 0.22254716808782596, 'subsample': 0.5788754903750212, 'rsm': 0.35642825510068543, 'min_data_in_leaf': 196, 'model_size_reg': 9.844259609739117, 'leaf_estimation_iterations': 8} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,781] Trial 7 failed with value None.
[W 2026-04-30 20:02:53,770] Trial 4 failed with value None.
[W 2026-04-30 20:02:53,770] Trial 9 failed with parameters: {'iterations': 214, 'learning_rate': 0.026672240397647377, 'depth': 7, 'l2_leaf_reg': 16.259134382910286, 'random_strength': 0.012326108634500508, 'subsample': 0.5130477692601949, 'rsm': 0.8051287533966085, 'min_data_in_leaf': 102, 'model_size_reg': 6.475950648428092, 'leaf_estimation_iterations': 6} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,773] Trial 8 failed with parameters: {'iterations': 350, 'learning_rate': 0.01629533308529257, 'depth': 8, 'l2_leaf_reg': 1.8238990497558054, 'random_strength': 2.3181574544525514, 'subsample': 0.6295097863514608, 'rsm': 0.6222763473578223, 'min_data_in_leaf': 147, 'model_size_reg': 2.211240576092469, 'leaf_estimation_iterations': 10} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,789] Trial 8 failed with value None.
[W 2026-04-30 20:02:53,768] Trial 3 failed with value None.
[W 2026-04-30 20:02:53,786] Trial 9 failed with value None.
[W 2026-04-30 20:02:53,786] Trial 10 failed with parameters: {'iterations': 279, 'learning_rate': 0.09743465204401094, 'depth': 4, 'l2_leaf_reg': 1.472265763984786, 'random_strength': 0.08801570846168481, 'subsample': 0.6667726243127945, 'rsm': 0.3797936249472639, 'min_data_in_leaf': 129, 'model_size_reg': 1.9526975317763262, 'leaf_estimation_iterations': 6} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,795] Trial 10 failed with value None.
[W 2026-04-30 20:02:53,766] Trial 6 failed with parameters: {'iterations': 367, 'learning_rate': 0.015692059777011894, 'depth': 6, 'l2_leaf_reg': 4.769457657081786, 'random_strength': 0.018040662010318752, 'subsample': 0.8158634186334967, 'rsm': 0.6420348649810057, 'min_data_in_leaf': 94, 'model_size_reg': 9.45299649935893, 'leaf_estimation_iterations': 5} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,798] Trial 6 failed with value None.
[W 2026-04-30 20:02:53,787] Trial 11 failed with parameters: {'iterations': 451, 'learning_rate': 0.012834375078963851, 'depth': 4, 'l2_leaf_reg': 2.192228303321951, 'random_strength': 0.07502600207324374, 'subsample': 0.6368858773166541, 'rsm': 0.48807812714695586, 'min_data_in_leaf': 54, 'model_size_reg': 1.4870329180357011, 'leaf_estimation_iterations': 7} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,802] Trial 11 failed with value None.
[W 2026-04-30 20:02:53,814] Trial 12 failed with parameters: {'iterations': 318, 'learning_rate': 0.034036564886017084, 'depth': 7, 'l2_leaf_reg': 1.01797835847101, 'random_strength': 4.75064235195191, 'subsample': 0.9564836181421308, 'rsm': 0.35233592747311426, 'min_data_in_leaf': 177, 'model_size_reg': 9.578666493610552, 'leaf_estimation_iterations': 2} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,816] Trial 13 failed with parameters: {'iterations': 340, 'learning_rate': 0.03421354406577025, 'depth': 5, 'l2_leaf_reg': 1.7479618515601487, 'random_strength': 0.06699227672059921, 'subsample': 0.9013196180901795, 'rsm': 0.6530780865152904, 'min_data_in_leaf': 76, 'model_size_reg': 2.530562206235248, 'leaf_estimation_iterations': 5} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,818] Trial 13 failed with value None.
[W 2026-04-30 20:02:53,817] Trial 14 failed with parameters: {'iterations': 446, 'learning_rate': 0.018024851598464415, 'depth': 5, 'l2_leaf_reg': 1.7630923356708867, 'random_strength': 8.98671410995883, 'subsample': 0.6879369036614038, 'rsm': 0.6698535366008429, 'min_data_in_leaf': 23, 'model_size_reg': 4.089161791279596, 'leaf_estimation_iterations': 4} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,820] Trial 14 failed with value None.
[W 2026-04-30 20:02:53,816] Trial 12 failed with value None.
[W 2026-04-30 20:02:53,821] Trial 15 failed with parameters: {'iterations': 399, 'learning_rate': 0.013923679085130628, 'depth': 6, 'l2_leaf_reg': 3.024747523231854, 'random_strength': 0.3459861063102275, 'subsample': 0.52274165473157, 'rsm': 0.741084980063681, 'min_data_in_leaf': 69, 'model_size_reg': 8.215656745056092, 'leaf_estimation_iterations': 5} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,821] Trial 17 failed with parameters: {'iterations': 406, 'learning_rate': 0.11488982030148409, 'depth': 7, 'l2_leaf_reg': 17.992061463786083, 'random_strength': 0.024427189319478116, 'subsample': 0.5024943361526272, 'rsm': 0.8050821515576336, 'min_data_in_leaf': 158, 'model_size_reg': 0.06800789639501259, 'leaf_estimation_iterations': 4} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,821] Trial 16 failed with parameters: {'iterations': 495, 'learning_rate': 0.0654146327231182, 'depth': 6, 'l2_leaf_reg': 2.236732191189541, 'random_strength': 0.019132303866808798, 'subsample': 0.5652767450005383, 'rsm': 0.396357363505534, 'min_data_in_leaf': 95, 'model_size_reg': 3.950527903889621, 'leaf_estimation_iterations': 7} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,822] Trial 19 failed with parameters: {'iterations': 376, 'learning_rate': 0.010284069607867426, 'depth': 5, 'l2_leaf_reg': 27.44372863222045, 'random_strength': 0.04336376739584246, 'subsample': 0.605540517019326, 'rsm': 0.6960342332016125, 'min_data_in_leaf': 32, 'model_size_reg': 6.661890007611314, 'leaf_estimation_iterations': 10} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,824] Trial 15 failed with value None.
[W 2026-04-30 20:02:53,824] Trial 18 failed with parameters: {'iterations': 386, 'learning_rate': 0.10035758048883836, 'depth': 5, 'l2_leaf_reg': 1.9658057038632955, 'random_strength': 0.01905823987032896, 'subsample': 0.8452758971798984, 'rsm': 0.8362604622204279, 'min_data_in_leaf': 15, 'model_size_reg': 1.1900346093055936, 'leaf_estimation_iterations': 8} because of the following error: NameError("name '_strata' is not defined").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1047, in objective
    return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 773, in _objective_all_classification
    for fold_i, (tr, va) in enumerate(self._cv_splits(_strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=_train_ratio)):
                                                      ^^^^^^^
NameError: name '_strata' is not defined
[W 2026-04-30 20:02:53,826] Trial 17 failed with value None.
[W 2026-04-30 20:02:53,828] Trial 16 failed with value None.
[W 2026-04-30 20:02:53,830] Trial 19 failed with value None.
[W 2026-04-30 20:02:53,832] Trial 18 failed with value None.
20:02:53 INFO [rubin.tuning] BLT 'catboost__propensity__classifier__all_direct__no_t__t': 0/20 Trials abgeschlossen (20 fehlgeschlagen, 0 gepruned, parallel=10).
20:02:53 WARNING [rubin.tuning] BLT 'catboost__propensity__classifier__all_direct__no_t__t': 20/20 Trials FEHLGESCHLAGEN. Fehlertypen:
20:02:53 WARNING [rubin.tuning]   [20×] Unbekannt
20:02:53 WARNING [rubin.tuning] BLT 'catboost__propensity__classifier__all_direct__no_t__t': Häufigster Fehler — vollständiger Traceback:
Unbekannt
20:02:53 WARNING [rubin.tuning] Tuning-Task 'catboost__propensity__classifier__all_direct__no_t__t': Keine abgeschlossenen Trials. Verwende Default-Parameter.
20:02:54 INFO [rubin.tuning] Tuning-Task 'catboost__pseudo_effect__regressor__group_specific_shared_params__no_t__d': X_input=299988 rows, indices=299988, X_task=(299988, 129), target=(299988,) (unique=[0, 1]), train_subsample=100%, T_task unique=[0, 1], cv_splits=5, target_name=D, objective=pseudo_effect
