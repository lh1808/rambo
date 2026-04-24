[W 2026-04-24 20:51:09,816] Trial 3 failed with parameters: {'iterations': 300, 'learning_rate': 0.05169298625369349, 'depth': 2, 'l2_leaf_reg': 79.00795794211078, 'random_strength': 6.323814953980304, 'subsample': 0.5158104051656958, 'rsm': 0.4383849405204935, 'min_data_in_leaf': 460, 'model_size_reg': 1.197246395033322, 'leaf_estimation_iterations': 5} because of the following error: AttributeError("'FinalModelTuner' object has no attribute '_apply_overfit_penalty'").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1402, in objective
    adjusted = self._apply_overfit_penalty(val_score, train_score, penalty=_op, tolerance=_ot)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'FinalModelTuner' object has no attribute '_apply_overfit_penalty'
[W 2026-04-24 20:51:09,822] Trial 3 failed with value None.
[W 2026-04-24 20:51:10,406] Trial 4 failed with parameters: {'iterations': 307, 'learning_rate': 0.022388852748244835, 'depth': 2, 'l2_leaf_reg': 11.81574864536997, 'random_strength': 2.4808359039696697, 'subsample': 0.7944867096876391, 'rsm': 0.5283409777775628, 'min_data_in_leaf': 99, 'model_size_reg': 0.2595047993186799, 'leaf_estimation_iterations': 3} because of the following error: AttributeError("'FinalModelTuner' object has no attribute '_apply_overfit_penalty'").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1402, in objective
    adjusted = self._apply_overfit_penalty(val_score, train_score, penalty=_op, tolerance=_ot)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'FinalModelTuner' object has no attribute '_apply_overfit_penalty'
[W 2026-04-24 20:51:10,416] Trial 4 failed with value None.
[W 2026-04-24 20:51:10,998] Trial 0 failed with parameters: {'iterations': 211, 'learning_rate': 0.00602865532519029, 'depth': 4, 'l2_leaf_reg': 15.51326065151898, 'random_strength': 1.2634534532824913, 'subsample': 0.7415631674840364, 'rsm': 0.6436186650420015, 'min_data_in_leaf': 285, 'model_size_reg': 9.35620373117854, 'leaf_estimation_iterations': 3} because of the following error: AttributeError("'FinalModelTuner' object has no attribute '_apply_overfit_penalty'").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1402, in objective
    adjusted = self._apply_overfit_penalty(val_score, train_score, penalty=_op, tolerance=_ot)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'FinalModelTuner' object has no attribute '_apply_overfit_penalty'
[W 2026-04-24 20:51:10,999] Trial 0 failed with value None.
[W 2026-04-24 20:51:11,686] Trial 1 failed with parameters: {'iterations': 333, 'learning_rate': 0.05380326167119583, 'depth': 3, 'l2_leaf_reg': 15.161820121607512, 'random_strength': 1.732275560886047, 'subsample': 0.7744495449353237, 'rsm': 0.6806891228173266, 'min_data_in_leaf': 170, 'model_size_reg': 5.6936429377833, 'leaf_estimation_iterations': 3} because of the following error: AttributeError("'FinalModelTuner' object has no attribute '_apply_overfit_penalty'").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1402, in objective
    adjusted = self._apply_overfit_penalty(val_score, train_score, penalty=_op, tolerance=_ot)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'FinalModelTuner' object has no attribute '_apply_overfit_penalty'
[W 2026-04-24 20:51:11,688] Trial 1 failed with value None.
[W 2026-04-24 20:51:12,732] Trial 2 failed with parameters: {'iterations': 323, 'learning_rate': 0.01542032518899923, 'depth': 4, 'l2_leaf_reg': 12.078843790658405, 'random_strength': 3.9051570290311624, 'subsample': 0.7475200998009468, 'rsm': 0.6167782566759752, 'min_data_in_leaf': 239, 'model_size_reg': 0.7501962486597917, 'leaf_estimation_iterations': 2} because of the following error: AttributeError("'FinalModelTuner' object has no attribute '_apply_overfit_penalty'").
Traceback (most recent call last):
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/optuna/study/_optimize.py", line 206, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1402, in objective
    adjusted = self._apply_overfit_penalty(val_score, train_score, penalty=_op, tolerance=_ot)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'FinalModelTuner' object has no attribute '_apply_overfit_penalty'
[W 2026-04-24 20:51:12,734] Trial 2 failed with value None.
