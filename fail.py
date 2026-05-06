I 2026-05-06 10:44:01,338] A new study created in memory with name: no-name-ef108854-789f-4768-b38d-6b4ce7363683
[W 2026-05-06 10:44:01,677] Trial 0 failed with parameters: {'train_length_weeks': 78, 'hidden_size': 80, 'hidden_continuous_size': 32, 'lstm_layers': 1, 'dropout': 0.45720795338776943, 'learning_rate': 0.0009392267032579403, 'weight_decay': 0.00018381295128882237, 'batch_size': 24} because of the following error: RuntimeError('The NVIDIA driver on your system is too old (found version 12080). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver.').
Traceback (most recent call last):
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 372, in <lambda>
    lambda trial: _optuna_objective_for_horizon(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 225, in _optuna_objective_for_horizon
    res = rolling_block_forecast(
          ^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/utils.py", line 268, in rolling_block_forecast
    model = model_builder()
            ^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 199, in model_builder
    return build_tft(hcfg, tuned_tft_cfg, extra_callbacks=pruning_callbacks)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/model.py", line 428, in build_tft
    return TFTModel(
           ^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/models/forecasting/forecasting_model.py", line 128, in __call__
    return super().__call__(**all_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/models/forecasting/tft_model.py", line 945, in __init__
    super().__init__(**self._extract_torch_model_params(**model_kwargs))
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/utils/torch.py", line 92, in decorator
    with fork_rng():
  File "/opt/conda/envs/generic/lib/python3.11/contextlib.py", line 137, in __enter__
    return next(self.gen)
           ^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/torch/random.py", line 210, in fork_rng
    device_rng_states = [device_mod.get_rng_state(device) for device in devices]
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/torch/random.py", line 210, in <listcomp>
    device_rng_states = [device_mod.get_rng_state(device) for device in devices]
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/torch/cuda/random.py", line 33, in get_rng_state
    _lazy_init()
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/torch/cuda/__init__.py", line 478, in _lazy_init
    torch._C._cuda_init()
RuntimeError: The NVIDIA driver on your system is too old (found version 12080). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver.
[W 2026-05-06 10:44:01,688] Trial 0 failed with value None.
INFO:pluto_multivariate_repository:DB2 connection successfully closed.
Traceback (most recent call last):
  File "/mnt/da-pluto-timeseries/pluto_forecast_job.py", line 235, in <module>
    run_pluto_multivariate_forecast_job()
  File "/mnt/da-pluto-timeseries/pluto_forecast_job.py", line 154, in run_pluto_multivariate_forecast_job
    results = run_full_job(df_daily, cfg=cfg, logger=None)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 641, in run_full_job
    artifacts, metrics_df, backtest, true_ts = train_and_evaluate_for_horizon(
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 371, in train_and_evaluate_for_horizon
    study.optimize(
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 372, in <lambda>
    lambda trial: _optuna_objective_for_horizon(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 225, in _optuna_objective_for_horizon
    res = rolling_block_forecast(
          ^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/utils.py", line 268, in rolling_block_forecast
    model = model_builder()
            ^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 199, in model_builder
    return build_tft(hcfg, tuned_tft_cfg, extra_callbacks=pruning_callbacks)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/model.py", line 428, in build_tft
    return TFTModel(
           ^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/models/forecasting/forecasting_model.py", line 128, in __call__
    return super().__call__(**all_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/models/forecasting/tft_model.py", line 945, in __init__
    super().__init__(**self._extract_torch_model_params(**model_kwargs))
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/utils/torch.py", line 92, in decorator
    with fork_rng():
  File "/opt/conda/envs/generic/lib/python3.11/contextlib.py", line 137, in __enter__
    return next(self.gen)
           ^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/torch/random.py", line 210, in fork_rng
    device_rng_states = [device_mod.get_rng_state(device) for device in devices]
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/torch/random.py", line 210, in <listcomp>
    device_rng_states = [device_mod.get_rng_state(device) for device in devices]
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/torch/cuda/random.py", line 33, in get_rng_state
    _lazy_init()
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/torch/cuda/__init__.py", line 478, in _lazy_init
    torch._C._cuda_init()
RuntimeError: The NVIDIA driver on your system is too old (found version 12080). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver.
