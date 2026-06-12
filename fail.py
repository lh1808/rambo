Runs-Verzeichnis: /mnt/runs
Bisherige Runs im Verzeichnis: 12
Starte Forecast Job...
The StatsForecast module could not be imported. To enable support for the AutoARIMA, AutoETS and Croston models, please consider installing it.
INFO:__main__:Lade Konfiguration aus /mnt/da-pluto-timeseries/config.smoke.yaml
INFO:__main__:Runs-Verzeichnis: /mnt/runs
INFO:pluto_multivariate_repository:Connection to DB2 database was successful.
INFO:pluto_multivariate_repository:Using source table: t7.TA_DA_PLUTO_SP_2025
INFO:pluto_multivariate_repository:Using target table: t7.TA_DA_PLUTO_SP_2025_PROGNOSE
INFO:pluto_multivariate_repository:Loaded multivariate time series from DB2. Rows: 1987, Columns (components): 80
WARNING:pluto_multivariate_repository:UNERWARTETE PRODUKTE in DB gefunden (nicht in product_list): ['Sonstiges']. Diese werden trotzdem verarbeitet.
WARNING:pluto_multivariate_repository:UNERWARTETER SCHADENSTATUS in DB gefunden (nicht in status_list): ['LFD', 'NEU']. Diese werden trotzdem verarbeitet.
WARNING:pluto_multivariate_repository:FEHLENDER SCHADENSTATUS: In der erwarteten Liste, aber nicht in der DB vorhanden: ['Folgebearbeitung', 'Neuschaden'].
INFO:pluto_multivariate_repository:DIM_ORGA: 3 distinkte Ausprägungen geladen.
INFO:pluto_multivariate_repository:DB2 connection successfully closed.
Traceback (most recent call last):
  File "/mnt/da-pluto-timeseries/pluto_forecast_job.py", line 273, in <module>
    run_pluto_multivariate_forecast_job()
  File "/mnt/da-pluto-timeseries/pluto_forecast_job.py", line 156, in run_pluto_multivariate_forecast_job
    results = run_full_job(df_daily, cfg=cfg, logger=None)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 789, in run_full_job
    artifacts, metrics_df, backtest, true_ts = train_and_evaluate_for_horizon(
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 594, in train_and_evaluate_for_horizon
    res = rolling_block_forecast(
          ^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/utils.py", line 268, in rolling_block_forecast
    model = model_builder()
            ^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 587, in model_builder_eval
    return build_tft(effective_hcfg, effective_tft_cfg)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/model.py", line 435, in build_tft
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
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/torch/random.py", line 227, in fork_rng
    device_rng_states = [device_mod.get_rng_state(device) for device in devices]
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/torch/random.py", line 227, in <listcomp>
    device_rng_states = [device_mod.get_rng_state(device) for device in devices]
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/torch/cuda/random.py", line 33, in get_rng_state
    _lazy_init()
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/torch/cuda/__init__.py", line 491, in _lazy_init
    torch._C._cuda_init()
RuntimeError: The NVIDIA driver on your system is too old (found version 12080). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver.
