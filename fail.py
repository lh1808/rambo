ERROR:main_logger:ValueError: The input `series` are too short to extract even a single sample. Expected min length: `156`, received max length: `121`.
INFO:pluto_multivariate_repository:DB2 connection successfully closed.
Traceback (most recent call last):
  File "/mnt/da-pluto-timeseries/pluto_forecast_job.py", line 201, in <module>
    run_pluto_multivariate_forecast_job()
  File "/mnt/da-pluto-timeseries/pluto_forecast_job.py", line 140, in run_pluto_multivariate_forecast_job
    results = run_full_job(df_daily, cfg=cfg, logger=None)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 513, in run_full_job
    artifacts, metrics_df, backtest, true_ts = train_and_evaluate_for_horizon(
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 338, in train_and_evaluate_for_horizon
    res = rolling_block_forecast(
          ^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/utils.py", line 220, in rolling_block_forecast
    model.fit(
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/utils/torch.py", line 94, in decorator
    return decorated(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/models/forecasting/torch_forecasting_model.py", line 934, in fit
    ) = self._setup_for_fit_from_dataset(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/models/forecasting/torch_forecasting_model.py", line 1044, in _setup_for_fit_from_dataset
    train_dataset = self._build_train_dataset(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/models/forecasting/tft_model.py", line 1177, in _build_train_dataset
    return super()._build_train_dataset(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/models/forecasting/torch_forecasting_model.py", line 562, in _build_train_dataset
    return SequentialTorchTrainingDataset(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/utils/data/torch_datasets/training_dataset.py", line 419, in __init__
    super().__init__(
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/utils/data/torch_datasets/training_dataset.py", line 178, in __init__
    raise_log(
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/darts/logging.py", line 132, in raise_log
    raise exception
ValueError: The input `series` are too short to extract even a single sample. Expected min length: `156`, received max length: `121`.
