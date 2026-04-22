WARNING:pluto_multivariate_repository:UNERWARTETE PRODUKTE in DB gefunden (nicht in product_list): ['HUS_Eingabe', 'KFZ_Kasko_Eingabe']. Diese werden trotzdem verarbeitet.
WARNING:pluto_multivariate_repository:UNERWARTETER SCHADENSTATUS in DB gefunden (nicht in status_list): ['LFD', 'NEU']. Diese werden trotzdem verarbeitet.
WARNING:pluto_multivariate_repository:FEHLENDER SCHADENSTATUS: In der erwarteten Liste, aber nicht in der DB vorhanden: ['Folgebearbeitung', 'Neuschaden'].
INFO:pluto_multivariate_repository:DB2 connection successfully closed.
Traceback (most recent call last):
  File "/mnt/da-pluto-timeseries/pluto_forecast_job.py", line 235, in <module>
    run_pluto_multivariate_forecast_job()
  File "/mnt/da-pluto-timeseries/pluto_forecast_job.py", line 154, in run_pluto_multivariate_forecast_job
    results = run_full_job(df_daily, cfg=cfg, logger=None)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 596, in run_full_job
    artifacts, metrics_df, backtest, true_ts = train_and_evaluate_for_horizon(
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 408, in train_and_evaluate_for_horizon
    res = rolling_block_forecast(
          ^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/utils.py", line 268, in rolling_block_forecast
    model = model_builder()
            ^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/pipeline.py", line 401, in model_builder_eval
    return build_tft(hcfg, effective_tft_cfg)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/da-pluto-timeseries/forecasting/model.py", line 403, in build_tft
    if tft_cfg.weight_decay > 0:
       ^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: '>' not supported between instances of 'str' and 'int'
