Verwende Config-Profil: smoke (config.smoke.yaml)
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
INFO:darts.models.forecasting.torch_forecasting_model:Train dataset contains 40 samples.
INFO:darts.models.forecasting.torch_forecasting_model:Time series values are 64-bits; casting model to float64.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
`Trainer.fit` stopped: `max_epochs=10` reached.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[Block] train [2023-12-24 .. 2025-12-14] -> pred [2025-12-21 .. 2026-03-15] (len=13)
INFO:darts.models.forecasting.torch_forecasting_model:Train dataset contains 40 samples.
INFO:darts.models.forecasting.torch_forecasting_model:Time series values are 64-bits; casting model to float64.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
`Trainer.fit` stopped: `max_epochs=10` reached.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[Block] train [2024-03-24 .. 2026-03-15] -> pred [2026-03-22 .. 2026-06-14] (len=13)
INFO:darts.models.forecasting.torch_forecasting_model:Train dataset contains 40 samples.
INFO:darts.models.forecasting.torch_forecasting_model:Time series values are 64-bits; casting model to float64.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
`Trainer.fit` stopped: `max_epochs=10` reached.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
INFO:darts.models.forecasting.torch_forecasting_model:Train dataset contains 27 samples.
INFO:darts.models.forecasting.torch_forecasting_model:Time series values are 64-bits; casting model to float64.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
`Trainer.fit` stopped: `max_epochs=10` reached.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[Block] train [2022-12-25 .. 2025-06-15] -> pred [2025-06-22 .. 2026-06-14] (len=52)
INFO:darts.models.forecasting.torch_forecasting_model:Train dataset contains 27 samples.
INFO:darts.models.forecasting.torch_forecasting_model:Time series values are 64-bits; casting model to float64.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
`Trainer.fit` stopped: `max_epochs=10` reached.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
INFO:__main__:Run-Artefakte für Horizont 13 unter /mnt/runs/2026-06-12T11-07_h13_model
INFO:__main__:Run-Artefakte für Horizont 52 unter /mnt/runs/2026-06-12T11-07_h52_model
INFO:__main__:Keine früheren Forecasts zur retrospektiven Evaluation vorhanden.
INFO:__main__:Wochentagsprofile: 80 Komponenten (Ebenen: {'component': 69, 'group': 11})
INFO:__main__:Disaggregation: 52 Wochen × 80 Komponenten → 364 Tage (weekend_policy=empirical).
INFO:__main__:Schreibe Forecast: 364 Tage ab 2026-06-15 bis 2027-06-13 (letztes Ist-Datum: 2026-06-11).
INFO:pluto_multivariate_repository:Forecast in t7.TA_DA_PLUTO_SP_2025_PROGNOSE geschrieben (28462 Zeilen).
INFO:__main__:PLUTO multivariate forecast job completed successfully.
INFO:pluto_multivariate_repository:DB2 connection successfully closed.
Job abgeschlossen.
