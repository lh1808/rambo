WARNING: Error parsing dependencies of nb-black: Expected matching RIGHT_PARENTHESIS for LEFT_PARENTHESIS, after version specifier
    yapf (>='0.28') ; python_version < "3.6"
         ~^
Starte Forecast Job...
The StatsForecast module could not be imported. To enable support for the AutoARIMA, AutoETS and Croston models, please consider installing it.
INFO:__main__:Keine Config-Datei unter /mnt/da-pluto-timeseries/config.yaml – nutze dataclass-Defaults.
INFO:pluto_multivariate_repository:Connection to DB2 database was successful.
INFO:pluto_multivariate_repository:Using source table: t7.TA_DA_PLUTO_SP_2025
INFO:pluto_multivariate_repository:Using target table: t7.TA_DA_PLUTO_SP_2025_PROGNOSE
INFO:pluto_multivariate_repository:Loaded multivariate time series from DB2. Rows: 1934, Columns (components): 40
INFO:darts.models.forecasting.torch_forecasting_model:Train dataset contains 109 samples.
INFO:darts.models.forecasting.torch_forecasting_model:Time series values are 64-bits; casting model to float64.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
You are using a CUDA device ('NVIDIA A100-PCIE-40GB MIG 3g.20gb') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/opt/conda/envs/generic/lib/python3.11/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
