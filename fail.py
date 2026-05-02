^C(generic) ubuntu@192.168.5.216 ~/rubin $ nvidia-smi | head -4
Sat May  2 21:29:30 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.211.01             Driver Version: 570.211.01     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
(generic) ubuntu@192.168.5.216 ~/rubin $ python -c "import catboost; print('CatBoost:', catboost.__version__)"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'catboost'
(generic) ubuntu@192.168.5.216 ~/rubin $ pixi -c "import catboost; print('CatBoost:', catboost.__version__)"
error: unexpected argument '-c' found

Usage: pixi [OPTIONS] [COMMAND]

For more information, try '--help'.
(generic) ubuntu@192.168.5.216 ~/rubin $ pixi run python -c "import catboost; print('CatBoost:', catboost.__version__)"
CatBoost: 1.2.10
(generic) ubuntu@192.168.5.216 ~/rubin $ nvidia-smi --query-gpu=name,compute_cap --format=csv
name, compute_cap
Tesla V100-PCIE-32GB, 7.0
(generic) ubuntu@192.168.5.216 ~/rubin $ 
