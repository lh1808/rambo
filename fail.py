# 1. NVIDIA-Treiber und CUDA-Version
nvidia-smi | head -4

# 2. CatBoost-Version
python -c "import catboost; print('CatBoost:', catboost.__version__)"

# 3. GPU-Modell und Compute Capability
nvidia-smi --query-gpu=name,compute_cap --format=csv
