# 1. Conda-CatBoost + CUDA 12.9 runtime entfernen
pixi remove catboost cuda-cudart cuda-cudart_linux-64 cuda-version

# 2. CatBoost von PyPI über pixi installieren (statisch gelinkte CUDA)
pixi add --pypi catboost==1.2.10

# 3. Verifizieren: Muss OHNE cuda129 im Build-String sein
pixi run python -c "import catboost; print(catboost.__version__, catboost.__file__)"

# 4. GPU testen
pixi run python -c "
from catboost import CatBoostRegressor
import numpy as np
m = CatBoostRegressor(iterations=5, task_type='GPU', verbose=1, allow_writing_files=False)
m.fit(np.random.rand(100,5).astype(np.float32), np.random.rand(100))
print('GPU OK!')
"
