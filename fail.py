Schritt 1 — Prüfe woher CatBoost kommt:
bashpixi run python -c "import catboost; print(catboost.__file__)"
Schritt 2 — Installiere CatBoost von PyPI statt conda:
bashpixi run pip install catboost==1.2.10 --force-reinstall --break-system-packages
Schritt 3 — Teste GPU:
bashpixi run python -c "
from catboost import CatBoostRegressor
import numpy as np
m = CatBoostRegressor(iterations=5, task_type='GPU', verbose=1, allow_writing_files=False)
m.fit(np.random.rand(100,5).astype(np.float32), np.random.rand(100))
print('GPU OK!')
"
Falls Schritt 2 nicht hilft, probiere eine ältere Version:
bashpixi run pip install catboost==1.2.7 --force-reinstall --break-system-packages
Poste mir die Ausgabe — dann kann ich gezielt weiter debuggen.
