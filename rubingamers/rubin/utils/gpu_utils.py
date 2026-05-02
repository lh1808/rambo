"""GPU-Erkennung für CatBoost.

Einmalige Prüfung ob NVIDIA GPU + CUDA verfügbar ist.
``get_gpu_device_count()`` gibt 0 zurück wenn:
  - Kein NVIDIA GPU vorhanden
  - CUDA-Treiber nicht installiert/inkompatibel
  - CatBoost-Paket ohne GPU-Support kompiliert
"""

from __future__ import annotations

import logging
from typing import Dict, Any

_logger = logging.getLogger("rubin.gpu")

# Parameter die auf GPU NICHT unterstützt werden (non-pairwise modes)
# GitHub catboost#983, #1433: rsm/colsample_bylevel nur für Pairwise-Ranking
GPU_INCOMPATIBLE_PARAMS = {"rsm", "colsample_bylevel"}

_CATBOOST_GPU_AVAILABLE: bool | None = None


def catboost_gpu_available() -> bool:
    """Prüft einmalig ob CatBoost GPU-Training tatsächlich funktioniert (cached).

    Zwei Stufen:
    1. ``get_gpu_device_count() > 0`` — GPU-Hardware + Treiber vorhanden
    2. Test-Fit mit ``task_type='GPU'`` — CUDA-Toolkit kompatibel

    Stufe 2 fängt Fehler wie CUDA error 222 ("PTX compiled with unsupported
    toolchain") ab, die auftreten wenn CatBoost mit einer neueren CUDA-Version
    kompiliert wurde als der installierte GPU-Treiber unterstützt.
    """
    global _CATBOOST_GPU_AVAILABLE
    if _CATBOOST_GPU_AVAILABLE is not None:
        return _CATBOOST_GPU_AVAILABLE
    try:
        from catboost.utils import get_gpu_device_count
        n_gpu = get_gpu_device_count()
        if n_gpu == 0:
            _CATBOOST_GPU_AVAILABLE = False
            _logger.debug("CatBoost GPU: Keine GPU erkannt → task_type='CPU'.")
            return False

        # Stufe 2: Test-Fit um CUDA-Kompatibilität zu prüfen
        import numpy as _np
        from catboost import CatBoostRegressor
        _test_model = CatBoostRegressor(
            iterations=1, depth=1, verbose=0,
            task_type="GPU", allow_writing_files=False,
        )
        _X_test = _np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=_np.float32)
        _y_test = _np.array([0.0, 1.0, 0.0], dtype=_np.float32)
        # cat_features=[] explizit setzen: Verhindert, dass der categorical_patch
        # die Indices des Hauptdatensatzes injiziert (Test-Array hat nur 2 Spalten).
        _test_model.fit(_X_test, _y_test, cat_features=[])
        del _test_model, _X_test, _y_test

        _CATBOOST_GPU_AVAILABLE = True
        _logger.info(
            "CatBoost GPU erkannt: %d NVIDIA GPU(s) verfügbar → task_type='GPU'. "
            "Inkompatible Parameter (%s) werden automatisch entfernt.",
            n_gpu, ", ".join(sorted(GPU_INCOMPATIBLE_PARAMS)),
        )
    except Exception as e:
        _CATBOOST_GPU_AVAILABLE = False
        _logger.warning(
            "CatBoost GPU: %d GPU(s) erkannt, aber GPU-Training nicht verfügbar "
            "(CUDA-Inkompatibilität). Fallback auf CPU. Fehler: %s",
            n_gpu if 'n_gpu' in dir() else 0, str(e)[:200],
        )
    return _CATBOOST_GPU_AVAILABLE


def catboost_gpu_params() -> Dict[str, Any]:
    """Gibt GPU-spezifische CatBoost-Parameter zurück, falls GPU verfügbar.

    ``task_type='GPU'``: Training auf GPU.
    ``one_hot_max_size`` wird nicht gesetzt — CatBoost wählt den optimalen
    Default selbst.

    Bekannte GPU-Inkompatibilitäten (werden in den Builder-Funktionen entfernt):
    - ``rsm`` / ``colsample_bylevel``: Nur für Pairwise-Modes auf GPU unterstützt.
      (GitHub catboost#983, #1433)
    """
    if not catboost_gpu_available():
        return {}
    return {
        "task_type": "GPU",
    }
