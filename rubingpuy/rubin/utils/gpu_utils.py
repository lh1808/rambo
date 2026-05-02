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

_CATBOOST_GPU_AVAILABLE: bool | None = None


def catboost_gpu_available() -> bool:
    """Prüft einmalig ob CatBoost GPU-Training verfügbar ist (cached)."""
    global _CATBOOST_GPU_AVAILABLE
    if _CATBOOST_GPU_AVAILABLE is not None:
        return _CATBOOST_GPU_AVAILABLE
    try:
        from catboost.utils import get_gpu_device_count
        n_gpu = get_gpu_device_count()
        _CATBOOST_GPU_AVAILABLE = n_gpu > 0
        if _CATBOOST_GPU_AVAILABLE:
            _logger.info(
                "CatBoost GPU erkannt: %d NVIDIA GPU(s) verfügbar → task_type='GPU'.",
                n_gpu,
            )
        else:
            _logger.debug("CatBoost GPU: Keine GPU erkannt → task_type='CPU'.")
    except Exception:
        _CATBOOST_GPU_AVAILABLE = False
    return _CATBOOST_GPU_AVAILABLE


def catboost_gpu_params() -> Dict[str, Any]:
    """Gibt GPU-spezifische CatBoost-Parameter zurück, falls GPU verfügbar.

    ``task_type='GPU'``: Training auf GPU.
    ``one_hot_max_size`` wird NICHT gesetzt — CatBoost wählt den optimalen
    Default selbst (2 bei supervisierten Tasks mit Target-Daten, identisch
    mit CPU-Default).
    """
    if not catboost_gpu_available():
        return {}
    return {
        "task_type": "GPU",
    }
