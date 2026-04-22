"""
Config-Loader: lädt ``GlobalConfig`` mit optionalem YAML-Overlay und
Env-Var-Overrides.

Design
------
Die dataclasses in ``config.py`` sind die kanonische Form. YAML und
Env-Vars überschreiben punktuell nur, was sie angeben – alles andere bleibt
bei den dataclass-Defaults.

Reihenfolge (später schlägt früher):
    1. dataclass-Defaults
    2. YAML-Datei (falls Pfad angegeben und Datei existiert)
    3. Env-Vars mit Prefix ``PLUTO__`` (z. B. ``PLUTO__TFT__N_EPOCHS=50``)

Unbekannte Feldnamen in YAML oder Env werfen ``ValueError`` – Tippfehler
fallen damit sofort auf statt still ignoriert zu werden.

Grenzen (bewusst minimal gehalten)
----------------------------------
- Neue Einträge in Dict-Feldern (``horizons``, ``tuning``) müssen
  vollständig angegeben werden; partielle Overrides gehen nur auf
  existierende Keys.
- Env-Vars casten auf die Primitiv-Typen bool/int/float/str, keine Listen
  oder Dicts. Listen/Dicts gehören in die YAML.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Type, Union

import yaml

from .config import (
    GlobalConfig,
    HorizonConfig,
    TuningConfig,
)

ENV_PREFIX = "PLUTO__"
ENV_SEP = "__"

# Für Dict-Felder in GlobalConfig: welchen dataclass-Typ haben ihre Values?
# Nötig, wenn per YAML ein NEUER Key (z. B. horizons: 26) angelegt wird.
_DICT_VALUE_TYPES: dict[str, Type] = {
    "horizons": HorizonConfig,
    "tuning": TuningConfig,
}


def _cast_dict_key(key: Any, existing_keys) -> Any:
    """Wenn die Dict-Keys im Default int sind, YAML-Strings nach int casten."""
    if existing_keys and all(isinstance(k, int) for k in existing_keys):
        try:
            return int(key)
        except (ValueError, TypeError):
            return key
    return key


def _deep_merge_into_dataclass(instance: Any, overrides: Mapping[str, Any], path: str = "") -> None:
    """
    Aktualisiert ein dataclass-Objekt rekursiv aus einem Mapping.

    Unbekannte Keys → ``ValueError`` mit voll qualifiziertem Pfad.
    """
    if not dataclasses.is_dataclass(instance):
        raise TypeError(f"Kein dataclass: {type(instance).__name__} (Pfad: {path or '<root>'})")

    field_map = {f.name: f for f in dataclasses.fields(instance)}

    for key, value in overrides.items():
        sub_path = f"{path}.{key}" if path else key

        if key not in field_map:
            valid = ", ".join(sorted(field_map.keys()))
            raise ValueError(
                f"Unbekanntes Config-Feld '{sub_path}' auf "
                f"{type(instance).__name__}. Erlaubt: {valid}"
            )

        current = getattr(instance, key)

        # Nested dataclass → rekursiv mergen
        if dataclasses.is_dataclass(current) and isinstance(value, Mapping):
            _deep_merge_into_dataclass(current, value, sub_path)
            continue

        # Dict-Feld (z. B. horizons, tuning) → per-Key mergen
        if isinstance(current, dict) and isinstance(value, Mapping):
            value_type = _DICT_VALUE_TYPES.get(key)
            for sub_key, sub_value in value.items():
                cast_key = _cast_dict_key(sub_key, current.keys())
                sub_sub_path = f"{sub_path}.{sub_key}"

                if cast_key in current and dataclasses.is_dataclass(current[cast_key]) and isinstance(sub_value, Mapping):
                    _deep_merge_into_dataclass(current[cast_key], sub_value, sub_sub_path)
                elif value_type is not None and isinstance(sub_value, Mapping):
                    # Neuer Dict-Key mit bekanntem Value-Typ → dataclass bauen
                    try:
                        current[cast_key] = value_type(**sub_value)
                    except TypeError as e:
                        raise ValueError(
                            f"Ungültige Felder für neuen Eintrag '{sub_sub_path}' "
                            f"({value_type.__name__}): {e}"
                        ) from e
                else:
                    current[cast_key] = sub_value
            continue

        # Primitive / Listen / sonstiges → direkt setzen
        setattr(instance, key, value)


def _parse_env_value(raw: str) -> Any:
    """Minimales Parsing: bool → int → float → str."""
    low = raw.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _set_by_path(obj: Any, path_parts: list[str], value: Any, full_path: str) -> None:
    """Navigiert durch Config-Struktur und setzt den Wert am Ende."""
    for i, part in enumerate(path_parts):
        is_last = i == len(path_parts) - 1

        if dataclasses.is_dataclass(obj):
            field_map = {f.name: f for f in dataclasses.fields(obj)}
            if part not in field_map:
                valid = ", ".join(sorted(field_map.keys()))
                raise ValueError(
                    f"Unbekanntes Feld im Env-Pfad '{full_path}' "
                    f"auf {type(obj).__name__}: '{part}'. Erlaubt: {valid}"
                )
            if is_last:
                setattr(obj, part, value)
            else:
                obj = getattr(obj, part)

        elif isinstance(obj, dict):
            cast_key = _cast_dict_key(part, obj.keys())
            if cast_key not in obj:
                raise ValueError(
                    f"Unbekannter Dict-Key im Env-Pfad '{full_path}': '{part}'"
                )
            if is_last:
                obj[cast_key] = value
            else:
                obj = obj[cast_key]
        else:
            raise ValueError(
                f"Kann Env-Pfad '{full_path}' nicht weiter navigieren "
                f"(bei '{part}', Typ {type(obj).__name__})"
            )


def _apply_env_overrides(cfg: GlobalConfig, env: Optional[Mapping[str, str]] = None) -> None:
    """
    Wendet alle Env-Vars mit Präfix ``PLUTO__`` als Overrides auf ``cfg`` an.

    Keys werden am Doppel-Unterstrich gesplittet und gegen die
    Dataclass-Struktur aufgelöst; Werte werden vor dem Setzen über
    :func:`_parse_env_value` auf bool/int/float/str gecastet. Unbekannte
    Pfade führen zu ``ValueError``.
    """
    env = env if env is not None else os.environ
    for key, raw_value in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path_parts = key[len(ENV_PREFIX):].lower().split(ENV_SEP)
        path_parts = [p for p in path_parts if p]
        if not path_parts:
            continue
        _set_by_path(cfg, path_parts, _parse_env_value(raw_value), full_path=key)


def load_config(
    yaml_path: Union[str, Path, None] = None,
    apply_env: bool = True,
) -> GlobalConfig:
    """
    Lädt die GlobalConfig mit optionalem YAML-Overlay und Env-Var-Overrides.

    Parameters
    ----------
    yaml_path
        Pfad zur YAML-Datei. Wenn ``None``, nur Defaults + Env.
        Wenn Pfad nicht existiert, wird er still übergangen – so kann
        ``load_config("config.yaml")`` in jeder Umgebung laufen, auch wenn
        keine Config-Datei hinterlegt ist.
    apply_env
        Ob Env-Vars mit Prefix ``PLUTO__`` angewendet werden.

    Raises
    ------
    ValueError
        Bei unbekannten Feldnamen in YAML oder Env-Vars.
    """
    cfg = GlobalConfig()

    if yaml_path is not None:
        path = Path(yaml_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"YAML {path} muss ein Mapping (Dict) auf oberster Ebene sein, "
                    f"ist {type(raw).__name__}"
                )
            _deep_merge_into_dataclass(cfg, raw)

    if apply_env:
        _apply_env_overrides(cfg)

    return cfg
