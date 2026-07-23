from __future__ import annotations

"""Orchestrierung der Production-Pipeline (Scoring).
Die Production-Pipeline:
- lädt ein Bundle,
- wendet das gespeicherte Preprocessing an,
- ruft die gespeicherten Modelle auf,
- schreibt konsistente Scores/Outputs.
Dadurch kann Scoring in einem separaten Job laufen (z. B. Batch),
ohne dass Analyse-spezifische Nebenwirkungen auftreten."""


from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Dict, Any, Iterable, Optional

import pandas as pd
import numpy as np
import pickle

import json
import logging

from rubin.training import _predict_effect, SURROGATE_MODEL_NAME
from rubin.utils.data_utils import load_dtypes_json
from rubin.model_management import read_registry


_logger = logging.getLogger("rubin.production")

class _LazyModelStore(Mapping):
    """Mapping Modellname → entpickeltes Modell, geladen erst beim Zugriff.

    Die verfügbaren Namen stammen aus den .pkl-Dateien im Bundle; geladene
    Modelle werden gecacht. ``loaded_names`` dient Tests und Diagnose."""

    def __init__(self, models_dir: Path) -> None:
        self._dir = models_dir
        self._names = sorted(p.stem for p in models_dir.glob("*.pkl"))
        self._cache: Dict[str, Any] = {}

    def __getitem__(self, name: str) -> Any:
        if name not in self._names:
            raise KeyError(name)
        if name not in self._cache:
            with open(self._dir / f"{name}.pkl", "rb") as f:
                self._cache[name] = pickle.load(f)
        return self._cache[name]

    def __iter__(self):
        return iter(self._names)

    def __contains__(self, name) -> bool:
        # Ohne Override lädt Mapping.__contains__ via __getitem__ — Membership-
        # Checks (Champion-Manifest, has_surrogate) sollen aber nichts laden.
        return name in self._names

    def __len__(self) -> int:
        return len(self._names)

    @property
    def loaded_names(self) -> list:
        return sorted(self._cache)


@dataclass
class ProductionOutputs:
    cate: pd.DataFrame
    metadata: Dict[str, Any]


class ProductionPipeline:
    """Designentscheidung:
- Production ist bewusst „schlank“ gehalten: kein Training, kein Tuning, keine Feature-Auswahl.
- Die Qualität/Kompatibilität wird über Bundle-Artefakte abgesichert (Preprocessor + Schema)."""
    def __init__(self, bundle_path: str) -> None:
        self.last_schema_report = None  # wird beim Scoring gesetzt
        self.bundle_root = Path(bundle_path)
        self.models_dir = self.bundle_root / "models"

        # Harmlose sklearn-Warnung unterdrücken (EconML mischt DataFrame/numpy intern)
        import warnings
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module=r"sklearn\.utils\.validation",
        )
        # EconML nutzt intern den in sklearn veralteten Parameter 'force_all_finite',
        # der in sklearn 1.6 zu 'ensure_all_finite' umbenannt wurde.
        warnings.filterwarnings(
            "ignore",
            message=".*force_all_finite.*was renamed to.*ensure_all_finite",
            category=FutureWarning,
            module=r"sklearn",
        )

        with open(self.bundle_root / "preprocessor.pkl", "rb") as f:
            self.preprocessor = pickle.load(f)

        # Modelle LAZY laden: Ein Bundle enthält alle trainierten Modelle
        # (inkl. Ensemble-Pickle, das seine Mitglieder mitserialisiert) — ein
        # Scoring-Lauf braucht davon typischerweise nur Champion + Surrogate.
        # Eager-Laden würde den Speicherbedarf vervielfachen, ohne Nutzen.
        # Mapping-Interface (in / [] / sorted / keys) bleibt identisch.
        self.models: "_LazyModelStore" = _LazyModelStore(self.models_dir)

        self.champion_model_name: Optional[str] = None
        try:
            manifest = read_registry(self.bundle_root)
            champion = manifest.get("champion")
            if isinstance(champion, str) and champion in self.models:
                self.champion_model_name = champion
        except Exception:
            self.champion_model_name = None

        self.bundle_dtypes = None
        dtypes_path = self.bundle_root / "dtypes.json"
        if dtypes_path.exists():
            try:
                self.bundle_dtypes = load_dtypes_json(str(dtypes_path))
            except Exception:
                self.bundle_dtypes = None

        # ML-Stack-Versionsabgleich: Bundles sind Pickles — Entpickeln/Scoren über
        # abweichende Versionen (v. a. econml) hinweg kann still falsche Ergebnisse
        # oder kryptische Fehler produzieren. Bei Mismatch: Warnung, kein Abbruch.
        # metadata.json und Versions-Stempel sind PFLICHT — jedes exportierte
        # Bundle schreibt beides (artifacts.py); ein Bundle ohne ist beschädigt
        # oder von Hand zusammengebaut und wird abgelehnt.
        meta_path = self.bundle_root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json fehlt im Bundle ({self.bundle_root}). "
                f"Bundles ohne Metadaten werden nicht geladen."
            )
        with open(meta_path, encoding="utf-8") as f:
            self.metadata: Dict[str, Any] = json.load(f)

        self.version_mismatches: Dict[str, tuple] = {}
        exported = self.metadata.get("ml_package_versions")
        if not exported:
            raise ValueError(
                f"ml_package_versions fehlt in metadata.json ({meta_path}). "
                f"Ohne Versions-Stempel ist die Pickle-Kompatibilität nicht prüfbar."
            )
        from importlib.metadata import version as _pkg_version
        for pkg, exp in exported.items():
            if exp is None:
                continue
            try:
                run = _pkg_version(pkg)
            except Exception:
                run = None
            if run != exp:
                self.version_mismatches[pkg] = (exp, run)
        if self.version_mismatches:
            details = ", ".join(f"{k}: Bundle={v[0]} / Laufzeit={v[1]}"
                                for k, v in sorted(self.version_mismatches.items()))
            _logger.warning(
                "Bundle wurde mit anderen Paketversionen exportiert (%s). "
                "Empfehlung: Scoring im selben Environment-Stand wie der Export "
                "(pixi solve-group stellt das für default/prod sicher).", details,
            )

    @property
    def has_surrogate(self) -> bool:
        """Prüft, ob ein Surrogate-Einzelbaum im Bundle vorhanden ist."""
        return SURROGATE_MODEL_NAME in self.models

    def score_surrogate(self, X_raw: pd.DataFrame) -> ProductionOutputs:
        """Scoring ausschließlich mit dem Surrogate-Einzelbaum.

        Convenience-Methode für Kunden, die über den interpretierbaren
        Einzelbaum statt über das vollständige CATE-Modell gescoret werden."""
        if self.has_surrogate:
            return self.score(X_raw, model_names=[SURROGATE_MODEL_NAME])
        raise ValueError(
            f"Das Bundle enthält keinen Surrogate-Einzelbaum ('{SURROGATE_MODEL_NAME}'). "
            f"Verfügbare Modelle: {sorted(self.models.keys())}"
        )

    def score(self, X_raw: pd.DataFrame, model_names: Optional[Iterable[str]] = None) -> ProductionOutputs:
        # Schema-Check vor dem Transform:
        # - fehlende Spalten werden später ergänzt, sollten aber gemeldet werden
        # - zusätzliche Spalten sind i. d. R. unkritisch (werden verworfen), können aber auf Datenänderungen hinweisen
        # Optionale Best-Effort-Typangleichung auf Basis der im Bundle gespeicherten Referenz.
        # Das erleichtert stabile Scorings, wenn Rohdaten aus anderen Exportwegen kommen.
        X_input = X_raw.copy()

        # ── Feature-Guard (analog zum Scoring-Flow in production/run_scoring.py) ──
        # Ohne diesen Check würden fehlende/anders geschriebene Feature-Spalten vom
        # Preprocessor still zu NaN reindexed: GRF-Modelle crashen dann kryptisch
        # ("Input contains NaN"), GBM-Modelle scoren stillschweigend Unsinn.
        _expected = list(self.metadata.get("feature_columns") or [])
        if not _expected and isinstance(self.bundle_dtypes, dict):
            _expected = list(self.bundle_dtypes.keys())
        if _expected:
            _missing = [c for c in _expected if c not in X_input.columns]
            if _missing:
                # rubin-Konvention: DataPrep schreibt alle Spalten uppercase — bei
                # Case-Mismatch werden die Input-Spalten angeglichen, sobald das
                # die Fehlmenge reduziert (auch teilweise: so meldet der Fehler
                # unten präzise nur die WIRKLICH fehlenden Spalten).
                _upper_map = {c: str(c).upper() for c in X_input.columns}
                _resolved = [c for c in _missing if c in set(_upper_map.values())]
                if _resolved:
                    X_input.columns = [_upper_map[c] for c in X_input.columns]
                    logging.getLogger("rubin.production").info(
                        "Feature-Spalten automatisch auf Großschreibung angeglichen "
                        "(Bundle-Konvention; %d Spalten betroffen).", len(_resolved),
                    )
                    _missing = [c for c in _expected if c not in X_input.columns]
            if _missing:
                raise ValueError(
                    f"Eingabedaten enthalten {len(_missing)} vom Bundle erwartete "
                    f"Feature-Spalte(n) nicht: {_missing[:10]}"
                    f"{' …' if len(_missing) > 10 else ''}. "
                    f"Vorhandene Spalten: {list(X_input.columns)[:10]}"
                    f"{' …' if len(X_input.columns) > 10 else ''}. "
                    f"Ohne diese Features würde der Preprocessor sie zu NaN auffüllen "
                    f"und das Scoring unbrauchbare Werte liefern."
                )

        if isinstance(self.bundle_dtypes, dict):
            for col, dt in self.bundle_dtypes.items():
                if col in X_input.columns:
                    try:
                        X_input[col] = X_input[col].astype(dt)
                    except Exception:
                        pass

        # NaN in kategorischen Spalten → explizite Kategorie "fehlend":
        # identische Repräsentation wie im Training (fill_missing_categories in
        # den Trainings-Ladepfaden). Ohne diesen Schritt würde CatBoost-Predict
        # an None/NaN in cat_features scheitern bzw. Scoring-Zeilen mit
        # fehlenden Kategorien anders behandelt als im Training.
        from rubin.utils.data_utils import decode_bytes_categories, fill_missing_categories
        decode_bytes_categories(X_input, logger=self._logger if hasattr(self, "_logger") else None)
        fill_missing_categories(X_input, logger=self._logger if hasattr(self, "_logger") else None)
        try:
            if hasattr(self.preprocessor, "validate"):
                res = self.preprocessor.validate(X_input, strict=False)
                self.last_schema_report = res.to_dict()
        except Exception:
            self.last_schema_report = None

        X = self.preprocessor.transform(X_input)

        selected_names = list(model_names) if model_names is not None else None
        if selected_names is None:
            if self.champion_model_name is not None:
                selected_names = [self.champion_model_name]
            else:
                selected_names = sorted(self.models.keys())

        missing = [name for name in selected_names if name not in self.models]
        if missing:
            raise ValueError(f"Unbekannte Modellnamen im Bundle: {missing}. Verfügbar: {sorted(self.models)}")

        out = pd.DataFrame(index=X.index)
        for name in selected_names:
            model = self.models[name]
            cate = _predict_effect(model, X)
            cate = np.asarray(cate)

            if cate.ndim == 2 and cate.shape[1] > 1:
                # Multi-Treatment: K-1 CATE-Spalten + optimale Zuweisung
                n_effects = cate.shape[1]
                for k in range(n_effects):
                    out[f"cate_{name}_T{k+1}"] = cate[:, k]
                # Optimale Zuweisung
                best_effect = np.max(cate, axis=1)
                best_arm = np.argmax(cate, axis=1) + 1
                out[f"optimal_treatment_{name}"] = np.where(best_effect > 0, best_arm, 0)
                # Confidence: Differenz zwischen bestem und zweitbestem Effekt
                if n_effects > 1:
                    sorted_cate = np.sort(cate, axis=1)[:, ::-1]
                    out[f"treatment_confidence_{name}"] = sorted_cate[:, 0] - sorted_cate[:, 1]
                else:
                    out[f"treatment_confidence_{name}"] = np.abs(cate[:, 0])
            else:
                out[f"cate_{name}"] = cate.reshape(-1)

        bundle_id = self.bundle_root.name
        out["bundle_id"] = bundle_id
        out["model_name"] = selected_names[0] if len(selected_names) == 1 else "multiple"

        return ProductionOutputs(
            cate=out,
            metadata={
                "bundle_id": bundle_id,
                "models_used": list(selected_names),
                "champion_model": self.champion_model_name,
            },
        )
