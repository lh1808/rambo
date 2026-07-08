"""Regressionstest für den Full-Data-Refit aller Modelle beim Bundle-Export.

Sichert konkret den Fix gegen den Production-Fehler
``'CausalForestDML' object has no attribute '_d_t_in'`` ab:

- Vor dem Fix referenzierte die gebündelte ``Ensemble.pkl`` ungefittete
  Mitgliedsmodelle (``_d_t_in`` wird erst in ``fit`` gesetzt). Ein
  ``const_marginal_effect`` in Production warf deshalb einen AttributeError —
  aber nur, wenn das Ensemble NICHT der Champion war.
- Nach dem Fix werden beim Export ALLE Modelle auf vollen Daten refittet und das
  Ensemble self-contained aus den refitteten Mitgliedern neu gebaut.

Der Test umgeht die schwere ``AnalysisPipeline``-Konstruktion: ``_run_bundle_export``
ist eine Methode, die nur ``self._logger`` benötigt (Surrogate-Pfad deaktiviert),
und lässt sich daher über eine ``__new__``-Instanz mit einem minimalen
``SimpleNamespace``-Config direkt aufrufen.
"""

from __future__ import annotations

import json
import logging
import pickle
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Harte Abhängigkeiten des getesteten Pfads — in Minimal-Umgebungen sauberes Skip.
econml = pytest.importorskip("econml")
pytest.importorskip("sklearn")

from econml.dml import LinearDML  # noqa: E402
from econml.score import EnsembleCateEstimator  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402

from rubin.pipelines.analysis_pipeline import AnalysisPipeline  # noqa: E402
from rubin.pipelines.production_pipeline import ProductionPipeline  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────
def _make_data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    T = rng.integers(0, 2, size=n)
    # Heterogener Effekt entlang f0, damit const_marginal_effect variiert.
    Y = (0.5 * X["f0"].to_numpy() + (0.3 + 0.4 * X["f0"].to_numpy()) * T
         + rng.normal(scale=0.1, size=n))
    return X, T, Y


def _make_model(seed: int) -> LinearDML:
    return LinearDML(
        model_y=LinearRegression(),
        model_t=LogisticRegression(max_iter=200),
        discrete_treatment=True,
        cv=2,
        random_state=seed,
    )


def _make_cfg(base_dir, cfg_file, x_file, *, manual_champion=None) -> SimpleNamespace:
    return SimpleNamespace(
        bundle=SimpleNamespace(
            enabled=True, base_dir=str(base_dir), bundle_id=None,
            log_to_mlflow=False,
        ),
        source_config_path=str(cfg_file),
        data_files=SimpleNamespace(x_file=str(x_file)),  # parent hat keine preprocessor.pkl → Schema-Fallback
        selection=SimpleNamespace(
            metric="qini", higher_is_better=True,
            manual_champion=manual_champion,
        ),
        surrogate_tree=SimpleNamespace(enabled=False),
        treatment=SimpleNamespace(type="binary", reference_group=0),
        base_learner=SimpleNamespace(type="catboost"),
    )


def _export_bundle(tmp_path):
    """Baut Modelle + ungefittetes Ensemble, ruft _run_bundle_export auf und gibt
    (bundle_path, original_ensemble, X, T, Y) zurück."""
    X, T, Y = _make_data()

    # Mitglieder UNGEFITTET lassen — exakt der Zustand, den models[name] nach der
    # Cross-Prediction im echten Lauf hat (gefittet werden nur Fold-Deepcopies).
    m_a, m_b = _make_model(1), _make_model(2)
    from rubin.training import ShapeSafeEnsembleCate
    original_ensemble = ShapeSafeEnsembleCate(
        cate_models=[m_a, m_b], member_names=["ModelA", "ModelB"],
        weights=np.array([0.5, 0.5]),
    )
    models = {"ModelA": m_a, "ModelB": m_b, "Ensemble": original_ensemble}

    # Nicht-Ensemble-Champion erzwingen: ModelA hat das beste Qini.
    eval_summary = {
        "ModelA": {"qini": 0.05},
        "ModelB": {"qini": 0.03},
        "Ensemble": {"qini": 0.04},
    }

    base_dir = tmp_path / "bundles"
    base_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = tmp_path / "config_snapshot.yml"
    cfg_file.write_text("# test config snapshot\n", encoding="utf-8")
    x_file = tmp_path / "dataprep" / "X.parquet"  # muss nicht existieren

    cfg = _make_cfg(base_dir, cfg_file, x_file)

    pipe = AnalysisPipeline.__new__(AnalysisPipeline)
    pipe._logger = logging.getLogger("test_bundle_export")

    bundle_id = f"test_{uuid.uuid4().hex[:8]}"
    pipe._run_bundle_export(
        cfg, models, eval_summary,
        X, T, Y,          # Train
        X, T, Y,          # X_full/T_full/Y_full (für den Test identisch)
        list(X.columns),  # selected_feature_columns
        None,             # holdout_data
        True,             # export_bundle
        str(base_dir),    # bundle_dir
        bundle_id,        # bundle_id
        MagicMock(),      # mlflow (log_to_mlflow=False → nicht genutzt)
    )

    bundle_path = base_dir / bundle_id
    assert bundle_path.is_dir(), "Bundle-Verzeichnis wurde nicht erstellt"
    return bundle_path, original_ensemble, X, T, Y


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────
def test_bundled_ensemble_scores_without_d_t_in(tmp_path):
    """Kern-Regression: gebündeltes Ensemble scort in Production ohne _d_t_in-Fehler,
    obwohl das Ensemble nicht der Champion ist und die Original-Mitglieder ungefittet sind."""
    bundle_path, original_ensemble, X, T, Y = _export_bundle(tmp_path)

    # Vorzustand dokumentieren: das Original-Ensemble (ungefittete Mitglieder) wirft.
    with pytest.raises(Exception):
        original_ensemble.const_marginal_effect(X)

    # Production-Scoring des gebündelten Ensembles muss fehlerfrei und endlich sein.
    prod = ProductionPipeline(str(bundle_path))
    assert "Ensemble" in prod.models, "Ensemble.pkl fehlt im Bundle"

    out = prod.score(X, model_names=["Ensemble"])
    cate = out.cate
    assert "cate_Ensemble" in cate.columns
    vals = np.asarray(cate["cate_Ensemble"].to_numpy(), dtype=float)
    assert len(vals) == len(X)
    assert np.all(np.isfinite(vals)), "Ensemble-CATE enthält NaN/Inf"


def test_bundled_ensemble_members_are_fitted(tmp_path):
    """Struktureller Beleg des Fixes: jedes Mitglied der gebündelten Ensemble.pkl
    trägt den Fitted-Marker _d_t_in."""
    bundle_path, *_ = _export_bundle(tmp_path)

    with open(bundle_path / "models" / "Ensemble.pkl", "rb") as f:
        ens_loaded = pickle.load(f)

    members = getattr(ens_loaded, "cate_models", None) or getattr(ens_loaded, "_cate_models", [])
    assert len(members) >= 2, "Ensemble sollte >=2 Mitglieder haben"
    for m in members:
        assert hasattr(m, "_d_t_in"), f"Ensemble-Mitglied {type(m).__name__} ist nicht gefittet"


def test_refit_all_exports_every_model_and_keeps_non_ensemble_champion(tmp_path):
    """Flexibilitäts-Garantie: der Export refittet und bündelt IMMER alle
    trainierten Modelle (frei wählbar); der Champion ist das Nicht-Ensemble-Modell."""
    bundle_path, *_ = _export_bundle(tmp_path)

    for name in ("ModelA", "ModelB", "Ensemble"):
        assert (bundle_path / "models" / f"{name}.pkl").is_file(), f"{name}.pkl fehlt"

    with open(bundle_path / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["champion"] == "ModelA"
    assert set(meta["models"]) >= {"ModelA", "ModelB", "Ensemble"}
    assert meta["champion_refit_on_full_data"] is True

    # Auch ein Nicht-Champion-Einzelmodell muss in Production scorebar sein.
    prod = ProductionPipeline(str(bundle_path))
    out = prod.score(_make_data()[0], model_names=["ModelB"])
    assert "cate_ModelB" in out.cate.columns
    assert np.all(np.isfinite(out.cate["cate_ModelB"].to_numpy()))


def test_shape_safe_ensemble_handles_heterogeneous_member_shapes():
    """Regression E2E-Befund: EconMLs EnsembleCateEstimator crasht bei gemischten
    Modellfamilien (Meta-Learner liefern (n,), DML (n,1)) mit "inhomogeneous shape".
    ShapeSafeEnsembleCate normalisiert die Mitglieder-Outputs vor der Mittelung."""
    from rubin.training import ShapeSafeEnsembleCate

    n = 50

    class _Flat:      # Meta-Learner-Stil: (n,)
        def const_marginal_effect(self, X): return np.full(n, 0.2)

    class _Col:       # DML-Stil: (n, 1)
        def const_marginal_effect(self, X): return np.full((n, 1), 0.4)

    class _Cube:      # ältere EconML-Variante: (n, 1, 1)
        def const_marginal_effect(self, X): return np.full((n, 1, 1), 0.6)

    ens = ShapeSafeEnsembleCate([_Flat(), _Col(), _Cube()], member_names=["A", "B", "C"])
    out = np.asarray(ens.const_marginal_effect(np.zeros((n, 3))))
    assert out.shape == (n,)
    assert np.allclose(out, 0.4)  # Gleichgewichtung: (0.2+0.4+0.6)/3

    # Referenz: das EconML-Original ist auf diesen Mitgliedern nicht nutzbar —
    # es scheitert je nach Version am BaseCateEstimator-Typecheck (Konstruktor)
    # oder am inhomogenen np.average (const_marginal_effect).
    with pytest.raises(Exception):
        econml_ens = EnsembleCateEstimator(cate_models=[_Flat(), _Col(), _Cube()],
                                           weights=np.ones(3) / 3)
        econml_ens.const_marginal_effect(np.zeros((n, 3)))

    # Echte Inkompatibilität (verschiedene Treatment-Dimensionen) wird weiterhin abgelehnt:
    class _MT:
        def const_marginal_effect(self, X): return np.full((n, 2), 0.1)
    with pytest.raises(ValueError):
        ShapeSafeEnsembleCate([_Flat(), _MT()], member_names=["A", "B"]).const_marginal_effect(np.zeros((n, 3)))


def test_predict_effect_normalizes_all_econml_shapes():
    """econml liefert je nach Version/Modell (n, d_y, d_t) mit der
    Singleton-Outcome-Achse in der MITTE — z. B. (n, 1, K-1) bei
    LinearDML/Multi-Treatment. Ein reines Trailing-squeeze(axis=-1) würfe
    dort "cannot select an axis to squeeze out". _predict_effect muss alle
    Varianten auf BT → (n,) bzw. MT → (n, K-1) normalisieren."""
    from rubin.training import _predict_effect

    n = 40

    def mk(shape, fill=0.3):
        class _M:
            def const_marginal_effect(self, X): return np.full(shape, fill)
        return _M()

    X = np.zeros((n, 3))
    # Binary-Treatment-Varianten → (n,)
    for shape in [(n,), (n, 1), (n, 1, 1)]:
        out = _predict_effect(mk(shape), X)
        assert out.shape == (n,), shape
    # Multi-Treatment-Varianten (K-1=2) → (n, 2)
    for shape in [(n, 2), (n, 2, 1), (n, 1, 2)]:  # letzteres: der Crash-Fall
        out = _predict_effect(mk(shape), X)
        assert out.shape == (n, 2), shape
    # Echt unerwartete Form wird klar abgelehnt statt still verbogen:
    with pytest.raises(ValueError):
        _predict_effect(mk((n, 2, 3)), X)


def test_bundle_records_and_checks_ml_package_versions(tmp_path, caplog):
    """Bundles stempeln die ML-Stack-Versionen (Pickle-Kompatibilität); die
    ProductionPipeline vergleicht beim Laden gegen die Laufzeit und warnt bei
    Abweichungen (kein Abbruch)."""
    import logging as _logging

    bundle_path, *_ = _export_bundle(tmp_path)

    meta = json.loads((bundle_path / "metadata.json").read_text(encoding="utf-8"))
    versions = meta.get("ml_package_versions")
    assert versions, "ml_package_versions fehlt in metadata.json"
    from importlib.metadata import version as _v
    assert versions["econml"] == _v("econml")
    assert versions["numpy"] == _v("numpy")

    # 1) Übereinstimmende Versionen → keine Mismatches, keine Warnung
    with caplog.at_level(_logging.WARNING, logger="rubin.production"):
        pipe = ProductionPipeline(str(bundle_path))
    assert pipe.version_mismatches == {}

    # 2) Manipulierter Stempel → Warnung + version_mismatches gefüllt
    meta["ml_package_versions"]["econml"] = "0.0.1"
    (bundle_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    caplog.clear()
    with caplog.at_level(_logging.WARNING, logger="rubin.production"):
        pipe2 = ProductionPipeline(str(bundle_path))
    assert "econml" in pipe2.version_mismatches
    assert pipe2.version_mismatches["econml"][0] == "0.0.1"
    assert any("anderen Paketversionen" in r.message for r in caplog.records)

    # 3) Bundle ohne Stempel → wird abgelehnt (Stempel ist Pflichtbestandteil)
    del meta["ml_package_versions"]
    (bundle_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="ml_package_versions"):
        ProductionPipeline(str(bundle_path))

    # 4) Bundle ohne metadata.json → wird abgelehnt
    (bundle_path / "metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        ProductionPipeline(str(bundle_path))


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble-Mitgliedschaft: evaluiert == exportiert (member_names)
# ──────────────────────────────────────────────────────────────────────────────
def test_shape_safe_ensemble_member_names_roundtrip():
    """member_names ist Pflicht, wird persistiert und muss zu cate_models passen."""
    from rubin.training import ShapeSafeEnsembleCate

    m_a, m_b = _make_model(1), _make_model(2)
    ens = ShapeSafeEnsembleCate(cate_models=[m_a, m_b], member_names=["ModelA", "ModelB"])
    assert ens.member_names == ["ModelA", "ModelB"]

    # Pickle-Roundtrip erhält die Namen
    ens2 = pickle.loads(pickle.dumps(ens))
    assert ens2.member_names == ["ModelA", "ModelB"]

    # Pflichtparameter: Konstruktion ohne member_names ist ein Programmierfehler
    with pytest.raises(TypeError):
        ShapeSafeEnsembleCate(cate_models=[m_a, m_b])

    # Längen-Mismatch wird abgelehnt
    with pytest.raises(ValueError):
        ShapeSafeEnsembleCate(cate_models=[m_a, m_b], member_names=["NurEins"])


def _export_bundle_with_membership(tmp_path, member_names):
    """Wie _export_bundle, aber mit drei Modellen im models-Dict — member_names
    steuert die evaluierte Ensemble-Zusammensetzung (echte Teilmenge)."""
    from rubin.training import ShapeSafeEnsembleCate

    X, T, Y = _make_data()
    m_a, m_b, m_c = _make_model(1), _make_model(2), _make_model(3)
    _by_name = {"ModelA": m_a, "ModelB": m_b, "ModelC": m_c}
    original_ensemble = ShapeSafeEnsembleCate(
        cate_models=[_by_name[n] for n in member_names],
        member_names=member_names,
    )
    models = {"ModelA": m_a, "ModelB": m_b, "ModelC": m_c, "Ensemble": original_ensemble}
    eval_summary = {n: {"qini": 0.05 - i * 0.01} for i, n in enumerate(models)}

    base_dir = tmp_path / "bundles"
    base_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = tmp_path / "config_snapshot.yml"
    cfg_file.write_text("# test config snapshot\n", encoding="utf-8")
    x_file = tmp_path / "dataprep" / "X.parquet"

    cfg = _make_cfg(base_dir, cfg_file, x_file)
    pipe = AnalysisPipeline.__new__(AnalysisPipeline)
    pipe._logger = logging.getLogger("test_bundle_export_membership")

    bundle_id = f"test_{uuid.uuid4().hex[:8]}"
    pipe._run_bundle_export(
        cfg, models, eval_summary,
        X, T, Y, X, T, Y,
        list(X.columns), None, True, str(base_dir), bundle_id, MagicMock(),
    )
    return base_dir / bundle_id


def test_bundle_ensemble_rebuild_honors_member_names(tmp_path):
    """Kern-Invariante: Das exportierte Ensemble enthält EXAKT die Mitglieder des
    evaluierten Ensembles (member_names) — auch wenn models weitere Modelle enthält."""
    bundle_path = _export_bundle_with_membership(tmp_path, member_names=["ModelA", "ModelB"])
    with open(bundle_path / "models" / "Ensemble.pkl", "rb") as f:
        bundled = pickle.load(f)
    assert bundled.member_names == ["ModelA", "ModelB"]
    assert len(bundled.cate_models) == 2, (
        "Export-Ensemble weicht von der evaluierten Zusammensetzung ab "
        f"({len(bundled.cate_models)} statt 2 Mitglieder)"
    )
