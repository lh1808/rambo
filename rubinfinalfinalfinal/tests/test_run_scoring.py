"""Tests für production/run_scoring.py: Config → Scoring → XPT + Monitoring-JSON.

Nutzt das Bundle-Harness aus test_bundle_ensemble_refit (echte, refittete
Modelle inkl. Ensemble; Schema-Fallback-Preprocessor ohne kategoriale Spalten).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("econml")
pytest.importorskip("pyreadstat")
import pyreadstat  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production"))
from run_scoring import load_scoring_config, run_scoring  # noqa: E402

from tests.test_bundle_ensemble_refit import _export_bundle, _make_data  # noqa: E402


def _write_cfg(tmp_path, bundle_path, *, score_b="SurrogateTree", extra=None,
               fmt_version=5, round_dec=6) -> str:
    X, _, _ = _make_data()
    inp = tmp_path / "scoring_input.parquet"
    df = X.copy()
    df.insert(0, "PARTNER_ID_V", np.arange(1_000_001, 1_000_001 + len(df)))
    df["UNUSED_EXTRA_COL"] = "x"          # muss verworfen werden
    df.loc[df.index[:3], "f0"] = np.inf   # Inf-Handling
    df.to_parquet(inp)

    cfg_text = f"""
name: kausalscore_test
bundle: {bundle_path}
input:
  path: {inp}
id_columns: [PARTNER_ID_V]
scoring:
  score_p_model: champion
  score_b_model: {"null" if score_b is None else score_b}
  extra_models: {list(extra or [])}
  batch_size: 128
  round_decimals: {round_dec}
output:
  xpt_path: {tmp_path / "out" / "kausalscore_test.xpt"}
  table_name: kau_test
  file_format_version: {fmt_version}
  meta_columns:
    SCORE_TYP: MO_SCORE_TEST
    VALID_TO: 31DEC9999
    SCORE_VERFAHREN: kausal
  column_order: [PARTNER_ID_V, SCORE_TYP, VALID_TO, SCORE_P, SCORE_B, SCORE_VERFAHREN, TIMESTAMP]
"""
    p = tmp_path / "scoring_test.yml"
    p.write_text(cfg_text, encoding="utf-8")
    return str(p)


def test_scoring_end_to_end_with_monitoring(tmp_path):
    bundle_path, *_ = _export_bundle(tmp_path)
    cfg = load_scoring_config(_write_cfg(tmp_path, bundle_path, score_b=None,
                                         extra=["Ensemble"]))
    report = run_scoring(cfg)

    # XPT existiert und ist per pyreadstat rücklesbar
    xpt = Path(cfg["output"]["xpt_path"])
    assert xpt.is_file()
    back, meta = pyreadstat.read_xport(str(xpt))
    assert len(back) == 300
    # XPT V5 kappt Namen auf 8 Zeichen (wie schon beim Alt-Skript):
    assert "SCORE_P" in back.columns and "PARTNER_" in back.columns
    assert "CATE_Ens" in back.columns
    assert list(back.columns)[0] == "PARTNER_"
    assert back["SCORE_TY"].iloc[0] == "MO_SCORE_TEST"

    # Rundung angewendet (Zeile 10: unbeeinflusst von der Inf-Injektion in Zeilen 0-2)
    v = float(back["SCORE_P"].iloc[10])
    assert np.isfinite(v) and v == round(v, 6)
    # Die Inf-Zeilen: Inf→NaN ersetzt; das Harness-Bundle nutzt den Schema-
    # Fallback-Preprocessor OHNE gelernte Imputation → NaN-Scores (mit echtem
    # FittedPreprocessor griffe hier die Median-Imputation). Monitoring zählt sie:

    # Monitoring: versionierte Datei + latest, Kerninhalte
    mon_dir = xpt.parent / "monitoring"
    files = sorted(mon_dir.glob("kausalscore_test_*.json"))
    assert files and (mon_dir / "kausalscore_test_latest.json").is_file()
    mon = json.loads((mon_dir / "kausalscore_test_latest.json").read_text(encoding="utf-8"))
    assert mon["input"]["n_rows"] == 300
    assert mon["preprocessing"]["inf_cells_replaced_with_nan"] == 3
    # Spalten-Pruning (Default): die Extra-Spalte wird gar nicht erst gelesen
    assert mon["input"]["column_pruning"] is True
    assert mon["preprocessing"]["extra_input_columns_dropped"] == 0
    assert "minus1_rate_per_categorical" in mon["preprocessing"]
    assert mon["bundle"]["champion"] == report["models"]["score_p"] == "ModelA"
    assert set(mon["scores"]) == {"SCORE_P", "CATE_Ensemble"}
    for stats in mon["scores"].values():
        assert stats["n"] == 300 and np.isfinite(stats["median"])
    assert mon["scores"]["SCORE_P"]["nan"] == 3
    assert mon["bundle"]["ml_package_versions"]["econml"]
    assert mon["output"]["v5_name_truncations"]["PARTNER_ID_V"] == "PARTNER_"


def test_score_b_optional_when_surrogate_missing(tmp_path):
    """Bundle OHNE SurrogateTree: SCORE_B entfällt still statt Fehler."""
    bundle_path, *_ = _export_bundle(tmp_path)  # Harness exportiert keinen Surrogate
    cfg = load_scoring_config(_write_cfg(tmp_path, bundle_path, score_b="SurrogateTree",
                                         fmt_version=8))
    report = run_scoring(cfg)
    assert report["models"]["score_b"] is None
    back, _ = pyreadstat.read_xport(cfg["output"]["xpt_path"])
    assert "SCORE_B" not in back.columns
    # V8: 32-Zeichen-Namen → ungekürzt
    assert "SCORE_P" in back.columns and "PARTNER_ID_V" in back.columns


def test_unknown_score_p_model_raises(tmp_path):
    bundle_path, *_ = _export_bundle(tmp_path)
    cfg = load_scoring_config(_write_cfg(tmp_path, bundle_path))
    cfg["scoring"]["score_p_model"] = "GibtEsNicht"
    with pytest.raises(ValueError, match="GibtEsNicht"):
        run_scoring(cfg)


def test_pruning_disabled_reads_and_drops_extra_columns(tmp_path):
    """pull_only_needed_columns: false → volle Tabelle wird gelesen; der
    Preprocessor droppt Extra-Spalten beim Transform (Monitoring zählt sie)."""
    bundle_path, *_ = _export_bundle(tmp_path)
    cfg_path = _write_cfg(tmp_path, bundle_path)
    import yaml as _yaml
    raw = _yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    raw["input"]["pull_only_needed_columns"] = False
    Path(cfg_path).write_text(_yaml.safe_dump(raw), encoding="utf-8")

    cfg = load_scoring_config(cfg_path)
    report = run_scoring(cfg)
    assert report["input"]["column_pruning"] is False
    assert report["preprocessing"]["extra_input_columns_dropped"] >= 1


def test_concrete_model_selection_independent_of_champion(tmp_path):
    """Die Modellwahl ist YAML-gesteuert und unabhängig vom Champion:
    score_p_model mit konkretem Namen scort exakt dieses Modell (nicht den
    Champion), Monitoring dokumentiert es."""
    bundle_path, _, X, _, _ = _export_bundle(tmp_path)
    cfg_path = _write_cfg(tmp_path, bundle_path, score_b=None)
    import yaml as _yaml
    raw = _yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    raw["scoring"]["score_p_model"] = "ModelB"   # Champion ist ModelA
    Path(cfg_path).write_text(_yaml.safe_dump(raw), encoding="utf-8")

    report = run_scoring(load_scoring_config(cfg_path))
    assert report["models"]["score_p"] == "ModelB"
    assert report["bundle"]["champion"] == "ModelA"

    # SCORE_P entspricht dem direkten ModelB-Scoring (und nicht dem Champion)
    back, _ = pyreadstat.read_xport(str(Path(raw["output"]["xpt_path"])))
    from rubin.pipelines.production_pipeline import ProductionPipeline
    pipe = ProductionPipeline(str(bundle_path))
    ref_b = pipe.score(X, model_names=["ModelB"]).cate["cate_ModelB"].to_numpy().round(6)
    ref_a = pipe.score(X, model_names=["ModelA"]).cate["cate_ModelA"].to_numpy().round(6)
    # Zeilen 0–2 tragen die vom Harness injizierten Inf-Werte (→ NaN-Scores
    # ohne gelernte Imputation) — Vergleich daher ab Zeile 3.
    got = back["SCORE_P"].to_numpy()[3:]
    np.testing.assert_allclose(got, ref_b[3:], atol=1e-9)
    assert not np.allclose(got, ref_a[3:])


def test_missing_feature_columns_guard(tmp_path, caplog):
    """Zwei Richtungen eines Vertrags: Fehlen ALLE Bundle-Features (klassischer
    uppercase-Mismatch), bricht der Lauf hart ab — keine NaN-Score-XPT.
    Fehlen einzelne, läuft er mit Warnung + NaN-Ergänzung durch."""
    import logging
    import yaml as _yaml
    import pandas as pd

    bundle_path, *_ = _export_bundle(tmp_path)
    cfg_path = _write_cfg(tmp_path, bundle_path)
    raw = _yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))

    # a) alle fehlen (uppercase-Mismatch) → Abbruch, nichts geschrieben
    raw["input"]["uppercase_columns"] = True   # Bundle erwartet f0..f2 (klein)
    raw["input"]["pull_only_needed_columns"] = False
    Path(cfg_path).write_text(_yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="KEINE der vom Bundle erwarteten"):
        run_scoring(load_scoring_config(cfg_path))
    assert not Path(raw["output"]["xpt_path"]).exists()

    # b) eine fehlt → Warnung + Lauf geht durch, Monitoring nennt sie
    raw["input"]["uppercase_columns"] = False
    Path(cfg_path).write_text(_yaml.safe_dump(raw), encoding="utf-8")
    inp = pd.read_parquet(raw["input"]["path"]).drop(columns=["f2"])
    inp.to_parquet(raw["input"]["path"])
    with caplog.at_level(logging.WARNING, logger="rubin.scoring"):
        report = run_scoring(load_scoring_config(cfg_path))
    assert report["preprocessing"]["missing_expected_columns"] == ["f2"]
    assert any("als NaN ergänzt" in r.message for r in caplog.records)


def test_scores_mode_contracts(tmp_path, monkeypatch):
    """Die zwei Kern-Verträge des Ein-Lade-Modus: (1) Konfigurationsfehler
    (nicht ladbares Bundle) brechen ATOMAR ab — nichts wird gelesen oder
    geschrieben. (2) Ein gültiger Lauf lädt die Quelle genau EINMAL und
    verarbeitet sie pro Eintrag unterschiedlich (Champion vs. Challenger)."""
    import yaml as _yaml
    import run_scoring as rs

    bundle_path, _, X, _, _ = _export_bundle(tmp_path)
    inp = tmp_path / "gesamt.parquet"
    d = X.copy(); d.insert(0, "PARTNER_ID_V", np.arange(len(d)) + 1)
    d.to_parquet(inp)
    base = {"name": "einladung", "input": {"path": str(inp)},
            "id_columns": ["PARTNER_ID_V"], "monitoring": {"dir": str(tmp_path / "mon")}}

    # (1) Atomarität
    cfg_path = tmp_path / "atomar.yml"
    cfg_path.write_text(_yaml.safe_dump({**base, "scores": [
        {"name": "ok", "bundle": str(bundle_path),
         "output": {"xpt_path": str(tmp_path / "a0.xpt"), "table_name": "a"}},
        {"name": "kaputt", "bundle": str(tmp_path / "existiert_nicht"),
         "output": {"xpt_path": str(tmp_path / "k.xpt"), "table_name": "k"}},
    ]}), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        rs.run_scoring(rs.load_scoring_config(str(cfg_path)))
    assert not (tmp_path / "a0.xpt").exists() and not (tmp_path / "mon").exists()

    # (2) genau EIN Ladevorgang, zwei unterschiedliche Verarbeitungen
    calls = {"n": 0}
    orig = rs.read_input
    def counting(icfg, columns=None):
        calls["n"] += 1
        return orig(icfg, columns=columns)
    monkeypatch.setattr(rs, "read_input", counting)
    cfg_path.write_text(_yaml.safe_dump({**base, "scores": [
        {"name": "champ", "bundle": str(bundle_path),
         "output": {"xpt_path": str(tmp_path / "a.xpt"), "table_name": "a"}},
        {"name": "chall", "bundle": str(bundle_path),
         "scoring": {"score_p_model": "ModelB"},
         "output": {"xpt_path": str(tmp_path / "b.xpt"), "table_name": "b"}},
    ]}), encoding="utf-8")
    reports = rs.run_scoring(rs.load_scoring_config(str(cfg_path)))
    assert calls["n"] == 1 and len(reports) == 2
    assert [r["models"]["score_p"] for r in reports] == ["ModelA", "ModelB"]
    a, _ = pyreadstat.read_xport(str(tmp_path / "a.xpt"))
    b, _ = pyreadstat.read_xport(str(tmp_path / "b.xpt"))
    assert not np.allclose(a["SCORE_P"], b["SCORE_P"])
    assert (tmp_path / "mon" / "einladung_champ_latest.json").is_file()
    assert (tmp_path / "mon" / "einladung_chall_latest.json").is_file()

