"""Die ausgelieferten Scoring-Configs müssen dauerhaft valide bleiben, und
die Loader müssen fehlerhafte Configs mit klaren Meldungen ablehnen.

Zwei Sammeltests statt vieler Mikrotests: (1) alle Positiv-Verträge der
ausgelieferten YAMLs, (2) alle Ablehnungs-Verträge der Validierung —
Tippfehler-Schlüssel, Kreuz-Runner, leere id_columns, Top-Level-bundle
neben scores, fehlendes monitoring.dir (saspy)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROD = Path(__file__).resolve().parents[1] / "production"
sys.path.insert(0, str(PROD))

from run_scoring import load_scoring_config  # noqa: E402
from run_scoring_saspy import load_saspy_scoring_config  # noqa: E402


def test_shipped_configs_valid():
    files = sorted(PROD.glob("*.yml"))
    assert files, "Keine Scoring-Configs in production/ gefunden"
    for p in files:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert isinstance(raw, dict), f"{p.name}: kein YAML-Mapping"
        assert raw.get("runner") in ("file", "saspy"), (
            f"{p.name}: Top-Level-Key 'runner' fehlt oder ungültig")
    for name in ("scoring_template_file.yml", "scoring_ph.yml"):
        cfg = load_scoring_config(str(PROD / name))
        assert cfg["runner"] == "file" and cfg["id_columns"]
    cfg = load_saspy_scoring_config(str(PROD / "scoring_template_saspy.yml"))
    assert cfg["runner"] == "saspy"
    assert cfg["output"]["write_mode"] in ("replace", "append") and cfg["monitoring"]["dir"]


def test_loaders_reject_invalid_configs(tmp_path):
    def w(name, d):
        p = tmp_path / name
        p.write_text(yaml.safe_dump(d), encoding="utf-8")
        return str(p)

    file_ok = {"name": "x", "bundle": "/b", "id_columns": ["ID"],
               "input": {"path": "/p"}, "output": {"xpt_path": "/o"}}
    saspy_ok = {"name": "x", "bundle": "/b", "id_columns": ["ID"],
                "input": {"libref": "A", "table": "B"},
                "output": {"libref": "C", "table": "D"}, "monitoring": {"dir": "/m"}}

    # Schlüssel-Tippfehler → Vorschlag (beide Flows)
    bad = {**file_ok, "scoring": {"score_p_modell": "X"}}
    with pytest.raises(ValueError, match="score_p_model"):
        load_scoring_config(w("t1.yml", bad))
    bad = {**saspy_ok, "input": {"libref": "A", "table": "B", "chunksize": 1}}
    with pytest.raises(ValueError, match="chunk_size"):
        load_saspy_scoring_config(w("t2.yml", bad))

    # meta_columns bleibt Free-Form (Positiv-Gegenprobe zur Schlüsselprüfung)
    ok = {**file_ok, "output": {"xpt_path": "/o", "meta_columns": {"BELIEBIG": 1}}}
    assert load_scoring_config(w("t3.yml", ok))["output"]["meta_columns"]["BELIEBIG"] == 1

    # Kreuz-Runner → nennt den richtigen Einstieg (beide Richtungen)
    with pytest.raises(ValueError, match="run_scoring_saspy.py"):
        load_scoring_config(str(PROD / "scoring_template_saspy.yml"))
    with pytest.raises(ValueError, match="run_scoring.py"):
        load_saspy_scoring_config(str(PROD / "scoring_ph.yml"))

    # id_columns Pflicht (beide Flows)
    with pytest.raises(ValueError, match="id_columns"):
        load_scoring_config(w("t4.yml", {k: v for k, v in file_ok.items() if k != "id_columns"}))
    with pytest.raises(ValueError, match="id_columns"):
        load_saspy_scoring_config(w("t5.yml", {**saspy_ok, "id_columns": []}))

    # scores-Liste: Top-Level-bundle verboten
    bad = {**file_ok, "scores": [{"name": "a", "bundle": "/b",
                                  "output": {"xpt_path": "/o2"}}]}
    with pytest.raises(ValueError, match="Top-Level"):
        load_scoring_config(w("t6.yml", bad))

    # saspy: monitoring.dir Pflicht
    with pytest.raises(ValueError, match="monitoring.dir"):
        load_saspy_scoring_config(w("t7.yml", {k: v for k, v in saspy_ok.items()
                                               if k != "monitoring"}))
