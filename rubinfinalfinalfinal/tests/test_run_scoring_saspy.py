"""Tests für production/run_scoring_saspy.py: SAS-Library → Bundle → SAS-Library.

Der saspy-Layer wird über eine FakeSASsession simuliert (kein SAS nötig);
das Bundle ist echt (Harness aus test_bundle_ensemble_refit, refittete
Modelle inkl. Ensemble). Verifiziert werden Chunked-Pull (firstobs/obs),
Scoring-Konsistenz gegen die ProductionPipeline, Chunked-Write-Back
(replace/append, PROC APPEND), SAS-Log-Fehlerbehandlung, Session-Teardown
und das Monitoring-JSON.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("econml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production"))
from run_scoring_saspy import (  # noqa: E402
    load_saspy_scoring_config,
    run_scoring_saspy,
)

from tests.test_bundle_ensemble_refit import _export_bundle, _make_data  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# FakeSAS: minimale saspy-Oberfläche (sd2df, df2sd, submit, endsas)
# ──────────────────────────────────────────────────────────────────────────────
class FakeSASsession:
    def __init__(self, source_df: pd.DataFrame, fail_on_append: bool = False):
        self._source = source_df.reset_index(drop=True)
        self.fail_on_append = fail_on_append
        self.submitted: list[str] = []
        self.written: list[pd.DataFrame] = []
        self.ended = False

    # saspy: firstobs/obs sind 1-basierte, inklusive Beobachtungsnummern;
    # keep= filtert Spalten (SAS: case-insensitiv)
    def sd2df(self, table, libref, dsopts=None, **_):
        dsopts = dsopts or {}
        self.last_dsopts = dict(dsopts)
        first = int(dsopts.get("firstobs", 1))
        last = int(dsopts.get("obs", len(self._source)))
        part = self._source.iloc[first - 1: last]
        keep = dsopts.get("keep")
        if keep:
            wanted = {c.upper() for c in keep.split()}
            part = part[[c for c in part.columns if c.upper() in wanted]]
        return part.reset_index(drop=True)

    def df2sd(self, df, table, libref, **_):
        self.written.append(df.copy())
        return object()  # SASdata-Platzhalter (nicht None = Erfolg)

    def submit(self, code):
        self.submitted.append(code)
        if self.fail_on_append and "proc append" in code.lower():
            return {"LOG": "ERROR: File SCOREOUT.KAUSALSCORE.DATA is locked."}
        return {"LOG": "NOTE: PROCEDURE beendet."}

    def endsas(self):
        self.ended = True


def _install_fake_saspy(monkeypatch, session: FakeSASsession):
    mod = types.ModuleType("saspy")
    mod.SASsession = lambda **kw: session
    monkeypatch.setitem(sys.modules, "saspy", mod)


def _write_cfg(tmp_path, bundle_path, **overrides) -> str:
    cfg = {
        "name": "saspy_test",
        "bundle": str(bundle_path),
        "saspy": {"cfgname": "testcfg"},
        "input": {"libref": "SCORING", "table": "eingang", "chunk_size": 100},
        "id_columns": ["KUNDE_ID"],
        "scoring": {"score_p_model": "champion", "score_b_model": None,
                    "batch_size": 500, "round_decimals": 6},
        "output": {"libref": "SCOREOUT", "table": "kausalscore",
                   "write_mode": "replace", "write_chunk_size": 120,
                   "meta_columns": {"SCORE_TYP": "TEST"}},
        "monitoring": {"dir": str(tmp_path / "monitoring")},
    }
    for key, val in overrides.items():
        cfg[key] = {**cfg.get(key, {}), **val} if isinstance(val, dict) else val
    p = tmp_path / "scoring_saspy_test.yml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(p)


def _make_sas_source(n=300) -> pd.DataFrame:
    """SAS-seitige Tabelle: Bundle-Features (f0..f2, klein geschrieben wie aus
    SAS üblich) + ID-Spalte. uppercase_columns hebt sie auf F0..F2 — das
    Bundle aus dem Harness erwartet f0..f2, daher hier uppercase aus."""
    X, _, _ = _make_data(n=n)
    df = X.copy()
    df.insert(0, "KUNDE_ID", np.arange(n) + 500000)
    return df


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("saspy_bundle")
    bundle_path, *_ = _export_bundle(tmp)
    return bundle_path


def test_end_to_end_pull_score_writeback(tmp_path, monkeypatch, bundle):
    src = _make_sas_source(n=300)
    fake = FakeSASsession(src)
    _install_fake_saspy(monkeypatch, fake)

    cfg = load_saspy_scoring_config(_write_cfg(
        tmp_path, bundle,
        saspy={"cfgname": "testcfg", "setup_code": "libname SCORING '/pfad';"},
        input={"libref": "SCORING", "table": "eingang",
               "chunk_size": 100, "uppercase_columns": False},
    ))
    run_scoring_saspy(cfg)
    # setup_code läuft als erster (log-geprüfter) submit
    assert fake.submitted[0] == "libname SCORING '/pfad';"

    # Rückgeschrieben: 300 Zeilen in 120er-Chunks → 3 df2sd-Aufrufe
    assert sum(len(w) for w in fake.written) == 300
    assert len(fake.written) == 3
    out_all = pd.concat(fake.written, ignore_index=True)
    assert list(out_all["KUNDE_ID"]) == list(src["KUNDE_ID"])
    assert "SCORE_P" in out_all.columns and out_all["SCORE_P"].notna().all()
    assert (out_all["SCORE_TYP"] == "TEST").all()

    # Scoring-Konsistenz: identisch zur direkten ProductionPipeline
    from rubin.pipelines.production_pipeline import ProductionPipeline
    pipe = ProductionPipeline(str(bundle))
    ref = pipe.score(src.drop(columns=["KUNDE_ID"]))
    ref_col = [c for c in ref.cate.columns if c.startswith("cate_")][0]
    np.testing.assert_allclose(out_all["SCORE_P"].to_numpy(),
                               ref.cate[ref_col].to_numpy().round(6), atol=1e-9)

    # replace: Zieltabelle wird vor dem ersten Append gelöscht; danach 3 Appends
    deletes = [c for c in fake.submitted if "delete kausalscore" in c.lower()]
    appends = [c for c in fake.submitted if "proc append" in c.lower()]
    assert len(deletes) == 1 and len(appends) == 3
    assert fake.ended, "SAS-Session wurde nicht beendet"


    # Monitoring-JSON
    mon = json.loads((tmp_path / "monitoring" / "saspy_test_latest.json").read_text())
    assert mon["input"]["n_rows"] == 300
    assert mon["output"]["write_mode"] == "replace"
    assert "SCORE_P" in mon["scores"]

    # write_mode=append: kein Delete, genau ein Append zusätzlich
    fake2 = FakeSASsession(_make_sas_source(n=150))
    _install_fake_saspy(monkeypatch, fake2)
    cfg2 = load_saspy_scoring_config(_write_cfg(
        tmp_path, bundle,
        input={"libref": "SCORING", "table": "eingang",
               "chunk_size": 500, "uppercase_columns": False},
        output={"libref": "SCOREOUT", "table": "kausalscore",
                "write_mode": "append", "write_chunk_size": 500},
    ))
    run_scoring_saspy(cfg2)
    assert not any("delete kausalscore" in c.lower() for c in fake2.submitted)
    assert sum("proc append" in c.lower() for c in fake2.submitted) == 1

def test_sas_error_in_log_raises_and_closes_session(tmp_path, monkeypatch, bundle):
    fake = FakeSASsession(_make_sas_source(n=50), fail_on_append=True)
    _install_fake_saspy(monkeypatch, fake)
    cfg = load_saspy_scoring_config(_write_cfg(
        tmp_path, bundle,
        input={"libref": "SCORING", "table": "eingang",
               "chunk_size": 100, "uppercase_columns": False},
    ))
    with pytest.raises(RuntimeError, match="SAS-Fehler"):
        run_scoring_saspy(cfg)
    assert fake.ended, "Session muss auch im Fehlerfall geschlossen werden"


def test_empty_table_raises(tmp_path, monkeypatch, bundle):
    fake = FakeSASsession(_make_sas_source(n=0))
    _install_fake_saspy(monkeypatch, fake)
    cfg = load_saspy_scoring_config(_write_cfg(
        tmp_path, bundle,
        input={"libref": "SCORING", "table": "eingang",
               "chunk_size": 100, "uppercase_columns": False},
    ))
    with pytest.raises(ValueError, match="0 Zeilen"):
        run_scoring_saspy(cfg)
    assert fake.ended


def test_keep_pruning_pulls_only_needed_columns(tmp_path, monkeypatch, bundle):
    """Verschiedene Scores brauchen verschiedene Feature-Teilmengen: Der Pull
    zieht per keep= nur Bundle-Features + IDs — Extra-Spalten der breiten
    Gesamttabelle verlassen SAS gar nicht erst."""
    src_df = _make_sas_source(n=200)
    for extra in ("BESTAND_XL", "IRRELEVANT_1", "IRRELEVANT_2"):
        src_df[extra] = 1.0
    fake = FakeSASsession(src_df)
    _install_fake_saspy(monkeypatch, fake)

    cfg = load_saspy_scoring_config(_write_cfg(
        tmp_path, bundle,
        input={"libref": "SCORING", "table": "eingang",
               "chunk_size": 500, "uppercase_columns": False},
    ))
    report = run_scoring_saspy(cfg)

    assert "keep" in fake.last_dsopts
    kept = set(fake.last_dsopts["keep"].split())
    assert kept == {"f0", "f1", "f2", "KUNDE_ID"}
    assert report["input"]["column_pruning"] is True
    assert report["input"]["n_columns"] == 4  # 3 Features + ID, keine Extras


def test_lazy_model_loading_during_scoring(tmp_path, monkeypatch, bundle):
    """Nur die für den Score benötigten Modell-Pickles werden geladen —
    Challenger-Pickles bleiben unangefasst (Speicher bei Multi-Modell-Bundles)."""
    fake = FakeSASsession(_make_sas_source(n=50))
    _install_fake_saspy(monkeypatch, fake)
    cfg = load_saspy_scoring_config(_write_cfg(
        tmp_path, bundle,
        input={"libref": "SCORING", "table": "eingang",
               "chunk_size": 100, "uppercase_columns": False},
    ))

    loaded_holder = {}
    from rubin.pipelines import production_pipeline as pp
    orig_init = pp.ProductionPipeline.__init__
    def spy_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        loaded_holder["models"] = self.models
    monkeypatch.setattr(pp.ProductionPipeline, "__init__", spy_init)

    run_scoring_saspy(cfg)
    loaded = loaded_holder["models"].loaded_names
    assert loaded == ["ModelA"], f"Nur der Champion sollte geladen sein, war: {loaded}"


def test_scores_list_single_pull_two_targets(tmp_path, monkeypatch, bundle):
    """saspy-Variante der Ein-Lade-Zusage: EIN sd2df-Pull, zwei Zieltabellen."""
    fake = FakeSASsession(_make_sas_source(n=80))
    _install_fake_saspy(monkeypatch, fake)
    import yaml as _yaml
    cfg_path = tmp_path / "multi.yml"
    cfg_path.write_text(_yaml.safe_dump({
        "name": "einpull", "runner": "saspy",
        "saspy": {"cfgname": "testcfg"},
        "input": {"libref": "SCORING", "table": "eingang",
                  "chunk_size": 100000, "uppercase_columns": False},
        "id_columns": ["KUNDE_ID"],
        "monitoring": {"dir": str(tmp_path / "mon")},
        "scores": [
            {"name": "champ", "bundle": str(bundle),
             "output": {"libref": "SCOREOUT", "table": "ziel_a"}},
            {"name": "chall", "bundle": str(bundle),
             "scoring": {"score_p_model": "ModelB"},
             "output": {"libref": "SCOREOUT", "table": "ziel_b"}},
        ],
    }), encoding="utf-8")

    pulls = {"n": 0}
    orig = fake.sd2df
    def counting(*a, **kw):
        pulls["n"] += 1
        return orig(*a, **kw)
    fake.sd2df = counting

    reports = run_scoring_saspy(load_saspy_scoring_config(str(cfg_path)))
    assert pulls["n"] == 1, f"{pulls['n']} Pulls (erwartet: 1)"
    assert [r["models"]["score_p"] for r in reports] == ["ModelA", "ModelB"]
    appends_a = sum("base=scoreout.ziel_a" in c.lower() for c in fake.submitted)
    appends_b = sum("base=scoreout.ziel_b" in c.lower() for c in fake.submitted)
    assert appends_a == 1 and appends_b == 1
    assert fake.ended
