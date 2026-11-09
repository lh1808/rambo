"""Tests für die Qini-Einordnung: Rauschband (Permutations-Null) + Segment-Benchmark.

Realer Hintergrund: Bei niedrigen Response-Raten (~10-14%) ist das
Bernoulli-Rauschen des Qini so groß wie realistische Effekte — ein
Eval-Qini von 0.0003 ist ohne Referenz nicht interpretierbar, und
Training-OOF-Werte sind durch Trial-/Modell-Selektion nach oben verzerrt
(Gewinner-Fluch). Die zwei Bausteine machen das im Report sichtbar.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from rubin.evaluation.uplift_metrics import (
    qini_null_band,
    uplift_curve,
    qini_coefficient,
)
from rubin.evaluation.segment_benchmark import (
    fit_segment_benchmark,
    fit_score_segment_benchmark_oof,
    resolve_benchmark_columns,
    score_segment_benchmark,
    _cluster_mean_var,
)


def _population(n=20_000, seed=0, flag_share=0.106, tau_f=0.019, tau_r=0.007):
    """Synthetisches Abbild des Use Case (Flag-Moderation, T-Rate 0.75)."""
    rng = np.random.default_rng(seed)
    flag = (rng.random(n) < flag_share).astype(int)
    t = (rng.random(n) < 0.75).astype(int)
    tau = np.where(flag == 1, tau_f, tau_r)
    y = (rng.random(n) < np.where(flag == 1, 0.173, 0.131) + t * tau).astype(int)
    return flag, t, y


# ───────────────────────────── Rauschband ─────────────────────────────

class TestQiniNullBand:
    def test_deterministic_and_shape(self):
        _, t, y = _population(4_000, seed=1)
        b1 = qini_null_band(y, t, n_permutations=40, seed=7)
        b2 = qini_null_band(y, t, n_permutations=40, seed=7)
        assert b1["sd"] == b2["sd"] and np.array_equal(b1["curve_lo"], b2["curve_lo"])
        assert b1["sd"] > 0 and b1["lo"] < 0 < b1["hi"]
        assert len(b1["frac_grid"]) == 101 == len(b1["curve_lo"]) == len(b1["curve_hi"])
        assert np.all(b1["curve_lo"] <= b1["curve_hi"])
        # Endpunkt der Kurve ist score-unabhängig → Envelope schließt sich dort
        assert abs(b1["curve_lo"][-1] - b1["curve_hi"][-1]) < 1e-9
        assert not b1["grouped"]

    def test_band_shrinks_with_n(self):
        _, t_s, y_s = _population(3_000, seed=2)
        _, t_l, y_l = _population(30_000, seed=2)
        sd_small = qini_null_band(y_s, t_s, n_permutations=60, seed=3)["sd"]
        sd_large = qini_null_band(y_l, t_l, n_permutations=60, seed=3)["sd"]
        assert sd_large < sd_small  # mehr Daten → schmaleres Band

    def test_grouped_band_wider_with_clustered_outcomes(self):
        """Zwillings-Szenario: Y je Gruppe identisch dupliziert → zeilenweises
        Permutieren unterschätzt das Band, gruppenweises nicht."""
        rng = np.random.default_rng(4)
        n_g = 1_500
        y_g = rng.integers(0, 2, n_g)
        t_g = rng.integers(0, 2, n_g)
        reps = 4
        y = np.repeat(y_g, reps)
        t = np.repeat(t_g, reps)
        groups = np.repeat(np.arange(n_g), reps)
        sd_row = qini_null_band(y, t, n_permutations=80, seed=5)["sd"]
        b_grp = qini_null_band(y, t, n_permutations=80, seed=5, groups=groups)
        assert b_grp["grouped"] and b_grp["n_groups"] == n_g
        assert b_grp["sd"] > 1.5 * sd_row

    def test_signal_exceeds_band_noise_does_not(self):
        flag, t, y = _population(60_000, seed=6, tau_f=0.08, tau_r=0.0)
        band = qini_null_band(y, t, n_permutations=80, seed=6)
        rng = np.random.default_rng(6)
        q_signal = qini_coefficient(uplift_curve(y=y, t=t, score=flag + rng.normal(0, 1e-6, len(y))))
        q_noise = qini_coefficient(uplift_curve(y=y, t=t, score=rng.normal(size=len(y))))
        assert q_signal > band["hi"]          # starkes Signal: klar außerhalb
        assert band["lo"] <= q_noise <= band["hi"]  # Rauschen: im Band

    def test_single_group_falls_back_to_rowwise(self):
        _, t, y = _population(2_000, seed=7)
        b = qini_null_band(y, t, n_permutations=20, seed=8, groups=np.zeros(len(y)))
        assert not b["grouped"]


# ──────────────────────────── Segment-Benchmark ────────────────────────────

class TestSegmentBenchmark:
    def test_resolve_columns_case_insensitive_and_max_levels(self):
        X = pd.DataFrame({"Flag": [0, 1] * 50, "FEIN": np.arange(100)})
        cols, skipped = resolve_benchmark_columns(["flag", "fein", "FEHLT"], list(X.columns), 25, X)
        assert cols == ["Flag"]
        reasons = {s["column"]: s["reason"] for s in skipped}
        assert "max_levels" in reasons["FEIN"] and "nicht in X" in reasons["FEHLT"]

    def test_moderation_detected_benchmark_ranks(self):
        """Starke Moderation: tau² > 0, Benchmark-Qini positiv und nahe am
        Flag-Oracle (der Benchmark IST das Flag-Oracle bei einer Flag)."""
        flag, t, y = _population(80_000, seed=10, tau_f=0.08, tau_r=0.0)
        X = pd.DataFrame({"FLAG": flag})
        scores, full = fit_score_segment_benchmark_oof(X, t, y, n_splits=5, seed=42)
        assert full.heterogeneity_detected and full.tau2 > 0
        q_bench = qini_coefficient(uplift_curve(y=y, t=t, score=scores))
        rng = np.random.default_rng(0)
        q_oracle = qini_coefficient(uplift_curve(
            y=y, t=t, score=np.where(flag == 1, 0.08, 0.0) + rng.normal(0, 1e-9, len(y))))
        assert q_bench > 0
        assert q_bench > 0.6 * q_oracle  # OOF-Benchmark nahe am wahren Deckel

    def test_homogeneous_collapses_to_constant(self):
        """Keine Moderation → tau²=0, Score ≈ konstant, Qini im Rauschband.
        Das ist die zentrale Ehrlichkeits-Garantie des Shrinkage."""
        rng = np.random.default_rng(11)
        n = 30_000
        seg = rng.integers(0, 8, n)  # 8 Segmente ohne echte Unterschiede
        t = rng.integers(0, 2, n)
        y = (rng.random(n) < 0.13 + t * 0.008).astype(int)
        X = pd.DataFrame({"SEG": seg})
        scores, full = fit_score_segment_benchmark_oof(X, t, y, n_splits=5, seed=42)
        assert not full.heterogeneity_detected
        # je Fold ein fixed-effect-Wert → höchstens n_splits distinct Werte
        assert len(np.unique(np.round(scores, 12))) <= 5
        band = qini_null_band(y, t, n_permutations=60, seed=11)
        q = qini_coefficient(uplift_curve(y=y, t=t, score=scores + rng.normal(0, 1e-12, n)))
        assert band["lo"] - band["sd"] <= q <= band["hi"] + band["sd"]

    def test_small_cell_shrinkage_prevents_capture(self):
        """Der '3,6pp ± 4,0'-Fall: winzige Zelle mit großem Zufalls-ATE darf
        das Ranking nicht anführen — Shrinkage zieht sie Richtung Gesamt-ATE."""
        rng = np.random.default_rng(12)
        n = 40_000
        seg = np.where(rng.random(n) < 0.004, 2, rng.integers(0, 2, n))  # ~160 Zeilen Zelle 2
        t = rng.integers(0, 2, n)
        tau = np.where(seg == 0, 0.0, np.where(seg == 1, 0.03, 0.0))  # Zelle 2: wahrer Effekt 0
        y = (rng.random(n) < 0.13 + t * tau).astype(int)
        bench = fit_segment_benchmark(pd.DataFrame({"S": seg}), t, y, min_rows=20)
        by_seg = {r["segment"]: r for r in bench.table}
        r2 = by_seg["2"]
        if r2["estimable"] and r2["ate"] is not None and abs(r2["ate"]) > 0.01:
            # roher Zufalls-ATE groß → geshrunkener deutlich näher am Gesamt-ATE
            assert abs(r2["ate_shrunk"] - bench.global_ate) < 0.5 * abs(r2["ate"] - bench.global_ate)

    def test_unknown_segment_scores_global_ate(self):
        flag, t, y = _population(10_000, seed=13)
        bench = fit_segment_benchmark(pd.DataFrame({"F": flag}), t, y)
        s = score_segment_benchmark(bench, pd.DataFrame({"F": [0, 1, 99]}))
        assert s[2] == pytest.approx(bench.global_ate)

    def test_cluster_se_wider_than_rowwise(self):
        rng = np.random.default_rng(14)
        n_g = 800
        y_g = rng.random(n_g)
        y = np.repeat(y_g, 5)  # perfekt geclusterte Outcomes
        clusters = np.repeat(np.arange(n_g), 5)
        v_row = _cluster_mean_var(y, None)
        v_cl = _cluster_mean_var(y, clusters)
        assert v_cl > 3 * v_row  # Designeffekt ~5 bei 5er-Duplikaten

    def test_oof_no_leakage_with_many_noise_segments(self):
        """Viele Segmente + reines Rauschen: In-Sample-Zell-ATEs würden ranken
        (Gewinner-Fluch), der OOF+Shrinkage-Benchmark bleibt im Rauschband."""
        rng = np.random.default_rng(15)
        n = 30_000
        seg = rng.integers(0, 30, n)
        t = rng.integers(0, 2, n)
        y = (rng.random(n) < 0.13 + t * 0.008).astype(int)
        scores, full = fit_score_segment_benchmark_oof(pd.DataFrame({"S": seg}), t, y, n_splits=5, seed=1)
        band = qini_null_band(y, t, n_permutations=60, seed=15)
        q = qini_coefficient(uplift_curve(y=y, t=t, score=scores + rng.normal(0, 1e-12, n)))
        assert q <= band["hi"] + band["sd"]

    def test_missing_values_form_own_segment(self):
        flag, t, y = _population(8_000, seed=16)
        col = flag.astype(float)
        col[:500] = np.nan
        bench = fit_segment_benchmark(pd.DataFrame({"F": col}), t, y)
        assert any("fehlend" in r["segment"] for r in bench.table)


# ──────────────────────── Pipeline-Integration ────────────────────────

def _mk_pipeline_stub(groups=None):
    import rubin.pipelines.analysis_pipeline as ap
    pipe = ap.AnalysisPipeline.__new__(ap.AnalysisPipeline)
    pipe._logger = logging.getLogger("rubin.test_einordnung")
    pipe._eval_scores_ctx = {}
    pipe._groups = groups
    pipe._report = SimpleNamespace(qini_noise_info={}, segment_benchmark_info={})
    return pipe


def _mk_cfg(**eval_kwargs):
    from rubin.settings import AnalysisConfig
    cfg = {
        "data_files": {"x_file": "X.parquet", "t_file": "T.parquet", "y_file": "Y.parquet"},
        "mlflow": {"experiment_name": "t"},
        "constants": {"SEED": 42},
        "evaluation": eval_kwargs,
    }
    return AnalysisConfig.model_validate(cfg)


class TestPipelineIntegration:
    def test_noise_band_method_cross_mode_with_groups(self):
        flag, t, y = _population(6_000, seed=20)
        groups = np.arange(len(y)) // 3
        pipe = _mk_pipeline_stub(groups=groups)
        cfg = _mk_cfg(qini_noise_permutations=30)
        mlf = MagicMock()
        pipe._compute_qini_noise_band(cfg, None, t, y, None, None, mlf, is_mt=False)
        assert pipe._qini_null_band is not None and pipe._qini_null_band["grouped"]
        logged = {c.args[0]: c.args[1] for c in mlf.log_metric.call_args_list}
        assert "qini_noise_sd" in logged and logged["qini_noise_sd"] > 0
        assert pipe._report.qini_noise_info["n_permutations"] == 30
        # _band_for: Längen-Guard
        assert pipe._band_for(len(y)) is not None and pipe._band_for(len(y) + 1) is None

    def test_noise_band_respects_mask_and_off_switch(self):
        flag, t, y = _population(4_000, seed=21)
        mask = np.zeros(len(y), dtype=bool)
        mask[:1500] = True
        pipe = _mk_pipeline_stub()
        cfg = _mk_cfg(qini_noise_permutations=20)
        pipe._compute_qini_noise_band(cfg, None, t, y, None, mask, MagicMock(), is_mt=False)
        assert pipe._qini_null_band["n"] == 1500
        pipe2 = _mk_pipeline_stub()
        cfg_off = _mk_cfg(qini_noise_band=False)
        pipe2._compute_qini_noise_band(cfg_off, None, t, y, None, None, MagicMock(), is_mt=False)
        assert pipe2._qini_null_band is None
        # cfg ohne evaluation-Attribut (alte SimpleNamespace-Aufrufer): kein Crash
        pipe3 = _mk_pipeline_stub()
        pipe3._compute_qini_noise_band(SimpleNamespace(), None, t, y, None, None, MagicMock(), is_mt=False)
        assert pipe3._qini_null_band is None

    def test_segment_benchmark_cross_mode_end_to_end(self):
        flag, t, y = _population(40_000, seed=22, tau_f=0.08, tau_r=0.0)
        X = pd.DataFrame({"NEUKUNDE_FLAG": flag, "A": np.random.default_rng(0).normal(size=len(y))})
        pipe = _mk_pipeline_stub()
        cfg = _mk_cfg(benchmark_segments=["neukunde_flag", "GIBTS_NICHT"])
        eval_summary = {}
        mlf = MagicMock()
        pipe._run_segment_benchmark(cfg, X, t, y, None, None, eval_summary, mlf, is_mt=False)
        assert "segment_benchmark" in eval_summary
        assert eval_summary["segment_benchmark"]["qini"] > 0
        assert "policy_value" in eval_summary["segment_benchmark"]  # rct-Politik
        assert "segment_benchmark" in pipe._eval_scores_ctx
        info = pipe._report.segment_benchmark_info
        assert info["mode"] == "cross" and info["columns"] == ["NEUKUNDE_FLAG"]
        assert info["heterogeneity_detected"] and len(info["table"]) == 2
        assert any(s["column"] == "GIBTS_NICHT" for s in info["skipped_columns"])
        logged = {c.args[0] for c in mlf.log_metric.call_args_list}
        assert "qini__segment_benchmark" in logged

    def test_segment_benchmark_tmes_fits_on_train_rows_only(self, monkeypatch):
        flag, t, y = _population(20_000, seed=23, tau_f=0.08, tau_r=0.0)
        X = pd.DataFrame({"F": flag})
        mask = np.zeros(len(y), dtype=bool)
        mask[:6000] = True
        pipe = _mk_pipeline_stub()
        cfg = _mk_cfg(benchmark_segments=["F"])
        seen = {}
        import rubin.pipelines.analysis_pipeline as ap
        import rubin.evaluation.segment_benchmark as sbm
        orig = sbm.fit_segment_benchmark
        def spy(seg_df, tt, yy, **kw):
            seen["n_fit"] = len(seg_df)
            return orig(seg_df, tt, yy, **kw)
        monkeypatch.setattr(ap, "fit_segment_benchmark", spy, raising=False)
        monkeypatch.setattr(sbm, "fit_segment_benchmark", spy)
        eval_summary = {}
        pipe._run_segment_benchmark(cfg, X, t, y, None, mask, eval_summary, MagicMock(), is_mt=False)
        assert seen["n_fit"] == int((~mask).sum())  # NIE aus Eval-Zeilen
        assert len(pipe._eval_scores_ctx["segment_benchmark"][0]) == int(mask.sum())
        assert pipe._report.segment_benchmark_info["mode"] == "tmes"

    def test_segment_benchmark_external_mode_and_mt_skip(self):
        flag, t, y = _population(15_000, seed=24, tau_f=0.08, tau_r=0.0)
        X = pd.DataFrame({"F": flag})
        f_ev, t_ev, y_ev = _population(5_000, seed=25, tau_f=0.08, tau_r=0.0)
        holdout = (pd.DataFrame({"F": f_ev}), t_ev, y_ev, None)
        pipe = _mk_pipeline_stub()
        cfg = _mk_cfg(benchmark_segments=["F"])
        eval_summary = {}
        pipe._run_segment_benchmark(cfg, X, t, y, holdout, None, eval_summary, MagicMock(), is_mt=False)
        assert pipe._report.segment_benchmark_info["mode"] == "external"
        assert len(pipe._eval_scores_ctx["segment_benchmark"][0]) == 5_000
        # MT → stiller Skip
        pipe2 = _mk_pipeline_stub()
        es2 = {}
        pipe2._run_segment_benchmark(cfg, X, t, y, None, None, es2, MagicMock(), is_mt=True)
        assert es2 == {}

    def test_assessment_three_stage_logic(self):
        """Rauschband dämpft die Einstufung; Benchmark-Vergleich liefert die
        Drei-Stufen-Diagnose (nichts da / Lücke / eingefangen)."""
        import rubin.pipelines.analysis_pipeline as ap
        pipe = ap.AnalysisPipeline.__new__(ap.AnalysisPipeline)
        pipe._logger = logging.getLogger("rubin.test_einordnung2")
        cfg = _mk_cfg()
        eval_summary = {
            "ModelA": {"qini": 0.0003, "policy_value": 0.001, "uplift_at_10pct": 0.01, "uplift_at_50pct": 0.005},
            "ModelB": {"qini": 0.0002, "policy_value": 0.001},
            "ModelC": {"qini": 0.0001, "policy_value": 0.001},
        }
        # Fall 1: Qini im Rauschband → Herabstufung + Detailzeile
        pipe._qini_null_band = {"sd": 0.0004, "lo": -0.0008, "hi": 0.0008, "n": 1000,
                                "n_permutations": 50, "grouped": False, "n_groups": 1000}
        pipe._segment_benchmark_summary = {"name": "segment_benchmark", "qini": 0.0001,
                                           "heterogeneity_detected": False, "tau2": 0.0,
                                           "n_segments": 2, "columns": ["F"], "mode": "cross"}
        ha = pipe._compute_heterogeneity_assessment(cfg, eval_summary, "ModelA")
        het = ha["heterogeneity"]
        assert het["noise_within"] is True and het["level"] in ("moderate", "weak")
        assert any("Rauschband" in d for d in het["details"])
        bc = ha["benchmark_comparison"]
        assert bc["level"] == "bench_nothing"
        # Fall 2: Benchmark trägt, Modell bleibt zurück → Lücke
        pipe._qini_null_band = {"sd": 0.0001, "lo": -0.0002, "hi": 0.0002, "n": 1000,
                                "n_permutations": 50, "grouped": False, "n_groups": 1000}
        pipe._segment_benchmark_summary = {"name": "segment_benchmark", "qini": 0.002,
                                           "heterogeneity_detected": True, "tau2": 1e-5,
                                           "n_segments": 2, "columns": ["F"], "mode": "cross"}
        ha2 = pipe._compute_heterogeneity_assessment(cfg, eval_summary, "ModelA")
        assert ha2["benchmark_comparison"]["level"] == "bench_gap"
        # Fall 3: Modell erreicht Benchmark → eingefangen
        pipe._segment_benchmark_summary["qini"] = 0.00035
        ha3 = pipe._compute_heterogeneity_assessment(cfg, eval_summary, "ModelA")
        assert ha3["benchmark_comparison"]["level"] == "bench_captured"
        # Benchmark/hist zählen nicht in den Modell-Konsens
        eval_summary["segment_benchmark"] = {"qini": 0.5}
        ha4 = pipe._compute_heterogeneity_assessment(cfg, eval_summary, "ModelA")
        assert ha4["heterogeneity"]["n_models"] == 3

    def test_report_renders_cards_and_footnote(self):
        from rubin.reporting.html_report import ReportCollector, _render_heterogeneity
        c = ReportCollector()
        c.heterogeneity_assessment = {
            "heterogeneity": {"level": "weak", "color": "#b35900", "bg": "#fff3e6",
                              "border": "#f0c78a", "label": "Schwache Heterogenität",
                              "details": ["x"], "champion": "ModelA", "champion_qini": 0.0003,
                              "champion_pv": 0.0, "n_positive_qini": 1, "n_models": 3,
                              "concentration": 1.0, "noise_two_sd": 0.0008, "noise_within": True},
            "hist_comparison": None,
            "benchmark_comparison": {"level": "bench_nothing", "color": "#b35900", "bg": "#fff3e6",
                                     "border": "#f0c78a", "label": "Deklarierte Segmente ohne messbare Ranking-Ausbeute",
                                     "details": ["τ² ≈ 0"], "bench_name": "segment_benchmark",
                                     "bench_qini": 0.0001, "champion_qini": 0.0003, "tau2": 0.0,
                                     "n_segments": 2, "columns": ["F"], "mode": "cross",
                                     "heterogeneity_detected": False, "within_noise": True},
        }
        c.qini_noise_info = {"sd": 0.0004, "lo": -0.0008, "hi": 0.0008, "n": 50_000,
                             "n_permutations": 200, "grouped": True, "n_groups": 20_000}
        c.segment_benchmark_info = {
            "name": "segment_benchmark", "mode": "cross", "columns": ["F"],
            "skipped_columns": [{"column": "X9", "reason": "nicht in X gefunden"}],
            "n_segments": 2, "tau2": 0.0, "heterogeneity_detected": False,
            "global_ate": 0.008, "grouped": True, "qini": 0.0001,
            "table": [{"segment": "1", "n": 5000, "n_treat": 3750, "t_rate": 0.75,
                       "ate": 0.019, "se": 0.006, "z": 3.1, "ate_shrunk": 0.015, "estimable": True},
                      {"segment": "0", "n": 45000, "n_treat": 33750, "t_rate": 0.75,
                       "ate": 0.007, "se": 0.002, "z": 3.5, "ate_shrunk": 0.0075, "estimable": True}],
        }
        html = _render_heterogeneity(c)
        for needle in ["Qini-Rauschband (Permutations-Null)", "gruppenweise permutiert",
                       "Segment-Benchmark: Treatment-Effekte", "ATE (geshrunken)",
                       "Wie wird das Rauschband erzeugt", "Wie entsteht der Segment-Benchmark",
                       "nichts zu holen", "Übersprungen"]:
            assert needle in html, needle
        # Leerer Collector → leere Sektion (Golden-Schutz)
        assert _render_heterogeneity(ReportCollector()) == ""

    def test_settings_yaml_roundtrip(self, tmp_path):
        import yaml
        from rubin.settings import AnalysisConfig
        cfg = {
            "data_files": {"x_file": "X.parquet", "t_file": "T.parquet", "y_file": "Y.parquet"},
            "mlflow": {"experiment_name": "t"},
            "evaluation": {"qini_noise_band": False, "qini_noise_permutations": 300,
                           "benchmark_segments": ["NEUKUNDE_FLAG", "LEAD_ART"]},
        }
        p = tmp_path / "c.yml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        loaded = AnalysisConfig.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
        assert loaded.evaluation.qini_noise_band is False
        assert loaded.evaluation.qini_noise_permutations == 300
        assert loaded.evaluation.benchmark_segments == ["NEUKUNDE_FLAG", "LEAD_ART"]
        # Defaults ohne Sektion
        default = AnalysisConfig.model_validate({k: v for k, v in cfg.items() if k != "evaluation"})
        assert default.evaluation.qini_noise_band is True
        assert default.evaluation.benchmark_segments == []


class TestBalanceDedupGroupInteraction:
    """Zusammenspiel von Treatment-Balance, Deduplizierung und Gruppen-Folds.

    Reihenfolge in DataPrep: Dedup → Balance → Gruppen-Puffer. Damit bleiben
    die gebalancten Raten final (Dedup verschiebt sie nicht nachträglich) und
    groups.parquet beschreibt exakt die überlebenden Zeilen (eigener Dauertest
    test_balance_keeps_groups_aligned). Hier: die Konfigurations-Guards.
    """

    def _leads_csv(self, tmp_path, name="a.csv", n_partner=120, t_rate=0.5, seed=1):
        rng = np.random.default_rng(seed)
        rows = []
        for pid in range(n_partner):
            for lead in range(2):
                rows.append({"PARTNER_ID": f"P{pid:04d}", "LEAD_ID": f"L{pid:04d}_{lead}",
                             "ALTER": 30 + pid % 40, "LEAD_ART": lead,
                             "T": int(rng.random() < t_rate), "Y": int(rng.random() < 0.3)})
        f = tmp_path / name
        pd.DataFrame(rows).to_csv(f, index=False)
        return f

    def test_dedup_and_group_same_column_warns_singleton(self, tmp_path, caplog):
        import yaml  # noqa: F401 - Muster wie test_group_cv
        from tests.test_group_cv import _dataprep_cfg
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        f = self._leads_csv(tmp_path)
        p, outd = _dataprep_cfg(tmp_path, [f], {
            "group_column": "PARTNER_ID",
            "deduplicate": True, "deduplicate_id_column": "PARTNER_ID",
            "features": ["ALTER", "LEAD_ART"]})
        with caplog.at_level(logging.WARNING, logger="rubin.dataprep"):
            DataPrepPipeline.from_config_path(str(p)).run()
        assert "Singleton" in caplog.text and "wirkungslos" in caplog.text
        # Und faktisch: jede Gruppe genau 1 Zeile
        g = pd.read_parquet(outd / "groups.parquet")["group"]
        assert g.nunique() == len(g)

    def test_dedup_and_group_different_columns_is_legit_combo(self, tmp_path, caplog):
        from tests.test_group_cv import _dataprep_cfg
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        f = self._leads_csv(tmp_path)
        # Exakte Zeilen-Duplikate je LEAD_ID entstehen hier nicht — es geht um
        # die Guard-Logik: verschiedene Spalten → Info, KEINE Warnung.
        p, outd = _dataprep_cfg(tmp_path, [f], {
            "group_column": "PARTNER_ID",
            "deduplicate": True, "deduplicate_id_column": "LEAD_ID",
            "features": ["ALTER", "LEAD_ART"]})
        with caplog.at_level(logging.INFO, logger="rubin.dataprep"):
            DataPrepPipeline.from_config_path(str(p)).run()
        assert "kombiniert" in caplog.text
        assert "wirkungslos" not in caplog.text
        g = pd.read_parquet(outd / "groups.parquet")["group"]
        assert g.value_counts().max() == 2  # Partner-Gruppen intakt

    def test_segment_t_rate_drift_warns_in_rct(self, caplog):
        """Realfall Score-Klassen: T-Rate driftet 0.70 vs. 0.79 über Segmente —
        balance_treatments (nur zwischen Dateien) behebt das nicht; der
        Benchmark macht es jetzt sichtbar."""
        rng = np.random.default_rng(30)
        n = 40_000
        seg = rng.integers(0, 2, n)
        t = (rng.random(n) < np.where(seg == 1, 0.79, 0.70)).astype(int)
        y = (rng.random(n) < 0.13 + t * np.where(seg == 1, 0.02, 0.005)).astype(int)
        X = pd.DataFrame({"SEG": seg})
        pipe = _mk_pipeline_stub()
        cfg = _mk_cfg(benchmark_segments=["SEG"])
        with caplog.at_level(logging.WARNING, logger="rubin.test_einordnung"):
            pipe._run_segment_benchmark(cfg, X, t, y, None, None, {}, MagicMock(), is_mt=False)
        assert "driften" in caplog.text and "balance_treatments" in caplog.text
        drift = pipe._report.segment_benchmark_info.get("t_rate_drift")
        assert drift is not None and 0.05 < drift < 0.13

    def test_no_drift_warning_when_rates_uniform(self, caplog):
        flag, t, y = _population(30_000, seed=31, tau_f=0.08, tau_r=0.0)
        pipe = _mk_pipeline_stub()
        cfg = _mk_cfg(benchmark_segments=["F"])
        with caplog.at_level(logging.WARNING, logger="rubin.test_einordnung"):
            pipe._run_segment_benchmark(cfg, pd.DataFrame({"F": flag}), t, y, None, None, {}, MagicMock(), is_mt=False)
        assert "driften" not in caplog.text
        assert pipe._report.segment_benchmark_info.get("t_rate_drift") is not None


class TestBenchmarkSurvivesFeatureSelection:
    """Realer Vorfall: FEATURES_FEHLEN_FLG war explizit als Benchmark-Segment
    deklariert, wurde aber von der Feature-Selektion aus X entfernt → der
    Benchmark entfiel still mit 'nicht in X gefunden'. Garantie jetzt: Die
    Benchmark-Spalten werden in run() VOR der Selektion gepuffert und in
    _run_segment_benchmark aus dem Puffer wiederhergestellt."""

    def _data(self, n=30_000, seed=40):
        rng = np.random.default_rng(seed)
        flag = (rng.random(n) < 0.106).astype(int)
        t = (rng.random(n) < 0.75).astype(int)
        y = (rng.random(n) < 0.131 + 0.042 * flag + t * (0.007 + 0.07 * flag)).astype(int)
        X_full = pd.DataFrame({"FEATURES_FEHLEN_FLG": flag,
                               "A": rng.normal(size=n), "B": rng.normal(size=n)})
        return X_full, t, y

    def test_restored_from_buffer_cross_mode(self, caplog):
        X_full, t, y = self._data()
        X_after_fs = X_full[["A", "B"]]  # FS hat die Flag entfernt
        pipe = _mk_pipeline_stub()
        pipe._bench_seg_train_df = X_full[["FEATURES_FEHLEN_FLG"]].copy()
        cfg = _mk_cfg(benchmark_segments=["features_fehlen_flg"])
        eval_summary = {}
        with caplog.at_level(logging.INFO, logger="rubin.test_einordnung"):
            pipe._run_segment_benchmark(cfg, X_after_fs, t, y, None, None, eval_summary, MagicMock(), is_mt=False)
        assert "segment_benchmark" in eval_summary
        assert eval_summary["segment_benchmark"]["qini"] > 0
        assert "wiederhergestellt" in caplog.text and "always_keep" in caplog.text
        info = pipe._report.segment_benchmark_info
        assert info["restored_after_fs"] == ["FEATURES_FEHLEN_FLG"]
        assert info["columns"] == ["FEATURES_FEHLEN_FLG"]

    def test_restored_in_tmes_and_external(self):
        X_full, t, y = self._data(20_000)
        X_after = X_full[["A"]]
        # TMES
        mask = np.zeros(len(y), dtype=bool); mask[:6000] = True
        pipe = _mk_pipeline_stub()
        pipe._bench_seg_train_df = X_full[["FEATURES_FEHLEN_FLG"]].copy()
        cfg = _mk_cfg(benchmark_segments=["FEATURES_FEHLEN_FLG"])
        es = {}
        pipe._run_segment_benchmark(cfg, X_after, t, y, None, mask, es, MagicMock(), is_mt=False)
        assert "segment_benchmark" in es and pipe._report.segment_benchmark_info["mode"] == "tmes"
        # External: Spalte auch im Eval-X vom Alignment entfernt → Eval-Puffer
        Xe_full, te, ye = self._data(8_000, seed=41)
        pipe2 = _mk_pipeline_stub()
        pipe2._bench_seg_train_df = X_full[["FEATURES_FEHLEN_FLG"]].copy()
        pipe2._bench_seg_eval_df = Xe_full[["FEATURES_FEHLEN_FLG"]].copy()
        es2 = {}
        holdout = (Xe_full[["A"]], te, ye, None)
        pipe2._run_segment_benchmark(cfg, X_after, t, y, holdout, None, es2, MagicMock(), is_mt=False)
        assert "segment_benchmark" in es2 and pipe2._report.segment_benchmark_info["mode"] == "external"
        assert len(pipe2._eval_scores_ctx["segment_benchmark"][0]) == 8_000

    def test_without_buffer_still_skips_with_warning(self, caplog):
        X_full, t, y = self._data(5_000)
        pipe = _mk_pipeline_stub()  # kein Puffer gesetzt
        cfg = _mk_cfg(benchmark_segments=["FEATURES_FEHLEN_FLG"])
        es = {}
        with caplog.at_level(logging.WARNING, logger="rubin.test_einordnung"):
            pipe._run_segment_benchmark(cfg, X_full[["A", "B"]], t, y, None, None, es, MagicMock(), is_mt=False)
        assert es == {} and "entfällt" in caplog.text

    def test_buffer_is_filled_before_feature_selection_in_run(self):
        """Quelltext-Garantie: Der Puffer-Block steht in run() VOR dem
        _run_feature_selection-Aufruf (sonst puffert er das beschnittene X)."""
        import inspect
        import rubin.pipelines.analysis_pipeline as ap
        src = inspect.getsource(ap.AnalysisPipeline)
        i_buf = src.index("self._bench_seg_train_df = None")
        i_fs = src.index("self._run_feature_selection(cfg, X, T, Y, mlflow)")
        assert i_buf < i_fs
        # und der Puffer liest aus X (vor der Reduktion), nicht aus removed
        i_fill = src.index("self._bench_seg_train_df = X[_found].copy()")
        assert i_fill < i_fs


class TestReferenceRowsExplained:
    """Der Benchmark (und der hist. Score) erscheinen im Modellvergleich und in
    der Per-File-Matrix — dort müssen sie als Referenz gekennzeichnet und ihre
    Entstehung erklärt sein (Nutzerfrage: 'Ist überall beschrieben, wie die
    Benchmark-Scores entstehen?')."""

    def test_comparison_table_badges_and_footnote(self):
        from rubin.reporting.html_report import ReportCollector, _render_comparison
        c = ReportCollector()
        c.model_metrics = {
            "NonParamDML": {"qini": 0.0007, "auuc": 0.0009},
            "historical_score": {"qini": 0.00006, "auuc": 0.00008},
            "segment_benchmark": {"qini": -0.0006, "auuc": -0.0008},
        }
        c.segment_benchmark_info = {"name": "segment_benchmark", "qini": -0.0006}
        c.heterogeneity_assessment = {"hist_comparison": {"hist_name": "historical_score"}}
        html = _render_comparison(c, {"study_type": "rct"}, "NonParamDML", "qini", True, False)
        assert html.count('class="badge-ref"') == 2 and html.count('>Referenz</span>') == 2
        assert "Referenz-Zeilen:" in html
        assert "Stufen-Score" in html and "Homogenitätstest + Shrinkage" in html
        assert "nehmen nicht an der Champion-Wahl teil" in html
        # Ohne Referenzen: keine Badges, keine Fußnote (Golden-Schutz)
        c2 = ReportCollector()
        c2.model_metrics = {"NonParamDML": {"qini": 0.0007}}
        html2 = _render_comparison(c2, {"study_type": "rct"}, "NonParamDML", "qini", True, False)
        assert "badge-ref" not in html2 and "Referenz-Zeilen" not in html2

    def test_heterogeneity_explainer_describes_scoring_mechanics(self):
        from rubin.reporting.html_report import ReportCollector, _render_heterogeneity
        c = ReportCollector()
        c.segment_benchmark_info = {
            "name": "segment_benchmark", "mode": "cross", "columns": ["F"],
            "skipped_columns": [], "n_segments": 2, "tau2": 0.0,
            "heterogeneity_detected": False, "global_ate": 0.009, "grouped": True,
            "qini": -0.0006, "q_pvalue": 0.341, "table": [],
        }
        html = _render_heterogeneity(c)
        for needle in ["derselben Qini-Mechanik", "seed-fester Tie-Break",
                       "Reihenfolge der Segmente", "Homogenitätstest p = 0.341"]:
            assert needle in html, needle


class TestPrioritizationDiagnostics:
    """Priorisierung/Depriorisierung je Segment: CIs, vs.-Rest-Kontraste mit
    Benjamini-Hochberg, Verdicts gegen die Relevanzschwelle, Forest-Plot."""

    def test_bh_adjustment_properties(self):
        from rubin.evaluation.segment_benchmark import _benjamini_hochberg
        p = [0.001, 0.04, 0.03, 0.5, 0.8]
        adj = _benjamini_hochberg(p)
        assert all(a >= r for a, r in zip(adj, p))          # adj >= roh
        assert all(0 <= a <= 1 for a in adj)
        order = sorted(range(5), key=lambda i: p[i])
        assert all(adj[order[i]] <= adj[order[i+1]] + 1e-12 for i in range(4))  # monoton
        assert adj[0] == pytest.approx(0.005)               # 0.001*5/1

    def test_verdicts_against_threshold(self):
        """Drei konstruierte Fälle: klar wirksam / eng um Null bei Schwelle>0 /
        breites CI → unentscheidbar."""
        rng = np.random.default_rng(50)
        n = 200_000
        # Segment 1 groß (enges CI um 0), Segment 2 winzig (breites CI)
        seg = rng.choice([0, 1, 2], size=n, p=[0.35, 0.62, 0.03])
        t = rng.integers(0, 2, n)
        tau = np.where(seg == 0, 0.08, np.where(seg == 1, 0.0, 0.01))
        y = (rng.random(n) < 0.08 + t * tau).astype(int)
        bench = fit_segment_benchmark(pd.DataFrame({"S": seg}), t, y, min_effect=0.005)
        by = {r["segment"]: r for r in bench.table}
        assert by["0"]["verdict"] == "effective"            # CI klar über der Schwelle
        assert by["1"]["verdict"] == "below_threshold"      # wahrer Effekt 0, enges CI < Schwelle
        assert by["2"]["verdict"] == "inconclusive"         # kleines n, breites CI
        assert by["0"]["ci_lo"] > 0.005 and by["1"]["ci_hi"] < 0.005
        assert bench.min_effect == pytest.approx(0.005)

    def test_direction_confirmed_only_after_fdr(self):
        """Ein echter Ausreißer wird ▲-belegt, Rausch-Segmente nicht — und bei
        vielen reinen Rausch-Segmenten wird NICHTS belegt (FDR-Schutz)."""
        rng = np.random.default_rng(51)
        n = 120_000
        seg = rng.integers(0, 6, n)
        t = rng.integers(0, 2, n)
        tau = np.where(seg == 0, 0.06, 0.008)               # Segment 0 hebt sich stark ab
        y = (rng.random(n) < 0.13 + t * tau).astype(int)
        bench = fit_segment_benchmark(pd.DataFrame({"S": seg}), t, y)
        by = {r["segment"]: r for r in bench.table}
        assert by["0"]["direction"] == "above"
        assert by["0"]["p_contrast_adj"] < 0.05
        assert sum(1 for r in bench.table if r.get("direction")) <= 3  # kein Rausch-Festival
        # Konsistenz: delta = ate - ate_rest
        r0 = by["0"]
        assert r0["delta_vs_rest"] == pytest.approx(r0["ate"] - r0["ate_rest"])
        # Reines Rauschen: keine belegten Richtungen
        y2 = (rng.random(len(t)) < 0.13 + t * 0.008).astype(int)
        bench2 = fit_segment_benchmark(pd.DataFrame({"S": seg}), t, y2)
        assert all(r.get("direction") is None for r in bench2.table)

    def test_tendency_marks_raw_but_not_adjusted(self):
        """Grenzfall wie Lukas Segment 0: roh auffällig, nach BH nicht →
        tendency gesetzt, direction nicht."""
        rng = np.random.default_rng(52)
        n = 50_000
        seg = rng.integers(0, 11, n)
        t = rng.integers(0, 2, n)
        tau = np.where(seg == 0, 0.0, 0.009)                # Seg 0 schwächer, moderat
        y = (rng.random(n) < 0.13 + t * tau).astype(int)
        bench = fit_segment_benchmark(pd.DataFrame({"S": seg}), t, y)
        r0 = {r["segment"]: r for r in bench.table}["0"]
        # je nach Seed roh signifikant oder belegt — beides zeigt Richtung "below"
        assert (r0["direction"] == "below") or (r0["tendency"] == "below") or r0["p_contrast"] > 0.05

    def test_forest_plot_renders(self):
        import matplotlib; matplotlib.use("Agg")
        from rubin.evaluation.evaluation_plots import generate_segment_forest_plot
        flag, t, y = _population(20_000, seed=53, tau_f=0.08, tau_r=0.0)
        bench = fit_segment_benchmark(pd.DataFrame({"F": flag}), t, y)
        fig = generate_segment_forest_plot(bench.table, bench.global_ate, 0.0)
        assert fig is not None
        import matplotlib.pyplot as plt; plt.close(fig)
        assert generate_segment_forest_plot([], 0.0) is None

    def test_pipeline_summary_carries_diagnostics_and_forest(self):
        flag, t, y = _population(40_000, seed=54, tau_f=0.08, tau_r=0.0)
        X = pd.DataFrame({"F": flag})
        pipe = _mk_pipeline_stub()
        cfg = _mk_cfg(benchmark_segments=["F"], benchmark_min_effect=0.002)
        es = {}
        pipe._run_segment_benchmark(cfg, X, t, y, None, None, es, MagicMock(), is_mt=False)
        info = pipe._report.segment_benchmark_info
        assert info["min_effect"] == pytest.approx(0.002)
        assert isinstance(info["forest_plot"], str) and len(info["forest_plot"]) > 1000
        r1 = {r["segment"]: r for r in info["table"]}["1"]
        assert r1["verdict"] == "effective" and r1["direction"] == "above"
        assert r1["p_contrast_adj"] is not None and r1["ci_lo"] is not None

    def test_report_renders_prio_columns_markers_and_explainer(self):
        from rubin.reporting.html_report import ReportCollector, _render_heterogeneity
        c = ReportCollector()
        c.segment_benchmark_info = {
            "name": "segment_benchmark", "mode": "cross", "columns": ["F"],
            "skipped_columns": [], "n_segments": 3, "tau2": 1e-5,
            "heterogeneity_detected": True, "global_ate": 0.009, "grouped": True,
            "qini": 0.0004, "q_pvalue": 0.01, "min_effect": 0.0,
            "forest_plot": "aWFtYXBuZw==",
            "table": [
                {"segment": "1", "n": 78348, "t_rate": 0.752, "ate": 0.0129, "se": 0.0026,
                 "z": 4.92, "ate_shrunk": 0.012, "estimable": True, "ci_lo": 0.0078, "ci_hi": 0.018,
                 "ate_rest": 0.007, "delta_vs_rest": 0.0059, "se_contrast": 0.003,
                 "ci_contrast_lo": 0.0, "ci_contrast_hi": 0.0118,
                 "p_contrast": 0.02, "p_contrast_adj": 0.04, "direction": "above",
                 "tendency": None, "verdict": "effective"},
                {"segment": "0", "n": 41718, "t_rate": 0.749, "ate": 0.0029, "se": 0.0036,
                 "z": 0.8, "ate_shrunk": 0.006, "estimable": True, "ci_lo": -0.0042, "ci_hi": 0.01,
                 "ate_rest": 0.0103, "delta_vs_rest": -0.0074, "se_contrast": 0.004,
                 "ci_contrast_lo": -0.0152, "ci_contrast_hi": 0.0004,
                 "p_contrast": 0.06, "p_contrast_adj": 0.26, "direction": None,
                 "tendency": "below", "verdict": "inconclusive"},
                {"segment": "alt", "n": 100, "t_rate": 0.7, "ate": None, "se": None, "z": None,
                 "ate_shrunk": 0.009, "estimable": False},  # alte/degenerierte Zeile → "–"
            ],
        }
        html = _render_heterogeneity(c)
        for needle in ["Δ vs. Rest [CI]", "p roh / adj.", "0.060 / 0.260", "(▼)", "<strong>▲</strong>",
                       "wirksam belegt", "unentscheidbar", "Benjamini-Hochberg",
                       "Wie lese ich Priorisierung", "VORREGISTRIERTE", "Segment-Forest-Plot",
                       "Priorisierung: belegt überdurchschnittlich: <strong>\u201e1\u201c</strong>",
                       "kein Segment unter der Schwelle belegt"]:
            assert needle in html, needle
        # Alte Zeile ohne neue Keys rendert als "–", kein Crash
        assert html.count("–") >= 4

    def test_settings_and_ui_roundtrip_min_effect(self, tmp_path):
        import yaml
        from rubin.settings import AnalysisConfig
        cfg = {"data_files": {"x_file": "X.parquet", "t_file": "T.parquet", "y_file": "Y.parquet"},
               "mlflow": {"experiment_name": "t"},
               "evaluation": {"benchmark_segments": ["F"], "benchmark_min_effect": 0.002}}
        loaded = AnalysisConfig.model_validate(cfg)
        assert loaded.evaluation.benchmark_min_effect == pytest.approx(0.002)
        assert AnalysisConfig.model_validate({k: v for k, v in cfg.items() if k != "evaluation"}).evaluation.benchmark_min_effect == 0.0
