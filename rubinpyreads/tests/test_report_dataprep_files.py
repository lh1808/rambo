"""Datengrundlage im Report: ALLE Quelldateien werden angezeigt (vorher
Kürzung auf 5 + Zähler), jede mit ihrer Rolle je Evaluationstyp:
Training / Training + Evaluation (TMES: eval_file_index-Dateien trainieren
in den übrigen Folds mit) / Evaluation (external: separates Eval-File).
Bei validate_on: external wird eine gesetzte Maske ignoriert — wie in der
Pipeline."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_report_golden import _fixture_collector  # noqa: E402

from rubin.reporting.html_report import generate_html_report  # noqa: E402

FILES = [f"batch_{i:02d}.parquet" for i in range(12)]


def _html(tmp_path, *, eval_file_index=None, eval_files=(), validate_on="cv"):
    c = _fixture_collector()
    c.config_summary["validate_on"] = validate_on
    c.dataprep_info["data_files"] = list(FILES)
    c.dataprep_info["eval_files"] = list(eval_files)
    c.dataprep_info["eval_file_index"] = eval_file_index
    out = tmp_path / "r.html"
    generate_html_report(c, str(out))
    return out.read_text(encoding="utf-8")


def _badges(html):
    sec = html[html.index("Quelldateien nach Rolle"):]
    rows = re.findall(r'>([^<>]+\.parquet)</span>.*?>(Training \+ Evaluation|Evaluation|Training)</span>', sec)
    return dict(rows)


class TestExcludeTile:
    def test_tile_prefers_applied_over_wishlist(self, tmp_path):
        from test_report_golden import _fixture_collector
        from rubin.reporting.html_report import generate_html_report
        c = _fixture_collector()
        c.dataprep_info["exclude_features"] = ["BEITRAG", "GIBTSNICHT"]
        c.dataprep_info["exclude_features_applied"] = ["BEITRAG"]
        out = tmp_path / "r.html"
        generate_html_report(c, str(out))
        html = out.read_text(encoding="utf-8")
        assert "Use-Case-Ausschlüsse" in html and "1: BEITRAG" in html
        assert "GIBTSNICHT" not in html   # Tippfehler nie als Ausschluss angezeigt


class TestDataFileRoles:
    def test_all_files_shown_no_truncation(self, tmp_path):
        html = _html(tmp_path)
        for f in FILES:
            assert f in html          # alle 12 sichtbar
        assert not re.search(r"\(\+\d+\)", html)  # kein Abschneide-Zähler (+N) mehr
        assert "12 Datei(en) — 12 Training" in html

    def test_tmes_marks_masked_files_as_train_plus_eval(self, tmp_path):
        html = _html(tmp_path, eval_file_index=[10, 11])
        roles = _badges(html)
        assert roles["batch_10.parquet"] == "Training + Evaluation"
        assert roles["batch_11.parquet"] == "Training + Evaluation"
        assert all(roles[f] == "Training" for f in FILES[:10])
        assert "10 Training, 2 Training + Evaluation" in html

    def test_scalar_index_and_external_eval_files(self, tmp_path):
        html = _html(tmp_path, eval_file_index=3, validate_on="cv")
        assert _badges(html)["batch_03.parquet"] == "Training + Evaluation"
        html = _html(tmp_path, eval_files=["holdout_2025.parquet"], validate_on="external")
        roles = _badges(html)
        assert roles["holdout_2025.parquet"] == "Evaluation"
        assert all(roles[f] == "Training" for f in FILES)
        assert "12 Training, 1 Evaluation" in html

    def test_external_ignores_stale_mask_index(self, tmp_path):
        # Pipeline-Verhalten gespiegelt: external ignoriert eine gesetzte Maske
        html = _html(tmp_path, eval_file_index=[5], eval_files=["h.parquet"], validate_on="external")
        assert _badges(html)["batch_05.parquet"] == "Training"



class TestPerFileQini:
    """Je-Datei-Qini (Diagnose gepoolter Experimente): Metrik-Helper und
    Report-Tabelle. Hintergrund: Gepoolte Qini kann Schein-Uplift belohnen,
    der nur Datei-Zugehörigkeit erkennt — die Aufschlüsselung macht das sichtbar."""

    def test_rows_per_file_plus_total_and_min_arm(self):
        import numpy as np
        from rubin.evaluation.uplift_metrics import per_file_qini_rows
        rng = np.random.default_rng(0)
        n = 400
        fs = np.array(["a.parquet"] * 300 + ["b.parquet"] * 100)
        t = (rng.random(n) > 0.5).astype(int)
        y = (rng.random(n) < 0.1 + 0.05 * t).astype(float)
        score = rng.normal(size=n)
        rows = per_file_qini_rows(y, t, score, fs, min_arm=60)
        assert [r["file"] for r in rows] == ["a.parquet", "b.parquet", "GESAMT"]
        assert rows[0]["qini"] is not None            # 300 Zeilen → beide Arme > 60
        assert rows[1]["qini"] is None                # 100 Zeilen → Arm < 60 → Platzhalter
        assert rows[2]["n"] == n and rows[2]["qini"] is not None
        assert abs(rows[2]["treat_rate"] - t.mean()) < 1e-9

    def test_report_renders_table(self, tmp_path):
        from rubin.reporting.html_report import ReportCollector, generate_html_report
        c = ReportCollector()
        c.per_file_qini = [
            {"file": "f<1>.parquet", "n": 100, "treat_rate": 0.5,
             "y_rate_t": 0.1, "y_rate_c": 0.08, "qini": 0.01},
            {"file": "GESAMT", "n": 100, "treat_rate": 0.5,
             "y_rate_t": 0.1, "y_rate_c": 0.08, "qini": None},
        ]
        out = tmp_path / "r.html"
        generate_html_report(c, str(out))
        h = out.read_text(encoding="utf-8")
        assert "Qini je Quelldatei" in h and "f&lt;1&gt;.parquet" in h and "0.0100" in h

    def test_report_renders_historical_column_and_all_models_details(self, tmp_path):
        from rubin.reporting.html_report import ReportCollector, generate_html_report
        c = ReportCollector()
        c.per_file_qini = [
            {"file": "a.parquet", "n": 100, "treat_rate": 0.5,
             "y_rate_t": 0.1, "y_rate_c": 0.08, "qini": 0.05},
            {"file": "GESAMT", "n": 100, "treat_rate": 0.5,
             "y_rate_t": 0.1, "y_rate_c": 0.08, "qini": 0.05},
        ]
        c.per_file_qini_champion = "NonParamDML"
        c.per_file_qini_hist_name = "AFF"
        c.per_file_qini_models = {
            "NonParamDML": {"a.parquet": 0.05, "GESAMT": 0.05},
            "SLearner": {"a.parquet": 0.02, "GESAMT": None},
            "AFF": {"a.parquet": 0.04, "GESAMT": 0.03},
        }
        out = tmp_path / "r2.html"
        generate_html_report(c, str(out))
        h = out.read_text(encoding="utf-8")
        assert "Qini NonParamDML (Champion)" in h and "Qini AFF (historisch)" in h
        assert "alle Modelle" in h and "SLearner" in h and "0.0400" in h


class TestPerFileQiniTmes:
    def test_tmes_mask_aligns_file_source_to_eval_files(self, tmp_path):
        """TMES: Eval-Arrays sind das Masken-Subset — die volle file_source-
        Zuordnung wird auf die Maske ausgerichtet; die Tabelle zeigt die
        EVAL-Dateien (realer Vorfall: Sektion fehlte bei TMES-Läufen komplett,
        weil der Längen-Check konservativ übersprang)."""
        import numpy as np
        import pandas as pd
        import logging
        from types import SimpleNamespace
        from rubin.pipelines.analysis_pipeline import AnalysisPipeline
        rng = np.random.default_rng(1)
        n_full, n_eval = 300, 120
        fs_full = np.array(["train_a.parquet"] * 180 + ["eval_b.parquet"] * 60 + ["eval_c.parquet"] * 60)
        mask = np.zeros(n_full, dtype=bool)
        mask[180:] = True
        pd.DataFrame({"file_source": fs_full}).to_parquet(tmp_path / "file_source.parquet")
        y = (rng.random(n_eval) < 0.1).astype(float)
        t = (rng.random(n_eval) > 0.5).astype(int)
        fake = SimpleNamespace(
            _logger=logging.getLogger("t"),
            _eval_scores_ctx={"Champ": (y, t, rng.normal(size=n_eval))},
            _eval_scores_mask=mask,
        )
        cfg = SimpleNamespace(data_files=SimpleNamespace(x_file=str(tmp_path / "X.parquet")),
                              historical_score=SimpleNamespace(name=None))
        res = AnalysisPipeline._compute_per_file_qini(fake, cfg, "Champ")
        assert [r["file"] for r in res["rows"]] == ["eval_b.parquet", "eval_c.parquet", "GESAMT"]
        assert res["rows"][2]["n"] == n_eval
        # Konsistenz-Anker: GESAMT-Qini der Sektion == direkter Qini auf (y,t,score)
        from rubin.evaluation.uplift_metrics import qini_coefficient, uplift_curve
        y2, t2, s2 = fake._eval_scores_ctx["Champ"]
        expected = float(qini_coefficient(uplift_curve(y=y2, t=t2, score=s2)))
        assert abs(res["rows"][2]["qini"] - expected) < 1e-12

    def test_external_guard_disables_section(self, tmp_path):
        import numpy as np
        import pandas as pd
        import logging
        from types import SimpleNamespace
        from rubin.pipelines.analysis_pipeline import AnalysisPipeline
        pd.DataFrame({"file_source": ["a"] * 5 + ["b"] * 5}).to_parquet(tmp_path / "file_source.parquet")
        fake = SimpleNamespace(_logger=logging.getLogger("t"),
                               _eval_scores_ctx={"C": (np.zeros(10), np.zeros(10, int), np.zeros(10))},
                               _eval_scores_mask=None)
        cfg = SimpleNamespace(data_files=SimpleNamespace(x_file=str(tmp_path / "X.parquet")),
                              historical_score=SimpleNamespace(name=None),
                              data_processing=SimpleNamespace(validate_on="external"))
        assert AnalysisPipeline._compute_per_file_qini(fake, cfg, "C") == {}
