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
