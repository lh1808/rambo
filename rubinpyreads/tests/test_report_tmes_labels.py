"""Report-Korrektheit: TMES (Train Many, Evaluate Some) vs. Externe
Evaluation müssen sauber unterschieden werden. Vorher erzählte der Report
bei TMES die External-Geschichte (separater Eval-Datensatz, "ungesehene
Daten", externes Nachtraining) — methodisch falsch: TMES trainiert und
cross-predictet auf ALLEN Zeilen (OOF) und wertet nur das eval_mask-Subset
aus. Die Golden-Tests sichern parallel, dass external/cross byte-gleich
bleiben."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_report_golden import _fixture_collector  # noqa: E402

from rubin.reporting.html_report import generate_html_report  # noqa: E402


def _html(validate_on, eval_mask_file, tmp_path):
    c = _fixture_collector()
    c.config_summary["validate_on"] = validate_on
    c.config_summary["eval_mask_file"] = eval_mask_file
    if eval_mask_file or validate_on == "external":
        c.add_eval_data_stats(
            X_eval=[[0.0], [0.0], [0.0], [0.0]],
            T_eval=[0, 1, 0, 1], Y_eval=[0.0, 1.0, 0.0, 1.0],
        )
    out = tmp_path / "r.html"
    generate_html_report(c, str(out))
    return out.read_text(encoding="utf-8")


class TestTmesVsExternalLabels:
    def test_tmes_report_has_tmes_wording_and_no_external_claims(self, tmp_path):
        html = _html("cv", "masks/eval_mask.parquet", tmp_path)
        assert "TMES-Validierung (Train Many, Evaluate Some)" in html
        assert "eval_mask-Subset" in html
        assert "Eval (TMES)" in html                       # Summary-Bar lebt jetzt
        assert "TMES: eval_mask-Subset" in html            # Datengrundlage-Header
        assert "Gesamtdatensatz inkl. Eval-Subset" in html  # Treatment-Verteilung
        # Die External-Behauptungen dürfen NICHT erscheinen:
        assert "Externe Validierung:" not in html
        assert "separater Datensatz" not in html
        assert "externen Evaluationsdatensatz" not in html
        assert "auf ungesehenen Daten evaluiert" not in html
        assert "externen Eval-Datensatz" not in html

    def test_external_report_keeps_external_wording_and_no_tmes(self, tmp_path):
        html = _html("external", None, tmp_path)
        assert "Externe Validierung:" in html
        assert "externen Evaluationsdatensatz" in html
        assert "TMES" not in html

    def test_cross_report_mentions_neither(self, tmp_path):
        html = _html("cv", None, tmp_path)
        assert "TMES" not in html and "Externe Validierung:" not in html
        assert "Out-of-Fold-Predictions (externe K-Fold Cross-Validation für alle Modelle)" in html
