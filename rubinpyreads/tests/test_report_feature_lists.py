"""Feature-Selektion im Report: nicht nur Zählungen, sondern die
vollständigen Namenslisten — welche Features selektiert wurden und welche
warum entfernt (Korrelation vs. Importance). Einklappbar, ungekürzt.
Alte Läufe ohne die Listen-Keys rendern unverändert ohne Crash."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_report_golden import _fixture_collector  # noqa: E402

from rubin.reporting.html_report import generate_html_report  # noqa: E402

SEL = [f"feat_{i:03d}" for i in range(80)]


def _html(tmp_path, **fs_overrides):
    c = _fixture_collector()
    c.feature_selection_info.update(fs_overrides)
    out = tmp_path / "r.html"
    generate_html_report(c, str(out))
    return out.read_text(encoding="utf-8")


class TestFeatureLists:
    def test_all_selected_and_removed_names_shown(self, tmp_path):
        html = _html(tmp_path, selected_features=SEL,
                     removed_correlation=["corr_a", "corr_b"],
                     removed_importance=["imp_x"])
        for name in SEL + ["corr_a", "corr_b", "imp_x"]:
            assert name in html
        assert "Selektierte Features (80) — vollständige Liste" in html
        assert "Entfernt durch Korrelationsfilter (2)" in html
        assert "Entfernt durch Importance-Ranking (1)" in html

    def test_old_collector_without_lists_renders_without_details(self, tmp_path):
        html = _html(tmp_path, selected_features=None,
                     removed_correlation=None, removed_importance=None)
        assert "vollständige Liste" not in html  # keine leeren details-Blöcke
