"""Modellvergleichs-Tabelle: richtungsbewusste Pfeile und Best-Marker je
Metrik (Fehlermaße wie *_se sind "niedriger = besser" — vorher galt das eine
higher-Flag der Selektionsmetrik pauschal für alle Spalten), int-Metriken
mit Marker statt leerer Spalte, NaN als "–" statt wörtlich "nan"."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_report_golden import _fixture_collector  # noqa: E402

from rubin.reporting.html_report import generate_html_report  # noqa: E402


def _html(tmp_path):
    c = _fixture_collector()
    for mn, mx in c.model_metrics.items():
        mx["ate_se"] = 0.05 if mn == c.champion_name else 0.09  # Champion: kleinster Fehler
        mx["n_folds"] = 5
    first = sorted(c.model_metrics)[0]
    c.model_metrics[first]["auuc_extra"] = float("nan")
    for mn in c.model_metrics:
        c.model_metrics[mn].setdefault("auuc_extra", 0.7 if mn == c.champion_name else 0.6)
    out = tmp_path / "r.html"
    generate_html_report(c, str(out))
    return out.read_text(encoding="utf-8")


class TestComparisonRobustness:
    def test_direction_aware_arrows(self, tmp_path):
        html = _html(tmp_path)
        assert "ate_se ▼" in html          # Fehlermaß: niedriger = besser
        assert "qini ▲" in html            # Standard: höher = besser

    def test_error_metric_best_marker_is_minimum(self, tmp_path):
        html = _html(tmp_path)
        assert '<td class="best-val">0.050000</td>' in html   # min(ate_se) markiert
        assert '<td class="best-val">0.090000</td>' not in html

    def test_int_metric_rendered_not_dashed(self, tmp_path):
        html = _html(tmp_path)
        assert '>5</td>' in html           # int-Zelle sichtbar (vorher "–"-Spalte)

    def test_nan_shown_as_dash_not_nan_text(self, tmp_path):
        html = _html(tmp_path)
        assert 'title="nicht berechnet (NaN/Inf)">–</td>' in html
        assert ">nan<" not in html
