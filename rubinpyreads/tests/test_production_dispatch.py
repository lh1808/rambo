"""Runner-Dispatch der Produktivsetzung: EIN Einstieg (run_scoring.py), der
Transport pro Config über den runner:-Key — saspy wird intern delegiert.

Ersetzt das frühere Shell-Routing (grep-basiertes YAML-Parsing in
run_scoring.sh, RUNNER_SCRIPT-Override) durch robustes YAML in Python.
"""
import sys
from pathlib import Path

import pytest

_PROD = Path(__file__).resolve().parents[1] / "production"
if str(_PROD) not in sys.path:
    sys.path.insert(0, str(_PROD))

import run_scoring  # noqa: E402


def _write(tmp_path, text):
    p = tmp_path / "cfg.yml"
    p.write_text(text, encoding="utf-8")
    return str(p)


class TestRunnerDispatch:
    def test_saspy_config_delegates(self, tmp_path, monkeypatch):
        cfg = _write(tmp_path, 'runner: saspy\nname: x\n')
        called = {}
        import run_scoring_saspy
        monkeypatch.setattr(run_scoring_saspy, "main", lambda: called.setdefault("saspy", True))
        monkeypatch.setattr(sys, "argv", ["run_scoring.py", "--config", cfg])
        run_scoring.main()
        assert called.get("saspy") is True

    def test_quoted_runner_value_still_routes(self, tmp_path, monkeypatch):
        """Der frühere grep/sed-Parser wäre an runner: "saspy" gescheitert
        (UNBEKANNT, Exit 2) — YAML macht Quotes transparent."""
        cfg = _write(tmp_path, 'runner: "saspy"\nname: x\n')
        called = {}
        import run_scoring_saspy
        monkeypatch.setattr(run_scoring_saspy, "main", lambda: called.setdefault("saspy", True))
        monkeypatch.setattr(sys, "argv", ["run_scoring.py", "--config", cfg])
        run_scoring.main()
        assert called.get("saspy") is True

    def test_unknown_runner_exits_clearly(self, tmp_path, monkeypatch):
        cfg = _write(tmp_path, "runner: ftp\nname: x\n")
        monkeypatch.setattr(sys, "argv", ["run_scoring.py", "--config", cfg])
        with pytest.raises(SystemExit, match="Unbekannter runner"):
            run_scoring.main()

    def test_file_config_proceeds_to_validation(self, tmp_path, monkeypatch):
        """file-Configs (auch implizit ohne runner-Key) laufen in die normale
        Validierung — hier: erwartbarer Pflichtfeld-Fehler statt Dispatch."""
        cfg = _write(tmp_path, "runner: file\n")
        monkeypatch.setattr(sys, "argv", ["run_scoring.py", "--config", cfg])
        with pytest.raises(ValueError, match="Pflichtfeld 'name'"):
            run_scoring.main()
