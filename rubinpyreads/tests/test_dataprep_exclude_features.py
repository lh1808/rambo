"""Use-Case-Ausschlüsse als Delta über dem Feature-Dictionary: Das zentrale
Dictionary bleibt die Wahrheit "was ist grundsätzlich Input";
dp.exclude_features nimmt davon pro Use Case gezielt Spalten aus. Unbekannte
Namen warnen (Tippfehler/umbenannt), Target/Treatment sind geschützt, und
die effektiv angewandte Liste wird separat als exclude_features_used.txt
weggeloggt (schnell greifbar; bei Retraining 1:1 wiederverwendbar)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from rubin.pipelines.data_prep_pipeline import DataPrepPipeline


def _make_raw(tmp_path, n=120):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "ALTER": rng.normal(40, 10, n).round(1),
        "BEITRAG": rng.normal(90, 20, n).round(2),
        "REGION": rng.choice(["N", "S", "W"], n),
        "T": rng.integers(0, 2, n),
        "Y": rng.integers(0, 2, n),
    })
    p = tmp_path / "raw.csv"
    df.to_csv(p, index=False)
    return p


def _dictionary(tmp_path):
    p = tmp_path / "features.csv"
    pd.DataFrame({
        "ROLE": ["INPUT", "INPUT", "INPUT", "TARGET", "TREATMENT"],
        "NAME": ["ALTER", "BEITRAG", "REGION", "Y", "T"],
        "LEVEL": ["METRIC", "METRIC", "NOMINAL", "METRIC", "METRIC"],
    }).to_csv(p, index=False)
    return p


def _cfg(tmp_path, raw_csv, out_dir, *, feature_path=None, exclude=None):
    dp = {
        "data_path": [str(raw_csv)],
        "output_path": str(out_dir),
        "target": "Y",
        "treatment": "T",
        "score_name": None,
        "log_to_mlflow": False,
    }
    if feature_path:
        dp["feature_path"] = str(feature_path)
    if exclude is not None:
        dp["exclude_features"] = exclude
    cfg = {
        "mlflow": {"experiment_name": "test_excl"},
        "constants": {"SEED": 1, "work_dir": str(tmp_path / "runs")},
        "data_files": {
            "x_file": str(out_dir / "X.parquet"),
            "t_file": str(out_dir / "T.parquet"),
            "y_file": str(out_dir / "Y.parquet"),
        },
        "data_prep": dp,
    }
    p = tmp_path / "cfg.yml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _run(cfg_path, out_dir):
    DataPrepPipeline.from_config_path(str(cfg_path)).run()
    return pd.read_parquet(out_dir / "X.parquet")


class TestExcludeFeatures:
    def test_dictionary_minus_usecase_excludes_and_txt_log(self, tmp_path, caplog):
        out = tmp_path / "out"; out.mkdir()
        raw = _make_raw(tmp_path)
        cfg = _cfg(tmp_path, raw, out, feature_path=_dictionary(tmp_path),
                   exclude=["beitrag", "GIBTSNICHT"])
        with caplog.at_level("WARNING"):
            X = _run(cfg, out)
        assert sorted(X.columns) == ["ALTER", "REGION"]
        assert any("GIBTSNICHT" in r.message for r in caplog.records)   # Tippfehler warnt
        # Separates Weglogging: ein Name pro Zeile, exakt die ANGEWANDTEN
        assert (out / "exclude_features_used.txt").read_text(encoding="utf-8") == "BEITRAG\n"
        # Und in der Config-Kopie des Laufs (Retraining-Reproduzierbarkeit)
        written = yaml.safe_load((out / "dataprep_config.yml").read_text(encoding="utf-8"))
        assert written["data_prep"]["exclude_features"] == ["beitrag", "GIBTSNICHT"]

    def test_no_false_typo_warning_for_already_non_input(self, tmp_path, caplog):
        """UI-Flow "Liste → Checkbox-Abwahl": Der Name steht dann in
        exclude_features UND fehlt bereits in der features-Liste — das ist
        kein Tippfehler und darf nicht warnen; ein Fantasiename schon."""
        out = tmp_path / "out"; out.mkdir()
        raw = _make_raw(tmp_path)
        cfg = _cfg(tmp_path, raw, out, exclude=["REGION", "GIBTSNICHT"])
        yaml_cfg = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        yaml_cfg["data_prep"]["features"] = ["ALTER", "BEITRAG"]  # REGION bereits abgewählt
        cfg.write_text(yaml.safe_dump(yaml_cfg), encoding="utf-8")
        import logging
        with caplog.at_level(logging.WARNING):
            X = _run(cfg, out)
        assert sorted(X.columns) == ["ALTER", "BEITRAG"]
        warn = " ".join(r.getMessage() for r in caplog.records)
        assert "GIBTSNICHT" in warn and "REGION" not in warn
        # Separates Weglogging dokumentiert die Use-Case-Liste auch dann,
        # wenn der Filter nichts mehr zu entfernen hatte (UI-Flow):
        assert (out / "exclude_features_used.txt").read_text(encoding="utf-8") == "REGION\n"

    def test_categorical_exclusion_and_protected_columns(self, tmp_path):
        out = tmp_path / "out"; out.mkdir()
        raw = _make_raw(tmp_path)
        cfg = _cfg(tmp_path, raw, out, feature_path=_dictionary(tmp_path),
                   exclude=["REGION", "Y", "T"])   # Y/T geschützt
        X = _run(cfg, out)
        assert sorted(X.columns) == ["ALTER", "BEITRAG"]
        assert (out / "Y.parquet").exists() and (out / "T.parquet").exists()

    def test_fallback_branch_without_dictionary(self, tmp_path):
        out = tmp_path / "out"; out.mkdir()
        raw = _make_raw(tmp_path)
        cfg = _cfg(tmp_path, raw, out, exclude=["ALTER"])
        X = _run(cfg, out)
        assert sorted(X.columns) == ["BEITRAG", "REGION"]

    def test_pure_typo_no_txt_and_empty_applied(self, tmp_path, caplog):
        """Nur Tippfehler in der Liste: Warnung ja, aber KEINE txt und
        applied=[] in der Kopie — die Report-Kachel zeigt dann nichts,
        statt Tippfehler als Ausschluss auszuweisen."""
        out = tmp_path / "out"; out.mkdir()
        raw = _make_raw(tmp_path)
        cfg = _cfg(tmp_path, raw, out, feature_path=_dictionary(tmp_path),
                   exclude=["GIBTSNICHT"])
        import logging
        with caplog.at_level(logging.WARNING):
            X = _run(cfg, out)
        assert sorted(X.columns) == ["ALTER", "BEITRAG", "REGION"]
        assert not (out / "exclude_features_used.txt").exists()
        written = yaml.safe_load((out / "dataprep_config.yml").read_text(encoding="utf-8"))
        assert written["data_prep"]["exclude_features_applied"] == []

    def test_applied_written_to_config_copy(self, tmp_path):
        out = tmp_path / "out"; out.mkdir()
        raw = _make_raw(tmp_path)
        cfg = _cfg(tmp_path, raw, out, feature_path=_dictionary(tmp_path),
                   exclude=["beitrag", "GIBTSNICHT"])
        _run(cfg, out)
        written = yaml.safe_load((out / "dataprep_config.yml").read_text(encoding="utf-8"))
        assert written["data_prep"]["exclude_features_applied"] == ["BEITRAG"]

    def test_without_exclude_unchanged_and_no_txt(self, tmp_path):
        out = tmp_path / "out"; out.mkdir()
        raw = _make_raw(tmp_path)
        cfg = _cfg(tmp_path, raw, out, feature_path=_dictionary(tmp_path))
        X = _run(cfg, out)
        assert sorted(X.columns) == ["ALTER", "BEITRAG", "REGION"]
        assert not (out / "exclude_features_used.txt").exists()


class TestDictionaryInputsRoute:
    def test_route_returns_inputs_and_validates(self, tmp_path):
        from app.server import app
        d = _dictionary(tmp_path)
        with app.test_client() as c:
            r = c.post("/api/dictionary-inputs", json={"path": str(d)})
            assert r.status_code == 200
            body = r.get_json()
            assert body["status"] == "ok" and body["inputs"] == ["ALTER", "BEITRAG", "REGION"]
            r2 = c.post("/api/dictionary-inputs", json={"path": str(tmp_path / "fehlt.csv")})
            assert r2.status_code == 404
