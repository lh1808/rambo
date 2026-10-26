"""Deduplizierung über mehrere Dateien: zufälliger Vertreter je ID.

Vorher gewann bei ID-Überschneidungen zwischen Dateien IMMER die frühere
Datei (drop_duplicates keep="first" auf dem konkatenierten Frame) — ein
Selektionsbias: spätere Dateien (z. B. das TMES-Eval-File oder jüngere
Kampagnenwellen) verloren systematisch alle Überschneidungen. Jetzt wird
seed-reproduzierbar gemischt, dedupliziert und die ursprüngliche
Zeilenordnung wiederhergestellt: je ID gewinnt ein zufälliger Vertreter,
gleichverteilt über die Dateien."""
import numpy as np
import pandas as pd
import yaml

from rubin.pipelines.data_prep_pipeline import DataPrepPipeline


def _run(tmp_path, sub, seed=7):
    tmp = tmp_path / sub
    tmp.mkdir()
    ids = np.arange(200)
    for j, name in enumerate(["f1.csv", "f2.csv"]):
        pd.DataFrame({"KUNDE_ID": ids, "ALTER": 30 + j, "BEITRAG": 100.0 + j,
                      "T": (ids % 2), "Y": ((ids // 2) % 2)}).to_csv(tmp / name, index=False)
    outd = tmp / "out"
    outd.mkdir()
    cfg = {"mlflow": {"experiment_name": "t"},
           "constants": {"SEED": seed, "work_dir": str(tmp / "runs")},
           "data_files": {"x_file": str(outd / "X.parquet"),
                          "y_file": str(outd / "Y.parquet"),
                          "t_file": str(outd / "T.parquet")},
           "data_prep": {"data_path": [str(tmp / "f1.csv"), str(tmp / "f2.csv")],
                         "output_path": str(outd), "target": "Y", "treatment": "T",
                         "features": ["ALTER", "BEITRAG"],
                         "deduplicate": True, "deduplicate_id_column": "KUNDE_ID",
                         "score_name": None, "log_to_mlflow": False}}
    p = tmp / "c.yml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    DataPrepPipeline.from_config_path(str(p)).run()
    return pd.read_parquet(outd / "X.parquet")


class TestDedupRandomAcrossFiles:
    def test_random_representative_not_first_file_wins(self, tmp_path):
        X = _run(tmp_path, "a")
        counts = X["ALTER"].value_counts().to_dict()  # 30=Datei 1, 31=Datei 2
        n1, n2 = counts.get(30, 0), counts.get(31, 0)
        assert n1 + n2 == 200
        # Vorher: 200/0. Jetzt: zufällig verteilt — beide Dateien substanziell vertreten.
        assert 60 <= n1 <= 140 and 60 <= n2 <= 140, (n1, n2)

    def test_same_seed_is_fully_reproducible(self, tmp_path):
        X1 = _run(tmp_path, "b1")
        X2 = _run(tmp_path, "b2")
        assert X1["ALTER"].tolist() == X2["ALTER"].tolist()

    def test_file_source_artifact_written_for_multi_file(self, tmp_path):
        # Nebenprodukt der Mehrdatei-Läufe: file_source.parquet ermöglicht
        # die Je-Datei-Qini-Aufschlüsselung im Report.
        X = _run(tmp_path, "fs")
        fs = pd.read_parquet(tmp_path / "fs" / "out" / "file_source.parquet")["file_source"]
        assert len(fs) == len(X) and fs.nunique() == 2

    def test_file_source_uses_filenames_and_survives_tmes(self, tmp_path):
        """Zuordnung trägt Dateinamen (nicht Indizes) und ist bei TMES VOLL
        (X wird ungesplittet gespeichert; die Analyse richtet über die Maske
        aus). Realer Vorfall: Indizes '0'/'1' statt Namen."""
        import numpy as np
        import yaml
        tmp = tmp_path / "tm"
        tmp.mkdir()
        ids = np.arange(80)
        for j, name in enumerate(["f1.csv", "f2.csv"]):
            pd.DataFrame({"KUNDE_ID": ids + j * 1000, "ALTER": 30 + j, "BEITRAG": 100.0,
                          "T": (ids % 2), "Y": ((ids // 2) % 2)}).to_csv(tmp / name, index=False)
        outd = tmp / "out"
        outd.mkdir()
        cfg = {"mlflow": {"experiment_name": "t"},
               "constants": {"SEED": 7, "work_dir": str(tmp / "runs")},
               "data_files": {"x_file": str(outd / "X.parquet"),
                              "y_file": str(outd / "Y.parquet"),
                              "t_file": str(outd / "T.parquet")},
               "data_prep": {"data_path": [str(tmp / "f1.csv"), str(tmp / "f2.csv")],
                             "output_path": str(outd), "target": "Y", "treatment": "T",
                             "features": ["ALTER", "BEITRAG"], "eval_file_index": 1,
                             "score_name": None, "log_to_mlflow": False}}
        p = tmp / "c.yml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        DataPrepPipeline.from_config_path(str(p)).run()
        X = pd.read_parquet(outd / "X.parquet")
        fs = pd.read_parquet(outd / "file_source.parquet")["file_source"]
        mask = np.load(outd / "eval_mask.npy").astype(bool)
        assert len(X) == len(fs) == len(mask) == 160
        assert set(fs[mask]) == {"f2.csv"} and set(fs[~mask]) == {"f1.csv"}

    def test_different_seed_changes_selection(self, tmp_path):
        X1 = _run(tmp_path, "c1", seed=7)
        X2 = _run(tmp_path, "c2", seed=8)
        assert X1["ALTER"].tolist() != X2["ALTER"].tolist()


class TestEvalMaskMlflowLogging:
    def test_mask_and_file_source_logged_as_artifacts(self, tmp_path):
        """Realer Vorfall: eval_mask.npy war in den mlflow-Logs nicht auffindbar —
        TMES-Maske und Datei-Zuordnung werden jetzt als Run-Artefakte geloggt
        (die geloggten Predictions enthalten ALLE Zeilen; erst die Maske macht
        das Eval-Subset im Nachhinein rekonstruierbar)."""
        import os
        import numpy as np
        import yaml
        import mlflow
        uri = f"sqlite:///{tmp_path / 'ml.db'}"
        os.environ["MLFLOW_TRACKING_URI"] = uri
        mlflow.set_tracking_uri(uri)
        rng = np.random.default_rng(3)
        for name in ["train_a.csv", "eval_b.csv"]:
            pd.DataFrame({"ALTER": rng.integers(18, 80, 60), "BEITRAG": 100.0,
                          "T": rng.integers(0, 2, 60), "Y": rng.integers(0, 2, 60)}
                         ).to_csv(tmp_path / name, index=False)
        outd = tmp_path / "out"
        outd.mkdir()
        cfg = {"mlflow": {"experiment_name": "mask_log_test"},
               "constants": {"SEED": 1, "work_dir": str(tmp_path / "runs")},
               "data_files": {"x_file": str(outd / "X.parquet"),
                              "y_file": str(outd / "Y.parquet"),
                              "t_file": str(outd / "T.parquet")},
               "data_prep": {"data_path": [str(tmp_path / "train_a.csv"), str(tmp_path / "eval_b.csv")],
                             "output_path": str(outd), "target": "Y", "treatment": "T",
                             "features": ["ALTER", "BEITRAG"], "eval_file_index": 1,
                             "score_name": None, "log_to_mlflow": True}}
        p = tmp_path / "c.yml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        DataPrepPipeline.from_config_path(str(p)).run()
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("mask_log_test")
        run = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"])[0]
        arts = {a.path for a in client.list_artifacts(run.info.run_id)}
        assert "eval_mask.npy" in arts and "file_source.parquet" in arts


class TestAllNanColumnFillFallback:
    """Realer Vorfall: CausalForest wurde wegen "fehlender Werte" übersprungen,
    obwohl DataPrep fill_na_method="max" hatte. Wurzel: Für KOMPLETT leere
    Spalten liefert max/mean selbst NaN (mode gar nichts) — fillna(NaN) ist ein
    No-op, die Spalte blieb voll NaN. Jetzt: 0.0-Fallback + Warnung."""

    def test_all_nan_column_gets_zero_fallback(self, tmp_path):
        import numpy as np
        import yaml
        import json
        rng = np.random.default_rng(3)
        n = 120
        df = pd.DataFrame({"ALTER": rng.integers(18, 80, n).astype(float),
                           "LEER": np.nan,
                           "T": rng.integers(0, 2, n), "Y": rng.integers(0, 2, n)})
        df.loc[rng.choice(n, 20, replace=False), "ALTER"] = np.nan
        df.to_csv(tmp_path / "f.csv", index=False)
        outd = tmp_path / "out"
        outd.mkdir()
        cfg = {"mlflow": {"experiment_name": "t"},
               "constants": {"SEED": 1, "work_dir": str(tmp_path / "runs")},
               "data_files": {"x_file": str(outd / "X.parquet"),
                              "y_file": str(outd / "Y.parquet"),
                              "t_file": str(outd / "T.parquet")},
               "data_prep": {"data_path": [str(tmp_path / "f.csv")], "output_path": str(outd),
                             "target": "Y", "treatment": "T", "features": ["ALTER", "LEER"],
                             "fill_na_method": "max", "score_name": None, "log_to_mlflow": False}}
        p = tmp_path / "c.yml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        DataPrepPipeline.from_config_path(str(p)).run()
        X = pd.read_parquet(outd / "X.parquet")
        assert int(X.isnull().sum().sum()) == 0
        mv = json.load(open(outd / "missing_values.json"))
        assert mv.get("LEER") == 0.0 and mv.get("ALTER") == df["ALTER"].max()


class TestCustomOutputPaths:
    """Eigene Ausgabepfade: DataPrep schreibt nur nach output_path (feste Namen),
    die Analyse liest data_files.* und sucht Nebenartefakte neben x_file.
    Divergenz wird jetzt laut gewarnt; die file_source-Suche hat einen Fallback
    über das Verzeichnis der konfigurierten eval_mask_file."""

    def test_divergence_warning_fires_only_on_mismatch(self, tmp_path, caplog):
        import logging
        import numpy as np
        import yaml
        for differs, sub in [(True, "div"), (False, "ok")]:
            tmp = tmp_path / sub
            tmp.mkdir()
            pd.DataFrame({"A": [1.0, 2.0] * 30, "T": [0, 1] * 30, "Y": [0, 1] * 30}).to_csv(tmp / "f.csv", index=False)
            outd = tmp / "out"
            outd.mkdir()
            other = tmp / "woanders"
            other.mkdir()
            base = other if differs else outd
            cfg = {"mlflow": {"experiment_name": "t"},
                   "constants": {"SEED": 1, "work_dir": str(tmp / "runs")},
                   "data_files": {"x_file": str(base / "X.parquet"),
                                  "y_file": str(outd / "Y.parquet"),
                                  "t_file": str(outd / "T.parquet")},
                   "data_prep": {"data_path": [str(tmp / "f.csv")], "output_path": str(outd),
                                 "target": "Y", "treatment": "T", "features": ["A"],
                                 "score_name": None, "log_to_mlflow": False}}
            p = tmp / "c.yml"
            p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with caplog.at_level(logging.WARNING):
                caplog.clear()
                DataPrepPipeline.from_config_path(str(p)).run()
            fired = any("zeigen NICHT auf output_path" in r.message for r in caplog.records)
            assert fired == differs

    def test_file_source_fallback_via_eval_mask_dir(self, tmp_path):
        import numpy as np
        import logging
        from types import SimpleNamespace
        from rubin.pipelines.analysis_pipeline import AnalysisPipeline
        art = tmp_path / "artefakte"
        art.mkdir()
        leer = tmp_path / "leer"
        leer.mkdir()
        n = 200
        fs = np.array(["a.parquet"] * 120 + ["b.parquet"] * 80)
        pd.DataFrame({"file_source": fs}).to_parquet(art / "file_source.parquet")
        rng = np.random.default_rng(0)
        y = (rng.random(n) < 0.1).astype(float)
        t = (rng.random(n) > 0.5).astype(int)
        fake = SimpleNamespace(_logger=logging.getLogger("t"),
                               _eval_scores_ctx={"C": (y, t, rng.normal(size=n))},
                               _eval_scores_ctx_full={}, _eval_scores_mask=None)
        cfg = SimpleNamespace(
            data_files=SimpleNamespace(x_file=str(leer / "X.parquet"),
                                       eval_mask_file=str(art / "eval_mask.npy")),
            historical_score=SimpleNamespace(name=None),
            data_processing=SimpleNamespace(validate_on="cross"))
        res = AnalysisPipeline._compute_per_file_qini(fake, cfg, "C")
        assert [r["file"] for r in res["rows"]] == ["a.parquet", "b.parquet", "GESAMT"]



class TestRowProvenance:
    """row_provenance.parquet: Nach Dedup (zufälliger Vertreter je ID) und
    Treatment-Balance ist im Nachhinein exakt rekonstruierbar, welche Zeilen
    der Originaldateien behalten wurden — Spalten: orig_row (0-basiert, je
    Datei), file (Basename), ID-Spalte (bei aktivem Dedup). Zeile i von
    X.parquet == Original-Zeile orig_row[i] der Datei file[i]."""

    def test_provenance_reconstructs_exact_original_rows(self, tmp_path):
        import numpy as np
        import yaml
        rng = np.random.default_rng(5)
        orig = {}
        for j, name in enumerate(["welle_a.csv", "welle_b.csv"]):
            n = 300
            ids = np.concatenate([np.arange(180), np.arange(1000 + j * 500, 1000 + j * 500 + 120)])
            rng.shuffle(ids)
            d = pd.DataFrame({"KUNDE_ID": ids,
                              "ALTER": rng.integers(18, 80, n).astype(float),
                              "MARKER": rng.normal(size=n).round(6),
                              "T": (rng.random(n) < (0.3 if j == 0 else 0.7)).astype(int),
                              "Y": rng.integers(0, 2, n)})
            d.to_csv(tmp_path / name, index=False)
            orig[name] = d
        outd = tmp_path / "out"
        outd.mkdir()
        cfg = {"mlflow": {"experiment_name": "t"},
               "constants": {"SEED": 7, "work_dir": str(tmp_path / "runs")},
               "data_files": {"x_file": str(outd / "X.parquet"),
                              "y_file": str(outd / "Y.parquet"),
                              "t_file": str(outd / "T.parquet")},
               "data_prep": {"data_path": [str(tmp_path / "welle_a.csv"), str(tmp_path / "welle_b.csv")],
                             "output_path": str(outd), "target": "Y", "treatment": "T",
                             "features": ["ALTER", "MARKER"],
                             "deduplicate": True, "deduplicate_id_column": "KUNDE_ID",
                             "balance_treatments": True,
                             "score_name": None, "log_to_mlflow": False}}
        p = tmp_path / "c.yml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        DataPrepPipeline.from_config_path(str(p)).run()
        X = pd.read_parquet(outd / "X.parquet")
        T = pd.read_parquet(outd / "T.parquet")["T"]
        prov = pd.read_parquet(outd / "row_provenance.parquet")
        assert list(prov.columns) == ["orig_row", "file", "KUNDE_ID"]
        assert len(prov) == len(X) < 600
        for i in range(len(prov)):
            o = orig[prov["file"].iloc[i]].iloc[int(prov["orig_row"].iloc[i])]
            assert o["KUNDE_ID"] == prov["KUNDE_ID"].iloc[i]
            # X ist durch die Memory-Reduktion float32 → Toleranz entsprechend
            assert abs(o["MARKER"] - float(X["MARKER"].iloc[i])) < 1e-5
            assert int(o["T"]) == int(T.iloc[i])
        assert prov.groupby("KUNDE_ID").size().max() == 1
