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

    def test_different_seed_changes_selection(self, tmp_path):
        X1 = _run(tmp_path, "c1", seed=7)
        X2 = _run(tmp_path, "c2", seed=8)
        assert X1["ALTER"].tolist() != X2["ALTER"].tolist()
