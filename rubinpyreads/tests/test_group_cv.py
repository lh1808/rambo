"""Gruppen-CV (Zwillings-Leakage-Prävention bei Mehrfach-Samples je Einheit).

Use Case: 2-3 echte Samples pro Partner (unterschiedliche Lead-Arten,
T konstant je Partner, Y kann variieren). Deduplizierung würde echte
Beobachtungen wegwerfen — stattdessen hält `data_prep.group_column` alle
Zeilen einer Gruppe im selben CV-Fold, durchgängig:
Cross-Prediction, BLT/FMT/CFT-Tuning und EconMLs internes Cross-Fitting.

Kernzusagen:
1. OHNE groups ändert sich NICHTS (GroupAwareStratifiedKFold delegiert
   seed-identisch an StratifiedKFold) — volle Rückwärtskompatibilität.
2. MIT groups liegt keine Gruppe in Train UND Validation (Fold-Reinheit).
3. DataPrep schreibt groups.parquet, die Analyse erkennt sie automatisch.
"""
import logging

import numpy as np
import pandas as pd
import pytest
import yaml

from rubin.utils.group_cv import GroupAwareStratifiedKFold


def _toy(n=400, n_groups=150, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("ABCD"))
    groups = rng.integers(0, n_groups, n).astype(str)
    T = rng.integers(0, 2, n)
    Y = ((0.3 * T + rng.normal(size=n)) > 0.5).astype(int)
    return X, T, Y, groups


def _assert_pure(folds, groups):
    for tr, va in folds:
        overlap = set(np.asarray(groups)[tr]) & set(np.asarray(groups)[va])
        assert not overlap, f"Gruppen in Train UND Val: {sorted(overlap)[:5]}"


class TestGroupAwareStratifiedKFold:
    def test_without_groups_identical_to_stratified_kfold(self):
        """Rückwärtskompatibilität: ohne groups exakt dieselben Folds wie bisher."""
        from sklearn.model_selection import StratifiedKFold
        _, T, Y, _ = _toy()
        strata = T * 10 + Y
        ga = list(GroupAwareStratifiedKFold(5, True, 42).split(np.zeros((len(strata), 1)), strata))
        sk = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros((len(strata), 1)), strata))
        assert len(ga) == len(sk)
        for (a_tr, a_va), (b_tr, b_va) in zip(ga, sk):
            assert (a_tr == b_tr).all() and (a_va == b_va).all()

    def test_with_groups_fold_purity(self):
        _, T, Y, groups = _toy()
        folds = list(GroupAwareStratifiedKFold(5, True, 42).split(np.zeros((len(T), 1)), T * 10 + Y, groups))
        assert len(folds) == 5
        _assert_pure(folds, groups)
        # Jede Zeile genau einmal in Val
        all_va = np.concatenate([va for _, va in folds])
        assert sorted(all_va.tolist()) == list(range(len(T)))

    def test_n_splits_capped_to_n_groups(self):
        """n_splits > Anzahl Gruppen → defensiv kappen statt crashen."""
        y = np.array([0, 1, 0, 1, 0, 1])
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        cv = GroupAwareStratifiedKFold(5, True, 1)
        folds = list(cv.split(np.zeros((6, 1)), y, groups))
        assert len(folds) == 3
        assert cv.get_n_splits(groups=groups) == 3
        _assert_pure(folds, groups)

    def test_single_group_falls_back_to_rowwise(self):
        y = np.array([0, 1] * 10)
        groups = np.array(["x"] * 20)
        folds = list(GroupAwareStratifiedKFold(4, True, 1).split(np.zeros((20, 1)), y, groups))
        assert len(folds) == 4  # Fallback: zeilenweise StratifiedKFold

    def test_not_a_sklearn_kfold_subclass(self):
        """EconML wirft TypeError bei fit(groups=) + (Stratified)KFold-INSTANZ.
        Der Wrapper darf deshalb keine Subklasse sein."""
        from sklearn.model_selection import KFold, StratifiedKFold
        assert not isinstance(GroupAwareStratifiedKFold(3), (KFold, StratifiedKFold))

    def test_econml_fit_with_and_without_groups(self):
        """Der entscheidende Integrationspunkt: EconML akzeptiert den Wrapper
        als cv sowohl MIT groups (→ StratifiedGroupKFold) als auch OHNE."""
        from econml.dml import LinearDML
        from lightgbm import LGBMClassifier
        X, T, Y, groups = _toy(300, 100, 3)

        def mk():
            return LinearDML(
                model_y=LGBMClassifier(n_estimators=10, verbosity=-1),
                model_t=LGBMClassifier(n_estimators=10, verbosity=-1),
                discrete_treatment=True, discrete_outcome=True,
                cv=GroupAwareStratifiedKFold(n_splits=3, shuffle=True, random_state=1),
                random_state=1,
            )

        est = mk()
        est.fit(Y, T, X=X)  # ohne groups: kein TypeError, normales Verhalten
        assert np.asarray(est.effect(X)).shape[0] == len(X)
        est2 = mk()
        est2.fit(Y, T, X=X, groups=groups)  # mit groups: kein TypeError
        assert np.asarray(est2.effect(X)).shape[0] == len(X)


class TestRegistryUsesGroupAwareCV:
    def test_dml_dr_factories_build_group_aware_cv(self):
        """Alle DML/DR-Estimatoren aus der Registry müssen den groups-fähigen
        Splitter tragen — sonst crasht fit(groups=...) mit EconML-TypeError."""
        from rubin.model_registry import default_registry, ModelContext
        reg = default_registry()
        ctx = ModelContext(seed=1, base_learner_type="lgbm", dml_crossfit_folds=3)
        for name in ["NonParamDML", "ParamDML", "CausalForestDML", "DRLearner"]:
            est = reg.create(name, ctx)
            assert isinstance(est.cv, GroupAwareStratifiedKFold), name


class TestTuningSplitsGrouped:
    def test_iter_stratified_grouped_purity(self):
        from rubin.tuning.common import _iter_stratified_or_kfold
        _, T, Y, groups = _toy()
        folds = list(_iter_stratified_or_kfold(T * 10 + Y, 4, 42, groups=groups))
        assert len(folds) == 4
        _assert_pure(folds, groups)

    def test_iter_stratified_without_groups_unchanged(self):
        from sklearn.model_selection import StratifiedKFold
        from rubin.tuning.common import _iter_stratified_or_kfold
        _, T, Y, _ = _toy()
        strata = T * 10 + Y
        a = [(tr.tolist(), va.tolist()) for tr, va in _iter_stratified_or_kfold(strata, 4, 42)]
        b = [(tr.tolist(), va.tolist())
             for tr, va in StratifiedKFold(4, shuffle=True, random_state=42).split(np.zeros(len(strata)), strata)]
        assert a == b

    def test_iter_stratified_group_length_mismatch_falls_back(self, caplog):
        from rubin.tuning.common import _iter_stratified_or_kfold
        _, T, Y, _ = _toy()
        with caplog.at_level(logging.WARNING, logger="rubin.tuning"):
            folds = list(_iter_stratified_or_kfold(T * 10 + Y, 3, 42, groups=np.array(["a", "b"])))
        assert len(folds) == 3
        assert any("Gruppen-CV ignoriert" in r.message for r in caplog.records)


class TestCrossPredictionGrouped:
    def _model(self):
        from econml.dml import LinearDML
        from lightgbm import LGBMClassifier
        return LinearDML(
            model_y=LGBMClassifier(n_estimators=10, verbosity=-1),
            model_t=LGBMClassifier(n_estimators=10, verbosity=-1),
            discrete_treatment=True, discrete_outcome=True,
            cv=GroupAwareStratifiedKFold(n_splits=2, shuffle=True, random_state=1),
            random_state=1,
        )

    def test_crosspredict_fold_purity_with_groups(self):
        from rubin.training import train_and_crosspredict_bt_bo, get_fold_aligned_data
        X, T, Y, groups = _toy()
        out = train_and_crosspredict_bt_bo(
            model=self._model(), X=X, Y=Y, T=T, n_splits=3,
            model_name="LinearDML", random_state=7, parallel_level=1, groups=groups,
        )
        _, fold_splits = get_fold_aligned_data(out)
        assert fold_splits and len(fold_splits) == 3
        _assert_pure(fold_splits, groups)
        assert out[f"Predictions_LinearDML"].notna().all()

    def test_crosspredict_group_length_mismatch_ignored(self, caplog):
        from rubin.training import train_and_crosspredict_bt_bo
        X, T, Y, _ = _toy(200, 80, 5)
        with caplog.at_level(logging.WARNING, logger="rubin.training"):
            out = train_and_crosspredict_bt_bo(
                model=self._model(), X=X, Y=Y, T=T, n_splits=2,
                model_name="LinearDML", random_state=7, parallel_level=1,
                groups=np.array(["a"] * 50),
            )
        assert len(out) == len(X)
        assert any("Gruppen-CV wird ignoriert" in r.message for r in caplog.records)

    def test_twin_leakage_actually_closed(self):
        """Beweis am Mechanismus: exakte Zwillinge (gleiches X, T und Y je
        Partner) + memorisierender Learner auf den von rubin erzeugten Folds.
        Zeilenweise Folds: der Zwilling liegt im Trainfold, das Modell erkennt
        ihn im Val wieder → OOF-Vorhersage korreliert stark mit purem
        Y-Rauschen (~0.6). Gruppen-Folds: kein Zwilling im Trainfold → ~0."""
        from rubin.training import train_and_crosspredict_bt_bo, get_fold_aligned_data
        from econml.metalearners import SLearner
        from lightgbm import LGBMRegressor

        rng = np.random.default_rng(11)
        n_partner = 200
        base = rng.normal(size=(n_partner, 3))
        X = pd.DataFrame(np.repeat(base, 2, axis=0), columns=list("ABC"))
        groups = np.repeat(np.arange(n_partner), 2).astype(str)
        T = np.repeat(rng.integers(0, 2, n_partner), 2)
        Y = np.repeat(rng.integers(0, 2, n_partner), 2)  # pures Rauschen, je Zwilling geteilt

        def folds_of(groups_arg):
            out = train_and_crosspredict_bt_bo(
                model=SLearner(overall_model=LGBMRegressor(n_estimators=5, verbosity=-1)),
                X=X, Y=Y, T=T, n_splits=3, model_name="S", random_state=3,
                parallel_level=1, return_train_predictions=False, groups=groups_arg)
            return get_fold_aligned_data(out)[1]

        def oof_corr(folds):
            p = np.full(len(Y), np.nan)
            for tr, va in folds:
                m = LGBMRegressor(n_estimators=300, num_leaves=255, min_child_samples=1,
                                  learning_rate=0.3, verbosity=-1, random_state=0)
                m.fit(X.iloc[tr], Y[tr])
                p[va] = m.predict(X.iloc[va])
            return float(abs(np.corrcoef(p, Y - Y.mean())[0, 1]))

        corr_rowwise = oof_corr(folds_of(None))
        corr_grouped = oof_corr(folds_of(groups))
        assert corr_rowwise > 0.4, corr_rowwise          # Leakage-Kanal offen
        assert corr_grouped < 0.15, corr_grouped         # Kanal geschlossen
        assert corr_grouped < corr_rowwise / 3


# ───────────────────────── DataPrep: group_column ─────────────────────────

def _dataprep_cfg(tmp, files, extra_dp=None):
    outd = tmp / "out"
    outd.mkdir(exist_ok=True)
    dp = {"data_path": [str(f) for f in files], "output_path": str(outd),
          "target": "Y", "treatment": "T", "score_name": None, "log_to_mlflow": False}
    dp.update(extra_dp or {})
    cfg = {"mlflow": {"experiment_name": "t"},
           "constants": {"SEED": 7, "work_dir": str(tmp / "runs")},
           "data_files": {"x_file": str(outd / "X.parquet"),
                          "y_file": str(outd / "Y.parquet"),
                          "t_file": str(outd / "T.parquet")},
           "data_prep": dp}
    p = tmp / "c.yml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p, outd


class TestDataPrepGroupColumn:
    def _write_leads(self, tmp, with_nan=False):
        rng = np.random.default_rng(1)
        n_partner = 120
        rows = []
        for pid in range(n_partner):
            k = 1 + (pid % 3)  # 1-3 Leads je Partner
            for lead in range(k):
                rows.append({"PARTNER_ID": f"P{pid:04d}", "ALTER": 30 + pid % 40,
                             "LEAD_ART": lead, "T": pid % 2, "Y": int(rng.random() < 0.3)})
        df = pd.DataFrame(rows)
        if with_nan:
            df.loc[df.index[:5], "PARTNER_ID"] = None
        f = tmp / "leads.csv"
        df.to_csv(f, index=False)
        return f, df

    def test_groups_parquet_written_and_aligned(self, tmp_path):
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        f, df = self._write_leads(tmp_path)
        p, outd = _dataprep_cfg(tmp_path, [f], {"group_column": "PARTNER_ID",
                                                "features": ["ALTER", "LEAD_ART"]})
        DataPrepPipeline.from_config_path(str(p)).run()
        X = pd.read_parquet(outd / "X.parquet")
        g = pd.read_parquet(outd / "groups.parquet")["group"]
        assert len(g) == len(X) == len(df)
        # Ohne Dedup/Balance bleibt die Reihenfolge erhalten → 1:1-Abgleich
        assert g.tolist() == df["PARTNER_ID"].astype(str).tolist()
        # Gruppen-Spalte ist KEIN Feature
        assert "PARTNER_ID" not in X.columns

    def test_group_column_excluded_from_auto_features(self, tmp_path):
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        f, _ = self._write_leads(tmp_path)
        # Keine explizite Feature-Liste → Auto-Branch (alle außer Y/T/ID/Gruppe)
        p, outd = _dataprep_cfg(tmp_path, [f], {"group_column": "PARTNER_ID"})
        DataPrepPipeline.from_config_path(str(p)).run()
        X = pd.read_parquet(outd / "X.parquet")
        assert "PARTNER_ID" not in X.columns
        assert "ALTER" in X.columns and "LEAD_ART" in X.columns

    def test_group_column_dropped_from_explicit_features_with_warning(self, tmp_path, caplog):
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        f, _ = self._write_leads(tmp_path)
        p, outd = _dataprep_cfg(tmp_path, [f], {"group_column": "PARTNER_ID",
                                                "features": ["ALTER", "LEAD_ART", "PARTNER_ID"]})
        with caplog.at_level(logging.WARNING):
            DataPrepPipeline.from_config_path(str(p)).run()
        X = pd.read_parquet(outd / "X.parquet")
        assert "PARTNER_ID" not in X.columns
        assert any("aus den Features entfernt" in r.message for r in caplog.records)

    def test_nan_group_ids_become_singletons(self, tmp_path):
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        f, df = self._write_leads(tmp_path, with_nan=True)
        p, outd = _dataprep_cfg(tmp_path, [f], {"group_column": "PARTNER_ID",
                                                "features": ["ALTER", "LEAD_ART"]})
        DataPrepPipeline.from_config_path(str(p)).run()
        g = pd.read_parquet(outd / "groups.parquet")["group"]
        singles = [v for v in g if str(v).startswith("__einzeln_")]
        assert len(singles) == 5
        assert len(set(singles)) == 5  # jede NaN-Zeile ihre eigene Gruppe
        assert "nan" not in set(g.astype(str))

    def test_missing_group_column_raises_with_clear_message(self, tmp_path):
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        f, _ = self._write_leads(tmp_path)
        p, _ = _dataprep_cfg(tmp_path, [f], {"group_column": "GIBTS_NICHT",
                                             "features": ["ALTER"]})
        with pytest.raises(ValueError, match="group_column 'GIBTS_NICHT'"):
            DataPrepPipeline.from_config_path(str(p)).run()

    def test_groups_survive_dedup_off_multifile(self, tmp_path):
        """Mehrdatei-Fall: groups.parquet deckt den konkatenierten Frame ab."""
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        ids = np.arange(100)
        files = []
        for j, name in enumerate(["f1.csv", "f2.csv"]):
            f = tmp_path / name
            pd.DataFrame({"PARTNER_ID": [f"P{i}" for i in ids], "ALTER": 30 + j,
                          "T": j, "Y": (ids % 2)}).to_csv(f, index=False)
            files.append(f)
        p, outd = _dataprep_cfg(tmp_path, files, {"group_column": "PARTNER_ID",
                                                  "features": ["ALTER"]})
        DataPrepPipeline.from_config_path(str(p)).run()
        X = pd.read_parquet(outd / "X.parquet")
        g = pd.read_parquet(outd / "groups.parquet")["group"]
        assert len(g) == len(X) == 200
        assert g.value_counts().max() == 2  # jede Partner-ID in beiden Dateien


# ───────────────────── Analyse: Laden + Auto-Erkennung ─────────────────────

def _write_xty(tmp, n=300, seed=2, with_groups=True, n_partner=100):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("ABC"))
    T = pd.DataFrame({"T": rng.integers(0, 2, n)})
    Y = pd.DataFrame({"Y": rng.integers(0, 2, n)})
    X.to_parquet(tmp / "X.parquet")
    T.to_parquet(tmp / "T.parquet")
    Y.to_parquet(tmp / "Y.parquet")
    if with_groups:
        g = rng.integers(0, n_partner, n).astype(str)
        pd.DataFrame({"group": g}).to_parquet(tmp / "groups.parquet")
        return g
    return None


def _analysis_pipeline(tmp, df_frac=None, groups_file=None):
    from rubin.settings import AnalysisConfig
    from rubin.pipelines.analysis_pipeline import AnalysisPipeline
    cfg = {
        "data_files": {"x_file": str(tmp / "X.parquet"), "t_file": str(tmp / "T.parquet"),
                       "y_file": str(tmp / "Y.parquet")},
        "mlflow": {"experiment_name": "t"},
        "constants": {"SEED": 5, "work_dir": str(tmp / "runs")},
    }
    if groups_file:
        cfg["data_files"]["groups_file"] = groups_file
    if df_frac:
        cfg["data_processing"] = {"df_frac": df_frac}
    return AnalysisPipeline(AnalysisConfig.model_validate(cfg))


class TestAnalysisLoadsGroups:
    def test_auto_detect_groups_next_to_x_file(self, tmp_path, caplog):
        g = _write_xty(tmp_path)
        pipe = _analysis_pipeline(tmp_path)
        with caplog.at_level(logging.INFO):
            X, T, Y, S, mask = pipe._load_inputs()
        assert pipe._groups is not None
        assert len(pipe._groups) == len(X)
        assert pipe._groups.tolist() == list(g)
        assert any("Gruppen-CV aktiv" in r.message and "auto-erkannt" in r.message
                   for r in caplog.records)

    def test_explicit_groups_file_config(self, tmp_path):
        g = _write_xty(tmp_path)
        # Datei umbenennen → nur über expliziten Pfad auffindbar
        (tmp_path / "groups.parquet").rename(tmp_path / "partner_gruppen.parquet")
        pipe = _analysis_pipeline(tmp_path, groups_file=str(tmp_path / "partner_gruppen.parquet"))
        X, *_ = pipe._load_inputs()
        assert pipe._groups is not None and len(pipe._groups) == len(X)

    def test_no_groups_file_means_none(self, tmp_path):
        _write_xty(tmp_path, with_groups=False)
        pipe = _analysis_pipeline(tmp_path)
        pipe._load_inputs()
        assert pipe._groups is None

    def test_length_mismatch_ignored_with_warning(self, tmp_path, caplog):
        _write_xty(tmp_path, with_groups=False)
        pd.DataFrame({"group": ["a", "b"]}).to_parquet(tmp_path / "groups.parquet")
        pipe = _analysis_pipeline(tmp_path)
        with caplog.at_level(logging.WARNING):
            pipe._load_inputs()
        assert pipe._groups is None
        assert any("passt nicht zu X" in r.message for r in caplog.records)

    def test_df_frac_subsamples_groups_consistently(self, tmp_path):
        g = _write_xty(tmp_path)
        # Zeilen-Fingerprint: Spalte A trägt den Gruppen-Code als Zahl — damit
        # lässt sich nach dem Subsampling (X wird auf RangeIndex resettet)
        # zeilengenau prüfen, dass _groups synchron mit-subsampled wurde.
        full_X = pd.read_parquet(tmp_path / "X.parquet")
        full_X["A"] = np.asarray(g, dtype=float)
        full_X.to_parquet(tmp_path / "X.parquet")
        pipe = _analysis_pipeline(tmp_path, df_frac=0.5)
        X, T, Y, S, mask = pipe._load_inputs()
        assert pipe._groups is not None
        assert len(pipe._groups) == len(X)
        expected = [str(int(round(v))) for v in X["A"].to_numpy(dtype=float)]
        assert pipe._groups.tolist() == expected


# ──────────────── Tuner-Integration: BLT / FMT / CFT mit Gruppen ────────────────

def _tuning_cfg(tmp, **over):
    """Vollständige AnalysisConfig (validierte Defaults) für Tuner-Smokes."""
    from rubin.settings import AnalysisConfig
    base = {
        "data_files": {"x_file": str(tmp / "X.parquet"), "t_file": str(tmp / "T.parquet"),
                       "y_file": str(tmp / "Y.parquet")},
        "mlflow": {"experiment_name": "t"},
        "constants": {"SEED": 5, "work_dir": str(tmp / "runs")},
        "base_learner": {"type": "lgbm", "fixed_params": {"n_estimators": 15, "num_leaves": 7}},
    }
    base.update(over)
    return AnalysisConfig.model_validate(base)


class TestTunerIntegrationWithGroups:
    """Durchstich: Die drei Tuning-Stufen laufen mit groups fehlerfrei durch —
    insbesondere EconMLs fit(groups=...) + GroupAwareStratifiedKFold-cv
    (würde ohne Splitter-Tausch mit TypeError crashen)."""

    def test_blt_tune_all_with_groups(self, tmp_path):
        from rubin.tuning.base_learner import BaseLearnerTuner
        X, T, Y, groups = _toy(250, 90, 7)
        cfg = _tuning_cfg(tmp_path, tuning={"enabled": True, "n_trials": 2, "cv_splits": 2})
        tuner = BaseLearnerTuner(cfg)
        out = tuner.tune_all(["SLearner"], X=X, Y=Y, T=T, groups=groups)
        assert tuner._groups_all is not None and len(tuner._groups_all) == len(X)
        assert "SLearner" in out and out["SLearner"]

    def test_fmt_nonparamdml_with_groups(self, tmp_path):
        """Kritischster Pfad: NonParamDML.fit(cache_values=True, groups=...)
        mit gruppiertem äußerem UND innerem CV."""
        from rubin.tuning.final_model import FinalModelTuner
        X, T, Y, groups = _toy(250, 90, 8)
        cfg = _tuning_cfg(
            tmp_path,
            final_model_tuning={"enabled": True, "n_trials": 2, "cv_splits": 2,
                                "single_fold": True, "scorer": "qini"},
            data_processing={"cross_validation_splits": 2},
        )
        fmt = FinalModelTuner(cfg)
        add = fmt.tune_final_model(
            model_name="NonParamDML", X=X, Y=Y, T=T,
            base_type="lgbm",
            base_fixed_params={"n_estimators": 15, "num_leaves": 7},
            tuned_roles={}, fmt_fixed_params={}, groups=groups,
        )
        assert "model_final" in add and add["model_final"], add

    def test_cft_causalforestdml_with_groups(self, tmp_path):
        from rubin.tuning.causal_forest import CausalForestTuner
        X, T, Y, groups = _toy(250, 90, 9)
        cfg = _tuning_cfg(
            tmp_path,
            causal_forest={"tune_enabled": True, "scorer": "qini"},
            data_processing={"cross_validation_splits": 2, "dml_crossfit_folds": 2},
        )
        cft = CausalForestTuner(cfg)
        result = cft.tune(
            model_type="CausalForestDML", X=X, Y=Y, T=T, n_trials=1,
            nuisance_params_y={"n_estimators": 15, "num_leaves": 7},
            nuisance_params_t={"n_estimators": 15, "num_leaves": 7},
            single_fold=True, groups=groups,
        )
        assert result.get("best_params"), result

    def test_pipeline_callsites_pass_groups(self):
        """Signatur-Kontrakt: Alle vier Andockstellen akzeptieren 'groups' —
        schützt die run()-Aufrufe vor stillen Signatur-Drifts."""
        import inspect
        from rubin.tuning.base_learner import BaseLearnerTuner
        from rubin.tuning.final_model import FinalModelTuner
        from rubin.tuning.causal_forest import CausalForestTuner
        from rubin.training import train_and_crosspredict_bt_bo
        for fn in (BaseLearnerTuner.tune_all, FinalModelTuner.tune_final_model,
                   CausalForestTuner.tune, train_and_crosspredict_bt_bo):
            assert "groups" in inspect.signature(fn).parameters, fn


# ───────────── Szenarien: TMES, Balance, External Eval, DRTester, Report ─────────────

class TestGroupScenarios:
    def _leads_files(self, tmp, n_partner=90, files=2, overlap=True):
        """files CSVs mit je 2 Leads pro Partner; bei overlap teilen sich die
        Dateien die Partner-IDs (TMES-typisch: Eval-Welle enthält bekannte Partner)."""
        rng = np.random.default_rng(3)
        paths = []
        for j in range(files):
            rows = []
            for pid in range(n_partner):
                gid = pid if overlap else pid + j * n_partner
                for lead in range(2):
                    rows.append({"PARTNER_ID": f"P{gid:04d}", "ALTER": 25 + gid % 40,
                                 "LEAD_ART": lead, "T": gid % 2,
                                 "Y": int(rng.random() < 0.3)})
            f = tmp / f"welle{j}.csv"
            pd.DataFrame(rows).to_csv(f, index=False)
            paths.append(f)
        return paths

    def test_tmes_groups_and_eval_mask_aligned(self, tmp_path):
        """TMES + group_column: eval_mask.npy und groups.parquet decken denselben
        finalen Frame ab; die Eval-Datei-Zeilen tragen dieselben Gruppen-IDs wie
        ihre Partner-Zwillinge in den Train-Dateien."""
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        files = self._leads_files(tmp_path)
        p, outd = _dataprep_cfg(tmp_path, files, {"group_column": "PARTNER_ID",
                                                  "features": ["ALTER", "LEAD_ART"],
                                                  "eval_file_index": 1})
        DataPrepPipeline.from_config_path(str(p)).run()
        X = pd.read_parquet(outd / "X.parquet")
        g = pd.read_parquet(outd / "groups.parquet")["group"].to_numpy()
        mask = np.load(outd / "eval_mask.npy").astype(bool)
        assert len(g) == len(mask) == len(X)
        # Überschneidende Partner: Eval-Zeilen und Train-Zeilen teilen Gruppen-IDs →
        # genau der Fall, den Gruppen-Folds bei der OOF-Bildung zusammenhalten.
        assert set(g[mask]) & set(g[~mask]), "Testaufbau: Überschneidung erwartet"
        # Analyse lädt beides konsistent
        pipe = _analysis_pipeline(outd)
        pipe.cfg.data_files.eval_mask_file = str(outd / "eval_mask.npy")
        X_l, T_l, Y_l, S_l, mask_l = pipe._load_inputs()
        assert pipe._groups is not None and len(pipe._groups) == len(X_l) == len(mask_l)

    def test_balance_keeps_groups_aligned(self, tmp_path):
        """Treatment-Balance verwirft Zeilen — groups.parquet muss die
        ÜBERLEBENDEN Zeilen beschreiben (zeilengleich zu X)."""
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        rng = np.random.default_rng(5)
        paths = []
        for j, rate in enumerate([0.2, 0.7]):  # stark unterschiedliche T-Raten
            rows = []
            for pid in range(150):
                for lead in range(2):
                    rows.append({"PARTNER_ID": f"W{j}_P{pid}", "ALTER": 30 + pid % 20,
                                 "LEAD_ART": lead, "T": int(rng.random() < rate),
                                 "Y": int(rng.random() < 0.3)})
            f = tmp_path / f"b{j}.csv"
            pd.DataFrame(rows).to_csv(f, index=False)
            paths.append(f)
        p, outd = _dataprep_cfg(tmp_path, paths, {"group_column": "PARTNER_ID",
                                                  "features": ["ALTER", "LEAD_ART"],
                                                  "balance_treatments": True})
        DataPrepPipeline.from_config_path(str(p)).run()
        X = pd.read_parquet(outd / "X.parquet")
        g = pd.read_parquet(outd / "groups.parquet")["group"]
        assert len(g) == len(X)
        assert len(X) < 600  # Balance hat wirklich Zeilen entfernt

    def test_external_eval_feature_alignment_without_group_column(self, tmp_path):
        """Explizite Feature-Liste ENTHÄLT die Gruppen-Spalte versehentlich:
        Sie muss aus X UND X_eval herausgehalten werden (identische Spalten)."""
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        tr, ev = self._leads_files(tmp_path, overlap=False)
        p, outd = _dataprep_cfg(tmp_path, [tr], {
            "group_column": "PARTNER_ID",
            "features": ["ALTER", "LEAD_ART", "PARTNER_ID"],
            "eval_data_path": [str(ev)],
        })
        DataPrepPipeline.from_config_path(str(p)).run()
        X = pd.read_parquet(outd / "X.parquet")
        X_eval = pd.read_parquet(outd / "X_eval.parquet")
        assert "PARTNER_ID" not in X.columns and "PARTNER_ID" not in X_eval.columns
        assert list(X.columns) == list(X_eval.columns)

    def test_external_eval_group_overlap_warns(self, tmp_path, caplog):
        """Dieselben Partner in Train- UND External-Eval-Daten: Gruppen-Folds
        schützen nur die interne CV — die Pipeline muss laut warnen."""
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        tr, ev = self._leads_files(tmp_path, overlap=True)
        p, outd = _dataprep_cfg(tmp_path, [tr], {
            "group_column": "PARTNER_ID",
            "features": ["ALTER", "LEAD_ART"],
            "eval_data_path": [str(ev)],
        })
        with caplog.at_level(logging.WARNING):
            DataPrepPipeline.from_config_path(str(p)).run()
        assert any("Leakage-Kanal" in r.message and "Gruppen" in r.message for r in caplog.records)

    def test_external_eval_disjoint_groups_no_warning(self, tmp_path, caplog):
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        tr, ev = self._leads_files(tmp_path, overlap=False)
        p, outd = _dataprep_cfg(tmp_path, [tr], {
            "group_column": "PARTNER_ID",
            "features": ["ALTER", "LEAD_ART"],
            "eval_data_path": [str(ev)],
        })
        with caplog.at_level(logging.WARNING):
            DataPrepPipeline.from_config_path(str(p)).run()
        assert not any("Leakage-Kanal" in r.message for r in caplog.records)


class TestDRTesterGroups:
    def test_drtester_nuisance_grouped(self, caplog):
        from rubin.evaluation.drtester_core import fit_drtester_nuisance
        from lightgbm import LGBMClassifier
        X, T, Y, groups = _toy(300, 100, 12)
        with caplog.at_level(logging.INFO, logger="rubin.evaluation"):
            tester = fit_drtester_nuisance(
                model_regression=LGBMClassifier(n_estimators=10, verbosity=-1),
                model_propensity=LGBMClassifier(n_estimators=10, verbosity=-1),
                X_val=X, T_val=T, Y_val=Y, cv=3, seed=1, groups=groups,
            )
        assert tester is not None
        assert any("Gruppen-CV für Nuisance-Cross-Fitting" in r.message for r in caplog.records)

    def test_drtester_without_groups_unchanged(self, caplog):
        from rubin.evaluation.drtester_core import fit_drtester_nuisance
        from lightgbm import LGBMClassifier
        X, T, Y, _ = _toy(200, 80, 13)
        with caplog.at_level(logging.INFO, logger="rubin.evaluation"):
            tester = fit_drtester_nuisance(
                model_regression=LGBMClassifier(n_estimators=10, verbosity=-1),
                model_propensity=LGBMClassifier(n_estimators=10, verbosity=-1),
                X_val=X, T_val=T, Y_val=Y, cv=3, seed=1,
            )
        assert tester is not None
        assert not any("Gruppen-CV" in r.message for r in caplog.records)


class TestReportShowsGroupCV:
    def test_dataprep_section_renders_group_tile_and_tag(self):
        """Der Report muss zeigen, WAS gemacht wurde: Kachel aus der Analyse
        (auch ohne DataPrep-Config) + Verarbeitungsschritt-Tag aus der
        dataprep_config.yml."""
        from rubin.reporting.html_report import ReportCollector, _render_dataprep
        c = ReportCollector()
        c.dataprep_info = {"data_files": ["leads.csv"], "target": "Y", "treatment": "T",
                           "group_column": "PARTNER_ID"}
        c.group_cv_info = {"n_rows": 800, "n_groups": 400, "max_size": 2,
                           "mean_size": 2.0, "source": "groups.parquet", "auto": True}
        html = _render_dataprep(c, {})
        assert "Gruppen-CV aktiv" in html
        assert "400" in html and "groups.parquet" in html and "auto-erkannt" in html
        assert "DRTester" in html  # die Kachel benennt die gesamte Kette
        assert "Gruppen-Folds statt Deduplizierung (PARTNER_ID)" in html

    def test_dataprep_section_tile_without_dataprep_config(self):
        """Eigene Pfade ohne dataprep_config.yml: Kachel erscheint trotzdem."""
        from rubin.reporting.html_report import ReportCollector, _render_dataprep
        c = ReportCollector()
        c.group_cv_info = {"n_rows": 100, "n_groups": 50, "max_size": 3,
                           "mean_size": 2.0, "source": "partner.parquet", "auto": False}
        html = _render_dataprep(c, {})
        assert "Gruppen-CV aktiv" in html and "explizit konfiguriert" in html

    def test_no_group_cv_no_tile(self):
        from rubin.reporting.html_report import ReportCollector, _render_dataprep
        c = ReportCollector()
        c.dataprep_info = {"data_files": ["a.csv"], "target": "Y", "treatment": "T"}
        html = _render_dataprep(c, {})
        assert "Gruppen-CV aktiv" not in html
        assert "Gruppen-Folds statt Deduplizierung" not in html

    def test_add_dataprep_info_reads_group_column(self, tmp_path):
        import yaml as _yaml
        from rubin.reporting.html_report import ReportCollector
        (tmp_path / "dataprep_config.yml").write_text(_yaml.safe_dump(
            {"data_prep": {"data_path": ["l.csv"], "target": "Y", "treatment": "T",
                           "group_column": "PARTNER_ID"}}), encoding="utf-8")
        c = ReportCollector()
        c.add_dataprep_info(tmp_path)
        assert c.dataprep_info.get("group_column") == "PARTNER_ID"


class TestGroupHardening:
    """Zweite Tiefenprüfung: stale Artefakte, DRLearner-FMT, CFT-rscore,
    modus-bewusste Report-Kachel."""

    def test_stale_groups_parquet_removed_on_rerun_without_group_column(self, tmp_path):
        """Lauf 1 mit group_column, Lauf 2 ohne: Die alte groups.parquet darf
        nicht überleben — sonst aktiviert ein späterer Analyse-Lauf stillschweigend
        Gruppen-CV mit veralteter (womöglich längen-passender!) Zuordnung."""
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        f = tmp_path / "l.csv"
        pd.DataFrame({"PARTNER_ID": [f"P{i//2}" for i in range(200)],
                      "ALTER": np.arange(200) % 40, "T": np.arange(200) % 2,
                      "Y": (np.arange(200) // 2) % 2}).to_csv(f, index=False)
        p1, outd = _dataprep_cfg(tmp_path, [f], {"group_column": "PARTNER_ID", "features": ["ALTER"]})
        DataPrepPipeline.from_config_path(str(p1)).run()
        assert (outd / "groups.parquet").exists()
        p2, _ = _dataprep_cfg(tmp_path, [f], {"features": ["ALTER"]})
        DataPrepPipeline.from_config_path(str(p2)).run()
        assert not (outd / "groups.parquet").exists()

    def test_fmt_drlearner_with_groups(self, tmp_path):
        from rubin.tuning.final_model import FinalModelTuner
        X, T, Y, groups = _toy(250, 90, 14)
        cfg = _tuning_cfg(
            tmp_path,
            final_model_tuning={"enabled": True, "n_trials": 2, "cv_splits": 2,
                                "single_fold": True, "scorer": "qini"},
            data_processing={"cross_validation_splits": 2},
        )
        add = FinalModelTuner(cfg).tune_final_model(
            model_name="DRLearner", X=X, Y=Y, T=T, base_type="lgbm",
            base_fixed_params={"n_estimators": 15, "num_leaves": 7},
            tuned_roles={}, fmt_fixed_params={}, groups=groups,
        )
        assert "model_final" in add and add["model_final"], add

    def test_cft_rscore_path_with_groups(self, tmp_path):
        """rscore-Pfad: RScorer-Nuisances werden mit Gruppen-Splits gefittet."""
        from rubin.tuning.causal_forest import CausalForestTuner
        X, T, Y, groups = _toy(250, 90, 15)
        cfg = _tuning_cfg(
            tmp_path,
            causal_forest={"tune_enabled": True, "scorer": "rscore"},
            data_processing={"cross_validation_splits": 2, "dml_crossfit_folds": 2},
        )
        r = CausalForestTuner(cfg).tune(
            model_type="CausalForestDML", X=X, Y=Y, T=T, n_trials=1,
            nuisance_params_y={"n_estimators": 15, "num_leaves": 7},
            nuisance_params_t={"n_estimators": 15, "num_leaves": 7},
            single_fold=True, groups=groups,
        )
        assert r.get("best_params"), r

    def test_report_tile_mode_external(self):
        """External: kein OOF-/DRTester-Claim — stattdessen Hinweis auf den
        nicht schließbaren Train↔Eval-Kanal."""
        from rubin.reporting.html_report import ReportCollector, _render_dataprep
        c = ReportCollector()
        c.group_cv_info = {"n_rows": 600, "n_groups": 300, "max_size": 2,
                           "mean_size": 2.0, "source": "groups.parquet",
                           "auto": True, "mode": "external"}
        html = _render_dataprep(c, {})
        assert "per Fold nicht schließbar" in html
        assert "Gilt durchgängig: Cross-Prediction" not in html

    def test_report_tile_mode_tmes(self):
        from rubin.reporting.html_report import ReportCollector, _render_dataprep
        c = ReportCollector()
        c.group_cv_info = {"n_rows": 600, "n_groups": 300, "max_size": 2,
                           "mean_size": 2.0, "source": "groups.parquet",
                           "auto": True, "mode": "tmes"}
        html = _render_dataprep(c, {})
        assert "bleibt gruppenrein" in html
        assert "Gilt durchgängig: Cross-Prediction" in html

    def test_estimator_with_group_cv_pickles_and_refits(self):
        """Bundle-Export pickelt gefittete Estimatoren: GroupAware-cv muss
        mitwandern, Predictions identisch, Refit mit UND ohne groups möglich."""
        import pickle
        from econml.dml import LinearDML
        from lightgbm import LGBMClassifier
        from rubin.training import fit_estimator
        X, T, Y, groups = _toy(200, 70, 16)
        est = LinearDML(model_y=LGBMClassifier(n_estimators=8, verbosity=-1),
                        model_t=LGBMClassifier(n_estimators=8, verbosity=-1),
                        discrete_treatment=True, discrete_outcome=True,
                        cv=GroupAwareStratifiedKFold(n_splits=2, shuffle=True, random_state=1),
                        random_state=1)
        fit_estimator(est, Y, T, X, groups=groups)
        est2 = pickle.loads(pickle.dumps(est))
        assert np.allclose(np.asarray(est.effect(X)), np.asarray(est2.effect(X)))
        fit_estimator(est2, Y, T, X, groups=groups)
        est2.fit(Y, T, X=X)  # Refit ohne groups (Scoring-Umgebung)


class TestThirdPassGuarantees:
    """Dritte Tiefenprüfung: numerische Bit-Identität ohne Gruppen über den
    kompletten Estimator (nicht nur Splits), seltene Strata, Dominanz-Warnung."""

    def test_estimator_results_bit_identical_without_groups(self):
        """Der Splitter-Tausch in der Registry darf Läufe OHNE Gruppen nicht
        um ein Bit verändern — voller LinearDML-Vergleich alt vs. neu."""
        from sklearn.model_selection import StratifiedKFold
        from econml.dml import LinearDML
        from lightgbm import LGBMClassifier
        rng = np.random.default_rng(9)
        n = 400
        X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("ABCD"))
        T = rng.integers(0, 2, n)
        Y = ((0.2 * T + 0.3 * X["A"] + rng.normal(size=n)) > 0.4).astype(int)

        def eff(cv):
            est = LinearDML(
                model_y=LGBMClassifier(n_estimators=25, random_state=1, verbosity=-1),
                model_t=LGBMClassifier(n_estimators=25, random_state=1, verbosity=-1),
                discrete_treatment=True, discrete_outcome=True, cv=cv, random_state=7)
            est.fit(Y, T, X=X)
            return np.asarray(est.effect(X))

        alt = eff(StratifiedKFold(n_splits=3, shuffle=True, random_state=7))
        neu = eff(GroupAwareStratifiedKFold(n_splits=3, shuffle=True, random_state=7))
        assert np.array_equal(alt, neu)

    def test_rare_strata_do_not_crash_grouped_splits(self):
        """Winzige T×Y-Zelle (1 Mitglied): SGKF warnt nur (wie SKF) — die
        gruppierten Zweige dürfen nicht härter sein als die alten."""
        import warnings as _w
        from rubin.tuning.common import _iter_stratified_or_kfold
        rng = np.random.default_rng(0)
        n = 200
        strata = np.zeros(n, dtype=int); strata[0] = 11
        groups = rng.integers(0, 80, n).astype(str)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            folds = list(_iter_stratified_or_kfold(strata, 5, 42, groups=groups))
        assert len(folds) == 5
        _assert_pure(folds, groups)
        all_va = np.concatenate([va for _, va in folds])
        assert sorted(all_va.tolist()) == list(range(n))

    def test_dominant_group_warns(self, tmp_path, caplog):
        """Fremde groups-Datei mit NaN → Mega-Gruppe 'nan': laute Warnung."""
        rng = np.random.default_rng(2)
        n = 300
        X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("ABC"))
        X.to_parquet(tmp_path / "X.parquet")
        pd.DataFrame({"T": rng.integers(0, 2, n)}).to_parquet(tmp_path / "T.parquet")
        pd.DataFrame({"Y": rng.integers(0, 2, n)}).to_parquet(tmp_path / "Y.parquet")
        g = pd.Series([None] * 120 + [f"P{i}" for i in range(180)], dtype=object)
        pd.DataFrame({"group": g}).to_parquet(tmp_path / "groups.parquet")
        pipe = _analysis_pipeline(tmp_path)
        with caplog.at_level(logging.WARNING):
            pipe._load_inputs()
        assert pipe._groups is not None
        assert any("größte Gruppe" in r.message and "unbalancierte Folds" in r.message
                   for r in caplog.records)

    def test_small_groups_no_dominance_warning(self, tmp_path, caplog):
        _write_xty(tmp_path, n=300, n_partner=150)
        pipe = _analysis_pipeline(tmp_path)
        with caplog.at_level(logging.WARNING):
            pipe._load_inputs()
        assert not any("größte Gruppe" in r.message for r in caplog.records)


class TestFourthPassRuntimePaths:
    """Vierte Tiefenprüfung: bisher nie AUSGEFÜHRTE Laufzeitpfade."""

    def test_parallel_fold_path_with_groups(self, caplog):
        """Der parallele Fold-Pfad (joblib-Threads) lief bisher in keinem Test
        mit Gruppen — Fold-Reinheit, vollständige Predictions und das
        Last-Fold-Modell (Explainability-Refit mit groups) hier erzwingen."""
        from rubin.training import train_and_crosspredict_bt_bo, get_fold_aligned_data
        from econml.metalearners import SLearner
        from lightgbm import LGBMRegressor
        X, T, Y, groups = _toy(400, 140, 17)
        with caplog.at_level(logging.INFO, logger="rubin.training"):
            out, last_model, last_va = train_and_crosspredict_bt_bo(
                model=SLearner(overall_model=LGBMRegressor(n_estimators=15, verbosity=-1)),
                X=X, Y=Y, T=T, n_splits=3, model_name="S", random_state=3,
                parallel_level=4, max_parallel_folds=3,
                keep_last_fold_model=True, groups=groups,
            )
        assert any("Folds parallel" in r.message for r in caplog.records), \
            "Testaufbau: paralleler Pfad wurde nicht genommen"
        _, fold_splits = get_fold_aligned_data(out)
        _assert_pure(fold_splits, groups)
        assert out["Predictions_S"].notna().all()
        assert last_model is not None and last_va is not None
        # Last-Fold-Modell wurde ohne seinen Val-Fold trainiert → Val-Gruppen
        # dürfen im Trainingsteil des letzten Folds nicht vorkommen.
        tr_last = fold_splits[-1][0]
        assert not (set(groups[tr_last]) & set(groups[last_va]))

    def test_bundle_export_and_scoring_with_groups(self, tmp_path):
        """Kompletter Zyklus: Bundle-Export (Full-Data-Refit via fit_estimator
        mit groups, GroupAware-cv im Estimator) → ProductionPipeline lädt und
        scort neue Daten. Deckt Pickle + Schema + Produktions-Predict."""
        import logging as _logging
        import uuid
        from types import SimpleNamespace
        from econml.dml import LinearDML
        from lightgbm import LGBMClassifier
        from rubin.pipelines.analysis_pipeline import AnalysisPipeline
        from rubin.pipelines.production_pipeline import ProductionPipeline

        rng = np.random.default_rng(18)
        n_partner = 150
        Xb = pd.DataFrame(rng.normal(size=(n_partner, 3)), columns=["f0", "f1", "f2"])
        X = pd.concat([Xb, Xb + rng.normal(0, 0.05, Xb.shape)], ignore_index=True)
        groups = np.tile([f"P{i}" for i in range(n_partner)], 2)
        T = np.tile(rng.integers(0, 2, n_partner), 2)
        Y = np.tile((rng.random(n_partner) < 0.4).astype(int), 2)

        model = LinearDML(
            model_y=LGBMClassifier(n_estimators=10, verbosity=-1),
            model_t=LGBMClassifier(n_estimators=10, verbosity=-1),
            discrete_treatment=True, discrete_outcome=True,
            cv=GroupAwareStratifiedKFold(n_splits=2, shuffle=True, random_state=1),
            random_state=1,
        )
        models = {"ModelA": model}
        eval_summary = {"ModelA": {"qini": 0.05}}

        base_dir = tmp_path / "bundles"; base_dir.mkdir()
        cfg_file = tmp_path / "config_snapshot.yml"
        cfg_file.write_text("# test\n", encoding="utf-8")
        cfg = SimpleNamespace(
            bundle=SimpleNamespace(enabled=True, base_dir=str(base_dir), bundle_id=None, log_to_mlflow=False),
            source_config_path=str(cfg_file),
            data_files=SimpleNamespace(x_file=str(tmp_path / "X.parquet")),
            selection=SimpleNamespace(metric="qini", higher_is_better=True, manual_champion=None),
            surrogate_tree=SimpleNamespace(enabled=False),
            treatment=SimpleNamespace(type="binary", reference_group=0),
            base_learner=SimpleNamespace(type="lgbm"),
        )
        pipe = AnalysisPipeline.__new__(AnalysisPipeline)
        pipe._logger = _logging.getLogger("test_bundle_groups")
        pipe._groups = groups  # → _refit_one fittet mit fit_estimator(groups=...)

        from unittest.mock import MagicMock
        bundle_id = f"grp_{uuid.uuid4().hex[:8]}"
        pipe._run_bundle_export(
            cfg, models, eval_summary,
            X, T, Y, X, T, Y, list(X.columns),
            None, True, str(base_dir), bundle_id, MagicMock(),
        )
        bundle_path = base_dir / bundle_id
        assert bundle_path.is_dir()

        prod = ProductionPipeline(str(bundle_path))
        X_new = pd.DataFrame(rng.normal(size=(50, 3)), columns=["f0", "f1", "f2"])
        out = prod.score(X_new, model_names=["ModelA"])
        cate = out.cate
        assert "cate_ModelA" in cate.columns
        vals = np.asarray(cate["cate_ModelA"].to_numpy(), dtype=float)
        assert len(vals) == len(X_new) and np.all(np.isfinite(vals))


class TestFifthPassDelivery:
    """Fünfte Tiefenprüfung: Lieferartefakt-Perspektive."""

    def test_dataprep_logs_groups_parquet_to_mlflow(self, tmp_path):
        """groups.parquet muss — wie eval_mask/file_source/row_provenance —
        als mlflow-Artefakt am DataPrep-Run hängen (echter sqlite-Store)."""
        import os
        import mlflow
        from rubin.pipelines.data_prep_pipeline import DataPrepPipeline
        uri = f"sqlite:///{tmp_path / 'ml.db'}"
        os.environ["MLFLOW_TRACKING_URI"] = uri
        mlflow.set_tracking_uri(uri)
        rng = np.random.default_rng(3)
        pd.DataFrame({"PARTNER_ID": [f"P{i//2}" for i in range(120)],
                      "ALTER": rng.integers(18, 80, 120),
                      "T": rng.integers(0, 2, 120),
                      "Y": rng.integers(0, 2, 120)}).to_csv(tmp_path / "l.csv", index=False)
        p, outd = _dataprep_cfg(tmp_path, [tmp_path / "l.csv"],
                                {"group_column": "PARTNER_ID", "features": ["ALTER"],
                                 "log_to_mlflow": True,
                                 "mlflow_experiment_name": "groups_log_test"})
        # _dataprep_cfg setzt log_to_mlflow=False als Default — Override oben zieht,
        # weil extra_dp NACH den Defaults gemerged wird.
        DataPrepPipeline.from_config_path(str(p)).run()
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("groups_log_test")
        assert exp is not None
        run = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"])[0]
        arts = {a.path for a in client.list_artifacts(run.info.run_id)}
        assert "groups.parquet" in arts, arts
