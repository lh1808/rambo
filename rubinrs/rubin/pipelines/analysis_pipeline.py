from __future__ import annotations

"""Orchestrierung der Analyse-Pipeline.
Die Analyse-Pipeline ist der zentrale Einstiegspunkt für Entwicklungs- und
Evaluationsläufe.
Aufgaben:
- Einlesen der vorbereiteten Input-Dateien (X/T/Y, optional S)
- optionale Feature-Filterung (Korrelation / Importance)
- optionales Base-Learner-Tuning mit Optuna
- Training der konfigurierten kausalen Learner
- Cross-Predictions (standardmäßig) für robuste Evaluation
- Uplift-Metriken (Qini, AUUC, Uplift@k, Policy Value)
- Logging nach MLflow
- optional: synchroner Bundle-Export (Production-Artefakte)
Wichtig:
Die Production-Pipeline arbeitet ausschließlich auf Bundles.
Der Bundle-Export ist deshalb bewusst am Ende des Analyselaufs angesiedelt."""

from dataclasses import dataclass
import copy
import gc
import tempfile
import time
from typing import Any, Dict, Optional
import json
import logging
import os

import numpy as np
import pandas as pd

from rubin.artifacts import ArtifactBundler
from rubin.feature_selection import (
    compute_importances,
    select_features_by_importance,
    remove_highly_correlated_features,
)
from rubin.model_management import ModelEntry, choose_champion, write_registry, float_metrics
from rubin.model_registry import ModelContext, ModelRegistry, default_registry
from rubin.preprocessing import build_simple_preprocessor_from_dataframe
from rubin.settings import AnalysisConfig
from rubin.training import _predict_effect, train_and_crosspredict_bt_bo, is_multi_treatment, SurrogateTreeWrapper, SURROGATE_MODEL_NAME
from rubin.utils.data_utils import reduce_mem_usage, available_cpu_count
from rubin.utils.io_utils import read_table
from rubin.utils.categorical_patch import patch_categorical_features
from rubin.tuning_optuna import BaseLearnerTuner, FinalModelTuner, build_base_learner
from rubin.evaluation.uplift_metrics import auuc, policy_value, qini_coefficient, uplift_at_k, uplift_curve, mt_eval_summary
from rubin.evaluation.drtester_plots import (
    CustomDRTester,
    evaluate_cate_with_plots,
    filter_tester_for_mask,
    fit_drtester_nuisance,
    generate_cate_distribution_plot,
    generate_sklift_plots,
    plot_custom_qini_curve,
    save_dataframe_as_png,
    policy_value_comparison_plots,
)
from rubin.reporting import ReportCollector, generate_html_report


@dataclass
class AnalysisResult:
    """Rückgabeobjekt der Analyse-Pipeline."""

    models: Dict[str, Any]
    predictions: Dict[str, pd.DataFrame]
    removed_features: Dict[str, Any]
    eval_summary: Dict[str, Dict[str, float]]


def _log_temp_artifact(mlflow, content_fn, filename: str) -> None:
    """Schreibt ein temporäres Artefakt, loggt es nach MLflow und räumt auf.

    Vermeidet Dateikonflikte bei parallelen Runs, indem ein temporäres
    Verzeichnis verwendet wird.

    Parameters
    ----------
    mlflow:
        Das MLflow-Modul.
    content_fn:
        Callable, das den vollständigen Dateipfad als Argument erhält und die
        Datei schreibt.
    filename:
        Gewünschter Dateiname im MLflow-Artefakt-Store.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        content_fn(path)
        mlflow.log_artifact(path)


def _log_figure_fast(mlflow, fig, filename: str) -> None:
    """Rendert eine Matplotlib-Figure EINMAL und loggt sie nach MLflow.

    Cached die PNG-Bytes auf dem Figure-Objekt (fig._rubin_png_cache),
    damit fig_to_base64() im HTML-Report die Bytes wiederverwenden kann
    statt erneut savefig() aufzurufen. Bei großen Datensätzen spart das
    mehrere Sekunden pro Plot.

    Ersetzt mlflow.log_figure(), das intern nochmal savefig() aufruft."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    png_bytes = buf.read()
    buf.close()
    # Cache für fig_to_base64
    fig._rubin_png_cache = png_bytes
    # MLflow: Bytes direkt als Artefakt loggen
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        with open(path, "wb") as f:
            f.write(png_bytes)
        mlflow.log_artifact(path)


class AnalysisPipeline:
    """Führt einen kompletten Analyselauf gemäß Konfiguration aus."""

    _logger = logging.getLogger("rubin.analysis")

    def __init__(self, cfg: AnalysisConfig, registry: Optional[ModelRegistry] = None) -> None:
        self.cfg = cfg
        self.registry = registry or default_registry()

    def _read_table(self, path: str, use_index: bool = True) -> pd.DataFrame:
        """Liest eine Tabelle aus CSV oder Parquet."""
        return read_table(path, use_index=use_index)

    def _load_inputs(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Lädt X, T, Y, optional S und optional eval_mask."""
        X = self._read_table(self.cfg.data_files.x_file)
        T_df = self._read_table(self.cfg.data_files.t_file)
        Y_df = self._read_table(self.cfg.data_files.y_file)
        T = T_df["T"].to_numpy()
        Y = Y_df["Y"].to_numpy()

        S: Optional[np.ndarray] = None
        S_df: Optional[pd.DataFrame] = None
        if self.cfg.data_files.s_file:
            try:
                col = self.cfg.historical_score.column
                S_df = self._read_table(self.cfg.data_files.s_file)
                if col not in S_df.columns:
                    self._logger.warning(
                        "Historischer Score: Spalte '%s' nicht in s_file gefunden. "
                        "Verfügbare Spalten: %s. Score wird ignoriert.",
                        col, list(S_df.columns)[:10],
                    )
                    S_df = None
                else:
                    S = S_df[col].to_numpy(dtype=float)
                    n_nan = int(np.isnan(S).sum())
                    if n_nan > 0:
                        self._logger.info("Historischer Score: %d NaN-Werte durch 0 ersetzt.", n_nan)
                        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
            except FileNotFoundError:
                S = None

        # ── Eval-Maske laden (Train Many, Evaluate One) ──
        # Wird VOR df_frac geladen, damit sie mit-subsampled wird.
        eval_mask: Optional[np.ndarray] = None
        if self.cfg.data_files.eval_mask_file:
            try:
                _raw_mask = np.load(self.cfg.data_files.eval_mask_file).astype(bool)
                if len(_raw_mask) == len(X):
                    eval_mask = _raw_mask
                else:
                    self._logger.warning(
                        "eval_mask Länge (%d) passt nicht zu X (%d) — Maske wird ignoriert.",
                        len(_raw_mask), len(X),
                    )
            except Exception:
                self._logger.warning("eval_mask konnte nicht geladen werden.", exc_info=True)

        if self.cfg.data_processing.df_frac:
            X = X.sample(frac=float(self.cfg.data_processing.df_frac), random_state=self.cfg.constants.random_seed)
            idx = X.index
            T = T_df["T"].loc[idx].to_numpy()
            Y = Y_df["Y"].loc[idx].to_numpy()
            if S is not None and S_df is not None:
                try:
                    col = self.cfg.historical_score.column
                    S = S_df[col].loc[idx].to_numpy(dtype=float)
                    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    S = None
            # ── Daten-Summary ──
            _t_arr = np.asarray(T)
            _y_arr = np.asarray(Y)
            _n_treat = int((_t_arr > 0).sum())
            _n_ctrl = int((_t_arr == 0).sum())
            _y_pos_pct = float((_y_arr > 0).mean()) * 100
            self._logger.info(
                "Daten: %s Zeilen, %d Features | T: %d Control (%.1f%%), %d Treatment (%.1f%%) | Y: %.1f%% positiv",
                f"{len(X):,}", X.shape[1],
                _n_ctrl, _n_ctrl / len(X) * 100, _n_treat, _n_treat / len(X) * 100,
                _y_pos_pct,
            )
            if S is not None:
                self._logger.info("Historischer Score (S): vorhanden (%d Werte)", len(S))

            if eval_mask is not None:
                # eval_mask ist ein numpy-Array → braucht positionelle Indizes.
                # X.index nach sample() enthält Label-Indizes, die bei RangeIndex
                # identisch mit Positionen sind. Für Sicherheit bei nicht-0-basiertem
                # Index: iloc-Positionen über get_indexer berechnen.
                try:
                    positions = T_df.index.get_indexer(idx)
                    eval_mask = eval_mask[positions]
                except Exception:
                    eval_mask = eval_mask[idx.to_numpy()]

        if self.cfg.data_files.dtypes_file:
            try:
                with open(self.cfg.data_files.dtypes_file, "r", encoding="utf-8") as f:
                    dtypes = json.load(f)
                for c, dt in dtypes.items():
                    if c in X.columns:
                        try:
                            X[c] = X[c].astype(dt)
                        except Exception:
                            pass
            except FileNotFoundError:
                pass

        # Memory-Reduktion: Datentypen downcasten (float64→float32, int64→int32 etc.)
        if getattr(self.cfg.data_processing, "reduce_memory", True):
            n_before = X.memory_usage(deep=True).sum()
            X = reduce_mem_usage(X)
            n_after = X.memory_usage(deep=True).sum()
            self._logger.info(
                "Memory-Reduktion: %.1f MB → %.1f MB (%.0f%% gespart).",
                n_before / 1e6, n_after / 1e6, (1 - n_after / max(n_before, 1)) * 100,
            )

        # Kategorische Spalten explizit als category-Dtype markieren.
        # Drei Quellen (in Prioritätsreihenfolge):
        #   1. cfg.data_processing.categorical_columns (explizit vom User)
        #   2. dtypes.json (von DataPrep → enthält "category" wenn korrekt erzeugt)
        #   3. Parquet-Metadaten (preserven category-Dtype automatisch)
        # Quelle 2+3 werden bereits oben bei dtypes-Anwendung / Parquet-Load abgedeckt.
        # Hier ergänzen wir Quelle 1 als Override: Falls der User explizite
        # categorical_columns angegeben hat, werden diese zu category konvertiert.
        cat_cols = getattr(self.cfg.data_processing, "categorical_columns", None)
        if cat_cols:
            n_cat = 0
            for c in cat_cols:
                if c in X.columns and not isinstance(X[c].dtype, pd.CategoricalDtype):
                    try:
                        X[c] = X[c].astype("category")
                        n_cat += 1
                    except Exception:
                        pass
            if n_cat > 0:
                self._logger.info("Kategorische Spalten aus Config angewendet: %d Spalten als category markiert.", n_cat)

        self._logger.info(
            "Daten geladen: X=%s, T=%s (unique=%s), Y=%s (unique=%s), S=%s",
            X.shape, T.shape, np.unique(T).tolist(), Y.shape, np.unique(Y).tolist(),
            S.shape if S is not None else "None",
        )

        # Defensiv: Index zurücksetzen, damit X.iloc[i] ↔ T[i] ↔ Y[i] ↔ S[i]
        # garantiert position-konsistent sind (nach sample wäre X.index nicht-konsekutiv).
        X = X.reset_index(drop=True)

        # ── Alignment-Assertion: X/T/Y/S müssen gleich lang sein ──
        n = len(X)
        assert len(T) == n, f"T-Länge ({len(T)}) ≠ X-Länge ({n}). Prüfe ob x_file und t_file zusammenpassen."
        assert len(Y) == n, f"Y-Länge ({len(Y)}) ≠ X-Länge ({n}). Prüfe ob x_file und y_file zusammenpassen."
        if S is not None:
            if len(S) != n:
                self._logger.warning(
                    "Historischer Score: S-Länge (%d) ≠ X-Länge (%d). "
                    "Vermutlich stammt s_file aus einem anderen DataPrep-Lauf "
                    "(z.B. vor balance_treatments). Score wird ignoriert.",
                    len(S), n,
                )
                S = None
        if eval_mask is not None:
            if len(eval_mask) != n:
                self._logger.warning(
                    "eval_mask-Länge (%d) ≠ X-Länge (%d) nach Subsampling — Maske wird ignoriert.",
                    len(eval_mask), n,
                )
                eval_mask = None

        if eval_mask is not None:
            self._logger.info(
                "Eval-Maske: %d von %d Zeilen für Evaluation (Train Many, Evaluate One).",
                int(eval_mask.sum()), len(eval_mask),
            )

        return X, T, Y, S, eval_mask

    # ------------------------------------------------------------------
    # Submethoden für die einzelnen Pipeline-Schritte
    # ------------------------------------------------------------------

    def _run_feature_selection(self, cfg, X, T, Y, mlflow):
        """Feature-Selektion: Korrelationsfilter (importance-gesteuert) → Importance-Threshold.

        Reihenfolge:
        1. Importances auf ALLEN Features berechnen
        2. Korrelationsfilter: bei korrelierten Paaren das weniger wichtige entfernen
        3. Importance-Threshold (Top-X%): auf den verbleibenden Features
        
        Vorteile gegenüber der umgekehrten Reihenfolge:
        - Korrelierte Features splitten die Importance → beide könnten unter den Threshold fallen
        - Durch vorherige Korrelationsentfernung konzentriert sich die Importance auf ein Feature
        - Der Korrelationsfilter nutzt die Importances, um das wichtigere Feature zu behalten
        """
        removed: Dict[str, Any] = {}
        fs = cfg.feature_selection

        # 1. Importances auf ALLEN Features berechnen
        importances = {}
        effective_methods = [m for m in fs.methods if m != "none"]
        if fs.enabled and effective_methods:
            importances = compute_importances(
                methods=effective_methods, X=X, T=T, Y=Y,
                seed=cfg.constants.random_seed,
                n_jobs=1 if cfg.constants.parallel_level <= 1 else -1,
                parallel_methods=cfg.constants.parallel_level >= 3,
            )

            # Importances als Artefakte speichern (vor Filterung, auf allen Features)
            if importances:
                mlflow.log_param("feature_selection_methods", ",".join(effective_methods))
                mlflow.log_param("feature_selection_top_pct", fs.top_pct)
                for method_name, imp in importances.items():
                    def _write_imp(p, _imp=imp, _method=method_name):
                        df = _imp.rename_axis("feature").reset_index()
                        df.columns = ["feature", _method]
                        df.to_csv(p, index=False)
                    _log_temp_artifact(mlflow, _write_imp, f"feature_importance_{method_name}.csv")

        # 2. Korrelationsfilter (importance-gesteuert)
        absorbed_by: Dict[str, str] = {}
        if fs.enabled and float(fs.correlation_threshold) > 0:
            X, removed_corr, absorbed_by = remove_highly_correlated_features(
                X, correlation_threshold=fs.correlation_threshold,
                correlation_methods=["pearson", "spearman"],
                importances=importances if importances else None,
            )
            removed["high_correlation"] = removed_corr
            mlflow.log_metric("features_removed_correlation", len(removed_corr))
            mlflow.log_param("feature_selection_corr_threshold", fs.correlation_threshold)

            def _write_corr(p):
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(removed_corr, fh, ensure_ascii=False, indent=2)

            _log_temp_artifact(mlflow, _write_corr, "removed_features_corr.json")

            # Importance-Umverteilung: entferntes Feature → überlebendem Partner dazuaddieren.
            # Dadurch wird das Importance-Splitting bei korrelierten Features korrigiert.
            if absorbed_by and importances:
                for method_name, imp in importances.items():
                    for dropped, keeper in absorbed_by.items():
                        if dropped in imp.index and keeper in imp.index:
                            imp[keeper] = imp[keeper] + imp[dropped]
                n_redistributed = len([d for d in absorbed_by if any(d in imp.index for imp in importances.values())])
                if n_redistributed:
                    self._logger.info(
                        "Importance-Umverteilung: %d entfernte Features → Importance auf Partner übertragen.",
                        n_redistributed,
                    )
        else:
            removed["high_correlation"] = []

        # 3. Importance-Threshold (Top-X%) auf verbleibenden Features
        if fs.enabled and effective_methods and importances:
            X, removed_imp, top_per_method = select_features_by_importance(
                X, importances, top_pct=fs.top_pct,
                max_features=fs.max_features,
            )
            removed["importance"] = removed_imp

            mlflow.log_metric("features_after_importance", len(X.columns))
            mlflow.log_metric("features_removed_importance", len(removed_imp))
            for method_name, top_features in top_per_method.items():
                mlflow.log_metric(f"features_top_{method_name}", len(top_features))

            def _write_removed(p):
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(removed_imp, fh, ensure_ascii=False, indent=2)
            _log_temp_artifact(mlflow, _write_removed, "removed_features_imp.json")

        # Summary logging
        n_imp = len(removed.get("importance", []))
        n_corr = len(removed.get("high_correlation", []))
        if n_imp or n_corr:
            self._logger.info(
                "Feature-Selektion gesamt: %d → %d Features (Korrelation: −%d, Importance: −%d)",
                X.shape[1] + n_imp + n_corr, X.shape[1], n_corr, n_imp,
            )

        return X, removed

    def _run_tuning(self, cfg, X, T, Y, mlflow, _progress_cb=None):
        """Base-Learner-Tuning (Optuna) und optionales Final-Model-Tuning."""
        tuned_params_by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}

        if cfg.tuning.enabled:
            tuner = BaseLearnerTuner(cfg)
            self._logger.info(
                "Starte Tuning: X=%s, Y=%s (unique=%s), T=%s (unique=%s)",
                X.shape, Y.shape, np.unique(Y).tolist(), T.shape, np.unique(T).tolist(),
            )
            tuned_params_by_model = tuner.tune_all(cfg.models.models_to_train, X=X, Y=Y, T=T)

            # Logger: Zusammenfassung
            n_tasks = len(tuner.best_scores)
            self._logger.info(
                "Base-Learner-Tuning: %d Tasks abgeschlossen.",
                n_tasks,
            )
            for _tk, _tv in sorted(tuner.best_scores.items()):
                self._logger.info("  %-40s best=%.6g", _tk, _tv)

            # Modellgüte der Base-Learner-Tuning-Tasks loggen
            for task_key, score in tuner.best_scores.items():
                mlflow.log_metric(f"tuning_best__{task_key}", score)
            mlflow.log_param("blt__n_trials", cfg.tuning.n_trials)
            mlflow.log_param("blt__cv_splits", cfg.tuning.cv_splits)
            mlflow.log_param("blt__metric", cfg.tuning.metric)
            mlflow.log_param("blt__metric_regression", getattr(cfg.tuning, "metric_regression", "neg_mse"))
            mlflow.log_param("blt__single_fold", getattr(cfg.tuning, "single_fold", False))
            if cfg.tuning.models is not None:
                mlflow.log_param("blt__models", ", ".join(cfg.tuning.models))

            # Kurzform: gut sichtbare Top-Level-Metriken pro Modell+Rolle
            for model_name, roles_dict in tuned_params_by_model.items():
                for role in roles_dict:
                    if role == "default":
                        continue
                    # Task-Key rekonstruieren und Score zuordnen
                    try:
                        tk = tuner._task_key(model_name, role)
                        if tk in tuner.best_scores:
                            short_name = f"bl_score__{model_name}__{role}"
                            mlflow.log_metric(short_name, tuner.best_scores[tk])
                    except Exception:
                        pass

            # Beste Parameter auch als MLflow-Params loggen (für schnelle Übersicht)
            # "default" ist ein interner Fallback-Lookup-Key (Kopie der ersten getunten Rolle)
            # und wird für User-facing Outputs herausgefiltert.
            for model_name, roles_dict in tuned_params_by_model.items():
                for role, params in roles_dict.items():
                    if role == "default":
                        continue
                    if isinstance(params, dict):
                        for pk, pv in params.items():
                            try:
                                mlflow.log_param(f"hp__{model_name}__{role}__{pk}", pv)
                            except Exception:
                                pass  # MLflow Param-Limit (500 chars) oder Duplikate

            # Tuned Params + Scores als JSON-Artefakt (leicht inspizierbar)
            # "default"-Fallback wird gefiltert (ist intern, kein konfigurierbarer Parameter).
            def _write_tuned_params(p):
                import json
                payload = {
                    "params": {k: {r: dict(v) for r, v in roles.items() if r != "default"}
                               for k, roles in tuned_params_by_model.items()},
                    "scores": dict(tuner.best_scores),
                }
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2, ensure_ascii=False, default=float)
            _log_temp_artifact(mlflow, _write_tuned_params, "tuned_baselearner_params.json")

            # Für HTML-Report sammeln
            if hasattr(self, '_report'):
                self._report.tuning_scores.update(tuner.best_scores)
                # Tuning-Plan (Task-Sharing-Tabelle)
                try:
                    plan = tuner._build_plan(cfg.models.models_to_train)
                    plan_list = []
                    for task in sorted(plan.values(), key=tuner._task_priority):
                        role_label = {
                            "outcome": "Outcome (Classifier)",
                            "outcome_regression": "Outcome (Regressor)",
                            "propensity": "Propensity (T)",
                            "grouped_outcome_regression": "Grouped Outcome (Regressor)",
                            "pseudo_effect": "Pseudo-Effekt",
                        }.get(task.objective_family, task.objective_family)
                        sig_parts = [cfg.base_learner.type, task.estimator_task, task.sample_scope,
                                     "with_t" if task.uses_treatment_feature else "no_t", task.target_name]
                        plan_list.append({
                            "task_key": task.key,
                            "role": role_label,
                            "models": [m for m, _ in task.roles],
                            "signature": " | ".join(sig_parts),
                        })
                    self._report.add_tuning_plan(plan_list)
                except Exception:
                    pass
                # Best Hyperparameter (pro Tuning-Task)
                try:
                    for task in plan.values():
                        if task.roles:
                            mname, role = task.roles[0]
                            params = tuned_params_by_model.get(mname, {}).get(role, {})
                            if params:
                                self._report.add_best_params(task.key, dict(params))
                except Exception:
                    pass

        if getattr(cfg, "final_model_tuning", None) is not None and cfg.final_model_tuning.enabled:
            if _progress_cb:
                _progress_cb("Final-Model-Tuning")
            final_tuner = FinalModelTuner(cfg)
            # Nur die in final_model_tuning.models konfigurierten Modelle tunen.
            # None = alle FMT-fähigen Modelle (NonParamDML, DRLearner).
            fmt_models = cfg.final_model_tuning.models
            for mname in cfg.models.models_to_train:
                if fmt_models is not None and mname not in fmt_models:
                    continue
                current = dict(tuned_params_by_model.get(mname, {}) or {})
                add = final_tuner.tune_final_model(
                    model_name=mname, X=X, Y=Y, T=T,
                    base_type=cfg.base_learner.type,
                    base_fixed_params=dict(cfg.base_learner.fixed_params or {}),
                    fmt_fixed_params=dict(cfg.final_model_tuning.fixed_params or {}),
                    tuned_roles=current,
                    crosspred_splits=cfg.data_processing.cross_validation_splits,
                )
                if add:
                    current.update(add)
                    tuned_params_by_model[mname] = current

            # Modellgüte der Final-Model-Tuning-Tasks loggen
            for task_key, score in final_tuner.best_scores.items():
                mlflow.log_metric(f"fmt_best__{task_key}", score)
            mlflow.log_param("fmt__enabled", True)
            mlflow.log_param("fmt__n_trials", cfg.final_model_tuning.n_trials)
            mlflow.log_param("fmt__cv_splits", getattr(cfg.final_model_tuning, "cv_splits", 3))
            mlflow.log_param("fmt__overfit_penalty", getattr(cfg.final_model_tuning, "overfit_penalty", 0.0))
            mlflow.log_param("fmt__single_fold", getattr(cfg.final_model_tuning, "single_fold", False))
            if cfg.final_model_tuning.models is not None:
                mlflow.log_param("fmt__models", ", ".join(cfg.final_model_tuning.models))

            # Logger: Zusammenfassung
            n_fmt = len(final_tuner.best_scores)
            
            if n_fmt > 0:
                for tk, sc in final_tuner.best_scores.items():
                    if "__adjusted" in tk:
                        # Penalized scores loggen wir zusammen mit dem raw-Score
                        continue
                    raw_key = tk
                    pen_key = tk + "__adjusted"
                    raw_val = sc
                    pen_val = final_tuner.best_scores.get(pen_key)
                    model_short = tk.replace("final__", "")
                    # Score-Label: je nach Modell und Methode
                    if True:  # Alle FMT-Modelle nutzen OOF-CV
                        score_label = "OOF-R-Score"
                    if pen_val is not None and abs(raw_val - pen_val) > 1e-10:
                        self._logger.info(
                            "FMT %s: %s → %.6g (raw) / %.6g (penalized), Δ=%.6g",
                            score_label, model_short, raw_val, pen_val, raw_val - pen_val)
                    else:
                        self._logger.info("FMT %s: %s → %.6g", score_label, model_short, raw_val)
            if hasattr(self, '_report'):
                # FMT-Scores NICHT in tuning_scores mischen (die gehören zum BL-Tuning).
                # Stattdessen in fmt_info["best_scores"] für die FMT-Report-Sektion.
                # FMT-Plan für Report
                try:
                    fmt_n_trials = cfg.final_model_tuning.n_trials
                    fmt_single_fold = getattr(cfg.final_model_tuning, "single_fold", False)
                    fmt_outer_cv = cfg.data_processing.cross_validation_splits
                    fmt_internal_cv = getattr(cfg.final_model_tuning, "cv_splits", 3)
                    _mc = cfg.data_processing.mc_iters or 1
                    _fits_per_dml = _mc * fmt_internal_cv * 2 + 1  # Nuisance + model_final
                    fmt_plan_list = []
                    for mname in cfg.models.models_to_train:
                        name_lower = (mname or "").lower()
                        if name_lower in ("nonparamdml", "drlearner"):
                            _outer = 1 if fmt_single_fold else fmt_outer_cv
                            _fpt = _outer * _fits_per_dml
                            fmt_plan_list.append({
                                "model": mname, "method": f"OOF {'1' if fmt_single_fold else str(_outer)}-Fold CV",
                                "trials": fmt_n_trials, "fits_per_trial": _fpt,
                                "total_fits": fmt_n_trials * _fpt,
                                "note": f"{_outer} äußere Fold(s) × {_fits_per_dml} Fits/Fold (est.fit(train) + est.score(val), cv={fmt_internal_cv}).",
                            })
                    if fmt_plan_list:
                        self._report.add_fmt_plan(fmt_plan_list)
                    # FMT-Info
                    self._report.add_fmt_info({
                        "n_trials": fmt_n_trials,
                        "single_fold": fmt_single_fold,
                        "cv_internal": fmt_internal_cv,
                        "cv_outer": fmt_outer_cv,
                        "method": "oof_cv",
                        "overfit_penalty": getattr(cfg.final_model_tuning, "overfit_penalty", 0.0),
                        "models": [m for m in cfg.models.models_to_train if (m or "").lower() in {"nonparamdml", "drlearner"}],
                        "best_scores": {
                            k.replace("final__", ""): v
                            for k, v in final_tuner.best_scores.items()
                        },
                    })
                except Exception:
                    pass
                # FMT Best Params
                try:
                    for mname in cfg.models.models_to_train:
                        final_params = tuned_params_by_model.get(mname, {}).get("model_final", {})
                        if final_params:
                            self._report.add_fmt_best_params(mname, dict(final_params))
                except Exception:
                    pass

        return tuned_params_by_model

    def _run_training(self, cfg, X, T, Y, tuned_params_by_model, holdout_data, mlflow, _progress_cb=None):
        """Training der kausalen Learner + Cross-Predictions.

        Progress-Callback: Wenn GRF-Tuning aktiv ist, werden GRF-Modelle zuerst
        trainiert und unter dem Label "GRF-Tuning" emittiert; danach wechselt
        das Label auf "Training & Predictions" für die restlichen Modelle.
        """
        models: Dict[str, Any] = {}
        preds: Dict[str, pd.DataFrame] = {}
        fold_models: Dict[str, tuple] = {}  # {name: (fitted_model, val_indices)} für Explainability
        has_missing = X.isnull().any().any()
        keep_fold = cfg.shap_values.calculate_shap_values and holdout_data is None

        # Progress-Steuerung: GRF-Tuning als eigene Phase vor dem Haupt-Training.
        # Dazu GRF-Modelle (CausalForestDML/CausalForest) in der Iteration zuerst
        # verarbeiten, wenn GRF-Tuning aktiv ist. Models-to-train sind voneinander
        # unabhängig, deshalb ist Umsortierung sicher.
        _grf_names = ("CausalForestDML", "CausalForest")
        _has_grf_tune = any(n in (cfg.models.models_to_train or []) for n in _grf_names) and getattr(cfg.causal_forest, "use_econml_tune", False)
        _models_order = list(cfg.models.models_to_train or [])
        if _has_grf_tune:
            _grf = [m for m in _models_order if m in _grf_names]
            _other = [m for m in _models_order if m not in _grf_names]
            _models_order = _grf + _other
        _last_label = None

        for name in _models_order:
            # Progress-Label emittieren, wenn Phase wechselt
            if _progress_cb is not None:
                _is_grf = name in _grf_names
                _label = "GRF-Tuning" if (_is_grf and _has_grf_tune) else "Training & Predictions"
                if _label != _last_label:
                    _progress_cb(_label)
                    _last_label = _label
            # CausalForestDML und CausalForest können keine fehlenden Werte verarbeiten.
            # Bei fehlenden Werten wird das Modell übersprungen.
            if name in ("CausalForestDML", "CausalForest") and has_missing:
                n_missing_cols = int(X.isnull().any().sum())
                self._logger.warning(
                    "%s übersprungen – Daten enthalten fehlende Werte "
                    "(%d Spalten betroffen). GRF-basierte Modelle können keine "
                    "fehlenden Werte verarbeiten. Alle anderen Modelle (mit "
                    "LightGBM/CatBoost als Base Learner) sind davon nicht betroffen.",
                    name, n_missing_cols,
                )
                mlflow.log_param(f"model_enabled__{name}", False)
                mlflow.log_param(f"model_skipped__{name}", "missing_values")
                continue

            mlflow.log_param(f"model_enabled__{name}", True)
            _model_t0 = time.perf_counter()

            # Globalen Random-State pro Modell zurücksetzen, damit die
            # Ergebnisse jedes Modells unabhängig von der Trainingsreihenfolge
            # reproduzierbar sind. Besonders wichtig für Meta-Learner
            # (XLearner, TLearner, SLearner), die keinen eigenen random_state
            # Parameter akzeptieren und intern np.random nutzen.
            np.random.seed(cfg.constants.random_seed)
            import random
            random.seed(cfg.constants.random_seed)

            # parallel_jobs: Kerne pro Base-Learner-Fit.
            # Bei Level 3/4 laufen mehrere Folds parallel → pro Fold weniger Kerne,
            # um CPU-Übersubskription zu vermeiden.
            # CatBoost braucht mehr Threads pro Fit als LightGBM (Symmetric Tree).
            # "both"-Modus: konservativ wie CatBoost (Worst Case) parallelisieren,
            # da pro Trial wechselweise LGBM oder CatBoost trainiert wird.
            pl = cfg.constants.parallel_level
            _bt = (cfg.base_learner.type or "").lower()
            is_catboost = _bt == "catboost" or _bt == "both"
            n_fold_workers = 1  # Default: sequentiell
            if pl <= 1:
                pj = 1
            elif pl <= 2:
                pj = -1  # alle Kerne, Folds sequentiell → kein Konflikt
            else:
                # Level 3/4: Folds parallel → Kerne aufteilen
                n_cpus = available_cpu_count()
                n_cv = cfg.data_processing.cross_validation_splits
                if pl >= 4:
                    if is_catboost:
                        # CatBoost: weniger parallele Folds, mehr Threads pro Fit
                        n_fold_workers = min(n_cv, max(1, n_cpus // 4))
                    else:
                        n_fold_workers = min(n_cv, n_cpus)
                else:
                    n_fold_workers = min(n_cv, max(1, n_cpus // 4), n_cpus)
                pj = max(1, n_cpus // max(1, n_fold_workers))

            # DML/DR-Modelle (+ CausalForestDML): Interne Cross-Fitting-Folds
            # für Nuisance-Residualisierung. EconML-Default=2.
            _dml_cv = cfg.data_processing.dml_crossfit_folds

            ctx = ModelContext(
                seed=cfg.constants.random_seed,
                base_learner_type=cfg.base_learner.type,
                base_fixed_params=dict(cfg.base_learner.fixed_params or {}),
                fmt_fixed_params=dict(cfg.final_model_tuning.fixed_params or {}),
                tuned_params=tuned_params_by_model.get(name, {}),
                parallel_jobs=pj,
                dml_crossfit_folds=_dml_cv,
                mc_iters=cfg.data_processing.mc_iters,
                mc_agg=cfg.data_processing.mc_agg,

            )

            if name.lower() == "causalforestdml":
                ctx.tuned_params = dict(ctx.tuned_params or {})
                forest_defaults = dict(getattr(cfg.causal_forest, "forest_fixed_params", {}) or {})
                if forest_defaults:
                    existing = dict(ctx.tuned_params.get("forest") or {})
                    ctx.tuned_params["forest"] = {**forest_defaults, **existing}
                # CausalForestDML-Folds laufen immer sequentiell (Deadlock-Prävention) →
                # Base Learner können alle Kerne nutzen.
                if pl >= 2:
                    ctx.parallel_jobs = -1

            if name == "CausalForest":
                # CausalForest forest_fixed_params über "grf" Key in tuned_params setzen
                ctx.tuned_params = dict(ctx.tuned_params or {})
                forest_defaults = dict(getattr(cfg.causal_forest, "forest_fixed_params", {}) or {})
                if forest_defaults:
                    existing = dict(ctx.tuned_params.get("grf") or {})
                    ctx.tuned_params["grf"] = {**forest_defaults, **existing}
                # CausalForest-Folds laufen immer sequentiell (Deadlock-Prävention) →
                # der Forest kann alle Kerne für die interne Baum-Parallelisierung nutzen.
                if pl >= 2:
                    ctx.parallel_jobs = -1

            model = self.registry.create(name, ctx)

            # Debug: Effektive Parameter für jede Rolle loggen
            if name.lower() in {"nonparamdml", "drlearner"}:
                final_params = dict(ctx.fmt_fixed_params or {})
                explicit_final = ctx.tuned_params.get("model_final")
                if explicit_final:
                    final_params.update(explicit_final)
                self._logger.info(
                    "%s model_final effektive Params: %s (explicit_tuned=%s, fmt_fixed=%s)",
                    name, {k: v for k, v in final_params.items() if k in (
                        "min_child_samples", "num_leaves", "max_depth", "n_estimators",
                        "min_data_in_leaf", "depth", "iterations", "min_child_weight",
                        "colsample_bytree", "rsm", "path_smooth", "model_size_reg",
                        "reg_lambda", "reg_alpha", "l2_leaf_reg",
                    )},
                    "ja" if explicit_final else "nein",
                    bool(ctx.fmt_fixed_params),
                )

            # Warnung: DML-Modelle mit model_final profitieren stark von FinalModelTuning
            if name.lower() in {"nonparamdml", "drlearner"} and not cfg.final_model_tuning.enabled:
                has_explicit_final = bool(ctx.tuned_params.get("model_final"))
                if not has_explicit_final:
                    self._logger.warning(
                        "%s: model_final hat keine getunten Parameter. "
                        "Das CATE-Effektmodell nutzt nur base_fixed_params. "
                        "Empfehlung: final_model_tuning.enabled=true aktivieren, "
                        "um model_final über R-Score zu optimieren.",
                        name,
                    )

            if name.lower() == "causalforestdml" and getattr(cfg.causal_forest, "use_econml_tune", False):
                try:
                    # Normal: EconML-Default-Grid (12 Kombis: min_weight_fraction_leaf × max_depth × min_var_fraction_leaf)
                    # Intensiv: Erweitertes Grid (48 Kombis: + criterion + mehr max_depth + min_var_fraction_leaf=None)
                    _intensive = getattr(cfg.causal_forest, "tune_intensive", False)
                    if _intensive:
                        _cfdml_params = {
                            "min_weight_fraction_leaf": [0.0001, 0.01],
                            "max_depth": [3, 5, 8, None],
                            "min_var_fraction_leaf": [None, 0.001, 0.01],
                            "criterion": ["mse", "het"],
                        }  # 2 × 4 × 3 × 2 = 48 Kombinationen
                    else:
                        _cfdml_params = "auto"  # EconML-Default: 12 Kombinationen
                    _cfdml_n = 48 if _intensive else 12
                    # tune() verwendet alle Daten — EconML handhabt Evaluation intern
                    # via RScorer (eigene OOB/Cross-Validation). Konsistent mit CausalForest (OOB).
                    model.tune(Y, T, X=X, params=_cfdml_params)
                    mlflow.log_param("causal_forest__econml_tune", True)
                    mlflow.log_param("causal_forest__tune_intensive", _intensive)
                    self._logger.info(
                        "CausalForestDML EconML tune(): %d Grid-Kombis%s, n=%d Beobachtungen.",
                        _cfdml_n, " (intensiv)" if _intensive else "", len(X),
                    )
                except Exception:
                    self._logger.warning("CausalForestDML EconML-Tune fehlgeschlagen.", exc_info=True)
                    mlflow.log_param("causal_forest__econml_tune", False)

            # CausalForest: Grid-Search über Wald-Parameter mit RScorer-Evaluation
            if name == "CausalForest" and getattr(cfg.causal_forest, "use_econml_tune", False):
                try:
                    _intensive = getattr(cfg.causal_forest, "tune_intensive", False)
                    # Nuisance-Modelle für RScorer: Dieselben Base-Learner wie
                    # NonParamDML/CausalForestDML (model_y, model_t), mit BLT-getunten Params.
                    _nuisance_y_params = dict(cfg.base_learner.fixed_params or {})
                    _nuisance_t_params = dict(cfg.base_learner.fixed_params or {})
                    for _mn in ["NonParamDML", "ParamDML", "CausalForestDML"]:
                        _roles = tuned_params_by_model.get(_mn, {})
                        if _roles.get("model_y"):
                            _nuisance_y_params.update(_roles["model_y"])
                            break
                    for _mn in ["NonParamDML", "ParamDML", "CausalForestDML"]:
                        _roles = tuned_params_by_model.get(_mn, {})
                        if _roles.get("model_t"):
                            _nuisance_t_params.update(_roles["model_t"])
                            break
                    _rscorer_model_y = build_base_learner(
                        cfg.base_learner.type, _nuisance_y_params,
                        seed=cfg.constants.random_seed, task="classifier", parallel_jobs=-1,
                    )
                    _rscorer_model_t = build_base_learner(
                        cfg.base_learner.type, _nuisance_t_params,
                        seed=cfg.constants.random_seed, task="classifier", parallel_jobs=-1,
                    )
                    result = model.tune(
                        X, T, Y, intensive=_intensive,
                        model_y=_rscorer_model_y, model_t=_rscorer_model_t,
                    )
                    best_p = result.get("best_params", {})
                    best_score = result.get("best_r_score")
                    mlflow.log_param("causal_forest_pure__tune_enabled", True)
                    mlflow.log_param("causal_forest_pure__tune_intensive", _intensive)
                    mlflow.log_param("causal_forest_pure__tune_best_params", str(best_p))
                    if best_score is not None:
                        mlflow.log_metric("causal_forest_pure__tune_best_r_score", best_score)
                    self._logger.info(
                        "CausalForest Tune (RScorer): %d Kombis%s, n=%d → best=%s, R-Score=%.6g",
                        result.get("n_combos", 0), " (intensiv)" if _intensive else "",
                        len(X), best_p, best_score or 0,
                    )
                except Exception:
                    self._logger.warning("CausalForest Tune fehlgeschlagen.", exc_info=True)
                    mlflow.log_param("causal_forest_pure__tune_enabled", False)

            if holdout_data is None:
                # ── Alle Modelle: Externe K-Fold Cross-Validation ──
                # Auch DML/DR-Modelle durchlaufen externe CV: Das interne
                # Cross-Fitting erzeugt zwar OOF-Nuisance-Residuals, aber
                # model_final/CATE sieht alle Trainingsdaten. Echte OOF-
                # Garantie für die CATE-Predictions erfordert externe CV.
                result = train_and_crosspredict_bt_bo(
                    model=model, X=X, Y=Y, T=T,
                    n_splits=cfg.data_processing.cross_validation_splits,
                    model_name=name, random_state=cfg.constants.random_seed,
                    parallel_level=cfg.constants.parallel_level,
                    max_parallel_folds=n_fold_workers,
                    keep_last_fold_model=keep_fold,
                )
                if keep_fold:
                    df_pred, fold_model, fold_val_idx = result
                    if fold_model is not None:
                        fold_models[name] = (fold_model, fold_val_idx)
                else:
                    df_pred = result
                current_model = model
            else:
                X_test, T_test, Y_test, _ = holdout_data
                model.fit(Y, T, X=X)
                current_model = model
                test_pred = _predict_effect(model, X_test)
                df_pred = pd.DataFrame({"Y": Y_test, "T": T_test})
                if test_pred.ndim == 2 and test_pred.shape[1] > 1:
                    n_effects = test_pred.shape[1]
                    for k in range(n_effects):
                        df_pred[f"Predictions_{name}_T{k+1}"] = test_pred[:, k]
                        df_pred[f"Train_{name}_T{k+1}"] = np.nan
                    best_eff = np.nanmax(test_pred, axis=1)
                    best_arm = np.nanargmax(test_pred, axis=1) + 1
                    df_pred[f"OptimalTreatment_{name}"] = np.where(best_eff > 0, best_arm, 0)
                else:
                    df_pred[f"Predictions_{name}"] = test_pred
                    df_pred[f"Train_{name}"] = np.nan

            preds[name] = df_pred
            models[name] = current_model
            _model_dur = time.perf_counter() - _model_t0
            self._logger.info("  %s: Training + Cross-Predictions in %.1fs", name, _model_dur)
            # Predictions mit voller Präzision speichern (float64, 10 signifikante Stellen)
            _log_temp_artifact(mlflow, lambda p, _df=df_pred: _df.to_csv(p, index=False, float_format="%.10g"), f"predictions_{name}.csv")

            # Diagnose: Warnung wenn CATEs kollabiert sind
            pred_cols = [c for c in df_pred.columns if c.startswith(f"Predictions_{name}")]
            for pc in pred_cols:
                vals = df_pred[pc].dropna()
                if len(vals) == 0:
                    continue
                n_unique = vals.nunique()
                val_range = float(vals.max() - vals.min())
                cv_folds = cfg.data_processing.cross_validation_splits

                if (vals == 0).all():
                    self._logger.warning(
                        "WARNUNG: %s hat ausschließlich CATE=0 Predictions! "
                        "Mögliche Ursachen: (1) Daten haben keinen Treatment-Effekt, "
                        "(2) Modell verwendet predict statt predict_proba (Meta-Learner), "
                        "(3) Extrem unbalancierte Klassen.", pc,
                    )
                elif n_unique <= cv_folds + 1 and val_range < abs(vals.mean()) * 0.1:
                    # CATE hat nur so viele Werte wie CV-Folds → model_final/CATE-Modell
                    # ist zu einem Intercept kollabiert (1 Wert pro Fold).
                    self._logger.warning(
                        "WARNUNG: %s hat nur %d distinkte Werte bei %d Folds (Range=%.2e, Mean=%.2e). "
                        "Das CATE-Modell ist wahrscheinlich zu einem Intercept kollabiert. "
                        "Empfehlungen: (1) final_model_tuning.enabled=true aktivieren, "
                        "(2) Prüfen ob base_fixed_params zu restriktiv sind "
                        "(min_child_samples, num_leaves, max_depth), "
                        "(3) Mehr Features oder Feature-Engineering.",
                        pc, n_unique, cv_folds, val_range, abs(vals.mean()),
                    )
                elif n_unique < 20 and len(vals) > 1000:
                    self._logger.warning(
                        "HINWEIS: %s hat nur %d distinkte Werte bei %d Samples. "
                        "Das Modell differenziert wenig zwischen Individuen.",
                        pc, n_unique, len(vals),
                    )
                else:
                    self._logger.info(
                        "%s: CATE min=%.6g, median=%.6g, max=%.6g, std=%.6g, "
                        "unique=%d/%d, non-zero=%d/%d",
                        pc, vals.min(), vals.median(), vals.max(), vals.std(),
                        n_unique, len(vals), (vals != 0).sum(), len(vals),
                    )

            gc.collect()  # Zwischen Modellen: deepcopy-Reste und Fold-Daten freigeben

        _n_trained = len(models)
        _n_skipped = len(cfg.models.models_to_train) - _n_trained
        if _n_skipped > 0:
            self._logger.info("Training: %d Modelle trainiert, %d übersprungen (NaN in Daten).", _n_trained, _n_skipped)

        return models, preds, fold_models

    def _run_evaluation(self, cfg, X, T, Y, S, holdout_data, preds, models, tuned_params_by_model, mlflow, eval_mask=None):
        """Uplift-Evaluation und Diagnose-Plots."""
        eval_summary: Dict[str, Dict[str, float]] = {}
        policy_values_dict: Dict[str, pd.DataFrame] = {}
        is_mt = is_multi_treatment(T)
        _is_rct = getattr(cfg, "study_type", "rct") == "rct"

        # Warnung: eval_mask wird bei external ignoriert
        if eval_mask is not None and holdout_data is not None:
            self._logger.warning(
                "eval_mask_file ist gesetzt, wird aber ignoriert, da validate_on='%s' "
                "einen eigenen Eval-Datensatz verwendet.",
                cfg.data_processing.validate_on,
            )

        # ── Bootstrap-Iterationen für DRTester Konfidenzintervalle ──
        n_bootstrap = 1000
        self._eval_n_bootstrap = n_bootstrap
        pl = cfg.constants.parallel_level
        import matplotlib.pyplot as plt

        # ── DRTester Nuisance EINMAL fitten (für alle Modelle gleich) ──
        # Beste Classifier-Params finden: DML model_y/model_t bevorzugt,
        # dann DRLearner model_propensity, dann base_learner.fixed_params
        _best_clf_params_y = dict(cfg.base_learner.fixed_params or {})
        _best_clf_params_t = dict(cfg.base_learner.fixed_params or {})
        for mname in ["NonParamDML", "ParamDML", "CausalForestDML", "DRLearner"]:
            roles = tuned_params_by_model.get(mname, {})
            if roles.get("model_y"):
                _best_clf_params_y = {**_best_clf_params_y, **roles["model_y"]}
                break
        for mname in ["NonParamDML", "ParamDML", "CausalForestDML", "DRLearner"]:
            roles = tuned_params_by_model.get(mname, {})
            if roles.get("model_t") or roles.get("model_propensity"):
                _best_clf_params_t = {**_best_clf_params_t, **(roles.get("model_t") or roles.get("model_propensity") or {})}
                break

        fitted_tester_bt = None  # Pre-fitted DRTester für BT
        fitted_tester_mt = {}    # Pre-fitted DRTester pro Arm für MT
        try:
            # DRTester-Nuisance-Modelle: Leichtere Varianten der getunten Modelle.
            # DRTester benötigt Nuisance-Predictions für DR-Outcomes. Die Qualität
            # muss gut sein, aber nicht perfekt — es geht um Diagnostik-Plots, nicht
            # um Champion-Selektion. Daher cappen wir n_estimators auf 100 (statt
            # potentiell 400+). CV=5 wie bei den DML-Modellen für konsistente
            # Nuisance-Qualität (~12-20s Mehrkosten gegenüber cv=3, vernachlässigbar).
            _drtester_n_estimators_cap = 100
            _drtester_cv = 5

            def _cap_estimators(params: dict) -> dict:
                """Begrenzt n_estimators/iterations für DRTester-Modelle."""
                p = dict(params)
                for key in ("n_estimators", "iterations"):
                    if key in p and int(p[key]) > _drtester_n_estimators_cap:
                        p[key] = _drtester_n_estimators_cap
                return p

            dr_params_y = _cap_estimators(_best_clf_params_y)
            dr_params_t = _cap_estimators(_best_clf_params_t)
            pj = 1 if cfg.constants.parallel_level <= 1 else -1
            model_reg = build_base_learner(cfg.base_learner.type, dr_params_y, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=pj)
            model_prop = build_base_learner(cfg.base_learner.type, dr_params_t, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=pj)

            X_val = holdout_data[0] if holdout_data is not None else X
            T_val = holdout_data[1] if holdout_data is not None else T
            Y_val = holdout_data[2] if holdout_data is not None else Y

            # Train-Daten für DRTester: Bei External Eval separater Datensatz,
            # bei Cross-Validation dieselben X/T/Y wie val (wie in causaluka).
            # DRTester benötigt Train-Daten für Calibration, Qini-CI und TOC-CI.
            # Im Cross-Modus sind die Train-Statistiken leicht optimistisch
            # (keine echte Train/Val-Trennung), aber die Plots werden erzeugt
            # und sind diagnostisch nützlich.
            # Prüfe ob Train-Preds vorhanden (für irgendein Modell)
            has_train = any(
                any(c.startswith("Train_") and not np.all(np.isnan(dfp[c].to_numpy(dtype=float)))
                    for c in dfp.columns if c.startswith("Train_"))
                for dfp in preds.values()
            )

            if not is_mt:
                fitted_tester_bt = fit_drtester_nuisance(
                    model_regression=model_reg,
                    model_propensity=model_prop,
                    X_val=X_val, T_val=T_val, Y_val=Y_val,
                    X_train=X if has_train else None,
                    T_train=T if has_train else None,
                    Y_train=Y if has_train else None,
                    cv=_drtester_cv,
                )
                self._logger.info("DRTester Nuisance einmalig gefittet (BT, cv=%d, n_est≤%d). Wird für alle Modelle wiederverwendet.", _drtester_cv, _drtester_n_estimators_cap)
            else:
                # MT: Pro Arm einen binären DRTester fitten (Control vs. Arm k)
                K = len(np.unique(T_val))

                def _fit_arm(arm):
                    arm_mask_val = (T_val == 0) | (T_val == arm)
                    arm_X_val = X_val.loc[X_val.index[arm_mask_val]].copy() if isinstance(X_val, pd.DataFrame) else X_val[arm_mask_val]
                    arm_T_val = (T_val[arm_mask_val] == arm).astype(int)
                    arm_Y_val = Y_val[arm_mask_val]

                    arm_X_train, arm_T_train, arm_Y_train = None, None, None
                    if has_train:
                        arm_mask_tr = (T == 0) | (T == arm)
                        arm_X_train = X.iloc[np.where(arm_mask_tr)[0]].copy()
                        arm_T_train = (T[arm_mask_tr] == arm).astype(int)
                        arm_Y_train = Y[arm_mask_tr]

                    tester = fit_drtester_nuisance(
                        model_regression=build_base_learner(cfg.base_learner.type, dr_params_y, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=pj),
                        model_propensity=build_base_learner(cfg.base_learner.type, dr_params_t, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=pj),
                        X_val=arm_X_val, T_val=arm_T_val, Y_val=arm_Y_val,
                        X_train=arm_X_train, T_train=arm_T_train, Y_train=arm_Y_train,
                        cv=_drtester_cv,
                    )
                    return arm, tester

                if pl >= 3 and K > 2:
                    # Level 3/4: Arme parallel fitten
                    try:
                        from joblib import Parallel, delayed
                        import os
                        n_cpus = available_cpu_count()
                        n_arm_workers = min(K - 1, n_cpus)
                        self._logger.info("DRTester MT: %d Arme parallel (n_jobs=%d).", K - 1, n_arm_workers)
                        results = Parallel(n_jobs=n_arm_workers, prefer="threads")(
                            delayed(_fit_arm)(arm) for arm in range(1, K)
                        )
                        for arm, tester in results:
                            fitted_tester_mt[arm] = tester
                    except Exception:
                        self._logger.warning("Parallele MT-Arm-Fits fehlgeschlagen, Fallback sequentiell.", exc_info=True)
                        for arm in range(1, K):
                            _, tester = _fit_arm(arm)
                            fitted_tester_mt[arm] = tester
                else:
                    for arm in range(1, K):
                        _, tester = _fit_arm(arm)
                        fitted_tester_mt[arm] = tester

                self._logger.info("DRTester Nuisance einmalig gefittet (MT, %d Arme). Wird für alle Modelle wiederverwendet.", K - 1)
        except Exception:
            self._logger.warning("DRTester Nuisance Pre-Fit fehlgeschlagen. Fallback auf Per-Modell-Fit.", exc_info=True)

        # ── Eval-Maske für DRTester: X_val/T_val/Y_val auch filtern ──
        if eval_mask is not None and holdout_data is None:
            X_val_eval = X.loc[eval_mask].reset_index(drop=True)
            T_val_eval = T[eval_mask]
            Y_val_eval = Y[eval_mask]
            self._logger.info("Eval-Maske aktiv: Metriken auf %d von %d Zeilen.", len(X_val_eval), len(X))
        else:
            X_val_eval = None  # Sentinel: benutze Standard-X_val

        for mname, dfp in preds.items():
            # ── Train Many, Evaluate One: Nur Eval-Zeilen für Metriken ──
            if eval_mask is not None and holdout_data is None:
                dfp = dfp.loc[eval_mask].reset_index(drop=True)
            y = dfp["Y"].to_numpy()
            t = dfp["T"].to_numpy()

            # Diagnose: CATE-Verteilung pro Modell loggen (vor Evaluation)
            pred_cols = [c for c in dfp.columns if c.startswith(f"Predictions_{mname}")]
            for pc in pred_cols:
                vals = dfp[pc].dropna()
                if len(vals) > 0:
                    self._logger.info(
                        "Evaluation %s: n=%d, min=%.6g, median=%.6g, max=%.6g, "
                        "std=%.6g, non-zero=%d/%d, unique=%.0f",
                        pc, len(vals), vals.min(), vals.median(), vals.max(),
                        vals.std(), (vals != 0).sum(), len(vals),
                        min(vals.nunique(), 999),
                    )

            # ── Phase 1: Schnelle Metriken (IMMER, für Champion-Selektion) ──
            if is_mt:
                mt_pred_cols = [c for c in dfp.columns if c.startswith(f"Predictions_{mname}_T")]
                if mt_pred_cols:
                    scores_2d = dfp[mt_pred_cols].to_numpy()
                    eval_summary[mname] = mt_eval_summary(y=y, t=t, scores_2d=scores_2d, propensity=None)
                    for key, val in eval_summary[mname].items():
                        if isinstance(val, (int, float)):
                            mlflow.log_metric(f"{key}__{mname}", float(val))
                        elif isinstance(val, dict):
                            for sub_key, sub_val in val.items():
                                mlflow.log_metric(f"{key}_{sub_key}__{mname}", float(sub_val))
            else:
                pred_col = f"Predictions_{mname}"
                if pred_col in dfp.columns:
                    s = dfp[pred_col].to_numpy()
                    curve = uplift_curve(y=y, t=t, score=s)
                    eval_summary[mname] = {
                        "qini": float(qini_coefficient(curve)),
                        "auuc": float(auuc(curve)),
                        "uplift_at_10pct": float(uplift_at_k(curve, k_fraction=0.10)),
                        "uplift_at_20pct": float(uplift_at_k(curve, k_fraction=0.20)),
                        "uplift_at_50pct": float(uplift_at_k(curve, k_fraction=0.50)),
                    }
                    if _is_rct:
                        eval_summary[mname]["policy_value"] = float(policy_value(y=y, t=t, score=s, threshold=0.0))
                    for key, val in eval_summary[mname].items():
                        short = {"uplift_at_10pct": "uplift10", "uplift_at_20pct": "uplift20", "uplift_at_50pct": "uplift50"}.get(key, key)
                        mlflow.log_metric(f"{short}__{mname}", val)

            # ── CATE-Verteilungs-Plots (IMMER, alle Modelle) ──
            # Schnell (~0.5s), reine Histogramme. Zeigen Training- und
            # Cross-Validated-Predictions nebeneinander.
            try:
                if is_mt:
                    mt_pred_cols = [c for c in dfp.columns if c.startswith(f"Predictions_{mname}_T")]
                    for k, pc in enumerate(mt_pred_cols):
                        arm = k + 1
                        val_arr = dfp[pc].to_numpy(dtype=float)
                        train_col = f"Train_{mname}_T{arm}"
                        train_arr = None
                        if train_col in dfp.columns and not np.all(np.isnan(dfp[train_col].to_numpy(dtype=float))):
                            train_arr = dfp[train_col].to_numpy(dtype=float)
                        fig_dist = generate_cate_distribution_plot(
                            cate_preds_val=val_arr, cate_preds_train=train_arr,
                            model_name=mname, arm_label=f"T{arm}",
                        )
                        if fig_dist is not None:
                            _log_figure_fast(mlflow, fig_dist, f"distribution__{mname}_T{arm}.png")
                            if hasattr(self, '_report'):
                                self._report.add_plot(f"{mname}_T{arm}", "cate_distribution", fig_dist)
                            plt.close(fig_dist)
                else:
                    pred_col = f"Predictions_{mname}"
                    if pred_col in dfp.columns:
                        val_arr = dfp[pred_col].to_numpy(dtype=float)
                        train_col = f"Train_{mname}"
                        train_arr = None
                        if train_col in dfp.columns and not np.all(np.isnan(dfp[train_col].to_numpy(dtype=float))):
                            train_arr = dfp[train_col].to_numpy(dtype=float)
                        fig_dist = generate_cate_distribution_plot(
                            cate_preds_val=val_arr, cate_preds_train=train_arr,
                            model_name=mname,
                        )
                        if fig_dist is not None:
                            _log_figure_fast(mlflow, fig_dist, f"distribution__{mname}.png")
                            if hasattr(self, '_report'):
                                self._report.add_plot(mname, "cate_distribution", fig_dist)
                            plt.close(fig_dist)
            except Exception:
                self._logger.warning("CATE-Verteilungsplot für %s fehlgeschlagen.", mname, exc_info=True)

        # ── Champion frühzeitig bestimmen (für Level-abhängige Plot-Steuerung) ──
        _early_champion = self._determine_champion(cfg, eval_summary, models) if eval_summary else None
        self._logger.info(
            "Metriken für %d Modelle berechnet. Vorläufiger Champion: %s. "
            "Diagnostik-Plots: %s",
            len(eval_summary), _early_champion or "–",
            "alle Modelle" if pl <= 2 else ("nur Champion" if pl >= 4 else "Champion + Challenger"),
        )

        # ── Phase 2: DRTester-Plots + sklift (Level-abhängig) ──
        # Level 1-2: Alle Modelle bekommen volle Diagnostik
        # Level 3:   Champion + bester Challenger
        # Level 4:   Nur Champion
        if pl >= 4:
            plot_models = {_early_champion} if _early_champion else set()
        elif pl >= 3:
            # Champion + 1 bester Challenger
            plot_models = {_early_champion} if _early_champion else set()
            sel_met = cfg.selection.metric
            other = [(m, eval_summary[m].get(sel_met, 0)) for m in eval_summary if m != _early_champion and m in models and isinstance(eval_summary[m].get(sel_met), (int, float))]
            if other:
                other.sort(key=lambda x: x[1], reverse=cfg.selection.higher_is_better)
                plot_models.add(other[0][0])
        else:
            plot_models = set(preds.keys())

        for mname, dfp in preds.items():
            if mname not in plot_models:
                continue
            y = dfp["Y"].to_numpy()
            t = dfp["T"].to_numpy()

            if is_mt:
                eval_summary, policy_values_dict = self._evaluate_mt(
                    cfg, X, T, Y, holdout_data, mname, dfp, y, t,
                    tuned_params_by_model, eval_summary, policy_values_dict, mlflow,
                    fitted_testers=fitted_tester_mt, eval_mask=eval_mask)
            else:
                eval_summary, policy_values_dict = self._evaluate_bt(
                    cfg, X, T, Y, holdout_data, mname, dfp, y, t,
                    tuned_params_by_model, eval_summary, policy_values_dict, mlflow,
                    fitted_tester=fitted_tester_bt, eval_mask=eval_mask)

        # ── Phase 3: scikit-uplift Plots (IMMER für alle Modelle) ──
        # Schnell (~2-5s pro Modell), reine Histogramm/Kurven-Plots ohne
        # Bootstrap-Berechnungen. Unabhängig von Phase 2 (DRTester), damit
        # alle Modelle Qini-Kurve, Uplift-by-Percentile und Treatment-Balance
        # im Report haben.
        eval_X = holdout_data[0] if holdout_data is not None else X
        eval_T = holdout_data[1] if holdout_data is not None else T
        eval_Y = holdout_data[2] if holdout_data is not None else Y
        for mname, dfp in preds.items():
            # ── Train Many, Evaluate One: auch sklift-Plots auf Eval-Subset ──
            if eval_mask is not None and holdout_data is None:
                dfp = dfp.loc[eval_mask].reset_index(drop=True)
                eval_T_sk = T[eval_mask]
                eval_Y_sk = Y[eval_mask]
            else:
                eval_T_sk = eval_T
                eval_Y_sk = eval_Y
            y = dfp["Y"].to_numpy()
            t = dfp["T"].to_numpy()
            if is_mt:
                mt_pred_cols = [c for c in dfp.columns if c.startswith(f"Predictions_{mname}_T")]
                if not mt_pred_cols:
                    continue
                K = len(np.unique(T))
                for k in range(1, K):
                    arm = k
                    arm_scores = dfp[mt_pred_cols[k-1]].to_numpy(dtype=float) if k-1 < len(mt_pred_cols) else None
                    if arm_scores is None:
                        continue
                    arm_mask = (eval_T_sk == 0) | (eval_T_sk == arm)
                    arm_cate = arm_scores[arm_mask]
                    arm_T_bin = (eval_T_sk[arm_mask] == arm).astype(int)
                    arm_Y = eval_Y_sk[arm_mask]
                    try:
                        sk_qini, sk_pct, sk_tb = generate_sklift_plots(arm_cate, arm_T_bin, arm_Y)
                        if not _is_rct:
                            sk_tb = None  # Treatment Balance nur bei RCT aussagekräftig
                        if sk_qini is not None:
                            _log_figure_fast(mlflow, sk_qini, f"sklift_qini__{mname}_T{arm}.png")
                        if sk_pct is not None:
                            _log_figure_fast(mlflow, sk_pct, f"sklift_percentile__{mname}_T{arm}.png")
                        if sk_tb is not None:
                            _log_figure_fast(mlflow, sk_tb, f"treatment_balance__{mname}_T{arm}.png")
                        if hasattr(self, '_report'):
                            label = f"{mname}_T{arm}"
                            for fig, key in [(sk_qini, "sklift_qini"), (sk_pct, "sklift_percentile"), (sk_tb, "treatment_balance")]:
                                if fig is not None:
                                    self._report.add_plot(label, key, fig)
                        for fig in [sk_qini, sk_pct, sk_tb]:
                            if fig is not None:
                                plt.close(fig)
                    except Exception:
                        self._logger.warning("sklift-Plots für %s T%d fehlgeschlagen.", mname, arm, exc_info=True)
            else:
                pred_col = f"Predictions_{mname}"
                if pred_col not in dfp.columns:
                    continue
                cate_vals = dfp[pred_col].to_numpy(dtype=float)
                try:
                    sk_qini, sk_pct, sk_tb = generate_sklift_plots(cate_vals, eval_T_sk, eval_Y_sk)
                    if not _is_rct:
                        sk_tb = None  # Treatment Balance nur bei RCT aussagekräftig
                    if sk_qini is not None:
                        _log_figure_fast(mlflow, sk_qini, f"sklift_qini__{mname}.png")
                    if sk_pct is not None:
                        _log_figure_fast(mlflow, sk_pct, f"sklift_percentile__{mname}.png")
                    if sk_tb is not None:
                        _log_figure_fast(mlflow, sk_tb, f"treatment_balance__{mname}.png")
                    if hasattr(self, '_report'):
                        for fig, key in [(sk_qini, "sklift_qini"), (sk_pct, "sklift_percentile"), (sk_tb, "treatment_balance")]:
                            if fig is not None:
                                self._report.add_plot(mname, key, fig)
                    for fig in [sk_qini, sk_pct, sk_tb]:
                        if fig is not None:
                            plt.close(fig)
                except Exception:
                    self._logger.warning("sklift-Plots für %s fehlgeschlagen.", mname, exc_info=True)

        # Historischer Score (nur BT)
        hist_score_eval = S
        if holdout_data is not None and holdout_data[3] is not None:
            hist_score_eval = holdout_data[3]
        elif eval_mask is not None and S is not None:
            hist_score_eval = S[eval_mask]

        if hist_score_eval is not None and is_mt:
            self._logger.info("Historischer Score (S) vorhanden, wird aber bei Multi-Treatment übersprungen.")

        if hist_score_eval is not None and not is_mt:
            eval_summary, policy_values_dict = self._evaluate_historical_score(
                cfg, X, T, Y, holdout_data, preds, hist_score_eval, eval_summary, policy_values_dict, mlflow,
                fitted_tester=fitted_tester_bt, eval_mask=eval_mask)

        if eval_summary:
            def _write_eval(p):
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(eval_summary, fh, ensure_ascii=False, indent=2)

            _log_temp_artifact(mlflow, _write_eval, "uplift_eval_summary.json")

        return eval_summary, policy_values_dict, fitted_tester_bt

    def _evaluate_bt(self, cfg, X, T, Y, holdout_data, mname, dfp, y, t, tuned_params_by_model, eval_summary, policy_values_dict, mlflow, fitted_tester=None, eval_mask=None):
        """Binary-Treatment: DRTester-Diagnostik-Plots für ein Modell.

        Metriken (Qini, AUUC, etc.) werden bereits in Phase 1 berechnet.
        Diese Methode erzeugt nur die teuren DRTester-Plots (Calibration,
        Qini/TOC mit Bootstrap-CIs) und scikit-uplift-Plots."""
        import matplotlib.pyplot as plt
        pred_col = f"Predictions_{mname}"
        if pred_col not in dfp.columns:
            return eval_summary, policy_values_dict

        # X/T/Y für DRTester-Plots: bei eval_mask auf Eval-Zeilen filtern
        use_mask = eval_mask is not None and holdout_data is None
        X_eval = X.loc[eval_mask].reset_index(drop=True) if use_mask else X
        T_eval = T[eval_mask] if use_mask else T
        Y_eval = Y[eval_mask] if use_mask else Y
        dfp_eval = dfp.loc[eval_mask].reset_index(drop=True) if use_mask else dfp

        # Wenn eval_mask aktiv ist, muss der fitted_tester auf das Eval-Subset gefiltert werden
        ft_eval = fitted_tester
        if use_mask and fitted_tester is not None:
            ft_eval = filter_tester_for_mask(fitted_tester, eval_mask, len(X_eval))

        try:
            train_col = f"Train_{mname}"
            cate_train = None
            # Train-CATEs nur verwenden wenn vorhanden und nicht-NaN
            if train_col in dfp_eval.columns and not np.all(np.isnan(dfp_eval[train_col].to_numpy(dtype=float))):
                cate_train = dfp_eval[train_col].to_numpy(dtype=float)

            bundle = evaluate_cate_with_plots(
                X_val=(holdout_data[0] if holdout_data is not None else X_eval),
                T_val=(holdout_data[1] if holdout_data is not None else T_eval),
                Y_val=(holdout_data[2] if holdout_data is not None else Y_eval),
                cate_preds_val=dfp_eval[pred_col].to_numpy(dtype=float),
                X_train=X if cate_train is not None else None,
                T_train=T if cate_train is not None else None,
                Y_train=Y if cate_train is not None else None,
                cate_preds_train=cate_train, n_groups=10,
                fitted_tester=ft_eval,
                n_bootstrap=getattr(self, '_eval_n_bootstrap', 1000),
                seed=cfg.constants.random_seed,
            )
            policy_values_dict[mname] = bundle.policy_values
            bundle.log_to_mlflow(mlflow, mname, log_temp_artifact_fn=_log_temp_artifact)
            if hasattr(self, '_report'):
                bundle.add_to_report(self._report, mname)
            bundle.close_figures()
        except Exception:
            self._logger.warning("DRTester/SkLift-Plots für %s fehlgeschlagen.", mname, exc_info=True)

        return eval_summary, policy_values_dict

    def _evaluate_mt(self, cfg, X, T, Y, holdout_data, mname, dfp, y, t, tuned_params_by_model, eval_summary, policy_values_dict, mlflow, fitted_testers=None, eval_mask=None):
        """Multi-Treatment: DRTester-Diagnostik-Plots für ein Modell.

        Metriken (per-Arm Qini, AUUC, globaler Policy Value) werden bereits
        in Phase 1 berechnet. Diese Methode erzeugt nur die teuren
        DRTester-Plots pro Treatment-Arm und scikit-uplift-Plots."""
        import matplotlib.pyplot as plt

        # eval_mask: auf Eval-Subset filtern (wie bei _evaluate_bt)
        use_mask = eval_mask is not None and holdout_data is None
        if use_mask:
            dfp = dfp.loc[eval_mask].reset_index(drop=True)
            X = X.loc[eval_mask].reset_index(drop=True)
            T = T[eval_mask]
            Y = Y[eval_mask]
            y = dfp["Y"].to_numpy()
            t = dfp["T"].to_numpy()

        mt_pred_cols = [c for c in dfp.columns if c.startswith(f"Predictions_{mname}_T")]
        if not mt_pred_cols:
            return eval_summary, policy_values_dict

        scores_2d = dfp[mt_pred_cols].to_numpy()

        try:
            n_effects = scores_2d.shape[1]
            for k in range(n_effects):
                arm = k + 1
                arm_scores = scores_2d[:, k]
                try:
                    eval_X = (holdout_data[0] if holdout_data is not None else X)
                    eval_T = (holdout_data[1] if holdout_data is not None else T)
                    eval_Y = (holdout_data[2] if holdout_data is not None else Y)
                    arm_mask = (eval_T == 0) | (eval_T == arm)
                    arm_X = eval_X.loc[eval_X.index[arm_mask]].copy()
                    arm_T = (eval_T[arm_mask] == arm).astype(int)
                    arm_Y = eval_Y[arm_mask]
                    arm_cate = arm_scores[arm_mask]

                    train_col = f"Train_{mname}_T{arm}"
                    cate_train, arm_X_train, arm_T_train, arm_Y_train = None, None, None, None
                    if holdout_data is not None and train_col in dfp.columns and not np.all(np.isnan(dfp[train_col].to_numpy(dtype=float))):
                        train_mask = (T == 0) | (T == arm)
                        cate_train = dfp[train_col].to_numpy(dtype=float)[train_mask]
                        arm_X_train = X.iloc[np.where(train_mask)[0]].copy()
                        arm_T_train = (T[train_mask] == arm).astype(int)
                        arm_Y_train = Y[train_mask]

                    # Pre-fitted DRTester für diesen Arm verwenden (wenn vorhanden)
                    arm_tester = (fitted_testers or {}).get(arm)
                    bundle = evaluate_cate_with_plots(
                        X_val=arm_X, T_val=arm_T, Y_val=arm_Y, cate_preds_val=arm_cate,
                        X_train=arm_X_train, T_train=arm_T_train, Y_train=arm_Y_train,
                        cate_preds_train=cate_train, n_groups=10,
                        fitted_tester=arm_tester,
                        n_bootstrap=getattr(self, '_eval_n_bootstrap', 1000),
                        seed=cfg.constants.random_seed,
                    )
                    arm_label = f"{mname}_T{arm}"
                    policy_values_dict[arm_label] = bundle.policy_values
                    bundle.log_to_mlflow(mlflow, arm_label, log_temp_artifact_fn=_log_temp_artifact)
                    if hasattr(self, '_report'):
                        bundle.add_to_report(self._report, arm_label)
                    bundle.close_figures()
                except Exception:
                    self._logger.warning("DRTester-Plots für %s T%d fehlgeschlagen.", mname, arm, exc_info=True)
        except Exception:
            self._logger.warning("DRTester-Plots für %s fehlgeschlagen.", mname, exc_info=True)

        return eval_summary, policy_values_dict

    def _evaluate_historical_score(self, cfg, X, T, Y, holdout_data, preds, hist_score_eval, eval_summary, policy_values_dict, mlflow, fitted_tester=None, eval_mask=None):
        """Vergleich der kausalen Modelle gegen einen historischen Score."""
        import matplotlib.pyplot as plt
        from rubin.utils.plot_theme import apply_rubin_theme
        apply_rubin_theme()

        hist_name = cfg.historical_score.name
        hist_score = np.asarray(hist_score_eval).astype(float)
        hist_score = np.nan_to_num(hist_score, nan=0.0, posinf=0.0, neginf=0.0)
        if not cfg.historical_score.higher_is_better:
            hist_score = -hist_score

        # Bei eval_mask: X/T/Y auf Eval-Subset filtern (hist_score ist bereits gefiltert)
        use_mask = eval_mask is not None and holdout_data is None
        eval_X = X.loc[eval_mask].reset_index(drop=True) if use_mask else X
        eval_y = holdout_data[2] if holdout_data is not None else (Y[eval_mask] if use_mask else Y)
        eval_t = holdout_data[1] if holdout_data is not None else (T[eval_mask] if use_mask else T)
        curve_h = uplift_curve(y=eval_y, t=eval_t, score=hist_score)
        eval_summary[hist_name] = {
            "qini": float(qini_coefficient(curve_h)),
            "auuc": float(auuc(curve_h)),
            "uplift_at_10pct": float(uplift_at_k(curve_h, k_fraction=0.10)),
            "uplift_at_20pct": float(uplift_at_k(curve_h, k_fraction=0.20)),
            "uplift_at_50pct": float(uplift_at_k(curve_h, k_fraction=0.50)),
        }
        if getattr(cfg, "study_type", "rct") == "rct":
            eval_summary[hist_name]["policy_value"] = float(policy_value(y=eval_y, t=eval_t, score=hist_score, threshold=0.0))
        for key, val in eval_summary[hist_name].items():
            short = {"uplift_at_10pct": "uplift10", "uplift_at_20pct": "uplift20", "uplift_at_50pct": "uplift50"}.get(key, key)
            mlflow.log_metric(f"{short}__{hist_name}", val)

        # Distribution-Plot für historischen Score
        try:
            fig_dist = generate_cate_distribution_plot(
                cate_preds_val=hist_score, model_name=hist_name,
            )
            if fig_dist is not None:
                _log_figure_fast(mlflow, fig_dist, f"distribution__{hist_name}.png")
                if hasattr(self, '_report'):
                    self._report.add_plot(hist_name, "cate_distribution", fig_dist)
                plt.close(fig_dist)
        except Exception:
            self._logger.warning("Distribution-Plot für %s fehlgeschlagen.", hist_name, exc_info=True)

        try:
            # Bei eval_mask: fitted_tester filtern
            ft_hist = fitted_tester
            if use_mask and fitted_tester is not None:
                ft_hist = filter_tester_for_mask(fitted_tester, eval_mask, len(eval_X))

            # Train-Daten: immer voller Datensatz (wie in der funktionierenden Version)
            if ft_hist is not None:
                bundle_h = evaluate_cate_with_plots(
                    fitted_tester=ft_hist,
                    X_val=eval_X, T_val=eval_t, Y_val=eval_y,
                    cate_preds_val=hist_score,
                    X_train=X, T_train=T, Y_train=Y,
                    cate_preds_train=hist_score if not use_mask else None,
                    n_groups=10,
                    n_bootstrap=getattr(self, '_eval_n_bootstrap', 1000),
                    seed=cfg.constants.random_seed,
                )
            else:
                # Fallback: eigene Nuisance fitten
                params_y = dict(cfg.base_learner.fixed_params or {})
                params_t = dict(cfg.base_learner.fixed_params or {})
                pj = 1 if cfg.constants.parallel_level <= 1 else -1
                model_reg = build_base_learner(cfg.base_learner.type, params_y, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=pj)
                model_prop = build_base_learner(cfg.base_learner.type, params_t, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=pj)
                bundle_h = evaluate_cate_with_plots(
                    model_regression=model_reg, model_propensity=model_prop,
                    X_val=eval_X, T_val=eval_t, Y_val=eval_y,
                    cate_preds_val=hist_score,
                    X_train=X, T_train=T, Y_train=Y,
                    cate_preds_train=hist_score,
                    n_groups=10,
                    n_bootstrap=getattr(self, '_eval_n_bootstrap', 1000),
                    seed=cfg.constants.random_seed,
                )
            policy_values_dict[hist_name] = bundle_h.policy_values
            bundle_h.log_to_mlflow(mlflow, hist_name, log_temp_artifact_fn=_log_temp_artifact)
            if hasattr(self, '_report'):
                bundle_h.add_to_report(self._report, hist_name)
            bundle_h.close_figures()

            for mname, dfp in preds.items():
                pred_col = f"Predictions_{mname}"
                if pred_col not in dfp.columns:
                    continue
                # Bei eval_mask: preds auf Eval-Subset filtern
                dfp_cmp = dfp.loc[eval_mask].reset_index(drop=True) if use_mask else dfp
                df_cmp = pd.DataFrame({
                    "Y": dfp_cmp["Y"].to_numpy(), "T": dfp_cmp["T"].to_numpy(),
                    mname: dfp_cmp[pred_col].to_numpy(dtype=float), hist_name: hist_score,
                })
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_custom_qini_curve(data=df_cmp, causal_score_label=mname, affinity_score_label=hist_name, ax=ax, relative_axes=True)
                _log_figure_fast(mlflow, fig, f"custom_qini__{mname}_vs_{hist_name}.png")
                if hasattr(self, '_report'):
                    self._report.add_plot(mname, f"qini_vs_{hist_name}", fig)
                plt.close(fig)
        except Exception:
            self._logger.warning("DRTester/SkLift-Plots für historischen Score fehlgeschlagen.", exc_info=True)

        try:
            hist_pv = policy_values_dict.get(hist_name)
            if (hist_pv is not None and len(hist_pv) > 0
                    and hasattr(hist_pv, 'columns') and "policy_value" in hist_pv.columns
                    and len(policy_values_dict) > 1):
                figs = policy_value_comparison_plots(policy_values_dict, comparison_model_name=hist_name)
                for model_name, fig in figs.items():
                    _log_figure_fast(mlflow, fig, f"policy_compare__{model_name}_vs_{hist_name}.png")
                    if hasattr(self, '_report'):
                        self._report.add_plot(model_name, f"policy_compare_vs_{hist_name}", fig)
                    plt.close(fig)
        except Exception:
            self._logger.warning("Policy-Value-Vergleich fehlgeschlagen.", exc_info=True)

        _log_temp_artifact(mlflow, lambda p: pd.DataFrame({"fraction": curve_h.fraction, "uplift": curve_h.uplift, "n_treat": curve_h.n_treat, "n_control": curve_h.n_control}).to_csv(p, index=False), f"uplift_curve__{hist_name}.csv")

        return eval_summary, policy_values_dict

    # ------------------------------------------------------------------
    # Surrogate-Einzelbaum (Teacher-Learner)
    # ------------------------------------------------------------------

    def _determine_champion(self, cfg, eval_summary, models):
        """Ermittelt den Champion-Modellnamen anhand der Konfiguration und eval_summary.

        Wird vor dem Bundle-Export aufgerufen, damit der Surrogate-Einzelbaum
        den Champion bereits kennt. Nur Einträge, die auch in `models` existieren,
        werden als Kandidaten betrachtet (historische Scores, Surrogates etc. ausgeschlossen)."""
        manual = (getattr(cfg.selection, "manual_champion", None) or "").strip() or None
        if manual and manual in models:
            return manual

        _exclude = {SURROGATE_MODEL_NAME}
        entries = [
            ModelEntry(name=name, artifact_path=f"models/{name}.pkl", metrics=float_metrics(metrics or {}))
            for name, metrics in eval_summary.items()
            if name not in _exclude and name in models
        ]
        champion = choose_champion(entries, metric=cfg.selection.metric, higher_is_better=cfg.selection.higher_is_better)
        if champion is None and entries:
            champion = entries[0].name
        return champion

    def _compute_heterogeneity_assessment(self, cfg, eval_summary, champion_name):
        """Bewertet die gefundene Heterogenität und den historischen Vergleich.

        Gibt ein Dict mit zwei getrennten Bewertungen zurück:
        - heterogeneity: Rein modellbasierte Heterogenitäts-Bewertung
        - hist_comparison: Vergleich mit historischem Score (falls vorhanden)
        """
        _exclude = {SURROGATE_MODEL_NAME, "Ensemble"}

        champ_metrics = eval_summary.get(champion_name, {})
        champ_qini = champ_metrics.get("qini", 0.0)
        champ_pv = champ_metrics.get("policy_value", 0.0)
        uplift_10 = champ_metrics.get("uplift_at_10pct", 0.0)
        uplift_50 = champ_metrics.get("uplift_at_50pct", 0.0)

        concentration = (uplift_10 / uplift_50) if uplift_50 > 0 and uplift_10 > 0 else 0.0

        model_qinis = {
            n: m.get("qini", 0.0) for n, m in eval_summary.items()
            if n not in _exclude
        }
        n_positive_qini = sum(1 for v in model_qinis.values() if v > 0)
        n_models = len(model_qinis)

        # ── 1. Heterogenitäts-Bewertung (rein modellbasiert) ──
        details = []
        if champ_qini <= 0:
            level = "none"
            details.append("Champion-Qini ≤ 0 — kein Modell sortiert besser als Random.")
        elif champ_pv < 0:
            level = "none"
            details.append("Policy Value negativ — Targeting schadet.")
        elif n_positive_qini >= 3 and champ_pv > 0 and concentration >= 1.5:
            level = "strong"
            details.append(f"Konzentration Top-10/50: {concentration:.1f}×.")
            details.append(f"Konsens: {n_positive_qini}/{n_models} Modelle mit positivem Qini.")
        elif n_positive_qini >= 2 and champ_pv >= 0:
            level = "moderate"
            details.append(f"Konsens: {n_positive_qini}/{n_models} Modelle mit positivem Qini.")
        else:
            level = "weak"
            details.append(f"Nur {n_positive_qini}/{n_models} Modell(e) mit positivem Qini.")

        _levels = {
            "strong":   {"color": "#1a7f37", "bg": "#e8f5ec", "border": "#a3d9b1", "label": "Starke Heterogenität"},
            "moderate": {"color": "#7a5a00", "bg": "#fffbeb", "border": "#e8d49c", "label": "Moderate Heterogenität"},
            "weak":     {"color": "#b35900", "bg": "#fff3e6", "border": "#f0c78a", "label": "Schwache Heterogenität"},
            "none":     {"color": "#9B111E", "bg": "#fef2f2", "border": "#f5c6cb", "label": "Keine Heterogenität"},
        }
        style = _levels[level]

        heterogeneity = {
            "level": level, **style,
            "details": details,
            "champion": champion_name,
            "champion_qini": round(champ_qini, 6),
            "champion_pv": round(champ_pv, 6),
            "n_positive_qini": n_positive_qini,
            "n_models": n_models,
            "concentration": round(concentration, 2),
        }

        # ── 2. Historischer Score-Vergleich (separat) ──
        hist_comparison = None
        hist_name = getattr(cfg.historical_score, "name", None)
        if hist_name and hist_name in eval_summary:
            hist_metrics = eval_summary[hist_name]
            hist_qini = hist_metrics.get("qini", 0.0)
            hist_pv = hist_metrics.get("policy_value", 0.0)
            qini_diff = champ_qini - hist_qini
            pv_diff = champ_pv - hist_pv
            beats_qini = qini_diff > 0
            beats_pv = pv_diff > 0

            # Level für historischen Vergleich
            if beats_qini and beats_pv:
                h_level = "beats_both"
                h_label = "Champion übertrifft historischen Score"
                h_style = _levels["strong"]
            elif beats_qini:
                h_level = "beats_qini"
                h_label = "Champion übertrifft historischen Score (Qini)"
                h_style = _levels["moderate"]
            elif hist_qini <= 0:
                h_level = "hist_negative"
                h_label = "Historischer Score hat keinen Uplift"
                h_style = _levels["weak"]
            else:
                h_level = "behind"
                h_label = "Champion hinter historischem Score"
                h_style = _levels["none"]

            hist_comparison = {
                "level": h_level, **h_style,
                "label": h_label,
                "hist_name": hist_name,
                "hist_qini": round(hist_qini, 6),
                "hist_pv": round(hist_pv, 6),
                "champion_qini": round(champ_qini, 6),
                "champion_pv": round(champ_pv, 6),
                "qini_diff": round(qini_diff, 6),
                "pv_diff": round(pv_diff, 6),
                "beats_qini": beats_qini,
                "beats_pv": beats_pv,
            }

        return {"heterogeneity": heterogeneity, "hist_comparison": hist_comparison}

    def _build_surrogate_regressor(self, cfg):
        """Erzeugt einen Einzelbaum-Regressor aus der Surrogate- und Base-Learner-Konfiguration."""
        tree_cfg = cfg.surrogate_tree
        base_type = cfg.base_learner.type.lower()
        seed = cfg.constants.random_seed

        # Surrogate-Tree ist ein Einzelbaum — "both" macht hier keinen Sinn,
        # da kein Tuning stattfindet. Fallback auf CatBoost (= globaler Default).
        if base_type == "both":
            base_type = "catboost"

        if base_type == "lgbm":
            params = {
                "n_estimators": 1,
                "num_leaves": tree_cfg.num_leaves,
                "min_child_samples": tree_cfg.min_samples_leaf,
                "max_depth": tree_cfg.max_depth if tree_cfg.max_depth is not None else -1,
                "learning_rate": 1.0,  # Einzelbaum: kein Shrinkage
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
            }
        elif base_type == "catboost":
            params = {
                "iterations": 1,
                "min_data_in_leaf": tree_cfg.min_samples_leaf,
                "depth": tree_cfg.max_depth if tree_cfg.max_depth is not None else 6,
                "learning_rate": 1.0,
                "bootstrap_type": "No",  # Kein Bootstrapping für Einzelbaum
                "rsm": 1.0,
                "l2_leaf_reg": 0.0,
            }
        else:
            raise ValueError(f"Unbekannter base_learner.type={base_type!r} für Surrogate-Tree.")

        return build_base_learner(base_type, params, seed=seed, task="regressor", parallel_jobs=1 if cfg.constants.parallel_level <= 1 else -1)

    @staticmethod
    def _log_surrogate_tree_info(tree_model, base_type: str):
        """Extrahiert Baumtiefe und Blattanzahl für Logging."""
        base_type = base_type.lower()
        depth, n_leaves = None, None
        try:
            if base_type == "lgbm":
                info = tree_model.booster_.dump_model()
                tree_info = info.get("tree_info", [{}])
                if tree_info:
                    n_leaves = tree_info[0].get("num_leaves")
                    depth = tree_info[0].get("max_depth")
            elif base_type == "catboost":
                all_params = tree_model.get_all_params()
                depth = all_params.get("depth")
                # CatBoost: symmetrischer Baum → 2^depth Blätter
                if depth is not None:
                    n_leaves = 2 ** int(depth)
        except Exception:
            pass
        return depth, n_leaves

    def _train_and_evaluate_surrogate(self, cfg, X, T, Y, teacher_name, preds, holdout_data, models, eval_summary, mlflow, surrogate_name=None, run_drtester=False, fitted_tester=None, eval_mask=None):
        """Trainiert den Surrogate-Einzelbaum und evaluiert ihn.

        Parameters
        ----------
        teacher_name : str
            Name des Modells, dessen CATE-Predictions als Regressionsziel dienen.
        surrogate_name : str, optional
            Name für den Surrogate im eval_summary/Report. Default: SURROGATE_MODEL_NAME.
        run_drtester : bool
            Wenn True, werden volle DRTester-Plots (BLP, Calibration, Qini, TOC, Policy Values) erzeugt.
        fitted_tester : CustomDRTester, optional
            Pre-fitted DRTester für DRTester-Plots (vermeidet redundantes Nuisance-Fitting).

        Returns
        -------
        surrogate_wrapper : SurrogateTreeWrapper
        surrogate_df : pd.DataFrame
        """
        from sklearn.model_selection import KFold

        base_type = cfg.base_learner.type
        # Surrogate-Tree: "both" macht hier keinen Sinn (kein Tuning, Einzelbaum).
        # Fallback auf CatBoost (= globaler Default).
        if (base_type or "").lower() == "both":
            base_type = "catboost"
        seed = cfg.constants.random_seed
        n_splits = cfg.data_processing.cross_validation_splits
        sname = surrogate_name or SURROGATE_MODEL_NAME
        is_mt = is_multi_treatment(T)

        champion_df = preds[teacher_name]

        if holdout_data is None:
            # --- Cross-Modus ---
            # Surrogate trainiert auf den Train-Predictions des Champions (Full-Data-Refit),
            # um die gelernte CATE-Funktion bestmöglich nachzulernen.
            # Cross-Validation erzeugt OOS-Predictions für faire Vergleichbarkeit mit dem Champion.
            if is_mt:
                mt_cols_train = sorted([c for c in champion_df.columns if c.startswith(f"Train_{teacher_name}_T")])
                mt_cols_pred = sorted([c for c in champion_df.columns if c.startswith(f"Predictions_{teacher_name}_T")])
                # Fallback auf OOF-Predictions wenn Train-Predictions nicht verfügbar
                if mt_cols_train and not np.all(np.isnan(champion_df[mt_cols_train[0]].to_numpy(dtype=float))):
                    target_2d = champion_df[mt_cols_train].to_numpy()
                    self._logger.info("Surrogate trainiert auf Train-Predictions (Full-Data-Refit) des Champions.")
                else:
                    target_2d = champion_df[mt_cols_pred].to_numpy()
                    self._logger.info("Surrogate: Train-Predictions nicht verfügbar, verwende OOF-Predictions.")
                n_effects = target_2d.shape[1]

                surrogate_preds = np.full_like(target_2d, np.nan, dtype=float)
                final_trees = {}
                for k in range(n_effects):
                    target_k = target_2d[:, k]
                    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    for tr_idx, va_idx in cv.split(X):
                        tree_k = self._build_surrogate_regressor(cfg)
                        tree_k.fit(X.iloc[tr_idx], target_k[tr_idx])
                        surrogate_preds[va_idx, k] = tree_k.predict(X.iloc[va_idx])

                    final_tree_k = self._build_surrogate_regressor(cfg)
                    final_tree_k.fit(X, target_k)
                    final_trees[k] = final_tree_k

                surrogate_df = pd.DataFrame({"Y": Y, "T": T})
                for k in range(n_effects):
                    surrogate_df[f"Predictions_{sname}_T{k+1}"] = surrogate_preds[:, k]
                    surrogate_df[f"Train_{sname}_T{k+1}"] = final_trees[k].predict(X)
                best_eff = np.nanmax(surrogate_preds, axis=1)
                best_arm = np.nanargmax(surrogate_preds, axis=1) + 1
                surrogate_df[f"OptimalTreatment_{sname}"] = np.where(best_eff > 0, best_arm, 0)

                final_tree = None  # Nicht genutzt bei MT
            else:
                train_col = f"Train_{teacher_name}"
                pred_col = f"Predictions_{teacher_name}"
                # Fallback auf OOF-Predictions wenn Train-Predictions nicht verfügbar
                if train_col in champion_df.columns and not np.all(np.isnan(champion_df[train_col].to_numpy(dtype=float))):
                    target = champion_df[train_col].to_numpy()
                    self._logger.info("Surrogate trainiert auf Train-Predictions (Full-Data-Refit) des Champions.")
                else:
                    target = champion_df[pred_col].to_numpy()
                    self._logger.info("Surrogate: Train-Predictions nicht verfügbar, verwende OOF-Predictions.")
                surrogate_preds = np.full_like(target, np.nan, dtype=float)
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                for tr_idx, va_idx in cv.split(X):
                    tree = self._build_surrogate_regressor(cfg)
                    tree.fit(X.iloc[tr_idx], target[tr_idx])
                    surrogate_preds[va_idx] = tree.predict(X.iloc[va_idx])

                final_tree = self._build_surrogate_regressor(cfg)
                final_tree.fit(X, target)
                final_trees = {}

                surrogate_df = pd.DataFrame({"Y": Y, "T": T})
                surrogate_df[f"Predictions_{sname}"] = surrogate_preds
                surrogate_df[f"Train_{sname}"] = final_tree.predict(X)

        else:
            # --- External-Eval-Modus ---
            champion_model = models[teacher_name]
            if teacher_name == "Ensemble":
                # Ensemble hat kein eigenes fit() — Predictions durch Mitteln der Einzelmodelle
                _ens_members = [n for n in models if n not in {"Ensemble", SURROGATE_MODEL_NAME}]
                _member_preds = [np.asarray(_predict_effect(models[n], X)) for n in _ens_members]
                train_target = np.mean(_member_preds, axis=0)
            else:
                train_target = np.asarray(_predict_effect(champion_model, X))

            X_test, T_test, Y_test, _ = holdout_data

            if is_mt and train_target.ndim == 2:
                n_effects = train_target.shape[1]
                final_trees = {}
                eval_preds = np.full((len(X_test), n_effects), np.nan, dtype=float)
                for k in range(n_effects):
                    tree_k = self._build_surrogate_regressor(cfg)
                    tree_k.fit(X, train_target[:, k])
                    eval_preds[:, k] = tree_k.predict(X_test)
                    final_trees[k] = tree_k
                final_tree = None

                surrogate_df = pd.DataFrame({"Y": Y_test, "T": T_test})
                for k in range(n_effects):
                    surrogate_df[f"Predictions_{sname}_T{k+1}"] = eval_preds[:, k]
                    surrogate_df[f"Train_{sname}_T{k+1}"] = np.nan
                best_eff = np.nanmax(eval_preds, axis=1)
                best_arm = np.nanargmax(eval_preds, axis=1) + 1
                surrogate_df[f"OptimalTreatment_{sname}"] = np.where(best_eff > 0, best_arm, 0)
            else:
                final_tree = self._build_surrogate_regressor(cfg)
                final_tree.fit(X, train_target.reshape(-1))
                eval_pred = final_tree.predict(X_test)
                final_trees = {}

                surrogate_df = pd.DataFrame({"Y": Y_test, "T": T_test})
                surrogate_df[f"Predictions_{sname}"] = eval_pred.reshape(-1)
                surrogate_df[f"Train_{sname}"] = np.nan

        surrogate_wrapper = SurrogateTreeWrapper(
            tree=final_tree, trees=final_trees, champion_name=teacher_name,
        )

        # Evaluation mit denselben Metriken
        # Bei eval_mask: Metriken nur auf Eval-Subset (konsistent mit Phase 1)
        use_mask = eval_mask is not None and holdout_data is None
        surr_eval_df = surrogate_df.loc[eval_mask].reset_index(drop=True) if use_mask else surrogate_df
        try:
            y_s = surr_eval_df["Y"].to_numpy()
            t_s = surr_eval_df["T"].to_numpy()
            if is_mt:
                mt_cols_s = [c for c in surr_eval_df.columns if c.startswith(f"Predictions_{sname}_T")]
                scores_2d = surr_eval_df[mt_cols_s].to_numpy()
                eval_summary[sname] = mt_eval_summary(y=y_s, t=t_s, scores_2d=scores_2d)
            else:
                s = surr_eval_df[f"Predictions_{sname}"].to_numpy()
                curve = uplift_curve(y=y_s, t=t_s, score=s)
                eval_summary[sname] = {
                    "qini": float(qini_coefficient(curve)),
                    "auuc": float(auuc(curve)),
                    "uplift_at_10pct": float(uplift_at_k(curve, k_fraction=0.10)),
                    "uplift_at_20pct": float(uplift_at_k(curve, k_fraction=0.20)),
                    "uplift_at_50pct": float(uplift_at_k(curve, k_fraction=0.50)),
                }
                if getattr(cfg, "study_type", "rct") == "rct":
                    eval_summary[sname]["policy_value"] = float(policy_value(y=y_s, t=t_s, score=s, threshold=0.0))
            for key, val in eval_summary[sname].items():
                if isinstance(val, (int, float)):
                    short = {"uplift_at_10pct": "uplift10", "uplift_at_20pct": "uplift20", "uplift_at_50pct": "uplift50"}.get(key, key)
                    mlflow.log_metric(f"{short}__{sname}", float(val))
                elif isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        mlflow.log_metric(f"{key}_{sub_key}__{sname}", float(sub_val))
            self._logger.info("Surrogate-Evaluation: %s", eval_summary[sname])
        except Exception:
            self._logger.warning("Surrogate-Evaluation fehlgeschlagen.", exc_info=True)

        # Surrogate CATE-Verteilungsplot
        try:
            import matplotlib.pyplot as plt
            if is_mt:
                mt_cols_s = [c for c in surrogate_df.columns if c.startswith(f"Predictions_{sname}_T")]
                for k, pc in enumerate(mt_cols_s):
                    arm = k + 1
                    fig_dist = generate_cate_distribution_plot(
                        cate_preds_val=surrogate_df[pc].to_numpy(dtype=float),
                        model_name=sname, arm_label=f"T{arm}",
                    )
                    if fig_dist is not None:
                        _log_figure_fast(mlflow, fig_dist, f"distribution__{sname}_T{arm}.png")
                        if hasattr(self, '_report'):
                            self._report.add_plot(f"{sname}_T{arm}", "cate_distribution", fig_dist)
                        plt.close(fig_dist)
            else:
                pred_col_s = f"Predictions_{sname}"
                if pred_col_s in surrogate_df.columns:
                    fig_dist = generate_cate_distribution_plot(
                        cate_preds_val=surrogate_df[pred_col_s].to_numpy(dtype=float),
                        model_name=sname,
                    )
                    if fig_dist is not None:
                        _log_figure_fast(mlflow, fig_dist, f"distribution__{sname}.png")
                        if hasattr(self, '_report'):
                            self._report.add_plot(sname, "cate_distribution", fig_dist)
                        plt.close(fig_dist)
        except Exception:
            self._logger.warning("Surrogate-CATE-Verteilungsplot fehlgeschlagen.", exc_info=True)

        # Surrogate sklift-Plots (Qini-Kurve, Uplift-by-Percentile, Treatment-Balance)
        try:
            import matplotlib.pyplot as plt
            y_s = surr_eval_df["Y"].to_numpy()
            t_s = surr_eval_df["T"].to_numpy()
            if is_mt:
                mt_cols_s = [c for c in surr_eval_df.columns if c.startswith(f"Predictions_{sname}_T")]
                K = len(np.unique(t_s))
                for k in range(1, K):
                    if k - 1 >= len(mt_cols_s):
                        continue
                    arm_scores = surr_eval_df[mt_cols_s[k - 1]].to_numpy(dtype=float)
                    arm_mask = (t_s == 0) | (t_s == k)
                    arm_cate = arm_scores[arm_mask]
                    arm_T_bin = (t_s[arm_mask] == k).astype(int)
                    arm_Y = y_s[arm_mask]
                    try:
                        sk_qini, sk_pct, sk_tb = generate_sklift_plots(arm_cate, arm_T_bin, arm_Y)
                        if getattr(cfg, "study_type", "rct") != "rct":
                            sk_tb = None
                        label = f"{sname}_T{k}"
                        for fig, key, fname in [
                            (sk_qini, "sklift_qini", f"sklift_qini__{sname}_T{k}.png"),
                            (sk_pct, "sklift_percentile", f"sklift_percentile__{sname}_T{k}.png"),
                            (sk_tb, "treatment_balance", f"treatment_balance__{sname}_T{k}.png"),
                        ]:
                            if fig is not None:
                                _log_figure_fast(mlflow, fig, fname)
                                if hasattr(self, '_report'):
                                    self._report.add_plot(label, key, fig)
                                plt.close(fig)
                    except Exception:
                        self._logger.warning("sklift-Plots für %s T%d fehlgeschlagen.", sname, k, exc_info=True)
            else:
                pred_col_s = f"Predictions_{sname}"
                if pred_col_s in surr_eval_df.columns:
                    cate_vals = surr_eval_df[pred_col_s].to_numpy(dtype=float)
                    try:
                        sk_qini, sk_pct, sk_tb = generate_sklift_plots(cate_vals, t_s, y_s)
                        if getattr(cfg, "study_type", "rct") != "rct":
                            sk_tb = None
                        for fig, key, fname in [
                            (sk_qini, "sklift_qini", f"sklift_qini__{sname}.png"),
                            (sk_pct, "sklift_percentile", f"sklift_percentile__{sname}.png"),
                            (sk_tb, "treatment_balance", f"treatment_balance__{sname}.png"),
                        ]:
                            if fig is not None:
                                _log_figure_fast(mlflow, fig, fname)
                                if hasattr(self, '_report'):
                                    self._report.add_plot(sname, key, fig)
                                plt.close(fig)
                    except Exception:
                        self._logger.warning("sklift-Plots für %s fehlgeschlagen.", sname, exc_info=True)
        except Exception:
            self._logger.warning("Surrogate-sklift-Plots fehlgeschlagen.", exc_info=True)

        # Baumtiefe und Blattanzahl loggen
        try:
            log_tree = final_tree if final_tree is not None else (final_trees.get(0) if final_trees else None)
            if log_tree is not None:
                depth, n_leaves = self._log_surrogate_tree_info(log_tree, base_type)
                if depth is not None:
                    mlflow.log_param("surrogate_tree_depth", int(depth))
                if n_leaves is not None:
                    mlflow.log_param("surrogate_tree_n_leaves", int(n_leaves))
            if is_mt:
                mlflow.log_param("surrogate_tree_n_arms", len(final_trees))
            mlflow.log_param("surrogate_tree_base_type", base_type)
            mlflow.log_param("surrogate_tree_teacher", teacher_name)
        except Exception:
            pass

        # ── DRTester-Diagnostik-Plots (nur wenn run_drtester=True) ──
        if run_drtester and not is_mt:
            try:
                import matplotlib.pyplot as plt
                pred_col_s = f"Predictions_{sname}"
                if pred_col_s in surrogate_df.columns:
                    cate_vals = surr_eval_df[pred_col_s].to_numpy(dtype=float)
                    y_s = surr_eval_df["Y"].to_numpy()
                    t_s = surr_eval_df["T"].to_numpy()
                    eval_X = holdout_data[0] if holdout_data is not None else (X.loc[eval_mask].reset_index(drop=True) if use_mask else X)

                    ft_surr = fitted_tester
                    if use_mask and fitted_tester is not None:
                        ft_surr = filter_tester_for_mask(fitted_tester, eval_mask, len(eval_X))

                    if ft_surr is not None:
                        bundle_s = evaluate_cate_with_plots(
                            fitted_tester=ft_surr,
                            X_val=eval_X, T_val=t_s, Y_val=y_s,
                            cate_preds_val=cate_vals,
                            X_train=X if holdout_data is not None else None,
                            T_train=T if holdout_data is not None else None,
                            Y_train=Y if holdout_data is not None else None,
                            cate_preds_train=None, n_groups=10,
                            n_bootstrap=getattr(self, '_eval_n_bootstrap', 1000),
                            seed=cfg.constants.random_seed,
                        )
                    else:
                        pj = 1 if cfg.constants.parallel_level <= 1 else -1
                        model_reg = build_base_learner(cfg.base_learner.type, {}, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=pj)
                        model_prop = build_base_learner(cfg.base_learner.type, {}, seed=cfg.constants.random_seed, task="classifier", parallel_jobs=pj)
                        bundle_s = evaluate_cate_with_plots(
                            model_regression=model_reg, model_propensity=model_prop,
                            X_val=eval_X, T_val=t_s, Y_val=y_s,
                            cate_preds_val=cate_vals,
                            X_train=X, T_train=T, Y_train=Y,
                            cate_preds_train=None, n_groups=10,
                            n_bootstrap=getattr(self, '_eval_n_bootstrap', 1000),
                            seed=cfg.constants.random_seed,
                        )

                    if bundle_s is not None:
                        bundle_s.log_to_mlflow(mlflow, sname, log_temp_artifact_fn=_log_temp_artifact)
                        if hasattr(self, '_report'):
                            bundle_s.add_to_report(self._report, sname)
                        bundle_s.close_figures()
                        self._logger.info("DRTester-Plots für %s erzeugt.", sname)
            except Exception:
                self._logger.warning("DRTester-Plots für %s fehlgeschlagen.", sname, exc_info=True)

        return surrogate_wrapper, surrogate_df

    def _run_bundle_export(self, cfg, models, eval_summary, X, T, Y, X_full, T_full, Y_full, selected_feature_columns, holdout_data, export_bundle, bundle_dir, bundle_id, mlflow):
        """Synchroner Bundle-Export am Ende des Analyselaufs."""
        bundle_cfg = getattr(cfg, "bundle", None)
        export_bundle_effective = bool(bundle_cfg.enabled) if bundle_cfg is not None else False
        if export_bundle is not None:
            export_bundle_effective = bool(export_bundle)
        if not export_bundle_effective:
            return

        bundle_base_dir = bundle_dir or (getattr(bundle_cfg, "base_dir", None) or "runs/bundles")
        bundle_id_effective = bundle_id or getattr(bundle_cfg, "bundle_id", None)
        bundle_include_challengers = bool(getattr(bundle_cfg, "include_challengers", True))
        bundle_log_to_mlflow = bool(getattr(bundle_cfg, "log_to_mlflow", True))

        bundler = ArtifactBundler(base_dir=bundle_base_dir)
        paths = bundler.create_bundle_dir(bundle_id=bundle_id_effective)
        bundler.write_config(paths, config_path=cfg.source_config_path)
        preproc = build_simple_preprocessor_from_dataframe(X)
        bundler.write_preprocessor(paths, preproc)

        # Champion-Auswahl: SurrogateTree wird nicht als Kandidat berücksichtigt.
        _surr_names = {SURROGATE_MODEL_NAME}
        cate_model_names = [n for n in models.keys() if n not in _surr_names]
        entries = [
            ModelEntry(name=mname, artifact_path=f"models/{mname}.pkl", metrics=float_metrics(eval_summary.get(mname, {}) or {}))
            for mname in cate_model_names
        ]
        manual_champion = (getattr(cfg.selection, "manual_champion", None) or "").strip() or None
        selection_cfg = {"metric": cfg.selection.metric, "higher_is_better": cfg.selection.higher_is_better, "manual_champion": manual_champion}
        if manual_champion is not None:
            champion = manual_champion
        else:
            champion = choose_champion(entries, metric=cfg.selection.metric, higher_is_better=cfg.selection.higher_is_better)
            if champion is None and entries:
                available_metrics = sorted(set(k for e in entries for k in e.metrics.keys()))
                self._logger.warning("Champion-Auswahl: Metrik '%s' nicht gefunden. Verfügbar: %s. Fallback auf erstes Modell.", cfg.selection.metric, available_metrics)
                champion = entries[0].name

        models_to_export = {n: m for n, m in models.items() if n not in _surr_names}
        registry_entries = list(entries)
        if not bundle_include_challengers and champion is not None:
            models_to_export = {champion: models[champion]}
            registry_entries = [e for e in entries if e.name == champion]

        champion_refit_on_full_data = False
        champion_refit_rows = None
        champion_fitted_obj = None
        _refit_full = bool(getattr(cfg.selection, "refit_champion_on_full_data", True))
        _ensemble_is_champion = champion == "Ensemble"
        X_refit = X_full.loc[:, selected_feature_columns].copy()

        # Wenn das Ensemble Champion ist und Refit aktiviert: ALLE Einzelmodelle
        # auf vollen Daten refitten, damit das Ensemble in Production die
        # bestmöglichen Vorhersagen liefert (pass-by-reference in EnsembleCateEstimator).
        _refit_all_for_ensemble = _ensemble_is_champion and _refit_full and not holdout_data

        for mname, mobj in models_to_export.items():
            obj_to_write = mobj
            _is_ensemble = mname == "Ensemble"

            if _is_ensemble:
                # EnsembleCateEstimator hat kein fit() — Refit überspringen.
                # Die Referenzen auf die Einzelmodelle werden unten aktualisiert.
                if mname == champion:
                    champion_fitted_obj = obj_to_write
            elif mname == champion and _refit_full:
                # Normaler Champion-Refit auf vollen Daten
                try:
                    obj_to_write = copy.deepcopy(mobj)
                except Exception:
                    self._logger.warning("deepcopy des Champion-Modells fehlgeschlagen, verwende Original.", exc_info=True)
                    obj_to_write = mobj
                obj_to_write.fit(Y_full, T_full, X=X_refit)
                champion_refit_on_full_data = True
                champion_refit_rows = int(len(X_refit))
                champion_fitted_obj = obj_to_write
            elif _refit_all_for_ensemble:
                # Ensemble ist Champion → alle Einzelmodelle auf vollen Daten refitten
                try:
                    obj_to_write = copy.deepcopy(mobj)
                except Exception:
                    self._logger.warning("deepcopy des Modells %s fehlgeschlagen.", mname, exc_info=True)
                    obj_to_write = mobj
                obj_to_write.fit(Y_full, T_full, X=X_refit)
                self._logger.info("Ensemble-Refit: %s auf vollen Daten refittet (%d Zeilen).", mname, len(X_refit))
            else:
                if not holdout_data:
                    try:
                        obj_to_write = copy.deepcopy(mobj)
                    except Exception:
                        self._logger.warning("deepcopy des Challenger-Modells %s fehlgeschlagen.", mname, exc_info=True)
                        obj_to_write = mobj
                    X_fit = X.loc[:, selected_feature_columns].copy()
                    obj_to_write.fit(Y, T, X=X_fit)
                if mname == champion:
                    champion_fitted_obj = obj_to_write

            # Refittetes Modell im models_to_export dict aktualisieren
            # (wichtig für Ensemble-Referenzen)
            models_to_export[mname] = obj_to_write
            bundler.write_model(paths, mname, obj_to_write)

        # Ensemble nach Refit der Einzelmodelle neu erstellen
        # (damit die Referenzen auf die refitteten Modelle zeigen)
        if _refit_all_for_ensemble and "Ensemble" in models_to_export:
            try:
                from econml.score import EnsembleCateEstimator
                _ens_members = [m for n, m in models_to_export.items() if n != "Ensemble"]
                _ens_weights = np.ones(len(_ens_members)) / len(_ens_members)
                ensemble_refitted = EnsembleCateEstimator(cate_models=_ens_members, weights=_ens_weights)
                bundler.write_model(paths, "Ensemble", ensemble_refitted)
                champion_fitted_obj = ensemble_refitted
                champion_refit_on_full_data = True
                champion_refit_rows = int(len(X_refit))
                self._logger.info(
                    "Ensemble-Champion: Neues EnsembleCateEstimator mit %d refitteten Modellen erstellt.",
                    len(_ens_members),
                )
            except Exception:
                self._logger.warning("Ensemble-Rebuild nach Refit fehlgeschlagen.", exc_info=True)

        # Surrogate-Einzelbaum für das Bundle:
        # Wird auf den (ggf. refitteten) Champion-Predictions trainiert.
        # Bei MT wird pro Treatment-Arm ein separater Baum trainiert.
        surrogate_exported = False
        if cfg.surrogate_tree.enabled and champion_fitted_obj is not None:
            try:
                if champion_refit_on_full_data:
                    X_surr = X_full.loc[:, selected_feature_columns].copy()
                else:
                    X_surr = X.loc[:, selected_feature_columns].copy()

                if _ensemble_is_champion:
                    # Ensemble: Predictions durch Mitteln der refitteten Einzelmodelle
                    _ens_members_export = {n: m for n, m in models_to_export.items() if n != "Ensemble"}
                    _member_preds = [np.asarray(_predict_effect(m, X_surr)) for m in _ens_members_export.values()]
                    champion_preds_for_surr = np.mean(_member_preds, axis=0)
                else:
                    champion_preds_for_surr = _predict_effect(champion_fitted_obj, X_surr)
                champion_preds_for_surr = np.asarray(champion_preds_for_surr)

                if champion_preds_for_surr.ndim == 2 and champion_preds_for_surr.shape[1] > 1:
                    # MT: pro Arm einen Baum trainieren
                    n_effects = champion_preds_for_surr.shape[1]
                    bundle_trees = {}
                    for k in range(n_effects):
                        tree_k = self._build_surrogate_regressor(cfg)
                        tree_k.fit(X_surr, champion_preds_for_surr[:, k])
                        bundle_trees[k] = tree_k
                    bundle_surrogate = SurrogateTreeWrapper(trees=bundle_trees, champion_name=champion)
                    log_tree = bundle_trees.get(0)
                else:
                    bundle_tree = self._build_surrogate_regressor(cfg)
                    bundle_tree.fit(X_surr, champion_preds_for_surr.reshape(-1))
                    bundle_surrogate = SurrogateTreeWrapper(tree=bundle_tree, champion_name=champion)
                    log_tree = bundle_tree

                bundler.write_model(paths, SURROGATE_MODEL_NAME, bundle_surrogate)
                surrogate_exported = True

                # Surrogate in Registry aufnehmen (ohne Champion-Konkurrenz)
                surrogate_entry = ModelEntry(
                    name=SURROGATE_MODEL_NAME,
                    artifact_path=f"models/{SURROGATE_MODEL_NAME}.pkl",
                    metrics=float_metrics(eval_summary.get(SURROGATE_MODEL_NAME, {}) or {}),
                )
                registry_entries.append(surrogate_entry)

                # Surrogate wurde mit Fallback "both"→"catboost" trainiert; gleiche Logik hier.
                _surr_bt = cfg.base_learner.type if (cfg.base_learner.type or "").lower() != "both" else "catboost"
                depth, n_leaves = self._log_surrogate_tree_info(log_tree, _surr_bt) if log_tree else (None, None)
                self._logger.info(
                    "Surrogate-Einzelbaum exportiert (Typ=%s, Tiefe=%s, Blätter=%s, trainiert auf %d Zeilen).",
                    _surr_bt, depth, n_leaves, len(X_surr),
                )
            except Exception:
                self._logger.warning("Surrogate-Tree Bundle-Export fehlgeschlagen.", exc_info=True)

        # Surrogate-Modelle: nur Champion-Surrogate im Bundle
        all_surrogate_names = []
        if surrogate_exported:
            all_surrogate_names.append(SURROGATE_MODEL_NAME)

        write_registry(paths.root, entries=registry_entries, champion=champion, selection=selection_cfg)

        bundler.write_metadata(paths, {
            "models": list(models_to_export.keys()) + all_surrogate_names,
            "champion": champion,
            "treatment_type": cfg.treatment.type,
            "n_treatment_arms": int(len(np.unique(T_full))),
            "reference_group": cfg.treatment.reference_group,
            "selection_metric": cfg.selection.metric, "selection_manual_champion": manual_champion,
            "created_from_run": True, "champion_refit_on_full_data": champion_refit_on_full_data,
            "champion_refit_rows": champion_refit_rows, "selected_feature_columns": selected_feature_columns,
            "bundle_include_challengers": bundle_include_challengers,
            "surrogate_tree_enabled": surrogate_exported,
        })
        if bundle_log_to_mlflow:
            mlflow.log_artifacts(str(paths.root), artifact_path=f"bundle_{paths.root.name}")
        self._logger.info("Bundle gespeichert: %s (%d Modelle, Champion=%s)", paths.root, len(models_to_export), champion)

    def _run_optional_output(self, cfg, eval_summary, removed, preds):
        """Optionale lokale Ausgaben (zusätzlich zu MLflow)."""
        if not cfg.optional_output.output_dir:
            return
        out_dir = cfg.optional_output.output_dir
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "removed_features.json"), "w", encoding="utf-8") as f:
            json.dump(removed, f, ensure_ascii=False, indent=2)
        if cfg.optional_output.save_predictions:
            fmt = (cfg.optional_output.predictions_format or "parquet").lower()
            max_rows = cfg.optional_output.max_prediction_rows
            for model_name, df_pred in preds.items():
                df_out = df_pred if max_rows is None or len(df_pred) <= max_rows else df_pred.iloc[:max_rows].copy()
                if fmt == "parquet":
                    try:
                        df_out.to_parquet(os.path.join(out_dir, f"predictions_{model_name}.parquet"), index=False)
                    except Exception:
                        self._logger.warning("Parquet-Export fehlgeschlagen, Fallback auf CSV.", exc_info=True)
                        df_out.to_csv(os.path.join(out_dir, f"predictions_{model_name}.csv"), index=False, float_format="%.10g")
                else:
                    df_out.to_csv(os.path.join(out_dir, f"predictions_{model_name}.csv"), index=False, float_format="%.10g")

    # ------------------------------------------------------------------
    # Explainability (SHAP)
    # ------------------------------------------------------------------

    def _run_explainability(self, cfg, X, T, Y, models, eval_summary, mlflow, report, holdout_data=None, fold_models=None):
        """SHAP-Analyse für den Champion.

        Out-of-Sample: SHAP-Werte werden immer auf Daten berechnet, die das
        Modell nie gesehen hat:
        - CV-Modus: Nutzt das letzte CV-Fold-Modell (bereits trainiert auf K-1
          Folds) und sampelt aus dessen Validation-Indices.
        - External/Mask-Modus: Nutzt das gefittete Modell und Eval-Daten.
        """
        shap_cfg = cfg.shap_values

        if not shap_cfg.calculate_shap_values:
            self._logger.info("Explainability: Deaktiviert (calculate_shap_values=false).")
            return

        # Champion bestimmen
        champion_name = self._determine_champion(cfg, eval_summary, models)
        if not champion_name or champion_name not in models:
            self._logger.warning("Explainability: Kein Champion-Modell gefunden, überspringe.")
            return

        # Ensemble-Fallback: SHAP auf dem besten Einzelmodell statt dem Ensemble.
        # EnsembleCateEstimator hat kein natives SHAP — KernelSHAP wäre langsam und
        # verrauscht. Das beste Einzelmodell liefert klarere, interpretierbare Erklärungen.
        if champion_name == "Ensemble" and eval_summary:
            sel_met = cfg.selection.metric
            higher = cfg.selection.higher_is_better
            _individual = [
                (n, eval_summary[n].get(sel_met, float("-inf") if higher else float("inf")))
                for n in eval_summary
                if n not in ("Ensemble", SURROGATE_MODEL_NAME) and n in models
                and isinstance(eval_summary[n].get(sel_met), (int, float))
            ]
            if _individual:
                _individual.sort(key=lambda x: x[1], reverse=higher)
                best_individual = _individual[0][0]
                self._logger.info(
                    "Explainability: Ensemble ist Champion → verwende bestes Einzelmodell '%s' "
                    "(Metrik %s = %.6g) für SHAP-Analyse.",
                    best_individual, sel_met, _individual[0][1],
                )
                champion_name = best_individual

        seed = cfg.constants.random_seed
        from rubin.training import _predict_effect
        fold_models = fold_models or {}

        # ── Modell + Out-of-Sample-Daten bestimmen ──
        if champion_name in fold_models:
            model, val_idx = fold_models[champion_name]
            if val_idx is not None:
                # CV-Modus: Letztes Fold-Modell, OOF-Samples
                X_expl = X.iloc[val_idx].copy()
                self._logger.info(
                    "Explainability für Champion '%s': CV-Fold-Modell (bereits trainiert), "
                    "%d out-of-fold Samples verfügbar.",
                    champion_name, len(X_expl),
                )
            else:
                # Fallback: Fold-Modell ohne val_idx → alle Daten verwenden
                X_expl = X.copy()
                self._logger.info(
                    "Explainability für Champion '%s': Fold-Modell ohne val_idx, "
                    "alle %d Samples verfügbar.",
                    champion_name, len(X_expl),
                )
        elif holdout_data is not None:
            # External / Eval-Mask: Eval-Daten als Explain-Set
            model = models[champion_name]
            X_expl = holdout_data[0]  # X_eval
            # Modell fitten falls nötig (im External-Modus schon gefittet)
            try:
                _predict_effect(model, X.iloc[:1])
            except (AttributeError, Exception):
                model.fit(Y, T, X=X)
            self._logger.info(
                "Explainability für Champion '%s': %d Eval-Samples (out-of-sample).",
                champion_name, len(X_expl),
            )
        else:
            # Fallback: kein Fold-Modell, kein Holdout → Fit + stratifizierter Split
            model = models[champion_name]
            from sklearn.model_selection import StratifiedShuffleSplit
            strat = np.asarray(T).astype(int) * 10 + np.asarray(Y).astype(int)
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            train_idx, expl_idx = next(splitter.split(X, strat))
            try:
                model.fit(Y[train_idx], T[train_idx], X=X.iloc[train_idx])
            except TypeError:
                model.fit(Y[train_idx], T[train_idx], X.iloc[train_idx])
            X_expl = X.iloc[expl_idx].copy()
            self._logger.info(
                "Explainability für Champion '%s': Fallback 80/20-Split, "
                "%d out-of-sample Samples.",
                champion_name, len(X_expl),
            )

        # Sampling auf n_shap_values
        if shap_cfg.n_shap_values and len(X_expl) > shap_cfg.n_shap_values:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X_expl), size=shap_cfg.n_shap_values, replace=False)
            X_expl = X_expl.iloc[idx].copy()

        self._logger.info("Explainability: %d Samples für SHAP.", len(X_expl))
        mlflow.log_param("explainability_model", champion_name)
        report.explainability_info["model_name"] = champion_name

        # Uplift-Predictions (out-of-sample)
        raw_uplift = np.asarray(_predict_effect(model, X_expl))
        if raw_uplift.ndim == 2 and raw_uplift.shape[1] > 1:
            uplift = np.max(raw_uplift, axis=1)
        else:
            uplift = raw_uplift.reshape(-1)

        top_n = shap_cfg.top_n_features
        num_bins = shap_cfg.num_bins

        # ── SHAP-Analyse ──
        try:
            from rubin.explainability import (
                shap_available, build_shap_plots, compute_shap_for_uplift,
            )
            from rubin.explainability.reporting import save_importance_barplot

            if not shap_available():
                raise ImportError("SHAP nicht installiert")

            try:
                # Vollständiger EconML-kompatibler SHAP-Plot-Satz
                shap_result = build_shap_plots(
                    model=model, X=X_expl, data=X_expl, cate=uplift,
                    top_n=top_n, num_bins=num_bins,
                )
                for plot_key, display_name, fig in [
                    ("summary", "SHAP Summary", shap_result.summary),
                    ("cate_profiles", "CATE-Profile", shap_result.cate_profiles),
                    ("dependence", "SHAP Dependence", shap_result.shap_dependence),
                    ("scatter", "SHAP Scatter", shap_result.shap_scatter),
                ]:
                    report.add_explainability_plot(display_name, fig)
                    def _save_plot(p, _fig=fig):
                        _fig.savefig(p, dpi=160, bbox_inches="tight")
                        import matplotlib.pyplot as plt
                        plt.close(_fig)
                    _log_temp_artifact(mlflow, _save_plot, f"SHAP_{plot_key}_{champion_name}.png")

                def _save_imp(p, _imp=shap_result.importance):
                    _imp.head(top_n).to_csv(p)
                _log_temp_artifact(mlflow, _save_imp, f"shap_importance_{champion_name}.csv")

                mlflow.log_param("explainability_method", "shap_plots")
                self._logger.info("SHAP-Plots für '%s' berechnet (%d Features).", champion_name, top_n)

            except Exception:
                self._logger.info("SHAP-Plot-Satz fehlgeschlagen, Fallback auf generische SHAP-Werte.", exc_info=True)
                from rubin.explainability import build_generic_shap_plots
                res = compute_shap_for_uplift(model=model, X=X_expl, seed=seed)
                imp = res.mean_abs_importance()

                def _save_imp2(p, _imp=imp):
                    _imp.head(top_n).to_csv(p)
                _log_temp_artifact(mlflow, _save_imp2, f"shap_importance_{champion_name}.csv")

                try:
                    # Generischer SHAP-Plot-Satz (funktioniert auch mit CausalForestDML)
                    shap_result = build_generic_shap_plots(
                        shap_result=res, X=X_expl, cate=uplift,
                        top_n=top_n, num_bins=num_bins,
                    )
                    for plot_key, display_name, fig in [
                        ("summary", "SHAP Summary", shap_result.summary),
                        ("cate_profiles", "CATE-Profile", shap_result.cate_profiles),
                        ("dependence", "SHAP Dependence", shap_result.shap_dependence),
                        ("scatter", "SHAP Scatter", shap_result.shap_scatter),
                    ]:
                        report.add_explainability_plot(display_name, fig)
                        def _save_plot2(p, _fig=fig):
                            _fig.savefig(p, dpi=160, bbox_inches="tight")
                            import matplotlib.pyplot as plt
                            plt.close(_fig)
                        _log_temp_artifact(mlflow, _save_plot2, f"SHAP_{plot_key}_{champion_name}.png")
                    self._logger.info("Generische SHAP-Plots für '%s' berechnet (%d Features).", champion_name, top_n)

                except Exception:
                    self._logger.info("Generischer SHAP-Plot-Satz fehlgeschlagen, nur Barplot.", exc_info=True)
                    def _save_bar2(p, _imp=imp):
                        save_importance_barplot(_imp, p, top_n=top_n, title=f"SHAP-Importance – {champion_name}")
                    _log_temp_artifact(mlflow, _save_bar2, f"shap_importance_{champion_name}.png")

                    # Barplot für HTML-Report
                    import matplotlib.pyplot as plt
                    _imp_plot = imp.head(top_n)[::-1]
                    _fig_bar, _ax = plt.subplots(figsize=(10, max(4, 0.25 * len(_imp_plot) + 2)))
                    _ax.barh(_imp_plot.index.astype(str), _imp_plot.values)
                    _ax.set_xlabel("Wichtigkeit"); _ax.set_title(f"SHAP-Importance – {champion_name}")
                    _fig_bar.tight_layout()
                    report.add_explainability_plot("SHAP-Importance", _fig_bar)
                    plt.close(_fig_bar)

                mlflow.log_param("explainability_method", "shap_generic")
                self._logger.info("Generische SHAP-Importance für '%s' berechnet.", champion_name)

        except Exception:
            self._logger.warning("SHAP-Analyse fehlgeschlagen.", exc_info=True)

    # ------------------------------------------------------------------
    # Hauptmethode
    # ------------------------------------------------------------------

    def run(self, export_bundle: bool | None = None, bundle_dir: str | None = None, bundle_id: str | None = None) -> AnalysisResult:
        """Startet den Analyselauf."""
        cfg = self.cfg

        # ── Harmlose sklearn-Warnung unterdrücken ──
        # EconML übergibt X intern mal als DataFrame (mit Spaltennamen), mal als
        # numpy-Array (ohne Namen). sklearn warnt dann bei predict(), obwohl die
        # Feature-Reihenfolge garantiert ist. Die Warnung ist rein kosmetisch und
        # würde das Log mit hunderten identischen Zeilen fluten.
        import warnings
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module=r"sklearn\.utils\.validation",
        )
        # MLflow filesystem-backend Deprecation-Warnung unterdrücken.
        # Die Migration zu SQLite/DB-Backend ist eine Infrastrukturentscheidung
        # und kein Handlungsbedarf für den einzelnen Analyselauf.
        warnings.filterwarnings(
            "ignore",
            message=".*filesystem tracking backend.*is deprecated",
            category=FutureWarning,
            module=r"mlflow",
        )
        # EconML nutzt intern den alten sklearn-Parameter 'force_all_finite',
        # der in sklearn 1.6 zu 'ensure_all_finite' umbenannt wurde. Die Warnung
        # ist rein kosmetisch — EconML wird das in einer zukünftigen Version
        # aktualisieren. Bis dahin unterdrücken wir die hundertfache Wiederholung.
        warnings.filterwarnings(
            "ignore",
            message=".*force_all_finite.*was renamed to.*ensure_all_finite",
            category=FutureWarning,
            module=r"sklearn",
        )
        # sklearn DataConversionWarning: "A column-vector y was passed when a 1d array
        # was expected." Tritt bei EconML-internen fit()-Aufrufen hundertfach auf.
        # Harmlos: ravel() wird automatisch angewendet.
        warnings.filterwarnings(
            "ignore",
            message=".*column-vector y was passed.*",
            category=UserWarning,
            module=r"sklearn",
        )
        warnings.filterwarnings(
            "ignore",
            message=".*column-vector y was passed.*",
            category=FutureWarning,
        )
        try:
            from sklearn.exceptions import DataConversionWarning
            warnings.filterwarnings("ignore", category=DataConversionWarning)
        except ImportError:
            pass
        # Optuna ExperimentalWarning: multivariate und constant_liar sind als
        # "experimental" markiert, aber seit Optuna 2.x stabil. Die Warnung
        # erscheint pro Study und flutet das Log unnötig.
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"optuna",
        )
        try:
            from optuna.exceptions import ExperimentalWarning
            warnings.filterwarnings("ignore", category=ExperimentalWarning)
        except ImportError:
            pass
        # EconML DRTester: Division-by-Zero und Invalid-Value Warnungen aus internen
        # Statistik-Berechnungen (cal_r_squared, TOC-Bootstrap, BLP p-values).
        # Numerisch harmlos: NaN/Inf werden von unserer _sanitize_dr abgefangen.
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"econml\.validate",
        )

        try:
            import mlflow
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("MLflow ist nicht installiert. Für Analyseläufe wird MLflow benötigt (pip install mlflow).") from e

        # ── Globale Seeds setzen für maximale Reproduzierbarkeit ──
        # Neben den Konstruktor-Seeds (random_state=...) nutzen einige Bibliotheken
        # (EconML GRF, sklearn, numpy) den globalen Random-State. Ohne diese Zeilen
        # können Feature-Importances und Subsampling bei gleichen Daten variieren.
        # Hinweis: Bei parallel_level > 1 (n_jobs > 1) bleibt Thread-Scheduling
        # nicht-deterministisch — volle Reproduzierbarkeit nur mit parallel_level=1.
        _seed = cfg.constants.random_seed
        np.random.seed(_seed)
        import random
        random.seed(_seed)

        # ── Schritte zählen ──
        total = 5  # load, fs, tune, train, eval
        if getattr(cfg, "final_model_tuning", None) is not None and cfg.final_model_tuning.enabled:
            total += 1  # fmt
        # GRF-Tuning als eigene Phase (vor dem Haupt-Training), wenn aktiviert
        _grf_tune_active = (
            any(n in (cfg.models.models_to_train or []) for n in ("CausalForestDML", "CausalForest"))
            and getattr(cfg.causal_forest, "use_econml_tune", False)
        )
        if _grf_tune_active:
            total += 1  # grf
        if cfg.surrogate_tree.enabled:
            total += 1
        bundle_enabled = export_bundle if export_bundle is not None else cfg.bundle.enabled
        if bundle_enabled:
            total += 1
        if cfg.shap_values.calculate_shap_values:
            total += 1
        total += 1  # report
        step = [0]
        step_times = {}
        _last_step_start = [time.perf_counter()]
        _last_step_label = [None]

        def _progress(label: str):
            now = time.perf_counter()
            if _last_step_label[0] is not None:
                step_times[_last_step_label[0]] = now - _last_step_start[0]
            _last_step_start[0] = now
            _last_step_label[0] = label
            step[0] += 1
            msg = f"[rubin] Step {step[0]}/{total}: {label}"
            print(msg, flush=True)

        # ── Arbeitsverzeichnis (work_dir) auflösen ──
        # Alle erzeugten Artefakte (MLflow, Report, Cache) landen hier.
        # Priorität: RUBIN_WORK_DIR (Env) > constants.work_dir (Config) > ./runs
        work_dir = cfg.constants.resolved_work_dir
        os.makedirs(work_dir, exist_ok=True)
        self._work_dir = work_dir
        self._logger.info("Arbeitsverzeichnis: %s", work_dir)

        # MLflow Tracking: Alle Artefakte landen konsolidiert unter work_dir.
        # Falls MLFLOW_TRACKING_URI gesetzt ist, hat diese Vorrang (z.B. Remote-Server).
        # Sonst: lokales Tracking unter {work_dir}/mlruns.
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow_dir = os.path.join(work_dir, "mlruns")
            mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_dir)}")

        mlflow.set_experiment(cfg.mlflow.experiment_name)

        # Prüfe ob DataPrep ein Experiment gespeichert hat (im Verzeichnis von x_file).
        # Falls ja und die Config noch den Default "rubin" verwendet, übernimm den
        # DataPrep-Experimentnamen, damit DataPrep- und Analyse-Runs im gleichen
        # Experiment landen.
        from pathlib import Path
        try:
            x_dir = Path(cfg.data_files.x_file).resolve().parent
            dp_exp_file = x_dir / ".mlflow_experiment"
            if dp_exp_file.is_file():
                dp_exp = dp_exp_file.read_text(encoding="utf-8").strip()
                if dp_exp and dp_exp == cfg.mlflow.experiment_name:
                    self._logger.info("MLflow-Experiment '%s' (identisch mit DataPrep).", dp_exp)
                elif dp_exp and cfg.mlflow.experiment_name == "rubin":
                    # Config hat den Default → DataPrep-Experiment übernehmen
                    mlflow.set_experiment(dp_exp)
                    self._logger.info(
                        "MLflow-Experiment '%s' aus DataPrep übernommen (Config hatte Default 'rubin').",
                        dp_exp,
                    )
                elif dp_exp:
                    self._logger.info(
                        "DataPrep-Experiment '%s', Config nutzt '%s'. Verwende Config-Wert.",
                        dp_exp, cfg.mlflow.experiment_name,
                    )
        except Exception:
            pass

        # Run-Name: Wenn DataPrep einen Run-Namen erzeugt hat, denselben Suffix
        # verwenden (z.B. DataPrep "Datenaufbereitung – roter-falke" → Analyse "Analyse – roter-falke").
        # Bei User-eingegebenen Namen (z.B. "mein-test") wird dieser als Suffix übernommen.
        # So sind zusammengehörige Runs sofort erkennbar.
        from rubin.utils.run_names import generate_run_name
        _run_name = generate_run_name("Analyse")
        try:
            dp_run_file = x_dir / ".mlflow_run_name"
            if dp_run_file.is_file():
                dp_run = dp_run_file.read_text(encoding="utf-8").strip()
                if dp_run:
                    # Auto-generiert: "Datenaufbereitung – roter-falke" → Suffix "roter-falke"
                    # User-eingegeben: "mein-test" → Suffix "mein-test"
                    suffix = dp_run.split(" – ", 1)[1] if " – " in dp_run else dp_run
                    _run_name = f"Analyse – {suffix}"
                    self._logger.info("Run-Name-Suffix '%s' aus DataPrep übernommen.", suffix)
        except Exception:
            pass

        with mlflow.start_run(run_name=_run_name):
            mlflow.log_param("seed", cfg.constants.random_seed)
            mlflow.log_param("base_learner", cfg.base_learner.type)
            mlflow.log_param("parallel_level", cfg.constants.parallel_level)
            mlflow.log_param("study_type", getattr(cfg, "study_type", "rct"))
            mlflow.log_param("cross_validation_splits", cfg.data_processing.cross_validation_splits)
            mlflow.log_param("dml_crossfit_folds", cfg.data_processing.dml_crossfit_folds)

            # ── Config-YAML nach MLflow loggen ──
            if cfg.source_config_path and os.path.isfile(cfg.source_config_path):
                try:
                    mlflow.log_artifact(cfg.source_config_path)
                except Exception:
                    self._logger.warning("Config-YAML konnte nicht nach MLflow geloggt werden.", exc_info=True)
            else:
                # Kein Datei-Pfad (z.B. UI-generiert) → Config als YAML-Text loggen
                try:
                    import yaml
                    cfg_dict = cfg.model_dump(mode="json", exclude_none=True)
                    _log_temp_artifact(mlflow, lambda p: open(p, "w", encoding="utf-8").write(yaml.dump(cfg_dict, allow_unicode=True, sort_keys=False)), "config.yml")
                except Exception:
                    self._logger.warning("Config konnte nicht als Artefakt geloggt werden.", exc_info=True)

            # ── DataPrep-Config mitloggen (falls vorhanden) ──
            # Die DataPrep-Pipeline speichert ihre Konfiguration als dataprep_config.yml
            # im Output-Verzeichnis. Da x_file typischerweise im selben Verzeichnis liegt,
            # suchen wir dort nach der Datei.
            try:
                from pathlib import Path
                x_dir = Path(cfg.data_files.x_file).resolve().parent
                dp_cfg_path = x_dir / "dataprep_config.yml"
                if dp_cfg_path.is_file():
                    mlflow.log_artifact(str(dp_cfg_path))
                    self._logger.info("DataPrep-Config nach MLflow geloggt: %s", dp_cfg_path)
            except Exception:
                pass  # Kein DataPrep-Output → kein Log, kein Fehler

            # Report-Collector initialisieren
            report = ReportCollector()
            report.add_config(cfg)
            report.run_name = _run_name
            try:
                report.add_dataprep_info(x_dir)
            except Exception:
                pass
            self._report = report

            import time as _time_mod
            _pipeline_start = _time_mod.perf_counter()

            # ── Pipeline Start ──
            self._logger.info("═" * 60)
            self._logger.info("rubin Pipeline Start")
            self._logger.info("═" * 60)
            _models_list = cfg.models.models_to_train
            self._logger.info(
                "Config: %d Modelle (%s), %s, %d-Fold CV, Parallel-Level %d",
                len(_models_list), ", ".join(_models_list),
                cfg.base_learner.type.upper(),
                cfg.data_processing.cross_validation_splits,
                cfg.constants.parallel_level,
            )
            # DML Cross-Fitting Info
            _dml_cv = cfg.data_processing.dml_crossfit_folds
            _mc = cfg.data_processing.mc_iters
            if _dml_cv != 2 or _mc:
                _mc_str = f", mc_iters={_mc}" if _mc else ""
                self._logger.info("DML Cross-Fitting: %d interne Folds%s", _dml_cv, _mc_str)

            _tuning_parts = []
            if cfg.tuning.enabled:
                _tuning_parts.append(f"BL-Tuning ({cfg.tuning.n_trials} Trials)")
            if getattr(cfg, "final_model_tuning", None) and cfg.final_model_tuning.enabled:
                _tuning_parts.append(f"FMT ({cfg.final_model_tuning.n_trials} Trials)")
            if getattr(cfg.causal_forest, "use_econml_tune", False):
                _tuning_parts.append("GRF-Tuning")
            if getattr(cfg.models, "ensemble", False):
                _tuning_parts.append("Ensemble")
            if _tuning_parts:
                self._logger.info("Aktiv: %s", " | ".join(_tuning_parts))

            _progress("Daten laden & Preprocessing")
            X, T, Y, S, eval_mask = self._load_inputs()

            if eval_mask is not None:
                mlflow.log_param("eval_mask_mode", "train_many_evaluate_one")
                mlflow.log_metric("eval_mask_n_eval", int(eval_mask.sum()))
                mlflow.log_metric("eval_mask_n_total", len(eval_mask))

            use_external = str(getattr(cfg.data_processing, "validate_on", "cross")).lower() == "external"
            holdout_data = None

            # ── TMEO (Train-Many-Evaluate-One): Mask-basierte Train/Eval-Trennung ──
            # Vor dieser Änderung: Trainings-Pipeline sah eval_mask-Zeilen → Data-Leakage
            # (Metriken auf Mask-Subset wurden auf Modelle berechnet, die Mask-Rows
            # mittrainiert haben). Neue Logik: Mask-Rows werden zum Holdout-Set, die
            # Train-Pipeline bekommt nur ~mask-Rows. Effektiv läuft TMEO dann wie
            # External Eval, nur mit implizit abgeleiteten Eval-Daten.
            if eval_mask is not None and not use_external:
                import numpy as _np
                _mask = _np.asarray(eval_mask).astype(bool)
                if _mask.any() and not _mask.all():
                    _eval_idx = _np.where(_mask)[0]
                    _train_idx = _np.where(~_mask)[0]
                    # Holdout aus Mask-Rows bilden (wie External-Eval-Pfad)
                    X_tmeo_eval = X.iloc[_eval_idx].reset_index(drop=True)
                    T_tmeo_eval = _np.asarray(T)[_eval_idx]
                    Y_tmeo_eval = _np.asarray(Y)[_eval_idx]
                    S_tmeo_eval = _np.asarray(S)[_eval_idx] if S is not None else None
                    holdout_data = (X_tmeo_eval, T_tmeo_eval, Y_tmeo_eval, S_tmeo_eval)
                    # Trainings-Daten auf ~mask beschränken
                    X = X.iloc[_train_idx].reset_index(drop=True)
                    T = _np.asarray(T)[_train_idx]
                    Y = _np.asarray(Y)[_train_idx]
                    if S is not None:
                        S = _np.asarray(S)[_train_idx]
                    # eval_mask geht NICHT mehr in evaluation, weil holdout_data genutzt wird
                    eval_mask = None
                    self._logger.info(
                        "TMEO aktiviert: %d Training-Zeilen (~mask), %d Eval-Zeilen (mask). "
                        "Kein Data-Leakage — läuft wie External-Eval-Pfad.",
                        len(_train_idx), len(_eval_idx),
                    )
                    mlflow.log_param("tmeo_train_n", len(_train_idx))
                    mlflow.log_param("tmeo_eval_n", len(_eval_idx))
                else:
                    self._logger.warning(
                        "eval_mask ist leer oder umfasst alle Zeilen — TMEO wird ignoriert."
                    )
                    eval_mask = None

            # X_full/T_full/Y_full werden nur für Bundle-Refit benötigt.
            # Kopie erst erstellen wenn nötig, spart ~2 GB bei normalen Läufen.
            # WICHTIG: Snapshot NACH TMEO-Split, damit Bundle-Refit konsistent zu
            # External Eval nur auf Trainings-Daten läuft (nicht auf Mask-Rows).
            if bundle_enabled:
                X_full, T_full, Y_full = X.copy(), np.asarray(T).copy(), np.asarray(Y).copy()
            else:
                X_full, T_full, Y_full = X, T, Y  # Referenz, keine Kopie

            if use_external:
                # Externe Evaluationsdaten laden
                self._logger.info("Validierungsmodus: external – lade separate Eval-Daten.")
                try:
                    X_eval = self._read_table(cfg.data_files.eval_x_file)
                    T_eval = self._read_table(cfg.data_files.eval_t_file)["T"].to_numpy()
                    Y_eval = self._read_table(cfg.data_files.eval_y_file)["Y"].to_numpy()
                except FileNotFoundError as e:
                    self._logger.warning(
                        "Externe Eval-Dateien nicht gefunden: %s. "
                        "Fallback auf Cross-Validation.", e,
                    )
                    use_external = False
                    X_eval, T_eval, Y_eval = None, None, None

            if use_external and X_eval is not None:
                # Längen-Alignment prüfen
                n_eval = len(X_eval)
                if len(T_eval) != n_eval or len(Y_eval) != n_eval:
                    self._logger.warning(
                        "Eval-Dateien haben unterschiedliche Längen: X=%d, T=%d, Y=%d. "
                        "Fallback auf Cross-Validation.",
                        n_eval, len(T_eval), len(Y_eval),
                    )
                    use_external = False
                    X_eval, T_eval, Y_eval = None, None, None

            if use_external and X_eval is not None:
                # Feature-Alignment: X_eval muss dieselben Spalten wie X haben
                missing_cols = set(X.columns) - set(X_eval.columns)
                extra_cols = set(X_eval.columns) - set(X.columns)
                if missing_cols:
                    self._logger.warning(
                        "Eval-Daten: %d Features fehlen (in X aber nicht in X_eval): %s. "
                        "Fehlende Spalten werden mit 0 aufgefüllt.",
                        len(missing_cols), sorted(missing_cols)[:10],
                    )
                    for c in missing_cols:
                        X_eval[c] = 0
                if extra_cols:
                    self._logger.info("Eval-Daten: %d extra Spalten ignoriert.", len(extra_cols))
                    X_eval = X_eval[[c for c in X.columns if c in X_eval.columns] + list(missing_cols)]
                # Spaltenreihenfolge an X angleichen
                X_eval = X_eval[X.columns]

                S_eval = None
                if cfg.data_files.eval_s_file:
                    try:
                        col = cfg.historical_score.column
                        S_eval = self._read_table(cfg.data_files.eval_s_file)[col].to_numpy(dtype=float)
                        S_eval = np.nan_to_num(S_eval, nan=0.0, posinf=0.0, neginf=0.0)
                        if len(S_eval) != n_eval:
                            self._logger.warning(
                                "Eval S-Länge (%d) ≠ X_eval-Länge (%d). Score wird ignoriert.",
                                len(S_eval), n_eval,
                            )
                            S_eval = None
                    except Exception:
                        S_eval = None
                # Dtype-Konsistenz: Eval-Daten durchlaufen dieselben Transformationen
                # wie die Trainingsdaten (reduce_memory, category-Dtypes), damit es
                # bei der Evaluation keine Dtype-Mismatches gibt.
                if getattr(cfg.data_processing, "reduce_memory", True):
                    X_eval = reduce_mem_usage(X_eval)
                cat_cols_eval = getattr(cfg.data_processing, "categorical_columns", None)
                if cat_cols_eval:
                    for c in cat_cols_eval:
                        if c in X_eval.columns and not isinstance(X_eval[c].dtype, pd.CategoricalDtype):
                            try:
                                X_eval[c] = X_eval[c].astype("category")
                            except Exception:
                                pass
                holdout_data = (X_eval, T_eval, Y_eval, S_eval)
                mlflow.log_param("validation_mode", "external")
                mlflow.log_param("eval_n_rows", len(X_eval))
                self._logger.info("Eval-Daten: %d Zeilen, %d Features.", len(X_eval), X_eval.shape[1])
            else:
                mlflow.log_param("validation_mode", "cross")

            report.add_data_stats(X, T, Y, S)
            if holdout_data is not None:
                # Gilt für External Eval UND TMEO (beide setzen holdout_data)
                report.add_eval_data_stats(*holdout_data)

            _progress("Feature-Selektion")
            # Kategorischer Patch auch für Feature-Selektion: LightGBM-Importance
            # braucht categorical_feature, sonst werden kategorische Features
            # systematisch unterbewertet und fliegen möglicherweise raus.
            with patch_categorical_features(X, base_learner_type=cfg.base_learner.type):
                X, removed = self._run_feature_selection(cfg, X, T, Y, mlflow)
            if removed.get("importance") or removed.get("high_correlation"):
                report.feature_selection_info["n_before"] = report.data_stats.get("n_features", 0)
                report.feature_selection_info["n_after"] = len(X.columns)
                report.feature_selection_info["n_removed_correlation"] = len(removed.get("high_correlation", []))
                report.feature_selection_info["n_removed_importance"] = len(removed.get("importance", []))
                n_corr = len(removed.get("high_correlation", []))
                report.feature_selection_info["n_after_correlation"] = report.data_stats.get("n_features", 0) - n_corr

            # Feature-Alignment: external Eval-Daten auf selektierte Features reduzieren
            if holdout_data is not None:
                X_h, T_h, Y_h, S_h = holdout_data
                if list(X_h.columns) != list(X.columns):
                    X_h = X_h.reindex(columns=X.columns)
                    self._logger.info(
                        "Eval-Daten: Feature-Alignment auf %d selektierte Spalten angewendet.",
                        len(X.columns),
                    )
                    holdout_data = (X_h, T_h, Y_h, S_h)

            if removed:
                total_removed = sum(len(v) for v in removed.values() if isinstance(v, list))
                if total_removed:
                    self._logger.info(
                        "Feature-Selektion: %d → %d Features (-%d entfernt)",
                        X.shape[1] + total_removed, X.shape[1], total_removed,
                    )
            elif cfg.feature_selection.enabled:
                self._logger.info("Feature-Selektion: Alle %d Features beibehalten.", X.shape[1])

            _progress("Base-Learner-Tuning")
            # Zweiter Patch mit post-FS Spaltenindizes (Spalten haben sich geändert).
            with patch_categorical_features(X, base_learner_type=cfg.base_learner.type) as cat_indices:
                if cat_indices:
                    mlflow.log_param("n_categorical_features", len(cat_indices))
                    _bt_label = (cfg.base_learner.type or "lgbm").upper()
                    print(
                        f"[rubin] Kategorische Features: {len(cat_indices)} von {X.shape[1]} "
                        f"Spalten → {_bt_label} erhält cat_feature-Indizes.",
                        flush=True,
                    )
                else:
                    print(
                        f"[rubin] Keine kategorialen Features erkannt → Standard-Encoding (numerisch).",
                        flush=True,
                    )

                tuned_params_by_model = self._run_tuning(cfg, X, T, Y, mlflow, _progress_cb=_progress)
                gc.collect()  # Tuner-Internals freigeben (Optuna Studies, Trial-Daten)

                # _run_training emittiert Progress-Labels "GRF-Tuning" bzw.
                # "Training & Cross-Predictions" pro Modell-Phase (siehe dort).
                models, preds, fold_models = self._run_training(cfg, X, T, Y, tuned_params_by_model, holdout_data, mlflow, _progress_cb=_progress)
                selected_feature_columns = list(X.columns)

                # ── Ensemble (optional): Gleichgewichtetes Mittel aller trainierten Modelle ──
                ENSEMBLE_NAME = "Ensemble"
                if getattr(cfg.models, "ensemble", False) and len(models) >= 2:
                    try:
                        from econml.score import EnsembleCateEstimator
                        is_mt = is_multi_treatment(T)
                        # Cross-Predictions mitteln
                        ensemble_df = pd.DataFrame(index=range(len(X)))
                        model_names = [n for n in models if n != ENSEMBLE_NAME]
                        # Y und T aus dem ersten Modell übernehmen (identisch für alle)
                        first_df = preds[model_names[0]]
                        if "Y" in first_df.columns:
                            ensemble_df["Y"] = first_df["Y"].values
                        if "T" in first_df.columns:
                            ensemble_df["T"] = first_df["T"].values
                        if is_mt:
                            n_arms = int(T.max()) if hasattr(T, 'max') else int(np.max(T))
                            for k in range(1, n_arms + 1):
                                arm_cols = [preds[n][f"Predictions_{n}_T{k}"] for n in model_names if f"Predictions_{n}_T{k}" in preds[n].columns]
                                if arm_cols:
                                    ensemble_df[f"Predictions_{ENSEMBLE_NAME}_T{k}"] = np.mean(arm_cols, axis=0)
                                    ensemble_df[f"Train_{ENSEMBLE_NAME}_T{k}"] = np.nan
                            # OptimalTreatment
                            arm_pred_cols = [c for c in ensemble_df.columns if c.startswith(f"Predictions_{ENSEMBLE_NAME}_T")]
                            if arm_pred_cols:
                                vals = ensemble_df[arm_pred_cols].values
                                best_eff = np.nanmax(vals, axis=1)
                                best_arm = np.nanargmax(vals, axis=1) + 1
                                ensemble_df[f"OptimalTreatment_{ENSEMBLE_NAME}"] = np.where(best_eff > 0, best_arm, 0)
                        else:
                            pred_cols = [preds[n][f"Predictions_{n}"] for n in model_names if f"Predictions_{n}" in preds[n].columns]
                            if pred_cols:
                                ensemble_df[f"Predictions_{ENSEMBLE_NAME}"] = np.mean(pred_cols, axis=0)
                                ensemble_df[f"Train_{ENSEMBLE_NAME}"] = np.nan

                        # EconML EnsembleCateEstimator: Gleichgewichtung aller gefitteten Modelle
                        model_list = [models[n] for n in model_names]
                        weights = np.ones(len(model_list)) / len(model_list)
                        ensemble_model = EnsembleCateEstimator(
                            cate_models=model_list,
                            weights=weights,
                        )
                        preds[ENSEMBLE_NAME] = ensemble_df
                        models[ENSEMBLE_NAME] = ensemble_model
                        _log_temp_artifact(mlflow, lambda p, _df=ensemble_df: _df.to_csv(p, index=False, float_format="%.10g"), f"predictions_{ENSEMBLE_NAME}.csv")
                        mlflow.log_param("ensemble_enabled", True)
                        mlflow.log_param("ensemble_models", ",".join(model_names))
                        self._logger.info(
                            "Ensemble erstellt (EconML EnsembleCateEstimator): %d Modelle gleichgewichtet (%s).",
                            len(model_names), ", ".join(model_names),
                        )
                    except Exception:
                        self._logger.warning("Ensemble-Erstellung fehlgeschlagen.", exc_info=True)

                _progress("Evaluation & Metriken")
                eval_summary: Dict[str, Dict[str, float]] = {}
                fitted_tester_bt = None
                try:
                    eval_summary, _, fitted_tester_bt = self._run_evaluation(cfg, X, T, Y, S, holdout_data, preds, models, tuned_params_by_model, mlflow, eval_mask=eval_mask)
                except Exception:
                    self._logger.warning("Uplift-Evaluation fehlgeschlagen.", exc_info=True)

                # ── Evaluations-Übersicht + Champion ──
                if eval_summary:
                    _sel_met = cfg.selection.metric
                    self._logger.info("─" * 60)
                    self._logger.info("Evaluation (%s):", _sel_met)
                    for _mn, _ms in sorted(eval_summary.items(), key=lambda x: x[1].get(_sel_met, 0), reverse=cfg.selection.higher_is_better):
                        _score = _ms.get(_sel_met)
                        _extra = " | ".join(f"{k}={v:.5g}" for k, v in _ms.items() if isinstance(v, (int, float)) and k != _sel_met and k in ("qini", "auuc", "policy_value"))
                        self._logger.info("  %-22s %s=%.6g  %s", _mn, _sel_met, _score if _score is not None else 0, _extra)
                    _champ = self._determine_champion(cfg, eval_summary, models)
                    if _champ:
                        _champ_score = eval_summary.get(_champ, {}).get(_sel_met, 0)
                        self._logger.info("═" * 60)
                        self._logger.info("Champion: %s (%s=%.6g)", _champ, _sel_met, _champ_score or 0)
                        self._logger.info("═" * 60)

                        # ── Dedizierte Top-Level-Metriken + Tags für MLflow-Overview ──
                        # Damit in der Experiment-Übersicht sofort sichtbar, welcher
                        # Run wie performt hat, ohne in den Run reinzuklicken.
                        # Columns müssen einmalig in der MLflow-UI als sichtbar ausgewählt
                        # werden (localStorage-persistiert, kein Server-Side-Default).
                        if _champ_score is not None:
                            mlflow.log_metric("champion_score", float(_champ_score))
                        mlflow.set_tag("rubin.champion_model", _champ)
                        mlflow.set_tag("rubin.selection_metric", _sel_met)
                        mlflow.set_tag("rubin.learner", cfg.base_learner.type or "catboost")
                        _val_mode = "external" if use_external else ("TMEO" if eval_mask is not None else "cross")
                        mlflow.set_tag("rubin.validation_mode", _val_mode)
                        # Zusätzliche Champion-Metriken für Schnellvergleich
                        _champ_metrics = eval_summary.get(_champ, {})
                        for _extra_key in ("qini", "auuc", "policy_value"):
                            _ev = _champ_metrics.get(_extra_key)
                            if isinstance(_ev, (int, float)):
                                mlflow.log_metric(f"champion_{_extra_key}", float(_ev))

                # ── Surrogate-Einzelbaum (nur Champion) ──
                if eval_summary and cfg.surrogate_tree.enabled:
                    _progress("Surrogate-Tree")
                    champion_name = self._determine_champion(cfg, eval_summary, models)

                    if champion_name and champion_name in preds:
                        try:
                            self._logger.info("Trainiere Surrogate auf Champion %s.", champion_name)
                            surrogate_wrapper, surrogate_df = self._train_and_evaluate_surrogate(
                                cfg, X, T, Y, champion_name, preds, holdout_data,
                                models, eval_summary, mlflow,
                                surrogate_name=SURROGATE_MODEL_NAME,
                                eval_mask=eval_mask,
                            )
                            preds[SURROGATE_MODEL_NAME] = surrogate_df
                            models[SURROGATE_MODEL_NAME] = surrogate_wrapper
                        except Exception:
                            self._logger.warning("SurrogateTree (Champion) fehlgeschlagen.", exc_info=True)
                    else:
                        self._logger.warning("Surrogate-Tree: Kein Champion ermittelt, überspringe.")

                gc.collect()  # Nach Surrogate freigeben
                self._logger.info("RAM-Optimierung: gc.collect() nach Surrogate.")

                if bundle_enabled:
                    _progress("Bundle-Export")
                self._run_bundle_export(cfg, models, eval_summary, X, T, Y, X_full, T_full, Y_full, selected_feature_columns, holdout_data, export_bundle, bundle_dir, bundle_id, mlflow)
                self._run_optional_output(cfg, eval_summary, removed, preds)

                # ── Explainability (SHAP) ──
                if cfg.shap_values.calculate_shap_values:
                    _progress("Explainability")
                    try:
                        self._run_explainability(cfg, X, T, Y, models, eval_summary, mlflow, report, holdout_data=holdout_data, fold_models=fold_models)
                    except Exception:
                        self._logger.warning("Explainability-Schritt fehlgeschlagen.", exc_info=True)
            # ── Ende categorical patch — originale .fit()-Methoden wiederhergestellt ──

            # ── RAM-Optimierung: Nicht mehr benötigte Objekte freigeben ──
            # Nach Bundle-Export und Prediction-Output werden nur noch
            # eval_summary und report für den HTML-Report benötigt.
            _champ = self._determine_champion(cfg, eval_summary, models) if eval_summary else None
            _keep = {_champ, SURROGATE_MODEL_NAME}
            for mname in list(models.keys()):
                if mname not in _keep:
                    del models[mname]
            preds.clear()
            del X_full, T_full, Y_full
            gc.collect()
            self._logger.info("RAM-Optimierung: Modelle, Predictions und X_full freigegeben.")

            # ── .rubin_cache: Immer schreiben, damit der Server den Report finden kann ──
            cache_dir = os.path.join(self._work_dir, ".rubin_cache")
            os.makedirs(cache_dir, exist_ok=True)

            # ── eval_summary.json → .rubin_cache + MLflow + output_dir ──
            try:
                cache_summary = os.path.join(cache_dir, "uplift_eval_summary.json")
                with open(cache_summary, "w", encoding="utf-8") as f:
                    json.dump(eval_summary, f, ensure_ascii=False, indent=2, default=float)
                _log_temp_artifact(mlflow, lambda p: open(p, "w", encoding="utf-8").write(
                    json.dumps(eval_summary, ensure_ascii=False, indent=2, default=float)
                ), "uplift_eval_summary.json")
                if cfg.optional_output.output_dir:
                    summary_path = os.path.join(cfg.optional_output.output_dir, "uplift_eval_summary.json")
                    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
                    import shutil
                    shutil.copy2(cache_summary, summary_path)
            except Exception:
                self._logger.warning("eval_summary.json konnte nicht gespeichert werden.", exc_info=True)

            # ── HTML-Report generieren ──
            _progress("HTML-Report")
            try:
                # Modell-Metriken + Champion
                for mname, metrics in eval_summary.items():
                    report.model_metrics[mname] = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                if eval_summary:
                    try:
                        report.champion_name = self._determine_champion(cfg, eval_summary, models) or ""
                    except Exception:
                        pass

                # ── Heterogeneity Assessment ──
                if eval_summary and report.champion_name:
                    try:
                        report.heterogeneity_assessment = self._compute_heterogeneity_assessment(
                            cfg, eval_summary, report.champion_name,
                        )
                    except Exception:
                        self._logger.warning("Heterogeneity-Assessment fehlgeschlagen.", exc_info=True)

                # Surrogate-Info
                if SURROGATE_MODEL_NAME in eval_summary:
                    report.surrogate_info = {
                        "champion": report.champion_name,
                        "metrics": {k: v for k, v in eval_summary[SURROGATE_MODEL_NAME].items() if isinstance(v, (int, float))},
                    }

                # Letzten Schritt abschließen + Timing (nach Setup, vor Generate)
                if _last_step_label[0] is not None:
                    step_times[_last_step_label[0]] = time.perf_counter() - _last_step_start[0]
                report.step_durations = step_times
                report.total_elapsed = sum(step_times.values())

                # Report generieren → .rubin_cache (immer) + MLflow + output_dir
                cache_report = os.path.join(cache_dir, "analysis_report.html")
                generate_html_report(report, cache_report)
                mlflow.log_artifact(cache_report)
                if cfg.optional_output.output_dir:
                    os.makedirs(cfg.optional_output.output_dir, exist_ok=True)
                    import shutil
                    shutil.copy2(cache_report, os.path.join(cfg.optional_output.output_dir, "analysis_report.html"))
                self._logger.info("HTML-Report geschrieben: %s", cache_report)
            except Exception:
                self._logger.warning("HTML-Report-Generierung fehlgeschlagen.", exc_info=True)

            _pipeline_elapsed = _time_mod.perf_counter() - _pipeline_start
            _mins = int(_pipeline_elapsed // 60)
            _secs = int(_pipeline_elapsed % 60)
            self._logger.info("═" * 60)
            self._logger.info("Pipeline abgeschlossen in %dm %ds", _mins, _secs)
            # Step-Zeiten
            for _step_name, _step_dur in step_times.items():
                self._logger.info("  %-28s %5.1fs", _step_name, _step_dur)
            self._logger.info("═" * 60)
            print(f"[rubin] Step {total}/{total}: Fertig", flush=True)
            return AnalysisResult(models=models, predictions=preds, removed_features=removed, eval_summary=eval_summary)
