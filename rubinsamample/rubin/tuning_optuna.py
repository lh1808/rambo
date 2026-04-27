from __future__ import annotations

"""Optuna-basiertes Hyperparameter-Tuning für Base Learner.
Kernidee:
- Es werden nicht "die kausalen Learner" direkt getunt, sondern die darunter
verwendeten Base Learner (z. B. Outcome-/Propensity-Modelle).
- Die verfügbare Datenmenge kann je kausalem Learner unterschiedlich sein
(S-Learner: alle Daten; T-/X-Learner: gruppenweise Daten). Diese Logik wird
beim Sampling der Tuning-Daten berücksichtigt, damit die gefundenen Hyperparameter
realistische Bedingungen widerspiegeln."""


from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import logging
import os
from rubin.utils.data_utils import available_cpu_count

from rubin.settings import AnalysisConfig, SearchSpaceConfig, SearchSpaceParameterConfig


def _iter_stratified_or_kfold(labels: np.ndarray, n_splits: int, seed: int):
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1:
        labels_arr = labels_arr.reshape(-1)

    # Safety: Mindestens 2 Folds für Cross-Validation
    n_splits = max(2, int(n_splits))
    if len(labels_arr) < 2:
        log = logging.getLogger("rubin.tuning")
        log.error(
            "Tuning-Split fehlgeschlagen: target hat nur %d Element(e). "
            "dtype=%s, unique=%s, shape=%s. "
            "Prüfe ob X/T/Y gleich viele Zeilen haben und ob df_frac aktiv ist.",
            len(labels_arr), labels_arr.dtype, np.unique(labels_arr).tolist(), labels_arr.shape,
        )
        raise ValueError(
            f"Für die Aufteilung werden mindestens 2 Beobachtungen benötigt "
            f"(erhalten: {len(labels_arr)}, unique={np.unique(labels_arr).tolist()}, "
            f"dtype={labels_arr.dtype})."
        )

    counts = pd.Series(labels_arr).value_counts(dropna=False)
    effective_splits = min(int(n_splits), len(labels_arr))
    if not counts.empty:
        effective_splits = min(effective_splits, int(counts.min()))

    if effective_splits >= 2:
        cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
        return cv.split(np.zeros(len(labels_arr)), labels_arr)

    fallback_splits = min(int(n_splits), len(labels_arr))
    if fallback_splits < 2:
        log = logging.getLogger("rubin.tuning")
        log.error(
            "Tuning-Split Fallback fehlgeschlagen: n_splits=%d, len=%d, unique=%s, counts=%s",
            n_splits, len(labels_arr), np.unique(labels_arr).tolist(),
            dict(pd.Series(labels_arr).value_counts(dropna=False)),
        )
        raise ValueError(
            f"Für die Aufteilung werden mindestens 2 Folds benötigt "
            f"(n_splits={n_splits}, len={len(labels_arr)}, "
            f"unique={np.unique(labels_arr).tolist()})."
        )
    cv = KFold(n_splits=fallback_splits, shuffle=True, random_state=seed)
    return cv.split(np.zeros(len(labels_arr)))


def _first_crossfit_train_indices(n: int, t: np.ndarray, n_splits: int, seed: int, y: np.ndarray = None) -> np.ndarray:
    """Ermittelt die Trainingsindizes des ersten Cross-Prediction-Folds.

    Wenn *y* übergeben wird, stratifiziert auf T×Y (identisch mit Cross-Predictions
    in training.py). Ohne *y* wird nur auf T stratifiziert (Legacy-Fallback).
    """
    t_int = np.asarray(t).astype(int)
    if y is not None:
        import pandas as pd
        from sklearn.model_selection import StratifiedKFold, KFold
        y_int = np.asarray(y).astype(int)
        strata = (pd.Series(t_int).astype(str) + "_" + pd.Series(y_int).astype(str)).to_numpy()
        strata_counts = pd.Series(strata).value_counts(dropna=False)
        eff_sp = min(int(n_splits), int(strata_counts.min())) if not strata_counts.empty else int(n_splits)
        if eff_sp >= 2:
            cv = StratifiedKFold(n_splits=eff_sp, shuffle=True, random_state=seed)
            for tr_idx, _ in cv.split(np.zeros(n), strata):
                return np.asarray(tr_idx, dtype=int)
        cv = KFold(n_splits=max(2, min(int(n_splits), n)), shuffle=True, random_state=seed)
        for tr_idx, _ in cv.split(np.zeros(n)):
            return np.asarray(tr_idx, dtype=int)
        return np.arange(n, dtype=int)
    split_iter = _iter_stratified_or_kfold(t_int, n_splits=n_splits, seed=seed)
    for tr_idx, _ in split_iter:
        return np.asarray(tr_idx, dtype=int)
    return np.arange(n, dtype=int)


def _safe_import_optuna():
    try:
        import optuna  # type: ignore
        # Performance: Optuna-Logging auf WARNING reduzieren
        # (unterdrückt "Trial X finished with value..." Meldungen)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna
    except Exception as e:
        raise ImportError(
            "Optuna-Tuning ist aktiviert, aber 'optuna' konnte nicht importiert werden. "
            "Bitte installieren mit: pip install optuna"
        ) from e


def _build_lgbm_classifier(params: Dict[str, Any], seed: int, parallel_jobs: int = -1):
    import lightgbm as lgbm
    # Möglichst deterministisches Verhalten
    fixed = dict(
        random_state=seed,
        verbose=-1,  # Unterdrückt C++-Level stdout (KRITISCH für Pipe-Performance)
    )
    # Default: DART-Boosting (weniger Overfitting auf orthogonalisierten Residuen,
    # bessere Generalisation bei CATE-Schätzung). Kann durch params überschrieben werden.
    fixed["boosting_type"] = "dart"
    fixed.update(params)
    # n_jobs NACH params setzen, damit fixed_params es nicht überschreiben können.
    fixed["n_jobs"] = parallel_jobs
    # Determinismus: Nur bei parallel_jobs=1 (Level 1) erzwingen. Bei n_jobs>1
    # erzwingt deterministic=True Single-Thread-Training — bis zu 5× langsamer.
    # random_state allein garantiert Reproduzierbarkeit der Splits und Initialisierung;
    # die minimale Floating-Point-Varianz durch parallele Histogramme ist vernachlässigbar.
    if parallel_jobs == 1:
        fixed["deterministic"] = True
        fixed["force_row_wise"] = True
    # Objective wird NICHT hardcodiert, damit LGBMClassifier binary vs.
    # multiclass automatisch aus den Trainingsdaten ableiten kann.
    # Für BT-Propensity/Outcome → binary; für MT-Propensity → multiclass.
    fixed.pop("objective", None)
    return lgbm.LGBMClassifier(**fixed)


def _build_lgbm_regressor(params: Dict[str, Any], seed: int, parallel_jobs: int = -1):
    import lightgbm as lgbm
    fixed = dict(
        random_state=seed,
        verbose=-1,  # Unterdrückt C++-Level stdout (KRITISCH für Pipe-Performance)
    )
    # Default: DART-Boosting (siehe Classifier-Kommentar)
    fixed["boosting_type"] = "dart"
    fixed.update(params)
    # n_jobs NACH params setzen, damit fixed_params es nicht überschreiben können.
    fixed["n_jobs"] = parallel_jobs
    # Determinismus nur bei Level 1 (siehe Classifier-Kommentar)
    if parallel_jobs == 1:
        fixed["deterministic"] = True
        fixed["force_row_wise"] = True
    # Für Effektmodelle ist Regression zwingend sinnvoll. Falls in fixed_params ein
    # Klassifikations-Objective gesetzt wurde, überschreiben wir es bewusst.
    fixed["objective"] = "regression"
    return lgbm.LGBMRegressor(**fixed)


def _build_catboost_classifier(params: Dict[str, Any], seed: int, parallel_jobs: int = -1):
    from catboost import CatBoostClassifier  # type: ignore
    fixed = dict(
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    fixed.update(params)
    # thread_count NACH params setzen, damit fixed_params es nicht überschreiben können.
    fixed["thread_count"] = parallel_jobs
    # KRITISCH: CatBoost Default bootstrap_type='Bayesian' ist NICHT kompatibel mit
    # subsample (= 'taken fraction'). Wenn subsample gesetzt wird (Search-Space oder
    # Defaults), MUSS bootstrap_type auf 'MVS' oder 'Bernoulli' stehen.
    # MVS (Minimal Variance Sampling) ist CatBoosts empfohlene Subsampling-Methode.
    if "subsample" in fixed and "bootstrap_type" not in fixed:
        fixed["bootstrap_type"] = "MVS"
    return CatBoostClassifier(**fixed)


def _build_catboost_regressor(params: Dict[str, Any], seed: int, parallel_jobs: int = -1):
    from catboost import CatBoostRegressor  # type: ignore
    fixed = dict(
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    fixed.update(params)
    # thread_count NACH params setzen, damit fixed_params es nicht überschreiben können.
    fixed["thread_count"] = parallel_jobs
    # KRITISCH: Analog zum Classifier — bootstrap_type='MVS' wenn subsample gesetzt.
    if "subsample" in fixed and "bootstrap_type" not in fixed:
        fixed["bootstrap_type"] = "MVS"
    fixed["loss_function"] = "RMSE"
    return CatBoostRegressor(**fixed)


def build_base_learner(base_type: str, params: Dict[str, Any], seed: int, task: str = "classifier", parallel_jobs: int = -1):
    """Die Funktion wird sowohl im Tuning als auch beim finalen Training verwendet, um sicherzustellen,
dass identische Defaults (z. B. Random Seed) gesetzt werden.
parallel_jobs: Anzahl Kerne für Base Learner. -1 = alle, 1 = single-core.

"Both"-Modus: Wenn base_type == "both", MUSS params einen Key "_learner_type" enthalten
(gesetzt von Optuna als kategorische Entscheidung pro Task). Andernfalls Fallback auf
CatBoost mit Warnung (passend zum globalen Default base_learner.type="catboost")."""
    base_type = (base_type or "lgbm").lower()
    task = (task or "classifier").lower()
    if task not in {"classifier", "regressor"}:
        raise ValueError(f"Unbekannter task={task!r}. Erwartet: 'classifier' oder 'regressor'.")

    # "Both"-Modus: Tatsächlichen Learner aus params lesen
    if base_type == "both":
        chosen = params.get("_learner_type") if isinstance(params, dict) else None
        if chosen not in ("lgbm", "catboost"):
            import logging
            logging.getLogger("rubin.tuning").warning(
                "base_type='both' aber kein '_learner_type' in params — Fallback auf 'catboost' "
                "(globaler Default). Das deutet auf ein fehlendes Tuning oder eine Rolle "
                "ohne getunte Params hin."
            )
            chosen = "catboost"
        base_type = chosen
        # Flache Params + nested per-Learner fixed_params korrekt zusammenfassen.
        # Struktur bei "both": {lgbm: {...}, catboost: {...}, <flat tuned params>, _learner_type}
        # Nur der gewählte Sub-Dict wird verwendet; Params für den anderen Learner werden
        # verworfen (sonst würden z.B. LGBM-spezifische Keys CatBoost-Builder crashen).
        flat_params = {}
        for k, v in (params or {}).items():
            if k == "_learner_type":
                continue
            if k in ("lgbm", "catboost") and isinstance(v, dict):
                if k == chosen:
                    flat_params.update(v)
                # else: skip — fixed_params für den nicht gewählten Learner
            else:
                # Flache Params (aus _suggest_params) gehören immer zum aktiven Learner
                flat_params[k] = v
        params = flat_params

    if base_type == "lgbm":
        return _build_lgbm_classifier(params, seed, parallel_jobs) if task == "classifier" else _build_lgbm_regressor(params, seed, parallel_jobs)
    if base_type == "catboost":
        return _build_catboost_classifier(params, seed, parallel_jobs) if task == "classifier" else _build_catboost_regressor(params, seed, parallel_jobs)
    raise ValueError(f"Unbekannter base_learner.type={base_type!r}. Erwartet: 'lgbm', 'catboost' oder 'both'.")


def _default_search_space(base_type: str) -> Dict[str, SearchSpaceParameterConfig]:
    base_type = (base_type or "").lower()
    if base_type == "lgbm":
        return {
            "n_estimators": SearchSpaceParameterConfig(type="int", low=200, high=600),
            "learning_rate": SearchSpaceParameterConfig(type="float", low=1e-2, high=1.5e-1, log=True),
            "num_leaves": SearchSpaceParameterConfig(type="int", low=15, high=127),
            "max_depth": SearchSpaceParameterConfig(type="int", low=3, high=8),
            "min_child_samples": SearchSpaceParameterConfig(type="int", low=10, high=200),
            "min_child_weight": SearchSpaceParameterConfig(type="float", low=1e-2, high=50.0, log=True),
            "subsample": SearchSpaceParameterConfig(type="float", low=0.5, high=1.0),
            "colsample_bytree": SearchSpaceParameterConfig(type="float", low=0.3, high=0.9),
            "min_split_gain": SearchSpaceParameterConfig(type="float", low=0.0, high=1.0),
            "reg_alpha": SearchSpaceParameterConfig(type="float", low=1e-6, high=20.0, log=True),
            "reg_lambda": SearchSpaceParameterConfig(type="float", low=1e-6, high=20.0, log=True),
            "path_smooth": SearchSpaceParameterConfig(type="float", low=0.0, high=10.0),
        }
    if base_type == "catboost":
        return {
            "iterations": SearchSpaceParameterConfig(type="int", low=200, high=600),
            "learning_rate": SearchSpaceParameterConfig(type="float", low=1e-2, high=1.5e-1, log=True),
            "depth": SearchSpaceParameterConfig(type="int", low=4, high=8),
            "l2_leaf_reg": SearchSpaceParameterConfig(type="float", low=1.0, high=30.0, log=True),
            "random_strength": SearchSpaceParameterConfig(type="float", low=1e-2, high=10.0, log=True),
            "subsample": SearchSpaceParameterConfig(type="float", low=0.5, high=1.0),
            "rsm": SearchSpaceParameterConfig(type="float", low=0.3, high=0.9),
            "min_data_in_leaf": SearchSpaceParameterConfig(type="int", low=10, high=200),
            "model_size_reg": SearchSpaceParameterConfig(type="float", low=0.0, high=10.0),
            "leaf_estimation_iterations": SearchSpaceParameterConfig(type="int", low=1, high=10),
        }
    raise ValueError(f"Unbekannter base_type {base_type}")


def _default_fmt_search_space(base_type: str) -> Dict[str, SearchSpaceParameterConfig]:
    """Default-Suchraum für Final-Model-Tuning (CATE-Regressor).

    BEWUSST konservativer als BL: weniger Bäume, flacher, stärkere
    Regularisierung. Verhindert Overfitting auf verrauschte Residuen."""
    base_type = (base_type or "").lower()
    if base_type == "lgbm":
        return {
            "n_estimators": SearchSpaceParameterConfig(type="int", low=100, high=400),
            "learning_rate": SearchSpaceParameterConfig(type="float", low=5e-3, high=1.2e-1, log=True),
            "num_leaves": SearchSpaceParameterConfig(type="int", low=7, high=63),
            "max_depth": SearchSpaceParameterConfig(type="int", low=2, high=6),
            "min_child_samples": SearchSpaceParameterConfig(type="int", low=20, high=500),
            "min_child_weight": SearchSpaceParameterConfig(type="float", low=0.5, high=80.0, log=True),
            "subsample": SearchSpaceParameterConfig(type="float", low=0.4, high=0.85),
            "colsample_bytree": SearchSpaceParameterConfig(type="float", low=0.25, high=0.7),
            "max_bin": SearchSpaceParameterConfig(type="int", low=15, high=127),
            "min_split_gain": SearchSpaceParameterConfig(type="float", low=0.0, high=3.0),
            "reg_alpha": SearchSpaceParameterConfig(type="float", low=0.1, high=50.0, log=True),
            "reg_lambda": SearchSpaceParameterConfig(type="float", low=1.0, high=100.0, log=True),
            "path_smooth": SearchSpaceParameterConfig(type="float", low=0.0, high=30.0),
        }
    if base_type == "catboost":
        return {
            "iterations": SearchSpaceParameterConfig(type="int", low=100, high=400),
            "learning_rate": SearchSpaceParameterConfig(type="float", low=5e-3, high=1.2e-1, log=True),
            "depth": SearchSpaceParameterConfig(type="int", low=2, high=6),
            "l2_leaf_reg": SearchSpaceParameterConfig(type="float", low=3.0, high=80.0, log=True),
            "random_strength": SearchSpaceParameterConfig(type="float", low=0.5, high=15.0, log=True),
            "subsample": SearchSpaceParameterConfig(type="float", low=0.4, high=0.85),
            "rsm": SearchSpaceParameterConfig(type="float", low=0.25, high=0.7),
            "min_data_in_leaf": SearchSpaceParameterConfig(type="int", low=20, high=500),
            "model_size_reg": SearchSpaceParameterConfig(type="float", low=0.1, high=30.0, log=True),
            "leaf_estimation_iterations": SearchSpaceParameterConfig(type="int", low=1, high=5),
        }
    raise ValueError(f"Unbekannter base_type {base_type}")


def _search_space_for(base_type: str, cfg_space: SearchSpaceConfig | None, is_fmt: bool = False) -> Dict[str, SearchSpaceParameterConfig]:
    base_type = (base_type or "").lower()
    custom = {}
    if cfg_space is not None:
        custom = dict(getattr(cfg_space, base_type, {}) or {})
    if custom:
        return custom
    return _default_fmt_search_space(base_type) if is_fmt else _default_search_space(base_type)


def _suggest_from_spec(trial, name: str, spec: SearchSpaceParameterConfig):
    if spec.type == "categorical":
        return trial.suggest_categorical(name, list(spec.choices or []))
    if spec.type == "int":
        step = None if spec.step is None else int(spec.step)
        return trial.suggest_int(name, int(spec.low), int(spec.high), step=step or 1, log=bool(spec.log))
    if spec.type == "float":
        kwargs = {"log": bool(spec.log)}
        if spec.step is not None and not spec.log:
            kwargs["step"] = float(spec.step)
        return trial.suggest_float(name, float(spec.low), float(spec.high), **kwargs)
    raise ValueError(f"Nicht unterstützter Parametertyp: {spec.type}")


def _apply_conditional_catboost_params(params: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(params)
    bootstrap_type = result.get("bootstrap_type")
    if bootstrap_type == "Bayesian":
        result.pop("subsample", None)
    elif bootstrap_type == "Bernoulli":
        result.pop("bagging_temperature", None)
    return result


def _suggest_params(trial, base_type: str, cfg_space: SearchSpaceConfig | None = None, is_fmt: bool = False) -> Dict[str, Any]:
    base_type = base_type.lower()

    # "Both"-Modus: Optuna wählt den Learner pro Trial als kategorische Dimension.
    # TPE lernt, welcher Learner für diesen Task besser ist, und konzentriert Trials
    # entsprechend. Der gewählte Learner wird in params["_learner_type"] gespeichert,
    # damit build_base_learner() später den richtigen Typ bauen kann.
    if base_type == "both":
        chosen_learner = trial.suggest_categorical("_learner_type", ["lgbm", "catboost"])
        params = _suggest_params(trial, chosen_learner, cfg_space, is_fmt=is_fmt)
        params["_learner_type"] = chosen_learner
        return params

    specs = _search_space_for(base_type, cfg_space, is_fmt=is_fmt)
    params: Dict[str, Any] = {}

    # Für CatBoost muss bootstrap_type vor bagging_temperature/subsample
    # gesampelt werden, damit die Abhängigkeit korrekt aufgelöst wird –
    # auch wenn die Search-Space-Definition (z. B. aus YAML) keine
    # deterministische Reihenfolge garantiert.
    if base_type == "catboost" and "bootstrap_type" in specs:
        params["bootstrap_type"] = _suggest_from_spec(trial, "bootstrap_type", specs["bootstrap_type"])

    for name, spec in specs.items():
        if name == "bootstrap_type" and base_type == "catboost":
            continue  # bereits oben gesampelt
        if base_type == "catboost":
            bootstrap_type = params.get("bootstrap_type")
            if name == "bagging_temperature" and bootstrap_type not in {None, "Bayesian"}:
                continue
            if name == "subsample" and bootstrap_type not in {None, "Bernoulli"}:
                continue
        params[name] = _suggest_from_spec(trial, name, spec)

    if base_type == "catboost":
        params = _apply_conditional_catboost_params(params)
    return params




@dataclass
class TunedSet:
    role_params: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class TuningTask:
    key: str
    objective_family: str
    estimator_task: str
    uses_treatment_feature: bool
    sample_scope: str
    target_name: str
    roles: Tuple[Tuple[str, str], ...]


class BaseLearnerTuner:
    """Optimiert Base-Learner-Aufgaben und teilt Ergebnisse zwischen kompatiblen Rollen.

    Die Logik ist task-basiert:
    - Aus den angeforderten kausalen Modellen werden die tatsächlich benötigten
      internen Lernaufgaben abgeleitet.
    - Identische Aufgaben werden nur einmal getunt.
    - Die besten Parameter werden anschließend allen passenden Rollen zugeordnet.
    """

    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg
        self.seed = int(cfg.constants.random_seed)
        self.optuna = _safe_import_optuna() if cfg.tuning.enabled else None
        self.best_scores: Dict[str, float] = {}
        self._blt_cv_folds = 1 if cfg.tuning.single_fold else (cfg.tuning.cv_splits or 5)

    def _create_study(self, study_key: str):
        optuna = self.optuna
        if optuna is None:
            raise RuntimeError("Optuna ist nicht verfügbar. Bitte optuna installieren.")

        # WICHTIG: Jede Study bekommt einen eigenen Seed, abgeleitet aus dem
        # Basis-Seed + study_key. Ohne dies schlägt der TPE-Sampler für verschiedene
        # Tasks (z. B. model_y vs. model_t) exakt dieselben Hyperparameter vor,
        # was zu identischen Tuning-Ergebnissen führt.
        # hashlib statt hash() für Determinismus über Python-Sessions hinweg.
        import hashlib
        base_seed = int(self.cfg.tuning.optuna_seed)
        key_hash = int(hashlib.sha256(study_key.encode()).hexdigest(), 16) % (2**31)
        study_seed = (base_seed + key_hash) % (2**31)

        # TPE-Sampler-Optimierung:
        # - multivariate=True: Modelliert Abhängigkeiten zwischen Parametern
        #   (z.B. learning_rate↔n_estimators). Besser als unabhängige Marginalverteilungen.
        # - constant_liar=True: Bei parallelen Trials (n_jobs>1) wird laufenden Trials
        #   ein Schätzwert zugewiesen, damit TPE nicht dieselben Regionen doppelt sampelt.
        # - n_startup_trials: Anzahl zufälliger Trials vor TPE-Exploration.
        #   Muss mindestens so groß wie parallel_jobs sein, damit TPE nach der
        #   ersten Welle genug abgeschlossene Datenpunkte zum Lernen hat.
        n_trials = int(self.cfg.tuning.n_trials)
        pj = self._tuning_n_jobs(self._blt_cv_folds)
        n_startup = max(pj, min(10, max(3, n_trials // 5)))
        # group=True: Partitioniert die Trial-Historie nach dem Parameter-Set des
        # jeweiligen Trials. Bei "both"-Modus wählt Optuna pro Trial entweder LGBM
        # oder CatBoost; die hyperparameter-Sets sind unterschiedlich (z.B. hat
        # CatBoost l2_leaf_reg, LGBM nicht). Ohne group fällt TPE bei diesen
        # "dynamischen" Params auf RandomSampler zurück, was die Sample-Qualität
        # deutlich verschlechtert. Mit group lernt TPE separate Modelle pro Zweig.
        try:
            sampler = optuna.samplers.TPESampler(
                seed=study_seed,
                multivariate=True,
                group=True,
                constant_liar=True,
                n_startup_trials=n_startup,
            )
        except TypeError:
            # group wurde in Optuna 2.8 eingeführt; bei älteren Versionen fallback
            try:
                sampler = optuna.samplers.TPESampler(
                    seed=study_seed,
                    multivariate=True,
                    constant_liar=True,
                    n_startup_trials=n_startup,
                )
            except TypeError:
                # Ältere Optuna-Version ohne multivariate/constant_liar
                try:
                    sampler = optuna.samplers.TPESampler(seed=study_seed)
                except Exception:
                    sampler = None

        # MedianPruner: Bricht Trials ab, deren Zwischenergebnis (pro Fold)
        # unter dem Median aller bisherigen Trials liegt. Spart ~30-50% der
        # Rechenzeit bei Multi-Fold-CV, da schlechte Parameterkombinationen
        # nach 1-2 Folds statt nach allen 5 abgebrochen werden.
        # n_warmup_steps=1: Erst ab dem 2. Fold prunen (1. Fold ist zu verrauscht).
        try:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=1)
        except Exception:
            pruner = None

        if self.cfg.tuning.storage_path:
            try:
                from pathlib import Path
                Path(self.cfg.tuning.storage_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            storage = f"sqlite:///{self.cfg.tuning.storage_path}"
            prefix = str(self.cfg.tuning.study_name_prefix).strip() or "baselearner"
            base_name = f"{prefix}__{study_key}"

            if bool(self.cfg.tuning.reuse_study_if_exists):
                return optuna.create_study(
                    study_name=base_name,
                    storage=storage,
                    direction="maximize",
                    load_if_exists=True,
                    sampler=sampler,
                    pruner=pruner,
                )

            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            return optuna.create_study(
                study_name=f"{base_name}__{ts}",
                storage=storage,
                direction="maximize",
                load_if_exists=False,
                sampler=sampler,
                pruner=pruner,
            )

        return optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def _role_signature(self, model_name: str, role: str) -> Tuple[str, str, bool, str, str]:
        """Signatur für Task-Sharing: (objective_family, estimator_task, uses_T, sample_scope, target).

        sample_scope unterscheidet, wie das Modell in Produktion trainiert wird:
        - "all": Innerhalb von DML/DR-internem Cross-Fitting (cv=dml_crossfit_folds, Default 5) → sieht ~80% des äußeren Folds.
        - "all_direct": Direkt auf dem äußeren Fold trainiert (Meta-Learner) → sieht ~100% des äußeren Folds.
        - "group_specific_shared_params": Pro Treatment-Gruppe trainiert.
        Diese Trennung verhindert, dass Modelle mit unterschiedlichen effektiven
        Trainingsmengen (40% vs 80% N) dasselbe Tuning teilen."""
        name = (model_name or "").lower()
        role = (role or "").lower()

        # ── Meta-Learner: Outcome-Modelle sind Regressoren ──
        # EconML's SLearner/TLearner/XLearner rufen model.predict() auf.
        # Regressor.predict() = E[Y|X] ∈ [0,1]; Classifier.predict() = {0,1}.
        # Meta-Learner haben kein internes CV → "all_direct" (80% N in Produktion).
        if name == "slearner" and role == "overall_model":
            return ("outcome_regression", "regressor", True, "all_direct", "Y")

        if name == "tlearner" and role == "models":
            return ("grouped_outcome_regression", "regressor", False, "group_specific_shared_params", "Y")

        if name == "xlearner":
            if role == "models":
                return ("grouped_outcome_regression", "regressor", False, "group_specific_shared_params", "Y")
            if role == "cate_models":
                return ("pseudo_effect", "regressor", False, "group_specific_shared_params", "D")
            if role == "propensity_model":
                # XLearner propensity wird direkt auf dem äußeren Fold trainiert (80% N),
                # während DML/DR-Propensity innerhalb von cv=dml_crossfit_folds trainiert wird (~64% N bei cv=5).
                return ("propensity", "classifier", False, "all_direct", "T")

        # ── DRLearner ──
        # Nuisance-Modelle laufen innerhalb DRLearner cv=dml_crossfit_folds → scope="all" (~64% N bei cv=5).
        if name == "drlearner":
            if role == "model_propensity":
                return ("propensity", "classifier", False, "all", "T")
            if role == "model_regression":
                return ("outcome_regression", "regressor", True, "all", "Y")

        # ── DML-Familie: Nuisance-Modelle sind Classifier ──
        # EconML wickelt predict_proba via discrete_outcome/discrete_treatment.
        # Laufen innerhalb DML cv=dml_crossfit_folds → scope="all" (~64% N bei cv=5).
        if name in {"nonparamdml", "paramdml", "causalforestdml"}:
            if role == "model_y":
                return ("outcome", "classifier", False, "all", "Y")
            if role == "model_t":
                return ("propensity", "classifier", False, "all", "T")

        raise KeyError(f"Keine Tuning-Signatur für {model_name}.{role} definiert.")

    def _roles_for_model(self, model_name: str) -> Dict[str, str]:
        name = model_name.lower()
        if name == "slearner":
            return {"overall_model": "y"}
        if name == "tlearner":
            return {"models": "y"}
        if name == "xlearner":
            return {"models": "y", "cate_models": "d", "propensity_model": "t"}
        if name == "drlearner":
            return {"model_propensity": "t", "model_regression": "y"}
        if name in {"nonparamdml", "paramdml", "causalforestdml"}:
            return {"model_y": "y", "model_t": "t"}
        return {}

    def _task_key(self, model_name: str, role: str) -> str:
        objective_family, estimator_task, uses_treatment_feature, sample_scope, target_name = self._role_signature(model_name, role)
        parts = [
            self.cfg.base_learner.type.lower(),
            objective_family,
            estimator_task,
            sample_scope,
            "with_t" if uses_treatment_feature else "no_t",
            target_name.lower(),
        ]
        if self.cfg.tuning.per_role:
            parts.append(role.lower())
        if self.cfg.tuning.per_learner:
            parts.append(model_name.lower())
        return "__".join(parts)

    def _build_plan(self, model_names: List[str]) -> Dict[str, TuningTask]:
        plan: Dict[str, TuningTask] = {}
        collectors: Dict[str, List[Tuple[str, str]]] = {}

        for model_name in model_names:
            roles = self._roles_for_model(model_name)
            for role in roles:
                key = self._task_key(model_name, role)
                collectors.setdefault(key, []).append((model_name, role))
                if key not in plan:
                    objective_family, estimator_task, uses_treatment_feature, sample_scope, target_name = self._role_signature(model_name, role)
                    plan[key] = TuningTask(
                        key=key,
                        objective_family=objective_family,
                        estimator_task=estimator_task,
                        uses_treatment_feature=uses_treatment_feature,
                        sample_scope=sample_scope,
                        target_name=target_name,
                        roles=tuple(),
                    )

        for key, roles in collectors.items():
            task = plan[key]
            plan[key] = TuningTask(
                key=task.key,
                objective_family=task.objective_family,
                estimator_task=task.estimator_task,
                uses_treatment_feature=task.uses_treatment_feature,
                sample_scope=task.sample_scope,
                target_name=task.target_name,
                roles=tuple(sorted(roles)),
            )
        return plan

    def _task_priority(self, task: TuningTask) -> int:
        """Tuning-Reihenfolge: Outcome/Regression zuerst, dann Propensity, dann Pseudo-Effekt."""
        order = {
            "outcome": 0, "outcome_regression": 1,
            "grouped_outcome": 2, "grouped_outcome_regression": 3,
            "propensity": 4, "pseudo_effect": 5,
        }
        return order.get(task.objective_family, 99)

    def _downsample_indices(self, model_name: str, T: np.ndarray, n: int) -> np.ndarray:
        """Gibt die Trainings-Indizes für ein Modell zurück.

        Historisch wurden hier modellspezifische Subsample-Anteile verwendet
        (LearnerDataUsageConfig). Seit deren Entfernung nutzen alle Modelle
        100% der Daten — die Methode existiert als Erweiterungspunkt."""
        return np.arange(n, dtype=int)

    def _combined_indices_for_task(self, task: TuningTask, T: np.ndarray, n_rows: int) -> np.ndarray:
        idx_sets: List[np.ndarray] = []
        for model_name, _ in task.roles:
            idx_sets.append(self._downsample_indices(model_name, T=T, n=n_rows))
        if not idx_sets:
            return np.arange(n_rows, dtype=int)
        return np.unique(np.concatenate(idx_sets)).astype(int)

    def _prepare_task_frame(self, task: TuningTask, X: pd.DataFrame, T: np.ndarray, indices: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        X_sub = X.iloc[indices]  # View, keine Kopie — to_numpy() in _tune_task erstellt Kopie
        T_sub = np.asarray(T)[indices].astype(int)
        if task.uses_treatment_feature:
            X_sub = X_sub.copy()  # Kopie nur wenn wir mutieren müssen
            X_sub["__treatment__"] = T_sub
        return X_sub, indices.astype(int), T_sub

    def _score_classifier(self, y_true: np.ndarray, proba: np.ndarray, multiclass: bool = False) -> float:
        """Bewertet einen Classifier anhand der konfigurierten Tuning-Metrik.

        Unterstützte Metriken (tuning.metric):
        - log_loss (Default): Negierter Log-Loss (höher = besser, da direction=maximize).
          Misst Kalibrierung — wichtig für DML-Residualisierung `Y - E[Y|X]`.
          Gut kalibrierte Nuisance-Predictions liefern informative Residuen.
        - pr_auc: Average Precision (PR-AUC). Misst Ranking-Qualität statt
          Kalibrierung. Robust bei Class Imbalance, aber für DML-Residualisierung
          weniger geeignet, da Ranking-optimale Vorhersagen nicht zwangsläufig
          kalibriert sind. Bei Multiclass: OvR-Mittel.
        - roc_auc: ROC-AUC, bei Multiclass roc_auc_ovr (weighted).
        - accuracy: Accuracy auf Basis der Klasse mit höchster Wahrscheinlichkeit.
        """
        metric = getattr(self.cfg.tuning, "metric", "log_loss")
        y_true = np.asarray(y_true).astype(int)
        if len(np.unique(y_true)) < 2:
            return 0.5 if metric in ("roc_auc", "pr_auc") else 0.0

        try:
            if metric == "pr_auc":
                from sklearn.metrics import average_precision_score
                if multiclass or len(np.unique(y_true)) > 2:
                    # Multiclass: One-vs-Rest macro-average
                    from sklearn.preprocessing import label_binarize
                    classes = np.unique(y_true)
                    y_bin = label_binarize(y_true, classes=classes)
                    if proba.ndim == 1 or proba.shape[1] < len(classes):
                        return 0.0
                    aps = []
                    for i, _ in enumerate(classes):
                        try:
                            aps.append(average_precision_score(y_bin[:, i], proba[:, i]))
                        except Exception:
                            pass
                    return float(np.mean(aps)) if aps else 0.0
                # Binary: proba für Klasse 1
                if proba.ndim == 2:
                    proba_pos = proba[:, 1]
                else:
                    proba_pos = proba
                return float(average_precision_score(y_true, proba_pos))

            if metric == "log_loss":
                from sklearn.metrics import log_loss as _log_loss
                # Negiert, da Optuna direction=maximize (weniger Loss = besser)
                if proba.ndim == 1:
                    proba_2d = np.column_stack([1 - proba, proba])
                else:
                    proba_2d = proba
                return -float(_log_loss(y_true, proba_2d))

            if metric == "accuracy":
                from sklearn.metrics import accuracy_score
                if proba.ndim == 1:
                    y_pred = (proba >= 0.5).astype(int)
                else:
                    y_pred = np.argmax(proba, axis=1)
                return float(accuracy_score(y_true, y_pred))

            # Default: roc_auc
            if multiclass or len(np.unique(y_true)) > 2:
                return float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted"))
            return float(roc_auc_score(y_true, proba))
        except Exception:
            return 0.5 if metric in ("roc_auc", "pr_auc") else 0.0

    def _fit_model(self, params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, estimator_task: str):
        pl = self.cfg.constants.parallel_level
        if pl <= 1:
            pj = 1
        elif pl <= 2:
            pj = -1  # alle Kerne, Trials sequentiell
        else:
            # Level 3/4: Mehrere Trials parallel → Kerne aufteilen
            n_cpus = available_cpu_count()
            n_trial_workers = self._tuning_n_jobs(self._blt_cv_folds)
            pj = max(1, n_cpus // max(1, n_trial_workers))
        model = build_base_learner(self.cfg.base_learner.type, params, seed=self.seed, task=estimator_task, parallel_jobs=pj)
        model.fit(X_train, y_train)
        return model

    def _cv_splits(self, labels: np.ndarray, single_fold: bool = False):
        """Erzeugt CV-Splits. Bei single_fold=True wird nur der erste Fold verwendet."""
        splits = _iter_stratified_or_kfold(labels, n_splits=self.cfg.tuning.cv_splits, seed=self.seed)
        if single_fold:
            # Nur den ersten Split zurückgeben (schneller, etwas verrauschter)
            for tr, va in splits:
                yield tr, va
                return
        else:
            yield from splits

    def _objective_all_classification(self, params: Dict[str, Any], X_mat: np.ndarray, target: np.ndarray, trial=None) -> float:
        scores: List[float] = []
        is_multiclass = len(np.unique(target)) > 2
        _op_active = getattr(self.cfg.tuning, "overfit_penalty", 0.0) > 0
        for fold_i, (tr, va) in enumerate(self._cv_splits(target.astype(int), single_fold=self.cfg.tuning.single_fold)):
            model = self._fit_model(params, X_mat[tr], target[tr].astype(int), "classifier")
            if is_multiclass:
                proba_val = model.predict_proba(X_mat[va])
                val_score = self._score_classifier(target[va], proba_val, multiclass=True)
                if _op_active:
                    proba_tr = model.predict_proba(X_mat[tr])
                    train_score = self._score_classifier(target[tr], proba_tr, multiclass=True)
                    val_score = self._apply_overfit_penalty(val_score, train_score)
            else:
                proba_val = model.predict_proba(X_mat[va])[:, 1]
                val_score = self._score_classifier(target[va], proba_val)
                if _op_active:
                    proba_tr = model.predict_proba(X_mat[tr])[:, 1]
                    train_score = self._score_classifier(target[tr], proba_tr)
                    val_score = self._apply_overfit_penalty(val_score, train_score)
            scores.append(val_score)
            # Optuna Pruning: nach jedem Fold Zwischenergebnis melden.
            # Unpromising Trials werden frühzeitig abgebrochen.
            if trial is not None:
                trial.report(float(np.mean(scores)), fold_i)
                if trial.should_prune():
                    raise self.optuna.TrialPruned()
        return float(np.mean(scores))

    # ── Regression-Objectives (für Meta-Learner + DRLearner model_regression) ──

    def _score_regressor(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Bewertet einen Regressor anhand der konfigurierten Tuning-Metrik.

        Unterstützte Metriken (tuning.metric_regression):
        - neg_mse (Default): Negierter MSE (höher = besser).
        - neg_rmse: Negierter RMSE.
        - neg_mae: Negierter MAE.
        - r2: R² (höher = besser, kein Negieren nötig).
        """
        metric = getattr(self.cfg.tuning, "metric_regression", "neg_mse")
        y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
        residuals = y_true - y_pred
        if metric == "neg_rmse":
            return -float(np.sqrt(np.mean(residuals ** 2)))
        if metric == "neg_mae":
            return -float(np.mean(np.abs(residuals)))
        if metric == "r2":
            ss_res = float(np.sum(residuals ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        # Default: neg_mse
        return -float(np.mean(residuals ** 2))

    def _apply_overfit_penalty(self, val_score: float, train_score: float,
                               penalty: float = None, tolerance: float = None) -> float:
        """Bestraft Overfit-Gap zwischen Train- und Val-Score.

        Motivation (aus Chernozhukov et al., 2016 / "Causal Inference for the
        Brave and True", Appendix):
        Nuisance-Modelle, die overfitten, absorbieren kausales Signal in den
        Residuals. Die OOF-Residuals Ỹ = Y - E[Y|X] werden systematisch zu
        klein, sodass model_final (CATE) weniger Heterogenität findet.

        Bei FMT: Analoges Problem — ein overfittendes model_final produziert
        gute Train-R-Scores, aber schlechte OOF-R-Scores. Der Gap misst
        direkt, ob model_final echte Heterogenität findet oder nur
        Trainingsrauschen gelernt hat.

        Formel (skalen-sicher, relative Tolerance):
            scale = |val_score|
            relative_gap = (train_score - val_score) / scale
            adjusted = val_score - penalty × scale × max(0, relative_gap - tolerance)

        tolerance ist ein relativer Anteil (z.B. 0.05 = 5% Gap wird toleriert).
        Damit funktioniert die Penalty identisch für alle Metriken (log_loss,
        R-Loss, DR-MSE etc.) unabhängig von deren Absolutskala.

        Parameter:
            val_score:   Score auf dem Validierungs-Fold (higher = better)
            train_score: Score auf dem Trainings-Fold (higher = better)
            penalty:     Penalty-Stärke (None → aus Config)
            tolerance:   Relativer Gap-Schwellwert (None → aus Config)
        """
        if penalty is None:
            penalty = getattr(self.cfg.tuning, "overfit_penalty", 0.0)
        if tolerance is None:
            tolerance = getattr(self.cfg.tuning, "overfit_tolerance", 0.05)
        if penalty <= 0:
            return val_score
        gap = train_score - val_score  # positiv = Overfitting
        scale = max(abs(val_score), 1e-10)
        relative_gap = gap / scale  # als Anteil der Score-Größenordnung
        return val_score - penalty * scale * max(0.0, relative_gap - tolerance)

    def _objective_all_regression(self, params: Dict[str, Any], X_mat: np.ndarray, target: np.ndarray, trial=None) -> float:
        """Tuning-Objective für Regressoren auf dem Gesamt-Datensatz (SLearner, DRLearner model_regression)."""
        scores: List[float] = []
        _op_active = getattr(self.cfg.tuning, "overfit_penalty", 0.0) > 0
        for fold_i, (tr, va) in enumerate(self._cv_splits(target.astype(float), single_fold=self.cfg.tuning.single_fold)):
            model = self._fit_model(params, X_mat[tr], target[tr].astype(float), "regressor")
            pred_val = model.predict(X_mat[va])
            val_score = self._score_regressor(target[va], pred_val)
            if _op_active:
                pred_tr = model.predict(X_mat[tr])
                train_score = self._score_regressor(target[tr], pred_tr)
                val_score = self._apply_overfit_penalty(val_score, train_score)
            scores.append(val_score)
            if trial is not None:
                trial.report(float(np.mean(scores)), fold_i)
                if trial.should_prune():
                    raise self.optuna.TrialPruned()
        return float(np.mean(scores)) if scores else -1.0

    def _objective_grouped_regression(self, params: Dict[str, Any], X_mat: np.ndarray, Y: np.ndarray, T: np.ndarray, trial=None) -> float:
        """Tuning-Objective für Regressoren pro Treatment-Gruppe (TLearner, XLearner)."""
        scores: List[float] = []
        _op_active = getattr(self.cfg.tuning, "overfit_penalty", 0.0) > 0
        K = len(np.unique(T))
        strat_labels = np.asarray(T).astype(int) * 10 + np.clip(np.asarray(Y), 0, 1).astype(int)
        for fold_i, (tr, va) in enumerate(self._cv_splits(strat_labels, single_fold=self.cfg.tuning.single_fold)):
            fold_scores: List[float] = []
            for group in range(K):
                tr_mask = T[tr] == group
                va_mask = T[va] == group
                if tr_mask.sum() < 2 or va_mask.sum() < 1:
                    continue
                y_train = Y[tr][tr_mask].astype(float)
                model = self._fit_model(params, X_mat[tr][tr_mask], y_train, "regressor")
                pred_val = model.predict(X_mat[va][va_mask])
                val_score = self._score_regressor(Y[va][va_mask], pred_val)
                if _op_active:
                    pred_tr = model.predict(X_mat[tr][tr_mask])
                    train_score = self._score_regressor(y_train, pred_tr)
                    val_score = self._apply_overfit_penalty(val_score, train_score)
                fold_scores.append(val_score)
            if fold_scores:
                scores.append(float(np.mean(fold_scores)))
            if trial is not None and scores:
                trial.report(float(np.mean(scores)), fold_i)
                if trial.should_prune():
                    raise self.optuna.TrialPruned()
        return float(np.mean(scores)) if scores else -1.0

    def _build_xlearner_pseudo_outcomes(self, X_mat: np.ndarray, Y: np.ndarray, T: np.ndarray, nuisance_params: Dict[str, Any]) -> np.ndarray:
        mu0 = np.zeros(len(Y), dtype=float)
        mu1 = np.zeros(len(Y), dtype=float)
        filled = np.zeros(len(Y), dtype=bool)
        strat_labels = np.asarray(T).astype(int) * 10 + np.asarray(Y).astype(int)

        for tr, va in _iter_stratified_or_kfold(strat_labels, n_splits=self.cfg.tuning.cv_splits, seed=self.seed):
            control_train = tr[T[tr] == 0]
            treated_train = tr[T[tr] == 1]
            if len(control_train) < 2 or len(treated_train) < 2:
                continue

            control_y = Y[control_train].astype(int)
            treated_y = Y[treated_train].astype(int)
            if len(np.unique(control_y)) < 2 or len(np.unique(treated_y)) < 2:
                continue

            m0 = self._fit_model(nuisance_params, X_mat[control_train], control_y, "classifier")
            m1 = self._fit_model(nuisance_params, X_mat[treated_train], treated_y, "classifier")
            mu0[va] = m0.predict_proba(X_mat[va])[:, 1]
            mu1[va] = m1.predict_proba(X_mat[va])[:, 1]
            filled[va] = True

        if not filled.all():
            control_idx = np.where(T == 0)[0]
            treated_idx = np.where(T == 1)[0]
            control_y = Y[control_idx].astype(int)
            treated_y = Y[treated_idx].astype(int)
            if len(control_idx) >= 2 and len(np.unique(control_y)) >= 2:
                m0 = self._fit_model(nuisance_params, X_mat[control_idx], control_y, "classifier")
                mu0[~filled] = m0.predict_proba(X_mat[~filled])[:, 1]
            if len(treated_idx) >= 2 and len(np.unique(treated_y)) >= 2:
                m1 = self._fit_model(nuisance_params, X_mat[treated_idx], treated_y, "classifier")
                mu1[~filled] = m1.predict_proba(X_mat[~filled])[:, 1]

        return np.where(T == 1, Y - mu0, mu1 - Y).astype(float)

    def _objective_xlearner_cate(self, params: Dict[str, Any], X_mat: np.ndarray, Y: np.ndarray, T: np.ndarray, nuisance_params: Dict[str, Any], trial=None) -> float:
        """Tuning-Objective für XLearner CATE-Modelle (cate_models).

        Bewertet Regressoren auf vorberechneten Pseudo-Outcomes pro Treatment-Gruppe.
        Nutzt _score_regressor (konfigurierbar via tuning.metric_regression) und
        optionale Overfit-Penalty (konsistent mit allen anderen BLT-Objectives).
        """
        pseudo = self._build_xlearner_pseudo_outcomes(X_mat, Y, T, nuisance_params=nuisance_params)
        scores: List[float] = []
        strat_labels = np.asarray(T).astype(int) * 10 + np.asarray(Y).astype(int)
        _op_active = getattr(self.cfg.tuning, "overfit_penalty", 0.0) > 0

        for fold_i, (tr, va) in enumerate(self._cv_splits(strat_labels, single_fold=self.cfg.tuning.single_fold)):
            fold_scores: List[float] = []
            for group in (0, 1):
                tr_mask = T[tr] == group
                va_mask = T[va] == group
                if tr_mask.sum() < 2 or va_mask.sum() < 1:
                    continue
                model = self._fit_model(params, X_mat[tr][tr_mask], pseudo[tr][tr_mask], "regressor")
                pred_val = np.asarray(model.predict(X_mat[va][va_mask]), dtype=float)
                val_score = self._score_regressor(pseudo[va][va_mask], pred_val)
                if _op_active:
                    pred_tr = np.asarray(model.predict(X_mat[tr][tr_mask]), dtype=float)
                    train_score = self._score_regressor(pseudo[tr][tr_mask], pred_tr)
                    val_score = self._apply_overfit_penalty(val_score, train_score)
                fold_scores.append(val_score)
            if fold_scores:
                scores.append(float(np.mean(fold_scores)))
            if trial is not None and scores:
                trial.report(float(np.mean(scores)), fold_i)
                if trial.should_prune():
                    raise self.optuna.TrialPruned()
        return float(np.mean(scores)) if scores else -1e12

    def _tune_task(self, task: TuningTask, X: pd.DataFrame, Y: np.ndarray, T: np.ndarray, shared_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        _is_both = (self.cfg.base_learner.type or "").lower() == "both"

        indices = self._combined_indices_for_task(task, T=T, n_rows=len(X))
        X_task, row_indices, T_task = self._prepare_task_frame(task, X=X, T=T, indices=indices)

        if task.target_name == "Y":
            target = np.asarray(Y)[row_indices]
        elif task.target_name == "T":
            target = np.asarray(T)[row_indices].astype(int)
        else:
            target = np.asarray(Y)[row_indices]

        max_rows = self.cfg.tuning.max_tuning_rows
        if max_rows is not None and len(X_task) > int(max_rows):
            rng = np.random.RandomState(self.seed)
            keep = np.asarray(rng.choice(np.arange(len(X_task)), size=int(max_rows), replace=False), dtype=int)
            X_task = X_task.iloc[keep]
            target = target[keep]
            row_indices = row_indices[keep]
            T_task = T_task[keep]
        X_mat = X_task.to_numpy()
        fixed_defaults = dict(self.cfg.base_learner.fixed_params or {})

        _tlog = logging.getLogger("rubin.tuning")
        _tlog.info(
            "Tuning-Task '%s': X_input=%d rows, indices=%d, X_task=%s, target=%s (unique=%s), "
            "T_task unique=%s, cv_splits=%d, target_name=%s, objective=%s",
            task.key, len(X), len(indices), X_mat.shape, target.shape,
            np.unique(target).tolist(), np.unique(T_task).tolist(),
            self.cfg.tuning.cv_splits, task.target_name, task.objective_family,
        )

        study = self._create_study(task.key)

        def objective(trial):
            params = _suggest_params(trial, self.cfg.base_learner.type, self.cfg.tuning.search_space)
            params = {**fixed_defaults, **params}

            # LGBM: subsample_freq=1 erzwingt Bagging in jedem Boosting-Schritt.
            # Ohne subsample_freq greift 'subsample' nur beim initialen Sampling,
            # nicht pro Iteration — das schwächt die Anti-Overfitting-Wirkung erheblich.
            # "both"-Modus: subsample_freq NUR setzen, wenn für diesen Trial LGBM gewählt
            # wurde (Optuna hat _learner_type="lgbm" geliefert).
            _bl = (self.cfg.base_learner.type or "").lower()
            _trial_is_lgbm = _bl == "lgbm" or (_bl == "both" and params.get("_learner_type") == "lgbm")
            if _trial_is_lgbm:
                params.setdefault("subsample_freq", 1)
            # Classifier-Objectives (Nuisance-Modelle: model_y, model_t, propensity)
            if task.objective_family in {"outcome", "propensity"}:
                return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), trial=trial)
            # Regressor-Objectives (Meta-Learner outcome, DRLearner model_regression)
            if task.objective_family == "outcome_regression":
                return self._objective_all_regression(params, X_mat=X_mat, target=target.astype(float), trial=trial)
            if task.objective_family == "grouped_outcome_regression":
                return self._objective_grouped_regression(params, X_mat=X_mat, Y=target.astype(float), T=T_task.astype(int), trial=trial)
            # XLearner CATE-Modelle
            if task.objective_family == "pseudo_effect":
                nuisance = dict(shared_params.get("xlearner__models") or fixed_defaults)
                return self._objective_xlearner_cate(
                    params,
                    X_mat=X_mat,
                    Y=np.asarray(Y)[row_indices].astype(int),
                    T=T_task.astype(int),
                    nuisance_params=nuisance,
                    trial=trial,
                )
            raise ValueError(f"Unbekannte objective_family={task.objective_family!r}")

        # RCT: Propensity-Tuning auf 20 Trials reduzieren (Diagnose-Check statt volles Tuning)
        _n_trials = int(self.cfg.tuning.n_trials)
        _is_rct = getattr(self.cfg, "study_type", "rct") == "rct"
        if _is_rct and task.objective_family == "propensity":
            _n_trials = min(_n_trials, 20)

        # "Both"-Modus: Search-Space ist verdoppelt (beide Learner-Familien).
        # Trials werden entsprechend verdoppelt, damit pro Familie die gleiche
        # Tuning-Qualität wie im Single-Learner-Fall erreicht wird.
        _is_both = (self.cfg.base_learner.type or "").lower() == "both"
        if _is_both:
            _n_trials *= 2

        # Timeout: Bei "both"-Modus wird der Timeout deaktiviert, weil
        # LGBM (DART-Boosting) systematisch langsamer ist als CatBoost.
        # Ein Timeout würde dazu führen, dass weniger LGBM-Trials fertig werden
        # → unfairer Vergleich. n_trials ist hier die primäre Steuerung.
        _timeout = self.cfg.tuning.timeout_seconds
        if _is_both and _timeout:
            logging.getLogger("rubin.tuning").warning(
                "timeout_seconds=%d wird bei base_learner.type='both' ignoriert, "
                "damit beide Learner-Familien gleich viele Trials durchlaufen. "
                "Steuerung erfolgt ausschließlich über n_trials=%d.",
                _timeout, _n_trials,
            )
            _timeout = None

        study.optimize(objective, n_trials=_n_trials, timeout=_timeout,
                       n_jobs=self._tuning_n_jobs(self._blt_cv_folds), catch=(Exception,))
        _tlog = logging.getLogger("rubin.tuning")
        # ── Trial-Diagnose: Abgeschlossene, fehlgeschlagene, geprunte Trials ──
        _pj = self._tuning_n_jobs(self._blt_cv_folds)
        _states = {}
        for tr in study.trials:
            sname = tr.state.name
            _states[sname] = _states.get(sname, 0) + 1
        n_complete = _states.get("COMPLETE", 0)
        n_fail = _states.get("FAIL", 0)
        n_pruned = _states.get("PRUNED", 0)
        _tlog.info(
            "BLT '%s': %d/%d Trials abgeschlossen (%d fehlgeschlagen, %d gepruned, parallel=%d).",
            task.key, n_complete, len(study.trials), n_fail, n_pruned, _pj,
        )
        # ── Fehlerdiagnose: Top-Fehler nach Typ gruppiert ──
        if n_fail > 0:
            error_groups: dict = {}
            for tr in study.trials:
                if tr.state.name == "FAIL":
                    reason = (tr.system_attrs or {}).get("fail_reason", "Unbekannt")
                    # Extrahiere die letzte Zeile (= eigentliche Fehlermeldung)
                    lines = [l.strip() for l in reason.strip().split("\n") if l.strip()]
                    short = lines[-1] if lines else "Unbekannt"
                    # Gruppieren nach Fehlertyp (erste 120 Zeichen)
                    key = short[:120]
                    if key not in error_groups:
                        error_groups[key] = {"count": 0, "full_example": reason}
                    error_groups[key]["count"] += 1
            _tlog.warning(
                "BLT '%s': %d/%d Trials FEHLGESCHLAGEN. Fehlertypen:", task.key, n_fail, len(study.trials),
            )
            for err_key, info in sorted(error_groups.items(), key=lambda x: -x[1]["count"]):
                _tlog.warning(
                    "  [%d×] %s", info["count"], err_key,
                )
            # Vollständigen Traceback des häufigsten Fehlers loggen
            most_common = max(error_groups.values(), key=lambda x: x["count"])
            _tlog.warning(
                "BLT '%s': Häufigster Fehler — vollständiger Traceback:\n%s",
                task.key, most_common["full_example"][:2000],
            )

        # RCT-Warnung: Wenn Propensity-Modell Treatment vorhersagen kann, ist die
        # Randomisierung möglicherweise verletzt oder Post-Treatment-Variablen im Datensatz.
        if _is_rct and task.objective_family == "propensity":
            try:
                best_val = float(study.best_value)
                # Schwellwerte je Metrik (alle entsprechen grob AUC > 0.65):
                #   log_loss:  log_loss < 0.55  → score = -log_loss > -0.55
                #   pr_auc:    AP > 0.65        → score > 0.65
                #                (Referenz-AP = Positiv-Rate; ~50% bei balanciertem RCT)
                #   roc_auc:   AUC > 0.65       → score > 0.65
                #   accuracy:  Acc > 0.65       → score > 0.65
                current_metric = getattr(self.cfg.tuning, "metric", "log_loss") or "log_loss"
                if current_metric == "log_loss":
                    _flag = best_val > -0.55
                else:
                    _flag = best_val > 0.65
                if _flag:
                    _tlog.warning(
                        "RCT-Warnung: Propensity-Modell erreicht %s=%.4f (bei randomisiertem "
                        "Treatment sollte das Modell nicht besser als Zufall sein). "
                        "Prüfe ob Treatment tatsächlich randomisiert ist oder ob Post-Treatment-"
                        "Variablen im Datensatz sind.",
                        current_metric, best_val,
                    )
            except Exception:
                pass
        try:
            self.best_scores[task.key] = float(study.best_value)
        except Exception:
            pass
        try:
            return {**fixed_defaults, **dict(study.best_params)}
        except ValueError:
            _tlog.warning(
                "Tuning-Task '%s': Keine abgeschlossenen Trials. Verwende Default-Parameter.",
                task.key,
            )
            return dict(fixed_defaults)

    def _tuning_n_jobs(self, cv_folds: int = 1) -> int:
        """Anzahl paralleler Optuna-Trials basierend auf parallel_level.

        Level 1-2: 1 (sequentiell, alle Kerne an den einzelnen Fit)
        Level 3-4: n_cpus // 4 (je 4 Kerne pro Trial, gleiche Anzahl
                   für CatBoost und LightGBM → vergleichbare TPE-Exploration)

        Bei K-Fold CV (cv_folds > 1) wird die Parallelität halbiert, weil
        jeder Trial K Fits sequentiell durchläuft und dabei K× länger
        Speicher belegt. Ohne Reduktion kann es bei großen Datensätzen
        zu OOM-Fehlern kommen, weil zu viele langlebige Trials gleichzeitig
        laufen und CatBoost/LGBM interne Puffer akkumulieren.
        """
        pl = self.cfg.constants.parallel_level
        if pl <= 2:
            return 1
        n_cpus = available_cpu_count()
        n_jobs = max(1, n_cpus // 4)
        if cv_folds > 1:
            n_jobs = max(1, n_jobs // 2)
        return n_jobs

    def tune_all(self, model_names: List[str], X: pd.DataFrame, Y: np.ndarray, T: np.ndarray) -> Dict[str, Dict[str, Dict[str, Any]]]:
        if not self.cfg.tuning.enabled:
            return {}

        # Modellfilter: Nur für ausgewählte Modelle Tuning-Tasks generieren.
        # Nicht-ausgewählte Modelle nutzen base_learner.fixed_params.
        tuning_models = self.cfg.tuning.models
        if tuning_models is not None:
            active_models = [m for m in model_names if m in tuning_models]
            skipped = [m for m in model_names if m not in tuning_models]
        else:
            active_models = list(model_names)
            skipped = []

        _tlog = logging.getLogger("rubin.tuning")
        # CPU-Diagnose: available_cpu_count() berücksichtigt cgroup/affinity
        _n_effective = available_cpu_count()
        _n_raw = os.cpu_count() or 1
        _cpu_diag = f"effective={_n_effective}"
        if _n_raw != _n_effective:
            _cpu_diag += f", os.cpu_count={_n_raw}"
        try:
            _n_aff = len(os.sched_getaffinity(0))
            if _n_aff != _n_effective:
                _cpu_diag += f", affinity={_n_aff}"
        except (AttributeError, OSError):
            pass
        _tlog.info(
            "tune_all gestartet: models=%s, X=%s, Y=%s (unique=%s), T=%s (unique=%s), "
            "cv_splits=%d, n_trials=%d, parallel_trials=%d, "
            "parallel_level=%d, cores_per_fit=%d, CPU=[%s]",
            active_models, X.shape, np.asarray(Y).shape, np.unique(Y).tolist(),
            np.asarray(T).shape, np.unique(T).tolist(),
            self.cfg.tuning.cv_splits, self.cfg.tuning.n_trials,
            self._tuning_n_jobs(self._blt_cv_folds),
            self.cfg.constants.parallel_level,
            max(1, _n_effective // max(1, self._tuning_n_jobs(self._blt_cv_folds))),
            _cpu_diag,
        )
        if skipped:
            _tlog.info(
                "BLT-Modellfilter: %s übersprungen (nutzen fixed_params). Aktiv: %s",
                skipped, active_models,
            )
        if not self.cfg.tuning.single_fold and self.cfg.constants.parallel_level >= 3:
            _full_pj = max(1, _n_effective // 4)
            _red_pj = self._tuning_n_jobs(self._blt_cv_folds)
            if _red_pj < _full_pj:
                _tlog.info(
                    "K-Fold aktiv (cv=%d): Parallele Trials reduziert %d → %d "
                    "(Speicherschutz: jeder Trial hält %d Fold-Fits gleichzeitig).",
                    self.cfg.tuning.cv_splits or 5, _full_pj, _red_pj, self.cfg.tuning.cv_splits or 5,
                )

        if any((m or "").lower() == "causalforestdml" for m in active_models):
            logging.getLogger("rubin.tuning").debug(
                "CausalForestDML erkannt: Wald-Parameter werden über EconML tune() bestimmt, "
                "nicht über Optuna. Optuna optimiert nur die Nuisance-Modelle."
            )

        plan = self._build_plan(active_models)
        tuned_by_task: Dict[str, Dict[str, Any]] = {}

        for task in sorted(plan.values(), key=self._task_priority):
            best = self._tune_task(task, X=X, Y=Y, T=T, shared_params=tuned_by_task)
            tuned_by_task[task.key] = best
            if task.objective_family == "grouped_outcome_regression" and any(m.lower() == "xlearner" for m, _ in task.roles):
                tuned_by_task.setdefault("xlearner__models", best)

        tuned_by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for task in plan.values():
            best = tuned_by_task.get(task.key, {})
            for model_name, role in task.roles:
                tuned_by_model.setdefault(model_name, {})
                tuned_by_model[model_name][role] = dict(best)

        for model_name, roles in tuned_by_model.items():
            if roles:
                # HINWEIS: "default" wird als Fallback für Rollen genutzt, die NICHT
                # explizit getunt wurden. Das betrifft v.a. model_final (CATE-Regression).
                # model_registry._base() verhindert, dass model_final diese Classifier-
                # Defaults erbt — model_final nutzt nur base_fixed_params oder explizit
                # getunte model_final-Params (via FinalModelTuning).
                roles.setdefault("default", next(iter(roles.values())))
        return tuned_by_model

    def tune(self, model_name: str, X: pd.DataFrame, Y: np.ndarray, T: np.ndarray) -> TunedSet:
        tuned = self.tune_all([model_name], X=X, Y=Y, T=T)
        return TunedSet(role_params=dict(tuned.get(model_name, {}) or {}))

    # ── Combined-Loss-Diagnostic (Bach et al., 2024) ──────────────────────
    # Die Combined Loss misst das Produkt der Nuisance-Fehler (Outcome × Propensity).
    # Ein niedriger Wert korreliert mit besserer kausaler Schätzqualität.
    # Wird NACH dem Tuning als Post-hoc-Diagnostic berechnet, nicht als Tuning-Metrik.

class FinalModelTuner:
    """Tuning des Final-Modells (model_final) über OOF-CV.

Beide unterstützten Modelle (NonParamDML, DRLearner) werden über ihre nativen
EconML est.score()-Methoden bewertet:

- **NonParamDML:** est.score() = R-Loss (Nie & Wager, 2021): E[(Y_res − τ·T_res)²].
  Misst, wie gut τ(X) die kausale Varianz in den Residuen erklärt.
- **DRLearner:** est.score() = DR-MSE: doubly-robust Pseudo-Outcome-MSE.
  Passt zum internen Trainings-Loss des DR model_final.

Beide Metriken sind kausal valide, nutzen aber unterschiedliche Dekompositionsansätze.
Der Vorteil nativer Metriken: Kein Train-Eval-Mismatch (das Modell wird mit
derselben Metrik evaluiert, auf die es intern optimiert).

Architektur:
- Äußere CV-Folds (K = cross_validation_splits): est.fit(train) + est.score(val)
- model_final wird komplett Out-of-Fold evaluiert → keine optimistischen Scores
- Optionale Overfit-Penalty: Skalen-sichere relative Tolerance

Score-Konvention:
Beide Metriken sind lower=better, werden für Optuna direction="maximize" negiert.

Overfit-Penalty (skalen-sicher):
Die Penalty nutzt relative Tolerance: tolerance=0.05 = "5% relative Gap toleriert".
Damit funktioniert die Penalty identisch für R-Loss (~0.001) und DR-MSE (~0.01)
ohne manuelle Skaleneinstellung.

Locking-Regel:
Die gefundenen Hyperparameter werden als tuned_params persistiert und
in allen Cross-Prediction-Folds wiederverwendet."""

    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg
        self.seed = int(cfg.constants.random_seed)
        self.optuna = _safe_import_optuna() if cfg.final_model_tuning.enabled else None
        self.best_scores: Dict[str, float] = {}

    def _apply_overfit_penalty(self, val_score: float, train_score: float,
                               penalty: float = 0.0, tolerance: float = 0.05) -> float:
        """Bestraft Overfit-Gap zwischen Train- und Val-Score (skalen-sicher).

        Formel (relative Tolerance):
            scale = |val_score|
            relative_gap = (train_score - val_score) / scale
            adjusted = val_score - penalty × scale × max(0, relative_gap - tolerance)

        tolerance ist ein relativer Anteil (z.B. 0.05 = 5% Gap wird toleriert).
        Funktioniert identisch für R-Loss und DR-MSE unabhängig von Absolutskala.
        """
        if penalty <= 0:
            return val_score
        gap = train_score - val_score  # positiv = Overfitting
        scale = max(abs(val_score), 1e-10)
        relative_gap = gap / scale
        return val_score - penalty * scale * max(0.0, relative_gap - tolerance)

    def _tuning_n_jobs(self, cv_folds: int = 1) -> int:
        """Anzahl paralleler Optuna-Trials basierend auf parallel_level.

        FMT-Trials sind deutlich schwerer als BLT-Trials: Jeder Trial fittet
        ein vollständiges kausalmodell (NonParamDML/DRLearner) mit internem
        Cross-Fitting (K Folds × model_y + model_t + model_final). Der
        RAM-Bedarf pro Trial ist ~5-10× höher als bei BLT.
        → Weniger parallele Trials, dafür mehr Kerne pro Fit.

        Level 1-2: 1 (sequentiell)
        Level 3-4: max(2, n_cpus // 8) — halb so viel wie BLT

        Bei K-Fold (cv_folds > 1) wird nochmals halbiert (wie beim BLT).
        """
        pl = self.cfg.constants.parallel_level
        if pl <= 2:
            return 1
        n_cpus = available_cpu_count()
        n = max(2, n_cpus // 8)
        if cv_folds > 1:
            n = max(1, n // 2)
        return n

    def _parallel_jobs_per_fit(self) -> int:
        """Kerne pro Base-Learner-Fit, berücksichtigt parallele Trials."""
        pl = self.cfg.constants.parallel_level
        if pl <= 1:
            return 1
        if pl <= 2:
            return -1
        n_cpus = available_cpu_count()
        n_trial_workers = self._tuning_n_jobs(
            1 if self.cfg.final_model_tuning.single_fold
            else (self.cfg.data_processing.cross_validation_splits or 5)
        )
        return max(1, n_cpus // max(1, n_trial_workers))

    def _create_study(self, study_key: str):
        optuna = self.optuna
        if optuna is None:
            raise RuntimeError("Optuna ist nicht verfügbar. Bitte optuna installieren.")
        import hashlib
        base_seed = int(self.cfg.tuning.optuna_seed)
        key_hash = int(hashlib.sha256(study_key.encode()).hexdigest(), 16) % (2**31)
        study_seed = (base_seed + key_hash) % (2**31)
        n_trials = int(self.cfg.final_model_tuning.n_trials)
        n_startup = min(7, max(2, n_trials // 4))
        # group=True: siehe BL-Tuning für Erläuterung (conditional search spaces)
        try:
            sampler = optuna.samplers.TPESampler(
                seed=study_seed,
                multivariate=True,
                group=True,
                constant_liar=True,
                n_startup_trials=n_startup,
            )
        except TypeError:
            try:
                sampler = optuna.samplers.TPESampler(
                    seed=study_seed,
                    multivariate=True,
                    constant_liar=True,
                    n_startup_trials=n_startup,
                )
            except TypeError:
                try:
                    sampler = optuna.samplers.TPESampler(seed=study_seed)
                except Exception:
                    sampler = None
        try:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=1)
        except Exception:
            pruner = None
        return optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def _build_classifier(self, base_type: str, fixed: Dict[str, Any], tuned: Dict[str, Any]) -> Any:
        params = dict(fixed or {})
        params.update(tuned or {})
        return build_base_learner(base_type, params, seed=self.seed, task="classifier", parallel_jobs=self._parallel_jobs_per_fit())

    def _build_regressor(self, base_type: str, fixed: Dict[str, Any], tuned: Dict[str, Any]) -> Any:
        params = dict(fixed or {})
        params.update(tuned or {})
        return build_base_learner(base_type, params, seed=self.seed, task="regressor", parallel_jobs=self._parallel_jobs_per_fit())

    def _log_trial_diagnostics(self, study, label: str):
        """Loggt Trial-Diagnose: Fehlertypen gruppiert, vollständiger Traceback des häufigsten."""
        _tlog = logging.getLogger("rubin.tuning")
        _states = {}
        for tr in study.trials:
            sname = tr.state.name
            _states[sname] = _states.get(sname, 0) + 1
        n_complete = _states.get("COMPLETE", 0)
        n_fail = _states.get("FAIL", 0)
        n_pruned = _states.get("PRUNED", 0)
        _tlog.info(
            "FMT '%s': %d/%d Trials abgeschlossen (%d fehlgeschlagen, %d gepruned).",
            label, n_complete, len(study.trials), n_fail, n_pruned,
        )
        if n_fail > 0:
            error_groups: dict = {}
            for tr in study.trials:
                if tr.state.name == "FAIL":
                    reason = (tr.system_attrs or {}).get("fail_reason", "Unbekannt")
                    lines = [l.strip() for l in reason.strip().split("\n") if l.strip()]
                    short = lines[-1] if lines else "Unbekannt"
                    key = short[:120]
                    if key not in error_groups:
                        error_groups[key] = {"count": 0, "full_example": reason}
                    error_groups[key]["count"] += 1
            _tlog.warning("FMT '%s': %d/%d Trials FEHLGESCHLAGEN. Fehlertypen:", label, n_fail, len(study.trials))
            for err_key, info in sorted(error_groups.items(), key=lambda x: -x[1]["count"]):
                _tlog.warning("  [%d×] %s", info["count"], err_key)
            most_common = max(error_groups.values(), key=lambda x: x["count"])
            _tlog.warning("FMT '%s': Häufigster Fehler — vollständiger Traceback:\n%s", label, most_common["full_example"][:2000])

    def tune_final_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        Y: np.ndarray,
        T: np.ndarray,
        base_type: str,
        base_fixed_params: Dict[str, Any],
        tuned_roles: Dict[str, Dict[str, Any]],
        crosspred_splits: int,
        fmt_fixed_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Ermittelt getunte Parameter für die Rolle 'model_final'.

        base_fixed_params: Für Nuisance-Modelle (model_y, model_t, model_propensity).
        fmt_fixed_params: Für model_final (final_model_tuning.fixed_params). Falls None → {}."""
        if fmt_fixed_params is None:
            fmt_fixed_params = {}

        if not self.cfg.final_model_tuning.enabled:
            return {}

        name = (model_name or "").lower()
        if name not in {"nonparamdml", "drlearner"}:
            return {}


        _tlog = logging.getLogger("rubin.tuning")

        # Locking: Falls model_final bereits vorhanden ist, nichts tun.
        if (tuned_roles.get("model_final") or {}) != {}:
            return {}

        optuna = self.optuna
        if optuna is None:
            return {}

        tr_idx = _first_crossfit_train_indices(len(X), T, n_splits=int(crosspred_splits), seed=self.seed, y=Y)
        X_tune = X.iloc[tr_idx]
        Y_tune = Y[tr_idx]
        T_tune = T[tr_idx]

        max_rows = self.cfg.final_model_tuning.max_tuning_rows
        if max_rows is not None and len(X_tune) > int(max_rows):
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(np.arange(len(X_tune)), size=int(max_rows), replace=False)
            X_tune = X_tune.iloc[idx]
            Y_tune = Y_tune[idx]
            T_tune = T_tune[idx]

        role = "model_final"
        base_type = (base_type or "lgbm").lower()

        if name == "nonparamdml":
            from econml.dml import NonParamDML

            study = self._create_study(f"final__{model_name}__{base_type}")
            _op = float(getattr(self.cfg.final_model_tuning, "overfit_penalty", 0.0))
            _ot = float(getattr(self.cfg.final_model_tuning, "overfit_tolerance", 0.05))
            _sf = bool(getattr(self.cfg.final_model_tuning, "single_fold", False))

            # Äußere CV-Folds: model_final wird auf Train gefittet und auf
            # komplett OOF-Val bewertet. Ohne äußere CV wäre die R-Score-
            # Schätzung optimistisch, weil model_final auf denselben Daten
            # evaluiert wird, auf denen es trainiert wurde.
            _outer_splits = int(self.cfg.data_processing.cross_validation_splits)

            def objective(trial):
                cand_params = _suggest_params(trial, base_type, self.cfg.final_model_tuning.search_space, is_fmt=True)
                if base_type == "lgbm" or (base_type == "both" and cand_params.get("_learner_type") == "lgbm"):
                    cand_params.setdefault("subsample_freq", 1)

                fold_scores_raw = []
                fold_scores_adj = []

                split_iter = _iter_stratified_or_kfold(
                    labels=T_tune.astype(int),
                    n_splits=_outer_splits,
                    seed=self.seed,
                )
                for fold_i, (tr, va) in enumerate(split_iter):
                    est = NonParamDML(
                        model_y=self._build_classifier(base_type, base_fixed_params, tuned_roles.get("model_y", {})),
                        model_t=self._build_classifier(base_type, base_fixed_params, tuned_roles.get("model_t", {})),
                        model_final=self._build_regressor(base_type, fmt_fixed_params, cand_params),
                        discrete_treatment=True, discrete_outcome=True,
                        cv=int(self.cfg.final_model_tuning.cv_splits),
                        mc_iters=self.cfg.data_processing.mc_iters,
                        mc_agg=self.cfg.data_processing.mc_agg,
                        random_state=self.seed,
                    )
                    est.fit(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr])
                    # NonParamDML.score() = R-Loss (Nie & Wager): E[(Y_res − τ·T_res)²].
                    # Nativ korrekt — das IST R-Loss. Negiert für direction="maximize".
                    val_score = -float(est.score(Y_tune[va], T_tune[va], X=X_tune.iloc[va]))
                    fold_scores_raw.append(val_score)

                    adjusted = val_score
                    if _op > 0:
                        train_score = -float(est.score(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr]))
                        adjusted = self._apply_overfit_penalty(val_score, train_score, penalty=_op, tolerance=_ot)
                    fold_scores_adj.append(adjusted)

                    trial.report(float(np.mean(fold_scores_adj)), fold_i)
                    if trial.should_prune():
                        raise self.optuna.TrialPruned()

                    if _sf:
                        break

                trial.set_user_attr("r_score_raw", float(np.mean(fold_scores_raw)))
                return float(np.mean(fold_scores_adj))

            _fmt_pj = self._tuning_n_jobs(1 if self.cfg.final_model_tuning.single_fold else (self.cfg.data_processing.cross_validation_splits or 5))
            _fmt_nt = int(self.cfg.final_model_tuning.n_trials)
            # "Both"-Modus: Search-Space verdoppelt — Trials auch verdoppeln.
            # Timeout wird bei "both" deaktiviert (LGBM-DART langsamer als CatBoost).
            _fmt_is_both = (self.cfg.base_learner.type or "").lower() == "both"
            if _fmt_is_both:
                _fmt_nt *= 2
            _fmt_timeout = self.cfg.final_model_tuning.timeout_seconds
            if _fmt_is_both and _fmt_timeout:
                logging.getLogger("rubin.tuning").warning(
                    "FMT timeout_seconds=%d wird bei 'both' ignoriert (faire Trial-Allokation).", _fmt_timeout)
                _fmt_timeout = None
            _tlog.info(
                "FMT '%s': Starte %d Trials (parallel=%d, Wellen≈%d).",
                model_name, _fmt_nt, _fmt_pj, max(1, _fmt_nt // max(1, _fmt_pj)),
            )
            study.optimize(objective, n_trials=_fmt_nt,
                           timeout=_fmt_timeout,
                           n_jobs=_fmt_pj, catch=(Exception,))
            self._log_trial_diagnostics(study, model_name)
            try:
                raw = study.best_trial.user_attrs.get("r_score_raw", study.best_value)
                self.best_scores[f"final__{model_name}"] = raw
                if _op > 0:
                    self.best_scores[f"final__{model_name}__adjusted"] = float(study.best_value)
            except Exception:
                pass
            try:
                return {role: dict(study.best_trial.params)}
            except ValueError:
                _tlog.warning("FMT '%s': Keine abgeschlossenen Trials. Verwende Default-Parameter.", model_name)
                return {}

        if name == "drlearner":
            from econml.dr import DRLearner

            study = self._create_study(f"final__{model_name}__{base_type}")
            _op = float(getattr(self.cfg.final_model_tuning, "overfit_penalty", 0.0))
            _ot = float(getattr(self.cfg.final_model_tuning, "overfit_tolerance", 0.05))
            _sf = bool(getattr(self.cfg.final_model_tuning, "single_fold", False))
            _outer_splits = int(self.cfg.data_processing.cross_validation_splits)

            def objective(trial):
                cand_params = _suggest_params(trial, base_type, self.cfg.final_model_tuning.search_space, is_fmt=True)
                if base_type == "lgbm" or (base_type == "both" and cand_params.get("_learner_type") == "lgbm"):
                    cand_params.setdefault("subsample_freq", 1)

                fold_scores_raw = []
                fold_scores_adj = []

                split_iter = _iter_stratified_or_kfold(
                    labels=T_tune.astype(int),
                    n_splits=_outer_splits,
                    seed=self.seed,
                )
                for fold_i, (tr, va) in enumerate(split_iter):
                    est = DRLearner(
                        model_propensity=self._build_classifier(base_type, base_fixed_params, tuned_roles.get("model_propensity", {})),
                        model_regression=self._build_regressor(base_type, base_fixed_params, tuned_roles.get("model_regression", {})),
                        model_final=self._build_regressor(base_type, fmt_fixed_params, cand_params),
                        cv=int(self.cfg.final_model_tuning.cv_splits),
                        mc_iters=self.cfg.data_processing.mc_iters,
                        mc_agg=self.cfg.data_processing.mc_agg,
                        random_state=self.seed,
                    )
                    est.fit(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr])
                    # DRLearner.score() = DR-MSE (doubly-robust Pseudo-Outcome-MSE).
                    # Nativ korrekt — die Metrik passt zum internen Trainings-Loss
                    # des DR model_final. Negiert für direction="maximize".
                    # (NonParamDML.score() = R-Loss — dort ist est.score() ebenfalls nativ.)
                    val_score = -float(est.score(Y_tune[va], T_tune[va], X=X_tune.iloc[va]))
                    fold_scores_raw.append(val_score)

                    adjusted = val_score
                    if _op > 0:
                        train_score = -float(est.score(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr]))
                        adjusted = self._apply_overfit_penalty(val_score, train_score, penalty=_op, tolerance=_ot)
                    fold_scores_adj.append(adjusted)

                    trial.report(float(np.mean(fold_scores_adj)), fold_i)
                    if trial.should_prune():
                        raise self.optuna.TrialPruned()

                    if _sf:
                        break

                trial.set_user_attr("r_score_raw", float(np.mean(fold_scores_raw)))
                return float(np.mean(fold_scores_adj))

            _fmt_pj = self._tuning_n_jobs(1 if self.cfg.final_model_tuning.single_fold else (self.cfg.data_processing.cross_validation_splits or 5))
            _fmt_nt = int(self.cfg.final_model_tuning.n_trials)
            # "Both"-Modus: Search-Space verdoppelt — Trials auch verdoppeln.
            # Timeout wird bei "both" deaktiviert (LGBM-DART langsamer als CatBoost).
            _fmt_is_both = (self.cfg.base_learner.type or "").lower() == "both"
            if _fmt_is_both:
                _fmt_nt *= 2
            _fmt_timeout = self.cfg.final_model_tuning.timeout_seconds
            if _fmt_is_both and _fmt_timeout:
                logging.getLogger("rubin.tuning").warning(
                    "FMT timeout_seconds=%d wird bei 'both' ignoriert (faire Trial-Allokation).", _fmt_timeout)
                _fmt_timeout = None
            _tlog.info(
                "FMT '%s': Starte %d Trials (parallel=%d, Wellen≈%d).",
                model_name, _fmt_nt, _fmt_pj, max(1, _fmt_nt // max(1, _fmt_pj)),
            )
            study.optimize(objective, n_trials=_fmt_nt,
                           timeout=_fmt_timeout,
                           n_jobs=_fmt_pj, catch=(Exception,))
            self._log_trial_diagnostics(study, model_name)
            try:
                raw = study.best_trial.user_attrs.get("r_score_raw", study.best_value)
                self.best_scores[f"final__{model_name}"] = raw
                if _op > 0:
                    self.best_scores[f"final__{model_name}__adjusted"] = float(study.best_value)
            except Exception:
                pass
            try:
                return {role: dict(study.best_trial.params)}
            except ValueError:
                _tlog.warning("FMT '%s': Keine abgeschlossenen Trials. Verwende Default-Parameter.", model_name)
                return {}

        return {}
