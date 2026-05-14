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
from sklearn.metrics import log_loss as _log_loss_fn
import logging
import os
import gc


def _compute_skill_score(best_neg_score: float, target: np.ndarray, is_classification: bool) -> float:
    """Berechnet den Skill Score (Verbesserung gegenüber naivem Baseline-Modell).

    Klassifikation: skill = 1 - log_loss / baseline_log_loss
      → baseline = naives Modell, das immer die Klassenverteilung vorhersagt
      → 0.0 = nicht besser als Zufall, 1.0 = perfekt, <0 = schlechter als Zufall

    Regression: skill = R² = 1 - MSE / Var(Y)
      → baseline = Mittelwert-Modell
      → 0.0 = nicht besser als Mittelwert, 1.0 = perfekt, <0 = schlechter

    Args:
        best_neg_score: Negierter Score aus Optuna (neg_log_loss oder neg_mse)
        target: Ziel-Array des Tasks
        is_classification: True für Klassifikation, False für Regression
    """
    try:
        model_loss = -best_neg_score  # Optuna maximiert neg_loss
        if is_classification:
            y = target.astype(int)
            p = np.clip(y.mean(), 1e-15, 1 - 1e-15)
            baseline = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        else:
            baseline = float(np.var(target.astype(float)))
        if baseline < 1e-15:
            return 0.0
        return 1.0 - model_loss / baseline
    except Exception:
        return 0.0
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
    # Default: GBDT-Boosting (Standard-Modus, stabil und schnell).
    fixed["boosting_type"] = "gbdt"
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
    # Default: GBDT-Boosting (Standard-Modus, stabil und schnell).
    fixed["boosting_type"] = "gbdt"
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


# ── CatBoost CPU-Optimierungen ──────────────────────────────────────────
# max_ctr_complexity=1: Keine automatischen Feature-Kombinationen.
#   Default ist 4 → bei 74 kat. Features: >1 Mio. Kombinationen → extrem langsam.
#   Mit 1: Nur individuelle OTS pro Feature (bleibt erhalten), keine Paare/Tripel.
#   CatBoost GitHub #1859: "If I raise max_ctr_complexity this can get to multiple hours."
# one_hot_max_size=2: CatBoost CPU-Default beibehalten. Features mit exakt 2
#   einzigartigen Werten bekommen One-Hot, alle anderen OTS.
_CATBOOST_CPU_OPTS = {
    "max_ctr_complexity": 1,
    "one_hot_max_size": 2,     # CatBoost CPU-Default beibehalten
}


def _build_catboost_classifier(params: Dict[str, Any], seed: int, parallel_jobs: int = -1):
    from catboost import CatBoostClassifier  # type: ignore
    fixed = dict(
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    fixed.update(_CATBOOST_CPU_OPTS)
    fixed.update(params)
    # thread_count NACH params setzen, damit fixed_params es nicht überschreiben können.
    fixed["thread_count"] = parallel_jobs
    # KRITISCH: CatBoost bootstrap_type='Bayesian' (MultiClass-Default) ist NICHT kompatibel mit
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
    fixed.update(_CATBOOST_CPU_OPTS)
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
            "min_child_weight": SearchSpaceParameterConfig(type="float", low=1e-2, high=50.0),
            "subsample": SearchSpaceParameterConfig(type="float", low=0.5, high=1.0),
            "colsample_bytree": SearchSpaceParameterConfig(type="float", low=0.3, high=0.9),
            "max_bin": SearchSpaceParameterConfig(type="int", low=15, high=127),
            "min_split_gain": SearchSpaceParameterConfig(type="float", low=0.0, high=1.0),
            "reg_alpha": SearchSpaceParameterConfig(type="float", low=1e-6, high=20.0),
            "reg_lambda": SearchSpaceParameterConfig(type="float", low=1e-6, high=20.0),
            "path_smooth": SearchSpaceParameterConfig(type="float", low=0.0, high=10.0),
        }
    if base_type == "catboost":
        ss = {
            "iterations": SearchSpaceParameterConfig(type="int", low=200, high=600),
            "learning_rate": SearchSpaceParameterConfig(type="float", low=1e-2, high=1.5e-1, log=True),
            "depth": SearchSpaceParameterConfig(type="int", low=4, high=8),
            "l2_leaf_reg": SearchSpaceParameterConfig(type="float", low=1.0, high=30.0),
            "random_strength": SearchSpaceParameterConfig(type="float", low=1e-2, high=10.0),
            "subsample": SearchSpaceParameterConfig(type="float", low=0.5, high=1.0),
            "rsm": SearchSpaceParameterConfig(type="float", low=0.3, high=0.9),
            "min_data_in_leaf": SearchSpaceParameterConfig(type="int", low=10, high=200),
            "model_size_reg": SearchSpaceParameterConfig(type="float", low=0.0, high=10.0),
            "leaf_estimation_iterations": SearchSpaceParameterConfig(type="int", low=1, high=10),
        }
        return ss
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
            "min_child_weight": SearchSpaceParameterConfig(type="float", low=0.5, high=80.0),
            "subsample": SearchSpaceParameterConfig(type="float", low=0.4, high=0.85),
            "colsample_bytree": SearchSpaceParameterConfig(type="float", low=0.25, high=0.7),
            "max_bin": SearchSpaceParameterConfig(type="int", low=15, high=127),
            "min_split_gain": SearchSpaceParameterConfig(type="float", low=0.0, high=3.0),
            "reg_alpha": SearchSpaceParameterConfig(type="float", low=0.1, high=50.0),
            "reg_lambda": SearchSpaceParameterConfig(type="float", low=1.0, high=100.0),
            "path_smooth": SearchSpaceParameterConfig(type="float", low=0.0, high=30.0),
        }
    if base_type == "catboost":
        ss = {
            "iterations": SearchSpaceParameterConfig(type="int", low=100, high=400),
            "learning_rate": SearchSpaceParameterConfig(type="float", low=5e-3, high=1.2e-1, log=True),
            "depth": SearchSpaceParameterConfig(type="int", low=2, high=6),
            "l2_leaf_reg": SearchSpaceParameterConfig(type="float", low=3.0, high=80.0),
            "random_strength": SearchSpaceParameterConfig(type="float", low=0.5, high=15.0),
            "subsample": SearchSpaceParameterConfig(type="float", low=0.4, high=0.85),
            "rsm": SearchSpaceParameterConfig(type="float", low=0.25, high=0.7),
            "min_data_in_leaf": SearchSpaceParameterConfig(type="int", low=20, high=500),
            "model_size_reg": SearchSpaceParameterConfig(type="float", low=0.1, high=30.0),
            "leaf_estimation_iterations": SearchSpaceParameterConfig(type="int", low=1, high=5),
        }
        return ss
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
    train_subsample_ratio: float = 1.0  # scope="all" → (K-1)/K, sonst 1.0


def _log_trial_diagnostics(study, label: str, n_jobs: int = 1, stage: str = "BLT"):
    """Loggt Trial-Diagnose: Abgeschlossene/fehlgeschlagene/geprunte Trials, Fehlertypen.

    Wird von BLT, FMT und CFT einheitlich aufgerufen.
    """
    _tlog = logging.getLogger("rubin.tuning")
    _states = {}
    for tr in study.trials:
        sname = tr.state.name
        _states[sname] = _states.get(sname, 0) + 1
    n_complete = _states.get("COMPLETE", 0)
    n_fail = _states.get("FAIL", 0)
    n_pruned = _states.get("PRUNED", 0)
    _tlog.info(
        "%s '%s': %d/%d Trials abgeschlossen (%d fehlgeschlagen, %d gepruned, parallel=%d).",
        stage, label, n_complete, len(study.trials), n_fail, n_pruned, n_jobs,
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
        _tlog.warning("%s '%s': %d/%d Trials FEHLGESCHLAGEN. Fehlertypen:", stage, label, n_fail, len(study.trials))
        for err_key, info in sorted(error_groups.items(), key=lambda x: -x[1]["count"]):
            _tlog.warning("  [%d×] %s", info["count"], err_key)
        most_common = max(error_groups.values(), key=lambda x: x["count"])
        _tlog.warning("%s '%s': Häufigster Fehler — vollständiger Traceback:\n%s", stage, label, most_common["full_example"][:2000])
    return n_complete, n_fail, n_pruned


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
        # Separater Seed für Tuning-CV, damit Fold-Zuordnung ≠ Cross-Prediction-Folds.
        self.tuning_cv_seed = int(cfg.constants.tuning_seed)
        if self.tuning_cv_seed == self.seed:
            logging.getLogger("rubin.tuning").warning(
                "tuning_seed (%d) == random_seed (%d): Tuning-CV und Cross-Prediction "
                "nutzen identische Folds → Val-Set-Overfitting möglich! "
                "Empfehlung: unterschiedliche Seeds setzen.", self.tuning_cv_seed, self.seed)
        self.optuna = _safe_import_optuna() if cfg.tuning.enabled else None
        self.best_scores: Dict[str, float] = {}
        self.skill_scores: Dict[str, float] = {}

    def _create_study(self, study_key: str, effective_n_trials: int = None):
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
        # - consider_endpoints=True: Verbessert Exploration an Suchraum-Grenzen.
        #   Wichtig wenn Optimum am Rand liegt (z.B. min_samples_leaf am oberen Rand).
        # - n_startup_trials: Basiert auf EFFEKTIVEN Trials pro Task (nicht global).
        #   Bei Propensity-Tasks (20 Trials) muss n_startup < n_trials sein,
        #   sonst ist alles Random Search und TPE wird nie aktiviert.
        #   Gekappt auf max. die Hälfte der Trials.
        _nt = effective_n_trials or int(self.cfg.tuning.n_trials)
        pj = self._tuning_n_jobs()
        n_startup = max(pj, min(10, max(3, _nt // 5)))
        n_startup = min(n_startup, max(3, _nt // 2))
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
                consider_endpoints=True,
                n_startup_trials=n_startup,
            )
        except TypeError:
            # group wurde in Optuna 2.8 eingeführt; bei älteren Versionen fallback
            try:
                sampler = optuna.samplers.TPESampler(
                    seed=study_seed,
                    multivariate=True,
                    constant_liar=True,
                    consider_endpoints=True,
                    n_startup_trials=n_startup,
                )
            except TypeError:
                try:
                    sampler = optuna.samplers.TPESampler(seed=study_seed)
                except Exception:
                    sampler = None

        # MedianPruner: Bricht Trials ab, deren Zwischenergebnis (pro Fold)
        # unter dem Median aller bisherigen Trials liegt. Spart ~30-50% der
        # Rechenzeit bei Multi-Fold-CV, da schlechte Parameterkombinationen
        # nach 2-3 Folds statt nach allen 5 abgebrochen werden.
        # n_warmup_steps=2: Erst ab dem 3. Fold prunen (1-2 Folds zu verrauscht).
        try:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=2)
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

        sample_scope dokumentiert, wie das Modell in Produktion trainiert wird:
        - "all": Innerhalb von DML/DR-internem Cross-Fitting (cv=dml_crossfit_folds, Default 5) → sieht ~80% des äußeren Folds.
        - "all_direct": Direkt auf dem äußeren Fold trainiert (Meta-Learner) → sieht ~100% des äußeren Folds.
        - "group_specific_shared_params": Pro Treatment-Gruppe trainiert.

        "all" und "all_direct" werden NICHT zusammengelegt, da sie durch
        train_subsample_ratio verschiedene effektive Trainingsmengen haben
        (scope="all" → (K-1)/K Subsampling, scope="all_direct" → kein Subsampling).
        "group_specific_shared_params" bleibt ebenfalls separat."""
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
                # DRLearner mit discrete_outcome=True → Classifier (predict_proba).
                # BLT und Training nutzen beide Logloss für optimale Hyperparameter.
                return ("outcome", "classifier", True, "all", "Y")

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
        # Kein Scope-Merge: "all" (DML/DR-Nuisance) und "all_direct" (Meta-Learner)
        # haben durch train_subsample_ratio verschiedene effektive Trainingsmengen
        # (64% vs 80% bei K=5). Getrennte Tuning-Tasks sind korrekt.
        effective_scope = sample_scope
        parts = [
            self.cfg.base_learner.type.lower(),
            objective_family,
            estimator_task,
            effective_scope,
            "with_t" if uses_treatment_feature else "no_t",
            target_name.lower(),
        ]
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
                    # scope="all": Nuisance-Modell wird in Produktion innerhalb von
                    # DML/DR-Cross-Fitting trainiert → sieht nur (K-1)/K der äußeren
                    # Fold-Daten. BLT simuliert das durch Subsampling der Trainingsmenge.
                    _inner_k = int(self.cfg.data_processing.dml_crossfit_folds)
                    _subsample = (_inner_k - 1) / _inner_k if sample_scope == "all" else 1.0
                    plan[key] = TuningTask(
                        key=key,
                        objective_family=objective_family,
                        estimator_task=estimator_task,
                        uses_treatment_feature=uses_treatment_feature,
                        sample_scope=sample_scope,
                        target_name=target_name,
                        roles=tuple(),
                        train_subsample_ratio=_subsample,
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
                train_subsample_ratio=task.train_subsample_ratio,
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
        """Bewertet einen Classifier via Log-Loss (negiert, höher = besser).

        Log-Loss misst die Kalibrierung der Nuisance-Vorhersagen P(T|X) und P(Y|X).
        Kalibrierung ist eine notwendige Bedingung für unverzerrte CATE-Schätzungen
        bei DML-Residualisierung (Bach et al., 2024). Andere Metriken (AUC, Accuracy)
        messen nur Diskrimination, nicht Kalibrierung.
        """
        y_true = np.asarray(y_true).astype(int)
        if len(np.unique(y_true)) < 2:
            return 0.0

        try:
            if proba.ndim == 1:
                proba_2d = np.column_stack([1 - proba, proba])
            else:
                proba_2d = proba
            return -float(_log_loss_fn(y_true, proba_2d))
        except Exception:
            return 0.0

    def _fit_model(self, params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, estimator_task: str):
        pl = self.cfg.constants.parallel_level
        if pl <= 1:
            pj = 1
        elif pl <= 2:
            pj = -1  # alle Kerne, Trials sequentiell
        else:
            # Level 3/4: Mehrere Trials parallel → Kerne aufteilen
            n_cpus = available_cpu_count()
            n_trial_workers = self._tuning_n_jobs()
            pj = max(1, n_cpus // max(1, n_trial_workers))
        model = build_base_learner(self.cfg.base_learner.type, params, seed=self.seed, task=estimator_task, parallel_jobs=pj)
        model.fit(X_train, y_train)
        return model

    def _cv_splits(self, labels: np.ndarray, single_fold: bool = False, train_subsample_ratio: float = 1.0):
        """Erzeugt CV-Splits. Bei single_fold=True wird nur der erste Fold verwendet.
        
        train_subsample_ratio < 1.0: Subsamplet die Trainingsindizes, um das
        Produktions-Datenregime zu simulieren. Bei scope="all" (DML/DR-Nuisance)
        sehen Modelle in Produktion nur (K-1)/K der äußeren Fold-Daten durch
        internes Cross-Fitting. BLT simuliert das hier."""
        splits = _iter_stratified_or_kfold(labels, n_splits=self.cfg.tuning.cv_splits, seed=self.tuning_cv_seed)
        for fold_i, (tr, va) in enumerate(splits):
            if train_subsample_ratio < 1.0 and len(tr) > 10:
                rng = np.random.RandomState(self.seed + fold_i)
                n_sub = max(10, int(len(tr) * train_subsample_ratio))
                tr = rng.choice(tr, size=n_sub, replace=False)
            yield tr, va
            if single_fold:
                return

    def _objective_all_classification(self, params: Dict[str, Any], X_mat: np.ndarray, target: np.ndarray, strata: np.ndarray, train_ratio: float = 1.0, trial=None) -> float:
        val_scores: List[float] = []
        train_scores: List[float] = []
        is_multiclass = len(np.unique(target)) > 2
        _op_active = getattr(self.cfg.tuning, "overfit_penalty", 0.0) > 0
        for fold_i, (tr, va) in enumerate(self._cv_splits(strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=train_ratio)):
            model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(int), "classifier")
            if is_multiclass:
                proba_val = model.predict_proba(X_mat.iloc[va])
                val_score = self._score_classifier(target[va], proba_val, multiclass=True)
                if _op_active:
                    proba_tr = model.predict_proba(X_mat.iloc[tr])
                    train_scores.append(self._score_classifier(target[tr], proba_tr, multiclass=True))
            else:
                proba_val = model.predict_proba(X_mat.iloc[va])[:, 1]
                val_score = self._score_classifier(target[va], proba_val)
                if _op_active:
                    proba_tr = model.predict_proba(X_mat.iloc[tr])[:, 1]
                    train_scores.append(self._score_classifier(target[tr], proba_tr))
            val_scores.append(val_score)
            del model  # CatBoost-nativen Speicher sofort freigeben
            # Pruning auf RAW Val-Score — Penalty darf Pruning nicht beeinflussen,
            # weil per-Fold-Gaps verrauscht sind und vielversprechende Trials
            # voreilig abbrechen würden.
            if trial is not None:
                trial.report(float(np.mean(val_scores)), fold_i)
                if trial.should_prune():
                    raise self.optuna.TrialPruned()
        mean_val = float(np.mean(val_scores)) if val_scores else -1e12
        if trial is not None:
            trial.set_user_attr("raw_val_score", mean_val)
        # Penalty auf STABILE Mittelwerte (nicht verrauschte Einzelfolds)
        if _op_active and train_scores:
            return self._apply_overfit_penalty(mean_val, float(np.mean(train_scores)))
        return mean_val

    # ── Regression-Objectives (für Meta-Learner Outcome-Modelle) ──

    def _score_regressor(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Bewertet einen Regressor via neg. MSE (höher = besser).

        MSE auf binären Targets Y∈{0,1} entspricht dem Brier Score und misst
        direkt die Kalibrierung von P(Y|X,T). Für DML-Residualisierung ist
        Kalibrierung entscheidend — analog zu Log-Loss bei Klassifikation.
        """
        y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
        return -float(np.mean((y_true - y_pred) ** 2))

    def _apply_overfit_penalty(self, val_score: float, train_score: float,
                               penalty: float = None, tolerance: float = None) -> float:
        """Bestraft Overfit-Gap zwischen Train- und Val-Score (skalen-sicher, relative Tolerance).

        Wird NUR auf den finalen Trial-Score angewendet (K-Fold-Mittelwerte),
        NICHT auf Einzelfold-Scores. Pruning basiert auf Raw-Val-Scores.

        Formel:
            scale = |val_score|
            relative_gap = (train_score - val_score) / scale
            adjusted = val_score - penalty × scale × max(0, relative_gap - tolerance)

        tolerance ist ein relativer Anteil (z.B. 0.15 = 15% Gap wird toleriert
        bei BLT, 0.05 = 5% bei FMT). Skalen-sicher über alle Metriken.
        """
        if penalty is None:
            penalty = getattr(self.cfg.tuning, "overfit_penalty", 0.0)
        if tolerance is None:
            tolerance = getattr(self.cfg.tuning, "overfit_tolerance", 0.15)
        if penalty <= 0:
            return val_score
        gap = train_score - val_score  # positiv = Overfitting
        scale = max(abs(val_score), 1e-10)
        relative_gap = gap / scale  # als Anteil der Score-Größenordnung
        return val_score - penalty * scale * max(0.0, relative_gap - tolerance)

    def _objective_all_regression(self, params: Dict[str, Any], X_mat: np.ndarray, target: np.ndarray, strata: np.ndarray, train_ratio: float = 1.0, trial=None) -> float:
        """Tuning-Objective für Regressoren auf dem Gesamt-Datensatz (SLearner overall_model).

        Pruning auf RAW Val-Score, Penalty auf stabile K-Fold-Mittelwerte.
        """
        val_scores: List[float] = []
        train_scores: List[float] = []
        _op_active = getattr(self.cfg.tuning, "overfit_penalty", 0.0) > 0
        for fold_i, (tr, va) in enumerate(self._cv_splits(strata, single_fold=self.cfg.tuning.single_fold, train_subsample_ratio=train_ratio)):
            model = self._fit_model(params, X_mat.iloc[tr], target[tr].astype(float), "regressor")
            pred_val = model.predict(X_mat.iloc[va])
            val_scores.append(self._score_regressor(target[va], pred_val))
            if _op_active:
                pred_tr = model.predict(X_mat.iloc[tr])
                train_scores.append(self._score_regressor(target[tr], pred_tr))
            del model  # CatBoost-nativen Speicher sofort freigeben
            if trial is not None:
                trial.report(float(np.mean(val_scores)), fold_i)
                if trial.should_prune():
                    raise self.optuna.TrialPruned()
        mean_val = float(np.mean(val_scores)) if val_scores else -1.0
        if trial is not None:
            trial.set_user_attr("raw_val_score", mean_val)
        if _op_active and train_scores:
            return self._apply_overfit_penalty(mean_val, float(np.mean(train_scores)))
        return mean_val

    def _objective_grouped_regression(self, params: Dict[str, Any], X_mat: np.ndarray, Y: np.ndarray, T: np.ndarray, trial=None) -> float:
        """Tuning-Objective für Regressoren pro Treatment-Gruppe (TLearner, XLearner)."""
        val_scores: List[float] = []
        train_scores: List[float] = []
        _op_active = getattr(self.cfg.tuning, "overfit_penalty", 0.0) > 0
        K = len(np.unique(T))
        strat_labels = np.asarray(T).astype(int) * 10 + np.clip(np.asarray(Y), 0, 1).astype(int)
        for fold_i, (tr, va) in enumerate(self._cv_splits(strat_labels, single_fold=self.cfg.tuning.single_fold)):
            fold_vals: List[float] = []
            fold_trains: List[float] = []
            for group in range(K):
                tr_mask = T[tr] == group
                va_mask = T[va] == group
                if tr_mask.sum() < 2 or va_mask.sum() < 1:
                    continue
                y_train = Y[tr][tr_mask].astype(float)
                model = self._fit_model(params, X_mat.iloc[tr][tr_mask], y_train, "regressor")
                pred_val = model.predict(X_mat.iloc[va][va_mask])
                fold_vals.append(self._score_regressor(Y[va][va_mask], pred_val))
                if _op_active:
                    pred_tr = model.predict(X_mat.iloc[tr][tr_mask])
                    fold_trains.append(self._score_regressor(y_train, pred_tr))
                del model  # CatBoost-nativen Speicher sofort freigeben
            if fold_vals:
                val_scores.append(float(np.mean(fold_vals)))
            if fold_trains:
                train_scores.append(float(np.mean(fold_trains)))
            if trial is not None and val_scores:
                trial.report(float(np.mean(val_scores)), fold_i)
                if trial.should_prune():
                    raise self.optuna.TrialPruned()
        mean_val = float(np.mean(val_scores)) if val_scores else -1.0
        if trial is not None:
            trial.set_user_attr("raw_val_score", mean_val)
        if _op_active and train_scores:
            return self._apply_overfit_penalty(mean_val, float(np.mean(train_scores)))
        return mean_val

    def _build_xlearner_pseudo_outcomes(self, X_mat: np.ndarray, Y: np.ndarray, T: np.ndarray, nuisance_params: Dict[str, Any]) -> np.ndarray:
        mu0 = np.zeros(len(Y), dtype=float)
        mu1 = np.zeros(len(Y), dtype=float)
        filled = np.zeros(len(Y), dtype=bool)
        strat_labels = np.asarray(T).astype(int) * 10 + np.asarray(Y).astype(int)

        for tr, va in _iter_stratified_or_kfold(strat_labels, n_splits=self.cfg.tuning.cv_splits, seed=self.tuning_cv_seed):
            control_train = tr[T[tr] == 0]
            treated_train = tr[T[tr] == 1]
            if len(control_train) < 2 or len(treated_train) < 2:
                continue

            control_y = Y[control_train].astype(int)
            treated_y = Y[treated_train].astype(int)
            if len(np.unique(control_y)) < 2 or len(np.unique(treated_y)) < 2:
                continue

            m0 = self._fit_model(nuisance_params, X_mat.iloc[control_train], control_y, "classifier")
            m1 = self._fit_model(nuisance_params, X_mat.iloc[treated_train], treated_y, "classifier")
            mu0[va] = m0.predict_proba(X_mat.iloc[va])[:, 1]
            mu1[va] = m1.predict_proba(X_mat.iloc[va])[:, 1]
            filled[va] = True
            del m0, m1  # CatBoost-nativen Speicher sofort freigeben

        if not filled.all():
            control_idx = np.where(T == 0)[0]
            treated_idx = np.where(T == 1)[0]
            control_y = Y[control_idx].astype(int)
            treated_y = Y[treated_idx].astype(int)
            if len(control_idx) >= 2 and len(np.unique(control_y)) >= 2:
                m0 = self._fit_model(nuisance_params, X_mat.iloc[control_idx], control_y, "classifier")
                mu0[~filled] = m0.predict_proba(X_mat.iloc[~filled])[:, 1]
            if len(treated_idx) >= 2 and len(np.unique(treated_y)) >= 2:
                m1 = self._fit_model(nuisance_params, X_mat.iloc[treated_idx], treated_y, "classifier")
                mu1[~filled] = m1.predict_proba(X_mat.iloc[~filled])[:, 1]

        return np.where(T == 1, Y - mu0, mu1 - Y).astype(float)

    def _objective_xlearner_cate(self, params: Dict[str, Any], X_mat: np.ndarray, Y: np.ndarray, T: np.ndarray, nuisance_params: Dict[str, Any], trial=None) -> float:
        """Tuning-Objective für XLearner CATE-Modelle (cate_models).

        Bewertet Regressoren auf vorberechneten Pseudo-Outcomes pro Treatment-Gruppe.
        Nutzt _score_regressor (neg. MSE) und
        optionale Overfit-Penalty (konsistent mit allen anderen BLT-Objectives).

        Pruning basiert auf RAW Val-Scores. Penalty wirkt nur auf den finalen
        Trial-Score (stabile Mittelwerte statt verrauschte Einzelfolds).
        """
        pseudo = self._build_xlearner_pseudo_outcomes(X_mat, Y, T, nuisance_params=nuisance_params)
        val_scores: List[float] = []
        train_scores: List[float] = []
        strat_labels = np.asarray(T).astype(int) * 10 + np.asarray(Y).astype(int)
        _op_active = getattr(self.cfg.tuning, "overfit_penalty", 0.0) > 0

        for fold_i, (tr, va) in enumerate(self._cv_splits(strat_labels, single_fold=self.cfg.tuning.single_fold)):
            fold_vals: List[float] = []
            fold_trains: List[float] = []
            for group in (0, 1):
                tr_mask = T[tr] == group
                va_mask = T[va] == group
                if tr_mask.sum() < 2 or va_mask.sum() < 1:
                    continue
                model = self._fit_model(params, X_mat.iloc[tr][tr_mask], pseudo[tr][tr_mask], "regressor")
                pred_val = np.asarray(model.predict(X_mat.iloc[va][va_mask]), dtype=float)
                fold_vals.append(self._score_regressor(pseudo[va][va_mask], pred_val))
                if _op_active:
                    pred_tr = np.asarray(model.predict(X_mat.iloc[tr][tr_mask]), dtype=float)
                    fold_trains.append(self._score_regressor(pseudo[tr][tr_mask], pred_tr))
                del model  # CatBoost-nativen Speicher sofort freigeben
            if fold_vals:
                val_scores.append(float(np.mean(fold_vals)))
            if fold_trains:
                train_scores.append(float(np.mean(fold_trains)))
            if trial is not None and val_scores:
                trial.report(float(np.mean(val_scores)), fold_i)
                if trial.should_prune():
                    raise self.optuna.TrialPruned()
        mean_val = float(np.mean(val_scores)) if val_scores else -1e12
        if trial is not None:
            trial.set_user_attr("raw_val_score", mean_val)
        if _op_active and train_scores:
            return self._apply_overfit_penalty(mean_val, float(np.mean(train_scores)))
        return mean_val

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
        X_mat = X_task  # DataFrame behalten — CatBoost/LGBM nutzen native Dtypes
        fixed_defaults = dict(self.cfg.base_learner.fixed_params or {})

        _tlog = logging.getLogger("rubin.tuning")
        _tlog.info(
            "BLT '%s': X=%s, target=%s (unique=%s), subsample=%.0f%%, cv=%d, objective=%s",
            task.key, X_mat.shape, target.shape,
            np.unique(target).tolist(), task.train_subsample_ratio * 100,
            self.cfg.tuning.cv_splits, task.objective_family,
        )

        # Effektive Trial-Zahl VORAB berechnen (benötigt für n_startup im Sampler)
        _n_trials = int(self.cfg.tuning.n_trials)
        _is_rct = getattr(self.cfg, "study_type", "rct") == "rct"
        if _is_rct and task.objective_family == "propensity":
            _n_trials = min(_n_trials, 20)
        _is_both = (self.cfg.base_learner.type or "").lower() == "both"
        if _is_both:
            _n_trials *= 2

        study = self._create_study(task.key, effective_n_trials=_n_trials)

        _train_ratio = task.train_subsample_ratio
        # T×Y-Strata für StratifiedKFold (wie in Cross-Prediction)
        _t_arr = np.asarray(T)[row_indices].astype(int)
        _y_arr = target.astype(int) if target.dtype in (int, np.int32, np.int64, bool) else np.zeros(len(target), dtype=int)
        _strata = (_t_arr * 10 + _y_arr)  # Kombiniert T und Y für Stratifizierung

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
            # Classifier-Objectives (Nuisance: model_y, model_t, propensity + DRLearner model_regression)
            if task.objective_family in {"outcome", "propensity"}:
                return self._objective_all_classification(params, X_mat=X_mat, target=target.astype(int), strata=_strata, train_ratio=_train_ratio, trial=trial)
            # Regressor-Objectives (SLearner overall_model)
            if task.objective_family == "outcome_regression":
                return self._objective_all_regression(params, X_mat=X_mat, target=target.astype(float), strata=_strata, train_ratio=_train_ratio, trial=trial)
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

        # Timeout: Bei "both"-Modus wird der Timeout deaktiviert, weil
        # LGBM systematisch langsamer ist als CatBoost.
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

        _pj_start = self._tuning_n_jobs(task.key)
        _sf = self.cfg.tuning.single_fold
        _cv_k = self.cfg.tuning.cv_splits
        _fold_lbl = "Single-Fold" if _sf else f"{_cv_k}-Fold"
        # n_jobs pro Base-Learner-Fit (gleiche Logik wie _fit_model)
        _pl = self.cfg.constants.parallel_level
        if _pl <= 1:
            _nj = 1
        elif _pl <= 2:
            _nj = -1
        else:
            _nj = max(1, available_cpu_count() // max(1, _pj_start))
        _tlog.info(
            "BLT '%s': Starte %d Trials (parallel=%d, %s, n_jobs=%d).",
            task.key, _n_trials, _pj_start, _fold_lbl, _nj,
        )
        study.optimize(objective, n_trials=_n_trials, timeout=_timeout,
                       n_jobs=_pj_start, catch=(Exception,))
        _tlog = logging.getLogger("rubin.tuning")
        _log_trial_diagnostics(study, task.key, n_jobs=self._tuning_n_jobs(task.key), stage="BLT")

        # RCT-Warnung: Wenn Propensity-Modell Treatment vorhersagen kann, ist die
        # Randomisierung möglicherweise verletzt oder Post-Treatment-Variablen im Datensatz.
        if _is_rct and task.objective_family == "propensity":
            try:
                best_val = float(study.best_value)
                # Log-Loss Schwellwert für RCT-Warnung:
                # Bei balanciertem RCT: Propensity ≈ 0.5 → log_loss ≈ -0.69.
                # Score > -0.55 heißt: Modell lernt Treatment-Zuweisung vorherzusagen.
                _flag = best_val > -0.55
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
            _op_active = getattr(self.cfg.tuning, "overfit_penalty", 0.0) > 0
            raw_score = study.best_trial.user_attrs.get("raw_val_score", study.best_value)
            self.best_scores[task.key] = raw_score
            if _op_active and raw_score != study.best_value:
                self.best_scores[f"{task.key}__adjusted"] = float(study.best_value)
                _tlog.info(
                    "BLT '%s': Raw=%.6g, Adjusted=%.6g (Penalty aktiv, Gap-Abzug=%.6g)",
                    task.key, raw_score, study.best_value, raw_score - study.best_value,
                )
            _is_clf = task.objective_family in {"outcome", "propensity"}
            self.skill_scores[task.key] = _compute_skill_score(
                raw_score, target, is_classification=_is_clf)
        except Exception:
            pass
        # Study freigeben
        try:
            result = {**fixed_defaults, **dict(study.best_params)}
        except ValueError:
            _tlog.warning(
                "BLT '%s': Keine abgeschlossenen Trials. Verwende Default-Parameter.",
                task.key,
            )
            result = dict(fixed_defaults)
        del study
        gc.collect()
        # malloc_trim: glibc-Allocator gibt freigegebenen C/C++-Speicher (CatBoost)
        # ans OS zurück. Ohne malloc_trim hält glibc den Speicher für zukünftige
        # Allokationen — nach 100 CatBoost-Trials × 20 parallel akkumuliert das.
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass  # Nicht-Linux oder libc nicht verfügbar
        _tlog.info("BLT '%s': Study freigegeben, gc.collect() + malloc_trim durchgeführt.", task.key)
        return result

    def _tuning_n_jobs(self, task_key: str = "") -> int:
        """Anzahl paralleler Optuna-Trials basierend auf parallel_level.

        Level 1-2: 1 (sequentiell, alle Kerne an den einzelnen Fit)
        Level 3-4: n_cpus // 4 (je 4 Kerne pro Trial, gleiche Anzahl
                   für CatBoost und LightGBM → vergleichbare TPE-Exploration)
        """
        pl = self.cfg.constants.parallel_level
        if pl <= 2:
            return 1
        n_cpus = available_cpu_count()
        return max(1, n_cpus // 4)

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
            self._tuning_n_jobs(),
            self.cfg.constants.parallel_level,
            max(1, _n_effective // max(1, self._tuning_n_jobs())),
            _cpu_diag,
        )
        if skipped:
            _tlog.info(
                "BLT-Modellfilter: %s übersprungen (nutzen fixed_params). Aktiv: %s",
                skipped, active_models,
            )

        if any((m or "").lower() == "causalforestdml" for m in active_models):
            logging.getLogger("rubin.tuning").debug(
                "CausalForestDML erkannt: Nuisance-Modelle (model_y, model_t) werden hier im BLT getunt. "
                "Wald-Parameter werden separat im CFT (CausalForest-Tuning) via Optuna TPE optimiert."
            )

        plan = self._build_plan(active_models)
        _tlog.info("Tuning-Plan: %d Tasks für %d Modelle.", len(plan), len(active_models))
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

Pruning + Penalty:
Pruning basiert auf RAW Val-Scores (echte Performance). Die Overfit-Penalty
wird erst am Ende auf stabile K-Fold-Mittelwerte angewendet — nicht auf
verrauschte Einzelfold-Scores. Relative Tolerance (0.05 = 5% Gap toleriert)
funktioniert skalen-sicher über R-Loss (~0.001) und DR-MSE (~0.01).

Locking-Regel:
Die gefundenen Hyperparameter werden als tuned_params persistiert und
in allen Cross-Prediction-Folds wiederverwendet."""

    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg
        self.seed = int(cfg.constants.random_seed)
        # Separater Seed für Tuning-CV, damit Fold-Zuordnung ≠ Cross-Prediction-Folds.
        self.tuning_cv_seed = int(cfg.constants.tuning_seed)
        if self.tuning_cv_seed == self.seed:
            logging.getLogger("rubin.tuning").warning(
                "FMT: tuning_seed == random_seed → identische Folds → Val-Set-Overfitting!")
        self.optuna = _safe_import_optuna() if cfg.final_model_tuning.enabled else None
        self.best_scores: Dict[str, float] = {}
        self.skill_scores: Dict[str, float] = {}  # FMT: R² (est.score() ist bereits R²)

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

    def _tuning_n_jobs(self) -> int:
        """Anzahl paralleler Optuna-Trials basierend auf parallel_level.

        FMT-Trials sind ~13× schwerer als BLT-Trials: Jeder Trial fittet
        ein vollständiges NonParamDML/DRLearner mit internem Cross-Fitting
        (K Folds × model_y + model_t + model_final + est.score()).

        Bei n_cpus//4 (= 10 bei 40 CPUs) laufen 130 gleichzeitige
        CatBoost-Fits → OOM oder Thread-Pool-Kontention.

        Level 1-2: 1 (sequentiell)
        Level 3-4: n_cpus // 8 (= 5 bei 40 CPUs, je 8 Kerne pro Trial)
        """
        pl = self.cfg.constants.parallel_level
        if pl <= 2:
            return 1
        n_cpus = available_cpu_count()
        return max(2, n_cpus // 8)

    def _parallel_jobs_per_fit(self) -> int:
        """Kerne pro Base-Learner-Fit, berücksichtigt parallele Trials."""
        pl = self.cfg.constants.parallel_level
        if pl <= 1:
            return 1
        if pl <= 2:
            return -1
        n_cpus = available_cpu_count()
        n_trial_workers = self._tuning_n_jobs()
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
        n_startup = min(n_startup, max(2, n_trials // 2))
        try:
            sampler = optuna.samplers.TPESampler(
                seed=study_seed,
                multivariate=True,
                group=True,
                constant_liar=True,
                consider_endpoints=True,
                n_startup_trials=n_startup,
            )
        except TypeError:
            try:
                sampler = optuna.samplers.TPESampler(
                    seed=study_seed,
                    multivariate=True,
                    constant_liar=True,
                    consider_endpoints=True,
                    n_startup_trials=n_startup,
                )
            except TypeError:
                try:
                    sampler = optuna.samplers.TPESampler(seed=study_seed)
                except Exception:
                    sampler = None
        try:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=2)
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

    def tune_final_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        Y: np.ndarray,
        T: np.ndarray,
        base_type: str,
        base_fixed_params: Dict[str, Any],
        tuned_roles: Dict[str, Dict[str, Any]],
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

        # FMT nutzt die VOLLEN Trainingsdaten — kein 80%-Subsetting.
        # Overfitting-Schutz wird durch den separaten tuning_seed gewährleistet
        # (Tuning-CV-Folds ≠ Cross-Prediction-Folds).
        # Für RAM-/Speed-Kontrolle: max_tuning_rows konfigurieren.
        X_tune = X
        Y_tune = Y
        T_tune = T

        max_rows = self.cfg.final_model_tuning.max_tuning_rows
        if max_rows is not None and len(X_tune) > int(max_rows):
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(np.arange(len(X_tune)), size=int(max_rows), replace=False)
            X_tune = X_tune.iloc[idx]
            Y_tune = Y_tune[idx]
            T_tune = T_tune[idx]

        role = "model_final"
        base_type = (base_type or "lgbm").lower()
        _fmt_is_rct = getattr(self.cfg, "study_type", "rct") == "rct"

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

            # ── cache_values-Optimierung: Nuisance-Modelle EINMALIG pro äußerem Fold fitten ──
            # Die äußeren Folds sind über alle Trials identisch (gleicher tuning_cv_seed).
            # Nur model_final ändert sich pro Trial → Nuisance-Fitting kann gecacht werden.
            # EconML's cache_values=True speichert die OOF-Residuals (Y_res, T_res) und die
            # gefitteten Nuisance-Modelle. refit_final() fittet dann NUR model_final neu.
            _fmt_strata = (T_tune.astype(int) * 10 + Y_tune.astype(int))
            _fmt_cv = StratifiedKFold(n_splits=int(self.cfg.final_model_tuning.cv_splits), shuffle=True, random_state=self.seed)
            _cached_folds = []  # [(est, tr, va), ...]
            split_iter = _iter_stratified_or_kfold(labels=_fmt_strata, n_splits=_outer_splits, seed=self.tuning_cv_seed)
            _n_cache_folds = 1 if _sf else _outer_splits
            for fold_i, (tr, va) in enumerate(split_iter):
                if fold_i >= _n_cache_folds:
                    break
                # RCT: Propensity (model_t) als DummyClassifier → P(T|X) = const = mean(T).
                # Verhindert Overfitting auf Rauschen → saubere T-Residuen.
                if _fmt_is_rct:
                    from sklearn.dummy import DummyClassifier
                    _fmt_model_t = DummyClassifier(strategy="prior")
                else:
                    _fmt_model_t = build_base_learner(base_type, {**base_fixed_params, **tuned_roles.get("model_t", {})}, seed=self.seed, task="classifier", parallel_jobs=-1)
                est = NonParamDML(
                    model_y=build_base_learner(base_type, {**base_fixed_params, **tuned_roles.get("model_y", {})}, seed=self.seed, task="classifier", parallel_jobs=-1),
                    model_t=_fmt_model_t,
                    model_final=build_base_learner(base_type, {**fmt_fixed_params}, seed=self.seed, task="regressor", parallel_jobs=-1),
                    discrete_treatment=True, discrete_outcome=True,
                    allow_missing=True,
                    cv=_fmt_cv,
                    mc_iters=self.cfg.data_processing.mc_iters,
                    mc_agg=self.cfg.data_processing.mc_agg,
                    random_state=self.seed,
                )
                est.fit(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr], cache_values=True)
                _cached_folds.append((est, tr, va))
            _rct_suffix = " model_t = DummyClassifier (RCT: konstante Propensity)." if _fmt_is_rct else ""
            _tlog.info(
                "FMT '%s': Nuisance gecacht für %d äußere Folds (je %d innere × model_y + model_t). "
                "Trials fitten nur noch model_final via refit_final().%s",
                model_name, len(_cached_folds), int(self.cfg.final_model_tuning.cv_splits), _rct_suffix,
            )

            def objective(trial):
                cand_params = _suggest_params(trial, base_type, self.cfg.final_model_tuning.search_space, is_fmt=True)
                if base_type == "lgbm" or (base_type == "both" and cand_params.get("_learner_type") == "lgbm"):
                    cand_params.setdefault("subsample_freq", 1)

                fold_scores_raw = []
                fold_train_scores = []

                for fold_i, (est, tr, va) in enumerate(_cached_folds):
                    # Nur model_final tauschen → refit_final() nutzt gecachte Residuals
                    est.model_final = build_base_learner(base_type, {**fmt_fixed_params, **cand_params}, seed=self.seed, task="regressor", parallel_jobs=-1)
                    est.refit_final()
                    val_score = -float(est.score(Y_tune[va], T_tune[va], X=X_tune.iloc[va]))
                    fold_scores_raw.append(val_score)
                    if _op > 0:
                        fold_train_scores.append(-float(est.score(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr])))

                    # Pruning auf RAW Val-Score — Penalty darf Pruning nicht verzerren
                    trial.report(float(np.mean(fold_scores_raw)), fold_i)
                    if trial.should_prune():
                        raise self.optuna.TrialPruned()

                mean_raw = float(np.mean(fold_scores_raw))
                trial.set_user_attr("raw_val_score", mean_raw)
                # Penalty auf stabile Mittelwerte (nicht verrauschte Einzelfolds)
                if _op > 0 and fold_train_scores:
                    return self._apply_overfit_penalty(mean_raw, float(np.mean(fold_train_scores)), penalty=_op, tolerance=_ot)
                return mean_raw

            _fmt_nt = int(self.cfg.final_model_tuning.n_trials)
            _fmt_is_both = (self.cfg.base_learner.type or "").lower() == "both"
            if _fmt_is_both:
                _fmt_nt *= 2
            _fmt_timeout = self.cfg.final_model_tuning.timeout_seconds
            if _fmt_is_both and _fmt_timeout:
                logging.getLogger("rubin.tuning").warning(
                    "FMT timeout_seconds=%d wird bei 'both' ignoriert (faire Trial-Allokation).", _fmt_timeout)
                _fmt_timeout = None
            _tlog.info(
                "FMT '%s': Starte %d Trials (cache_values, %s, parallel_jobs=-1).",
                model_name, _fmt_nt, "Single-Fold" if _sf else f"{_outer_splits}-Fold",
            )
            # n_jobs=1: refit_final() modifiziert gecachte Estimatoren in-place → nicht thread-safe.
            # Sequentielle Trials sind trotzdem schneller als parallel ohne Cache,
            # weil die Nuisance-Phase (~80% der Laufzeit) komplett entfällt.
            study.optimize(objective, n_trials=_fmt_nt,
                           timeout=_fmt_timeout,
                           n_jobs=1, catch=(Exception,))
            _log_trial_diagnostics(study, model_name, n_jobs=1, stage="FMT")
            try:
                raw = study.best_trial.user_attrs.get("raw_val_score", study.best_value)
                self.best_scores[f"final__{model_name}"] = raw
                if _op > 0 and raw != study.best_value:
                    self.best_scores[f"final__{model_name}__adjusted"] = float(study.best_value)
                    _tlog.info(
                        "FMT '%s': Raw=%.6g, Adjusted=%.6g (Penalty aktiv, Gap-Abzug=%.6g)",
                        model_name, raw, study.best_value, raw - study.best_value,
                    )
            except Exception:
                pass
            try:
                result = {role: dict(study.best_trial.params)}
            except ValueError:
                _tlog.warning("FMT '%s': Keine abgeschlossenen Trials. Verwende Default-Parameter.", model_name)
                result = {}
            # Gecachte Estimatoren + Study freigeben
            for _ce, _, _ in _cached_folds:
                del _ce
            del _cached_folds, study
            import gc as _gc
            _gc.collect()
            _tlog.info("FMT '%s': Study + Cache freigegeben, gc.collect() durchgeführt.", model_name)
            return result

        elif name == "drlearner":
            from econml.dr import DRLearner

            study = self._create_study(f"final__{model_name}__{base_type}")
            _op = float(getattr(self.cfg.final_model_tuning, "overfit_penalty", 0.0))
            _ot = float(getattr(self.cfg.final_model_tuning, "overfit_tolerance", 0.05))
            _sf = bool(getattr(self.cfg.final_model_tuning, "single_fold", False))
            _outer_splits = int(self.cfg.data_processing.cross_validation_splits)

            # ── cache_values-Optimierung (analog NonParamDML) ──
            _fmt_strata = (T_tune.astype(int) * 10 + Y_tune.astype(int))
            _fmt_cv = StratifiedKFold(n_splits=int(self.cfg.final_model_tuning.cv_splits), shuffle=True, random_state=self.seed)
            _cached_folds = []
            split_iter = _iter_stratified_or_kfold(labels=_fmt_strata, n_splits=_outer_splits, seed=self.tuning_cv_seed)
            _n_cache_folds = 1 if _sf else _outer_splits
            for fold_i, (tr, va) in enumerate(split_iter):
                if fold_i >= _n_cache_folds:
                    break
                # RCT: Propensity als DummyClassifier → P(T|X) = const.
                if _fmt_is_rct:
                    from sklearn.dummy import DummyClassifier
                    _fmt_prop = DummyClassifier(strategy="prior")
                else:
                    _fmt_prop = build_base_learner(base_type, {**base_fixed_params, **tuned_roles.get("model_propensity", {})}, seed=self.seed, task="classifier", parallel_jobs=-1)
                est = DRLearner(
                    model_propensity=_fmt_prop,
                    model_regression=build_base_learner(base_type, {**base_fixed_params, **tuned_roles.get("model_regression", {})}, seed=self.seed, task="classifier", parallel_jobs=-1),
                    model_final=build_base_learner(base_type, {**fmt_fixed_params}, seed=self.seed, task="regressor", parallel_jobs=-1),
                    discrete_outcome=True,
                    allow_missing=True,
                    cv=_fmt_cv,
                    mc_iters=self.cfg.data_processing.mc_iters,
                    mc_agg=self.cfg.data_processing.mc_agg,
                    random_state=self.seed,
                )
                est.fit(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr], cache_values=True)
                _cached_folds.append((est, tr, va))
            _rct_suffix = " model_propensity = DummyClassifier (RCT: konstante Propensity)." if _fmt_is_rct else ""
            _tlog.info(
                "FMT '%s': Nuisance gecacht für %d äußere Folds (je %d innere × model_propensity + model_regression). "
                "Trials fitten nur noch model_final via refit_final().%s",
                model_name, len(_cached_folds), int(self.cfg.final_model_tuning.cv_splits), _rct_suffix,
            )

            def objective(trial):
                cand_params = _suggest_params(trial, base_type, self.cfg.final_model_tuning.search_space, is_fmt=True)
                if base_type == "lgbm" or (base_type == "both" and cand_params.get("_learner_type") == "lgbm"):
                    cand_params.setdefault("subsample_freq", 1)

                fold_scores_raw = []
                fold_train_scores = []

                for fold_i, (est, tr, va) in enumerate(_cached_folds):
                    est.model_final = build_base_learner(base_type, {**fmt_fixed_params, **cand_params}, seed=self.seed, task="regressor", parallel_jobs=-1)
                    est.refit_final()
                    val_score = -float(est.score(Y_tune[va], T_tune[va], X=X_tune.iloc[va]))
                    fold_scores_raw.append(val_score)
                    if _op > 0:
                        fold_train_scores.append(-float(est.score(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr])))

                    trial.report(float(np.mean(fold_scores_raw)), fold_i)
                    if trial.should_prune():
                        raise self.optuna.TrialPruned()

                mean_raw = float(np.mean(fold_scores_raw))
                trial.set_user_attr("raw_val_score", mean_raw)
                if _op > 0 and fold_train_scores:
                    return self._apply_overfit_penalty(mean_raw, float(np.mean(fold_train_scores)), penalty=_op, tolerance=_ot)
                return mean_raw

            _fmt_nt = int(self.cfg.final_model_tuning.n_trials)
            _fmt_is_both = (self.cfg.base_learner.type or "").lower() == "both"
            if _fmt_is_both:
                _fmt_nt *= 2
            _fmt_timeout = self.cfg.final_model_tuning.timeout_seconds
            if _fmt_is_both and _fmt_timeout:
                logging.getLogger("rubin.tuning").warning(
                    "FMT timeout_seconds=%d wird bei 'both' ignoriert (faire Trial-Allokation).", _fmt_timeout)
                _fmt_timeout = None
            _tlog.info(
                "FMT '%s': Starte %d Trials (cache_values, %s, parallel_jobs=-1).",
                model_name, _fmt_nt, "Single-Fold" if _sf else f"{_outer_splits}-Fold",
            )
            study.optimize(objective, n_trials=_fmt_nt,
                           timeout=_fmt_timeout,
                           n_jobs=1, catch=(Exception,))
            _log_trial_diagnostics(study, model_name, n_jobs=1, stage="FMT")
            try:
                raw = study.best_trial.user_attrs.get("raw_val_score", study.best_value)
                self.best_scores[f"final__{model_name}"] = raw
                if _op > 0 and raw != study.best_value:
                    self.best_scores[f"final__{model_name}__adjusted"] = float(study.best_value)
                    _tlog.info(
                        "FMT '%s': Raw=%.6g, Adjusted=%.6g (Penalty aktiv, Gap-Abzug=%.6g)",
                        model_name, raw, study.best_value, raw - study.best_value,
                    )
            except Exception:
                pass
            try:
                result = {role: dict(study.best_trial.params)}
            except ValueError:
                _tlog.warning("FMT '%s': Keine abgeschlossenen Trials. Verwende Default-Parameter.", model_name)
                result = {}
            for _ce, _, _ in _cached_folds:
                del _ce
            del _cached_folds, study
            import gc as _gc
            _gc.collect()
            _tlog.info("FMT '%s': Study + Cache freigegeben, gc.collect() durchgeführt.", model_name)
            return result

        return {}


# ── CausalForest Optuna Tuner ──

_CF_SEARCH_SPACE = {
    "n_estimators":         {"type": "int",   "low": 100,   "high": 500},
    "max_depth":            {"type": "int",   "low": 3,     "high": 30},     # EconML Default: None → 30 effektiv unbegrenzt
    "min_samples_leaf":     {"type": "int",   "low": 5,     "high": 200},  # EconML Default: 5
    "min_samples_split":    {"type": "int",   "low": 10,    "high": 500},  # EconML Default: 10
    "max_features":         {"type": "float", "low": 0.1,   "high": 1.0},  # EconML Default: 'auto' = 1.0 (alle Features)
    "max_samples":          {"type": "float", "low": 0.15,  "high": 0.5},   # EconML: max 0.5 bei inference=True
    "min_var_fraction_leaf":{"type": "float", "low": 1e-6,  "high": 0.05},  # EconML Default: None → 1e-6 ≈ None
    "min_impurity_decrease":{"type": "float", "low": 0.0,   "high": 0.01},
    "criterion":            {"type": "cat",   "choices": ["mse", "het"]},
}


def tune_causal_forest(
    cfg,
    X: "pd.DataFrame",
    Y: "np.ndarray",
    T: "np.ndarray",
    model_type: str = "CausalForestDML",
    n_trials: int = 50,
    nuisance_params_y: dict = None,
    nuisance_params_t: dict = None,
    single_fold: bool = False,
) -> Dict[str, Any]:
    """Optuna-basiertes Tuning für CausalForestDML und CausalForest.

    Evaluiert jede Parameterkombination via est.score() (R-Loss) auf
    äußeren Tuning-CV-Folds (tuning_cv_seed). Modell-Internals nutzen
    random_seed für Konsistenz mit der Produktion.

    CPU-Parallelisierung (siehe docs/architektur.md):
    - Trials laufen sequentiell (n_jobs=1 in study.optimize), da jeder
      Forest intern alle Kerne für die Baumbildung nutzt (n_jobs=-1).
    - GRF Residuen-Vorberechnung: Standalone Nuisance-Fits mit parallel_jobs=-1
      (alle Kerne frei, kein EconML-Estimator aktiv).
    - CFDML-interne Nuisance-Modelle: parallel_jobs=-1, da EconML's
      Cross-Fitting-Loop sequentiell ist (_OrthoLearner, kein ray).
      CausalForestDML(n_jobs=-1) parallelisiert den CausalForest-Baum-Fit,
      NICHT die Nuisance-Folds → keine Übersubskription.

    Suchraum-Overrides: cfg.causal_forest.search_space kann pro Parameter
    {low, high} überschreiben. Nicht überschriebene Parameter nutzen
    die Defaults aus _CF_SEARCH_SPACE.

    Returns:
        Dict mit 'best_params', 'best_score', 'n_trials_completed'.
    """
    optuna = _safe_import_optuna()
    _tlog = logging.getLogger("rubin.tuning")

    seed = int(cfg.constants.random_seed)
    tuning_cv_seed = int(cfg.constants.tuning_seed)
    inner_cv = int(cfg.data_processing.dml_crossfit_folds)

    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    Y_np = np.asarray(Y).ravel()
    T_np = np.asarray(T).ravel()

    base_type = (cfg.base_learner.type or "catboost").lower()
    if base_type == "both":
        base_type = "catboost"

    is_cfdml = model_type.lower() == "causalforestdml"

    # Merge user overrides into search space
    user_ss = {}
    if hasattr(cfg, "causal_forest") and hasattr(cfg.causal_forest, "search_space"):
        raw = cfg.causal_forest.search_space
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, dict):
                    user_ss[k] = v
                elif hasattr(v, "low"):
                    user_ss[k] = {"low": v.low, "high": v.high}
    effective_ss = {}
    for pname, spec in _CF_SEARCH_SPACE.items():
        entry = dict(spec)
        if pname in user_ss:
            if user_ss[pname].get("low") is not None:
                entry["low"] = user_ss[pname]["low"]
            if user_ss[pname].get("high") is not None:
                entry["high"] = user_ss[pname]["high"]
        effective_ss[pname] = entry

    # ── GRF (nicht-DML): Residuen EINMALIG vorberechnen ──
    # R-Loss braucht E[Y|X] und E[T|X]. Diese ändern sich nicht zwischen Trials
    # (gleiche Daten, gleiche Folds, gleicher Seed). Daher: einmalig vorberechnen
    # mit den BLT-getunten Nuisance-Modellen (konsistent mit CFDML-Pfad).
    _grf_fold_data = None
    _cft_is_rct = getattr(cfg, "study_type", "rct") == "rct"
    if not is_cfdml:
        _cf_strata_pre = T_np.astype(int) * 10 + np.clip(Y_np, 0, 1).astype(int)
        _splits_pre = list(_iter_stratified_or_kfold(
            labels=_cf_strata_pre,
            n_splits=inner_cv,
            seed=tuning_cv_seed,
        ))
        _n_pre = 1 if single_fold else inner_cv
        _grf_fold_data = []
        for _fi, (_tr, _va) in enumerate(_splits_pre[:_n_pre]):
            # BLT-getunte Nuisance-Modelle für Residuen-Vorberechnung.
            _m_y = build_base_learner(base_type, nuisance_params_y or {}, seed=seed, task="classifier", parallel_jobs=-1)
            # RCT: Propensity als DummyClassifier → T_res = T - mean(T), keine Korrelation mit X.
            if _cft_is_rct:
                from sklearn.dummy import DummyClassifier
                _m_t = DummyClassifier(strategy="prior")
            else:
                _m_t = build_base_learner(base_type, nuisance_params_t or {}, seed=seed, task="classifier", parallel_jobs=-1)
            _m_y.fit(X_np[_tr], Y_np[_tr])
            _m_t.fit(X_np[_tr], T_np[_tr])
            _Y_res = Y_np[_va] - _m_y.predict_proba(X_np[_va])[:, 1]
            _T_res = T_np[_va] - _m_t.predict_proba(X_np[_va])[:, 1]
            _grf_fold_data.append((_tr, _va, _Y_res, _T_res))
            del _m_y, _m_t
        import gc; gc.collect()
        _rct_suffix = " model_t = DummyClassifier (RCT: konstante Propensity)." if _cft_is_rct else ""
        _tlog.info(
            "GRF R-Loss: Nuisance-Residuen vorberechnet (%d Folds, %s Base-Learner).%s",
            len(_grf_fold_data), base_type.upper(), _rct_suffix,
        )

    # ── CFDML: Nuisance EINMALIG pro äußerem Fold cachen (cache_values) ──
    # CausalForestDML's Nuisance-Modelle (model_y, model_t) + OOF-Residuals werden
    # einmalig pro Fold gefittet und gecacht. Trials ändern nur die GRF-Forest-Parameter
    # via set_params() + refit_final(). Die äußeren Folds sind über alle Trials identisch
    # (gleicher tuning_cv_seed) → Nuisance-Fitting entfällt komplett in der Trial-Schleife.
    _cfdml_cached_folds = None
    if is_cfdml:
        from econml.dml import CausalForestDML
        _cf_strata_pre = T_np.astype(int) * 10 + np.clip(Y_np, 0, 1).astype(int)
        _splits_pre = list(_iter_stratified_or_kfold(
            labels=_cf_strata_pre,
            n_splits=inner_cv,
            seed=tuning_cv_seed,
        ))
        _n_pre = 1 if single_fold else inner_cv
        _cfdml_cached_folds = []
        # Default-GRF-Params für den initialen Fit (werden pro Trial überschrieben)
        _default_forest = {k: spec.get("low", spec.get("choices", [None])[0])
                          for k, spec in effective_ss.items() if k != "criterion"}
        for _fi, (_tr, _va) in enumerate(_splits_pre[:_n_pre]):
            # RCT: Propensity als DummyClassifier → P(T|X) = const.
            if _cft_is_rct:
                from sklearn.dummy import DummyClassifier
                _cft_model_t = DummyClassifier(strategy="prior")
            else:
                _cft_model_t = build_base_learner(base_type, nuisance_params_t or {}, seed=seed, task="classifier", parallel_jobs=-1)
            est = CausalForestDML(
                model_y=build_base_learner(base_type, nuisance_params_y or {}, seed=seed, task="classifier", parallel_jobs=-1),
                model_t=_cft_model_t,
                discrete_treatment=True,
                discrete_outcome=True,
                cv=StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed),
                random_state=seed,
                n_jobs=-1,
                **_default_forest,
            )
            est.fit(Y_np[_tr], T_np[_tr], X=X_np[_tr], cache_values=True)
            _cfdml_cached_folds.append((est, _tr, _va))
        import gc; gc.collect()
        _rct_suffix = " model_t = DummyClassifier (RCT: konstante Propensity)." if _cft_is_rct else ""
        _tlog.info(
            "CFT '%s': Nuisance gecacht für %d äußere Folds (je %d innere × model_y + model_t). "
            "Trials fitten nur noch den CausalForest via refit_final().%s",
            model_type, len(_cfdml_cached_folds), inner_cv, _rct_suffix,
        )

    def objective(trial):
        # Suggest params aus effektivem Suchraum (Defaults + User-Overrides)
        params = {}
        for pname, spec in effective_ss.items():
            # criterion ist nur für CausalForest (grf), nicht CausalForestDML
            if pname == "criterion" and is_cfdml:
                continue
            if spec["type"] == "int":
                # n_estimators muss durch subforest_size (Default=4) teilbar sein.
                # step=4 stellt sicher, dass Optuna NUR Vielfache von 4 vorschlägt —
                # sowohl im Trial als auch in study.best_trial.params.
                _step = 4 if pname == "n_estimators" else 1
                params[pname] = trial.suggest_int(pname, int(spec["low"]), int(spec["high"]), step=_step, log=spec.get("log", False) if _step == 1 else False)
            elif spec["type"] == "float":
                params[pname] = trial.suggest_float(pname, float(spec["low"]), float(spec["high"]), log=spec.get("log", False))
            elif spec["type"] == "cat":
                params[pname] = trial.suggest_categorical(pname, spec["choices"])

        fold_scores = []

        if is_cfdml:
            # CFDML: Gecachte Estimatoren nutzen — nur GRF-Parameter ändern + refit_final()
            for fold_i, (est, tr, va) in enumerate(_cfdml_cached_folds):
                try:
                    _forest_params = {k: v for k, v in params.items() if k not in ("criterion",)}
                    # CausalForestDML hat KEIN set_params() (kein sklearn BaseEstimator).
                    # Stattdessen: direkt Attribute setzen. _prefit() in refit_final()
                    # liest self.max_depth, self.n_estimators etc. und baut einen
                    # FRISCHEN CausalForest mit den aktualisierten Werten.
                    for k, v in _forest_params.items():
                        setattr(est, k, v)
                    est.refit_final()
                    score = -float(est.score(Y_np[va], T_np[va], X=X_np[va]))
                    fold_scores.append(score)
                except Exception:
                    fold_scores.append(-1e12)

                trial.report(float(np.mean(fold_scores)), fold_i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        else:
            # GRF: Vorberechnete Residuen nutzen (BLT-getunte Nuisance, T×Y-Folds)
            for fold_i, (tr, va, Y_res_va, T_res_va) in enumerate(_grf_fold_data):
                try:
                    from econml.grf import CausalForest as _GRF_CF
                    est = _GRF_CF(
                        random_state=seed,
                        n_jobs=-1,
                        **params,
                    )
                    est.fit(X_np[tr], T_np[tr], Y_np[tr])
                    cate_va = est.predict(X_np[va]).flatten()
                    # R-loss = E[(Y_res - τ·T_res)²], lower = better → negieren
                    score = -float(np.mean((Y_res_va - cate_va * T_res_va) ** 2))
                    fold_scores.append(score)
                    del est
                    import gc; gc.collect()
                except Exception:
                    fold_scores.append(-1e12)

                trial.report(float(np.mean(fold_scores)), fold_i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    _cft_n_startup = max(5, n_trials // 5)
    _cft_n_startup = min(_cft_n_startup, max(3, n_trials // 2))
    sampler = optuna.samplers.TPESampler(
        seed=tuning_cv_seed,
        multivariate=True,
        group=True,
        consider_endpoints=True,
        n_startup_trials=_cft_n_startup,
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(3, n_trials // 10), n_warmup_steps=2)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"cf_tune__{model_type}",
    )

    _fold_lbl = "Single-Fold" if single_fold else f"{inner_cv}-Fold"
    _tlog.info(
        "CFT '%s': Starte %d Trials (cache_values, %s, parallel_jobs=-1).",
        model_type, n_trials, _fold_lbl,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, catch=(Exception,))
    n_complete, _, _ = _log_trial_diagnostics(study, model_type, n_jobs=1, stage="CFT")

    try:
        best = dict(study.best_trial.params)
        best_score = float(study.best_value)
    except ValueError:
        _tlog.warning("CFT '%s': Keine abgeschlossenen Trials. Verwende Default-Parameter.", model_type)
        best = {}
        best_score = None

    del study
    # Gecachte CFDML-Estimatoren freigeben (Nuisance-Modelle + Residuals)
    if _cfdml_cached_folds is not None:
        for _ce, _, _ in _cfdml_cached_folds:
            del _ce
        del _cfdml_cached_folds
    import gc as _gc
    _gc.collect()
    _tlog.info("CFT '%s': Study + Cache freigegeben, gc.collect() durchgeführt.", model_type)

    return {"best_params": best, "best_score": best_score,
            "n_trials_completed": n_complete}
