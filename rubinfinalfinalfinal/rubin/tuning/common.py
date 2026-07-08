from __future__ import annotations

"""Optuna-basiertes Hyperparameter-Tuning (BLT, FMT, CFT).

Drei Tuning-Stufen:
1. BLT (Base-Learner-Tuning): Optimiert Nuisance-Modelle (Outcome, Propensity)
   mit aufgabenspezifischen CV-Objectives. Bei RCT dient das Propensity-Tuning
   als Diagnose-Check (Skill ≈ 0 bestätigt Randomisierung).
2. FMT (Final-Model-Tuning): Optimiert model_final (CATE-Regressor) via OOF-CV
   mit gecachten Nuisance-Residuals (cache_values).
3. CFT (CausalForest-Tuning): Optimiert Forest-Parameter für CausalForestDML
   (setattr + refit_final) und CausalForest (frischer Fit pro Trial).

Besonderheiten:
- RCT-Modus: DummyClassifier für Propensity in FMT/CFT Nuisance-Caching.
- CausalForestDML hat kein set_params() → setattr() + refit_final() stattdessen.
- n_estimators wird auf Vielfache von subforest_size=4 beschränkt (step=4).
- TPE-Sampler: multivariate=True, group=True, n_startup gekappt.
- MedianPruner: n_warmup_steps=2 für stabileres Pruning."""


from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
import logging


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
            # Baseline = Log-Loss des konstanten Klassen-Prior-Prädiktors
            # = Entropie der Klassenverteilung H(p) = -Σ p_k · ln(p_k).
            # Multiclass-fähig (z.B. MT-Propensity mit K Treatment-Armen);
            # für binäres Y reduziert sich das exakt auf -(p·ln p + (1-p)·ln(1-p)).
            y = target.astype(int)
            _, counts = np.unique(y, return_counts=True)
            freqs = np.clip(counts / counts.sum(), 1e-15, 1.0)
            baseline = float(-np.sum(freqs * np.log(freqs)))
        else:
            baseline = float(np.var(target.astype(float)))
        if baseline < 1e-15:
            return 0.0
        return 1.0 - model_loss / baseline
    except Exception:
        return 0.0

from rubin.settings import SearchSpaceConfig, SearchSpaceParameterConfig


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


# ── R-Score Helpers (historisch) ──
# FMT und CFT nutzen jetzt den externen EconML RScorer mit unabhängiger
# Nuisance (2-Fold T×Y Cross-Fitting). Der RScorer berechnet R-Score intern:
#   R-Score = 1 − MSE(heterogen) / MSE(konstant)
# Einheitliche Metrik über alle Learner (NonParamDML, DRLearner, CFDML, GRF).
# Scorer werden zwischen Modellen gecacht (FMT + CFT).
# Die folgenden Helper-Funktionen werden in keinem aktiven Scoring-Pfad
# verwendet — sie sind reine Utilities.
# Referenz: Nie & Wager (2021), Schuler et al. (2018), EconML RScorer.

def _compute_base_mse_from_residuals(Y_res: "np.ndarray", T_res: "np.ndarray") -> float:
    """Base-MSE für R-Loss: min_θ E[(Y_res − θ · T_res)²].

    OLS-Lösung: θ* = Σ(Y_res · T_res) / Σ(T_res²).
    Wird einmalig pro Fold VOR dem Trial-Loop berechnet.
    """
    denom = float(np.dot(T_res, T_res))
    if denom < 1e-10:
        # Kein Treatment-Signal → base_mse = Var(Y_res)
        return float(np.mean(Y_res ** 2))
    theta_const = float(np.dot(Y_res, T_res)) / denom
    return float(np.mean((Y_res - theta_const * T_res) ** 2))


def _nuisance_residuals_for_val(
    est, Y_va: "np.ndarray", T_va: "np.ndarray", X_va: "np.ndarray",
) -> "Tuple[np.ndarray, np.ndarray]":
    """Nuisance-Residuals für Validierungsdaten aus gecachten Modellen extrahieren.

    Nutzt est.models_y und est.models_t (öffentliche EconML-Attribute nach
    fit(cache_values=True)). Mittelt Predictions über alle Crossfit-Folds
    und MC-Iterationen für stabilere Residuals.
    """
    Y_va_flat = np.asarray(Y_va).ravel()
    T_va_flat = np.asarray(T_va).ravel()

    # Y-Nuisance: E[Y|X]
    y_preds = []
    for mc_models in est.models_y:
        for m in mc_models:
            if hasattr(m, "predict_proba"):
                y_preds.append(m.predict_proba(X_va)[:, 1])
            else:
                y_preds.append(np.asarray(m.predict(X_va)).ravel())
    Y_res = Y_va_flat - np.mean(y_preds, axis=0)

    # T-Nuisance: E[T|X]
    t_preds = []
    for mc_models in est.models_t:
        for m in mc_models:
            if hasattr(m, "predict_proba"):
                t_preds.append(m.predict_proba(X_va)[:, 1])
            else:
                t_preds.append(np.asarray(m.predict(X_va)).ravel())
    T_res = T_va_flat - np.mean(t_preds, axis=0)

    return Y_res, T_res


def _compute_base_mse_via_dummy(est, Y_va, T_va, X_va) -> float:
    """Base-MSE über DummyRegressor + refit_final() berechnen.

    Funktioniert für NonParamDML und DRLearner (model_final austauschbar).
    NICHT für CausalForestDML (Forest ist model_final selbst → _nuisance_residuals nutzen).

    Für DRLearner ist die Base-MSE exakt (= Var(Γ), DummyRegressor = mean(Γ)).
    Für NonParamDML entsteht durch EconML's Reweighting-Wrapper eine minimale
    Abweichung (~0.015%) vom theoretischen OLS-Optimum. Die Ranking-Erhaltung
    (monotone Transformation) ist davon nicht betroffen.
    """
    from sklearn.dummy import DummyRegressor
    _orig_final = est.model_final
    try:
        est.model_final = DummyRegressor(strategy="mean")
        est.refit_final()
        base_mse = float(est.score(Y_va, T_va, X=X_va))
    finally:
        est.model_final = _orig_final
    return base_mse


def _to_r_score(trial_mse: float, base_mse: float) -> float:
    """R-Score = 1 − trial_mse / base_mse.  Maximieren = besser."""
    if base_mse < 1e-10:
        return 0.0
    return 1.0 - trial_mse / base_mse


# ── Tuning n_estimators: Kleine Forests (100) für Speed, Production nutzt mehr ──


def _create_fold_scorer(
    base_type: str,
    base_fixed_params: dict,
    tuned_model_y_params: dict,
    tuned_model_t_params: dict,
    Y, T, X,
    seed: int,
    is_rct: bool = False,
    build_base_learner_fn=None,
):
    """Erstellt einen EconML RScorer mit unabhängiger Nuisance auf den gegebenen Daten.

    Der RScorer fittet eigene Nuisance-Modelle (model_y, model_t) mit internem
    2-Fold T×Y-stratifiziertem Cross-Fitting auf den übergebenen Daten.

    Warum cv=2 (nicht höher):
    - EconML's eigenes CausalForestDML.tune() nutzt cv=est.cv=2 für den RScorer.
    - Chernozhukov et al. (2018): K≥2 genügt für Elimination des First-Order
      Overfitting-Bias. Höheres K verbessert Nuisance-Qualität marginal, aber
      der RScorer dient dem MODEL SELECTION (Ranking-Erhaltung), nicht der
      CATE-Estimation. Für Ranking ist cv=2 ausreichend (Saito & Yasui, 2019).
    - Nesting-Problem: Die Val-Daten sind bereits ein Subset (z.B. 20% bei
      K=5 äußeren Folds). Internes cv=5 würde auf 16% der Gesamtdaten
      trainieren, cv=2 auf 10% — der Unterschied ist marginal, aber cv=2
      ist 2.5× schneller (2 statt 5 Nuisance-Fits pro Scorer).
    - Jacob (2020) empfiehlt 5-Fold für CATE-Estimation, nicht für Scoring.

    Warum T×Y-Stratifizierung (statt EconML-Default nur T):
    - EconML's StratifiedKFold stratifiziert nur nach T (Treatment).
    - T×Y-Stratifizierung stellt sicher, dass jeder interne Fold eine
      repräsentative Treatment-Outcome-Verteilung hat (wichtig bei
      unbalanciertem Y). Pre-computed als Iterable übergeben, weil
      RScorer intern split(X, T) aufruft (nur T-Stratifizierung).

    Referenz: EconML CausalForestDML.tune(), Chernozhukov et al. (2018),
    Schuler et al. (2018), Nie & Wager (2021), Saito & Yasui (2019).
    """
    import numpy as np
    from econml.score import RScorer
    from sklearn.model_selection import StratifiedKFold
    if build_base_learner_fn is None:
        build_base_learner_fn = build_base_learner  # tuning/common.py

    _my_params = {**base_fixed_params, **tuned_model_y_params}
    model_y = build_base_learner_fn(base_type, _my_params, seed=seed, task="classifier", parallel_jobs=-1)

    if is_rct:
        from sklearn.dummy import DummyClassifier
        model_t = DummyClassifier(strategy="prior")
    else:
        _mt_params = {**base_fixed_params, **tuned_model_t_params}
        model_t = build_base_learner_fn(base_type, _mt_params, seed=seed, task="classifier", parallel_jobs=-1)

    _Y_np = np.asarray(Y).ravel()
    _T_np = np.asarray(T).ravel()
    # NaN-Handling: RScorer's interner LinearDML hat kein allow_missing.
    # CatBoost/LightGBM handeln NaN nativ, aber LinearDML's check_input_arrays()
    # lehnt NaN ab. Lösung: Zeilen mit NaN in X vor dem Scorer-Fit entfernen.
    _X_np = np.asarray(X) if not isinstance(X, np.ndarray) else X
    _nan_mask = np.isnan(_X_np).any(axis=1) if _X_np.ndim == 2 else np.isnan(_X_np)
    if _nan_mask.any():
        _clean = ~_nan_mask
        _Y_np = _Y_np[_clean]
        _T_np = _T_np[_clean]
        _X_np = _X_np[_clean]

    # 2-Fold T×Y-stratifiziertes Cross-Fitting (EconML-aligned, cv=2).
    _strata = _T_np.astype(int) * 10 + np.clip(_Y_np, 0, 1).astype(int)
    _skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    _splits = list(_skf.split(np.zeros(len(_T_np)), _strata))

    scorer = RScorer(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        discrete_outcome=True,
        cv=_splits,
        random_state=seed,
    )
    scorer.fit(_Y_np, _T_np, X=_X_np)
    return scorer


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
#
# ÜBERSCHREIBBAR: Beide Werte können via base_learner.fixed_params überschrieben
# werden (z.B. max_ctr_complexity: 2 für Datensätze mit wenigen kat. Features,
# bei denen Feature-Interaktionen die Nuisance-Qualität verbessern können).
# Die fixed_params werden NACH _CATBOOST_CPU_OPTS angewendet.
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
            "reg_alpha": SearchSpaceParameterConfig(type="float", low=0.0, high=20.0),
            "reg_lambda": SearchSpaceParameterConfig(type="float", low=0.0, high=20.0),
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

    BEWUSST weniger Regularisierung als BL erlaubt: Das CATE-Signal ist
    schwächer als das Outcome/Propensity-Signal. Zu hohe Regularisierung
    (reg_alpha, reg_lambda, min_child_samples, path_smooth) führt bei LGBM
    zum Intercept-Kollaps — Optuna konvergiert auf maximale Regularisierung,
    weil eine Konstante R-Score ≈ 0 erreicht und jeder Heterogenitäts-Versuch
    R-Score < 0 liefert. Weniger Bäume und flachere Tiefe als BL,
    aber Regularisierungs-Obergrenzen NIEDRIGER als BL."""
    base_type = (base_type or "").lower()
    if base_type == "lgbm":
        return {
            "n_estimators": SearchSpaceParameterConfig(type="int", low=100, high=400),
            "learning_rate": SearchSpaceParameterConfig(type="float", low=5e-3, high=1.2e-1, log=True),
            "num_leaves": SearchSpaceParameterConfig(type="int", low=7, high=63),
            "max_depth": SearchSpaceParameterConfig(type="int", low=2, high=6),
            "min_child_samples": SearchSpaceParameterConfig(type="int", low=20, high=200),
            "min_child_weight": SearchSpaceParameterConfig(type="float", low=0.5, high=20.0),
            "subsample": SearchSpaceParameterConfig(type="float", low=0.4, high=0.85),
            "colsample_bytree": SearchSpaceParameterConfig(type="float", low=0.3, high=0.7),
            "max_bin": SearchSpaceParameterConfig(type="int", low=15, high=127),
            "min_split_gain": SearchSpaceParameterConfig(type="float", low=0.0, high=0.5),
            "reg_alpha": SearchSpaceParameterConfig(type="float", low=0.0, high=10.0),
            "reg_lambda": SearchSpaceParameterConfig(type="float", low=0.0, high=10.0),
            "path_smooth": SearchSpaceParameterConfig(type="float", low=0.0, high=5.0),
        }
    if base_type == "catboost":
        ss = {
            "iterations": SearchSpaceParameterConfig(type="int", low=100, high=400),
            "learning_rate": SearchSpaceParameterConfig(type="float", low=5e-3, high=1.2e-1, log=True),
            "depth": SearchSpaceParameterConfig(type="int", low=2, high=6),
            "l2_leaf_reg": SearchSpaceParameterConfig(type="float", low=3.0, high=30.0),
            "random_strength": SearchSpaceParameterConfig(type="float", low=0.5, high=10.0),
            "subsample": SearchSpaceParameterConfig(type="float", low=0.4, high=0.85),
            "rsm": SearchSpaceParameterConfig(type="float", low=0.3, high=0.7),
            "min_data_in_leaf": SearchSpaceParameterConfig(type="int", low=20, high=200),
            "model_size_reg": SearchSpaceParameterConfig(type="float", low=0.1, high=10.0),
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
    """Ergebnis eines BLT-Tasks: getunte Hyperparameter pro Modell-Rolle."""
    role_params: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class TuningTask:
    """Signatur eines BLT-Tuning-Tasks (frozen, hashable für Dedup).

    Tasks mit identischer Signatur werden geteilt: z.B. NonParamDML.model_t
    und DRLearner.model_propensity teilen denselben Propensity-Task.
    """
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


