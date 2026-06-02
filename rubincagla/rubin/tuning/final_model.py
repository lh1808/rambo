from __future__ import annotations

"""Final-Model-Tuning (FMT) mit Optuna.

Optimiert model_final (CATE-Regressor) via OOF-CV mit gecachten
Nuisance-Residuals (cache_values).
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import logging
import gc

from sklearn.model_selection import StratifiedKFold

from rubin.tuning.common import (
    _iter_stratified_or_kfold, _safe_import_optuna, _create_fold_scorer,
    _suggest_params, build_base_learner, _log_trial_diagnostics,
)
from rubin.evaluation.uplift_metrics import uplift_curve, qini_coefficient
from rubin.settings import AnalysisConfig
from rubin.utils.data_utils import available_cpu_count

class FinalModelTuner:
    """Tuning des Final-Modells (model_final) über OOF-CV.

Beide unterstützten Modelle (NonParamDML, DRLearner) werden über den EconML
RScorer bewertet — mit unabhängiger Nuisance (2-Fold T×Y Cross-Fitting auf
Val-Daten) und einheitlichem R-Score als Metrik.

R-Score (Schuler et al., 2018; Nie & Wager, 2021):
  R-Score = 1 − MSE(heterogen) / MSE(konstant)
  Intercept-Only → R-Score ≈ 0, echte Heterogenität → > 0, Overfitting → < 0.
  Der RScorer berechnet Base-MSE und R-Score intern — kein DummyRegressor nötig.

Vorteile des externen RScorer:
- Nuisance-Unabhängigkeit (kein Overfitting-Leak vom Estimator)
- Einheitliche Metrik über NonParamDML und DRLearner (direkt vergleichbar)
- Konsistenz mit EconML CausalForestDML.tune()
- Scorer werden zwischen Modellen gecacht (self._scorer_cache_*)

Architektur:
- Nuisance-Modelle (model_y, model_t/model_propensity) werden einmalig pro
  äußerem Fold gecacht (cache_values=True + est.fit()).
- RScorer pro Fold erstellt (unabhängige Nuisance, 2-Fold T×Y).
- Pro Trial wird NUR model_final als neues Objekt erstellt (est.model_final = ...)
  und via est.refit_final() gefittet. Bewertung via scorer.score(est).
- Bei RCT wird model_t/model_propensity als DummyClassifier(strategy="prior")
  im Nuisance-Cache verwendet → konstante Propensity.

Score-Konvention:
R-Score (higher=better), direkt kompatibel mit Optuna direction="maximize".

Pruning + Penalty:
Pruning basiert auf R-Score Val-Scores (echte Performance). Die Overfit-Penalty
wird erst am Ende auf stabile K-Fold-Mittelwerte angewendet. Mit dem externen
RScorer ist die Penalty weniger kritisch (EconML tune() nutzt keine Penalty).

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
        self.skill_scores: Dict[str, float] = {}  # FMT: R-Score
        # Scorer-Cache: Wiederverwendung über Modelle (NonParamDML → DRLearner)
        # wenn gleiche Fold-Konfiguration + Daten. Spart ~80s RScorer-Setup.
        self._scorer_cache_val: list = []
        self._scorer_cache_train: list = []
        self._scorer_cache_key: tuple = ()

    def _apply_overfit_penalty(self, val_score: float, train_score: float,
                               penalty: float = 0.0, tolerance: float = 0.05) -> float:
        """Bestraft Overfit-Gap zwischen Train- und Val-Score (skalen-sicher).

        Formel (relative Tolerance):
            scale = max(|val_score|, 0.1)
            relative_gap = (train_score - val_score) / scale
            adjusted = val_score - penalty × scale × max(0, relative_gap - tolerance)

        tolerance ist ein relativer Anteil (z.B. 0.05 = 5% Gap wird toleriert).
        Skaleninvariant: Funktioniert identisch für R-Score (~0.3) und Qini (~0.0001).
        """
        if penalty <= 0:
            return val_score
        gap = train_score - val_score  # positiv = Overfitting
        # Epsilon-Floor: Verhindert Division-by-Zero bei val_score ≈ 0.
        # Kein metrischer Floor — skaleninvariant für R-Score und Qini.
        scale = max(abs(val_score), 1e-8)
        relative_gap = gap / scale
        return val_score - penalty * scale * max(0.0, relative_gap - tolerance)

    def _tuning_n_jobs(self) -> int:
        """Anzahl paralleler Optuna-Trials basierend auf parallel_level.

        FMT-Trials nutzen cache_values: Nuisance wird einmalig pro Fold
        gecacht, Trials fitten nur model_final via refit_final() + RScorer.score().
        Trotzdem schwerer als BLT-Trials wegen RScorer-Setup (einmalig pro Fold).

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

    def _create_study(self, study_key: str, use_pruner: bool = True):
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
            pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=2) if use_pruner else None
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

        Nuisance-Modelle (model_y, model_t/model_propensity) werden einmalig pro
        äußerem Fold gecacht (cache_values=True). Pro Trial wird NUR model_final
        als neues Objekt erstellt und via est.refit_final() gefittet. Bei RCT
        wird model_t/model_propensity als DummyClassifier(strategy="prior")
        im Nuisance-Cache verwendet.

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

            # ── Scorer-Wahl: Qini (aggregierter OOF) oder RScore ──
            _scorer_type = self.cfg.final_model_tuning.scorer  # "qini" oder "rscore" (auto bereits aufgelöst)

            _fmt_nt = int(self.cfg.final_model_tuning.n_trials)
            _fmt_is_both = (self.cfg.base_learner.type or "").lower() == "both"
            if _fmt_is_both:
                _fmt_nt *= 2
            _fmt_timeout = self.cfg.final_model_tuning.timeout_seconds
            if _fmt_is_both and _fmt_timeout:
                logging.getLogger("rubin.tuning").warning(
                    "FMT timeout_seconds=%d wird bei 'both' ignoriert (faire Trial-Allokation).", _fmt_timeout)
                _fmt_timeout = None

            if _scorer_type == "qini":
                # ═══ QiniScorer: Aggregierter OOF-Qini (kein RScorer, kein Pruning) ═══
                _tlog.info("FMT '%s': Scorer='qini' — Pruning deaktiviert (QiniScorer benötigt alle Folds).", model_name)
                study = self._create_study(f"final__{model_name}__{base_type}", use_pruner=False)

                def objective(trial):
                    cand_params = _suggest_params(trial, base_type, self.cfg.final_model_tuning.search_space, is_fmt=True)
                    if base_type == "lgbm" or (base_type == "both" and cand_params.get("_learner_type") == "lgbm"):
                        cand_params.setdefault("subsample_freq", 1)

                    all_cate_val, all_y_val, all_t_val = [], [], []
                    fold_train_qinis = []
                    for fold_i, (est, tr, va) in enumerate(_cached_folds):
                        est.model_final = build_base_learner(base_type, {**fmt_fixed_params, **cand_params}, seed=self.seed, task="regressor", parallel_jobs=-1)
                        est.refit_final()
                        cate_val = np.asarray(est.effect(X_tune.iloc[va])).squeeze()
                        all_cate_val.append(cate_val)
                        all_y_val.append(Y_tune[va])
                        all_t_val.append(T_tune[va])
                        # Train-Qini: In-Sample-Prediction → Overfitting-Erkennung
                        if _op > 0:
                            cate_train = np.asarray(est.effect(X_tune.iloc[tr])).squeeze()
                            fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))

                    cate = np.concatenate(all_cate_val)
                    y_agg = np.concatenate(all_y_val)
                    t_agg = np.concatenate(all_t_val)
                    oof_qini = float(qini_coefficient(uplift_curve(y_agg, t_agg, cate)))
                    trial.set_user_attr("raw_val_score", oof_qini)
                    if _op > 0 and fold_train_qinis:
                        return self._apply_overfit_penalty(oof_qini, float(np.mean(fold_train_qinis)), penalty=_op, tolerance=_ot)
                    return oof_qini

                _tlog.info(
                    "FMT '%s': Starte %d Trials (cache_values, OOF-Qini, %s).",
                    model_name, _fmt_nt, "Single-Fold" if _sf else f"{_outer_splits}-Fold",
                )

            else:
                # ═══ RScorer: EconML R-Score pro Fold (bisheriger Pfad) ═══
                study = self._create_study(f"final__{model_name}__{base_type}", use_pruner=True)

                # RScorer pro Fold: Cache-Check + ggf. Erstellen
                _cache_key = (_outer_splits, self.tuning_cv_seed, _sf, len(Y_tune), _op > 0)
                if self._scorer_cache_key == _cache_key and self._scorer_cache_val:
                    _fold_scorers = self._scorer_cache_val
                    _fold_train_scorers = self._scorer_cache_train
                    _tlog.info(
                        "FMT '%s': Gecachte RScorer wiederverwendet (%d Val + %d Train).",
                        model_name, len(_fold_scorers), len(_fold_train_scorers),
                    )
                else:
                    _fold_scorers = []
                    _fold_train_scorers = []
                    for _est_b, _tr_b, _va_b in _cached_folds:
                        _scorer_val = _create_fold_scorer(
                            base_type, dict(base_fixed_params),
                            tuned_roles.get("model_y", tuned_roles.get("model_regression", {})),
                            tuned_roles.get("model_t", tuned_roles.get("model_propensity", {})),
                            Y_tune[_va_b], T_tune[_va_b], X_tune.iloc[_va_b],
                            seed=self.seed, is_rct=_fmt_is_rct,
                            build_base_learner_fn=build_base_learner,
                        )
                        _fold_scorers.append(_scorer_val)
                        if _op > 0:
                            _scorer_tr = _create_fold_scorer(
                                base_type, dict(base_fixed_params),
                                tuned_roles.get("model_y", tuned_roles.get("model_regression", {})),
                                tuned_roles.get("model_t", tuned_roles.get("model_propensity", {})),
                                Y_tune[_tr_b], T_tune[_tr_b], X_tune.iloc[_tr_b],
                                seed=self.seed, is_rct=_fmt_is_rct,
                                build_base_learner_fn=build_base_learner,
                            )
                            _fold_train_scorers.append(_scorer_tr)
                    self._scorer_cache_val = _fold_scorers
                    self._scorer_cache_train = _fold_train_scorers
                    self._scorer_cache_key = _cache_key
                    _tlog.info(
                        "FMT '%s': RScorer (unabhängige Nuisance, 2-Fold T×Y) erstellt + gecacht (%d Val + %d Train).",
                        model_name, len(_fold_scorers), len(_fold_train_scorers),
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
                        val_score = float(_fold_scorers[fold_i].score(est))
                        fold_scores_raw.append(val_score)
                        if _op > 0:
                            fold_train_scores.append(float(_fold_train_scorers[fold_i].score(est)))

                        # Pruning auf RAW Val-Score — Penalty darf Pruning nicht verzerren
                        trial.report(float(np.mean(fold_scores_raw)), fold_i)
                        if trial.should_prune():
                            raise self.optuna.TrialPruned()

                    mean_raw = float(np.mean(fold_scores_raw))
                    trial.set_user_attr("raw_val_score", mean_raw)
                    if _op > 0 and fold_train_scores:
                        return self._apply_overfit_penalty(mean_raw, float(np.mean(fold_train_scores)), penalty=_op, tolerance=_ot)
                    return mean_raw

                _tlog.info(
                    "FMT '%s': Starte %d Trials (cache_values, R-Score, %s).",
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
            # Gecachte Estimatoren + Scorers + Study freigeben
            for _ce, _, _ in _cached_folds:
                del _ce
            del _cached_folds, study  # Scorers bleiben im Cache (self._scorer_cache_*)
            gc.collect()
            _tlog.info("FMT '%s': Study + Cache freigegeben, gc.collect() durchgeführt.", model_name)
            return result

        elif name == "drlearner":
            from econml.dr import DRLearner

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

            # ── Scorer-Wahl ──
            _scorer_type = self.cfg.final_model_tuning.scorer

            _fmt_nt = int(self.cfg.final_model_tuning.n_trials)
            _fmt_is_both = (self.cfg.base_learner.type or "").lower() == "both"
            if _fmt_is_both:
                _fmt_nt *= 2
            _fmt_timeout = self.cfg.final_model_tuning.timeout_seconds
            if _fmt_is_both and _fmt_timeout:
                logging.getLogger("rubin.tuning").warning(
                    "FMT timeout_seconds=%d wird bei 'both' ignoriert (faire Trial-Allokation).", _fmt_timeout)
                _fmt_timeout = None

            if _scorer_type == "qini":
                _tlog.info("FMT '%s': Scorer='qini' — Pruning deaktiviert (QiniScorer benötigt alle Folds).", model_name)
                study = self._create_study(f"final__{model_name}__{base_type}", use_pruner=False)

                def objective(trial):
                    cand_params = _suggest_params(trial, base_type, self.cfg.final_model_tuning.search_space, is_fmt=True)
                    if base_type == "lgbm" or (base_type == "both" and cand_params.get("_learner_type") == "lgbm"):
                        cand_params.setdefault("subsample_freq", 1)

                    all_cate_val, all_y_val, all_t_val = [], [], []
                    fold_train_qinis = []
                    for fold_i, (est, tr, va) in enumerate(_cached_folds):
                        est.model_final = build_base_learner(base_type, {**fmt_fixed_params, **cand_params}, seed=self.seed, task="regressor", parallel_jobs=-1)
                        est.refit_final()
                        cate_val = np.asarray(est.effect(X_tune.iloc[va])).squeeze()
                        all_cate_val.append(cate_val)
                        all_y_val.append(Y_tune[va])
                        all_t_val.append(T_tune[va])
                        if _op > 0:
                            cate_train = np.asarray(est.effect(X_tune.iloc[tr])).squeeze()
                            fold_train_qinis.append(float(qini_coefficient(uplift_curve(Y_tune[tr], T_tune[tr], cate_train))))

                    cate = np.concatenate(all_cate_val)
                    y_agg = np.concatenate(all_y_val)
                    t_agg = np.concatenate(all_t_val)
                    oof_qini = float(qini_coefficient(uplift_curve(y_agg, t_agg, cate)))
                    trial.set_user_attr("raw_val_score", oof_qini)
                    if _op > 0 and fold_train_qinis:
                        return self._apply_overfit_penalty(oof_qini, float(np.mean(fold_train_qinis)), penalty=_op, tolerance=_ot)
                    return oof_qini

                _tlog.info(
                    "FMT '%s': Starte %d Trials (cache_values, OOF-Qini, %s).",
                    model_name, _fmt_nt, "Single-Fold" if _sf else f"{_outer_splits}-Fold",
                )

            else:
                study = self._create_study(f"final__{model_name}__{base_type}", use_pruner=True)

                _cache_key = (_outer_splits, self.tuning_cv_seed, _sf, len(Y_tune), _op > 0)
                if self._scorer_cache_key == _cache_key and self._scorer_cache_val:
                    _fold_scorers = self._scorer_cache_val
                    _fold_train_scorers = self._scorer_cache_train
                    _tlog.info(
                        "FMT '%s': Gecachte RScorer wiederverwendet (%d Val + %d Train).",
                        model_name, len(_fold_scorers), len(_fold_train_scorers),
                    )
                else:
                    _fold_scorers = []
                    _fold_train_scorers = []
                    for _est_b, _tr_b, _va_b in _cached_folds:
                        _scorer_val = _create_fold_scorer(
                            base_type, dict(base_fixed_params),
                            tuned_roles.get("model_regression", tuned_roles.get("model_y", {})),
                            tuned_roles.get("model_propensity", tuned_roles.get("model_t", {})),
                            Y_tune[_va_b], T_tune[_va_b], X_tune.iloc[_va_b],
                            seed=self.seed, is_rct=_fmt_is_rct,
                            build_base_learner_fn=build_base_learner,
                        )
                        _fold_scorers.append(_scorer_val)
                        if _op > 0:
                            _scorer_tr = _create_fold_scorer(
                                base_type, dict(base_fixed_params),
                                tuned_roles.get("model_regression", tuned_roles.get("model_y", {})),
                                tuned_roles.get("model_propensity", tuned_roles.get("model_t", {})),
                                Y_tune[_tr_b], T_tune[_tr_b], X_tune.iloc[_tr_b],
                                seed=self.seed, is_rct=_fmt_is_rct,
                                build_base_learner_fn=build_base_learner,
                            )
                            _fold_train_scorers.append(_scorer_tr)
                    self._scorer_cache_val = _fold_scorers
                    self._scorer_cache_train = _fold_train_scorers
                    self._scorer_cache_key = _cache_key
                    _tlog.info(
                        "FMT '%s': RScorer (unabhängige Nuisance, 2-Fold T×Y) erstellt + gecacht (%d Val + %d Train).",
                        model_name, len(_fold_scorers), len(_fold_train_scorers),
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
                        val_score = float(_fold_scorers[fold_i].score(est))
                        fold_scores_raw.append(val_score)
                        if _op > 0:
                            fold_train_scores.append(float(_fold_train_scorers[fold_i].score(est)))

                        trial.report(float(np.mean(fold_scores_raw)), fold_i)
                        if trial.should_prune():
                            raise self.optuna.TrialPruned()

                    mean_raw = float(np.mean(fold_scores_raw))
                    trial.set_user_attr("raw_val_score", mean_raw)
                    if _op > 0 and fold_train_scores:
                        return self._apply_overfit_penalty(mean_raw, float(np.mean(fold_train_scores)), penalty=_op, tolerance=_ot)
                    return mean_raw

                _tlog.info(
                    "FMT '%s': Starte %d Trials (cache_values, R-Score, %s).",
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
            del _cached_folds, study  # Scorers bleiben im Cache (self._scorer_cache_*)
            gc.collect()
            _tlog.info("FMT '%s': Study + Cache freigegeben, gc.collect() durchgeführt.", model_name)
            return result

        return {}


# ── CausalForest Optuna Tuner ──

# CFT Suchraum: 4 Parameter (aligniert mit EconML CausalForestDML.tune())
# EconML tune() optimiert: min_weight_fraction_leaf, max_depth, min_var_fraction_leaf.
# Rubin ergänzt criterion (mse/het). Alle anderen Parameter werden auf sichere
# EconML-Defaults fixiert — das verhindert degenerierte Konfigurationen
# (insb. den Intercept-Kollaps bei zu hohem min_samples_leaf + niedrigem max_samples)
# und reduziert die Suchraum-Dimension von 9 auf 4 für effizienteres TPE.
# Referenz: EconML CausalForestDML.tune() Source Code, Schuler et al. (2018).
_CF_SEARCH_SPACE = {
    "max_depth":                {"type": "cat",   "choices": [3, 5, 7, 10, 15, None]},
    "min_weight_fraction_leaf": {"type": "float", "low": 0.0001, "high": 0.05, "log": True},
    "min_var_fraction_leaf":    {"type": "float", "low": 0.0005, "high": 0.05, "log": True},
    "criterion":                {"type": "cat",   "choices": ["mse", "het"]},
}

# Fixe Defaults für nicht-getunte Wald-Parameter (EconML Defaults).
# Diese werden bei jedem Trial gesetzt, um konsistentes Verhalten sicherzustellen.
_CF_FIXED_DEFAULTS = {
    "n_estimators":         100,    # EconML Default
    "min_samples_leaf":     5,      # EconML Default
    "min_samples_split":    10,     # EconML Default
    "max_features":         "auto", # EconML Default (= alle Features)
    "max_samples":          0.45,   # EconML Default
    "min_impurity_decrease": 0.0,   # EconML Default
}


