from __future__ import annotations

"""CausalForest-Tuning (CFT) mit Optuna.

Optimiert Forest-Parameter für CausalForestDML (setattr + refit_final)
und CausalForest (frischer Fit pro Trial).

Architektur: Klasse `CausalForestTuner` — symmetrisch zu BaseLearnerTuner
und FinalModelTuner. Hält den RScorer-Cache als Instanz-State, damit
Scorer zwischen CausalForestDML- und CausalForest-Calls wiederverwendet
werden können (spart ~80s pro zusätzlichem Modell).
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
import logging
import gc

from sklearn.model_selection import StratifiedKFold

from rubin.tuning.common import (
    _iter_stratified_or_kfold, _safe_import_optuna, _create_fold_scorer,
    _log_trial_diagnostics, build_base_learner,
)
from rubin.evaluation.uplift_metrics import uplift_curve, uplift_curve_mt_argmax, qini_coefficient

# Tuning n_estimators: Kleine Forests (100) für Speed, Production nutzt mehr
_CFT_TUNING_N_ESTIMATORS = 100  # EconML tune() Konvention

# CFT Suchraum: 4 Parameter (aligniert mit EconML CausalForestDML.tune())
_CF_SEARCH_SPACE = {
    "max_depth":                {"type": "cat",   "choices": [3, 5, 7, 10, 15, None]},
    "min_weight_fraction_leaf": {"type": "float", "low": 0.0001, "high": 0.05, "log": True},
    "min_var_fraction_leaf":    {"type": "float", "low": 0.0005, "high": 0.05, "log": True},
    "criterion":                {"type": "cat",   "choices": ["mse", "het"]},
}

# Fixe Defaults für nicht-getunte Forest-Parameter (EconML Defaults).
_CF_FIXED_DEFAULTS = {
    "n_estimators":         100,
    "min_samples_leaf":     5,
    "min_samples_split":    10,
    "max_features":         "auto",
    "max_samples":          0.45,
    "min_impurity_decrease": 0.0,
}


class CausalForestTuner:
    """Optuna-basiertes Tuning für CausalForestDML und CausalForest.

    Symmetrisch zu BaseLearnerTuner und FinalModelTuner. Hält den
    RScorer-Cache als Instanz-State, damit Scorer zwischen
    CausalForestDML- und CausalForest-Calls wiederverwendet werden.

    Suchraum: 4 Parameter (aligniert mit EconML CausalForestDML.tune()):
    - max_depth, min_weight_fraction_leaf, min_var_fraction_leaf, criterion
    Alle anderen Forest-Parameter auf EconML-Defaults fixiert.

    Scoring: Externer EconML RScorer (unabhängige Nuisance, 2-Fold T×Y).

    Zwei Pfade:
    - CausalForestDML: Nuisance einmalig gecacht (cache_values=True),
      Trials ändern Forest-Parameter via setattr() + refit_final().
    - CausalForest (GRF): Pro Trial neuer CausalForestAdapter.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.seed = int(cfg.constants.random_seed)
        self.tuning_cv_seed = int(cfg.constants.tuning_seed)
        self.optuna = _safe_import_optuna()
        self.best_scores: Dict[str, float] = {}
        self._scorer_cache: list = []
        self._scorer_cache_key: tuple = ()

    @staticmethod
    def _scorer_fingerprint(*dicts) -> str:
        """Deterministischer Fingerprint einer Nuisance-Konfiguration für den Scorer-Cache-Key."""
        import json
        return json.dumps(
            [dict(sorted((d or {}).items())) for d in dicts],
            sort_keys=True, default=str,
        )

    def _study_seed(self, study_key: str) -> int:
        """Pro-Study-Seed aus tuning_cv_seed + sha256(study_key) ableiten.

        Verhindert, dass CausalForestDML- und CausalForest-Studies (gleicher
        Suchraum) exakt dieselbe Sampler-Sequenz durchlaufen — analog zu
        BaseLearnerTuner/FinalModelTuner.
        """
        import hashlib
        key_hash = int(hashlib.sha256(study_key.encode()).hexdigest(), 16) % (2**31)
        return (int(self.tuning_cv_seed) + key_hash) % (2**31)

    @staticmethod
    def _apply_overfit_penalty(val_score: float, train_score: float,
                               penalty: float = 0.0, tolerance: float = 0.10,
                               max_penalized_gap: float = 1.0) -> float:
        """Skaleninvariante Train-Val-Gap-Penalty (identisch zu BaseLearnerTuner).

        max_penalized_gap deckelt den bestraften Exzess-Gap (Saturierung); <=0 = kein Cap.
        """
        if penalty <= 0:
            return val_score
        scale = max(abs(val_score), 1e-8)
        relative_gap = (train_score - val_score) / scale
        excess = max(0.0, relative_gap - tolerance)
        if max_penalized_gap is not None and max_penalized_gap > 0:
            excess = min(excess, max_penalized_gap)
        return val_score - penalty * scale * excess

    def tune(
        self,
        model_type: str,
        X: pd.DataFrame,
        Y: np.ndarray,
        T: np.ndarray,
        n_trials: int = 50,
        nuisance_params_y: dict = None,
        nuisance_params_t: dict = None,
        single_fold: bool = False,
    ) -> Dict[str, Any]:
        """Tuned CausalForestDML oder CausalForest.

        RScorer werden intern gecacht (self._scorer_cache). Beim ersten
        tune()-Aufruf (z.B. CausalForestDML) werden die Scorer erstellt;
        beim zweiten (z.B. CausalForest) wiederverwendet.

        Returns:
            Dict mit 'best_params', 'best_score', 'n_trials_completed'.
        """
        optuna = self.optuna
        _tlog = logging.getLogger("rubin.tuning")

        seed = self.seed
        tuning_cv_seed = self.tuning_cv_seed
        inner_cv = int(self.cfg.data_processing.dml_crossfit_folds)

        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        Y_np = np.asarray(Y).ravel()
        T_np = np.asarray(T).ravel()

        base_type = (self.cfg.base_learner.type or "catboost").lower()
        # "both" NICHT vorab auf "catboost" kollabieren: Die BLT-getunten
        # Nuisance-Params enthalten im both-Modus '_learner_type' (+ ggf.
        # genestete lgbm/catboost-fixed_params). Nur der both-Zweig in
        # build_base_learner() wählt daraus den richtigen Learner und entfernt
        # die Meta-Keys — beim Kollabieren landeten sie ungefiltert im
        # CatBoost-Konstruktor (TypeError '_learner_type') bzw. LGBM-getunte
        # Params im falschen Builder.

        is_cfdml = model_type.lower() == "causalforestdml"

        # ── Merge user overrides into search space ──
        user_ss = {}
        if hasattr(self.cfg, "causal_forest") and hasattr(self.cfg.causal_forest, "search_space"):
            raw = self.cfg.causal_forest.search_space
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
                if user_ss[pname].get("choices") is not None:
                    entry["choices"] = user_ss[pname]["choices"]
            effective_ss[pname] = entry

        # UI-Chip-Selektoren
        _depth_choices = getattr(self.cfg.causal_forest, "depth_choices", None)
        if _depth_choices and isinstance(_depth_choices, list) and len(_depth_choices) > 0:
            effective_ss["max_depth"]["choices"] = [
                None if v == "None" else v for v in _depth_choices
            ]
        _criterion_choices = getattr(self.cfg.causal_forest, "criterion_choices", None)
        if _criterion_choices and isinstance(_criterion_choices, list) and len(_criterion_choices) > 0:
            effective_ss["criterion"]["choices"] = _criterion_choices

        # Fixe Defaults + User-Overrides
        _effective_fixed = dict(_CF_FIXED_DEFAULTS)
        _user_fixed = getattr(self.cfg.causal_forest, "forest_fixed_params", {}) or {}
        if isinstance(_user_fixed, dict):
            _effective_fixed.update(_user_fixed)
        _effective_fixed["n_estimators"] = _CFT_TUNING_N_ESTIMATORS

        _cft_is_rct = getattr(self.cfg, "study_type", "rct") == "rct"
        _CFDML_EXPLICIT_KEYS = {
            "model_y", "model_t", "discrete_treatment", "discrete_outcome",
            "cv", "random_state", "n_jobs", "inference", "mc_iters", "mc_agg",
        }

        # ── Fold-Daten vorbereiten ──
        _cf_strata_pre = T_np.astype(int) * 10 + np.clip(Y_np, 0, 1).astype(int)
        _splits_pre = list(_iter_stratified_or_kfold(
            labels=_cf_strata_pre, n_splits=inner_cv, seed=tuning_cv_seed,
        ))
        _n_pre = 1 if single_fold else inner_cv

        # ── GRF: Fold-Splits vorberechnen ──
        _grf_fold_data = None
        if not is_cfdml:
            _grf_fold_data = [(_tr, _va) for _tr, _va in _splits_pre[:_n_pre]]
            _rct_suffix = " model_t = DummyClassifier (RCT: konstante Propensity)." if _cft_is_rct else ""
            _tlog.info("CFT '%s': %d Fold-Splits vorberechnet.%s", model_type, len(_grf_fold_data), _rct_suffix)

        # ── CFDML: Nuisance cachen ──
        _cfdml_cached_folds = None
        from rubin.model_registry import CausalForestAdapter
        if is_cfdml:
            from econml.dml import CausalForestDML
            _default_forest = dict(_effective_fixed)
            _default_forest.update({
                k: spec.get("low", spec.get("choices", [None])[0])
                for k, spec in effective_ss.items()
            })
            for _ek in _CFDML_EXPLICIT_KEYS:
                _default_forest.pop(_ek, None)
            _cfdml_cached_folds = []
            for _fi, (_tr, _va) in enumerate(_splits_pre[:_n_pre]):
                if _cft_is_rct:
                    from sklearn.dummy import DummyClassifier
                    _cft_model_t = DummyClassifier(strategy="prior")
                else:
                    _cft_model_t = build_base_learner(base_type, nuisance_params_t or {}, seed=seed, task="classifier", parallel_jobs=-1)
                est = CausalForestDML(
                    model_y=build_base_learner(base_type, nuisance_params_y or {}, seed=seed, task="classifier", parallel_jobs=-1),
                    model_t=_cft_model_t,
                    discrete_treatment=True, discrete_outcome=True,
                    cv=StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed),
                    random_state=seed, n_jobs=-1, inference=False,
                    # mc_iters/mc_agg wie im finalen Training (model_registry) und in FMT,
                    # damit die Forest-Parameter auf denselben mc-aggregierten Nuisance-
                    # Residuen getunt werden, die das Produktionsmodell sieht.
                    mc_iters=self.cfg.data_processing.mc_iters,
                    mc_agg=self.cfg.data_processing.mc_agg,
                    **_default_forest,
                )
                est.fit(Y_np[_tr], T_np[_tr], X=X_np[_tr], cache_values=True)
                _cfdml_cached_folds.append((est, _tr, _va))
            gc.collect()
            _rct_suffix = " model_t = DummyClassifier (RCT: konstante Propensity)." if _cft_is_rct else ""
            _tlog.info(
                "CFT '%s': Nuisance gecacht für %d äußere Folds (je %d innere × model_y + model_t). "
                "Trials fitten nur noch Forest via refit_final().%s",
                model_type, len(_cfdml_cached_folds), inner_cv, _rct_suffix,
            )

        # ── Scorer-Wahl ──
        _scorer_type = getattr(self.cfg.causal_forest, "scorer", "rscore")

        if _scorer_type in ("qini", "qini_argmax"):
            # ═══ QiniScorer: Aggregierter OOF-Qini (kein RScorer, kein Pruning) ═══
            # "qini": binäres Ranking. "qini_argmax": Multi-Treatment-Ranking über
            # alle Arme (uplift_curve_mt_argmax; bei K=2 exakt == "qini").
            _use_argmax = _scorer_type == "qini_argmax"
            _op = getattr(self.cfg.causal_forest, "overfit_penalty", 0.0)
            _ot = getattr(self.cfg.causal_forest, "overfit_tolerance", 0.10)
            _omg = getattr(self.cfg.causal_forest, "overfit_max_penalized_gap", 1.0)
            _tlog.info("CFT '%s': Scorer='%s' — Pruning deaktiviert (QiniScorer benötigt alle Folds).",
                       model_type, _scorer_type)

            def _cft_cate(est_or_adapter, X_part):
                if _use_argmax:
                    return np.asarray(est_or_adapter.const_marginal_effect(X_part)).reshape(len(X_part), -1)
                return np.asarray(est_or_adapter.effect(X_part)).squeeze()

            def _cft_qini(y_part, t_part, cate_part):
                y_i, t_i = np.asarray(y_part).astype(int), np.asarray(t_part).astype(int)
                if _use_argmax:
                    return float(qini_coefficient(uplift_curve_mt_argmax(
                        y_i, t_i, np.asarray(cate_part).reshape(len(y_i), -1))))
                return float(qini_coefficient(uplift_curve(y_i, t_i, cate_part)))

            def objective(trial):
                params = {}
                for pname, spec in effective_ss.items():
                    if spec["type"] == "int":
                        _step = 4 if pname == "n_estimators" else 1
                        params[pname] = trial.suggest_int(pname, int(spec["low"]), int(spec["high"]), step=_step, log=spec.get("log", False) if _step == 1 else False)
                    elif spec["type"] == "float":
                        params[pname] = trial.suggest_float(pname, float(spec["low"]), float(spec["high"]), log=spec.get("log", False))
                    elif spec["type"] == "cat":
                        params[pname] = trial.suggest_categorical(pname, spec["choices"])

                all_params = dict(_effective_fixed)
                all_params.update(params)
                all_params["n_estimators"] = _CFT_TUNING_N_ESTIMATORS
                for _ek in ("model_y", "model_t", "discrete_treatment", "discrete_outcome",
                             "cv", "random_state", "n_jobs", "inference"):
                    all_params.pop(_ek, None)

                all_cate_val, all_y_val, all_t_val = [], [], []
                fold_train_qinis = []
                if is_cfdml:
                    for fold_i, (est, tr, va) in enumerate(_cfdml_cached_folds):
                        try:
                            for k, v in all_params.items():
                                setattr(est, k, v)
                            est.refit_final()
                            cate_val = _cft_cate(est, X_np[va])
                            all_cate_val.append(cate_val)
                            all_y_val.append(Y_np[va])
                            all_t_val.append(T_np[va])
                            if _op > 0:
                                cate_train = _cft_cate(est, X_np[tr])
                                fold_train_qinis.append(_cft_qini(Y_np[tr], T_np[tr], cate_train))
                        except Exception as _e:
                            _tlog.debug("CFT '%s' Trial %d Fold %d: %s", model_type, trial.number, fold_i, _e)
                            return -1e12
                else:
                    for fold_i, (tr, va) in enumerate(_grf_fold_data):
                        try:
                            adapter = CausalForestAdapter(
                                random_state=seed, n_jobs=-1, inference=False,
                                **all_params,
                            )
                            adapter.fit(Y_np[tr], T_np[tr], X=X_np[tr])
                            cate_val = _cft_cate(adapter, X_np[va])
                            all_cate_val.append(cate_val)
                            all_y_val.append(Y_np[va])
                            all_t_val.append(T_np[va])
                            if _op > 0:
                                cate_train = _cft_cate(adapter, X_np[tr])
                                fold_train_qinis.append(_cft_qini(Y_np[tr], T_np[tr], cate_train))
                            del adapter
                            gc.collect()
                        except Exception as _e:
                            _tlog.debug("CFT '%s' Trial %d Fold %d: %s", model_type, trial.number, fold_i, _e)
                            return -1e12

                cate = np.concatenate(all_cate_val)
                y_agg = np.concatenate(all_y_val)
                t_agg = np.concatenate(all_t_val)
                oof_qini = _cft_qini(y_agg, t_agg, cate)
                trial.set_user_attr("raw_val_score", oof_qini)
                if _op > 0 and fold_train_qinis:
                    return self._apply_overfit_penalty(oof_qini, float(np.mean(fold_train_qinis)), penalty=_op, tolerance=_ot, max_penalized_gap=_omg)
                return oof_qini

            # Study ohne Pruner
            _cft_n_startup = max(5, n_trials // 5)
            _cft_n_startup = min(_cft_n_startup, max(3, n_trials // 2))
            sampler = optuna.samplers.TPESampler(
                seed=self._study_seed(f"cf_tune__{model_type}"), multivariate=True, group=True,
                n_startup_trials=_cft_n_startup,
            )
            study = optuna.create_study(
                direction="maximize", sampler=sampler, pruner=optuna.pruners.NopPruner(),
                study_name=f"cf_tune__{model_type}",
            )
            _fold_lbl = "Single-Fold" if single_fold else f"{inner_cv}-Fold"
            _cft_mode = "cache_values" if is_cfdml else "CausalForestAdapter"
            _tlog.info("CFT '%s': Starte %d Trials (%s, OOF-Qini, %s).", model_type, n_trials, _cft_mode, _fold_lbl)

        else:
            # ═══ RScorer: EconML R-Score pro Fold ═══
            # RScorer: Cache prüfen oder neu erstellen.
            # Cache-Key inkl. Nuisance-Fingerprint + Fold-Konfiguration: Scorer
            # werden zwischen CausalForestDML- und CausalForest-Calls nur dann
            # wiederverwendet, wenn Nuisance-Params, base_type, RCT-Status und
            # Fold-Setup identisch sind (in der Praxis der Fall, da der Caller
            # für beide dieselben BLT-Nuisances + Splits liefert). Verhindert
            # blinde Wiederverwendung bei abweichender Konfiguration.
            _cft_cache_key = (
                base_type, bool(_cft_is_rct), inner_cv, int(tuning_cv_seed),
                bool(single_fold), len(Y_np),
                self._scorer_fingerprint(nuisance_params_y, nuisance_params_t),
            )
            if self._scorer_cache and self._scorer_cache_key == _cft_cache_key:
                _cft_fold_scorers = self._scorer_cache
                _tlog.info("CFT '%s': Gecachte RScorer wiederverwendet (%d Val).", model_type, len(_cft_fold_scorers))
            else:
                _cft_fold_scorers = []
                _scorer_folds = (
                    [(_tr_b, _va_b) for _, _tr_b, _va_b in _cfdml_cached_folds]
                    if is_cfdml else _grf_fold_data
                )
                for _tr_b, _va_b in _scorer_folds:
                    _scorer = _create_fold_scorer(
                        base_type, {},
                        nuisance_params_y or {}, nuisance_params_t or {},
                        Y_np[_va_b], T_np[_va_b], X_np[_va_b],
                        seed=seed, is_rct=_cft_is_rct,
                        build_base_learner_fn=build_base_learner,
                    )
                    _cft_fold_scorers.append(_scorer)
                self._scorer_cache = _cft_fold_scorers
                self._scorer_cache_key = _cft_cache_key
                _tlog.info("CFT '%s': RScorer (unabhängige Nuisance, 2-Fold T×Y) erstellt + gecacht (%d Val).", model_type, len(_cft_fold_scorers))

            def objective(trial):
                params = {}
                for pname, spec in effective_ss.items():
                    if spec["type"] == "int":
                        _step = 4 if pname == "n_estimators" else 1
                        params[pname] = trial.suggest_int(pname, int(spec["low"]), int(spec["high"]), step=_step, log=spec.get("log", False) if _step == 1 else False)
                    elif spec["type"] == "float":
                        params[pname] = trial.suggest_float(pname, float(spec["low"]), float(spec["high"]), log=spec.get("log", False))
                    elif spec["type"] == "cat":
                        params[pname] = trial.suggest_categorical(pname, spec["choices"])

                all_params = dict(_effective_fixed)
                all_params.update(params)
                all_params["n_estimators"] = _CFT_TUNING_N_ESTIMATORS
                for _ek in ("model_y", "model_t", "discrete_treatment", "discrete_outcome",
                             "cv", "random_state", "n_jobs", "inference"):
                    all_params.pop(_ek, None)

                fold_scores = []
                if is_cfdml:
                    for fold_i, (est, tr, va) in enumerate(_cfdml_cached_folds):
                        try:
                            for k, v in all_params.items():
                                setattr(est, k, v)
                            est.refit_final()
                            score = float(_cft_fold_scorers[fold_i].score(est))
                            fold_scores.append(score)
                        except Exception as _e:
                            _tlog.debug("CFT '%s' Trial %d Fold %d: %s", model_type, trial.number, fold_i, _e)
                            fold_scores.append(-1e12)
                        trial.report(float(np.mean(fold_scores)), fold_i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                else:
                    for fold_i, (tr, va) in enumerate(_grf_fold_data):
                        try:
                            adapter = CausalForestAdapter(
                                random_state=seed, n_jobs=-1, inference=False,
                                **all_params,
                            )
                            adapter.fit(Y_np[tr], T_np[tr], X=X_np[tr])
                            score = float(_cft_fold_scorers[fold_i].score(adapter))
                            fold_scores.append(score)
                            del adapter
                            gc.collect()
                        except Exception as _e:
                            _tlog.debug("CFT '%s' Trial %d Fold %d: %s", model_type, trial.number, fold_i, _e)
                            fold_scores.append(-1e12)
                        trial.report(float(np.mean(fold_scores)), fold_i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                return float(np.mean(fold_scores))

            # Study mit Pruner
            _cft_n_startup = max(5, n_trials // 5)
            _cft_n_startup = min(_cft_n_startup, max(3, n_trials // 2))
            sampler = optuna.samplers.TPESampler(
                seed=self._study_seed(f"cf_tune__{model_type}"), multivariate=True, group=True,
                n_startup_trials=_cft_n_startup,
            )
            pruner = optuna.pruners.MedianPruner(n_startup_trials=max(3, n_trials // 10), n_warmup_steps=2)
            study = optuna.create_study(
                direction="maximize", sampler=sampler, pruner=pruner,
                study_name=f"cf_tune__{model_type}",
            )
            _fold_lbl = "Single-Fold" if single_fold else f"{inner_cv}-Fold"
            _cft_mode = "cache_values" if is_cfdml else "CausalForestAdapter"
            _tlog.info("CFT '%s': Starte %d Trials (%s, R-Score, %s).", model_type, n_trials, _cft_mode, _fold_lbl)
        study.optimize(objective, n_trials=n_trials, n_jobs=1, catch=(Exception,))
        n_complete, _, _ = _log_trial_diagnostics(study, model_type, n_jobs=1, stage="CFT")

        try:
            best = dict(study.best_trial.params)
            best_score = float(study.best_value)
        except ValueError:
            _tlog.warning("CFT '%s': Keine abgeschlossenen Trials. Verwende Default-Parameter.", model_type)
            best = {}
            best_score = None

        if best_score is not None and best_score <= -1e11:
            _tlog.warning("CFT '%s': Alle Trials fehlgeschlagen (best_score=%.2g). Verwende Default-Parameter.", model_type, best_score)
            best = {}
            best_score = None

        if best_score is not None:
            self.best_scores[model_type] = best_score

        # Cleanup
        del study
        if _cfdml_cached_folds is not None:
            for _ce, _, _ in _cfdml_cached_folds:
                del _ce
            del _cfdml_cached_folds
        gc.collect()
        _tlog.info("CFT '%s': Study + Cache freigegeben, gc.collect() durchgeführt.", model_type)

        return {"best_params": best, "best_score": best_score, "n_trials_completed": n_complete}
