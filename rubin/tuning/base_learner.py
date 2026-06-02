from __future__ import annotations

"""Base-Learner-Tuning (BLT) mit Optuna.

Optimiert Nuisance-Modelle (Outcome, Propensity) mit aufgabenspezifischen
CV-Objectives. Bei RCT dient das Propensity-Tuning als Diagnose-Check.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import logging
import os
import gc

from sklearn.metrics import log_loss as _log_loss_fn

from rubin.tuning.common import (
    _compute_skill_score, _iter_stratified_or_kfold, _safe_import_optuna,
    _suggest_params, build_base_learner, _log_trial_diagnostics,
    TunedSet, TuningTask,
)
from rubin.settings import AnalysisConfig, SearchSpaceConfig
from rubin.utils.data_utils import available_cpu_count

class BaseLearnerTuner:
    """Optimiert Base-Learner-Aufgaben und teilt Ergebnisse zwischen kompatiblen Rollen.

    Die Logik ist task-basiert:
    - Aus den angeforderten kausalen Modellen werden die tatsächlich benötigten
      internen Lernaufgaben abgeleitet.
    - Identische Aufgaben werden nur einmal getunt.
    - Die besten Parameter werden anschließend allen passenden Rollen zugeordnet.

    TPE-Sampler: multivariate=True, group=True, constant_liar=True.
    n_startup_trials wird auf max. 50% der effektiven Trials pro Task gekappt.
    MedianPruner: n_warmup_steps=2 (erst ab 3. Fold prunen).

    Bei RCT: Propensity-Tasks laufen mit 20 Trials als Diagnose-Check (Skill ≈ 0
    bestätigt Randomisierung). Training nutzt dann DummyClassifier.
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

    def _score_classifier(self, y_true: np.ndarray, proba: np.ndarray) -> float:
        """Bewertet einen Classifier via Log-Loss (negiert, höher = besser).

        Behandelt binär und multiclass einheitlich: bei 1D-proba (binär) wird
        zu einer 2-Spalten-Matrix erweitert, bei 2D-proba (multiclass) direkt
        an log_loss übergeben.

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
                val_score = self._score_classifier(target[va], proba_val)
                if _op_active:
                    proba_tr = model.predict_proba(X_mat.iloc[tr])
                    train_scores.append(self._score_classifier(target[tr], proba_tr))
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
        """Bestraft Overfit-Gap zwischen Train- und Val-Score (skaleninvariant, relative Tolerance).

        Wird NUR auf den finalen Trial-Score angewendet (K-Fold-Mittelwerte),
        NICHT auf Einzelfold-Scores. Pruning basiert auf Raw-Val-Scores.

        Formel:
            scale = max(|val_score|, 1e-8)
            relative_gap = (train_score - val_score) / scale
            adjusted = val_score - penalty × scale × max(0, relative_gap - tolerance)

        Skaleninvariant: Ein 20%-Gap wird identisch bestraft unabhängig davon,
        ob die Metrik R-Score (~0.3) oder Qini (~0.0001) ist.
        tolerance ist ein relativer Anteil (z.B. 0.05 = 5% Gap wird toleriert).
        """
        if penalty is None:
            penalty = getattr(self.cfg.tuning, "overfit_penalty", 0.0)
        if tolerance is None:
            tolerance = getattr(self.cfg.tuning, "overfit_tolerance", 0.15)
        if penalty <= 0:
            return val_score
        gap = train_score - val_score  # positiv = Overfitting
        # Epsilon-Floor: Verhindert Division-by-Zero bei val_score ≈ 0.
        # Kein metrischer Floor (war vorher 0.1) — die Penalty ist damit
        # skaleninvariant und funktioniert identisch für R-Score (~0.3)
        # und Qini (~0.0001). Ein 20%-Gap ist ein 20%-Gap unabhängig von der Skala.
        scale = max(abs(val_score), 1e-8)
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

    def _objective_xlearner_cate(self, params: Dict[str, Any], X_mat: np.ndarray, Y: np.ndarray, T: np.ndarray, pseudo: np.ndarray, trial=None) -> float:
        """Tuning-Objective für XLearner CATE-Modelle (cate_models).

        Bewertet Regressoren auf vorberechneten Pseudo-Outcomes pro Treatment-Gruppe.
        Nutzt _score_regressor (neg. MSE) und
        optionale Overfit-Penalty (konsistent mit allen anderen BLT-Objectives).

        Die Pseudo-Outcomes werden EINMALIG in _tune_task vorberechnet und
        hereingereicht (sie hängen nur von den getunten Outcome-Nuisances +
        tuning_cv_seed ab, nicht von den Trial-Params der cate_models).

        Pruning basiert auf RAW Val-Scores. Penalty wirkt nur auf den finalen
        Trial-Score (stabile Mittelwerte statt verrauschte Einzelfolds).
        """
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

        # XLearner pseudo_effect: Pseudo-Outcomes EINMALIG vorberechnen.
        # Sie hängen nur von den getunten Outcome-Nuisances (shared_params,
        # fix über Trials) + tuning_cv_seed (fix) ab, NICHT von den Trial-Params
        # der cate_models. Vorher wurden sie pro Trial neu gebaut (n_trials×
        # redundante 2K Nuisance-Fits). Identisches Ergebnis, deutlich schneller.
        _xl_pseudo = None
        if task.objective_family == "pseudo_effect":
            _xl_nuisance = dict(shared_params.get("xlearner__models") or fixed_defaults)
            _xl_pseudo = self._build_xlearner_pseudo_outcomes(
                X_mat,
                np.asarray(Y)[row_indices].astype(int),
                T_task.astype(int),
                nuisance_params=_xl_nuisance,
            )

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
            # XLearner CATE-Modelle (Pseudo-Outcomes oben einmalig vorberechnet)
            if task.objective_family == "pseudo_effect":
                return self._objective_xlearner_cate(
                    params,
                    X_mat=X_mat,
                    Y=np.asarray(Y)[row_indices].astype(int),
                    T=T_task.astype(int),
                    pseudo=_xl_pseudo,
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
                        "RCT-Warnung: Propensity-Modell erreicht neg_log_loss=%.4f (bei randomisiertem "
                        "Treatment sollte das Modell nicht besser als Zufall sein). "
                        "Prüfe ob Treatment tatsächlich randomisiert ist oder ob Post-Treatment-"
                        "Variablen im Datensatz sind.",
                        best_val,
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

