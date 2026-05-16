from __future__ import annotations

"""Registry für kausale Learner und Base Learner.
Die Registry kapselt,
- welche kausalen Modelle trainiert werden können,
- wie ein Modell instanziiert wird,
- wie Base Learner (LightGBM/CatBoost) konsistent gebaut werden.
Die Konfiguration bleibt damit schlank; die Factory kümmert sich um die
konkrete Modellinstanziierung.
Bei RCT (is_rct=True) werden Propensity-Rollen automatisch durch
DummyClassifier(strategy="prior") ersetzt. CausalForest-Parameter
werden via _sanitize_forest_params validiert (n_estimators ÷ 4)."""


from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional

from econml.dml import CausalForestDML, NonParamDML, LinearDML
from econml.dr import DRLearner
from sklearn.model_selection import StratifiedKFold
from econml.metalearners import SLearner, TLearner, XLearner
from econml._cate_estimator import BaseCateEstimator

from rubin.tuning_optuna import build_base_learner


class CausalForestAdapter(BaseCateEstimator):
    """Adapter für ``econml.grf.CausalForest`` → EconML DML-kompatible API.

    Der reine CausalForest (GRF) schätzt Treatment-Effekte direkt (Honest Splitting, eigene
    Propensity-/Outcome-Schätzung im Wald), ohne vorgelagerte Nuisance-Modelle.
    Die API weicht von der DML-Familie ab:
    - ``fit(X, T, Y)`` statt ``fit(Y, T, X=X)``
    - ``predict(X)`` statt ``effect(X)``

    Dieser Adapter macht CausalForest transparent für die rubin-Pipeline
    und ist als BaseCateEstimator kompatibel mit EnsembleCateEstimator.
    """

    def __init__(self, **kwargs):
        from econml.grf import CausalForest
        self._cf = CausalForest(**kwargs)
        self._kwargs = kwargs

    def fit(self, Y, T, X=None):
        import numpy as np
        X_np = np.asarray(X, dtype=np.float64) if X is not None else None
        T_np = np.asarray(T, dtype=np.float64).ravel()
        Y_np = np.asarray(Y, dtype=np.float64).ravel()
        self._cf.fit(X_np, T_np, Y_np)
        return self

    def effect(self, X):
        import numpy as np
        X_np = np.asarray(X, dtype=np.float64)
        return self._cf.predict(X_np)

    def const_marginal_effect(self, X):
        return self.effect(X)

    def marginal_effect(self, T, X):
        return self.effect(X)

    @property
    def feature_importances_(self):
        if hasattr(self._cf, "feature_importances_"):
            return self._cf.feature_importances_
        return None

    @classmethod
    def from_fitted(cls, cf):
        """Factory-Method: Wraps einen bereits gefitteten CausalForest.

        Sicherer als __new__()-Pattern, weil alle Attribute konsistent gesetzt werden.
        Verwendung im RScorer-Grid-Search: cf.fit() → Adapter.from_fitted(cf) → rscorer.score(adapter).
        """
        adapter = cls.__new__(cls)
        adapter._cf = cf
        adapter._kwargs = {}
        adapter._tune_result = {}
        return adapter

    # ── CausalForest Tune Grid ──
    # Identisch mit CausalForestDML.tune(params='auto') für Konsistenz.
    # min_weight_fraction_leaf: Mindestanteil am Gesamtgewicht pro Blatt (Regularisierung)
    # max_depth: Baumtiefe (None → unbegrenzt, niedrig → stärkere Regularisierung)
    # min_var_fraction_leaf: Mindest-Treatment-Varianz pro Blatt (Identifikations-Schutz)
    TUNE_GRID = [
        {"min_weight_fraction_leaf": mwfl, "max_depth": md, "min_var_fraction_leaf": mvfl}
        for mwfl in [0.0001, 0.01]
        for md in [3, 5, None]
        for mvfl in [0.001, 0.01]
    ]  # 2 × 3 × 2 = 12 Kombinationen (= EconML Default)

    TUNE_GRID_INTENSIVE = [
        {"min_weight_fraction_leaf": mwfl, "max_depth": md, "min_var_fraction_leaf": mvfl, "criterion": cr}
        for mwfl in [0.0001, 0.01]
        for md in [3, 5, 8, None]
        for mvfl in [None, 0.001, 0.01]
        for cr in ["mse", "het"]
    ]  # 2 × 4 × 3 × 2 = 48 Kombinationen (identisch mit CausalForestDML intensiv)

    def tune(self, X, T, Y, intensive=False, rscorer=None, model_y=None, model_t=None, nuisance_cv=5):
        """Grid-Search über CausalForest-Waldparameter mit RScorer-Evaluation.

        Verwendet EconML's RScorer (Nie & Wager, 2021) zur Bewertung jeder
        Parameterkombination. RScorer fittet eigene Nuisance-Modelle (model_y,
        model_t) und berechnet den normalisierten R-Score:

            R-Score = 1 - E[(Y_res - τ(X)·T_res)²] / base_loss

        Höher = besser. Misst, wie viel zusätzliche Varianz das CATE-Modell
        gegenüber einem konstanten Effekt erklärt. Identisch mit der R-Score-Metrik,
        die im CFT (CausalForest-Tuning via Optuna) und FMT verwendet wird.

        Args:
            rscorer: Vorgefitteter RScorer. Wenn None, wird einer erstellt.
            model_y: Nuisance-Modell für E[Y|X]. Wenn None, RandomForest-Fallback.
            model_t: Nuisance-Modell für E[T|X]. Wenn None, RandomForest-Fallback.
                     Idealerweise dieselben Base-Learner wie NonParamDML/CausalForestDML.
            nuisance_cv: CV-Folds für Nuisance-Cross-Fitting im RScorer.
                         Sollte der inneren CV-Konfiguration entsprechen.
        """
        import numpy as np
        from econml.grf import CausalForest

        X_np = np.asarray(X, dtype=np.float64)
        T_np = np.asarray(T, dtype=np.float64).ravel()
        Y_np = np.asarray(Y, dtype=np.float64).ravel()

        # RScorer: Nuisance-Modelle einmalig fitten, dann jede Kombi bewerten
        if rscorer is None:
            from econml.score import RScorer
            if model_y is None:
                # Fallback: RandomForestClassifier, weil discrete_outcome=True
                # → EconML ruft intern predict_proba() auf model_y auf.
                from sklearn.ensemble import RandomForestClassifier as _RFC
                model_y = _RFC(n_estimators=100, min_samples_leaf=20, n_jobs=-1)
            if model_t is None:
                from sklearn.ensemble import RandomForestClassifier
                model_t = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, n_jobs=-1)
            rscorer = RScorer(
                model_y=model_y, model_t=model_t,
                discrete_treatment=True, discrete_outcome=True,
                cv=StratifiedKFold(n_splits=nuisance_cv, shuffle=True, random_state=self._kwargs.get("random_state", 42)),
                random_state=self._kwargs.get("random_state", 42),
            )
            rscorer.fit(Y_np, T_np, X=X_np)

        fixed_params = {
            "n_estimators": self._kwargs.get("n_estimators", 100),
            "random_state": self._kwargs.get("random_state", 42),
            "n_jobs": self._kwargs.get("n_jobs", -1),
        }

        best_score = -np.inf
        best_params = {}
        grid = self.TUNE_GRID_INTENSIVE if intensive else self.TUNE_GRID

        for combo in grid:
            try:
                cf = CausalForest(**fixed_params, **combo)
                cf.fit(X_np, T_np, Y_np)
                adapter = CausalForestAdapter.from_fitted(cf)
                score = float(rscorer.score(adapter))
                if score > best_score:
                    best_score = score
                    best_params = combo
            except Exception:
                continue

        if best_params:
            self._kwargs.update(best_params)
            self._cf = CausalForest(**fixed_params, **best_params)

        self._tune_result = {
            "best_params": best_params,
            "best_r_score": float(best_score) if best_params else None,
            "n_combos": len(grid),
        }
        return self._tune_result


@dataclass
class ModelContext:
    seed: int = 42
    base_learner_type: str = "lgbm"  # "lgbm" | "catboost"
    # Fixe Standardparameter für Base Learner aus der globalen Konfiguration.
    # Diese werden immer gesetzt und können durch getunte Parameter ergänzt/überschrieben werden.
    base_fixed_params: Dict[str, Any] = field(default_factory=dict)
    # Fixe Parameter für model_final (CATE-Effektmodell).
    # Separat von base_fixed_params, da BL-Classifier-Params für Regression ungeeignet sein können.
    fmt_fixed_params: Dict[str, Any] = field(default_factory=dict)

    # Getunte (oder modell-/rollen-spezifisch gesetzte) Parameter.
    # Struktur: role -> {param: value}
    tuned_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Anzahl Kerne für Base Learner (LightGBM n_jobs / CatBoost thread_count).
    # -1 = alle Kerne, 1 = ein Kern (minimaler RAM).
    # Wird vom parallel_level in der Config gesteuert.
    parallel_jobs: int = -1

    # DML-interne Cross-Fitting-Parameter
    dml_crossfit_folds: int = 5
    mc_iters: Optional[int] = None
    mc_agg: str = "mean"

    # RCT-Modus: Bei randomisierten Experimenten wird für Propensity-Rollen
    # (model_t, model_propensity) ein DummyClassifier(strategy="prior") verwendet,
    # der P(T|X) = empirische Treatment-Rate als Konstante vorhersagt.
    # BLT diagnostiziert vorab, dass Propensity-Skill ≈ 0 (bestätigt RCT).
    is_rct: bool = False

    def params_for(self, role: str) -> Dict[str, Any]:
        # Zuerst rollen-spezifisch, sonst 'default', sonst leer.
        return dict(self.tuned_params.get(role) or self.tuned_params.get("default") or {})


Factory = Callable[[ModelContext], Any]


class ModelRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, Factory] = {}

    def register(self, name: str, factory: Factory) -> None:
        self._factories[name] = factory

    def create(self, name: str, ctx: ModelContext) -> Any:
        if name not in self._factories:
            raise KeyError(f"Unbekanntes Modell '{name}'. Registriert: {sorted(self._factories)}")
        return self._factories[name](ctx)

    def list(self) -> List[str]:
        return sorted(self._factories.keys())


def default_registry() -> ModelRegistry:
    """Standard-Registry der verfügbaren kausalen Learner.

Alle Base Learner werden konsistent über `ctx.base_learner_type` und `ctx.tuned_params`
erzeugt. Bei `ctx.is_rct=True` werden Propensity-Rollen (model_t, model_propensity,
propensity_model) durch `DummyClassifier(strategy="prior")` ersetzt.
CausalForest-Parameter werden via `_sanitize_forest_params` validiert
(n_estimators muss durch subforest_size=4 teilbar sein)."""
    reg = ModelRegistry()

    def _base(ctx: ModelContext, role: str):
        # Wichtig: fixe Defaults aus der Konfiguration immer berücksichtigen.
        # Getunte Werte (rollen-spezifisch) überschreiben bei Schlüsselkonflikten.
        #
        # KRITISCH: model_final (CATE-Effektmodell) darf NICHT die "default"-Params
        # erben, die vom Base-Learner-Tuning stammen. Diese wurden für Nuisance-
        # Klassifikation (Y/T) optimiert und sind für CATE-Regression ungeeignet.
        # Typisches Symptom: min_child_samples=121 + num_leaves=13 kollabiert den
        # CATE-Baum zu einem Intercept → konstante Vorhersagen für alle Samples.
        # model_final nutzt stattdessen fmt_fixed_params (final_model_tuning.fixed_params)
        # als Basis, ergänzt durch FMT-getunte Parameter falls vorhanden.
        cate_model_roles = {"model_final"}
        if role in cate_model_roles:
            params = dict(ctx.fmt_fixed_params or {})
            explicit = ctx.tuned_params.get(role)
            if explicit:
                params.update(explicit)
        else:
            params = dict(ctx.base_fixed_params or {})
            params.update(ctx.params_for(role))

        # Task-Auswahl pro Rolle — Regressor vs. Classifier.
        #
        # REGRESSOR (predict() → E[Y|X] ∈ [0,1], kontinuierlich):
        #   model_final       — CATE-Effektmodell (DML, DRLearner)
        #   cate_models        — Pseudo-Outcome-Regression (XLearner)
        #   overall_model      — SLearner: EconML ruft predict() auf, NICHT predict_proba
        #   models             — TLearner/XLearner: EconML ruft predict() auf
        #
        # CLASSIFIER (EconML nutzt intern predict_proba):
        #   model_y            — NonParamDML: discrete_outcome=True → predict_proba
        #   model_t            — NonParamDML: discrete_treatment=True → predict_proba
        #   model_propensity   — DRLearner: Propensity-Score via predict_proba
        #   propensity_model   — XLearner: Propensity-Score via predict_proba
        #   model_regression   — DRLearner: discrete_outcome=True → Classifier (predict_proba)
        #                        BLT und Training nutzen beide Logloss für konsistente Hyperparameter.
        #
        regressor_roles = {
            "model_final", "cate_models",
            "overall_model", "models",       # Meta-Learner Outcome-Modelle
        }
        task = "regressor" if role in regressor_roles else "classifier"

        # RCT-Modus: Propensity-Rollen erhalten DummyClassifier(strategy="prior"),
        # der P(T|X) = empirische Treatment-Rate als Konstante vorhersagt.
        # BLT diagnostiziert vorab, ob Propensity-Skill ≈ 0 (bestätigt RCT).
        # Ohne dies overfittet das Propensity-Modell auf Rauschen → verzerrte DML-Residuen.
        _propensity_roles = {"model_t", "model_propensity", "propensity_model"}
        if ctx.is_rct and role in _propensity_roles:
            from sklearn.dummy import DummyClassifier
            return DummyClassifier(strategy="prior")

        return build_base_learner(ctx.base_learner_type, params, seed=ctx.seed, task=task, parallel_jobs=ctx.parallel_jobs)

    # DML family
    reg.register(
        "NonParamDML",
        lambda ctx: NonParamDML(
            model_y=_base(ctx, "model_y"),
            model_t=_base(ctx, "model_t"),
            # Das Final-Modell ist frei wählbar und wird optional über R-Score getunt.
            model_final=_base(ctx, "model_final"),
            discrete_treatment=True,
            discrete_outcome=True,
            cv=StratifiedKFold(n_splits=ctx.dml_crossfit_folds, shuffle=True, random_state=ctx.seed),
            mc_iters=ctx.mc_iters,
            mc_agg=ctx.mc_agg,
            random_state=ctx.seed,
            allow_missing=True,
        ),
    )
    # ParamDML nutzt EconMLs LinearDML, d. h. das Final-Modell nimmt eine lineare
    # CATE-Struktur an. Für nichtlineare parametrische CATE-Schätzung eignet sich
    # NonParamDML besser.
    reg.register(
        "ParamDML",
        lambda ctx: LinearDML(
            model_y=_base(ctx, "model_y"),
            model_t=_base(ctx, "model_t"),
            discrete_treatment=True,
            discrete_outcome=True,
            cv=StratifiedKFold(n_splits=ctx.dml_crossfit_folds, shuffle=True, random_state=ctx.seed),
            mc_iters=ctx.mc_iters,
            mc_agg=ctx.mc_agg,
            random_state=ctx.seed,
            allow_missing=True,  # Nur W — LinearDML final model kann kein NaN in X
        ),
    )

    # CausalForestDML kombiniert DML-Residualisierung (mit Nuisance-Modellen für Outcome und
    # Treatment) mit einem Causal Forest als letzter Stufe. Daher werden auch hier Base Learner
    # (model_y, model_t) verwendet. Wald-Parameter orientieren sich an EconML-Defaults.
    # CFT (CausalForest-Tuning via Optuna) kann max_depth, min_weight_fraction_leaf,
    # min_var_fraction_leaf und criterion weiter optimieren.
    _CFDML_DEFAULTS = {
        "n_estimators": 500,                  # Production Default (Tuning nutzt 100)
        "criterion": "mse",                   # EconML Default
        "max_depth": None,                    # EconML Default: unbegrenzt — CFT kann optimieren
        "min_samples_leaf": 5,                # EconML Default
        "min_samples_split": 10,              # EconML Default
        "min_weight_fraction_leaf": 0.0,      # EconML Default — CFT kann optimieren
        "max_features": "auto",               # EconML Default: 'auto' = n_features (alle Features pro Split)
        "max_samples": 0.45,                  # EconML Default
        "min_var_fraction_leaf": None,         # EconML Default: None — CFT kann optimieren
        "min_impurity_decrease": 0.0,          # EconML Default
    }
    def _sanitize_forest_params(defaults: dict, tuned: dict) -> dict:
        """Merged Forest-Params mit Validierung.

        Entfernt Keys die im Konstruktor explizit gesetzt werden (model_y, model_t,
        discrete_treatment, discrete_outcome, cv, random_state, n_jobs, inference,
        mc_iters, mc_agg) → verhindert 'got multiple values for keyword argument'.
        n_estimators muss durch subforest_size=4 teilbar sein.
        """
        merged = {**defaults, **tuned}
        # Keys die im CausalForestDML/CausalForestAdapter-Konstruktor
        # explizit gesetzt werden, dürfen nicht im **kwargs-Unpack sein.
        for _ek in ("model_y", "model_t", "discrete_treatment", "discrete_outcome",
                     "cv", "random_state", "n_jobs", "inference", "mc_iters", "mc_agg"):
            merged.pop(_ek, None)
        if "n_estimators" in merged:
            ne = int(merged["n_estimators"])
            if ne % 4 != 0:
                merged["n_estimators"] = max(4, (ne // 4) * 4)
        return merged

    reg.register(
        "CausalForestDML",
        lambda ctx: CausalForestDML(
            model_y=_base(ctx, "model_y"),
            model_t=_base(ctx, "model_t"),
            discrete_treatment=True,
            discrete_outcome=True,
            cv=StratifiedKFold(n_splits=ctx.dml_crossfit_folds, shuffle=True, random_state=ctx.seed),
            mc_iters=ctx.mc_iters,
            mc_agg=ctx.mc_agg,
            random_state=ctx.seed,
            # KEIN allow_missing: GRF final model kann kein NaN in X (Split-Features),
            # und rubin nutzt kein W. allow_missing=True würde NaN in X durchlassen
            # und zu kryptischem GRF-Crash führen statt klarem Validierungsfehler.
            **_sanitize_forest_params(_CFDML_DEFAULTS, ctx.params_for("forest")),
        ),
    )

    # DRLearner
    reg.register(
        "DRLearner",
        lambda ctx: DRLearner(
            model_propensity=_base(ctx, "model_propensity"),
            model_regression=_base(ctx, "model_regression"),
            # Final-Modell für die CATE-Schätzung (Regression der DR-Pseudo-Outcomes auf X).
            # Kann optional über R-Score getunt werden.
            model_final=_base(ctx, "model_final"),
            discrete_outcome=True,
            cv=StratifiedKFold(n_splits=ctx.dml_crossfit_folds, shuffle=True, random_state=ctx.seed),
            mc_iters=ctx.mc_iters,
            mc_agg=ctx.mc_agg,
            random_state=ctx.seed,
            allow_missing=True,
        ),
    )

    # Meta-learners
    reg.register(
        "XLearner",
        lambda ctx: XLearner(
            models=_base(ctx, "models"),
            cate_models=_base(ctx, "cate_models"),
            propensity_model=_base(ctx, "propensity_model"),
            allow_missing=True,
        ),
    )
    reg.register("TLearner", lambda ctx: TLearner(models=_base(ctx, "models"), allow_missing=True))
    reg.register("SLearner", lambda ctx: SLearner(overall_model=_base(ctx, "overall_model"), allow_missing=True))

    # GRF (reiner Causal Forest ohne DML-Residualisierung)
    # Kein Base Learner nötig — der Wald schätzt Treatment-Effekte direkt.
    # Defaults orientieren sich an EconML (identisch mit _CF_FIXED_DEFAULTS in tuning_optuna.py).
    # CFT kann max_depth, min_weight_fraction_leaf, min_var_fraction_leaf und criterion optimieren.
    _CF_DEFAULTS = {
        "n_estimators": 500,                  # Production Default (Tuning nutzt 100)
        "criterion": "mse",                   # EconML Default
        "max_depth": None,                    # EconML Default: unbegrenzt
        "min_samples_leaf": 5,                # EconML Default
        "min_samples_split": 10,              # EconML Default
        "min_weight_fraction_leaf": 0.0,      # EconML Default — CFT kann optimieren
        "max_features": "auto",               # EconML Default: 'auto' = n_features
        "max_samples": 0.45,                  # EconML Default
        "min_var_fraction_leaf": None,         # EconML Default: None — CFT kann optimieren
        "min_impurity_decrease": 0.0,          # EconML Default
    }
    reg.register(
        "CausalForest",
        lambda ctx: CausalForestAdapter(
            random_state=ctx.seed,
            n_jobs=ctx.parallel_jobs,
            **_sanitize_forest_params(_CF_DEFAULTS, ctx.params_for("grf")),
        ),
    )

    return reg
