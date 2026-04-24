from __future__ import annotations

"""Registry für kausale Learner und Base Learner.
Die Registry kapselt,
- welche kausalen Modelle trainiert werden können,
- wie ein Modell instanziiert wird,
- wie Base Learner (LightGBM/CatBoost) konsistent gebaut werden.
Die Konfiguration bleibt damit schlank; die Factory kümmert sich um die
konkrete Modellinstanziierung."""


from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional

from econml.dml import CausalForestDML, NonParamDML, LinearDML
from econml.dr import DRLearner
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

    def tune(self, X, T, Y, intensive=False, rscorer=None, model_y=None, model_t=None):
        """Grid-Search über CausalForest-Waldparameter mit RScorer-Evaluation.

        Verwendet EconML's RScorer (Nie & Wager, 2021) zur Bewertung jeder
        Parameterkombination. RScorer fittet eigene Nuisance-Modelle (model_y,
        model_t) und berechnet den normalisierten R-Score:

            R-Score = 1 - E[(Y_res - τ(X)·T_res)²] / base_loss

        Höher = besser. Misst, wie viel zusätzliche Varianz das CATE-Modell
        gegenüber einem konstanten Effekt erklärt. Identisch mit der Metrik,
        die CausalForestDML.tune() intern verwendet.

        Args:
            rscorer: Vorgefitteter RScorer. Wenn None, wird einer erstellt.
            model_y: Nuisance-Modell für E[Y|X]. Wenn None, RandomForest-Fallback.
            model_t: Nuisance-Modell für E[T|X]. Wenn None, RandomForest-Fallback.
                     Idealerweise dieselben Base-Learner wie NonParamDML/CausalForestDML.
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
                from sklearn.ensemble import RandomForestRegressor
                model_y = RandomForestRegressor(n_estimators=100, min_samples_leaf=20, n_jobs=-1)
            if model_t is None:
                from sklearn.ensemble import RandomForestClassifier
                model_t = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, n_jobs=-1)
            rscorer = RScorer(
                model_y=model_y, model_t=model_t,
                discrete_treatment=True, discrete_outcome=True,
                cv=3, random_state=self._kwargs.get("random_state", 42),
            )
            rscorer.fit(Y_np, T_np, X=X_np)

        fixed_params = {
            "n_estimators": self._kwargs.get("n_estimators", 200),
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
                # Adapter-Wrapper für RScorer (braucht .effect()-Methode)
                adapter = CausalForestAdapter.__new__(CausalForestAdapter)
                adapter._cf = cf
                adapter._kwargs = {}
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
erzeugt."""
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
        #   model_regression   — DRLearner: E[Y|X,T]-Modell. DRLearner hat KEIN
        #                        discrete_outcome → ruft predict() direkt auf.
        #                        Classifier → predict()={0,1} → DR-Pseudo-Outcomes kaputt!
        #                        Regressor → predict()=E[Y|X] → korrekt.
        #
        # CLASSIFIER (EconML nutzt intern predict_proba):
        #   model_y            — NonParamDML: discrete_outcome=True → predict_proba
        #   model_t            — NonParamDML: discrete_treatment=True → predict_proba
        #   model_propensity   — DRLearner: Propensity-Score via predict_proba
        #   propensity_model   — XLearner: Propensity-Score via predict_proba
        #
        regressor_roles = {
            "model_final", "cate_models",
            "overall_model", "models",       # Meta-Learner Outcome-Modelle
            "model_regression",              # DRLearner Outcome-Modell
        }
        task = "regressor" if role in regressor_roles else "classifier"
        return build_base_learner(ctx.base_learner_type, params, seed=ctx.seed, task=task, parallel_jobs=ctx.parallel_jobs)

    # DML family
    reg.register(
        "NonParamDML",
        lambda ctx: NonParamDML(
            model_y=_base(ctx, "model_y"),
            model_t=_base(ctx, "model_t"),
            # Das Final-Modell ist frei wählbar und wird optional über R-Loss/R-Score getunt.
            model_final=_base(ctx, "model_final"),
            discrete_treatment=True,
            discrete_outcome=True,
            cv=ctx.dml_crossfit_folds,
            mc_iters=ctx.mc_iters,
            mc_agg=ctx.mc_agg,
            random_state=ctx.seed,
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
            cv=ctx.dml_crossfit_folds,
            mc_iters=ctx.mc_iters,
            mc_agg=ctx.mc_agg,
            random_state=ctx.seed,
        ),
    )

    # CausalForestDML kombiniert DML-Residualisierung (mit Nuisance-Modellen für Outcome und
    # Treatment) mit einem Causal Forest als letzter Stufe. Daher werden auch hier Base Learner
    # (model_y, model_t) verwendet. Zusätzlich können Wald-Parameter (z. B. n_estimators,
    # max_depth, honest, subsample_fr) gesetzt werden.
    reg.register(
        "CausalForestDML",
        lambda ctx: CausalForestDML(
            model_y=_base(ctx, "model_y"),
            model_t=_base(ctx, "model_t"),
            discrete_treatment=True,
            discrete_outcome=True,
            cv=ctx.dml_crossfit_folds,
            mc_iters=ctx.mc_iters,
            mc_agg=ctx.mc_agg,
            n_estimators=ctx.params_for("forest").get("n_estimators", 200),
            random_state=ctx.seed,
            **{k: v for k, v in ctx.params_for("forest").items() if k != "n_estimators"},
        ),
    )

    # DRLearner
    reg.register(
        "DRLearner",
        lambda ctx: DRLearner(
            model_propensity=_base(ctx, "model_propensity"),
            model_regression=_base(ctx, "model_regression"),
            # Final-Modell für die CATE-Schätzung (Regression der DR-Pseudo-Outcomes auf X).
            # Kann optional über R-Loss/R-Score getunt werden.
            model_final=_base(ctx, "model_final"),
            cv=ctx.dml_crossfit_folds,
            mc_iters=ctx.mc_iters,
            mc_agg=ctx.mc_agg,
            random_state=ctx.seed,
        ),
    )

    # Meta-learners
    reg.register(
        "XLearner",
        lambda ctx: XLearner(
            models=_base(ctx, "models"),
            cate_models=_base(ctx, "cate_models"),
            propensity_model=_base(ctx, "propensity_model"),
        ),
    )
    reg.register("TLearner", lambda ctx: TLearner(models=_base(ctx, "models")))
    reg.register("SLearner", lambda ctx: SLearner(overall_model=_base(ctx, "overall_model")))

    # GRF (reiner Causal Forest ohne DML-Residualisierung)
    # Kein Base Learner nötig — der Wald schätzt Treatment-Effekte direkt.
    # Parameter werden über ctx.params_for("forest") oder ctx.tuned_params.get("grf") gesetzt.
    reg.register(
        "CausalForest",
        lambda ctx: CausalForestAdapter(
            n_estimators=ctx.params_for("grf").get("n_estimators", 200),
            criterion=ctx.params_for("grf").get("criterion", "mse"),
            min_samples_leaf=ctx.params_for("grf").get("min_samples_leaf", 20),
            max_depth=ctx.params_for("grf").get("max_depth", None),
            random_state=ctx.seed,
            n_jobs=ctx.parallel_jobs,
            **{k: v for k, v in ctx.params_for("grf").items()
               if k not in ("n_estimators", "min_samples_leaf", "max_depth", "n_jobs", "random_state", "criterion")},
        ),
    )

    return reg
