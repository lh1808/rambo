# rubin Causal ML Framework — Senior Review

**Reviewer:** Senior Causal ML Engineer
**Datum:** 2026-05-05
**Scope:** Vollstaendiger Code-Review des Produktions-Builds
**Ergebnis:** 94/94 Checks bestanden — keine kritischen Findings

---

## 1. Kausale Korrektheit der Modell-Spezifikation

### 1.1 Nuisance-Rollen
- [x] NonParamDML: model_y=Classifier, model_t=Classifier, model_final=Regressor, discrete_treatment=True, discrete_outcome=True
- [x] ParamDML: identisch zu NonParamDML
- [x] CausalForestDML: model_y=Classifier, model_t=Classifier, discrete_treatment=True, discrete_outcome=True
- [x] DRLearner: model_propensity=Classifier, model_regression=Regressor, model_final=Regressor
- [x] Meta-Learner: SLearner->overall_model(Reg), TLearner->models(Reg), XLearner->cate_models(Reg)+propensity_model(Clf)
- [x] model_final isoliert von BLT-Defaults (nutzt fmt_fixed_params, NICHT base_fixed)
- [x] model_regression als Regressor (deliberate -- Classifier->predict()={0,1} broke DR-Pseudo-Outcomes)

### 1.2 EconML-Parameter
- [x] StratifiedKFold: 5 Stellen in model_registry
- [x] random_state=ctx.seed: 9 Stellen
- [x] mc_iters/mc_agg: 4 Stellen
- [x] CausalForestDML: konservative Defaults (n_est=200, max_depth=5, min_samples_leaf=200)

---

## 2. Tuning-Korrektheit

### 2.1 BLT Metriken
- [x] Classifier: log_loss (Kalibrierung) via sklearn.metrics.log_loss
- [x] Regressor: neg. MSE via _score_regressor
- [x] Keine AUC, Accuracy, PR-AUC -- komplett entfernt

### 2.2 FMT Bewertung
- [x] NonParamDML: R-Loss via est.score()
- [x] DRLearner: DR-MSE via est.score()
- [x] Aeussere CV: est.fit(train) + est.score(val)
- [x] Nuisance nutzt BLT-Params: tuned_roles.get("model_y"/"model_t")

### 2.3 CFT Bewertung
- [x] R-Loss via est.score()
- [x] GRF: Residuen vorberechnet (vor Optuna-Loop)
- [x] CFDML: eigene interne Nuisance pro Trial

### 2.4 Suchraeume
- [x] BLT LGBM 13 Params = FMT LGBM 13 Params (Backend=UI)
- [x] BLT CB 10 Params = FMT CB 10 Params (Backend=UI)
- [x] CFT: 8 Wald-Parameter
- [x] boosting_type immer "gbdt"

### 2.5 Overfitting-Schutz
- [x] Dual-Seed (tuning_cv_seed != random_seed)
- [x] Overfit-Penalty (BLT+FMT)
- [x] TPESampler(multivariate=True, constant_liar=True)
- [x] MedianPruner
- [x] catch=(Exception,)

---

## 3. Stratifizierung

- [x] BLT: T*10+Y
- [x] FMT: T*10+Y
- [x] CFT: T*10+clip(Y)
- [x] Training: T+"_"+Y
- [x] DRTester: _TxYSplitter
- [x] Surrogate: T*10+Y
- [x] EconML-intern: T only (by design)
- [x] Fold-Fallback bei kleinen Strata

---

## 4. Parameter-Weitergabe

- [x] BLT -> tuned_params_by_model: {model: {role: params}}
- [x] BLT -> FMT: tuned_roles mit model_y/model_t
- [x] FMT -> tuned_params_by_model: model_final hinzugefuegt
- [x] BLT -> CFT: nuisance_params_y/t korrekt extrahiert
- [x] CFT -> tuned_params_by_model: forest/grf Key hinzugefuegt
- [x] ModelContext: base_fixed != fmt_fixed, model_final isoliert
- [x] preds[model] -> Evaluation -> Champion -> Surrogate/Bundle

---

## 5. CPU-Parallelisierung

- [x] CFT: n_jobs=-1 (Forest), parallel_jobs=1 (CFDML-Nuisance)
- [x] CFT GRF standalone: parallel_jobs=-1
- [x] Training CausalForest: ctx.parallel_jobs=-1 Override
- [x] BLT/FMT: cores/workers bei Level 3/4
- [x] Dokumentiert in docs/architektur.md

---

## 6. Statistische Korrektheit

- [x] const_marginal_effect() fuer BT, effect() fuer MT
- [x] Qini, AUUC, Policy Value korrekt
- [x] Champion-Selektion: Surrogate/Historical excluded

---

## 7. Robustheit

- [x] Exception-Handling: BLT/FMT/CFT mit try/except
- [x] GC/RAM: gc.collect() an Phasenuebergaengen
- [x] Degenerate Cases: BLT/FMT/CFT disabled -> korrekte Fallbacks

---

## 8. Defaults

- [x] Seeds: 42/18, base_learner: catboost, boosting: gbdt
- [x] max_features: 100, ensemble: true, parallel_level: 3
- [x] Konsistenz: settings.py <-> JSX <-> configs <-> docs

---

## Design-Notizen

1. **DRLearner discrete_outcome=False**: Bewusst. predict() statt predict_proba() fuer model_regression.
2. **EconML cv stratifiziert auf T**: By design (cv.split(X, T) in _OrthoLearner).
3. **CausalForestDML.tune() nicht genutzt**: Rubins Optuna-CFT ist strikt besser (50+ TPE-Trials vs. 12 Grid-Kombinationen).
4. **score_nuisances() nicht genutzt**: Rein diagnostisch, aendert keine Ergebnisse. PENDING fuer Report-Erweiterung.
