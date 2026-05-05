# HTML-Report (html_report.py) — Senior Review

**Datum:** 2026-05-05
**Ergebnis:** 4 Issues gefunden und gefixt, 0 verbleibend

---

## Sektionen geprueft

### 1. Overview (_sec "overview")
- [x] Modell-Liste: alle 8 Modelle korrekt angezeigt
- [x] Pipeline-Schritte korrekt
- [x] Kein DART, per_role, per_learner, top_pct

### 2. Heterogeneity (_sec "heterogeneity")
- [x] CATE-Verteilungsplots korrekt eingebettet

### 3. Data (_sec "data")
- [x] Trainingsdaten-Statistiken angezeigt
- [x] Treatment-Verteilung mit Outcome-Rate
- [x] Evaluationsdaten (bei externem Modus)

### 4. DataPrep (_sec "dataprep")
- [x] Verarbeitungsschritte angezeigt

### 5. Feature Selection (_sec "feature_sel")
- [x] Keine top_pct Referenz

### 6. Tuning (_sec "tuning")
- [x] Task-Sharing Tabelle korrekt (ohne per_role/per_learner)
- [x] Best Scores pro Task
- [x] Both-Mode (CatBoost + LGBM) korrekt behandelt
- [x] Metriken jetzt fest: "log_loss (fest)" + "neg_mse (fest)" GEFIXT

### 7. FMT (_sec "fmt")
- [x] FMT-Plan pro Modell angezeigt
- [x] Best FMT-Scores angezeigt
- [x] R-Loss / DR-MSE korrekt

### 8. Comparison (_sec "comparison")
- [x] Ranking nach Champion-Metrik
- [x] Champion hervorgehoben
- [x] Diagnose-Plots eingebettet

### 9. Surrogate (_sec "surrogate")
- [x] Surrogate vs. Champion Vergleichstabelle
- [x] Retention-Metrik

### 10. Explainability (_sec "explainability")
- [x] SHAP-Plots eingebettet

### 11. Timing (_sec "timing")
- [x] Laufzeiten pro Schritt

---

## Pipeline-Integration
- [x] add_best_params() - BLT-Ergebnisse
- [x] add_fmt_best_params() - FMT-Ergebnisse
- [x] add_fmt_info() - FMT-Konfiguration
- [x] add_fmt_plan() - FMT-Plan
- [x] add_plot() - Diagnose-Plots
- [x] add_tuning_plan() - BLT-Plan

## Fixes

| # | Stelle | Vorher | Nachher |
|---|--------|--------|---------|
| 1 | Z.118 | getattr(cfg.tuning, "metric", "log_loss") | "log_loss" (fest) |
| 2 | Z.119 | getattr(cfg.tuning, "metric_regression", "neg_mse") | "neg_mse" (fest) |
| 3 | Z.726 | cs.get("tuning_metric","log_loss") | "log_loss (fest)" |
| 4 | Z.727 | cs.get("tuning_metric_regression","neg_mse") | "neg_mse (fest)" |
