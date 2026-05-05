# rubin_overview.html — Senior Review

**Datum:** 2026-05-05
**Ergebnis:** 11 Issues gefunden und gefixt, 0 verbleibend

---

## Pruefplan & Ergebnisse

### 1. Ueberblick (Sektion 1)
- [x] Pipeline-Beschreibung korrekt (11 Schritte)
- [x] Modell-Auflistung vollstaendig (8 Modelle)

### 2. Architektur & Ablauf (Sektion 2)
- [x] Pipeline-Schritte stimmen mit Code ueberein

### 3. Konfiguration (Sektion 5)
- [x] 5.1 constants: parallel_level Default 2->3 GEFIXT
- [x] 5.6 feature_selection: top_pct entfernt GEFIXT
- [x] 5.6 feature_selection: max_features 50->100 GEFIXT
- [x] 5.7 models: ensemble Default false->true GEFIXT
- [x] 5.8 tuning: per_role/per_learner entfernt GEFIXT (2 Tabellenzeilen + 2 Erklaerungen)
- [x] 5.8 tuning: DART->GBDT GEFIXT
- [x] 5.8 tuning: Metriken-Alternativen (pr_auc, roc_auc, accuracy) entfernt GEFIXT
- [x] 5.8 tuning: "tuning.metric" Referenz entfernt (Metrik ist jetzt fest)

### 4. Kausale Modelle (Sektion 6)
- [x] Alle 8 Modelle korrekt beschrieben
- [x] Rollen-Zuweisungen stimmen mit model_registry.py ueberein

### 5. Tuning (Sektion 7)
- [x] BLT-Tabelle: "log_loss/roc_auc" -> "log_loss" GEFIXT
- [x] FMT-Beschreibung korrekt (R-Loss, DR-MSE)
- [x] CFT-Beschreibung korrekt (R-Loss, Wald-Parameter)

### 6. Evaluation (Sektion 10)
- [x] Diagnose-Plots korrekt aufgelistet
- [x] DRTester-Beschreibung aktuell
- [x] Metriken: Qini, AUUC, Policy Value, Uplift@k

### 7. Multi-Treatment (Sektion 13)
- [x] BT vs MT Vergleichstabelle: Propensity-Tuning "Binaere AUC"->"log_loss" GEFIXT

### 8. Web UI (Sektion 20)
- [x] Eingebettete App-Vorschau synchronisiert mit aktuellem JSX-Build

---

## Fixes (11 Stellen)

| # | Sektion | Vorher | Nachher |
|---|---------|--------|---------|
| 1 | 5.1 | parallel_level Default: 2 | 3 |
| 2 | 5.6 | top_pct Tabellenzeile | entfernt |
| 3 | 5.6 | top_pct: 15.0 Empfehlung | max_features: 100 |
| 4 | 5.6 | max_features: 50 | max_features: 60 (bei >200 Features) |
| 5 | 5.7 | ensemble: false | ensemble: true |
| 6 | 5.8 | per_learner + per_role (2 Zeilen) | entfernt |
| 7 | 5.8 | per_learner/per_role Erklaerung | automatisches Sharing |
| 8 | 5.8 | per_learner/per_role Empfehlung | entfernt |
| 9 | 5.8 | DART Beschreibung | GBDT Beschreibung |
| 10 | 5.8 | Metriken: pr_auc, roc_auc, accuracy | fest: log_loss + MSE |
| 11 | 7 | BLT: log_loss/roc_auc | log_loss |
| 12 | 13 | Propensity: Binaere AUC / roc_auc_ovr | log_loss |
