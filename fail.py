Modell
Aufschlüsselung
Fits
CausalForestDML
10 Nuisance + 48 Forest-Grid
58
Nuisance: cv=5 × 2 Modelle = 10. Dann 48 Forest-Kombis (intensiv) auf den Residuals.
CausalForest
48 Forest-Grid (kein Nuisance)
48
R-Loss Grid-Search auf erstem CV-Fold (48 Kombis inkl. criterion).
Intensives Grid (48 statt 12 Kombis)
Normal: 12 Kombis (min_samples_leaf × max_depth × max_samples). Intensiv: 48 Kombis (+ criterion, mehr max_depth-Stufen).
Search-Grid
Parameter
Werte
min_samples_leaf
5, 10, 20
max_depth
None, 10, 20, 30
max_samples
0.3, 0.5
criterion
mse, het
Nicht-getunte Parameter verwenden EconML-Defaults (n_estimators=200, honest=true). CausalForestDML optimiert zusätzlich min_weight_fraction_leaf und min_var_fraction_leaf.
