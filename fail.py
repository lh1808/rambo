09:33:22 INFO [rubin.analysis] Champion: CausalForest (uplift_at_10pct=0.00309095)
09:33:22 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
09:33:23 INFO [rubin.analysis] Trainiere Surrogate auf Champion CausalForest (volle Evaluation inkl. DRTester).
09:33:23 INFO [rubin.analysis] Surrogate K-Fold CV: Fold-Aligned Predictions (5 Folds, komplett leakage-frei). Jeder Surrogate-Fold nutzt Champion-Predictions von einem Modell, das den Val-Fold nie gesehen hat.
09:33:25 INFO [rubin.analysis] Surrogate Final-Fit: Trainiert auf Full-Data-Refit-Predictions (Produktion).
09:33:25 INFO [rubin.analysis] Surrogate-Evaluation: {'qini': 5.915958127850194e-05, 'auuc': 8.81185304103223e-05, 'uplift_at_10pct': 0.002226409087563594, 'uplift_at_20pct': 0.0014928853252877588, 'uplift_at_50pct': 0.0012491272615197564, 'policy_value': 0.0009166673740861522}
09:33:56 INFO [rubin.analysis] DRTester-Plots für SurrogateTree erzeugt.
09:33:56 INFO [rubin.analysis] Bundle-Preprocessor: DataPrep-FittedPreprocessor übernommen (rohdaten-fähig: Encoding-Maps + NA-Fill für 77 Features).
09:54:27 WARNING [rubin.analysis] Surrogate-Tree Bundle-Export fehlgeschlagen.
Traceback (most recent call last):
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 2619, in _run_bundle_export
    depth, n_leaves = self._log_surrogate_tree_info(log_tree, _surr_bt) if log_tree else (None, None)
    ^^^^^^^^^^^^^^^
TypeError: cannot unpack non-iterable NoneType object
09:54:28 INFO [rubin.analysis] Bundle gespeichert: runs/bundles/bundle_20260604_093356 (6 Modelle, Champion=CausalForest)
09:54:28 INFO [rubin.analysis] Explainability für Champion 'CausalForest': CV-Fold-Modell (bereits trainiert), 83360 out-of-fold Samples verfügbar.
09:54:28 INFO [rubin.analysis] Explainability: 10000 Samples für SHAP.
09:54:28 INFO [rubin.analysis] SHAP-Plot-Satz fehlgeschlagen, Fallback auf generische SHAP-Werte.
Traceback (most recent call last):
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 2807, in _run_explainability
    shap_result = build_shap_plots(
                  ^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/explainability/shap_uplift.py", line 239, in build_shap_plots
    raise TypeError(
TypeError: Das Modell stellt keine Methode 'shap_values' bereit. Für diesen Plot-Satz wird ein EconML-kompatibles Modell benötigt.
