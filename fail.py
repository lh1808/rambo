06:01:28 INFO [rubin.analysis] Metriken für 5 Modelle berechnet. Vorläufiger Champion: Ensemble. Diagnostik-Plots: Champion + Challenger
06:01:52 WARNING [rubin.evaluation.drtester_plots] Native Uplift-by-Percentile fehlgeschlagen: name 'y_s' is not defined
06:02:18 WARNING [rubin.evaluation.drtester_plots] Native Uplift-by-Percentile fehlgeschlagen: name 'y_s' is not defined
06:02:20 WARNING [rubin.evaluation.drtester_plots] Native Uplift-by-Percentile fehlgeschlagen: name 'y_s' is not defined
06:02:22 WARNING [rubin.evaluation.drtester_plots] Native Uplift-by-Percentile fehlgeschlagen: name 'y_s' is not defined
06:02:23 WARNING [rubin.evaluation.drtester_plots] Native Uplift-by-Percentile fehlgeschlagen: name 'y_s' is not defined
06:02:25 WARNING [rubin.evaluation.drtester_plots] Native Uplift-by-Percentile fehlgeschlagen: name 'y_s' is not defined
06:02:26 WARNING [rubin.evaluation.drtester_plots] Native Uplift-by-Percentile fehlgeschlagen: name 'y_s' is not defined
06:02:52 WARNING [rubin.evaluation.drtester_plots] Native Uplift-by-Percentile fehlgeschlagen: name 'y_s' is not defined
06:03:03 INFO [rubin.analysis] ────────────────────────────────────────────────────────────
06:03:03 INFO [rubin.analysis] Evaluation (qini):
06:03:03 INFO [rubin.analysis]   Ensemble               qini=0.000489261  auuc=0.0018884 | policy_value=0.0027802
06:03:03 INFO [rubin.analysis]   CausalForestDML        qini=0.00047994  auuc=0.0018791 | policy_value=0.0027983
06:03:03 INFO [rubin.analysis]   DRLearner              qini=0.000479887  auuc=0.001879 | policy_value=0.0027621
06:03:03 INFO [rubin.analysis]   NonParamDML            qini=0.00047044  auuc=0.0018696 | policy_value=0.0027521
06:03:03 INFO [rubin.analysis]   CausalForest           qini=0.000463931  auuc=0.0018631 | policy_value=0.0027983
06:03:03 INFO [rubin.analysis]   historical_score       qini=0.000200301  auuc=0.0015994 | policy_value=0.0027983
06:03:03 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
06:03:03 INFO [rubin.analysis] Champion: Ensemble (qini=0.000489261)
06:03:03 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
06:03:03 INFO [rubin.analysis] Trainiere Surrogate auf Champion Ensemble.
06:03:03 INFO [rubin.analysis] Surrogate: Train-Predictions nicht verfügbar, verwende OOF-Predictions.
06:03:06 INFO [rubin.analysis] Surrogate-Evaluation: {'qini': 0.000537449516717225, 'auuc': 0.0019365962665607748, 'uplift_at_10pct': 0.0007451712886229396, 'uplift_at_20pct': 0.0013140438433389403, 'uplift_at_50pct': 0.0021276849363798624, 'policy_value': 0.002798293499701821}
06:03:07 WARNING [rubin.evaluation.drtester_plots] Native Uplift-by-Percentile fehlgeschlagen: name 'y_s' is not defined
06:06:40 INFO [rubin.analysis] Ensemble-Refit: CausalForestDML auf vollen Daten refittet (435980 Zeilen).
06:07:45 INFO [rubin.analysis] Ensemble-Refit: CausalForest auf vollen Daten refittet (435980 Zeilen).
06:11:02 INFO [rubin.analysis] Ensemble-Refit: NonParamDML auf vollen Daten refittet (435980 Zeilen).
06:14:04 INFO [rubin.analysis] Ensemble-Refit: DRLearner auf vollen Daten refittet (435980 Zeilen).
06:14:04 INFO [rubin.analysis] Ensemble-Champion: Neues EnsembleCateEstimator mit 4 refitteten Modellen erstellt.
06:14:14 INFO [rubin.analysis] Surrogate-Einzelbaum exportiert (Typ=catboost, Tiefe=6, Blätter=64, trainiert auf 435980 Zeilen).
06:14:14 INFO [rubin.analysis] Bundle gespeichert: runs/bundles/bundle_20260506_060308 (5 Modelle, Champion=Ensemble)
06:14:14 INFO [rubin.analysis] Explainability: Ensemble ist Champion → verwende bestes Einzelmodell 'CausalForestDML' (Metrik qini = 0.00047994) für SHAP-Analyse.
06:14:14 INFO [rubin.analysis] Explainability für Champion 'CausalForestDML': CV-Fold-Modell (bereits trainiert), 87196 out-of-fold Samples verfügbar.
06:14:14 INFO [rubin.analysis] Explainability: 10000 Samples für SHAP.

 17%|===                 | 1723/10000 [00:11<00:52]
 19%|====                | 1880/10000 [00:12<00:51]
 20%|====                | 2041/10000 [00:13<00:50]
 22%|====                | 2197/10000 [00:14<00:49]
 24%|=====               | 2356/10000 [00:15<00:48]
 25%|=====               | 2515/10000 [00:16<00:47]
 27%|=====               | 2670/10000 [00:17<00:46]
 28%|======              | 2831/10000 [00:18<00:45]
 30%|======              | 2991/10000 [00:19<00:44]
 32%|======              | 3152/10000 [00:20<00:43]
 33%|=======             | 3312/10000 [00:21<00:42]
 35%|=======             | 3475/10000 [00:22<00:41]
 36%|=======             | 3637/10000 [00:23<00:40]
 38%|========            | 3797/10000 [00:24<00:39]
 40%|========            | 3959/10000 [00:25<00:38]
 41%|========            | 4119/10000 [00:26<00:37]
 43%|=========           | 4278/10000 [00:27<00:36]
 44%|=========           | 4436/10000 [00:28<00:35]
 46%|=========           | 4597/10000 [00:29<00:34]
 48%|==========          | 4760/10000 [00:30<00:33]
 49%|==========          | 4921/10000 [00:31<00:31]
 51%|==========          | 5086/10000 [00:32<00:30]
 53%|===========         | 5251/10000 [00:33<00:29]
 54%|===========         | 5409/10000 [00:34<00:28]
 56%|===========         | 5563/10000 [00:35<00:27]
 57%|===========         | 5719/10000 [00:36<00:26]
 59%|============        | 5880/10000 [00:37<00:25]
 60%|============        | 6041/10000 [00:38<00:24]
 62%|============        | 6196/10000 [00:39<00:23]
 64%|=============       | 6360/10000 [00:40<00:22]
 65%|=============       | 6523/10000 [00:41<00:21]
 67%|=============       | 6688/10000 [00:42<00:20]
 68%|==============      | 6848/10000 [00:43<00:19]
 70%|==============      | 7009/10000 [00:44<00:18]
 72%|==============      | 7169/10000 [00:45<00:17]
 73%|===============     | 7332/10000 [00:46<00:16]
 75%|===============     | 7487/10000 [00:47<00:15]
 76%|===============     | 7647/10000 [00:48<00:14]
 78%|================    | 7807/10000 [00:49<00:13]
 80%|================    | 7968/10000 [00:50<00:12]
 81%|================    | 8131/10000 [00:51<00:11]
 83%|=================   | 8291/10000 [00:52<00:10]
 84%|=================   | 8450/10000 [00:53<00:09]
 86%|=================   | 8599/10000 [00:54<00:08]
 87%|=================   | 8749/10000 [00:55<00:07]
 89%|==================  | 8908/10000 [00:56<00:06]
 91%|==================  | 9070/10000 [00:57<00:05]
 92%|==================  | 9232/10000 [00:58<00:04]
 94%|=================== | 9387/10000 [00:59<00:03]
 95%|=================== | 9549/10000 [01:00<00:02]
 97%|=================== | 9712/10000 [01:01<00:01]
 99%|===================| 9873/10000 [01:02<00:00]       /mnt/rubin/rubin/explainability/shap_uplift.py:169: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  df.groupby("feature_value", dropna=False)["value"]
/mnt/rubin/rubin/explainability/shap_uplift.py:169: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  df.groupby("feature_value", dropna=False)["value"]
/mnt/rubin/rubin/explainability/shap_uplift.py:169: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  df.groupby("feature_value", dropna=False)["value"]
/mnt/rubin/rubin/explainability/shap_uplift.py:169: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  df.groupby("feature_value", dropna=False)["value"]
/mnt/rubin/rubin/explainability/shap_uplift.py:169: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  df.groupby("feature_value", dropna=False)["value"]
/mnt/rubin/rubin/explainability/shap_uplift.py:169: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  df.groupby("feature_value", dropna=False)["value"]
/mnt/rubin/rubin/explainability/shap_uplift.py:169: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  df.groupby("feature_value", dropna=False)["value"]
/mnt/rubin/rubin/explainability/shap_uplift.py:169: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  df.groupby("feature_value", dropna=False)["value"]
/mnt/rubin/rubin/explainability/shap_uplift.py:169: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  df.groupby("feature_value", dropna=False)["value"]
/mnt/rubin/rubin/explainability/shap_uplift.py:169: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  df.groupby("feature_value", dropna=False)["value"]
/mnt/rubin/rubin/explainability/shap_uplift.py:169: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  df.groupby("feature_value", dropna=False)["value"]
/mnt/rubin/rubin/explainability/shap_uplift.py:169: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  df.groupby("feature_value", dropna=False)["value"]
/mnt/rubin/rubin/explainability/shap_uplift.py:169: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  df.groupby("feature_value", dropna=False)["value"]
/mnt/rubin/rubin/explainability/shap_uplift.py:169: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  df.groupby("feature_value", dropna=False)["value"]
