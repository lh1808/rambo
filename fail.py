15:47:47 INFO [rubin.analysis] Metriken für 9 Modelle berechnet. Vorläufiger Champion: Ensemble. Diagnostik-Plots: Champion + Challenger
15:49:23 WARNING [rubin.evaluation.drtester_plots] DRTester evaluate_all fehlgeschlagen: 'NoneType' object is not subscriptable
Traceback (most recent call last):
  File "/mnt/rubin/rubin/evaluation/drtester_plots.py", line 829, in evaluate_cate_with_plots
    res = tester.evaluate_all(X_val.values, X_train.values if X_train is not None else None, n_groups=n_groups, n_bootstrap=n_bootstrap, seed=seed)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/evaluation/drtester_plots.py", line 383, in evaluate_all
    cal_res = self.evaluate_cal(n_groups=n_groups)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/validate/drtester.py", line 405, in evaluate_cal
    cuts = np.quantile(self.cate_preds_train_[:, k], np.linspace(0, 1, n_groups + 1))
                       ~~~~~~~~~~~~~~~~~~~~~~^^^^^^
TypeError: 'NoneType' object is not subscriptable
