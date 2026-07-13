12:55:48 INFO [rubin.analysis]   both__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y best=-1.37788e-07  skill=1.0000
12:55:48 INFO [rubin.analysis]   both__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y__adjusted best=-1.58982e-07  skill=0.0000
12:55:48 INFO [rubin.analysis]   both__outcome__classifier__all__no_t__y  best=-4.56809e-06  skill=1.0000
12:55:48 INFO [rubin.analysis]   both__outcome__classifier__all__with_t__y best=-3.13619e-05  skill=0.9999
12:55:48 INFO [rubin.analysis]   both__propensity__classifier__all__no_t__t best=-0.849901  skill=0.2264
12:55:48 INFO [rubin.analysis]   both__propensity__classifier__all_direct__no_t__t best=-0.853308  skill=0.2233
12:55:48 INFO [rubin.analysis]   both__pseudo_effect__regressor__group_specific_shared_params__no_t__d best=-3.89162e-07  skill=1.0000
12:55:48 INFO [rubin.analysis]   both__pseudo_effect__regressor__group_specific_shared_params__no_t__d__adjusted best=-4.02828e-07  skill=0.0000
12:55:48 WARNING [rubin.analysis] RCT-Diagnose: Propensity-Skill = 0.2264 > 0.01. Das Propensity-Modell kann Treatment besser als Zufall vorhersagen — die Daten sind möglicherweise NICHT randomisiert. Prüfe die Treatment-Zuweisung. Training verwendet trotzdem konstante Propensity (DummyClassifier), wie für RCT konfiguriert.
12:55:50 WARNING [rubin.tuning] base_type='both' aber kein '_learner_type' in params — Fallback auf 'catboost' (globaler Default). Das deutet auf ein fehlendes Tuning oder eine Rolle ohne getunte Params hin.
12:55:51 INFO [rubin.categorical] CatBoost categorical patch (predict): 5 Spalten float→int konvertiert.
12:56:16 ERROR [rubin.analysis] FMT 'NonParamDML' fehlgeschlagen — bisherige Ergebnisse bleiben erhalten. Modell wird mit Default-/BLT-Parametern trainiert.
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/rubin/pipelines/analysis_pipeline.py", line 645, in _run_tuning
    add = final_tuner.tune_final_model(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/tuning/final_model.py", line 449, in tune_final_model
    est.fit(Y_tune[tr], T_tune[tr], X=X_tune.iloc[tr], cache_values=True)
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 1703, in fit
    return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 501, in fit
    return super().fit(Y, T, X=X, W=W,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_cate_estimator.py", line 134, in call
    m(self, Y, T, *args, **kwargs)
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 876, in fit
    self._fit_final(Y=Y,
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/_ortho_learner.py", line 979, in _fit_final
    self._ortho_learner_model_final.fit(Y, T, **filter_none_kwargs(X=X, W=W, Z=Z,
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 101, in fit
    self._model_final.fit(X, T, T_res, Y_res, sample_weight=sample_weight,
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/econml/dml/dml.py", line 198, in fit
    raise AttributeError("This method can only be used with single-dimensional continuous treatment "
AttributeError: This method can only be used with single-dimensional continuous treatment or binary categorical treatment.
12:56:16 WARNING [rubin.tuning] base_type='both' aber kein '_learner_type' in params — Fallback auf 'catboost' (globaler Default). Das deutet auf ein fehlendes Tuning oder eine Rolle ohne getunte Params hin.
12:57:07 WARNING [rubin.tuning] base_type='both' aber kein '_learner_type' in params — Fallback auf 'catboost' (globaler Default). Das deutet auf ein fehlendes Tuning oder eine Rolle ohne getunte Params hin.
12:57:59 WARNING [rubin.tuning] base_type='both' aber kein '_learner_type' in params — Fallback auf 'catboost' (globaler Default). Das deutet auf ein fehlendes Tuning oder eine Rolle ohne getunte Params hin.
12:58:51 WARNING [rubin.tuning] base_type='both' aber kein '_learner_type' in params — Fallback auf 'catboost' (globaler Default). Das deutet auf ein fehlendes Tuning oder eine Rolle ohne getunte Params hin.
12:59:43 WARNING [rubin.tuning] base_type='both' aber kein '_learner_type' in params — Fallback auf 'catboost' (globaler Default). Das deutet auf ein fehlendes Tuning oder eine Rolle ohne getunte Params hin.
Wird ausgeführt ...
Config-Vorschau
