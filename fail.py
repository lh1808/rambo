10:00:52 ERROR [rubin.analysis] FMT 'DRLearner' fehlgeschlagen — bisherige Ergebnisse bleiben erhalten. Modell wird mit Default-/BLT-Parametern trainiert.
Traceback (most recent call last):
  File "/mnt/rubin/rubin/pipelines/analysis_pipeline.py", line 610, in _run_tuning
    add = final_tuner.tune_final_model(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rubin/rubin/tuning_optuna.py", line 1655, in tune_final_model
    if _fmt_is_rct:
       ^^^^^^^^^^^
UnboundLocalError: cannot access local variable '_fmt_is_rct' where it is not associated with a value
