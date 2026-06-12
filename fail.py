INFO:__main__:Run-Artefakte für Horizont 13 unter /mnt/runs/2026-06-12T14-31_h13_model
INFO:__main__:Run-Artefakte für Horizont 52 unter /mnt/runs/2026-06-12T14-31_h52_model
INFO:__main__:Keine früheren Forecasts zur retrospektiven Evaluation vorhanden.
INFO:__main__:Wochentagsprofile: 80 Komponenten (Ebenen: {'component': 69, 'group': 11})
INFO:__main__:Disaggregation: 52 Wochen × 80 Komponenten → 364 Tage (weekend_policy=empirical).
INFO:__main__:Schreibe Forecast: 364 Tage ab 2026-06-15 bis 2027-06-13 (letztes Ist-Datum: 2026-06-11).
ERROR:pluto_multivariate_repository:Fehler beim Schreiben der Prognose: ibm_db_dbi::Error: Error 1: SQLExecute failed: [IBM][CLI Driver] CLI0111E  Numeric value out of range. SQLSTATE=22003 SQLCODE=-99999
Error 2: [IBM][CLI Driver] CLI0111E  Numeric value out of range. SQLSTATE=22003 SQLCODE=-99999

INFO:pluto_multivariate_repository:DB2 connection successfully closed.
Traceback (most recent call last):
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/ibm_db_dbi.py", line 1866, in executemany
    self.__rowcount = ibm_db.execute_many(self.stmt_handler, seq_parameters)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Exception: Error 1: SQLExecute failed: [IBM][CLI Driver] CLI0111E  Numeric value out of range. SQLSTATE=22003 SQLCODE=-99999
Error 2: [IBM][CLI Driver] CLI0111E  Numeric value out of range. SQLSTATE=22003 SQLCODE=-99999


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/da-pluto-timeseries/pluto_forecast_job.py", line 273, in <module>
    run_pluto_multivariate_forecast_job()
  File "/mnt/da-pluto-timeseries/pluto_forecast_job.py", line 263, in run_pluto_multivariate_forecast_job
    repo.write_forecast(df_write)
  File "/mnt/da-pluto-timeseries/pluto_multivariate_repository.py", line 446, in write_forecast
    cursor.executemany(insert_sql, rows)
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/ibm_db_dbi.py", line 1885, in executemany
    raise self.messages[len(self.messages) - 1]
ibm_db_dbi.Error: ibm_db_dbi::Error: Error 1: SQLExecute failed: [IBM][CLI Driver] CLI0111E  Numeric value out of range. SQLSTATE=22003 SQLCODE=-99999
Error 2: [IBM][CLI Driver] CLI0111E  Numeric value out of range. SQLSTATE=22003 SQLCODE=-99999
