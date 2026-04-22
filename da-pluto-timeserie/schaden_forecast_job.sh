/opt/conda/envs/generic/lib/python3.11/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
INFO:__main__:Run-Artefakte für Horizont 13 unter /mnt/runs/2026-04-21T15-21_h13_model
INFO:__main__:Run-Artefakte für Horizont 52 unter /mnt/runs/2026-04-21T15-21_h52_model
INFO:__main__:Keine früheren Forecasts zur retrospektiven Evaluation vorhanden.
INFO:pluto_multivariate_repository:Staging-Tabelle t7.TA_DA_PLUTO_SP_2025_PROGNOSE_STAGING existiert nicht – wird angelegt (CREATE TABLE LIKE t7.TA_DA_PLUTO_SP_2025_PROGNOSE).
ERROR:pluto_multivariate_repository:Fehler beim Schreiben der Prognose: Konnte Staging-Tabelle t7.TA_DA_PLUTO_SP_2025_PROGNOSE_STAGING nicht anlegen. Bitte vom DBA als strukturelle Kopie von t7.TA_DA_PLUTO_SP_2025_PROGNOSE einrichten lassen. Ursprünglicher Fehler: ibm_db_dbi::ProgrammingError: Statement Execute Failed: [IBM][CLI Driver][DB2/LINUXX8664] SQL0552N  "DA00103" does not have the privilege to perform operation "CREATE TABLE".  SQLSTATE=42502 SQLCODE=-552
INFO:pluto_multivariate_repository:DB2 connection successfully closed.
Traceback (most recent call last):
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/ibm_db_dbi.py", line 1652, in _execute_helper
    return_value = ibm_db.execute(self.stmt_handler)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Exception: Statement Execute Failed: [IBM][CLI Driver][DB2/LINUXX8664] SQL0552N  "DA00103" does not have the privilege to perform operation "CREATE TABLE".  SQLSTATE=42502 SQLCODE=-552

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/da-pluto-timeseries/pluto_multivariate_repository.py", line 390, in _ensure_staging_table
    cursor.execute(f"CREATE TABLE {staging_table} LIKE {self.target_table}")
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/ibm_db_dbi.py", line 1812, in execute
    self._execute_helper(parameters)
  File "/opt/conda/envs/generic/lib/python3.11/site-packages/ibm_db_dbi.py", line 1667, in _execute_helper
    raise self.messages[len(self.messages) - 1]
ibm_db_dbi.ProgrammingError: ibm_db_dbi::ProgrammingError: Statement Execute Failed: [IBM][CLI Driver][DB2/LINUXX8664] SQL0552N  "DA00103" does not have the privilege to perform operation "CREATE TABLE".  SQLSTATE=42502 SQLCODE=-552

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/mnt/da-pluto-timeseries/pluto_forecast_job.py", line 212, in <module>
    run_pluto_multivariate_forecast_job()
  File "/mnt/da-pluto-timeseries/pluto_forecast_job.py", line 202, in run_pluto_multivariate_forecast_job
    repo.write_forecast(df_combined)
  File "/mnt/da-pluto-timeseries/pluto_multivariate_repository.py", line 298, in write_forecast
    self._ensure_staging_table(cursor, staging_table)
  File "/mnt/da-pluto-timeseries/pluto_multivariate_repository.py", line 393, in _ensure_staging_table
    raise RuntimeError(
RuntimeError: Konnte Staging-Tabelle t7.TA_DA_PLUTO_SP_2025_PROGNOSE_STAGING nicht anlegen. Bitte vom DBA als strukturelle Kopie von t7.TA_DA_PLUTO_SP_2025_PROGNOSE einrichten lassen. Ursprünglicher Fehler: ibm_db_dbi::ProgrammingError: Statement Execute Failed: [IBM][CLI Driver][DB2/LINUXX8664] SQL0552N  "DA00103" does not have the privilege to perform operation "CREATE TABLE".  SQLSTATE=42502 SQLCODE=-552
