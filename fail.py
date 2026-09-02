Cell In[6], line 43, in verbindung(config)
     41 if fehlend:
     42     raise ValueError(f"Verbindungsdaten unvollständig: {', '.join(fehlend)}")
---> 43 conn = ibm_db_dbi.connect(dsn(config), "", "")
     44 try:
     45     yield conn

File /opt/conda/envs/generic/lib/python3.11/site-packages/ibm_db_dbi.py:825, in connect(dsn, user, password, host, database, conn_options)
    823 except Exception as inst:
    824     LogMsg(EXCEPTION, f"An exception occurred while connecting: {inst}")
--> 825     raise _get_exception(inst)

OperationalError: ibm_db_dbi::OperationalError: [IBM][CLI Driver] SQL30081N  A communication error has been detected. Communication protocol being used: "TCP/IP".  Communication API being used: "SOCKETS".  Location where the error was detected: "10.131.68.132".  Communication function detecting the error: "recv".  Protocol specific error code(s): "104", "*", "0".  SQLSTATE=08001 SQLCODE=-30081
