Analyse fehlgeschlagen: Fehlgeschlagen (Exit -6)

Details:
21:09:46 INFO [rubin.analysis] Arbeitsverzeichnis: /mnt/rubin/runs
21:09:46 INFO [rubin.analysis] MLflow-Experiment 'rubin_GRP_PBV' (identisch mit DataPrep).
21:09:46 INFO [rubin.analysis] Run-Name-Suffix 'wilder-eisvogel' aus DataPrep übernommen.
21:09:47 INFO [rubin.analysis] DataPrep-Config nach MLflow geloggt: /mnt/rubin/runs/data/dataprep_config.yml
21:09:47 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
21:09:47 INFO [rubin.analysis] rubin Pipeline Start
21:09:47 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
21:09:47 INFO [rubin.analysis] Config: 8 Modelle (NonParamDML, DRLearner, SLearner, TLearner, XLearner, ParamDML, CausalForestDML, CausalForest), CATBOOST, 5-Fold CV, Parallel-Level 3
21:09:47 INFO [rubin.analysis] Nuisance Cross-Fitting (DML + DR): 5 interne Folds
21:09:47 INFO [rubin.analysis] Aktiv: BL-Tuning (100 Trials) | FMT (100 Trials) | CausalForest-Tuning | Ensemble
21:09:47 INFO [rubin.analysis] Validierungsmodus: Cross-Validation (5 Folds, Seed=42, Tuning-Seed=18)
21:09:47 INFO [rubin.analysis] Historischer Score: 3 NaN-Werte durch 0 ersetzt.
21:09:47 INFO [rubin.analysis] dtypes.json auto-erkannt (runs/data/dtypes.json): 254 Spalten-Dtypes wiederhergestellt (254 kategorial).
21:09:48 INFO [rubin.analysis] Memory-Reduktion: 240.7 MB → 240.7 MB (0% gespart).
21:09:48 INFO [rubin.analysis] Daten geladen: X=(299988, 745), T=(299988,) (unique=[0, 1]), Y=(299988,) (unique=[0, 1]), S=(299988,)
21:09:48 INFO [rubin.categorical] Kategorische Spalten erkannt: 254 von 745 Features (['D_FRAU', 'GESELLSCHAFT_MM', 'BEAMTER_FLG']... (+251)). Patche BOTH .fit()-Methoden für EconML-Kompatibilität.
21:09:49 INFO [rubin.feature_selection] Feature-Selektion: 2 Methoden sequentiell (alle Kerne pro Methode).
21:09:49 INFO [rubin.gpu] CatBoost GPU erkannt: 1 NVIDIA GPU(s) verfügbar → task_type='GPU'. Inkompatible Parameter (colsample_bylevel, rsm) werden automatisch entfernt.
21:09:49 INFO [rubin.categorical] CatBoost categorical patch (fit): 254 cat_features injiziert (DataFrame).
Application terminated with error: ??+0 (0x7F1B59EB989A)
??+0 (0x7F1B5970F439)
??+0 (0x7F1B5AD146CD)
??+0 (0x7F1B5AD134C6)
??+0 (0x7F1B5AD14988)
??+0 (0x7F1B5993BC9F)
??+0 (0x7F1B5993BA9E)
??+0 (0x7F203BD32AA4)
??+0 (0x7F203BDBFC6C)

(TCudaException) catboost/cuda/cuda_lib/cuda_base.h:245: CUDA error 222: the provided PTX was compiled with an unsupported toolchain.
Terminating due to uncaught exception 0x4419acc0410    what() -> "catboost/cuda/cuda_lib/cuda_base.h:245: CUDA error 222: the provided PTX was compiled with an unsupported toolchain."
 of type TCudaException
