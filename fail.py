16:25:27 INFO [rubin.tuning] Tuning-Task 'lgbm__outcome__classifier__all__no_t__y': X_input=389988 rows, indices=389988, X_task=(389988, 81), target=(389988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome
16:26:25 INFO [rubin.tuning] Tuning-Task 'lgbm__outcome_regression__regressor__all__with_t__y': X_input=389988 rows, indices=389988, X_task=(389988, 82), target=(389988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome_regression
16:26:55 INFO [rubin.tuning] Tuning-Task 'lgbm__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': X_input=389988 rows, indices=389984, X_task=(389984, 81), target=(389984,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=grouped_outcome_regression
16:27:47 INFO [rubin.tuning] Tuning-Task 'lgbm__propensity__classifier__all__no_t__t': X_input=389988 rows, indices=389988, X_task=(389988, 81), target=(389988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=T, objective=propensity
16:29:04 INFO [rubin.tuning] Tuning-Task 'lgbm__pseudo_effect__regressor__group_specific_shared_params__no_t__d': X_input=389988 rows, indices=389984, X_task=(389984, 81), target=(389984,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=D, objective=pseudo_effect
16:39:18 INFO [rubin.analysis] Base-Learner-Tuning: 5 Tasks abgeschlossen (50 Trials/Task). Ø Score: -0.142225
16:43:11 INFO [rubin.analysis] FMT R-Score: NonParamDML → -4.35607e-11
16:43:11 INFO [rubin.analysis] FMT R-Score: DRLearner → 0.00734105
16:43:12 INFO [rubin.analysis] NonParamDML model_final effektive Params: {'n_estimators': 126, 'num_leaves': 15, 'max_depth': 6, 'min_child_samples': 33, 'min_child_weight': 3.960964849196579, 'colsample_bytree': 0.6808258044947768, 'reg_alpha': 43.407585331754674, 'reg_lambda': 6.064354375521055, 'path_smooth': 12.209820375801016} (explicit_tuned=ja, fmt_fixed=False)
16:43:12 INFO [rubin.training] NonParamDML: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
16:47:56 INFO [rubin.analysis]   NonParamDML: Training + Cross-Predictions in 284.7s
16:47:57 WARNING [rubin.analysis] WARNUNG: Predictions_NonParamDML hat nur 5 distinkte Werte bei 5 Folds (Range=1.80e-05, Mean=8.48e-04). Das CATE-Modell ist wahrscheinlich zu einem Intercept kollabiert. Empfehlungen: (1) final_model_tuning.enabled=true aktivieren, (2) Prüfen ob base_fixed_params zu restriktiv sind (min_child_samples, num_leaves, max_depth), (3) Mehr Features oder Feature-Engineering.
16:47:58 INFO [rubin.analysis] DRLearner model_final effektive Params: {'n_estimators': 159, 'num_leaves': 13, 'max_depth': 4, 'min_child_samples': 380, 'min_child_weight': 23.422439507845066, 'colsample_bytree': 0.5037253733042868, 'reg_alpha': 0.6544804858123543, 'reg_lambda': 4.881743905451841, 'path_smooth': 17.94150245811999} (explicit_tuned=ja, fmt_fixed=False)
16:47:58 INFO [rubin.training] DRLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
16:51:59 INFO [rubin.analysis]   DRLearner: Training + Cross-Predictions in 241.5s
16:52:00 WARNING [rubin.analysis] WARNUNG: Predictions_DRLearner hat nur 5 distinkte Werte bei 5 Folds (Range=1.58e-05, Mean=8.47e-04). Das CATE-Modell ist wahrscheinlich zu einem Intercept kollabiert. Empfehlungen: (1) final_model_tuning.enabled=true aktivieren, (2) Prüfen ob base_fixed_params zu restriktiv sind (min_child_samples, num_leaves, max_depth), (3) Mehr Features oder Feature-Engineering.
16:52:01 INFO [rubin.training] SLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
16:52:09 INFO [rubin.analysis]   SLearner: Training + Cross-Predictions in 9.2s
16:52:10 INFO [rubin.analysis] Predictions_SLearner: CATE min=-0.0145903, median=0.000336104, max=0.0160248, std=0.000511996, unique=103976/389988, non-zero=389965/389988
16:52:11 INFO [rubin.training] TLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
16:52:27 INFO [rubin.analysis]   TLearner: Training + Cross-Predictions in 17.0s
16:52:28 INFO [rubin.analysis] Predictions_TLearner: CATE min=-0.081372, median=0.000530895, max=0.0987318, std=0.0022127, unique=99587/389988, non-zero=389988/389988
16:52:29 INFO [rubin.training] XLearner: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
16:53:15 INFO [rubin.analysis]   XLearner: Training + Cross-Predictions in 46.0s
16:53:16 INFO [rubin.analysis] Predictions_XLearner: CATE min=-0.032005, median=0.000559598, max=0.027456, std=0.00141874, unique=388562/389988, non-zero=389988/389988
16:53:16 INFO [rubin.training] ParamDML: 5 Folds parallel (n_jobs=5, threads) auf 64 Kernen.
16:58:23 INFO [rubin.analysis]   ParamDML: Training + Cross-Predictions in 306.7s
16:58:24 INFO [rubin.analysis] Predictions_ParamDML: CATE min=-0.0887955, median=0.000670935, max=0.0301567, std=0.00215106, unique=389451/389988, non-zero=389988/389988
18:41:51 INFO [rubin.analysis] CausalForestDML EconML tune(): train=311990 Beobachtungen. Wald-Parameter optimiert.
18:41:51 INFO [rubin.training] CausalForestDML: Erzwinge sequentielle Folds (GRF nutzt intern joblib-Prozesse, die in Threads zu Deadlocks führen).
18:41:51 INFO [rubin.training] CausalForestDML: 5 Folds sequentiell.
