18:10:32 INFO [rubin.tuning] Tuning-Task 'both__outcome_regression__regressor__all__with_t__y': X_input=299988 rows, indices=299988, X_task=(299988, 130), target=(299988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=outcome_regression
18:42:00 INFO [rubin.tuning] BLT 'both__outcome_regression__regressor__all__with_t__y': 320/320 Trials abgeschlossen (parallel=10, Wellen≈32).
18:42:01 INFO [rubin.tuning] Tuning-Task 'both__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': X_input=299988 rows, indices=299988, X_task=(299988, 129), target=(299988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=Y, objective=grouped_outcome_regression
19:22:09 INFO [rubin.tuning] BLT 'both__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y': 320/320 Trials abgeschlossen (parallel=10, Wellen≈32).
19:22:11 INFO [rubin.tuning] Tuning-Task 'both__propensity__classifier__all__no_t__t': X_input=299988 rows, indices=299988, X_task=(299988, 129), target=(299988,) (unique=[0, 1]), T_task unique=[0, 1], cv_splits=5, target_name=T, objective=propensity
19:32:12 INFO [rubin.tuning] BLT 'both__propensity__classifier__all__no_t__t': 40/40 Trials abgeschlossen (parallel=10, Wellen≈4).
19:32:12 INFO [rubin.analysis] Base-Learner-Tuning: 4 Tasks abgeschlossen (160 Trials/Task). Ø Score: 0.128788
19:38:01 INFO [rubin.tuning] FMT 'NonParamDML': Starte 320 Trials (parallel=10, Wellen≈32).
