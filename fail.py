18:00:25 INFO [rubin.analysis] Base-Learner-Tuning: 7 Tasks abgeschlossen.
18:00:25 INFO [rubin.analysis]   catboost__grouped_outcome_regression__regressor__group_specific_shared_params__no_t__y best=-0.0018101
18:00:25 INFO [rubin.analysis]   catboost__outcome__classifier__all__no_t__y best=-0.0127236
18:00:25 INFO [rubin.analysis]   catboost__outcome_regression__regressor__all__with_t__y best=-0.00180985
18:00:25 INFO [rubin.analysis]   catboost__outcome_regression__regressor__all_direct__with_t__y best=-0.00180961
18:00:25 INFO [rubin.analysis]   catboost__propensity__classifier__all__no_t__t best=-0.693151
18:00:25 INFO [rubin.analysis]   catboost__propensity__classifier__all_direct__no_t__t best=-0.693168
18:00:25 INFO [rubin.analysis]   catboost__pseudo_effect__regressor__group_specific_shared_params__no_t__d best=-0.00180989
18:00:28 INFO [rubin.tuning] FMT 'NonParamDML': Starte 100 Trials (parallel=10, Wellen≈10).
18:02:03 INFO [rubin.categorical] CatBoost categorical patch (predict): 42 Spalten float→int konvertiert.
08:15:09 INFO [rubin.tuning] FMT 'NonParamDML': 100/100 Trials abgeschlossen (0 fehlgeschlagen, 0 gepruned).
08:15:09 INFO [rubin.tuning] FMT 'NonParamDML': Study freigegeben, gc.collect() durchgeführt.
08:15:10 INFO [rubin.analysis] FMT OOF-R-Loss: NonParamDML → -0.00180983
08:15:10 INFO [rubin.tuning] CausalForest-Tuning (CausalForestDML): Starte 100 Optuna-Trials, 5 äußere Folds (Seed=18).
