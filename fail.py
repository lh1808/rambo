---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[194], line 3
      1 m_name = 'TREE'
      2 m = model_dict.get(m_name)
----> 3 df_CATE[m_name] = predict_in_batches(X, m, 100000)

Cell In[177], line 7, in predict_in_batches(df, model, batch_size)
      3     predictions_list = []
      4 
      5     for i in range(0, len(df), batch_size):
      6         batch = df.iloc[i:i + batch_size]
----> 7         batch_predictions = model.const_marginal_effect(batch).flatten()
      8         predictions_list.append(batch_predictions)
      9 
     10     return list(chain(*predictions_list))

File /home/ubuntu/da-hf1-rubin/rubin/training.py:513, in SurrogateTreeWrapper.const_marginal_effect(self, X)
    511     preds = np.column_stack([self.trees[k].predict(X) for k in arm_keys])
    512     return preds
--> 513 return self.tree.predict(X)

File /home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/lightgbm/sklearn.py:1144, in LGBMModel.predict(self, X, raw_score, start_iteration, num_iteration, pred_leaf, pred_contrib, validate_features, **kwargs)
   1141 predict_params = _choose_param_value("num_threads", predict_params, self.n_jobs)
   1142 predict_params["num_threads"] = self._process_n_jobs(predict_params["num_threads"])
-> 1144 return self._Booster.predict(  # type: ignore[union-attr]
   1145     X,
   1146     raw_score=raw_score,
   1147     start_iteration=start_iteration,
   1148     num_iteration=num_iteration,
   1149     pred_leaf=pred_leaf,
   1150     pred_contrib=pred_contrib,
   1151     validate_features=validate_features,
   1152     **predict_params,
   1153 )

File /home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/lightgbm/basic.py:4767, in Booster.predict(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features, **kwargs)
   4765     else:
   4766         num_iteration = -1
-> 4767 return predictor.predict(
   4768     data=data,
   4769     start_iteration=start_iteration,
   4770     num_iteration=num_iteration,
   4771     raw_score=raw_score,
   4772     pred_leaf=pred_leaf,
   4773     pred_contrib=pred_contrib,
   4774     data_has_header=data_has_header,
   4775     validate_features=validate_features,
   4776 )

File /home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/lightgbm/basic.py:1158, in _InnerPredictor.predict(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features)
   1149     _safe_call(
   1150         _LIB.LGBM_BoosterValidateFeatureNames(
   1151             self._handle,
   (...)   1154         )
   1155     )
   1157 if isinstance(data, pd_DataFrame):
-> 1158     data = _data_from_pandas(
   1159         data=data,
   1160         feature_name="auto",
   1161         categorical_feature="auto",
   1162         pandas_categorical=self.pandas_categorical,
   1163     )[0]
   1165 predict_type = _C_API_PREDICT_NORMAL
   1166 if raw_score:

File /home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/lightgbm/basic.py:851, in _data_from_pandas(data, feature_name, categorical_feature, pandas_categorical)
    849 else:
    850     if len(cat_cols) != len(pandas_categorical):
--> 851         raise ValueError("train and valid dataset categorical_feature do not match.")
    852     for col, category in zip(cat_cols, pandas_categorical):
    853         if list(data[col].cat.categories) != list(category):

ValueError: train and valid dataset categorical_feature do not match.
