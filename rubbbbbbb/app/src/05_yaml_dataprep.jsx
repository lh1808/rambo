const buildDataPrepYaml = (dp, cfg) => {
  const l=[],a=s=>l.push(s);
  const outPath = dp.outputPath||"runs/data";
  a("# rubin DataPrep – generiert von der Web-UI");
  a("");
  // data_files ist Pflicht in AnalysisConfig – Pfade zeigen auf DataPrep-Output
  a("data_files:");
  a(`  x_file: ${outPath}/X.parquet`);
  a(`  t_file: ${outPath}/T.parquet`);
  a(`  y_file: ${outPath}/Y.parquet`);
  if(dp.evalFiles && dp.evalFiles.filter(f=>f).length > 0) {
    a(`  eval_x_file: ${outPath}/X_eval.parquet`);
    a(`  eval_t_file: ${outPath}/T_eval.parquet`);
    a(`  eval_y_file: ${outPath}/Y_eval.parquet`);
  }
  if((dp.evalFileIdxs||[]).length > 0 || dp.evalFileIdx!=null) {
    a(`  eval_mask_file: ${outPath}/eval_mask.npy`);
  }
  a("");
  a("data_prep:");
  a("  data_path:");
  (dp.files||[""]).filter(f=>f).forEach(f => a(`    - "${f}"`));
  const _tgts = (dp.targets||[""]).filter(t=>t.trim());
  a(`  target: ${_tgts.length===1 ? _tgts[0] : "["+_tgts.join(", ")+"]"}`);
  a(`  treatment: ${dp.treatment||""}`);
  if(dp.scoreName) a(`  score_name: ${dp.scoreName}`);
  if(dp.featurePath) a(`  feature_path: "${dp.featurePath}"`);
  a(`  output_path: ${outPath}`);
  a(`  delimiter: "${dp.delimiter||","}"`);
  if(dp.chunksize) a(`  chunksize: ${dp.chunksize}`);
  a(`  sas_encoding: ${dp.sasEncoding||"utf-8"}`);
  if(dp.fillNa && dp.fillNa!=="(keine)") a(`  fill_na_method: ${dp.fillNa}`);
  if(dp.binaryTarget !== undefined) a(`  binary_target: ${dp.binaryTarget ? "true" : "false"}`);
  if(dp.dedup && dp.dedupCol) {
    a("  deduplicate: true");
    a(`  deduplicate_id_column: ${dp.dedupCol}`);
  }
  if(dp.scoreAsFeature) a("  score_as_feature: true");
  if(dp.multiOpt) a(`  multiple_files_option: ${dp.multiOpt}`);
  if(dp.controlFileIndex>0) a(`  control_file_index: ${dp.controlFileIndex}`);
  if(dp.balanceTreat) a("  balance_treatments: true");
  if((dp.evalFileIdxs||[]).length > 1) {
    a(`  eval_file_index:`);
    (dp.evalFileIdxs||[]).forEach(i => a(`    - ${i}`));
  } else if((dp.evalFileIdxs||[]).length === 1) {
    a(`  eval_file_index: ${dp.evalFileIdxs[0]}`);
  } else if(dp.evalFileIdx!=null) {
    a(`  eval_file_index: ${dp.evalFileIdx}`);
  }
  // Explizite Feature-Auswahl (manuell oder Dictionary)
  const fs = dp.featureSelection||{};
  const selectedFeatures = Object.entries(fs).filter(([k,v])=>v===true).map(([k])=>k);
  const deselectedExists = Object.values(fs).some(v=>v===false);
  if(deselectedExists && selectedFeatures.length > 0) {
    a("  features:");
    selectedFeatures.forEach(f => a(`    - "${f}"`));
  }
  // Explizite Datentypen (cat/num)
  const ct = dp.colTypes||{};
  const catCols = Object.entries(ct).filter(([k,v])=>v==="cat").map(([k])=>k);
  if(catCols.length > 0) {
    a("  categorical_columns:");
    catCols.forEach(c => a(`    - "${c}"`));
  }
  // Treatment-Mapping (replacement)
  const tm = dp.treatMap||{};
  if(Object.keys(tm).length>0) {
    a("  treatment_replacement:");
    Object.entries(tm).forEach(([k,v]) => a(`    "${k}": ${v}`));
  }
  // MLflow
  if(dp.dpMlflow) {
    a("  log_to_mlflow: true");
    a(`  mlflow_experiment_name: ${cfg.expName||"rubin"}`);
    if(dp.dpRunName) a(`  mlflow_run_name: ${dp.dpRunName}`);
  }
  // Eval-Dateien (separater Datensatz, Preprocessor wird nur auf Train gefittet)
  if(dp.evalFiles && dp.evalFiles.filter(f=>f).length > 0) {
    a("  eval_data_path:");
    dp.evalFiles.filter(f=>f).forEach(f => a(`    - "${f}"`));
  }
  return l.join("\n");
};

// ── DataPrep Page ──