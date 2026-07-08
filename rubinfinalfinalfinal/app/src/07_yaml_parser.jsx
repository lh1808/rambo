const YAML_TO_CFG = {
  "study_type":"studyType","mlflow.experiment_name":"expName","constants.SEED":"seed","constants.tuning_seed":"tuningSeed","constants.parallel_level":"parallelLevel","constants.work_dir":"workDir",
  "data_files.x_file":"x_file","data_files.t_file":"t_file","data_files.y_file":"y_file","data_files.s_file":"s_file",
  "data_files.eval_x_file":"eval_x_file","data_files.eval_t_file":"eval_t_file","data_files.eval_y_file":"eval_y_file","data_files.eval_s_file":"eval_s_file","data_files.eval_mask_file":"eval_mask_file",
  "treatment.type":"treatmentType","treatment.reference_group":"refGroup",
  "historical_score.name":"histScoreName","historical_score.column":"histScoreCol","historical_score.higher_is_better":"histScoreHigher",
  "data_processing.validate_on":"validateOn","data_processing.cross_validation_splits":"cvSplits","data_processing.mc_iters":"mcIters","data_processing.mc_agg":"mcAgg","data_processing.dml_crossfit_folds":"dmlCrossfitFolds",
  "data_processing.df_frac":"dfFrac","data_processing.reduce_memory":"reduceMem",
  "feature_selection.enabled":"fsEnabled","feature_selection.methods":"fsMethods","feature_selection.correlation_threshold":"fsCorrThresh","feature_selection.max_features":"fsMaxFeatures",
  "models.models_to_train":"models","models.ensemble":"ensembleEnabled",
  "base_learner.type":"baseLearner","base_learner.fixed_params":"blFixed",
  "causal_forest.tune_enabled":"cfTune","causal_forest.search_space":"_cfSS","causal_forest.depth_choices":"_cfDepthChoices","causal_forest.criterion_choices":"_cfCriterionChoices","causal_forest.n_trials":"cfTrials","causal_forest.single_fold":"cfSingleFold","causal_forest.scorer":"cfScorer","causal_forest.overfit_penalty":"cfOverfitPenalty","causal_forest.overfit_tolerance":"cfOverfitTolerance","causal_forest.overfit_max_penalized_gap":"cfOverfitMaxGap","causal_forest.tune_models":"cfTuneModels",
  "tuning.enabled":"tuningEnabled","tuning.n_trials":"tuningTrials","tuning.cv_splits":"dmlCrossfitFolds","tuning.single_fold":"tuningSingleFold","tuning.overfit_penalty":"overfitPenalty","tuning.overfit_tolerance":"overfitTolerance","tuning.overfit_max_penalized_gap":"overfitMaxGap",
  "tuning.timeout_seconds":"tuningTimeout","tuning.max_tuning_rows":"tuningMaxRows","tuning.models":"tuningModels",
  "final_model_tuning.enabled":"fmtEnabled","final_model_tuning.models":"fmtModels","final_model_tuning.cv_splits":"dmlCrossfitFolds","final_model_tuning.single_fold":"fmtSingleFold","final_model_tuning.overfit_penalty":"fmtOverfitPenalty","final_model_tuning.overfit_tolerance":"fmtOverfitTolerance","final_model_tuning.overfit_max_penalized_gap":"fmtOverfitMaxGap","final_model_tuning.scorer":"fmtScorer","final_model_tuning.n_trials":"fmtTrials",
  "final_model_tuning.max_tuning_rows":"fmtMaxRows","final_model_tuning.timeout_seconds":"fmtTimeout","final_model_tuning.fixed_params":"fmtFixed",
  "shap_values.calculate_shap_values":"explEnabled","shap_values.n_shap_values":"explSampleSize","shap_values.top_n_features":"explTopN","shap_values.num_bins":"shapBins",
  "selection.metric":"selMetric","selection.higher_is_better":"higherBetter","selection.manual_champion":"manualChamp",
  "surrogate_tree.enabled":"surrEnabled","surrogate_tree.min_samples_leaf":"surrMinLeaf","surrogate_tree.num_leaves":"surrLeaves","surrogate_tree.max_depth":"surrDepth",
  "optional_output.output_dir":"outputDir","optional_output.save_predictions":"savePreds","optional_output.predictions_format":"predsFormat","optional_output.max_prediction_rows":"maxPredRows",
  "bundle.enabled":"bundleEnabled","bundle.base_dir":"bundleDir","bundle.log_to_mlflow":"bundleMlflow",
};

const parseYamlToCfg = (yamlText) => {
  const result = {...DEFAULT_CFG};
  let section = "";
  for(const raw of yamlText.split("\n")) {
    const line = raw.replace(/#.*$/,"").trimEnd(); // strip comments
    if(!line.trim()) continue;
    const secMatch = line.match(/^([a-z_]+):\s*$/);
    if(secMatch) { section = secMatch[1]; continue; }
    // Top-level key-value (kein Indent, hat Wert)
    const topKv = line.match(/^([a-z_]+):\s+(.+)$/);
    if(topKv) {
      const tlKey = topKv[1];
      const tlCfgKey = YAML_TO_CFG[tlKey];
      if(tlCfgKey) {
        let v = topKv[2].trim();
        if(v === "true" || v === "True") v = true;
        else if(v === "false" || v === "False") v = false;
        else if(v === "null" || v === "None") v = null;
        else if(!isNaN(Number(v)) && v !== "") v = Number(v);
        else v = v.replace(/^"|"$/g,"");
        result[tlCfgKey] = v;
      }
      continue;
    }
    const kvMatch = line.match(/^\s+([a-zA-Z0-9_]+):\s*(.+)$/);
    if(!kvMatch) continue;
    const [,key,rawVal] = kvMatch;
    const fullKey = section + "." + key;
    const cfgKey = YAML_TO_CFG[fullKey];
    if(!cfgKey) continue;
    // Parse value
    let val = rawVal.trim();
    if(val === "true" || val === "True") val = true;
    else if(val === "false" || val === "False") val = false;
    else if(val === "null" || val === "None") val = null;
    else if(val.startsWith("[") && val.endsWith("]")) {
      val = val.slice(1,-1).split(",").map(s=>{
        s=s.trim(); if(!s) return undefined;
        if(s==="true"||s==="True") return true;
        if(s==="false"||s==="False") return false;
        if(!isNaN(Number(s))&&s!=="") return Number(s);
        return s;
      }).filter(v=>v!==undefined);
    } else if(val.startsWith("{") && val.endsWith("}")) {
      try {
        const inner = val.slice(1,-1).trim();
        if(!inner) { val = {}; } else {
          const obj = {};
          inner.split(",").forEach(p => {
            const [k,...rest] = p.split(":");
            let v = rest.join(":").trim();
            if(v==="true") v=true; else if(v==="false") v=false; else if(v==="null") v=null;
            else if(!isNaN(Number(v))) v=Number(v); else v=v.replace(/^"|"$/g,"");
            obj[k.trim()] = v;
          });
          val = obj;
        }
      } catch(e) { val = {}; }
    } else if(!isNaN(Number(val)) && val !== "") val = Number(val);
    result[cfgKey] = val;
  }
  // Derive downsample from df_frac
  if(result.dfFrac && result.dfFrac < 1 && result.dfFrac > 0) result.downsample = true;
  // Parse methods if string
  if(typeof result.fsMethods === "string") result.fsMethods = [result.fsMethods];
  // Parse forest_fixed_params (nested YAML block)
  const cfLines = yamlText.split("\n");
  let inForest = false, forestParams = {};
  for(const line of cfLines) {
    if(line.match(/^\s+forest_fixed_params:\s*$/)) { inForest = true; continue; }
    if(inForest) {
      const m = line.match(/^\s{4,}(\w+):\s*(.+)$/);
      if(m) {
        let v = m[2].trim();
        if(v==="true") v=true; else if(v==="false") v=false; else if(v==="null") v=null; else if(!isNaN(Number(v))) v=Number(v);
        forestParams[m[1]] = v;
      } else { inForest = false; }
    }
  }
  if(Object.keys(forestParams).length > 0) result.cfFixed = forestParams;
  // Parse nested fixed_params for "both"-mode (base_learner + final_model_tuning)
  const parseNestedFixedParams = (sectionName, resultKey) => {
    const lines = yamlText.split("\n");
    let inSection = false, inFixed = false;
    const nested = {};
    for(const line of lines) {
      if(line.match(new RegExp("^" + sectionName + ":\\s*$"))) { inSection = true; inFixed = false; continue; }
      if(inSection && line.match(/^[a-z_]+:\s*$/)) { inSection = false; inFixed = false; continue; }
      if(inSection && line.match(/^\s+fixed_params:\s*$/)) { inFixed = true; continue; }
      if(inFixed) {
        const m = line.match(/^\s{4,}(lgbm|catboost):\s*\{(.+)\}\s*$/);
        if(m) {
          const obj = {};
          m[2].split(",").forEach(p => {
            const [k,...rest] = p.split(":");
            if(!k) return;
            let v = rest.join(":").trim();
            if(v==="true") v=true;
            else if(v==="false") v=false;
            else if(v==="null") v=null;
            else if(!isNaN(Number(v)) && v !== "") v=Number(v);
            else v = v.replace(/^"|"$/g,"");
            obj[k.trim()] = v;
          });
          nested[m[1]] = obj;
        } else if(line.match(/^\s+[a-z_]+:/) && !line.match(/^\s{4,}/)) {
          inFixed = false;
        }
      }
    }
    if(Object.keys(nested).length > 0) result[resultKey] = nested;
  };
  parseNestedFixedParams("base_learner", "blFixed");
  parseNestedFixedParams("final_model_tuning", "fmtFixed");
  // Parse feature_selection methods
  const fsMatch = yamlText.match(/methods:\s*\[([^\]]+)\]/);
  if(fsMatch) result.fsMethods = fsMatch[1].split(",").map(s=>s.trim()).filter(Boolean);

  // Parse search_space für BL (tuning.search_space) und FMT (final_model_tuning.search_space)
  // YAML-Struktur:
  //   tuning:
  //     search_space:
  //       catboost:
  //         iterations: {type: int, low: 100, high: 300}
  //       lgbm:
  //         n_estimators: {type: int, low: 200, high: 400}
  // → wird in React-State `sp`/`spFmt` Struktur überführt:
  //   {catboost: {iterations: {low: 100, high: 300}}, lgbm: {n_estimators: {low: 200, high: 400}}}
  const parseSearchSpace = (sectionName) => {
    const lines = yamlText.split("\n");
    const out = {catboost: {}, lgbm: {}};
    let inSection = false, inSS = false, curLearner = null;
    for (const raw of lines) {
      const line = raw.replace(/#.*$/, "").trimEnd();
      if (!line.trim()) continue;
      // Start der Ziel-Sektion
      if (line.match(new RegExp("^" + sectionName + ":\\s*$"))) {
        inSection = true; inSS = false; curLearner = null; continue;
      }
      // Ende Sektion (neue top-level Sektion)
      if (inSection && line.match(/^[a-z_]+:\s*$/)) {
        inSection = false; inSS = false; curLearner = null; continue;
      }
      if (!inSection) continue;
      // search_space: Block-Start (2-space indent)
      if (line.match(/^\s{2}search_space:\s*$/)) { inSS = true; curLearner = null; continue; }
      // andere 2-space-Keys beenden search_space
      if (inSS && line.match(/^\s{2}[a-z_]+:/)) { inSS = false; curLearner = null; continue; }
      if (!inSS) continue;
      // Learner-Header (4-space indent): "    catboost:" oder "    lgbm:"
      const learnerMatch = line.match(/^\s{4}(catboost|lgbm):\s*$/);
      if (learnerMatch) { curLearner = learnerMatch[1]; continue; }
      // Parameter-Zeile (6-space indent): "      iterations: {type: int, low: 100, high: 300}"
      if (curLearner) {
        const pMatch = line.match(/^\s{6}([a-z0-9_]+):\s*\{([^}]+)\}\s*$/);
        if (pMatch) {
          const [, pname, inner] = pMatch;
          const parts = inner.split(",").map(s => s.trim());
          const obj = {};
          parts.forEach(p => {
            const [k, ...rest] = p.split(":");
            if (!k) return;
            const v = rest.join(":").trim();
            const n = Number(v);
            obj[k.trim()] = isNaN(n) ? v.replace(/^"|"$/g, "") : n;
          });
          // Konvertiere backend-Format {type, low, high, [log]} → UI-Format {low, high}
          // Das log-Flag wird NICHT in sp gespeichert (es kommt aus UI-Defs), aber wir
          // lesen es, damit der Parser nicht stolpert.
          if (obj.low !== undefined && obj.high !== undefined) {
            out[curLearner][pname] = {low: obj.low, high: obj.high};
          }
        }
      }
    }
    // Nur zurückgeben wenn mindestens eine Änderung drin ist
    const hasAny = Object.values(out).some(d => Object.keys(d).length > 0);
    return hasAny ? out : null;
  };
  const _bl_sp = parseSearchSpace("tuning");
  if (_bl_sp) result.__sp = _bl_sp;
  const _fmt_sp = parseSearchSpace("final_model_tuning");
  if (_fmt_sp) result.__spFmt = _fmt_sp;

  return result;
};
