// ── YAML Builder + Validate ──
const buildYaml = (cfg, sp, spFmt) => {
  const l=[],a=s=>l.push(s),bl=cfg.baseLearner||"catboost";
  const jsonInline = (obj) => {
    if(!obj || Object.keys(obj).length===0) return "{}";
    return "{ "+Object.entries(obj).map(([k,v])=>k+": "+(v===null?"null":v===true?"true":v===false?"false":typeof v==="string"?"\""+v+"\"":v)).join(", ")+" }";
  };
  const emitSS = (ssObj, indent) => {
    const pad = " ".repeat(indent);
    // Bei "both" beide Learner emittieren; sonst nur den aktiven
    const emitForLearner = (lname, defs) => {
      const cur = (ssObj && ssObj[lname]) || {};
      const modified = Object.entries(defs).filter(([k, d]) => {
        const c = cur[k];
        if (!c) return false;
        const lo = c.low ?? d.min, hi = c.high ?? d.max;
        return lo !== d.min || hi !== d.max;
      });
      if (modified.length === 0) return false;
      a(pad + "  " + lname + ":");
      modified.forEach(([k, d]) => {
        const c = cur[k];
        let lo = c.low ?? d.min, hi = c.high ?? d.max;
        const t = d.type === "int" ? "int" : "float";
        // log=True erfordert low > 0 (Optuna-Constraint)
        if (d.log && lo <= 0) lo = 1e-6;
        const logSuffix = d.log ? ", log: true" : "";
        a(pad + `    ${k}: {type: ${t}, low: ${lo}, high: ${hi}${logSuffix}}`);
      });
      return true;
    };
    if (bl === "both") {
      const cbHas = Object.entries(CB).some(([k, d]) => {
        const c = (ssObj && ssObj.catboost || {})[k];
        return c && ((c.low ?? d.min) !== d.min || (c.high ?? d.max) !== d.max);
      });
      const lgHas = Object.entries(LGBM).some(([k, d]) => {
        const c = (ssObj && ssObj.lgbm || {})[k];
        return c && ((c.low ?? d.min) !== d.min || (c.high ?? d.max) !== d.max);
      });
      if (!cbHas && !lgHas) return;
      a(pad + "search_space:");
      emitForLearner("catboost", CB);
      emitForLearner("lgbm", LGBM);
    } else {
      const defs = bl === "catboost" ? CB : LGBM;
      const hasAny = Object.entries(defs).some(([k, d]) => {
        const c = (ssObj && ssObj[bl] || {})[k];
        return c && ((c.low ?? d.min) !== d.min || (c.high ?? d.max) !== d.max);
      });
      if (!hasAny) return;
      a(pad + "search_space:");
      emitForLearner(bl, defs);
    }
  };
  const emitSSFmt = (ssObj, indent) => {
    const pad = " ".repeat(indent);
    const emitForLearner = (lname, defs) => {
      const cur = (ssObj && ssObj[lname]) || {};
      const modified = Object.entries(defs).filter(([k, d]) => {
        const c = cur[k];
        if (!c) return false;
        const lo = c.low ?? d.min, hi = c.high ?? d.max;
        return lo !== d.min || hi !== d.max;
      });
      if (modified.length === 0) return false;
      a(pad + "  " + lname + ":");
      modified.forEach(([k, d]) => {
        const c = cur[k];
        let lo = c.low ?? d.min, hi = c.high ?? d.max;
        const t = d.type === "int" ? "int" : "float";
        if (d.log && lo <= 0) lo = 1e-6;
        const logSuffix = d.log ? ", log: true" : "";
        a(pad + `    ${k}: {type: ${t}, low: ${lo}, high: ${hi}${logSuffix}}`);
      });
      return true;
    };
    if (bl === "both") {
      const cbHas = Object.entries(CB_FMT).some(([k, d]) => {
        const c = (ssObj && ssObj.catboost || {})[k];
        return c && ((c.low ?? d.min) !== d.min || (c.high ?? d.max) !== d.max);
      });
      const lgHas = Object.entries(LGBM_FMT).some(([k, d]) => {
        const c = (ssObj && ssObj.lgbm || {})[k];
        return c && ((c.low ?? d.min) !== d.min || (c.high ?? d.max) !== d.max);
      });
      if (!cbHas && !lgHas) return;
      a(pad + "search_space:");
      emitForLearner("catboost", CB_FMT);
      emitForLearner("lgbm", LGBM_FMT);
    } else {
      const defs = bl === "catboost" ? CB_FMT : LGBM_FMT;
      const hasAny = Object.entries(defs).some(([k, d]) => {
        const c = (ssObj && ssObj[bl] || {})[k];
        return c && ((c.low ?? d.min) !== d.min || (c.high ?? d.max) !== d.max);
      });
      if (!hasAny) return;
      a(pad + "search_space:");
      emitForLearner(bl, defs);
    }
  };
  const mt = cfg.treatmentType==="multi";

  if((cfg.studyType||"rct") !== "rct") a(`study_type: ${cfg.studyType}`);
  a("");
  a("mlflow:");
  a(`  experiment_name: ${cfg.expName||"rubin"}`);
  a("");
  a("constants:");
  a(`  SEED: ${cfg.seed==null?42:cfg.seed}`);
  a(`  tuning_seed: ${cfg.tuningSeed==null?18:cfg.tuningSeed}`);
  if((cfg.parallelLevel||3) !== 3) a(`  parallel_level: ${cfg.parallelLevel}`);
  if(cfg.workDir) a(`  work_dir: "${cfg.workDir}"`);
  a("");
  a("data_files:");
  a(`  x_file: ${cfg.x_file||"runs/data/X.parquet"}`);
  a(`  t_file: ${cfg.t_file||"runs/data/T.parquet"}`);
  a(`  y_file: ${cfg.y_file||"runs/data/Y.parquet"}`);
  a(`  s_file: ${cfg.s_file||"null"}`);
  if(cfg.validateOn==="external") {
    if(cfg.eval_x_file) a(`  eval_x_file: ${cfg.eval_x_file}`);
    if(cfg.eval_t_file) a(`  eval_t_file: ${cfg.eval_t_file}`);
    if(cfg.eval_y_file) a(`  eval_y_file: ${cfg.eval_y_file}`);
    if(cfg.eval_s_file) a(`  eval_s_file: ${cfg.eval_s_file}`);
  }
  if(cfg.eval_mask_file) {
    if(Array.isArray(cfg.eval_mask_file)) {
      a(`  eval_mask_file:`);
      cfg.eval_mask_file.forEach(f => a(`    - ${f}`));
    } else {
      a(`  eval_mask_file: ${cfg.eval_mask_file}`);
    }
  }
  a("");
  a("treatment:");
  a(`  type: ${cfg.treatmentType||"binary"}`);
  a(`  reference_group: ${cfg.refGroup||0}`);
  a("");
  a("historical_score:");
  a(`  name: ${cfg.histScoreName||"historical_score"}`);
  a(`  column: ${cfg.histScoreCol||"S"}`);
  a(`  higher_is_better: ${cfg.histScoreHigher!==false}`);
  a("");
  a("data_processing:");
  a(`  validate_on: ${cfg.validateOn||"cross"}`);
  if(cfg.validateOn==="cross"||cfg.validateOn===undefined) a(`  cross_validation_splits: ${cfg.cvSplits||5}`);
  if(cfg.validateOn==="external") a(`  cross_validation_splits: ${cfg.cvSplits||5}`);
  if(cfg.downsample) a(`  df_frac: ${cfg.dfFrac||0.1}`);
  a(`  reduce_memory: ${cfg.reduceMem!==false}`);
  if((cfg.dmlCrossfitFolds||5) !== 5) a(`  dml_crossfit_folds: ${cfg.dmlCrossfitFolds}`);
  if(cfg.mcIters && cfg.mcIters > 0) {
    a(`  mc_iters: ${cfg.mcIters}`);
    if((cfg.mcAgg||"mean") !== "mean") a(`  mc_agg: ${cfg.mcAgg}`);
  }
  a("");
  a("feature_selection:");
  a(`  enabled: ${!!cfg.fsEnabled}`);
  if(cfg.fsEnabled) {
    a(`  methods: [${(cfg.fsMethods||["catboost_importance"]).join(", ")}]`);
    a(`  max_features: ${cfg.fsMaxFeatures||77}`);
    a(`  correlation_threshold: ${cfg.fsCorrThresh||0.9}`);
  }
  a("");
  a("models:");
  a(`  models_to_train: [${(cfg.models||["NonParamDML"]).join(", ")}]`);
  if(cfg.ensembleEnabled) a("  ensemble: true");
  a("");
  a("base_learner:");
  a(`  type: ${bl}`);
  const blF = cfg.blFixed||{};
  if(bl === "both") {
    const cbF = blF.catboost||{}, lgbmF = blF.lgbm||{};
    if(Object.keys(cbF).length > 0 || Object.keys(lgbmF).length > 0) {
      a("  fixed_params:");
      if(Object.keys(cbF).length > 0) a(`    catboost: ${jsonInline(cbF)}`);
      if(Object.keys(lgbmF).length > 0) a(`    lgbm: ${jsonInline(lgbmF)}`);
    }
  } else if(Object.keys(blF).length > 0) a(`  fixed_params: ${jsonInline(blF)}`);
  a("");
  if((cfg.models||[]).includes("CausalForestDML") || (cfg.models||[]).includes("CausalForest")) {
    a("causal_forest:");
    const cfF = cfg.cfFixed||{};
    if(Object.keys(cfF).length > 0) {
      a("  forest_fixed_params:");
      Object.entries(cfF).forEach(([k,v]) => a(`    ${k}: ${v===null?"null":v}`));
    } else {
      a("  forest_fixed_params: { n_jobs: -1 }");
    }
    a(`  tune_enabled: ${!!cfg.cfTune}`);
    a(`  n_trials: ${cfg.cfTrials||50}`);
    if(cfg.cfSingleFold) a("  single_fold: true");
    if(cfg.cfScorer && cfg.cfScorer !== "auto") a(`  scorer: ${cfg.cfScorer}`);
    if((cfg.cfOverfitPenalty||0)>0){a(`  overfit_penalty: ${cfg.cfOverfitPenalty}`);a(`  overfit_tolerance: ${cfg.cfOverfitTolerance===undefined?0.1:cfg.cfOverfitTolerance}`);a(`  overfit_max_penalized_gap: ${cfg.cfOverfitMaxGap===undefined?1.0:cfg.cfOverfitMaxGap}`);}
    if((cfg.cfTuneModels||[]).length > 0) a(`  tune_models: [${cfg.cfTuneModels.join(", ")}]`);
    if(cfg._cfDepthChoices && cfg._cfDepthChoices.length > 0) {
      a(`  depth_choices: [${cfg._cfDepthChoices.join(", ")}]`);
    }
    if(cfg._cfCriterionChoices && cfg._cfCriterionChoices.length > 0) {
      a(`  criterion_choices: [${cfg._cfCriterionChoices.join(", ")}]`);
    }
    const cfSS = cfg._cfSS||{};
    if(Object.keys(cfSS).length > 0) {
      a("  search_space:");
      Object.entries(cfSS).forEach(([k,v]) => {
        a(`    ${k}:`);
        if(v.low !== undefined) a(`      low: ${v.low}`);
        if(v.high !== undefined) a(`      high: ${v.high}`);
      });
    }
        a("");
  }
  a("tuning:");
  a(`  enabled: ${!!cfg.tuningEnabled}`);
  if(cfg.tuningEnabled) {
    a(`  n_trials: ${cfg.tuningTrials||50}`);
    if((cfg.dmlCrossfitFolds||5) !== 5) a(`  cv_splits: ${cfg.dmlCrossfitFolds}`);
    if(cfg.tuningSingleFold) a("  single_fold: true");if((cfg.overfitPenalty||0)>0){a(`  overfit_penalty: ${cfg.overfitPenalty}`);a(`  overfit_tolerance: ${cfg.overfitTolerance===undefined?0.2:cfg.overfitTolerance}`);a(`  overfit_max_penalized_gap: ${cfg.overfitMaxGap===undefined?1.0:cfg.overfitMaxGap}`);}
    if(cfg.tuningTimeout) a(`  timeout_seconds: ${cfg.tuningTimeout}`);
    if(cfg.tuningMaxRows) a(`  max_tuning_rows: ${cfg.tuningMaxRows}`);
    if((cfg.tuningModels||[]).length>0) a(`  models: [${cfg.tuningModels.join(", ")}]`);
      }
  if(sp) emitSS(sp, 2);
  a("");
  a("final_model_tuning:");
  a(`  enabled: ${!!cfg.fmtEnabled}`);
  if(cfg.fmtEnabled) {
    a(`  n_trials: ${cfg.fmtTrials||50}`);
    if((cfg.dmlCrossfitFolds||5) !== 5) a(`  cv_splits: ${cfg.dmlCrossfitFolds}`);
    if((cfg.fmtModels||[]).length > 0) a(`  models: [${cfg.fmtModels.join(", ")}]`);
    if(cfg.fmtSingleFold) a("  single_fold: true");
    if((cfg.fmtOverfitPenalty||0)>0){a(`  overfit_penalty: ${cfg.fmtOverfitPenalty}`);a(`  overfit_tolerance: ${cfg.fmtOverfitTolerance===undefined?0.1:cfg.fmtOverfitTolerance}`);a(`  overfit_max_penalized_gap: ${cfg.fmtOverfitMaxGap===undefined?1.0:cfg.fmtOverfitMaxGap}`);}
    if(cfg.fmtScorer && cfg.fmtScorer !== "auto") a(`  scorer: ${cfg.fmtScorer}`);
        if(cfg.fmtTimeout) a(`  timeout_seconds: ${cfg.fmtTimeout}`);
    if(cfg.fmtMaxRows) a(`  max_tuning_rows: ${cfg.fmtMaxRows}`);
  }
  const fmtF = cfg.fmtFixed||{};
  if(bl === "both") {
    const cbF2 = fmtF.catboost||{}, lgbmF2 = fmtF.lgbm||{};
    if(Object.keys(cbF2).length > 0 || Object.keys(lgbmF2).length > 0) {
      a("  fixed_params:");
      if(Object.keys(cbF2).length > 0) a(`    catboost: ${jsonInline(cbF2)}`);
      if(Object.keys(lgbmF2).length > 0) a(`    lgbm: ${jsonInline(lgbmF2)}`);
    }
  } else if(Object.keys(fmtF).length > 0) a(`  fixed_params: ${jsonInline(fmtF)}`);
  if(spFmt) emitSSFmt(spFmt, 2);
  a("");
  a("shap_values:");
  a(`  calculate_shap_values: ${!!cfg.explEnabled}`);
  if(cfg.explEnabled) {
    a(`  n_shap_values: ${cfg.explSampleSize||10000}`);
    a(`  top_n_features: ${cfg.explTopN||20}`);
    a(`  num_bins: ${cfg.shapBins||10}`);
  }
  a("");
  a("selection:");
  a(`  metric: ${cfg.selMetric||(mt?"policy_value":"qini")}`);
  a(`  higher_is_better: ${cfg.higherBetter!==false}`);
  if(cfg.manualChamp) a(`  manual_champion: ${cfg.manualChamp}`);
  a("");
  a("surrogate_tree:");
  a(`  enabled: ${!!cfg.surrEnabled}`);
  if(cfg.surrEnabled) {
    a(`  min_samples_leaf: ${cfg.surrMinLeaf||50}`);
    a(`  num_leaves: ${cfg.surrLeaves||31}`);
    if(cfg.surrDepth) a(`  max_depth: ${cfg.surrDepth}`);
  }
  a("");
  a("optional_output:");
  a(`  output_dir: ${cfg.outputDir||"null"}`);
  if(cfg.maxPredRows) a(`  max_prediction_rows: ${cfg.maxPredRows}`);
  a("");
  a("bundle:");
  a(`  enabled: ${!!cfg.bundleEnabled}`);
  if(cfg.bundleEnabled) {
    a(`  base_dir: ${cfg.bundleDir||"runs/bundles"}`);
    a(`  log_to_mlflow: ${cfg.bundleMlflow!==false}`);
  }
  return l.join("\n");
};
const validate = cfg => {
  const i=[];
  if(!cfg.expName)i.push("Experiment-Name fehlt.");
  if(!cfg.x_file)i.push("X-Datei nicht angegeben.");
  if(!cfg.t_file)i.push("T-Datei nicht angegeben.");
  if(!cfg.y_file)i.push("Y-Datei nicht angegeben.");
  if(cfg.validateOn==="external") {
    if(!cfg.eval_x_file)i.push("Eval X-Datei nicht angegeben (externe Validierung).");
    if(!cfg.eval_t_file)i.push("Eval T-Datei nicht angegeben (externe Validierung).");
    if(!cfg.eval_y_file)i.push("Eval Y-Datei nicht angegeben (externe Validierung).");
  }
  // Effektive Modelle: NaN-blockierte und MT-inkompatible herausfiltern
  const nanBlocked = cfg.hasNaN ? ["CausalForestDML","CausalForest"] : [];
  const effectiveModels = (cfg.models||[]).filter(m => !nanBlocked.includes(m));
  if(!effectiveModels.length)i.push("Keine Modelle ausgewählt.");
  if(cfg.treatmentType==="multi"){const b=effectiveModels.filter(m=>btOnly.has(m));if(b.length > 0)i.push(`MT nicht kompatibel mit: ${b.join(", ")}`);}
  // Manual Champion muss in Modellen sein (spiegelt Backend-Validator)
  if(cfg.manualChamp && !(cfg.models||[]).includes(cfg.manualChamp)){
    i.push(`Manueller Champion „${cfg.manualChamp}" ist nicht in der Modell-Liste enthalten.`);
  }
  // "both"-Modus: fixed_params muss verschachtelt sein (spiegelt Backend-Validator)
  if((cfg.baseLearner||"")==="both"){
    const checkNested = (name, fp) => {
      if(!fp || Object.keys(fp).length===0) return;
      const flatKeys = Object.entries(fp).filter(([k,v])=>!["lgbm","catboost"].includes(k) || (typeof v!=="object" || v===null)).map(([k])=>k);
      if(flatKeys.length > 0){
        i.push(`${name} bei „CatBoost & LGBM" muss verschachtelt sein (lgbm/catboost Sub-Dicts). Flache Keys: ${flatKeys.join(", ")}`);
      }
    };
    checkNested("base_learner.fixed_params", cfg.blFixed);
    checkNested("final_model_tuning.fixed_params", cfg.fmtFixed);
  }
  // Reference Group: muss gültig sein (bei binary nur 0 erlaubt)
  if(cfg.treatmentType==="binary" && cfg.refGroup!==0 && cfg.refGroup!==undefined && cfg.refGroup!==null){
    i.push(`treatment.reference_group=${cfg.refGroup}: Bei binary Treatment nur 0 erlaubt.`);
  }
  // MT-incompatible Selection-Metrics (spiegelt Backend-Validator _bt_only_metrics).
  // Hinweis: 'policy_value' (global IPW) ist bei Multi-Treatment die EMPFOHLENE
  // Metrik (siehe settings.py) und gehört nicht in diese Liste.
  if(cfg.treatmentType==="multi"){
    const btOnlyMetrics = new Set(["qini","auuc","uplift_at_10pct","uplift_at_20pct","uplift_at_50pct"]);
    if(btOnlyMetrics.has(cfg.selMetric)){
      i.push(`selection.metric='${cfg.selMetric}' existiert bei Multi-Treatment nicht. Empfohlen: 'policy_value' (global), 'policy_value_T1', 'qini_T1', 'qini_T2' etc.`);
    }
    // Qini-Scorer (FMT/CFT) ist binär-only — spiegelt Backend-Validator.
    if(cfg.fmtEnabled && cfg.fmtScorer==="qini"){
      i.push(`final_model_tuning.scorer='qini' ist bei Multi-Treatment nicht möglich (binär-only). Bitte 'rscore' oder 'auto' wählen.`);
    }
    if(cfg.cfTune && cfg.cfScorer==="qini"){
      i.push(`causal_forest.scorer='qini' ist bei Multi-Treatment nicht möglich (binär-only). Bitte 'rscore' oder 'auto' wählen.`);
    }
  }
  return i;
};

const PPreview = ({cfg,sp,spFmt,totalFits}) => {const y=buildYaml(cfg,sp,spFmt),issues=validate(cfg);const mt=cfg.treatmentType==="multi";return(<>
  <Row gap={10}>
    <MC value={(cfg.models||[]).length} label="Modelle"/>
    <MC value={(cfg.baseLearner||"catboost")==="both"?"Both":cfg.baseLearner==="lgbm"?"LGBM":"CB"} label="Learner"/>
    <MC value={cfg.tuningEnabled?(cfg.tuningTrials||50)+"T":"Aus"} label="BLT"/>
    <MC value={cfg.fmtEnabled?(cfg.fmtTrials||50)+"T":"Aus"} label="FMT"/>
    <MC value={cfg.cfTune?(cfg.cfTrials||50)+"T":"Aus"} label="CFT"/>
    <MC value={cfg.validateOn==="external"?"Ext.":cfg.eval_mask_file?"TMES":"CV-"+(cfg.cvSplits||5)} label="Validierung"/>
  </Row>
  <div style={{height:16}}/>
  <Sec title="Config-Vorschau"><div style={{background:"#1e1e2e",border:"none",borderRadius:10,padding:"20px 24px",fontFamily:MONO,fontSize:12,whiteSpace:"pre-wrap",maxHeight:520,overflowY:"auto",lineHeight:1.7,color:"#cdd6f4",boxShadow:"inset 0 2px 8px rgba(0,0,0,0.15)"}}>{y}</div><div style={{marginTop:14,display:"flex",gap:10}}><Btn small onClick={()=>navigator.clipboard?.writeText(y)}>Kopieren</Btn><Btn small secondary onClick={()=>{const b=new Blob([y],{type:"text/yaml"});const u=URL.createObjectURL(b);const a=document.createElement("a");a.href=u;a.download="config.yml";a.click();}}>YAML herunterladen</Btn><Btn small secondary onClick={()=>fetch("./api/save-config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({yaml:y,filename:"config.yml"})}).then(r=>r.json()).then(d=>{if(d.status==="done")alert("Config gespeichert: "+d.path)}).catch(()=>alert("Backend nicht erreichbar"))}>Auf Server speichern</Btn></div></Sec><Sec title="Validierung">{issues.length>0?issues.map((i,x)=><Info key={x} type="warn">{i}</Info>):<Info type="success">Konfiguration ist valide.</Info>}</Sec></>);};
