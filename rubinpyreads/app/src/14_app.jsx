function App() {
  // ── Session-Persistenz: State überlebt Browser-Refresh ──
  const _ssKey = "rubin_app_state";
  const _ssLoad = () => { try { return JSON.parse(sessionStorage.getItem(_ssKey)) || {}; } catch { return {}; } };
  const _ssSave = (patch) => {
    try {
      const cur = _ssLoad();
      sessionStorage.setItem(_ssKey, JSON.stringify({...cur, ...patch}));
    } catch {}
  };
  const _ss0 = _ssLoad();

  const [pg,setPg] = useState(_ss0.pg || "overview");

  // Scroll-to-top bei Seitenwechsel
  useEffect(() => {
    // Robust scroll-to-top: verschiedene Browser und Hosting-Umgebungen
    // (Iframe in Domino, verschiedene Scroll-Container) behandeln.
    window.scrollTo(0, 0);
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
    // Fallback: nach kurzer Verzögerung nochmal scrollen, falls der
    // neue Seiteninhalt das Layout erst nach dem Render aufbaut.
    const t = setTimeout(() => {
      window.scrollTo(0, 0);
      document.documentElement.scrollTop = 0;
      document.body.scrollTop = 0;
    }, 50);
    return () => clearTimeout(t);
  }, [pg]);
  const [sp,setSp] = useState(_ss0.sp || {lgbm:{},catboost:{}});
  const [activeBase,setActiveBase] = useState(_ss0.activeBase || null);
  const [activeAddons,setActiveAddons] = useState(new Set(_ss0.activeAddons || []));
  const [spFmt,setSpFmt] = useState(_ss0.spFmt || {lgbm:{},catboost:{}});
  const [analysisRunning,setAnalysisRunning] = useState(false);
  const [analysisDone,setAnalysisDone] = useState(false);
  const [resetKey,setResetKey] = useState(0);
  const [serverOk,setServerOk] = useState(null); // null=checking, true=ok, false=error
  const [serverError,setServerError] = useState("");
  const [sysInfo,setSysInfo] = useState(null); // {ram:{percent,used_mb,total_mb}, cpu:{percent,cores}, process:{status,pid,step,percent}}

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const r = await fetch("./api/health");
        if(!r.ok) throw new Error("HTTP " + r.status);
        const d = await r.json();
        if(mounted) {
          setServerOk(true); setServerError("");
          if(d.ram || d.process || d.cpu) setSysInfo({ram: d.ram || {}, process: d.process || {}, cpu: d.cpu || {}});
        }
      } catch(e) {
        if(mounted) { setServerOk(false); setServerError(e.message || "Nicht erreichbar"); setSysInfo(null); }
      }
    };
    check();
    const iv = setInterval(check, 15000);
    return () => { mounted = false; clearInterval(iv); };
  }, []);
  const INIT_DP = {files:[""],evalFiles:[],featurePath:"",targets:[""],treatment:"",scoreName:"",multiOpt:"merge",controlFileIndex:0,fillNa:"(keine)",binaryTarget:false,dedup:false,dedupCol:"",scoreAsFeature:false,outputPath:"runs/data",delimiter:",",detectedCols:null,featureSelection:{},treatValues:[],treatMap:{},colTypes:{},targetValues:[],nanCols:[],colStats:{},dictResult:null,chunksize:300000,sasEncoding:"utf-8",dpMlflow:true,dpMlflowExp:"",balanceTreat:false,evalFileIdx:null,nRows:0};
  const INIT_CFG = {studyType:"rct",expName:"rubin",seed:42,tuningSeed:18,parallelLevel:3,workDir:null,x_file:"",t_file:"",y_file:"",s_file:"",treatmentType:"binary",refGroup:0,hasNaN:false,nanCols:[],validateOn:"cross",cvSplits:5,downsample:false,dfFrac:0.1,reduceMem:true,fsEnabled:false,fsMethods:["catboost_importance"],fsCorrThresh:0.9,fsMaxFeatures:77,outputDir:"",models:["NonParamDML","DRLearner","SLearner","TLearner","XLearner","ParamDML","CausalForestDML","CausalForest"],baseLearner:"catboost",tuningEnabled:false,tuningTrials:50,tuningSingleFold:false,tuningModels:[],fmtEnabled:false,fmtModels:[],fmtSingleFold:false,fmtTrials:50,fmtMaxRows:0,fmtScorer:"auto",cfTune:false,cfTrials:50,cfSingleFold:false,cfScorer:"auto",cfTuneModels:[],cfOverfitPenalty:0.0,cfOverfitTolerance:0.1,selMetric:"qini",higherBetter:true,manualChamp:null,surrEnabled:false,surrMinLeaf:50,surrLeaves:31,surrDepth:0,bundleEnabled:false,bundleDir:"runs/bundles",bundleMlflow:true,explEnabled:false,explSampleSize:10000,explTopN:20,shapModels:[],shapBins:10,histScoreName:"historical_score",histScoreCol:"S",histScoreHigher:true,maxPredRows:0,tuningTimeout:0,tuningMaxRows:0,fmtTimeout:0,blFixed:{},fmtFixed:{},cfFixed:{},ensembleEnabled:true,mcIters:null,mcAgg:"mean",fmtOverfitPenalty:0.0,fmtOverfitTolerance:0.1,dmlCrossfitFolds:5,overfitPenalty:0.0,overfitTolerance:0.2};

  const [dp,setDp] = useState(_ss0.dp ? {...INIT_DP, ..._ss0.dp} : {...INIT_DP});
  const [cfg,set] = useState(_ss0.cfg ? {...INIT_CFG, ..._ss0.cfg} : {...INIT_CFG});

  // ── Session-Persistenz: State bei jeder Änderung in sessionStorage schreiben ──
  useEffect(() => { _ssSave({pg}); }, [pg]);
  useEffect(() => { _ssSave({dp}); }, [dp]);
  useEffect(() => { _ssSave({cfg}); }, [cfg]);
  useEffect(() => { _ssSave({sp}); }, [sp]);
  useEffect(() => { _ssSave({spFmt}); }, [spFmt]);
  useEffect(() => { _ssSave({activeBase}); }, [activeBase]);
  useEffect(() => { _ssSave({activeAddons: [...activeAddons]}); }, [activeAddons]);

  // ── Datenstatistiken (aus DataPrep detect-columns) für Single-Fold-Warnung ──
  const dataStats = useMemo(() => {
    const cs = dp.colStats; if (!cs || !dp.treatment) return null;
    const treatCol = dp.treatment.toUpperCase();
    const targetCol = ((dp.targets||[])[0]||"").toUpperCase();
    const tMean = cs[treatCol]?.mean; const yMean = cs[targetCol]?.mean;
    const n = dp.nRows || 0;
    if (!n || tMean == null || yMean == null) return null;
    const nTreated = Math.round(n * tMean);
    const nPositive = Math.round(n * yMean);
    return { n, nTreated, nPositive, minority: Math.min(nTreated, nPositive), treatRate: tMean, outcomeRate: yMean };
  }, [dp.colStats, dp.treatment, dp.targets, dp.nRows]);

  const resetApp = () => {
    if(!window.confirm("Gesamte App zurücksetzen? Alle Einstellungen und Daten gehen verloren.")) return;
    setDp({...INIT_DP});
    set({...INIT_CFG});
    setSp({lgbm:{},catboost:{}});
    setSpFmt({lgbm:{},catboost:{}});
    setActiveBase(null);
    setActiveAddons(new Set());
    setAnalysisRunning(false);
    setAnalysisDone(false);
    setResetKey(k => k + 1);
    setPg("overview");
    try { sessionStorage.removeItem(_ssKey); } catch {}
    // Server-State zurücksetzen (beendet ggf. laufenden Prozess)
    fetch("./api/reset", {method:"POST"}).catch(()=>{});
  };

  // ── Sidebar Status Computation ──
  const pageStatus = (() => {
    const s = {};
    // Datenvorbereitung (optional)
    if(dp.detectedCols) s.dataprep = {st:"done",detail:"Spalten erkannt"};
    else if(dp.files?.some(f=>f.trim())) s.dataprep = {st:"active",detail:"Dateien eingetragen"};
    else s.dataprep = {st:"open",detail:"Optional"};

    // Daten
    const hasFiles = !!(cfg.x_file && cfg.t_file && cfg.y_file);
    if(!hasFiles) s.datafiles = {st:"required",detail:"X/T/Y-Dateien fehlen"};
    else { const fParts=[cfg.x_file?"X":"",cfg.t_file?"T":"",cfg.y_file?"Y":"",cfg.s_file?"S":""].filter(Boolean); s.datafiles = {st:"done",detail:fParts.join(", ")+" geladen"}; }

    // Vorlage
    const hasModels = (cfg.models||[]).length > 0;
    const btConflict = cfg.treatmentType==="multi" && (cfg.models||[]).some(m=>btOnly.has(m));
    if(!hasModels) s.template = {st:"required",detail:"Keine Modelle ausgewählt"};
    else if(btConflict) s.template = {st:"required",detail:"MT-inkompatible Modelle"};
    else { const study=(cfg.studyType||"rct")==="rct"?"RCT":"Observational"; s.template = {st:"done",detail:(cfg.models||[]).length+" Modelle, "+study+", "+cfg.treatmentType.toUpperCase()}; }

    // Konfiguration
    if(!cfg.expName) s.config = {st:"required",detail:"Experiment-Name fehlt"};
    else {
      const valLabel = cfg.validateOn==="cross" ? cfg.cvSplits+"F Cross-Val" : cfg.validateOn==="external" ? "External Eval" : "TMES";
      const parts = [valLabel];
      if(cfg.downsample) parts.push(Math.round(cfg.dfFrac*100)+"%");
      if(cfg.fsEnabled) parts.push("FS");
      parts.push("Seed "+cfg.seed+"/"+cfg.tuningSeed);
      s.config = {st:"done",detail:parts.join(", ")};
    }

    // Modelle & Tuning
    const nanBlocked = cfg.hasNaN ? new Set(["CausalForestDML","CausalForest"]) : new Set();
    const effectiveModels = (cfg.models||[]).filter(m => !nanBlocked.has(m));
    if(!effectiveModels.length) s.models = {st:"required",detail:"Keine Modelle"};
    else {
      const bl = (cfg.baseLearner||"catboost")==="both"?"CB+LGBM":(cfg.baseLearner||"catboost")==="catboost"?"CB":"LGBM";
      const parts = [effectiveModels.length+"M", bl];
      if(cfg.tuningEnabled) parts.push(cfg.tuningTrials+"T");
      if(cfg.fmtEnabled) parts.push("FMT "+cfg.fmtTrials+"T");
      if(!cfg.tuningEnabled && !cfg.fmtEnabled) parts.push("kein Tuning");
      s.models = {st:"done",detail:parts.join(", ")};
    }

    // Auswahl & Export
    const selParts = [cfg.selMetric];
    if(cfg.manualChamp) selParts.push("Champ: "+cfg.manualChamp);
    if(cfg.surrEnabled) selParts.push("Surrogate");
    if(cfg.bundleEnabled) selParts.push("Bundle");
    s.selection = {st:"done",detail:selParts.join(", ")};

    // Explainability
    if(!cfg.explEnabled) s.explain = {st:"open",detail:"Deaktiviert"};
    else {
      s.explain = {st:"done",detail:"SHAP, "+cfg.explSampleSize+" Samples, Champion"};
    }

    // Config-Vorschau
    const issues = validate(cfg);
    if(issues.length > 0) s.preview = {st:"required",detail:issues.length+" Problem"+(issues.length>1?"e":"")};
    else s.preview = {st:"done",detail:"Valide"};

    // Analyse starten
    if(analysisDone) s.run = {st:"done",detail:"Analyse abgeschlossen"};
    else if(analysisRunning) s.run = {st:"active",detail:"Analyse läuft …"};
    else if(issues.length > 0) s.run = {st:"required",detail:"Nicht startbar"};
    else s.run = {st:"open",detail:"Bereit"};

    return s;
  })();

  const statusColor = {done:"#4ade80",active:"#60a5fa",open:"rgba(255,255,255,0.3)",required:"#ff6b6b"};

  const nRequired = Object.values(pageStatus).filter(s=>s.st==="required").length;
  // ── Geschätzte Modell-Fits (vollständig) ──
  const estimateFits = () => {
    const m = cfg.models || []; if (!m.length) return 0;
    const outerCv = cfg.cvSplits || 5;     // Äußere Cross-Predictions
    const innerCv = cfg.dmlCrossfitFolds || 5; // Innere CV (BLT, FMT, DML — synchronisiert)
    const mc = cfg.mcIters || 1;
    const K = 2; // Binary Treatment
    const bundle = !!cfg.bundleEnabled;
    let f = 0;

    // Hilfsfunktion: interne Fits pro model.fit() Aufruf
    // DML/DR: internes Cross-Fitting (Nuisance) + model_final
    // mc_iters wiederholt nur Nuisance, nicht model_final
    const fitsPerFit = (x) => {
      if (["NonParamDML","ParamDML","CausalForestDML"].includes(x)) return mc*innerCv*2+1;
      if (x==="DRLearner") return mc*innerCv*2+1; // propensity + regression pro Fold
      if (x==="TLearner") return K;
      if (x==="XLearner") return 1+2*K;
      return 1; // SLearner, CausalForest
    };

    // 1. Feature-Selektion
    if (cfg.fsEnabled) (cfg.fsMethods||[]).forEach(x => { if (x==="catboost_importance"||x==="lgbm_importance") f++; if (x==="causal_forest") f++; });

    // 2. BL-Tuning (Signatur-getrennte Tasks)
    const bothMult = (cfg.baseLearner||"catboost") === "both" ? 2 : 1;
    if (cfg.tuningEnabled) {
      const bltM = (cfg.tuningModels||[]).length > 0 ? cfg.tuningModels : m;
      const _isRct = (cfg.studyType||"rct") === "rct";
      const tr = (cfg.tuningTrials||50) * bothMult, tc = cfg.tuningSingleFold ? 1 : innerCv;
      const propTr = (_isRct ? Math.min(cfg.tuningTrials||50, 20) : (cfg.tuningTrials||50)) * bothMult;
      const hasDml = bltM.some(x=>["NonParamDML","ParamDML","CausalForestDML"].includes(x));
      const hasDr = bltM.includes("DRLearner"), hasSl = bltM.includes("SLearner");
      const hasTl = bltM.includes("TLearner"), hasXl = bltM.includes("XLearner");
      // Scopes sind immer getrennt (kein Scope-Merge, train_subsample_ratio verschieden)
      // Task 1: Outcome (Classifier, DML model_y)
      if (hasDml) f += tr * tc;
      // Task 2+3: Propensity — getrennt: DML/DR (scope=all) + XL (scope=all_direct)
      if (hasDml||hasDr) f += propTr * tc;
      if (hasXl) f += propTr * tc;
      // Task 4+5: Outcome Regression — getrennt: DR (scope=all) + SL (scope=all_direct)
      if (hasDr) f += tr * tc;
      if (hasSl) f += tr * tc;
      // Task 6: Grouped Outcome Regression (TL + XL models)
      if (hasTl||hasXl) f += tr * tc * K;
      // Task 7: Pseudo-Effekt (XL cate_models): Pseudo-Outcome-Nuisance (μ₀,μ₁) wird
      // EINMALIG vor den Trials berechnet (innerCv Folds × 2 Modelle: Control + Treated) —
      // sie hängt nur von den getunten Outcome-Nuisances ab, nicht von den Trial-Params.
      // Der seltene Degenerations-Fallback ersetzt übersprungene Folds (kein additiver Term).
      // Pro Trial laufen nur die K CATE-Regressoren über tc Folds.
      if (hasXl) f += tr * (tc * K) + (innerCv * 2);
    }

    // 3. FMT (Final-Model-Tuning) — cache_values: Nuisance EINMALIG + RScorer + Trials nur model_final
    const fmtFitsPerDmlFit = mc * innerCv * 2 + 1;   // NonParamDML: model_y + model_t + model_final
    const fmtFitsPerDrFit = mc * innerCv * 2 + 1;   // DRLearner: propensity + regression + model_final
    const _fmtScRes = cfg.treatmentType==="multi" ? (((cfg.fmtScorer||"auto")!=="auto"&&cfg.fmtScorer!=="qini") ? cfg.fmtScorer : "rscore") : ((cfg.fmtScorer||"auto")==="auto" ? ((cfg.studyType||"rct")==="rct"?"qini":"rscore") : cfg.fmtScorer);
    const fmtScorerFitsPerFold = _fmtScRes==="qini" ? 0 : 4;  // RScorer: cv=2 × (model_y + model_t); QiniScorer: 0
    if (cfg.fmtEnabled) {
      const fmtTr = (cfg.fmtTrials||50) * bothMult;
      const fmtOuterFolds = cfg.fmtSingleFold ? 1 : outerCv;
      (cfg.fmtModels||[]).forEach(x => {
        if (x==="NonParamDML") {
          // Pre-fit: fmtOuterFolds × (Nuisance + RScorer), Trials: fmtTr × fmtOuterFolds × 1 (model_final)
          f += fmtOuterFolds * (fmtFitsPerDmlFit + fmtScorerFitsPerFold) + fmtTr * fmtOuterFolds;
        } else if (x==="DRLearner") {
          f += fmtOuterFolds * (fmtFitsPerDrFit + fmtScorerFitsPerFold) + fmtTr * fmtOuterFolds;
        }
      });
    }

    // 4. CF Tuning — cache_values: Nuisance EINMALIG + RScorer + Trials nur Forest/GRF refit
    if (cfg.cfTune) {
      const cfTrials = cfg.cfTrials || 50;
      const cfFolds = cfg.cfSingleFold ? 1 : innerCv;
      const _cfScRes = cfg.treatmentType==="multi" ? (((cfg.cfScorer||"auto")!=="auto"&&cfg.cfScorer!=="qini") ? cfg.cfScorer : "rscore") : ((cfg.cfScorer||"auto")==="auto" ? ((cfg.studyType||"rct")==="rct"?"qini":"rscore") : cfg.cfScorer);
      const cfScorerFitsPerFold = _cfScRes==="qini" ? 0 : 4; // RScorer: cv=2 × (model_y + model_t); QiniScorer: 0
      // CausalForestDML: Pre-fit cfFolds × (Nuisance + RScorer) + cfTrials × cfFolds × 1 (forest refit)
      if ((cfg.cfTuneModels||[]).includes("CausalForestDML") && m.includes("CausalForestDML")) f += cfFolds * (mc*innerCv*2+1+cfScorerFitsPerFold) + cfTrials * cfFolds;
      // CausalForest: RScorer-Setup (cfFolds × 4) + cfTrials × cfFolds × 1 Forest
      if ((cfg.cfTuneModels||[]).includes("CausalForest") && m.includes("CausalForest")) f += cfTrials * cfFolds + cfFolds * cfScorerFitsPerFold;
    }

    // 5. Training
    // Cross/TMES: K-Fold CV + Train-Predictions-Fit (+1 bei SHAP für keep-fold)
    // External: nur 1 Fit auf Trainingsdaten, Predictions via predict() auf Holdout
    // TMES: normales K-Fold CV (Training auf allen Daten), Evaluation nur auf Mask-Subset
    const isHoldoutMode = cfg.validateOn === "external";
    const isExternal = isHoldoutMode;
    m.forEach(x => {
      const pf = fitsPerFit(x);
      if (isExternal) {
        f += pf;                                         // 1 Fit auf Trainingsdaten
      } else {
        f += pf * outerCv;                                    // Externe CV-Folds
        f += pf;                                          // Train-Predictions (full-data fit)
        // keep_last_fold_model bei SHAP: kein extra Fit, nur Referenz auf letztes Fold-Modell
      }
    });

    // 6. DRTester Nuisance (val-only, beide Modi)
    // CustomDRTester.fit_nuisance berechnet nur noch dr_val_ — der dr_train_-Pfad
    // wurde als ungenutzt/redundant entfernt. Daher in BEIDEN Modi nur die Val-
    // Nuisance: ~2 Modelle (regression + propensity) × cv=5 ≈ 10 Fits, einmalig
    // (shared pre-fit, über alle Modelle wiederverwendet).
    f += 10;

    // 7. Surrogate tree (Cross-Validated + Final)
    if (cfg.surrEnabled) f += outerCv + 1;

    // 8. Bundle: alle Modelle brauchen Refit auf vollen Daten
    if (bundle) {
      m.forEach(x => { f += fitsPerFit(x); });
      if (cfg.surrEnabled) f += 1;
    }

    return f;
  };
  const totalFits = estimateFits();

  return (
    <div style={{display:"flex",minHeight:"100vh",fontFamily:FONT,fontSize:15,color:"#24292f",background:"#faf7f8"}}>
      <nav style={{width:280,minWidth:280,background:"linear-gradient(180deg,#4a080f 0%,#6B0D15 30%,#8a0f1a 100%)",color:"#fff",position:"sticky",top:0,height:"100vh",overflowY:"auto",flexShrink:0,display:"flex",flexDirection:"column",boxShadow:"0 8px 40px rgba(107,13,21,0.25), 0 2px 12px rgba(0,0,0,0.08)",borderRadius:16}}>
        <div style={{display:"flex",flexDirection:"column",alignItems:"center",padding:"26px 24px 14px"}}>
          <div style={{background:"rgba(255,255,255,0.05)",borderRadius:20,padding:"14px 22px 10px",border:"1px solid rgba(255,255,255,0.06)"}}>
            <RubinLogo size={76} light/>
            <h1 style={{fontSize:30,fontWeight:700,margin:"10px 0 0",letterSpacing:1.5,textAlign:"center"}}>rubin</h1>
          </div>
          <p style={{fontSize:11,opacity:0.4,margin:"3px 0 0",letterSpacing:0.8}}>Causal ML Framework</p>
        </div>
        <div style={{borderTop:"1px solid rgba(255,255,255,0.08)",margin:"4px 28px 10px"}}/>
        <div style={{flex:1,padding:"0 10px"}}>
          {pages.map((p,idx) => {
            const active = pg===p.key;
            const st = pageStatus[p.key];
            const isOverview = p.key==="overview";
            const iColor = isOverview ? (active?"#fff":"rgba(255,255,255,0.5)") : statusColor[st?.st||"open"];
            const prevGroup = idx > 0 ? pages[idx-1].group : null;
            const showGroup = p.group && p.group !== prevGroup;
            return (
              <div key={p.key}>
                {showGroup && <div style={{fontSize:9,fontWeight:600,textTransform:"uppercase",letterSpacing:"1px",color:"rgba(255,255,255,0.2)",padding:"8px 18px 3px",marginTop:2}}>{p.group}</div>}
                <button onClick={()=>setPg(p.key)} style={{display:"flex",alignItems:"center",gap:10,width:"100%",textAlign:"left",padding:"7px 18px",background:active?"rgba(255,255,255,0.12)":"transparent",color:active?"#fff":"rgba(255,255,255,0.6)",border:"none",borderLeft:active?"3px solid rgba(255,255,255,0.85)":"3px solid transparent",borderRadius:active?"0 6px 6px 0":"0",cursor:"pointer",fontSize:13,fontWeight:active?600:400,fontFamily:"inherit",transition:"all 0.15s",letterSpacing:0.2,marginBottom:0}}
                onMouseEnter={e=>{if(!active)e.currentTarget.style.background="rgba(255,255,255,0.05)"}}
                onMouseLeave={e=>{if(!active)e.currentTarget.style.background="transparent"}}>
                  <span style={{width:18,height:18,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,position:"relative"}}><div style={{width:7,height:7,borderRadius:"50%",background:analysisRunning&&p.key==="run"?"#D4A853":iColor,boxShadow:st?.st==="done"?"0 0 6px rgba(74,222,128,0.35)":active?"0 0 8px rgba(255,255,255,0.35)":st?.st==="required"?"0 0 6px rgba(255,107,107,0.35)":"none",transition:"all 0.3s",animation:analysisRunning&&p.key==="run"?"rubin-dot-breathe 2.4s ease-in-out infinite":"none"}}/>{analysisRunning && p.key==="run" && <svg width="22" height="22" viewBox="0 0 22 22" style={{position:"absolute",left:-2,top:-2,animation:"rubin-spin 3s linear infinite"}}><circle cx="11" cy="11" r="9.5" fill="none" stroke="#D4A853" strokeWidth="1.5" strokeDasharray="15 45" strokeLinecap="round"/></svg>}</span>
                  <span style={{flex:1}}>{p.label}</span>
                  {p.optional && st?.st==="open" && <span style={{fontSize:8.5,background:"rgba(255,255,255,0.1)",padding:"1px 6px",borderRadius:6,color:"rgba(255,255,255,0.35)"}}>opt</span>}
                </button>
                {st && st.st==="required" && (
                  <div style={{fontSize:9.5,color:"#ff8a8a",padding:"0 18px 3px 49px",lineHeight:1.3}}>{st.detail}</div>
                )}
                {st && st.st==="done" && !isOverview && (
                  <div style={{fontSize:9.5,color:"rgba(255,255,255,0.3)",padding:"0 18px 2px 49px",lineHeight:1.3}}>{st.detail}</div>
                )}
              </div>
            );
          })}
        </div>

        <div style={{borderTop:"1px solid rgba(255,255,255,0.08)",margin:"8px 22px 0"}}/>
        <style>{_sidebarStyles}</style>
        <div style={{padding:"8px 18px 6px"}}>
          {analysisRunning ? (
            <div style={{background:"rgba(212,168,83,0.08)",borderRadius:10,padding:"10px 10px 8px",border:"1px solid rgba(212,168,83,0.18)"}}>
              <svg viewBox="0 0 200 72" width="100%" height="72" style={{display:"block",marginBottom:6}}>
                <defs>
                  <radialGradient id="tp"><stop offset="0%" stopColor="#D4A853" stopOpacity="0.5"/><stop offset="100%" stopColor="#D4A853" stopOpacity="0"/></radialGradient>
                </defs>
                <style>{`
                  @keyframes ht-pulse { 0%,100% { r:3.5; opacity:.65 } 50% { r:5.5; opacity:1 } }
                  @keyframes ht-glow { 0%,100% { r:12; opacity:.12 } 50% { r:20; opacity:.22 } }
                  .ht-pulse { animation: ht-pulse 1.8s ease-in-out infinite; }
                  .ht-glow { animation: ht-glow 1.8s ease-in-out infinite; }
                  @keyframes hc0 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.45}90%{opacity:.35}100%{cx:84;cy:33;opacity:0} }
                  @keyframes hc1 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.4}90%{opacity:.3}100%{cx:84;cy:39;opacity:0} }
                  @keyframes hc2 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.5}90%{opacity:.4}100%{cx:84;cy:34;opacity:0} }
                  @keyframes hc3 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.35}90%{opacity:.3}100%{cx:84;cy:38;opacity:0} }
                  @keyframes hc4 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.45}90%{opacity:.35}100%{cx:84;cy:32;opacity:0} }
                  @keyframes hc5 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.4}90%{opacity:.3}100%{cx:84;cy:40;opacity:0} }
                  @keyframes hc6 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.5}90%{opacity:.4}100%{cx:84;cy:35;opacity:0} }
                  @keyframes hc7 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.35}90%{opacity:.25}100%{cx:84;cy:37;opacity:0} }
                  @keyframes hu0 { 0%{cx:96;cy:36;opacity:0;r:1.4}6%{opacity:.85;r:2.2}100%{cx:194;cy:8;opacity:0;r:1} }
                  @keyframes hu1 { 0%{cx:96;cy:36;opacity:0;r:1.4}6%{opacity:.75;r:2}100%{cx:190;cy:14;opacity:0;r:1.1} }
                  @keyframes hu2 { 0%{cx:96;cy:36;opacity:0;r:1.4}6%{opacity:.8;r:1.9}100%{cx:188;cy:20;opacity:0;r:1.2} }
                  @keyframes hu3 { 0%{cx:96;cy:36;opacity:0;r:1.4}6%{opacity:.7;r:2.1}100%{cx:192;cy:11;opacity:0;r:0.9} }
                  @keyframes hu4 { 0%{cx:96;cy:36;opacity:0;r:1.4}6%{opacity:.85;r:1.8}100%{cx:186;cy:25;opacity:0;r:1.3} }
                  @keyframes hu5 { 0%{cx:96;cy:36;opacity:0;r:1.4}6%{opacity:.65;r:2.3}100%{cx:195;cy:6;opacity:0;r:0.8} }
                  @keyframes hu6 { 0%{cx:96;cy:36;opacity:0;r:1.4}6%{opacity:.75;r:1.7}100%{cx:185;cy:28;opacity:0;r:1.1} }
                  @keyframes hd0 { 0%{cx:96;cy:36;opacity:0;r:1.3}6%{opacity:.65;r:1.9}100%{cx:194;cy:64;opacity:0;r:0.9} }
                  @keyframes hd1 { 0%{cx:96;cy:36;opacity:0;r:1.3}6%{opacity:.55;r:1.7}100%{cx:190;cy:58;opacity:0;r:1} }
                  @keyframes hd2 { 0%{cx:96;cy:36;opacity:0;r:1.3}6%{opacity:.6;r:1.8}100%{cx:188;cy:52;opacity:0;r:1.1} }
                  @keyframes hd3 { 0%{cx:96;cy:36;opacity:0;r:1.3}6%{opacity:.5;r:2}100%{cx:192;cy:61;opacity:0;r:0.8} }
                  @keyframes hd4 { 0%{cx:96;cy:36;opacity:0;r:1.3}6%{opacity:.6;r:1.6}100%{cx:186;cy:48;opacity:0;r:1.2} }
                  @keyframes hd5 { 0%{cx:96;cy:36;opacity:0;r:1.3}6%{opacity:.45;r:2.1}100%{cx:195;cy:66;opacity:0;r:0.7} }
                `}</style>
                <line x1="8" y1="36" x2="86" y2="36" stroke="rgba(255,255,255,0.12)" strokeWidth="1" strokeDasharray="4 3"/>
                <path d="M96,36 Q140,32 194,10" fill="none" stroke="rgba(74,222,128,0.08)" strokeWidth="0.7"/>
                <path d="M96,36 Q140,34 190,22" fill="none" stroke="rgba(74,222,128,0.05)" strokeWidth="0.5"/>
                <path d="M96,36 Q140,40 194,62" fill="none" stroke="rgba(248,113,113,0.12)" strokeWidth="0.7"/>
                <path d="M96,36 Q140,38 190,50" fill="none" stroke="rgba(248,113,113,0.08)" strokeWidth="0.5"/>
                <circle cx="91" cy="36" fill="url(#tp)" className="ht-glow"/>
                <circle cx="91" cy="36" fill="#D4A853" className="ht-pulse"/>
                {[
                  {a:"hc0",d:0,dur:3.4},{a:"hc1",d:0.42,dur:3.7},{a:"hc2",d:0.85,dur:3.2},
                  {a:"hc3",d:1.3,dur:3.6},{a:"hc4",d:1.7,dur:3.3},{a:"hc5",d:2.15,dur:3.8},
                  {a:"hc6",d:2.6,dur:3.1},{a:"hc7",d:3.05,dur:3.5},
                ].map((p,i) => <circle key={`c${i}`} cx="6" cy="36" r="1.4" fill="rgba(255,255,255,0.35)"
                  style={{animation:`${p.a} ${p.dur}s ease-in-out infinite`,animationDelay:`${p.d}s`}}/>)}
                {[
                  {a:"hu0",d:0,dur:3.1},{a:"hu1",d:0.45,dur:3.4},{a:"hu2",d:0.9,dur:2.9},
                  {a:"hu3",d:1.35,dur:3.3},{a:"hu4",d:1.8,dur:3.0},{a:"hu5",d:2.25,dur:3.5},
                  {a:"hu6",d:2.7,dur:2.8},
                ].map((p,i) => <circle key={`u${i}`} cx="96" cy="36" r="2" fill="#4ade80"
                  style={{animation:`${p.a} ${p.dur}s ease-out infinite`,animationDelay:`${p.d}s`}}/>)}
                {[
                  {a:"hd0",d:0.2,dur:3.3},{a:"hd1",d:0.65,dur:3.0},{a:"hd2",d:1.1,dur:3.5},
                  {a:"hd3",d:1.55,dur:2.9},{a:"hd4",d:2.0,dur:3.2},{a:"hd5",d:2.5,dur:3.4},
                ].map((p,i) => <circle key={`d${i}`} cx="96" cy="36" r="1.8" fill="#f87171"
                  style={{animation:`${p.a} ${p.dur}s ease-out infinite`,animationDelay:`${p.d}s`}}/>)}
              </svg>
              <div style={{fontSize:10.5,fontWeight:600,color:"#D4A853",textAlign:"center",letterSpacing:0.3}}>Heterogenität wird aufgedeckt …</div>
            </div>
          ) : analysisDone ? (
            <div style={{background:"rgba(74,222,128,0.1)",borderRadius:8,padding:"9px 12px",border:"1px solid rgba(74,222,128,0.15)"}}>
              <div style={{display:"flex",alignItems:"center",gap:6,fontSize:11.5,fontWeight:600,color:"#4ade80"}}><div style={{width:6,height:6,borderRadius:"50%",background:"#4ade80",boxShadow:"0 0 6px rgba(74,222,128,0.3)"}}/> Analyse abgeschlossen</div>
              <div style={{fontSize:10,color:"rgba(255,255,255,0.4)",marginTop:2}}>Ergebnisse verfügbar</div>
            </div>
          ) : nRequired > 0 ? (
            <div style={{background:"rgba(255,100,100,0.12)",borderRadius:8,padding:"9px 12px",border:"1px solid rgba(255,100,100,0.2)"}}>
              <div style={{display:"flex",alignItems:"center",gap:6,fontSize:11.5,fontWeight:600,color:"#ff6b6b"}}><div style={{width:6,height:6,borderRadius:"50%",background:"#ff6b6b",boxShadow:"0 0 6px rgba(255,107,107,0.3)"}}/> {nRequired} Schritt{nRequired>1?"e":""} offen</div>
              <div style={{fontSize:10,color:"rgba(255,255,255,0.4)",marginTop:2}}>Pflichtangaben fehlen</div>
            </div>
          ) : (
            <div style={{background:"rgba(74,222,128,0.1)",borderRadius:8,padding:"9px 12px",border:"1px solid rgba(74,222,128,0.15)"}}>
              <div style={{display:"flex",alignItems:"center",gap:6,fontSize:11.5,fontWeight:600,color:"#4ade80"}}><div style={{width:6,height:6,borderRadius:"50%",background:"#4ade80",boxShadow:"0 0 6px rgba(74,222,128,0.3)"}}/> Bereit</div>
              <div style={{fontSize:10,color:"rgba(255,255,255,0.4)",marginTop:2}}>Analyse kann gestartet werden</div>
            </div>
          )}

        </div>
        {totalFits > 0 && <div style={{padding:"4px 18px 0",display:"flex",alignItems:"center",justifyContent:"space-between",fontSize:10,color:"rgba(255,255,255,0.3)"}}>
          <span>Modell-Fits</span>
          <span style={{fontFamily:MONO,fontWeight:600,color:"rgba(255,255,255,0.5)"}}>{totalFits.toLocaleString("de-DE")}</span>
        </div>}
        <div style={{fontSize:10,opacity:0.25,textAlign:"center",padding:"10px 0 6px"}}>v1.0</div>
        <div style={{padding:"0 16px 4px"}}>
          <button onClick={resetApp} style={{display:"flex",alignItems:"center",justifyContent:"center",gap:6,width:"100%",padding:"7px 0",background:"rgba(255,255,255,0.06)",border:"1px solid rgba(255,255,255,0.1)",borderRadius:6,color:"rgba(255,255,255,0.45)",fontSize:10.5,fontWeight:500,cursor:"pointer",fontFamily:"inherit",transition:"all 0.15s",letterSpacing:0.3}}
            onMouseEnter={e=>{e.currentTarget.style.background="rgba(255,100,100,0.15)";e.currentTarget.style.color="rgba(255,200,200,0.8)";e.currentTarget.style.borderColor="rgba(255,100,100,0.3)"}}
            onMouseLeave={e=>{e.currentTarget.style.background="rgba(255,255,255,0.06)";e.currentTarget.style.color="rgba(255,255,255,0.45)";e.currentTarget.style.borderColor="rgba(255,255,255,0.1)"}}>
            ↺ App zurücksetzen
          </button>
        </div>
        <div style={{padding:"0 16px 14px"}}>
          {/* Server-Verbindung */}
          <div style={{display:"flex",alignItems:"center",gap:5,fontSize:10,color:serverOk===true?"rgba(74,222,128,0.7)":serverOk===false?"rgba(255,100,100,0.8)":"rgba(255,255,255,0.3)",marginBottom:6}}>
            <div style={{width:6,height:6,borderRadius:3,background:serverOk===true?"#4ade80":serverOk===false?"#ff6b6b":"#888"}}/>
            {serverOk===true?"Server verbunden":serverOk===false?"Nicht erreichbar":"Prüfe..."}
          </div>
          {serverOk===false && serverError && <div style={{fontSize:9,color:"rgba(255,100,100,0.5)",marginBottom:6}}>{serverError}</div>}
          {/* Prozess-Status */}
          {sysInfo && sysInfo.process && (()=>{const ps=sysInfo.process;const stMap={idle:{c:"rgba(255,255,255,0.3)",bg:"#888",t:"Bereit"},running:{c:"rgba(74,222,128,0.7)",bg:"#4ade80",t:"Läuft"+(ps.step?" – "+ps.step:"")},done:{c:"rgba(74,222,128,0.7)",bg:"#22c55e",t:"Fertig"},error:{c:"rgba(255,100,100,0.8)",bg:"#ff6b6b",t:"Fehler"},crashed:{c:"rgba(255,100,100,0.8)",bg:"#ff6b6b",t:"Abgestürzt"}};const s=stMap[ps.status]||stMap.idle;return(<><div style={{display:"flex",alignItems:"center",gap:5,fontSize:10,color:s.c,marginBottom:4}}><div style={{width:6,height:6,borderRadius:3,background:s.bg}}/><span style={{flex:1,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{s.t}</span>{ps.pid&&<span style={{opacity:0.4,fontSize:9}}>PID {ps.pid}</span>}</div>{ps.status==="running"&&ps.percent>0&&(<div style={{height:2,background:"rgba(255,255,255,0.1)",borderRadius:1,overflow:"hidden",marginBottom:4}}><div style={{height:"100%",width:ps.percent+"%",background:"#4ade80",borderRadius:1,transition:"width 0.5s"}}/></div>)}{(ps.status==="error"||ps.status==="crashed")&&(<button onClick={async()=>{try{await fetch("./api/restart-process",{method:"POST"});setSysInfo(p=>p?{...p,process:{...p.process,status:"idle",pid:null}}:p)}catch(e){}}} style={{display:"block",width:"100%",padding:"4px 0",marginBottom:4,background:"rgba(255,100,100,0.15)",border:"1px solid rgba(255,100,100,0.3)",borderRadius:4,color:"rgba(255,100,100,0.8)",fontSize:10,fontWeight:600,cursor:"pointer",fontFamily:"inherit"}}>Prozess neu starten</button>)}</>)})()}
          {/* CPU + RAM Auslastung */}
          {sysInfo && sysInfo.ram && sysInfo.ram.total_mb>0 && (()=>{const r=sysInfo.ram;const rPct=r.percent||0;const rColor=rPct>85?"#ff6b6b":rPct>70?"#f59e0b":"rgba(74,222,128,0.6)";const c=sysInfo.cpu||{};const cPct=c.percent||0;const cColor=cPct>85?"#ff6b6b":cPct>70?"#f59e0b":"rgba(74,222,128,0.6)";return(<div style={{marginTop:2,display:"flex",gap:10}}><div style={{flex:1}}><div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"rgba(255,255,255,0.35)",marginBottom:2}}><span>CPU</span><span>{cPct}%{c.cores?" · "+Math.round(cPct*c.cores/100)+"/"+c.cores+"C":""}</span></div><div style={{height:3,background:"rgba(255,255,255,0.08)",borderRadius:2,overflow:"hidden"}}><div style={{height:"100%",width:cPct+"%",background:cColor,borderRadius:2,transition:"width 1s"}}/></div></div><div style={{flex:1}}><div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"rgba(255,255,255,0.35)",marginBottom:2}}><span>RAM</span><span>{rPct}%{r.used_mb?" · "+(r.used_mb/1024).toFixed(1)+"/"+(r.total_mb/1024).toFixed(1)+" GB":""}</span></div><div style={{height:3,background:"rgba(255,255,255,0.08)",borderRadius:2,overflow:"hidden"}}><div style={{height:"100%",width:rPct+"%",background:rColor,borderRadius:2,transition:"width 1s"}}/></div></div></div>)})()}
        </div>
      </nav>
      <main style={{flex:1,maxWidth:960,padding:"36px 52px 80px",margin:"0 auto"}}>
        {pg==="overview" && <POverview setPg={setPg}/>}
        <div key={"dp-"+resetKey} style={{display: pg==="dataprep" ? "block" : "none"}}><PDataPrep dp={dp} setDp={setDp} cfg={cfg} setCfg={set} setPg={setPg}/></div>
        {pg==="datafiles" && <PData cfg={cfg} set={set} setCfg={set} activeBase={activeBase} setActiveBase={setActiveBase} activeAddons={activeAddons} setActiveAddons={setActiveAddons} setSp={setSp} setSpFmt={setSpFmt} view="files" sysInfo={sysInfo}/>}
        {pg==="template" && <PData cfg={cfg} set={set} setCfg={set} activeBase={activeBase} setActiveBase={setActiveBase} activeAddons={activeAddons} setActiveAddons={setActiveAddons} setSp={setSp} setSpFmt={setSpFmt} view="template" sysInfo={sysInfo}/>}
        {pg==="config" && <PConfig cfg={cfg} set={set}/>}
        {pg==="models" && <PModels cfg={cfg} set={set} sp={sp} setSp={setSp} spFmt={spFmt} setSpFmt={setSpFmt} dataStats={dataStats} sysInfo={sysInfo}/>}
        {pg==="selection" && <PSelection cfg={cfg} set={set}/>}
        {pg==="explain" && <PExplain cfg={cfg} set={set}/>}
        {pg==="preview" && <PPreview cfg={cfg} sp={sp} spFmt={spFmt} totalFits={totalFits}/>}
        <div key={"run-"+resetKey} style={{display: pg==="run" ? "block" : "none"}}><PRun cfg={cfg} setPg={setPg} sp={sp} spFmt={spFmt} onRunningChange={setAnalysisRunning} onDoneChange={setAnalysisDone} totalFits={totalFits}/></div>
        <div style={{display:"flex",flexDirection:"column",alignItems:"center",color:"#ccc",fontSize:11,marginTop:24,paddingTop:16,borderTop:`1px solid ${C.borderLight}`,gap:10}}><RubinLogo size={24}/><span>rubin – Causal ML Framework</span></div>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);