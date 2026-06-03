const rolesForModel = m => {
  const n = m.toLowerCase();
  if(n==="slearner") return [{role:"overall_model",label:"Outcome-Regression (mit T)",sig:"outcome_regression__reg__with_t__all_direct__Y"}];
  if(n==="tlearner") return [{role:"models",label:"Outcome-Regression (pro Gruppe)",sig:"grouped_outcome_regression__reg__no_t__group__Y"}];
  if(n==="xlearner") return [
    {role:"models",label:"Outcome-Regression (pro Gruppe)",sig:"grouped_outcome_regression__reg__no_t__group__Y"},
    {role:"cate_models",label:"Pseudo-Effekt (Regressor)",sig:"pseudo_effect__reg__no_t__group__D"},
    {role:"propensity_model",label:"Propensity (direkt)",sig:"propensity__clf__no_t__all_direct__T"},
  ];
  if(n==="drlearner") return [
    {role:"model_propensity",label:"Propensity",sig:"propensity__clf__no_t__all__T"},
    {role:"model_regression",label:"Outcome-Klassifikation (mit T)",sig:"outcome__clf__with_t__all__Y"},
  ];
  if(["nonparamdml","paramdml","causalforestdml"].includes(n)) return [
    {role:"model_y",label:"Outcome (Classifier)",sig:"outcome__clf__no_t__all__Y"},
    {role:"model_t",label:"Propensity (Classifier)",sig:"propensity__clf__no_t__all__T"},
  ];
  return [];
};

const computeTuningPlan = (models, outerCv, innerCv) => {
  const tasks = {};
  (models || []).forEach(m => {
    rolesForModel(m).forEach(({role, label, sig}) => {
      let key = sig;
      if(!tasks[key]) tasks[key] = {key, label, sig, models: [], roles: []};
      tasks[key].models.push(m);
      tasks[key].roles.push({model: m, role});
    });
  });
  // Sortierung: logische Pipeline-Reihenfolge
  const order = {"outcome":1,"propensity":2,"outcome_regression":3,"grouped_outcome_regression":4,"pseudo_effect":5};
  return Object.values(tasks).sort((a,b) => {
    const fa = a.sig.split("__")[0], fb = b.sig.split("__")[0];
    return (order[fa]||9) - (order[fb]||9);
  });
};

const TuningPlanPreview = ({models, trials, cv, outerCv, innerCv: innerCvProp, isBoth, isRct}) => {
  const bothMultiplier = isBoth ? 2 : 1;
  const nTrialsBase = trials || 100;
  const RCT_PROP_CAP = 20;
  const nCv = cv || 5;          // Trial-CV (1 bei single_fold, sonst innerCv)
  const fullCv = innerCvProp || 5;  // Immer volle CV-Folds (für Nuisance-Berechnung)
  const K = 2; // Binary Treatment
  const plan = computeTuningPlan(models, outerCv, fullCv);
  if(plan.length === 0) return <Info type="warn">Keine Modelle ausgewählt – kein Tuning noetig.</Info>;

  // Trials pro Task: bei RCT werden Propensity-Tasks auf 20 gecappt (vor both-Verdopplung)
  const trialsForTask = (sig) => {
    const fam = sig.split("__")[0];
    const base = (isRct && fam === "propensity") ? Math.min(nTrialsBase, RCT_PROP_CAP) : nTrialsBase;
    return base * bothMultiplier;
  };

  // Fits pro Trial pro Task-Typ berechnen
  const fitsForTask = (sig) => {
    const fam = sig.split("__")[0];
    if (fam==="grouped_outcome_regression") return nCv * K;
    // Pseudo-Effekt: nur die CATE-Regressoren laufen PRO TRIAL (nCv × K).
    // Die Pseudo-Outcome-Nuisance wird EINMALIG vor den Trials berechnet (s. pseudoNuisanceOnce).
    if (fam==="pseudo_effect") return nCv * K;
    return nCv;
  };

  // Pseudo-Outcome-Nuisance (μ₀, μ₁) wird je Pseudo-Effekt-Task EINMALIG vor den Trials
  // berechnet (fullCv×2 OOF + bis zu 2 Fallback) — nicht pro Trial, da trial-unabhängig.
  const pseudoNuisanceOnce = plan.filter(t => t.sig.split("__")[0]==="pseudo_effect").length * (fullCv * 2 + 2);
  const totalFits = plan.reduce((a,t) => a + trialsForTask(t.sig) * fitsForTask(t.sig), 0) + pseudoNuisanceOnce;
  const hasPropCap = isRct && nTrialsBase > RCT_PROP_CAP && plan.some(t => t.sig.split("__")[0] === "propensity");

  const sigColors = {"outcome":"#9B111E","outcome_regression":"#d97706","propensity":"#D4A853","grouped_outcome_regression":"#7c3aed","pseudo_effect":"#059669"};
  const sigLabel = sig => {
    const fam = sig.split("__")[0];
    const isDirect = sig.includes("all_direct");
    if(fam==="outcome_regression") return isDirect ? "Outcome (Reg., direkt)" : "Outcome (Reg., DML/DR)";
    if(fam==="outcome") return "Outcome (Classifier)";
    if(fam==="propensity") return isDirect ? "Propensity (direkt)" : "Propensity (DML/DR)";
    if(fam==="grouped_outcome_regression") return "Grouped Outcome (Reg.)";
    if(fam==="pseudo_effect") return "Pseudo-Effekt (XLearner)";
    return sig;
  };
  const fitsDetail = (sig) => {
    const fam = sig.split("__")[0];
    const nt = trialsForTask(sig);
    if (fam==="grouped_outcome_regression") return nt+"T × "+nCv+"F × "+K+"G";
    if (fam==="pseudo_effect") return nt+"T × "+nCv+"F×"+K+"G cate + "+(fullCv*2+2)+" nuis (einmalig)";
    return nt+"T × "+nCv+"F";
  };

  return (
    <div>
      <div style={{display:"flex",gap:10,marginBottom:12,flexWrap:"wrap"}}>
        <div style={{padding:"8px 16px",background:"linear-gradient(135deg,#6B0D15,#9B111E)",borderRadius:8,color:"#fff"}}>
          <div style={{fontSize:22,fontWeight:700}}>{plan.length}</div>
          <div style={{fontSize:10,textTransform:"uppercase",letterSpacing:0.5,opacity:0.7}}>Optuna-Studies</div>
        </div>
        <div style={{padding:"8px 16px",background:"#FDF2F3",borderRadius:8,border:"1px solid "+C.border}}>
          <div style={{fontSize:22,fontWeight:700,color:"#C4343F"}}>{totalFits.toLocaleString()}</div>
          <div style={{fontSize:10,textTransform:"uppercase",letterSpacing:0.5,color:"#888"}}>Base-Learner-Fits</div>
        </div>
        <div style={{padding:"8px 16px",background:"#faf6f6",borderRadius:8,border:"1px solid #ede6e7"}}>
          <div style={{fontSize:22,fontWeight:700,color:"#333"}}>{(models||[]).length}</div>
          <div style={{fontSize:10,textTransform:"uppercase",letterSpacing:0.5,color:"#888"}}>Modelle</div>
        </div>
      </div>
      <div style={{fontSize:10.5,color:"#888",marginBottom:10,lineHeight:1.5}}>
        Jeder Trial fittet einen einzelnen Base Learner (kein internes Cross-Fitting). T = Trials, F = CV-Folds, G = Treatment-Gruppen. mc_iters wirkt hier nicht — betrifft nur das finale Training.
        {isBoth && <span style={{display:"block",marginTop:6,color:"#7a5a00",background:"#fffbeb",padding:"4px 8px",borderRadius:6,border:"1px solid #e8d49c",fontSize:10.5}}><strong>CatBoost &amp; LGBM aktiv:</strong> Trials sind verdoppelt ({nTrialsBase}T × 2 = {nTrialsBase*2}T), damit beide Learner-Familien ausreichend Budget bekommen.</span>}
        {hasPropCap && <span style={{display:"block",marginTop:6,color:"#6366f1",background:"#f6f7ff",padding:"4px 8px",borderRadius:6,border:"1px solid #c7d2fe",fontSize:10.5}}><strong>RCT-Modus:</strong> Propensity-Tasks sind auf {RCT_PROP_CAP}{isBoth ? ` × 2 = ${RCT_PROP_CAP*2}` : ""} Trials begrenzt (bei randomisiertem Treatment reicht ein Diagnose-Check statt vollem Tuning).</span>}
      </div>
      <div style={{border:"1px solid #ede6e7",borderRadius:8,overflow:"hidden"}}>
        <div style={{display:"grid",gridTemplateColumns:"36px 1fr 1fr 1fr 70px",fontSize:11,fontWeight:600,padding:"8px 14px",background:"#6B0D15",color:"#fff"}}>
          <span>#</span><span>Tuning-Task</span><span>Genutzt von</span><span>Berechnung</span><span>Fits</span>
        </div>
        {plan.map((t,i) => {
          const family = t.sig.split("__")[0];
          const col = sigColors[family] || "#888";
          const shared = t.roles.length > 1;
          const rowFits = trialsForTask(t.sig) * fitsForTask(t.sig);
          const isPropCapped = hasPropCap && t.sig.split("__")[0] === "propensity";
          return (
            <div key={t.key} style={{display:"grid",gridTemplateColumns:"36px 1fr 1fr 1fr 70px",padding:"7px 14px",borderBottom:"1px solid #f5f0f0",background:i%2===0?"#fff":"#fdfbfb",alignItems:"center",fontSize:12}}>
              <span style={{color:"#bbb",fontFamily:MONO,fontWeight:600}}>{i+1}</span>
              <div>
                <div style={{display:"flex",alignItems:"center",gap:6,flexWrap:"wrap"}}>
                  <span style={{display:"inline-flex",alignItems:"center",gap:6}}>
                    <span style={{display:"inline-block",width:8,height:8,borderRadius:4,background:col}}/>
                    <span style={{fontWeight:500}}>{sigLabel(t.sig)}</span>
                  </span>
                  {shared && <span style={{fontSize:10,color:"#059669",fontWeight:600}}>shared</span>}
                  {t.sig.includes("__all__") && <span style={{fontSize:9,color:"#6b7280",fontWeight:500,background:"#f3f4f6",padding:"1px 6px",borderRadius:8,border:"1px solid #d1d5db"}}>Train: {Math.round((1-(1/(fullCv||5)))*100)}%</span>}
                  {isPropCapped && <span style={{fontSize:9,color:"#6366f1",fontWeight:600,background:"#f6f7ff",padding:"1px 6px",borderRadius:8,border:"1px solid #c7d2fe"}}>RCT: {trialsForTask(t.sig)}T</span>}
                </div>
              </div>
              <div style={{display:"flex",gap:3,flexWrap:"wrap"}}>
                {t.roles.map((r,j) => (
                  <span key={j} style={{fontSize:10.5,background:"#f5f0f0",padding:"1px 7px",borderRadius:10,color:"#555"}}>
                    {r.model}<span style={{color:"#ccc"}}>.</span>{r.role}
                  </span>
                ))}
              </div>
              <span style={{fontSize:10.5,color:"#888",fontFamily:MONO}}>{fitsDetail(t.sig)}</span>
              <span style={{fontFamily:MONO,color:"#C4343F",fontWeight:600}}>{rowFits.toLocaleString()}</span>
            </div>
          );
        })}
      </div>
      {plan.length < (models||[]).reduce((a,m)=>a+rolesForModel(m).length,0) && (
        <div style={{fontSize:11.5,color:"#888",marginTop:8,fontStyle:"italic"}}>
          Tasks mit gleicher Signatur werden geteilt – {(models||[]).reduce((a,m)=>a+rolesForModel(m).length,0) - plan.length} Tuning-Laeufe eingespart.
        </div>
      )}
      {plan.some(t=>t.sig.includes("__all__")) && plan.some(t=>t.sig.includes("__all_direct__")) && (
        <div style={{fontSize:10.5,color:"#6b7280",marginTop:6,lineHeight:1.4}}>
          DML/DR-Nuisance-Tasks <span style={{background:"#f3f4f6",padding:"1px 5px",borderRadius:4,fontSize:9.5}}>Train: {Math.round((1-(1/(fullCv||5)))*100)}%</span> subsamplen die Trainingsmenge, um das DML-interne Cross-Fitting zu simulieren. Meta-Learner-Tasks trainieren auf der vollen Fold-Trainingsmenge.
        </div>
      )}
      {plan.some(t=>t.sig.split("__")[0]==="pseudo_effect") && (
        <div style={{fontSize:10.5,color:"#6b7280",marginTop:6,lineHeight:1.4}}>
          <strong style={{color:"#374151"}}>Pseudo-Effekt:</strong> Die Nuisance-Modelle (μ₀, μ₁) mit den <em>getunten</em> Params aus dem Grouped-Outcome-Task werden <em>einmalig vor den Trials</em> berechnet ({fullCv}CV × 2 Gruppen + 2 Fallback = {fullCv*2+2} Fits) — sie hängen nicht von den Trial-Params der CATE-Regressoren ab. Anschließend werden pro Trial nur die CATE-Regressoren auf den resultierenden Pseudo-Outcomes evaluiert ({nCv}F × {K}G = {nCv*K} Fits).
        </div>
      )}
    </div>
  );
};

// ── Final-Model Tuning Plan ──
const FinalTuningPlanPreview = ({models, fmtEnabled, fmtModels, fmtSingleFold, fmtTrials, fmtCv, outerCv: outerCvProp, mcIters, isBoth}) => {
  const bothMultiplier = isBoth ? 2 : 1;
  const nTrials = (fmtTrials || 50) * bothMultiplier;
  const internalCv = fmtCv || 5;  // Innere CV (Default=5, synchronisiert mit BLT und DML)
  const outerCv = outerCvProp || 5; // Äußere Score-Folds (= cross_validation_splits)
  const mc = mcIters || 1;
  const eligible = (models||[]).filter(m => ["NonParamDML","DRLearner"].includes(m) && (fmtModels === undefined || fmtModels === null || fmtModels.includes(m)));

  // cache_values-Architektur: Nuisance EINMALIG pro äußerem Fold, Trials nur model_final
  const nuisanceFitsPerDmlFold = mc * internalCv * 2 + 1; // model_y + model_t + initial model_final
  const nuisanceFitsPerDrFold = mc * internalCv * 2 + 1; // propensity + regression + initial model_final
  const scorerFitsPerFold = 2 * 2; // RScorer: cv=2 × (model_y + model_t)
  const outerFolds = fmtSingleFold ? 1 : outerCv;

  const rows = [];
  eligible.forEach(m => {
    if(!fmtEnabled) return;
    const mn = m.toLowerCase();
    const isDr = mn === "drlearner";
    const nuisancePerFold = isDr ? nuisanceFitsPerDrFold : nuisanceFitsPerDmlFold;
    // Pre-fit (einmalig): Nuisance + RScorer; Trials × Folds × 1 refit_final
    const preFit = outerFolds * (nuisancePerFold + scorerFitsPerFold);
    const trialFits = nTrials * outerFolds;
    const total = preFit + trialFits;
    const modelLabel = isDr ? "DRLearner" : "NonParamDML";
    const nuisanceDesc = isDr
      ? `${internalCv}×propensity + ${internalCv}×regression + 1×final`
      : `${internalCv}×model_y + ${internalCv}×model_t + 1×final`;
    rows.push({
      model: m,
      method: fmtSingleFold ? "OOF 1-Fold" : `OOF ${outerFolds}-Fold CV`,
      trials: nTrials,
      totalFits: total,
      detail: `${outerFolds}F×(${nuisancePerFold}+${scorerFitsPerFold}) + ${nTrials}T×${outerFolds}F`,
      note: `Pre-fit: ${outerFolds} Fold(s) × (Nuisance ${nuisanceDesc} + RScorer cv=2). Pro Trial: ${outerFolds}× refit_final() + scorer.score(est).`
    });
  });

  const totalFits = rows.reduce((a,r)=>a+r.totalFits,0);

  if(rows.length === 0) {
    const noEligible = (models||[]).filter(m=>["NonParamDML","DRLearner"].includes(m)).length === 0;
    return <Info type="warn">{noEligible ? "Kein Modell mit R-Score-Unterstuetzung ausgewaehlt (nur NonParamDML, DRLearner)." : "Final-Model-Tuning ist deaktiviert."}</Info>;
  }

  return (
    <div>
      <div style={{display:"flex",gap:10,marginBottom:12,flexWrap:"wrap"}}>
        <div style={{padding:"8px 16px",background:"linear-gradient(135deg,#b8860b,#D4A853)",borderRadius:8,color:"#fff"}}>
          <div style={{fontSize:22,fontWeight:700}}>{rows.length}</div>
          <div style={{fontSize:10,textTransform:"uppercase",letterSpacing:0.5,opacity:0.7}}>Optuna-Studies</div>
        </div>
        <div style={{padding:"8px 16px",background:"#fffbeb",borderRadius:8,border:"1px solid #D4A853"}}>
          <div style={{fontSize:22,fontWeight:700,color:"#b8860b"}}>{totalFits.toLocaleString()}</div>
          <div style={{fontSize:10,textTransform:"uppercase",letterSpacing:0.5,color:"#888"}}>Modell-Fits</div>
        </div>
        <div style={{padding:"8px 16px",background:"#faf7f7",borderRadius:8,border:"1px solid #ede6e7"}}>
          <div style={{fontSize:22,fontWeight:700,color:"#333"}}>{nTrials}</div>
          <div style={{fontSize:10,textTransform:"uppercase",letterSpacing:0.5,color:"#888"}}>Trials / Study</div>
        </div>
      </div>
      <div style={{fontSize:10.5,color:"#888",marginBottom:10,lineHeight:1.5}}>
        Nuisance-Modelle werden <strong>einmalig</strong> pro äußerem Fold mit cache_values gecacht. Trials fitten nur model_final via refit_final() auf gecachten Residuals (sequentiell, n_jobs=1). OOF-Evaluation via externem RScorer (unabhängige Nuisance, 2-Fold T×Y).
        {isBoth && <span style={{display:"block",marginTop:6,color:"#7a5a00",background:"#fffbeb",padding:"4px 8px",borderRadius:6,border:"1px solid #e8d49c",fontSize:10.5}}><strong>CatBoost &amp; LGBM aktiv:</strong> FM-Tuning-Trials sind verdoppelt ({(fmtTrials||50)}T × 2 = {nTrials}T), damit beide Learner-Familien ausreichend Budget bekommen.</span>}
      </div>
      <div style={{border:"1px solid #ede6e7",borderRadius:8,overflow:"hidden"}}>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 70px",fontSize:11,fontWeight:600,padding:"8px 14px",background:"#b8860b",color:"#fff"}}>
          <span>Modell</span><span>Berechnung</span><span>Fits</span>
        </div>
        {rows.map((r,i) => (
          <div key={r.model} style={{borderBottom:"1px solid #f5f0f0",background:i%2===0?"#fff":"#fdfbfb"}}>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 70px",padding:"8px 14px",alignItems:"center",fontSize:12.5}}>
              <div>
                <span style={{fontWeight:600,color:"#333"}}>{r.model}</span>
                <span style={{fontSize:10,color:"#7a5a00",marginLeft:8}}>{r.method}</span>
              </div>
              <span style={{fontSize:10.5,color:"#888",fontFamily:MONO}}>{r.detail}</span>
              <span style={{fontFamily:MONO,color:"#b8860b",fontWeight:700}}>{r.totalFits.toLocaleString()}</span>
            </div>
            <div style={{fontSize:10.5,color:"#999",padding:"0 14px 6px",marginTop:-2,lineHeight:1.4}}>{r.note}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

const PModelsOnly = ({cfg,set}) => {
  const mt=cfg.treatmentType==="multi";const nanB=cfg.hasNaN?new Set(["CausalForestDML","CausalForest"]):new Set();const mtB=mt?btOnly:new Set();const av=allModels.filter(m=>!mtB.has(m)&&!nanB.has(m));
  const _isObs = (cfg.studyType||"rct") !== "rct";
  const _noConfModels = new Set(["SLearner","TLearner","CausalForest"]);
  return (<Sec title="Kausale Modelle">{mt&&<Info type="warn"><strong>Multi-Treatment:</strong> Nur Modelle mit Residualisierung verfügbar (NonParamDML, ParamDML, DRLearner, CausalForestDML). Meta-Learner und CausalForest unterstützen kein Multi-Treatment.</Info>}{cfg.hasNaN&&<Info type="warn"><strong>Fehlende Werte:</strong> CausalForestDML und CausalForest werden automatisch übersprungen (GRF-basierte Modelle können keine fehlenden Werte verarbeiten).</Info>}
    {[
      {label:"Meta-Learner",color:"#9B111E",bg:"#fdf2f3",models:[
        {m:"SLearner",d:"Ein Modell für alle – einfachste Baseline"},
        {m:"TLearner",d:"Getrennte Modelle pro Treatment-Gruppe"},
        {m:"XLearner",d:"Cross-Learner mit Propensity-Gewichtung"},
      ]},
      {label:"DML & Doubly Robust",color:"#D4A853",bg:"#fffbeb",models:[
        {m:"NonParamDML",d:"Orthogonalisiert – flexibles CATE-Modell (model_final)"},
        {m:"ParamDML",d:"Orthogonalisiert – lineares CATE-Modell (LinearDML)"},
        {m:"DRLearner",d:"Doubly Robust – konsistent wenn Outcome- oder Propensity-Modell korrekt"},
      ]},
      {label:"GRF (Causal Forests)",color:"#2d6a4f",bg:"#e8f5ec",models:[
        {m:"CausalForestDML",d:"DML-Residualisierung + Causal Forest"},
        {m:"CausalForest",d:"Direkte Effektschätzung ohne Nuisance-Modelle"},
      ]},
    ].map(grp => {
      const vis = grp.models.filter(({m}) => allModels.includes(m));
      if(!vis.length) return null;
      return (<div key={grp.label} style={{marginBottom:14}}>
        <div style={{fontSize:10.5,fontWeight:700,textTransform:"uppercase",letterSpacing:".5px",color:grp.color,marginBottom:8}}>{grp.label}</div>
        <div style={{display:"grid",gridTemplateColumns:vis.length===2?"1fr 1fr":"1fr 1fr 1fr",gap:8}}>
          {vis.map(({m,d}) => {
            const ok=av.includes(m), on=(cfg.models||[]).includes(m)&&ok;
            const reason=mtB.has(m)?"Nur Binary":nanB.has(m)?"Keine NaN":null;
            return (
              <button key={m} onClick={()=>{if(!ok)return;const s=new Set(cfg.models||[]);on?s.delete(m):s.add(m);set({...cfg,models:[...s]})}}
                style={{display:"flex",alignItems:"flex-start",gap:10,padding:"12px 14px",background:on?grp.bg:"#fff",border:on?`2px solid ${grp.color}`:"1.5px solid "+C.border,borderRadius:10,cursor:ok?"pointer":"not-allowed",textAlign:"left",transition:"all 0.15s",opacity:ok?1:0.35}}>
                <div style={{width:16,height:16,borderRadius:4,border:on?`2px solid ${grp.color}`:"2px solid #ccc",background:on?grp.color:"#fff",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,marginTop:2}}>
                  {on && <span style={{color:"#fff",fontSize:10,fontWeight:700}}>✓</span>}
                </div>
                <div>
                  <div style={{fontSize:12.5,fontWeight:600,color:on?grp.color:"#333"}}>{m}</div>
                  <div style={{fontSize:10.5,color:"#888",marginTop:2,lineHeight:1.4}}>{d}{!ok&&reason?` – ${reason}`:""}{_isObs&&on&&_noConfModels.has(m)?" ⚠ Keine Confounding-Korrektur":""}</div>
                </div>
              </button>
            );
          })}
        </div>
      </div>);
    })}
    <div style={{fontSize:11,color:"#999",marginTop:4}}>{(cfg.models||[]).filter(m=>av.includes(m)).length} von {av.length} verfügbaren Modellen ausgewählt</div>
    <Divider/><div style={{background:cfg.ensembleEnabled?"rgba(155,17,30,0.04)":"transparent",border:cfg.ensembleEnabled?"1.5px solid rgba(155,17,30,0.15)":"1.5px dashed "+C.border,borderRadius:10,padding:"12px 14px",transition:"all 0.2s"}}><Toggle label="Ensemble (Gleichgewichtet)" checked={cfg.ensembleEnabled||false} onChange={v=>set({...cfg,ensembleEnabled:v})} help="Mittelt die CATE-Vorhersagen aller trainierten Modelle. Nimmt an der Champion-Selektion teil."/>{cfg.ensembleEnabled&&<>{(()=>{const active=(cfg.models||[]).filter(m=>av.includes(m));return active.length>=2?<div style={{fontSize:11,color:"#999",marginTop:8,lineHeight:1.8}}>Ensemble aus {active.map((m,i)=><span key={m}>{i>0&&<span style={{color:"#ddd"}}> · </span>}<span style={{fontSize:10.5,padding:"2px 8px",borderRadius:8,background:"rgba(155,17,30,0.08)",color:C.ruby,fontWeight:600,whiteSpace:"nowrap"}}>{m}</span></span>)} <span style={{fontStyle:"italic",whiteSpace:"nowrap"}}>({active.length} Modelle, je 1/{active.length})</span></div>:null})()}{(cfg.models||[]).filter(m=>av.includes(m)).length<2&&<Info type="warn">Mindestens 2 aktive Modelle nötig.</Info>}</>}</div></Sec>);
};

const PModels = ({cfg,set,sp,setSp,spFmt,setSpFmt,dataStats,sysInfo}) => {
  const _bl = cfg.baseLearner || "catboost";
  const _isBoth = _bl === "both";
  return (<><Sec title="Learner">
<div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12}}>
          {[
            {key:"catboost",logos:[CB_LOGO],name:"CatBoost",sub:"Gut bei vielen kategorialen Features"},
            {key:"lgbm",logos:[LGBM_LOGO],name:"LightGBM",sub:"Schneller, GBDT-Boosting"},
            {key:"both",logos:[CB_LOGO,LGBM_LOGO],name:"CatBoost & LGBM",sub:"Tuning wählt besten Learner pro Task"},
          ].map(bl => {
            const active = _bl === bl.key;
            return (
              <button key={bl.key} onClick={()=>set({...cfg,baseLearner:bl.key,blFixed:{},fmtFixed:{}})}
                style={{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:6,padding:"18px 14px 14px",background:active?"#faf6f6":"#fff",border:active?"2px solid "+C.ruby:"1.5px solid "+C.border,borderRadius:12,cursor:"pointer",transition:"all 0.15s",position:"relative",minHeight:110}}
                onMouseEnter={e=>{if(!active){e.currentTarget.style.borderColor="#ccc";e.currentTarget.style.background="#fdfbfb"}}}
                onMouseLeave={e=>{if(!active){e.currentTarget.style.borderColor=C.border;e.currentTarget.style.background="#fff"}}}>
                {active && <div style={{position:"absolute",top:8,right:10,width:18,height:18,borderRadius:9,background:C.ruby,display:"flex",alignItems:"center",justifyContent:"center"}}><span style={{color:"#fff",fontSize:10,fontWeight:700}}>✓</span></div>}
                <div style={{display:"flex",alignItems:"center",justifyContent:"center",gap:bl.logos.length>1?10:0,height:40}}>
                  {bl.logos.map((lg,i)=>(<img key={i} src={lg} alt={bl.name} style={{height:bl.logos.length>1?32:40,objectFit:"contain",filter:active?"none":"grayscale(40%) opacity(0.6)"}}/>))}
                </div>
                <div style={{fontSize:13,fontWeight:active?700:500,color:active?C.dark:"#999",letterSpacing:0.3,marginTop:2}}>{bl.name}</div>
                <div style={{fontSize:10.5,color:active?C.textSec:"#aaa",textAlign:"center",lineHeight:1.3,marginTop:2,maxWidth:160}}>{bl.sub}</div>
              </button>
            );
          })}
        </div>{_isBoth && <div style={{background:"#fff8e7",border:"1px solid #e8d49c",borderRadius:10,padding:"12px 16px",marginTop:12,fontSize:12,color:C.textSec,lineHeight:1.55}}><strong style={{color:"#7a5a00"}}>So funktioniert &quot;CatBoost &amp; LGBM&quot;:</strong> Dieser Modus entscheidet sich erst <strong>während des Tunings</strong> pro Task (model_y, model_t, model_final, ...) zwischen LightGBM und CatBoost. Das Tuning lernt, welcher Learner für die jeweilige Rolle besser abschneidet — am Ende kann model_y z.B. CatBoost nutzen und model_t LightGBM. Die Trial-Zahl wird dabei automatisch verdoppelt, damit beide Learner-Familien gleich viel Budget bekommen. <strong>Hinweis:</strong> Der Modus benötigt aktives Tuning. Ohne Tuning gibt es keine Entscheidungsgrundlage — dann wird CatBoost (globaler Default) verwendet.</div>}</Sec><Sec title="Base-Learner-Tuning" accent="#C4343F"><Info>Optimiert die Nuisance-Modelle (Outcome, Propensity) via Optuna. Die getunten Parameter werden automatisch an FMT, CFT und Training weitergereicht.</Info><Row><Col><Toggle label="Base-Learner-Tuning aktivieren" checked={cfg.tuningEnabled} onChange={v=>set({...cfg,tuningEnabled:v})}/></Col><Col><Toggle label="Single-Fold-Tuning" checked={cfg.tuningSingleFold||false} onChange={v=>set({...cfg,tuningSingleFold:v})} help="Bewertet jeden Trial auf 1 statt K Folds — schneller, aber verrauschtere Log-Loss/MSE-Schätzung."/></Col></Row>{_isBoth && !cfg.tuningEnabled && <div style={{fontSize:11.5,color:"#7a2e0e",background:"#fef2f2",padding:"10px 14px",borderRadius:8,border:"1px solid #e8b4b8",lineHeight:1.5,marginTop:8}}><strong>⚠ Hinweis:</strong> Im Modus &quot;CatBoost &amp; LGBM&quot; entscheidet Optuna pro Task, welcher Learner besser performt — dafür muss Tuning aktiv sein. Ohne Tuning kann keine Learner-Auswahl stattfinden, und es wird ausschließlich <strong>CatBoost</strong> als Fallback verwendet. Empfehlung: Tuning aktivieren, oder oben direkt den gewünschten Learner wählen.</div>}{!cfg.tuningEnabled && <><Divider/><div style={{display:"flex",alignItems:"center",gap:10,marginBottom:12,flexWrap:"wrap"}}><span style={{fontSize:13,fontWeight:600,color:C.dark}}>Base-Learner-Hyperparameter</span><span style={{fontSize:10,background:"#FDF2F3",color:"#9B111E",padding:"3px 12px",borderRadius:12,border:"1px solid #e8b4b8",fontWeight:600,letterSpacing:0.2}}>Direkt verwendet</span></div><Info>Tuning ist deaktiviert. Diese Parameter werden direkt für alle Base-Learner-Modelle verwendet (Outcome, Propensity, CATE-Hilfsmodelle).</Info>{_isBoth ? (<>
<div style={{fontSize:12,fontWeight:600,color:C.dark,marginBottom:6,marginTop:4}}>CatBoost-Parameter</div>
<FixedParamsEditor params={(cfg.blFixed||{}).catboost||{}} defaults={CB_DEFAULTS} onChange={v=>set({...cfg,blFixed:{...(cfg.blFixed||{}),catboost:v}})} label="CatBoost Hyperparameter (Fixed)"/>
<div style={{fontSize:12,fontWeight:600,color:C.dark,marginBottom:6,marginTop:14}}>LightGBM-Parameter</div>
<FixedParamsEditor params={(cfg.blFixed||{}).lgbm||{}} defaults={LGBM_DEFAULTS} onChange={v=>set({...cfg,blFixed:{...(cfg.blFixed||{}),lgbm:v}})} label="LightGBM Hyperparameter (Fixed)"/>
</>) : (<FixedParamsEditor params={cfg.blFixed||{}} defaults={(cfg.baseLearner||"catboost")==="catboost"?CB_DEFAULTS:LGBM_DEFAULTS} onChange={v=>set({...cfg,blFixed:v})} label="Hyperparameter (Fixed)"/>)}</>}{cfg.tuningEnabled&&<><Divider/><div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:8}}>Tuning-Plan</div><TuningPlanPreview models={(cfg.tuningModels||[]).length>0?cfg.tuningModels:cfg.models} trials={cfg.tuningTrials||50} cv={cfg.tuningSingleFold?1:(cfg.dmlCrossfitFolds||5)} outerCv={cfg.cvSplits||5} innerCv={cfg.dmlCrossfitFolds||5} isBoth={_isBoth} isRct={(cfg.studyType||"rct")==="rct"}/><Divider/><div style={{fontSize:13,fontWeight:600,color:"#6B0D15",marginBottom:8}}>Modelle für BL-Tuning</div><Info>Standardmäßig werden alle ausgewählten Modelle getuned. Hier kann das Tuning auf bestimmte Modelle eingeschränkt werden — nicht ausgewählte Modelle nutzen die festen Hyperparameter (fixed_params).</Info><div style={{display:"flex",flexWrap:"wrap",gap:10,marginBottom:14}}>{(cfg.models||[]).map(m=>{const active=(cfg.tuningModels||[]).length===0||(cfg.tuningModels||[]).includes(m);return(<label key={m} style={{display:"inline-flex",alignItems:"center",gap:6,padding:"6px 14px",borderRadius:8,border:active?"1.5px solid #C4343F":"1.5px solid "+C.border,background:active?"#FDF2F3":"#fff",cursor:"pointer",fontSize:12.5,fontWeight:active?600:400,transition:"all 0.15s"}}><input type="checkbox" checked={active} style={{accentColor:"#C4343F"}} onChange={e=>{const cur=cfg.tuningModels||[];if(cur.length===0){const s=new Set((cfg.models||[]).filter(x=>x!==m));if(s.size>0)set({...cfg,tuningModels:[...s]})}else{const s=new Set(cur);e.target.checked?s.add(m):s.delete(m);if(s.size===0)return;set({...cfg,tuningModels:s.size===(cfg.models||[]).length?[]:[...s]})}}}/>{m}</label>)})}</div>{(cfg.tuningModels||[]).length>0 && <div style={{fontSize:11,color:C.textMuted,background:C.rose,padding:"6px 12px",borderRadius:8,marginBottom:10,lineHeight:1.4}}>Eingeschränkt: Nur <strong style={{color:C.ruby}}>{(cfg.tuningModels||[]).join(", ")}</strong> werden getuned. Alle anderen nutzen fixed_params.</div>}<Divider/>{(()=>{