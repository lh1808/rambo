const nCores = sysInfo?.cpu?.cores || 0;
const pl = cfg.parallelLevel||3;
const par = nCores && pl >= 3 ? Math.max(1, Math.floor(nCores / 4)) : (pl <= 2 ? 1 : 0);
const trials = cfg.tuningTrials||50;
const waves = par > 0 ? Math.ceil(trials / par) : 0;
const waveColor = waves < 3 ? "#dc2626" : waves < 5 ? "#d97706" : "#059669";
const setByWaves = (w) => set({...cfg, tuningTrials: Math.max(par||5, (par||5) * w)});
return (<>
<div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:8}}>Tuning-Intensität</div>
{par > 0 ? (<>
  <div style={{display:"flex",gap:8,marginBottom:8}}>
    {[{w:3,l:"Schnell",d:"Grenzwertig, aber schnell"},{w:5,l:"Standard",d:"Solide TPE-Exploration"},{w:8,l:"Gründlich",d:"Hohe Suchqualität"}].map(p => {
      const active = waves === p.w;
      return <button key={p.w} onClick={()=>setByWaves(p.w)} style={{flex:1,padding:"10px 12px",borderRadius:10,border:active?"1.5px solid #C4343F":"1.5px solid "+C.border,background:active?C.rose:"#fff",cursor:"pointer",textAlign:"left",transition:"all 0.15s"}}>
        <div style={{fontSize:13,fontWeight:600,color:active?C.ruby:C.dark}}>{p.l}</div>
        <div style={{fontSize:11,color:active?C.ruby:C.textMuted,marginTop:2}}>{p.w} Wellen = {par*p.w} Trials</div>
      </button>;
    })}
  </div>
  <div style={{fontSize:11,color:C.textSec,background:"#f9fafb",padding:"8px 12px",borderRadius:8,lineHeight:1.5,border:"1px solid #e5e7eb"}}>
    <strong>{par}</strong> parallele Trials ({nCores} Kerne / 4) × <strong style={{color:waveColor}}>{waves} Wellen</strong> = <strong>{trials}</strong> Trials.
    {waves < 3 && " TPE kann bei weniger als 3 Wellen kaum zwischen guten und schlechten Parametern unterscheiden."}
    {waves >= 3 && waves < 5 && " Grenzwertig — TPE beginnt erst nach den Startup-Trials zu lernen."}
    {waves >= 5 && " Genug Wellen für stabile TPE-Exploration und Exploitation."}
  </div>
  <Expander title="Manuell anpassen">
    <Row><Col><Inp label="Trials (manuell)" type="number" value={trials} onChange={v=>set({...cfg,tuningTrials:Number(v)})}/></Col><Col><div style={{fontSize:11,color:C.textMuted,marginTop:24}}>= {par} parallel × {waves} Wellen</div></Col></Row>
  </Expander>
</>) : (
  <Inp label="Trials" type="number" value={trials} onChange={v=>set({...cfg,tuningTrials:Number(v)})} help="Server-Verbindung nötig für Wellen-Berechnung."/>
)}
</>);
})()}<Divider/>{_isBoth ? (<>
<SSEditor bl="catboost" sp={sp} setSp={setSp} accent="#C4343F"/>
<div style={{height:8}}/>
<SSEditor bl="lgbm" sp={sp} setSp={setSp} accent="#C4343F"/>
</>) : (<SSEditor bl={cfg.baseLearner||"catboost"} sp={sp} setSp={setSp} accent="#C4343F"/>)}<Expander title="Base-Learner-Defaults (Fixed)" accent="#C4343F"><div style={{fontSize:11,color:"#6b7280",lineHeight:1.5,marginBottom:10}}>Startpunkte für nicht-getunte Rollen und Default-Werte für Parameter außerhalb des Suchraums.</div>{_isBoth ? (<>
<FixedParamsEditor params={(cfg.blFixed||{}).catboost||{}} defaults={CB_DEFAULTS} onChange={v=>set({...cfg,blFixed:{...(cfg.blFixed||{}),catboost:v}})} label="CatBoost"/>
<FixedParamsEditor params={(cfg.blFixed||{}).lgbm||{}} defaults={LGBM_DEFAULTS} onChange={v=>set({...cfg,blFixed:{...(cfg.blFixed||{}),lgbm:v}})} label="LightGBM"/>
</>) : (<FixedParamsEditor params={cfg.blFixed||{}} defaults={(cfg.baseLearner||"catboost")==="catboost"?CB_DEFAULTS:LGBM_DEFAULTS} onChange={v=>set({...cfg,blFixed:v})} label="Hyperparameter"/>)}</Expander><Expander title="Erweiterte Tuning-Einstellungen"><div style={{fontSize:11,color:C.textSec,background:"#f9fafb",padding:"6px 10px",borderRadius:8,marginBottom:10,lineHeight:1.4,border:"1px solid #e5e7eb"}}><strong>Metriken (fest):</strong> Log-Loss (Klassifikation) und neg. MSE (Regression) — messen Kalibrierung für unverzerrte CATE-Schätzungen.</div><Row><Col><Inp label="Timeout (Sek., 0=unbegrenzt)" type="number" value={cfg.tuningTimeout||0} onChange={v=>set({...cfg,tuningTimeout:Number(v)})} help="Zeitlimit pro Study in Sekunden"/></Col></Row><Divider/><div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:6}}>Overfit-Penalty (nur Meta-Learner)</div>
{(()=>{const p=cfg.overfitPenalty||0,t=cfg.overfitTolerance===undefined?0.2:cfg.overfitTolerance;const lbl=p===0?"Aus":(p===0.2&&t===0.2)?"Moderat":(p===0.3&&t===0.2)?"Stark":"Benutzerdefiniert";const clr=p===0?"#059669":"#d97706";return <div style={{fontSize:11,color:clr,background:p===0?"#f0fdf4":"#fffbeb",padding:"6px 12px",borderRadius:8,marginBottom:8,border:`1px solid ${clr}33`}}><strong>Aktuelle Einstellung: {lbl}</strong>{p>0&&` (Penalty ${p}, Tol. ${t})`} — global steuerbar unter Pipeline-Optionen → Tuning-Regularisierung.</div>;})()}<div style={{fontSize:11,color:C.textSec,lineHeight:1.5,marginBottom:10}}>Bestraft Hyperparameter-Konfigurationen mit großem Train-Val-Gap. Wirkt <strong>nur auf Meta-Learner</strong> (S-/T-/X-Learner), die ohne internes Cross-Fitting direkt den CATE bilden — dort ist sie der primäre Overfitting-Hebel. <strong>DML/DR-Nuisances werden nie bestraft</strong> (Cross-Fitting + Orthogonalität fangen deren Overfitting ab, Bach et al. 2024) — dieser Regler hat auf sie keinen Effekt. Die Tolerance ist relativ: 0.2 = 20% Gap wird toleriert.</div><Row><Col><Sld label="Penalty-Faktor" min={0} max={1} step={0.05} value={cfg.overfitPenalty||0} onChange={v=>set({...cfg,overfitPenalty:v})} help="Stärke der Bestrafung der Meta-Learner-Base-Models (S/T/X). Default 0. DML/DR-Nuisances bleiben unberührt (nie bestraft)."/></Col><Col><Sld label="Toleranz (Gap-Schwelle)" min={0} max={0.3} step={0.01} value={cfg.overfitTolerance===undefined?0.2:cfg.overfitTolerance} onChange={v=>set({...cfg,overfitTolerance:v})} help="Relativer Gap-Schwellwert: 0.2 = 20% des Score-Betrags. Greift nur für Meta-Learner."/></Col></Row>{(cfg.overfitPenalty||0)>0 && <div style={{fontSize:11,color:C.ruby,background:C.rose,padding:"8px 12px",borderRadius:8,marginTop:6,lineHeight:1.4,border:"1px solid "+C.border}}><strong>Aktiv (nur Meta-Learner):</strong> Penalty={cfg.overfitPenalty}, Toleranz={cfg.overfitTolerance===undefined?0.2:cfg.overfitTolerance}. Relative Tolerance: {(cfg.overfitTolerance===undefined?20:Math.round(cfg.overfitTolerance*100))}% Gap toleriert. DML/DR-Nuisances bleiben unbestraft.</div>}{_isBoth && (cfg.tuningTimeout||0)>0 && <Info type="warn">Timeout wird bei „CatBoost &amp; LGBM" automatisch ignoriert, damit beide Learner gleich viele Trials durchlaufen (LGBM ist langsamer). Steuerung erfolgt über Trials.</Info>}</Expander></>}</Sec><Sec title="Final-Model-Tuning" accent="#D4A853"><Info>Optimiert das CATE-Effektmodell (model_final) via Optuna. Bewertung per Scorer (auto = Qini bei RCT, R-Score bei Beobachtungsdaten; bei Multi-Treatment immer R-Score).</Info><Row><Col><Toggle label="Final-Model-Tuning aktivieren" checked={cfg.fmtEnabled} onChange={v=>{const u={fmtEnabled:v};if(v&&(cfg.fmtModels||[]).length===0){u.fmtModels=(cfg.models||[]).filter(m=>["NonParamDML","DRLearner"].includes(m));}set({...cfg,...u});}}/></Col><Col><Toggle label="Single-Fold-Tuning" checked={cfg.fmtSingleFold||false} onChange={v=>set({...cfg,fmtSingleFold:v})} help="Bewertet jeden Trial auf 1 statt K OOF-Folds — schneller, aber verrauschtere Score-Schätzung."/></Col></Row>{cfg.fmtEnabled&&<><Row><Col><div style={{fontSize:12,fontWeight:500,color:C.text,marginBottom:4}}>Scorer</div><select value={cfg.fmtScorer||"auto"} onChange={e=>{const v=e.target.value;set({...cfg,fmtScorer:v,cfScorer:v});}} style={{width:"100%",padding:"7px 10px",borderRadius:8,border:"1px solid "+C.border,fontSize:12.5,background:"#fff"}}><option value="auto">{cfg.treatmentType==="multi"?"auto (R-Score — bei Multi-Treatment immer)":"auto (Qini bei RCT, R-Score bei Beobachtungsdaten)"}</option><option value="qini" disabled={cfg.treatmentType==="multi"}>Qini — optimiert Ranking-Qualität (OOF-aggregiert){cfg.treatmentType==="multi"?" — nicht bei Multi-Treatment (binär-only)":""}</option><option value="rscore">R-Score — optimiert CATE-Genauigkeit (EconML RScorer)</option></select><div style={{fontSize:10.5,color:C.textSec,marginTop:3}}>Qini: Direkt auf der Uplift-Metrik tunen (kein Pruning, robust bei schwachem Signal). R-Score: Pointwise CATE-MSE (Pruning möglich, EconML-Standard). Wird synchron mit CFT gesetzt.{cfg.treatmentType==="multi"?" Bei Multi-Treatment ist nur R-Score möglich (Qini verlangt binäres T und 1-d CATE).":""}</div></Col></Row><Divider/><div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:8}}>Tuning-Plan</div><FinalTuningPlanPreview models={cfg.models} fmtEnabled={cfg.fmtEnabled} fmtModels={cfg.fmtModels||[]} fmtSingleFold={cfg.fmtSingleFold||false} fmtTrials={cfg.fmtTrials||50} fmtCv={cfg.dmlCrossfitFolds||5} outerCv={cfg.cvSplits||5} mcIters={cfg.mcIters||1} isBoth={_isBoth} fmtScorer={cfg.fmtScorer} studyType={cfg.studyType} isMulti={cfg.treatmentType==="multi"}/><Divider/><div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:8}}>Modelle für Final-Tuning</div><Info>Wähle, welche Modelle per OOF-CV optimiert werden. Nicht ausgewählte Modelle verwenden die festen Hyperparameter (unten).</Info><div style={{display:"flex",flexWrap:"wrap",gap:10,marginBottom:14}}>{(cfg.models||[]).filter(m=>["NonParamDML","DRLearner"].includes(m)).map(m=>{const active=(cfg.fmtModels||[]).includes(m);return(<label key={m} style={{display:"inline-flex",alignItems:"center",gap:6,padding:"6px 14px",borderRadius:8,border:active?"1.5px solid #D4A853":"1.5px solid "+C.border,background:active?"#fffbeb":"#fff",cursor:"pointer",fontSize:12.5,fontWeight:active?600:400,transition:"all 0.15s"}}><input type="checkbox" checked={active} style={{accentColor:"#D4A853"}} onChange={e=>{const s=new Set(cfg.fmtModels||[]);e.target.checked?s.add(m):s.delete(m);set({...cfg,fmtModels:[...s]})}}/>{m}</label>)})}{!(cfg.models||[]).some(m=>["NonParamDML","DRLearner"].includes(m))&&<Info type="warn">Keine FMT-fähigen Modelle (NonParamDML/DRLearner) ausgewählt.</Info>}{(cfg.models||[]).some(m=>["NonParamDML","DRLearner"].includes(m)) && (cfg.fmtModels||[]).length===0 && <Info type="warn">Kein Modell für Final-Tuning ausgewählt. Alle verwenden die festen Hyperparameter.</Info>}</div><Divider/>{(()=>{
const trials = cfg.fmtTrials||50;
const trialColor = trials < 15 ? "#dc2626" : trials < 30 ? "#d97706" : "#059669";
return (<>
<div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:8}}>Tuning-Intensität</div>
<div style={{display:"flex",gap:8,marginBottom:8}}>
  {[{t:30,l:"Schnell",d:"Grenzwertig, aber schnell"},{t:50,l:"Standard",d:"Solide TPE-Exploration"},{t:100,l:"Gründlich",d:"Hohe Suchqualität"}].map(p => {
    const active = trials === p.t;
    return <button key={p.t} onClick={()=>set({...cfg, fmtTrials: p.t})} style={{flex:1,padding:"10px 12px",borderRadius:10,border:active?"1.5px solid #D4A853":"1.5px solid "+C.border,background:active?"#fffbeb":"#fff",cursor:"pointer",textAlign:"left",transition:"all 0.15s"}}>
      <div style={{fontSize:13,fontWeight:600,color:active?"#7a5a00":C.dark}}>{p.l}</div>
      <div style={{fontSize:11,color:active?"#7a5a00":C.textMuted,marginTop:2}}>{p.t} Trials</div>
    </button>;
  })}
</div>
<div style={{fontSize:11,color:C.textSec,background:"#f9fafb",padding:"8px 12px",borderRadius:8,lineHeight:1.5,border:"1px solid #e5e7eb"}}>
  Final-Model-Tuning läuft <strong>sequentiell</strong> (Nuisance einmalig mit cache_values gecacht, nur model_final wird pro Trial via refit_final() neu gefittet — alle CPU-Kerne pro Fit). <strong style={{color:trialColor}}>{trials}</strong> Trials insgesamt.
  {trials < 15 && " TPE kann bei weniger als 15 Trials kaum zwischen guten und schlechten Parametern unterscheiden."}
  {trials >= 15 && trials < 30 && " Grenzwertig — TPE beginnt erst nach den Startup-Trials zu lernen."}
  {trials >= 30 && " Genug Trials für stabile TPE-Exploration und Exploitation."}
</div>
<Expander title="Manuell anpassen">
  <Inp label="Trials" type="number" value={trials} onChange={v=>set({...cfg,fmtTrials:Number(v)})} help="Anzahl Optuna-Trials. Nuisance-Modelle werden einmalig gecacht, Trials fitten nur model_final (sequentiell, alle Kerne)."/>
</Expander>
</>);
})()}{cfg.fmtSingleFold && <div style={{fontSize:11,color:C.textMuted,background:C.rose,padding:"6px 12px",borderRadius:8,marginTop:6,lineHeight:1.4}}>Single-Fold aktiv: Jeder Trial wird auf <strong style={{color:C.ruby}}>1</strong> statt {cfg.cvSplits||5} OOF-Folds evaluiert — {cfg.cvSplits||5}× schneller.</div>}{cfg.fmtSingleFold && (()=>{const K=cfg.cvSplits||5;if(dataStats){const ppf=Math.floor(dataStats.minority/K);if(ppf<100){const severe=ppf<50;return <div style={{fontSize:11,color:severe?"#991b1b":"#92400e",background:severe?"#fef2f2":"#fffbeb",padding:"8px 12px",borderRadius:8,marginTop:6,lineHeight:1.5,border:`1px solid ${severe?"#fca5a5":"#fbbf24"}`}}><strong>{severe?"⚠ Nicht empfohlen":"⚠ Grenzwertig"}:</strong> Bei Ihren Daten nur <strong>{ppf}</strong> Minority-Fälle im äußeren Val-Fold (empfohlen: ≥100, Minimum: ≥50). {severe?"OOF-Score ist nicht aussagekräftig. K-Fold CV dringend empfohlen.":"Die Score-Schätzung (R-Score) ist merklich verrauscht — K-Fold CV empfohlen."}</div>;}}else return <div style={{fontSize:11,color:"#6b7280",background:"#f9fafb",padding:"6px 12px",borderRadius:8,marginTop:6,lineHeight:1.5,border:"1px solid #e5e7eb"}}>Faustregel: min(n_treated, n_positive) / {K} ≥ 100 pro Val-Fold für zuverlässige Metrik-Schätzung (Collins et al.: min. 100, idealerweise 200+ Events für stabile Validation).</div>;return null;})()}<Divider/>{_isBoth ? (<>
<div style={{fontSize:12,fontWeight:600,color:C.dark,marginBottom:6,marginTop:4}}>CatBoost</div>
<SSEditor bl="catboost" sp={spFmt||{lgbm:{},catboost:{}}} setSp={setSpFmt||setSp} fmt accent="#D4A853"/>
<div style={{fontSize:12,fontWeight:600,color:C.dark,marginBottom:6,marginTop:14}}>LightGBM</div>
<SSEditor bl="lgbm" sp={spFmt||{lgbm:{},catboost:{}}} setSp={setSpFmt||setSp} fmt accent="#D4A853"/>
</>) : (<SSEditor bl={cfg.baseLearner||"catboost"} sp={spFmt||{lgbm:{},catboost:{}}} setSp={setSpFmt||setSp} fmt accent="#D4A853"/>)}<Expander title="CATE-Modell-Defaults (Fixed)" accent="#D4A853"><div style={{fontSize:11,color:"#6b7280",lineHeight:1.5,marginBottom:10}}>Startpunkte für model_final. Bei aktivem FMT überschreibt Optuna die Parameter innerhalb des Suchraums — hier festgelegte Werte gelten nur für Parameter außerhalb des Suchraums.</div>{_isBoth ? (<>
<FixedParamsEditor params={(cfg.fmtFixed||{}).catboost||{}} defaults={CB_FINAL_DEFAULTS} onChange={v=>set({...cfg,fmtFixed:{...(cfg.fmtFixed||{}),catboost:v}})} label="CatBoost" accent="#D4A853"/>
<FixedParamsEditor params={(cfg.fmtFixed||{}).lgbm||{}} defaults={LGBM_FINAL_DEFAULTS} onChange={v=>set({...cfg,fmtFixed:{...(cfg.fmtFixed||{}),lgbm:v}})} label="LightGBM" accent="#D4A853"/>
</>) : (<FixedParamsEditor params={cfg.fmtFixed||{}} defaults={(cfg.baseLearner||"catboost")==="catboost"?CB_FINAL_DEFAULTS:LGBM_FINAL_DEFAULTS} onChange={v=>set({...cfg,fmtFixed:v})} label="CATE-Parameter" accent="#D4A853"/>)}</Expander><Expander title="Erweiterte Final-Tuning-Einstellungen"><Row><Col><Inp label="Timeout (Sek., 0=unbegrenzt)" type="number" value={cfg.fmtTimeout||0} onChange={v=>set({...cfg,fmtTimeout:Number(v)})} help="Zeitlimit pro Final-Model-Optuna-Study"/></Col></Row>{_isBoth && (cfg.fmtTimeout||0)>0 && <Info type="warn">Timeout wird bei „CatBoost &amp; LGBM" automatisch ignoriert (faire Trial-Allokation).</Info>}<Divider/><div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:6}}>Overfit-Penalty (model_final)</div>
{(()=>{const p=cfg.fmtOverfitPenalty||0,t=cfg.fmtOverfitTolerance===undefined?0.1:cfg.fmtOverfitTolerance;const lbl=p===0?"Aus":p===0.2&&t===0.1?"Moderat":p===0.35&&t===0.1?"Stark":"Benutzerdefiniert";const clr=p===0?"#059669":p<=0.2?"#d97706":"#dc2626";return <div style={{fontSize:11,color:clr,background:p===0?"#f0fdf4":"#fffbeb",padding:"6px 12px",borderRadius:8,marginBottom:8,border:`1px solid ${clr}33`}}><strong>Aktuelle Einstellung: {lbl}</strong>{p>0&&` (Penalty ${p}, Tol. ${t})`} — global steuerbar unter Pipeline-Optionen → Tuning-Regularisierung.</div>;})()}<div style={{fontSize:11,color:C.textSec,lineHeight:1.5,marginBottom:10}}>Bestraft model_final-Konfigurationen, deren Score auf den Trainingsdaten deutlich besser ist als auf der Validierung (OOF). Ein großer Train-Val-Gap bedeutet, dass model_final Rauschen statt echte Heterogenität gelernt hat. Bei 0 (Default) wird nur der OOF-Score optimiert. Tolerance ist strenger als bei BLT (10% statt 20%), weil CATE-Signal schwächer ist als Outcome-Signal.</div><Row><Col><Sld label="Penalty-Faktor" min={0} max={1} step={0.05} value={cfg.fmtOverfitPenalty||0} onChange={v=>set({...cfg,fmtOverfitPenalty:v})} help="Stärke der Bestrafung. 0 = aus, 0.2 = moderat, 0.35 = stark."/></Col><Col><Sld label="Toleranz (Gap-Schwelle)" min={0} max={0.2} step={0.01} value={cfg.fmtOverfitTolerance===undefined?0.1:cfg.fmtOverfitTolerance} onChange={v=>set({...cfg,fmtOverfitTolerance:v})} help="Relativer Gap-Schwellwert: 0.1 = 10% des Score-Betrags. Skalen-sicher (R-Score ist normalisiert)."/></Col></Row>{(cfg.fmtOverfitPenalty||0)>0 && <div style={{fontSize:11,color:"#92400e",background:"#fffbeb",padding:"8px 12px",borderRadius:8,marginTop:6,lineHeight:1.4,border:"1px solid #fbbf24"}}><strong>Aktiv:</strong> Penalty={cfg.fmtOverfitPenalty}, Toleranz={cfg.fmtOverfitTolerance===undefined?0.1:cfg.fmtOverfitTolerance}</div>}</Expander></>}{!cfg.fmtEnabled && (cfg.models||[]).some(m=>["NonParamDML","DRLearner"].includes(m)) && <><Divider/><div style={{display:"flex",alignItems:"center",gap:10,marginBottom:12,flexWrap:"wrap"}}><span style={{fontSize:13,fontWeight:600,color:C.dark}}>Final-Modell-Hyperparameter</span><span style={{fontSize:10,background:"#fffbeb",color:"#7a5a00",padding:"3px 12px",borderRadius:12,border:"1px solid #e8d49c",fontWeight:600,letterSpacing:0.2}}>Direkt verwendet</span></div><Info>Final-Model-Tuning ist deaktiviert. Diese Parameter werden direkt für das CATE-Effektmodell (model_final) von NonParamDML und DRLearner verwendet.</Info>{_isBoth ? (<>
<div style={{fontSize:12,fontWeight:600,color:C.dark,marginBottom:6,marginTop:4}}>CatBoost-Parameter (Final-Modell)</div>
<FixedParamsEditor params={(cfg.fmtFixed||{}).catboost||{}} defaults={CB_FINAL_DEFAULTS} onChange={v=>set({...cfg,fmtFixed:{...(cfg.fmtFixed||{}),catboost:v}})} label="CatBoost CATE-Parameter (Fixed)" accent="#D4A853"/>
<div style={{fontSize:12,fontWeight:600,color:C.dark,marginBottom:6,marginTop:14}}>LightGBM-Parameter (Final-Modell)</div>
<FixedParamsEditor params={(cfg.fmtFixed||{}).lgbm||{}} defaults={LGBM_FINAL_DEFAULTS} onChange={v=>set({...cfg,fmtFixed:{...(cfg.fmtFixed||{}),lgbm:v}})} label="LightGBM CATE-Parameter (Fixed)" accent="#D4A853"/>
</>) : (<FixedParamsEditor params={cfg.fmtFixed||{}} defaults={(cfg.baseLearner||"catboost")==="catboost"?CB_FINAL_DEFAULTS:LGBM_FINAL_DEFAULTS} onChange={v=>set({...cfg,fmtFixed:v})} label="CATE-Parameter (Fixed)" accent="#D4A853"/>)}</>}</Sec>{((cfg.models||[]).includes("CausalForestDML")||(cfg.models||[]).includes("CausalForest"))&&<Sec title="CausalForest-Tuning" accent="#2d6a4f"><Info>Optimiert 4 kausale Parameter (max_depth, min_weight_fraction_leaf, min_var_fraction_leaf, criterion) via Optuna. Restliche Forest-Parameter auf EconML-Defaults fixiert (n_estimators=100 (Tuning) / 500 (Production), min_samples_leaf=5, max_samples=0.45). Bewertung per QiniScorer (RCT) oder RScorer (obs.), konfigurierbar über Scorer-Auswahl; bei Multi-Treatment immer RScorer. n_estimators=100 beim Tuning, 500 bei Production.</Info><Row><Col><Toggle label="CausalForest-Tuning aktivieren" checked={cfg.cfTune||false} onChange={v=>{const u={cfTune:v};if(v&&(cfg.cfTuneModels||[]).length===0){u.cfTuneModels=(cfg.models||[]).filter(m=>["CausalForestDML","CausalForest"].includes(m));}set({...cfg,...u});}}/></Col><Col><Toggle label="Single-Fold-Tuning" checked={cfg.cfSingleFold||false} onChange={v=>set({...cfg,cfSingleFold:v})} help="Bewertet jeden Trial auf 1 statt K Folds — schneller, aber verrauschtere Score-Schätzung."/></Col></Row>{cfg.cfTune&&<><Row><Col><div style={{fontSize:12,fontWeight:500,color:C.text,marginBottom:4}}>Scorer</div><select value={cfg.cfScorer||"auto"} onChange={e=>{const v=e.target.value;set({...cfg,cfScorer:v,fmtScorer:v});}} style={{width:"100%",padding:"7px 10px",borderRadius:8,border:"1px solid "+C.border,fontSize:12.5,background:"#fff"}}><option value="auto">{cfg.treatmentType==="multi"?"auto (R-Score — bei Multi-Treatment immer)":"auto (Qini bei RCT, R-Score bei Beobachtungsdaten)"}</option><option value="qini" disabled={cfg.treatmentType==="multi"}>Qini — optimiert Ranking-Qualität (OOF-aggregiert){cfg.treatmentType==="multi"?" — nicht bei Multi-Treatment (binär-only)":""}</option><option value="rscore">R-Score — optimiert CATE-Genauigkeit (EconML RScorer)</option></select><div style={{fontSize:10.5,color:C.textSec,marginTop:3}}>Qini: Direkt auf der Uplift-Metrik tunen (kein Pruning, robust bei schwachem Signal). R-Score: Pointwise CATE-MSE (Pruning möglich, EconML-Standard). Wird synchron mit FMT gesetzt.{cfg.treatmentType==="multi"?" Bei Multi-Treatment ist nur R-Score möglich (Qini verlangt binäres T und 1-d CATE).":""}</div></Col></Row><Divider/><div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:8}}>Tuning-Plan</div>{(() => {
              const hasCF = (cfg.cfTuneModels||[]).includes("CausalForestDML");
              const hasGRF = (cfg.cfTuneModels||[]).includes("CausalForest");
              const nTrials = cfg.cfTrials || 50;
              const cfSF = cfg.cfSingleFold;
              const nStudies = (hasCF?1:0)+(hasGRF?1:0);
              const dc = cfg.dmlCrossfitFolds||5;
              const mc = cfg.mcIters||1;
              const fitsPerCfdmlFold = mc*dc*2+1;
              const _cfScRes = cfg.treatmentType==="multi" ? "rscore" : ((cfg.cfScorer||"auto")==="auto" ? ((cfg.studyType||"rct")==="rct"?"qini":"rscore") : cfg.cfScorer);
              const scorerFitsPerFold = _cfScRes==="qini" ? 0 : 2*2; // RScorer: cv=2 × (model_y+model_t); QiniScorer: 0
              const evalFolds = cfSF ? 1 : dc;
              const cfdmlPreFit = evalFolds * (fitsPerCfdmlFold + scorerFitsPerFold);
              const cfdmlTrialFits = nTrials * evalFolds;
              const cfdmlTotal = cfdmlPreFit + cfdmlTrialFits;
              const cfScorerFits = evalFolds * scorerFitsPerFold; // RScorer setup for GRF
              const cfTotal = nTrials * evalFolds + cfScorerFits;
              return (<>
                <div style={{display:"flex",gap:10,marginBottom:12,flexWrap:"wrap"}}>
                  <div style={{padding:"8px 16px",background:"linear-gradient(135deg,#2d6a4f,#40916c)",borderRadius:8,color:"#fff"}}><div style={{fontSize:22,fontWeight:700}}>{nStudies}</div><div style={{fontSize:10,textTransform:"uppercase",letterSpacing:0.5,opacity:0.7}}>Optuna-Studies</div></div>
                  <div style={{padding:"8px 16px",background:"#e8f5ec",borderRadius:8,border:"1px solid #a3d9b1"}}><div style={{fontSize:22,fontWeight:700,color:"#2d6a4f"}}>{((hasCF?cfdmlTotal:0)+(hasGRF?cfTotal:0)).toLocaleString()}</div><div style={{fontSize:10,textTransform:"uppercase",letterSpacing:0.5,color:"#888"}}>Modell-Fits</div></div>
                  <div style={{padding:"8px 16px",background:"#faf7f7",borderRadius:8,border:"1px solid #ede6e7"}}><div style={{fontSize:22,fontWeight:700,color:"#333"}}>{nTrials}</div><div style={{fontSize:10,textTransform:"uppercase",letterSpacing:0.5,color:"#888"}}>Trials / Study</div></div>
                </div>
                <div style={{fontSize:10.5,color:"#888",marginBottom:10,lineHeight:1.5}}>
                  Pre-fit: Nuisance-Modelle einmalig pro Fold gecacht (cache_values). Pro Trial: nur Forest-Parameter ändern + refit. Details zur Ausführung unter „Tuning-Intensität".
                </div>
                <div style={{border:"1px solid #ede6e7",borderRadius:8,overflow:"hidden"}}>
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 80px",fontSize:11,fontWeight:600,padding:"8px 14px",background:"#2d6a4f",color:"#fff"}}><span>Modell</span><span>Berechnung</span><span>Fits</span></div>
                  {hasCF && <div style={{borderBottom:"1px solid #f0f0f0"}}>
                    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 80px",padding:"8px 14px",fontSize:12}}>
                      <span style={{fontWeight:600}}>CausalForestDML</span>
                      <span style={{fontSize:10.5,color:"#2d6a4f",fontFamily:MONO}}>{evalFolds}F×({fitsPerCfdmlFold}+{scorerFitsPerFold}) + {nTrials}T×{evalFolds}F</span>
                      <span style={{fontFamily:MONO,fontWeight:700,color:"#2d6a4f"}}>{cfdmlTotal.toLocaleString()}</span>
                    </div>
                    <div style={{fontSize:10.5,color:"#999",padding:"0 14px 6px",lineHeight:1.4}}>Pre-fit: {evalFolds}F × (Nuisance {dc}×model_y + {dc}×model_t + 1×Forest + RScorer cv=2). Pro Trial: {evalFolds}× refit_final() + scorer.score(est). 4 Parameter (3 EconML + criterion).</div>
                  </div>}
                  {hasGRF && <div style={{background:hasCF?"#fdfbfb":"#fff"}}>
                    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 80px",padding:"8px 14px",fontSize:12}}>
                      <span style={{fontWeight:600}}>CausalForest</span>
                      <span style={{fontSize:10.5,color:"#2d6a4f",fontFamily:MONO}}>{nTrials}T × {evalFolds}F + {evalFolds}×{scorerFitsPerFold} RScorer</span>
                      <span style={{fontFamily:MONO,fontWeight:700,color:"#2d6a4f"}}>{cfTotal.toLocaleString()}</span>
                    </div>
                    <div style={{fontSize:10.5,color:"#999",padding:"0 14px 6px",lineHeight:1.4}}>Pre-fit: {evalFolds}F × RScorer cv=2. Pro Trial: {evalFolds}× CausalForestAdapter.fit() + scorer.score(adapter). 4 Parameter (3 EconML + criterion). n_est=100 Tuning, 500 Prod.</div>
                  </div>}
                </div>
              </>);
            })()}<Divider/><div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:8}}>Modelle für CausalForest-Tuning</div><Info>Wähle, welche Forest-Modelle per Optuna optimiert werden. Nicht ausgewählte Modelle verwenden Default-Parameter.</Info><div style={{display:"flex",flexWrap:"wrap",gap:10,marginBottom:14}}>{(cfg.models||[]).filter(m=>["CausalForestDML","CausalForest"].includes(m)).map(m=>{const active=(cfg.cfTuneModels||[]).includes(m);return(<label key={m} style={{display:"inline-flex",alignItems:"center",gap:6,padding:"6px 14px",borderRadius:8,border:active?"1.5px solid #2d6a4f":"1.5px solid "+C.border,background:active?"#e8f5ec":"#fff",cursor:"pointer",fontSize:12.5,fontWeight:active?600:400,transition:"all 0.15s"}}><input type="checkbox" checked={active} style={{accentColor:"#2d6a4f"}} onChange={e=>{const s=new Set(cfg.cfTuneModels||[]);e.target.checked?s.add(m):s.delete(m);set({...cfg,cfTuneModels:[...s]})}}/>{m}</label>)})}{!(cfg.models||[]).some(m=>["CausalForestDML","CausalForest"].includes(m))&&<Info type="warn">Keine Forest-Modelle (CausalForestDML/CausalForest) in der Modellauswahl.</Info>}{(cfg.models||[]).some(m=>["CausalForestDML","CausalForest"].includes(m)) && (cfg.cfTuneModels||[]).length===0 && <Info type="warn">Kein Modell für CausalForest-Tuning ausgewählt. Alle verwenden Default-Parameter.</Info>}</div><Divider/>{(()=>{
const trials = cfg.cfTrials||50;
const trialColor = trials < 15 ? "#dc2626" : trials < 30 ? "#d97706" : "#059669";
return (<>
<div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:8}}>Tuning-Intensität</div>
<div style={{display:"flex",gap:8,marginBottom:8}}>
  {[{t:30,l:"Schnell",d:"Grenzwertig, aber schnell"},{t:50,l:"Standard",d:"Solide TPE-Exploration"},{t:100,l:"Gründlich",d:"Hohe Suchqualität"}].map(p => {
    const active = trials === p.t;
    return <button key={p.t} onClick={()=>set({...cfg, cfTrials: p.t})} style={{flex:1,padding:"10px 12px",borderRadius:10,border:active?"1.5px solid #2d6a4f":"1.5px solid "+C.border,background:active?"#e8f5ec":"#fff",cursor:"pointer",textAlign:"left",transition:"all 0.15s"}}>
      <div style={{fontSize:13,fontWeight:600,color:active?"#2d6a4f":C.dark}}>{p.l}</div>
      <div style={{fontSize:11,color:active?"#2d6a4f":C.textMuted,marginTop:2}}>{p.t} Trials</div>
    </button>;
  })}
</div>
<div style={{fontSize:11,color:C.textSec,background:"#f9fafb",padding:"8px 12px",borderRadius:8,lineHeight:1.5,border:"1px solid #e5e7eb"}}>
  CausalForest-Tuning läuft <strong>sequentiell</strong> (CausalForestDML: Nuisance einmalig mit cache_values gecacht, Trials via setattr() + refit_final(). CausalForest: via CausalForestAdapter). Scorer: {cfg.treatmentType==="multi"?"R-Score (Multi-Treatment)":cfg.cfScorer==="qini"?"OOF-Qini":cfg.cfScorer==="rscore"?"R-Score (RScorer)":"auto ("+((cfg.studyType||"rct")==="rct"?"Qini":"R-Score")+")"}. Alle CPU-Kerne pro Fit (n_jobs=-1). <strong style={{color:trialColor}}>{trials}</strong> Trials insgesamt.
  {trials < 15 && " TPE kann bei weniger als 15 Trials kaum zwischen guten und schlechten Parametern unterscheiden."}
  {trials >= 15 && trials < 30 && " Grenzwertig — TPE beginnt erst nach den Startup-Trials zu lernen."}
  {trials >= 30 && " Genug Trials für stabile TPE-Exploration und Exploitation."}
</div>
<Expander title="Manuell anpassen">
  <Inp label="Trials" type="number" value={trials} onChange={v=>set({...cfg,cfTrials:Number(v)})} help="Anzahl Optuna-Trials. Nuisance-Modelle werden einmalig gecacht, Trials fitten nur den Forest (sequentiell, alle Kerne)."/>
</Expander>
</>);
})()}{cfg.cfSingleFold && <div style={{fontSize:11,color:C.textMuted,background:C.rose,padding:"6px 12px",borderRadius:8,marginTop:6,lineHeight:1.4}}>Single-Fold aktiv: Jeder Trial wird auf <strong style={{color:"#2d6a4f"}}>1</strong> statt {cfg.dmlCrossfitFolds||5} Folds evaluiert — {cfg.dmlCrossfitFolds||5}× schneller.</div>}</>}{cfg.cfTune && <><Divider/>
{(()=>{
  const cfSS = cfg._cfSS || {};
  const g = k => ({low: cfSS[k]?.low ?? ({min_weight_fraction_leaf:0.0001,min_var_fraction_leaf:0.0005})[k], high: cfSS[k]?.high ?? ({min_weight_fraction_leaf:0.05,min_var_fraction_leaf:0.05})[k]});
  const s = (k,l,h) => set({...cfg, _cfSS:{...cfSS,[k]:{low:l,high:h}}});
  return (<div>
    <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:12}}>
      <span style={{fontSize:13.5,fontWeight:700,color:"#2d6a4f"}}>Suchraum (4 Parameter: 3 EconML + criterion)</span>
      <span style={{fontSize:10.5,color:"#2d6a4f",opacity:0.7,background:"#2d6a4f12",padding:"2px 10px",borderRadius:10,border:"1px solid #2d6a4f30",fontWeight:600}}>Optuna</span>
    </div>
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"0 14px"}}>
      <div style={{marginBottom:8,padding:"9px 14px",background:"#faf7f7",borderRadius:R,border:"1px solid "+C.borderLight}}>
        <div style={{display:"flex",justifyContent:"space-between",marginBottom:6}}>
          <span style={{fontSize:11.5,fontWeight:600,color:"#444"}}>max_depth</span>
          <span style={{fontSize:10,color:"#6b7280"}}>kategorisch</span>
        </div>
        <div style={{display:"flex",gap:3,flexWrap:"wrap"}}>
          {[3,5,7,10,15,"None"].map(v=>{
            const vals = cfg._cfDepthChoices || [3,5,7,10,15,"None"];
            const active = vals.includes(v);
            return (<button key={v} onClick={()=>{const nv = active ? vals.filter(x=>x!==v) : [...vals,v]; if(nv.length>0) set({...cfg, _cfDepthChoices:nv});}} style={{padding:"2px 7px",borderRadius:4,border:active?"1.5px solid #2d6a4f":"1px solid "+C.borderLight,background:active?"#ecfdf5":"#fff",color:active?"#2d6a4f":"#aaa",fontSize:10.5,fontFamily:MONO,fontWeight:active?600:400,cursor:"pointer",transition:"all 0.15s",lineHeight:"16px"}}>{v === "None" ? "∞" : v}</button>);
          })}
        </div>
      </div>
      <div style={{marginBottom:8,padding:"9px 14px",background:"#faf7f7",borderRadius:R,border:"1px solid "+C.borderLight}}>
        <div style={{display:"flex",justifyContent:"space-between",marginBottom:6}}>
          <span style={{fontSize:11.5,fontWeight:600,color:"#444"}}>criterion</span>
          <span style={{fontSize:10,color:"#6b7280"}}>Split-Qualität</span>
        </div>
        <div style={{display:"flex",gap:3}}>
          {["mse","het"].map(v=>{
            const vals = cfg._cfCriterionChoices || ["mse","het"];
            const active = vals.includes(v);
            return (<button key={v} onClick={()=>{const nv = active ? vals.filter(x=>x!==v) : [...vals,v]; if(nv.length>0) set({...cfg, _cfCriterionChoices:nv});}} style={{padding:"2px 10px",borderRadius:4,border:active?"1.5px solid #2d6a4f":"1px solid "+C.borderLight,background:active?"#ecfdf5":"#fff",color:active?"#2d6a4f":"#aaa",fontSize:10.5,fontFamily:MONO,fontWeight:active?600:400,cursor:"pointer",transition:"all 0.15s",lineHeight:"16px"}}>{v}</button>);
          })}
        </div>
      </div>
      {Object.entries(CF_SS).map(([k,d])=>{const r=g(k);const lbl=d.log?k+" (log)":k;return (<RSlider key={k} label={lbl} min={d.min} max={d.max} step={d.step} low={r.low} high={r.high} type={d.type} onLow={v=>s(k,v,r.high)} onHigh={v=>s(k,r.low,v)} accent="#2d6a4f"/>);})}
    </div>
  </div>);
})()}
<Divider/><Expander title="Forest-Parameter (Defaults / Fixed)" accent="#2d6a4f"><div style={{fontSize:11,color:"#6b7280",lineHeight:1.5,marginBottom:10}}>Diese Parameter werden als Ausgangswerte verwendet. Bei aktivem Tuning überschreibt Optuna die Parameter innerhalb des Suchraums — hier festgelegte Werte gelten dann nur für Parameter, die nicht im Suchraum liegen.</div><FixedParamsEditor params={cfg.cfFixed||{}} defaults={CF_FOREST_DEFAULTS} onChange={v=>set({...cfg,cfFixed:v})} label="Forest-Parameter" accent="#2d6a4f"/></Expander>
<Expander title="Overfit-Penalty (CausalForest)"><div style={{fontSize:11,color:C.textSec,lineHeight:1.5,marginBottom:10}}>Bestraft Konfigurationen, deren In-Sample-Score (Train) deutlich besser ist als der OOF-Score (Val). Bei 0 (Default) wird nur der OOF-Score optimiert — kein zusätzlicher Rechenaufwand. Bei Penalty &gt; 0 wird pro Fold zusätzlich effect(X_train) berechnet (~50% mehr Rechenzeit).</div><Row><Col><Sld label="Penalty-Faktor" min={0} max={1} step={0.05} value={cfg.cfOverfitPenalty||0} onChange={v=>set({...cfg,cfOverfitPenalty:v})} help="Stärke der Bestrafung. 0 = aus, 0.2 = moderat, 0.35 = stark."/></Col><Col><Sld label="Toleranz (Gap-Schwelle)" min={0} max={0.2} step={0.01} value={cfg.cfOverfitTolerance===undefined?0.1:cfg.cfOverfitTolerance} onChange={v=>set({...cfg,cfOverfitTolerance:v})} help="Relativer Gap-Schwellwert: 0.1 = 10% des Score-Betrags."/></Col></Row>{(cfg.cfOverfitPenalty||0)>0 && <div style={{fontSize:11,color:"#92400e",background:"#fffbeb",padding:"8px 12px",borderRadius:8,marginTop:6,lineHeight:1.4,border:"1px solid #fbbf24"}}><strong>Aktiv:</strong> Penalty={cfg.cfOverfitPenalty}, Toleranz={cfg.cfOverfitTolerance===undefined?0.1:cfg.cfOverfitTolerance}. Zusätzlicher Rechenaufwand: ~50% pro Trial (Train-Predictions).</div>}</Expander>
</>}
{!cfg.cfTune && <><Divider/><div style={{display:"flex",alignItems:"center",gap:10,marginBottom:12,flexWrap:"wrap"}}><span style={{fontSize:13,fontWeight:600,color:C.dark}}>Forest-Parameter</span><span style={{fontSize:10,background:"#f0fdf4",color:"#059669",padding:"3px 12px",borderRadius:12,border:"1px solid #a7f3d0",fontWeight:600,letterSpacing:0.2}}>Direkt verwendet</span></div><Info>CausalForest-Tuning ist deaktiviert. Diese Parameter werden direkt für alle Forest-Modelle (CausalForestDML, CausalForest) verwendet.</Info><FixedParamsEditor params={cfg.cfFixed||{}} defaults={CF_FOREST_DEFAULTS} onChange={v=>set({...cfg,cfFixed:v})} label="Forest-Parameter" accent="#2d6a4f"/></>}</Sec>}</>);
};

const PSelection = ({cfg,set}) => {
  const mt=cfg.treatmentType==="multi";
  const _isObs = (cfg.studyType||"rct") !== "rct";
  const mo=mt?["policy_value","policy_value_T1","qini_T1","auuc_T1"].filter(m=>!_isObs||!m.startsWith("policy"))
    :["qini","auuc","uplift_at_10pct","uplift_at_20pct","uplift_at_50pct",...(_isObs?[]:["policy_value"])];
  const metricDescs={
    qini:"Fläche unter der Qini-Kurve. Misst den kumulativen Uplift über alle Dezile. Standard bei Binary Treatment.",
    auuc:"Area Under Uplift Curve. Ähnlich wie Qini, normiert auf zufällige Behandlung.",
    uplift_at_10pct:"Uplift in den Top-10% der Kunden mit höchstem CATE. Für aggressive Targeting-Strategien.",
    uplift_at_20pct:"Uplift in den Top-20%. Guter Kompromiss zwischen Reichweite und Genauigkeit.",
    uplift_at_50pct:"Uplift in der oberen Hälfte. Konservativere Strategie mit breiterem Targeting.",
    policy_value:"Policy Value: Gesamtnutzen der Policy über alle Personen (Uplift + vermiedener Schaden, gewichtet). Bei MT: Optimale Zuweisung über alle Arme.",
    policy_value_T1:"Policy Value für Treatment-Arm 1.",
    qini_T1:"Qini für Treatment-Arm 1.",
    auuc_T1:"AUUC für Treatment-Arm 1.",
  };

  return (<>
    <Sec title="Champion-Auswahl">
      <Info>Nach der Evaluation wird automatisch ein <strong>Champion</strong> gekürt – das Modell mit dem besten Wert auf der gewählten Metrik.</Info>
      <Sel label="Auswahl-Metrik" options={mo} value={cfg.selMetric||(mt?"policy_value":"qini")} onChange={v=>set({...cfg,selMetric:v})}/>
      <div style={{fontSize:11.5,color:C.textMuted,lineHeight:1.5,marginTop:-6,marginBottom:14}}>{metricDescs[cfg.selMetric||"qini"]||""}</div>
      <Expander title="Erweiterte Champion-Einstellungen">
        <Row>
          <Col><Toggle label="Höher = besser" checked={cfg.higherBetter!==false} onChange={v=>set({...cfg,higherBetter:v})} help="Für die meisten Uplift-Metriken gilt: höher = besser. Nur bei Spezialmetriken umkehren."/></Col>
          <Col><Sel label="Manueller Champion" options={["(automatisch)",...(cfg.models||[])]} value={cfg.manualChamp||"(automatisch)"} onChange={v=>set({...cfg,manualChamp:v==="(automatisch)"?null:v})} help="Überschreibt die automatische Auswahl."/></Col>
        </Row>
      </Expander>
    </Sec>

    <Sec title="Surrogate-Einzelbaum" accent="#D4A853">
      <Info>Ein interpretierbarer <strong>Entscheidungsbaum</strong>, der die CATE-Predictions des Champions nachlernt (Teacher-Student-Prinzip). Nützlich für Fachbereiche, die regelbasierte Erklärungen benötigen, oder wenn regulatorische Anforderungen keine Black-Box-Modelle erlauben.</Info>
      <Toggle label="Surrogate-Tree aktivieren" checked={cfg.surrEnabled} onChange={v=>set({...cfg,surrEnabled:v})}/>
      {cfg.surrEnabled&&<>
        <Divider/>
        <Info type="warn">Der Surrogate-Baum approximiert den Champion – er ist nicht das Champion-Modell selbst. Je weniger Blätter, desto interpretierbarer, aber auch ungenauer.</Info>
        <Row>
          <Col><Sld label="Min. Samples pro Blatt" min={10} max={500} step={10} value={cfg.surrMinLeaf||50} onChange={v=>set({...cfg,surrMinLeaf:v})} help="Verhindert zu kleine Segmente. Höher = stabilere Regeln."/></Col>
          <Col><Sld label="Max. Blätter" min={2} max={128} step={1} value={cfg.surrLeaves||31} onChange={v=>set({...cfg,surrLeaves:v})} help="Komplexität des Baums. 8–16 für einfache Regeln, 31+ für mehr Detail."/></Col>
          <Col><Sld label="Max. Tiefe (0=unbegrenzt)" min={0} max={20} step={1} value={cfg.surrDepth||0} onChange={v=>set({...cfg,surrDepth:v})} help="Alternative Komplexitätsbegrenzung. 0 = nur durch Blätter begrenzt."/></Col>
        </Row>
      </>}
    </Sec>

    <Sec title="Bundle-Export" accent="#059669">
      <Info>Exportiert alle Artefakte in ein Verzeichnis: Modelle (.pkl), Preprocessor, Config, Registry. Grundlage für <strong>run_production.py</strong> (Batch-Scoring) und <strong>run_promote.py</strong> (Champion wechseln). Ohne Bundle können Modelle nicht außerhalb der Analyse-Pipeline verwendet werden.</Info>
      <Toggle label="Bundle-Export aktivieren" checked={cfg.bundleEnabled} onChange={v=>set({...cfg,bundleEnabled:v})}/>
      {cfg.bundleEnabled&&<>
        <Divider/>
        <Row>
          <Col>
          </Col>
          <Col>
            <Toggle label="Bundle in MLflow loggen" checked={cfg.bundleMlflow!==false} onChange={v=>set({...cfg,bundleMlflow:v})} help="Das Bundle-Verzeichnis als MLflow-Artifact hochladen. Macht es über das MLflow-UI auffindbar."/>
          </Col>
        </Row>
        <div style={{fontSize:11,color:C.textMuted,marginTop:8}}>Ziel: <code style={{fontSize:10,background:C.rose,padding:"1px 6px",borderRadius:4}}>{cfg.bundleDir||"runs/bundles"}/&lt;timestamp-id&gt;/</code></div>
      </>}
    </Sec>
  </>);
};

const PExplain = ({cfg,set}) => (<>
  <Sec title="SHAP-Analyse" accent="#D4A853">
    <Info>Berechnet SHAP-Werte für den Champion: Feature-Wichtigkeit (global) und Feature-Effekte (lokal). Wird nach der Analyse automatisch ausgeführt. EconML-Modelle nutzen den nativen SHAP-Pfad, alle anderen den generischen TreeExplainer/KernelExplainer.</Info>
    <Toggle label="SHAP-Analyse aktivieren" checked={cfg.explEnabled} onChange={v=>set({...cfg,explEnabled:v})}/>
    {cfg.explEnabled&&<>
      <Divider/>
      <Row>
        <Col><Inp label="Sample Size" type="number" value={cfg.explSampleSize||10000} onChange={v=>set({...cfg,explSampleSize:Number(v)})} help="Anzahl Beobachtungen für SHAP. Bei großen Datensätzen wird gesampled."/></Col>
        <Col><Inp label="Top-N Features" type="number" value={cfg.explTopN||20} onChange={v=>set({...cfg,explTopN:Number(v)})} help="Anzahl Features im Importance-Barplot und in den SHAP-Dependency-Plots"/></Col>
        <Col><Sld label="Bins (Dependency-Plots)" min={4} max={20} step={1} value={cfg.shapBins||10} onChange={v=>set({...cfg,shapBins:v})} help="Anzahl Bins für SHAP-Dependency-Plots"/></Col>
      </Row>
    </>}
  </Sec>

</>);
