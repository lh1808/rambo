const PConfig = ({cfg,set}) => {
  const [workDirInfo,setWorkDirInfo] = useState(null);
  useEffect(() => { fetch("./api/work-dir").then(r=>r.json()).then(d=>setWorkDirInfo(d)).catch(()=>{}); }, []);
  return (<>
  <Sec title="Arbeitsverzeichnis">
    <Info>Alle erzeugten Artefakte landen hier. Konfigurierbar per Env <code style={{fontSize:10,background:C.rose,padding:"1px 6px",borderRadius:4}}>RUBIN_WORK_DIR</code> oder <code style={{fontSize:10,background:C.rose,padding:"1px 6px",borderRadius:4}}>constants.work_dir</code>.</Info>
    {workDirInfo ? (<>
    <div style={{background:"#f6f8fa",border:"1px solid "+C.border,borderRadius:8,padding:"10px 14px",fontSize:11,fontFamily:MONO,lineHeight:1.7}}>
      <div><span style={{color:"#57606a",minWidth:80,display:"inline-block"}}>Pfad:</span> <strong style={{color:C.dark}}>{workDirInfo.work_dir}</strong></div>
      <div><span style={{color:"#57606a",minWidth:80,display:"inline-block"}}>Quelle:</span> {workDirInfo.source}</div>
      <div><span style={{color:"#57606a",minWidth:80,display:"inline-block"}}>Upload-Limit:</span> {workDirInfo.max_upload_mb || 500} MB</div>
      <div style={{marginTop:6,color:"#57606a",fontSize:10}}>├── mlruns/ — MLflow Tracking</div>
      <div style={{color:"#57606a",fontSize:10}}>├── data/ — DataPrep-Ausgabe</div>
      <div style={{color:"#57606a",fontSize:10}}>├── bundles/ — Bundle-Export</div>
      <div style={{color:"#57606a",fontSize:10}}>├── configs/ — Gespeicherte Konfigurationen</div>
      <div style={{color:"#57606a",fontSize:10}}>├── exports/ — Feature Dictionary</div>
      <div style={{color:"#57606a",fontSize:10}}>├── uploads/ — Server-Uploads</div>
      <div style={{color:"#57606a",fontSize:10}}>└── .rubin_cache/ — Reports, Pipeline-Cache</div>
    </div>
    </>) : (
      <div style={{fontSize:11,color:"#6b7280",background:"#f9fafb",padding:"8px 12px",borderRadius:8,border:"1px solid #e5e7eb",lineHeight:1.5}}>Server-Verbindung wird hergestellt — Pfad wird automatisch angezeigt.</div>
    )}
    <Inp label="Eigenes Arbeitsverzeichnis (optional)" type="text" value={cfg.workDir||""} onChange={v=>set({...cfg,workDir:v||null})} help="Überschreibt den Standard-Pfad ./runs. Leer lassen für den Standard."/>
  </Sec>

  <Sec title="Parallelisierung">
    <Info>Steuert, wie viele CPU-Kerne gleichzeitig genutzt werden. Höhere Level beschleunigen die Pipeline, benötigen aber mehr RAM.</Info>
    <div style={{display:"grid",gridTemplateColumns:"repeat(4, 1fr)",gap:8}}>
      {[{v:1,l:"Minimal",d:"1 Kern, sequentiell. Sicher, minimaler RAM."},{v:2,l:"Moderat",d:"Alle Kerne pro Fit, Folds sequentiell. Guter Kompromiss."},{v:3,l:"Hoch",d:"Folds und Trials parallel. DRTester reduziert. Mehr RAM."},{v:4,l:"Maximum",d:"Max. Parallelisierung. DRTester nur Champion. Höchster RAM."}].map(o=>{
        const active=(cfg.parallelLevel||3)===o.v;
        return(<button key={o.v} onClick={()=>set({...cfg,parallelLevel:o.v})} style={{padding:"12px 14px",borderRadius:10,border:active?"2px solid #9B111E":"1.5px solid "+C.border,background:active?"#FDF2F3":"#fff",cursor:"pointer",textAlign:"left",transition:"all 0.15s"}}>
          <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:4}}><span style={{fontSize:13,fontWeight:active?700:500,color:active?"#6B0D15":"#24292f"}}>{o.l}</span><span style={{fontSize:10,color:"#999"}}>L{o.v}</span></div>
          <div style={{fontSize:10.5,color:"#57606a",lineHeight:1.4}}>{o.d}</div>
        </button>);
      })}
    </div>
  </Sec>

  <Sec title="Seeds (Reproduzierbarkeit)">
    <Info>Zwei getrennte Seeds für Training und Tuning verhindern Val-Set-Overfitting. Gleiche Seeds = gleiche Ergebnisse bei identischen Daten.</Info>
    <Row><Col>
    <div>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:8}}>
        <span style={{fontSize:12,fontWeight:600,color:C.text}}>Random Seed</span>
        <span style={{color:C.ruby,fontFamily:MONO,fontSize:13,background:C.rose,padding:"2px 12px",borderRadius:6,fontWeight:700,minWidth:36,textAlign:"center"}}>{cfg.seed}</span>
      </div>
      <style>{`
        .random-seed-slider { width:100%; }
        .random-seed-slider::-webkit-slider-runnable-track { background: linear-gradient(to right, #9B111E 0%, #9B111E var(--pct), #d0d7de var(--pct), #d0d7de 100%); height:6px; border-radius:3px; }
        .random-seed-slider::-moz-range-track { background: #d0d7de; height:6px; border-radius:3px; }
        .random-seed-slider::-moz-range-progress { background: #9B111E; height:6px; border-radius:3px; }
        .random-seed-slider::-webkit-slider-thumb { -webkit-appearance:none; width:16px; height:16px; border-radius:50%; background:#9B111E; border:2px solid #fff; margin-top:-5px; cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,0.15); }
        .random-seed-slider::-moz-range-thumb { width:12px; height:12px; border-radius:50%; background:#9B111E; border:2px solid #fff; cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,0.15); }
      `}</style>
      <input type="range" min={0} max={999} step={1} value={cfg.seed||42} onChange={e=>set({...cfg,seed:Number(e.target.value)})} className="random-seed-slider" style={{"--pct":`${(cfg.seed||42)/999*100}%`,WebkitAppearance:"none",appearance:"none",background:"transparent"}}/>
      <div style={{fontSize:10.5,color:C.textMuted,marginTop:6,lineHeight:1.4}}>Cross-Prediction Seed: gleicher Seed = gleiche Fold-Zuordnung bei identischen Daten</div>
    </div>
    </Col><Col>
    <div>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:8}}>
        <span style={{fontSize:12,fontWeight:600,color:C.text}}>Tuning Seed</span>
        <span style={{color:"#D4A853",fontFamily:MONO,fontSize:13,background:"#fef9ee",padding:"2px 12px",borderRadius:6,fontWeight:700,minWidth:36,textAlign:"center"}}>{cfg.tuningSeed}</span>
      </div>
      <style>{`
        .tuning-seed-slider { width:100%; }
        .tuning-seed-slider::-webkit-slider-runnable-track { background: linear-gradient(to right, #D4A853 0%, #D4A853 var(--pct), #d0d7de var(--pct), #d0d7de 100%); height:6px; border-radius:3px; }
        .tuning-seed-slider::-moz-range-track { background: #d0d7de; height:6px; border-radius:3px; }
        .tuning-seed-slider::-moz-range-progress { background: #D4A853; height:6px; border-radius:3px; }
        .tuning-seed-slider::-webkit-slider-thumb { -webkit-appearance:none; width:16px; height:16px; border-radius:50%; background:#D4A853; border:2px solid #fff; margin-top:-5px; cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,0.15); }
        .tuning-seed-slider::-moz-range-thumb { width:12px; height:12px; border-radius:50%; background:#D4A853; border:2px solid #fff; cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,0.15); }
      `}</style>
      <input type="range" min={0} max={999} step={1} value={cfg.tuningSeed||18} onChange={e=>set({...cfg,tuningSeed:Number(e.target.value)})} className="tuning-seed-slider" style={{"--pct":`${(cfg.tuningSeed||18)/999*100}%`,WebkitAppearance:"none",appearance:"none",background:"transparent"}}/>
      <div style={{fontSize:10.5,color:C.textMuted,marginTop:6,lineHeight:1.4}}>Eigener Seed für Tuning-CV-Folds, damit Optuna auf <em>anderen</em> Folds bewertet als die spätere Cross-Prediction.</div>
    </div>
    </Col></Row>
    {cfg.tuningSeed===cfg.seed && <div style={{fontSize:10.5,color:"#d97706",background:"#fffbeb",padding:"6px 10px",borderRadius:6,marginTop:6,border:"1px solid #fbbf24",lineHeight:1.4}}>Tuning Seed = Random Seed — Tuning und Cross-Prediction nutzen identische Folds. Empfehlung: unterschiedliche Werte.</div>}
  </Sec>


  <Sec title="Evaluierungsmodus">
    <Info>{cfg.dpRunName
      ? "Evaluationsmodus wurde durch DataPrep festgelegt und kann hier nicht geändert werden."
      : "Bestimmt, wie die Modelle evaluiert werden. Wird bei Nutzung von DataPrep oder Daten automatisch vorbelegt."
    }</Info>
    {(()=>{
      const locked = !!cfg.dpRunName;
      const modes = [
        {k:"cross",l:"Cross-Validation",d:"K-Fold OOF-Predictions — alle Daten effizient genutzt"},
        {k:"tmes",l:"TMES",d:"Eval-Maske aus DataPrep — Evaluation nur auf ausgewähltem Subset"},
        {k:"external",l:"External Eval",d:"Separater Eval-Datensatz — leakage-frei"},
      ];
      const currentMode = cfg.eval_mask_file && cfg.validateOn!=="external" ? "tmes" : cfg.validateOn==="external" ? "external" : "cross";
      return (
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:10,marginBottom:14}}>
          {modes.map(o => {
            const active = currentMode === o.k;
            const dis = locked || (o.k==="tmes" && !cfg.eval_mask_file);
            return (
              <button key={o.k} disabled={dis} onClick={()=>{
                if(dis) return;
                if(o.k==="cross") set({...cfg, validateOn:"cross", eval_mask_file:""});
                else if(o.k==="tmes" && cfg.eval_mask_file) set({...cfg, validateOn:"cross"});
                else if(o.k==="external") set({...cfg, validateOn:"external"});
              }} style={{padding:"14px 16px",borderRadius:10,border:active?"2px solid "+C.ruby:"1.5px solid "+C.border,background:active?C.rose:"#fff",cursor:dis?"not-allowed":"pointer",opacity:(!active&&dis)?0.4:1,textAlign:"left",transition:"all 0.15s"}}>
                <div style={{display:"flex",alignItems:"center",gap:8}}>
                  <div style={{width:14,height:14,borderRadius:7,border:active?"4px solid "+C.ruby:"2px solid #ccc",background:"#fff",flexShrink:0}}/>
                  <div>
                    <div style={{fontSize:13,fontWeight:600,color:active?C.dark:"#555"}}>{o.l}{active && locked && <span style={{fontSize:9,fontWeight:600,color:C.ruby,background:C.rose,padding:"1px 6px",borderRadius:6,marginLeft:6}}>DataPrep</span>}</div>
                    <div style={{fontSize:10.5,color:C.textMuted,marginTop:2,lineHeight:1.4}}>{o.d}</div>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      );
    })()}
    <Inp label="Äußere CV-Folds (K)" type="number" value={cfg.cvSplits} onChange={v=>set({...cfg,cvSplits:v})} help="Äußere Cross-Validation für Out-of-Fold CATE-Predictions (StratifiedKFold auf T×Y). Interne CVs (BLT, FMT, DML/DR-Cross-Fitting) laufen unabhängig davon."/>
    {cfg.validateOn==="external" && (
      cfg.eval_x_file && cfg.eval_t_file && cfg.eval_y_file
        ? <Info type="success">✓ External Eval vollständig konfiguriert. Eval X: <code style={{background:"rgba(255,255,255,0.5)",padding:"1px 5px",borderRadius:3}}>{cfg.eval_x_file.split("/").pop()}</code></Info>
        : <Info type="warn">External Eval aktiviert, aber es fehlen noch Eval-Dateien. Diese kannst du auf der Seite <strong>Daten</strong> eintragen{cfg.eval_x_file?` (Eval X bereits gesetzt: ${cfg.eval_x_file.split("/").pop()}, weitere fehlen)`:""}.</Info>
    )}
    {cfg.eval_mask_file && cfg.validateOn!=="external" && <Info type="warn"><strong>TMES aktiv:</strong> Evaluation nur auf Mask-Subset. Training auf allen Daten. {Array.isArray(cfg.eval_mask_file) ? `${cfg.eval_mask_file.length} Masken per OR kombiniert.` : `Maske: ${cfg.eval_mask_file.split("/").pop()}`}</Info>}
    {cfg.eval_mask_file && cfg.validateOn==="external" && <Info type="warn"><strong>Konflikt:</strong> eval_mask_file und External Eval aktiv. Maske wird <strong>ignoriert</strong>.</Info>}
  </Sec>

  <Sec title="Innere Cross-Validation">
    <Info>{cfg.validateOn === "external"
      ? `Evaluation auf separatem Holdout-Datensatz. Jedes Modell wird EINMAL auf den Trainingsdaten gefittet und direkt auf dem Eval-Set evaluiert — kein äußeres CV. DML/DR-Modelle nutzen intern Cross-Fitting (cv=${cfg.dmlCrossfitFolds||5}) für die Nuisance-Residualisierung.`
      : cfg.eval_mask_file
      ? `TMES: Alle ${cfg.cvSplits||5}-Fold Cross-Predictions werden auf allen Daten berechnet. Evaluation (Metriken, Plots) nur auf dem Mask-Subset. DML/DR-Modelle nutzen intern Cross-Fitting (cv=${cfg.dmlCrossfitFolds||5}).`
      : `Alle Modelle durchlaufen externe ${cfg.cvSplits||5}-Fold Cross-Validation für echte Out-of-Fold CATE-Predictions. DML/DR-Modelle nutzen zusätzlich internes Cross-Fitting (cv=${cfg.dmlCrossfitFolds||5}) für die Nuisance-Residualisierung innerhalb jedes äußeren Folds.`
    }</Info>
    <Row><Col><Inp label="Innere CV-Folds" type="number" value={cfg.dmlCrossfitFolds||5} onChange={v=>set({...cfg,dmlCrossfitFolds:Number(v)})} help="Gilt für alle internen Cross-Validations: BLT, FMT und DML/DR-Nuisance-Cross-Fitting. Überall StratifiedKFold (shuffle=True) für balancierte Treatment-Gruppen. Default=5."/></Col></Row>
    {(cfg.dmlCrossfitFolds||5) > 2 && <div style={{fontSize:11,color:"#7a5a00",background:"#fffbeb",padding:"8px 12px",borderRadius:8,marginTop:6,lineHeight:1.5}}>cv={cfg.dmlCrossfitFolds}: Jedes Nuisance-Modell trainiert auf {Math.round((1-1/(cfg.dmlCrossfitFolds||5))*100)}% der Daten (statt 50% bei cv=2). Stabilere Residuals, aber {cfg.dmlCrossfitFolds||5}× statt 2× Nuisance-Fits pro DML/DR-Modell.</div>}
    <Divider/>
    <Toggle label="Monte-Carlo-Iterationen aktivieren" checked={(cfg.mcIters||0)>0} onChange={v=>set({...cfg,mcIters:v?3:null})} help="Wiederholt das interne Cross-Fitting (Nuisance) mehrfach und mittelt die Residuals. Stabilere CATE-Schätzungen, aber proportional längere Laufzeit."/>
    {(cfg.mcIters||0)>0 && <div style={{fontSize:11,color:"#7a5a00",background:"#fffbeb",padding:"8px 12px",borderRadius:8,marginTop:6,lineHeight:1.5}}>Cross-Fitting wird <strong>{cfg.mcIters}x</strong> wiederholt. Interne Fits pro DML/DR-Fold: ({cfg.dmlCrossfitFolds||5} Folds x 2 Nuisance + 1 Final) x {cfg.mcIters} mc = <strong>{((((cfg.dmlCrossfitFolds||5))*2+1)*(cfg.mcIters||1))}</strong> Nuisance-Fits. {cfg.validateOn === "external" ? `Evaluation auf separatem Holdout — kein äußeres CV, jedes Modell wird einmal auf Trainingsdaten gefittet.` : `Zusätzlich ${cfg.cvSplits||5} äußere CV-Folds für OOF-CATE-Predictions.${cfg.eval_mask_file ? " TMES: Metriken nur auf Mask-Subset." : ""}`}</div>}
    {(cfg.mcIters||0)>0 && <Row>
      <Col><Sld label="Anzahl Iterationen" min={2} max={5} step={1} value={cfg.mcIters||3} onChange={v=>set({...cfg,mcIters:Number(v)})} help="2–3 für guten Kompromiss, 5 für maximale Stabilität"/></Col>
      <Col><Sel label="Aggregation" options={["mean","median"]} value={cfg.mcAgg||"mean"} onChange={v=>set({...cfg,mcAgg:v})} help="mean (Standard) oder median (robuster bei Ausreißern)"/></Col>
    </Row>}
  </Sec>

  <Sec title="Tuning-Regularisierung" accent="#9B111E">
    <Info>Steuert, wie aggressiv Overfitting beim Tuning bestraft wird. Die BLT-Penalty wirkt <strong>ausschließlich auf Meta-Learner</strong> (S-/T-/X-Learner), die ohne internes Cross-Fitting direkt den CATE bilden; <strong>DML/DR-Nuisances werden nie bestraft</strong> (Cross-Fitting + Orthogonalität fangen deren Overfitting bereits ab — unabhängig von dieser Einstellung). Die Presets setzen eine milde Meta-Learner-BLT-Penalty plus eine Penalty auf der CATE-Finalstufe (FMT/CFT). „Stark" erhöht die Penaltys (BLT 0.2→0.3, FMT/CFT 0.2→0.35), nicht die Toleranz (relativ 20% BLT / 10% Final).</Info>
    {(()=>{
      const REG_PRESETS = [
        {key:"moderate", label:"Moderat", desc:"Milde Penalty: Meta-BLT 20%, Final (FMT/CFT) 20%. Für moderate Stichproben.",
         blt_p:0.2, blt_t:0.2, fmt_p:0.2, fmt_t:0.1, cft_p:0.2, cft_t:0.1},
        {key:"strong", label:"Stark", desc:"Starke Penalty: Meta-BLT 30%, Final (FMT/CFT) 35%. Für kleine Stichproben.",
         blt_p:0.3, blt_t:0.2, fmt_p:0.35, fmt_t:0.1, cft_p:0.35, cft_t:0.1},
      ];
      const bp=cfg.overfitPenalty||0, bt=cfg.overfitTolerance===undefined?0.2:cfg.overfitTolerance;
      const fp=cfg.fmtOverfitPenalty||0, ft=cfg.fmtOverfitTolerance===undefined?0.1:cfg.fmtOverfitTolerance;
      const cp=cfg.cfOverfitPenalty||0, ct=cfg.cfOverfitTolerance===undefined?0.1:cfg.cfOverfitTolerance;
      const matched = REG_PRESETS.find(r => r.blt_p===bp && r.blt_t===bt && r.fmt_p===fp && r.fmt_t===ft && r.cft_p===cp && r.cft_t===ct);
      const activeKey = matched ? matched.key : (bp===0 && fp===0 && cp===0 ? null : "custom");
      const toggleReg = (r) => {
        if(activeKey===r.key) set({...cfg, overfitPenalty:0, overfitTolerance:0.2, fmtOverfitPenalty:0, fmtOverfitTolerance:0.1, cfOverfitPenalty:0, cfOverfitTolerance:0.1});
        else set({...cfg, overfitPenalty:r.blt_p, overfitTolerance:r.blt_t, fmtOverfitPenalty:r.fmt_p, fmtOverfitTolerance:r.fmt_t, cfOverfitPenalty:r.cft_p, cfOverfitTolerance:r.cft_t});
      };
      return (<>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10,marginTop:4}}>
          {REG_PRESETS.map(r => {
            const active = activeKey===r.key;
            return (
              <label key={r.key} onClick={(e)=>{e.preventDefault();toggleReg(r)}} style={{display:"flex",flexDirection:"column",padding:"12px 14px",borderRadius:10,border:active?"1.5px solid "+C.ruby:"1.5px solid "+C.border,background:active?"rgba(155,17,30,0.03)":"#fff",cursor:"pointer",transition:"all 0.15s"}}>
                <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
                  <input type="checkbox" checked={active} readOnly style={{accentColor:C.ruby,pointerEvents:"none"}}/>
                  <span style={{fontSize:13,fontWeight:600,color:C.dark}}>{r.label}</span>
                </div>
                <div style={{fontSize:11,color:C.textMuted,lineHeight:1.4}}>{r.desc}</div>
              </label>
            );
          })}
        </div>
        {activeKey===null && <div style={{fontSize:11,color:"#6b7280",marginTop:8,lineHeight:1.4}}>Keine Regularisierung aktiv — Tuning optimiert rein auf Val-Score.</div>}
        {activeKey==="custom" && <div style={{fontSize:11,color:C.ruby,background:C.rose,padding:"8px 12px",borderRadius:8,marginTop:8,lineHeight:1.5,border:"1px solid "+C.border}}>
          <strong>Benutzerdefiniert:</strong> BLT p={bp}/t={bt} | FMT p={fp}/t={ft}<br/><span style={{fontSize:10,color:"#9ca3af"}}>Anpassbar unter Learner &amp; Tuning → Erweiterte Einstellungen</span>
        </div>}
      </>);
    })()}
  </Sec>

  <Sec title="Feature-Selektion" accent="#0d9488">
    <Info>Reduziert die Feature-Matrix auf die relevantesten Spalten. Der Prozess läuft in drei Stufen: (1) Korrelationsfilter entfernt redundante Features, (2) Importance-Berechnung pro Methode, (3) Top-N pro Methode per Union zusammenführen und auf das exakte Feature-Budget bringen — bei Überschreitung per Konsens-Rang kappen, bei Unterschreitung (durch Überschneidungen zwischen Methoden) mit den global bestbewerteten Features auffüllen.</Info>
    <Toggle label="Feature-Selektion aktivieren" checked={cfg.fsEnabled} onChange={v=>set({...cfg,fsEnabled:v,fsMethods:v?["catboost_importance","causal_forest"]:(cfg.fsMethods||["catboost_importance"])})}/>
    {cfg.fsEnabled&&<>
      <Divider/>
      <div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:10}}>Importance-Methoden</div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:10}}>
        {[
          {m:"catboost_importance",label:"CatBoost Gain",desc:"Outcome-Regressor mit nativen kat. Splits. Misst prädiktive Feature-Relevanz für Y."},
          {m:"lgbm_importance",label:"LightGBM Gain",desc:"Outcome-Regressor (GBDT). Gain-Importance über alle Splits — schnell, gut bei vielen Features."},
          {m:"causal_forest",label:"CausalForest",desc:"Kausale Relevanz: misst, wie stark ein Feature die Heterogenität im Treatment-Effekt treibt."},
        ].map(({m,label,desc}) => {
          const blocked = m==="causal_forest" && cfg.hasNaN;
          const checked = (cfg.fsMethods||[]).includes(m)&&!blocked;
          return (
            <label key={m} style={{display:"flex",flexDirection:"column",padding:"12px 14px",borderRadius:10,border:checked?"1.5px solid "+C.ruby:"1.5px solid "+C.border,background:checked?"rgba(155,17,30,0.03)":"#fff",opacity:blocked?0.35:1,cursor:blocked?"not-allowed":"pointer",transition:"all 0.15s"}}>
              <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
                <input type="checkbox" checked={checked} disabled={blocked} style={{accentColor:C.ruby}} onChange={e=>{const s=new Set(cfg.fsMethods||["catboost_importance"]);e.target.checked?s.add(m):s.delete(m);set({...cfg,fsMethods:[...s]})}}/>
                <span style={{fontSize:13,fontWeight:600,color:blocked?"#aaa":C.dark}}>{label}</span>
              </div>
              <div style={{fontSize:11,color:C.textMuted,lineHeight:1.4}}>{desc}{blocked?" Keine NaN-Toleranz.":""}</div>
            </label>
          );
        })}
      </div>
      <Divider/>
      <div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:10}}>Feature-Budget</div>
      <Inp label="Ziel-Feature-Anzahl" type="number" value={cfg.fsMaxFeatures||77} onChange={v=>set({...cfg,fsMaxFeatures:Number(v)})} help={`Exakte Anzahl Features nach Selektion: das Ergebnis enthält genau ${cfg.fsMaxFeatures||77} Features (sofern nach dem Korrelationsfilter genügend vorhanden sind, sonst alle verbleibenden). Bei ${(cfg.fsMethods||["catboost_importance"]).length} Methode(n) liefert jede ihre Top-${Math.ceil((cfg.fsMaxFeatures||77)/Math.max(1,(cfg.fsMethods||["catboost_importance"]).length))} (garantierte Repräsentation); die Restmenge wird per Konsens-Rang aufgefüllt bzw. gekappt.`}/>
      {(()=>{
        const nM = (cfg.fsMethods||["catboost_importance"]).length;
        const maxF = cfg.fsMaxFeatures||77;
        const perM = Math.ceil(maxF / Math.max(1, nM));
        return <div style={{fontSize:11,color:C.textSec,background:"#f9fafb",padding:"8px 12px",borderRadius:8,marginTop:6,lineHeight:1.5,border:"1px solid #e5e7eb"}}>
          <strong>{nM}</strong> Methode{nM>1?"n":""} × <strong>{perM}</strong> Features (Top-N pro Methode) → Ergebnis auf <strong>exakt {maxF}</strong> gebracht.
          {nM>1 && " Überschneidungen zwischen Methoden verkleinern nicht mehr die Endzahl: freie Budget-Plätze werden per Konsens-Rang mit den nächstbesten Features aufgefüllt."}
          {nM>1 && " Bei Überschreitung entscheidet der Konsens-Rang (Summe der Positionen über alle Methoden), welche Features bleiben."}
          {" Weniger als "}{maxF}{" nur, wenn der Pool nach dem Korrelationsfilter kleiner ist."}
        </div>;
      })()}
      <Divider/>
      <div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:10}}>Korrelationsfilter</div>
      <Sld label="Korrelations-Schwelle" min={0.5} max={1} step={0.05} value={cfg.fsCorrThresh||0.9} onChange={v=>set({...cfg,fsCorrThresh:v})} help="Schwellwert für |r|. Feature-Paare mit höherer Korrelation werden reduziert — das weniger wichtige Feature wird entfernt."/>
      <div style={{fontSize:11,color:C.textSec,background:"#f9fafb",padding:"8px 12px",borderRadius:8,marginTop:6,lineHeight:1.5,border:"1px solid #e5e7eb"}}>
        Zwei Korrelationsmaße werden nacheinander berechnet: <strong>Pearson</strong> (lineare Zusammenhänge) und <strong>Spearman</strong> (monotone, auch nicht-lineare Zusammenhänge). Ein Feature-Paar gilt als redundant, sobald es in <em>einem</em> der beiden Maße |r| {">"} {cfg.fsCorrThresh||0.9} erreicht. Das Feature mit der niedrigeren aggregierten Importance wird entfernt — seine Importance wird auf den überlebenden Partner addiert (Importance-Umverteilung). Spearman wird nur auf den nach Pearson verbliebenen Features berechnet (Performance-Optimierung).
      </div>
    </>}
  </Sec>


  <Sec title="Performance">
    <Info>Optionale Maßnahmen zur Reduktion von Laufzeit und Speicherverbrauch.</Info>
    <Row>
      <Col>
        <Toggle label="Downsampling" checked={cfg.downsample} onChange={v=>set({...cfg,downsample:v})} help="Reduziert den Datensatz vor dem Training"/>
        {cfg.downsample&&<Sld label="Anteil" min={0.01} max={1} step={0.01} value={cfg.dfFrac||0.1} onChange={v=>set({...cfg,dfFrac:v})}/>}
      </Col>
      <Col>
        <Toggle label="Memory-Reduktion" checked={cfg.reduceMem!==false} onChange={v=>set({...cfg,reduceMem:v})} help="Datentypen downcasten: float64 → float32 etc. Spart ca. 40–60% RAM."/>
      </Col>
    </Row>
  </Sec>

  <Sec title="Ausgabe">
    <Info>Ergebnisse werden automatisch in <strong>MLflow</strong> geloggt (Metriken, Cross-Predictions, Plots, Report).</Info>
    <Expander title="Erweiterte Ausgabe-Einstellungen">
      <Row>
        <Col><Inp label="Zusätzlicher Ausgabeordner (optional)" placeholder="" value={cfg.outputDir} onChange={v=>set({...cfg,outputDir:v})} help="Kopiert Report, Metriken und Predictions zusätzlich in diesen Ordner (neben MLflow). Leer = nur MLflow."/></Col>
        <Col><Inp label="Max. Prediction-Rows (0=alle)" type="number" value={cfg.maxPredRows||0} onChange={v=>set({...cfg,maxPredRows:Number(v)})} help="Begrenzt die Anzahl gespeicherter Predictions – nützlich bei sehr großen Datensätzen"/></Col>
      </Row>
    </Expander>
  </Sec>
</>); };

// ── Tuning Plan (mirrors rubin/tuning/ logic) ──