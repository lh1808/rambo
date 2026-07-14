const PData = ({cfg,set,setCfg,activeBase,setActiveBase,activeAddons,setActiveAddons,setSp,setSpFmt,view,sysInfo}) => {
  const [tab,setTab] = useState("Dateipfade");
  const [tt,setTt] = useState(cfg.treatmentType||"binary");


  const switchTreatment = (newTt) => {
    setTt(newTt);
    setActiveBase(null);
    const paths = {x_file:cfg.x_file,t_file:cfg.t_file,y_file:cfg.y_file,s_file:cfg.s_file,eval_x_file:cfg.eval_x_file,eval_t_file:cfg.eval_t_file,eval_y_file:cfg.eval_y_file,eval_s_file:cfg.eval_s_file,hasNaN:cfg.hasNaN,nanCols:cfg.nanCols,seed:cfg.seed,outputDir:cfg.outputDir,histScoreName:cfg.histScoreName,histScoreCol:cfg.histScoreCol,histScoreHigher:cfg.histScoreHigher,expName:cfg.expName,dpRunName:cfg.dpRunName,eval_mask_file:cfg.eval_mask_file};
    const addonOverlay = {};
    activeAddons.forEach(k => {const a=ADDON_PRESETS.find(x=>x.key===k);if(a){Object.assign(addonOverlay,a.cfg);if(a.waves)addonOverlay[a.waves.field]=_wavesToTrials(a.waves.w, a.waves.stage||"blt");}});
    // Filter models: remove BT-only models when switching to multi
    const defaultModels = newTt==="multi" ? DEFAULT_CFG.models.filter(m=>!btOnly.has(m)) : DEFAULT_CFG.models;
    // Qini-Scorer (FMT/CFT) ist binär-only — bei Multi auf "auto" zurücksetzen
    // (Backend löst auto bei Multi-Treatment zu rscore auf).
    const scorerReset = newTt==="multi" ? {fmtScorer:"auto",cfScorer:"auto"} : {};
    setCfg({...DEFAULT_CFG,...paths,...addonOverlay,...scorerReset,models:defaultModels,treatmentType:newTt,refGroup:0,selMetric:newTt==="multi"?"policy_value":"qini"});
  };


  const _getParallel = (stage="blt") => {
    const nCores = sysInfo?.cpu?.cores || 0;
    const pl = cfg.parallelLevel||3;
    if(!nCores || pl <= 2) return pl <= 2 ? 1 : 5;
    if(stage === "fmt") return Math.max(2, Math.floor(nCores / 8));
    return Math.max(1, Math.floor(nCores / 4));
  };
  const _wavesToTrials = (w, stage="blt") => Math.max(10, _getParallel(stage) * w);

  const MUTEX = [["bl_tuning_schnell","bl_tuning","bl_tuning_intensiv"],["fmt_schnell","fmt","fmt_intensiv"],["grf_tuning_schnell","grf_tuning","grf_tuning_intensiv"],["reg_moderate","reg_strong"]];
  const toggleAddon = (p) => {
    const next = new Set(activeAddons);
    if(next.has(p.key)) {
      next.delete(p.key);
      const reset = {};
      Object.keys(p.cfg).forEach(k => {reset[k]=DEFAULT_CFG[k];});
      if(p.waves) reset[p.waves.field] = DEFAULT_CFG[p.waves.field];
      setCfg(prev => ({...prev,...reset}));
    } else {
      next.add(p.key);
      MUTEX.forEach(group => {
        if(group.includes(p.key)) {
          group.filter(k=>k!==p.key).forEach(k=>{
            if(next.has(k)){
              next.delete(k);
              const other=ADDON_PRESETS.find(x=>x.key===k);
              if(other){const r={};Object.keys(other.cfg).forEach(ck=>{r[ck]=DEFAULT_CFG[ck]});if(other.waves)r[other.waves.field]=DEFAULT_CFG[other.waves.field];setCfg(prev=>({...prev,...r}));}
            }
          });
        }
      });
      let applyCfg = {...p.cfg};
      // Wave-based: compute trials from waves × parallel
      if(p.waves) applyCfg[p.waves.field] = _wavesToTrials(p.waves.w, p.waves.stage||"blt");
      if(p.key === "feature_reduction" && cfg.hasNaN) {
        applyCfg.fsMethods = (applyCfg.fsMethods || []).filter(m => m !== "causal_forest");
      }
      setCfg(prev => ({...prev,...applyCfg}));
    }
    setActiveAddons(next);
  };

  const [importMode,setImportMode] = useState(false);
  const [importStatus,setImportStatus] = useState(null);
  const fileInputRef = useRef(null);

  const handleYamlImport = (text) => {
    try {
      const parsed = parseYamlToCfg(text);
      // Extract sp/spFmt BEVOR sie in setCfg landen (sind keine cfg-Keys)
      const _importedSp = parsed.__sp;
      const _importedSpFmt = parsed.__spFmt;
      delete parsed.__sp;
      delete parsed.__spFmt;
      setCfg(parsed);
      setTt(parsed.treatmentType || "binary");
      setActiveBase(null);
      setActiveAddons(new Set());
      if (_importedSp) setSp(_importedSp);
      if (_importedSpFmt) setSpFmt(_importedSpFmt);
      const _importedMsgs = [];
      if (_importedSp) _importedMsgs.push("Suchraum BL");
      if (_importedSpFmt) _importedMsgs.push("Suchraum FM-Tuning");
      const _extraMsg = _importedMsgs.length > 0 ? ` (inkl. ${_importedMsgs.join(", ")})` : "";
      setImportStatus({ok:true, msg:"Konfiguration erfolgreich geladen – " + Object.keys(parsed).length + " Felder übernommen" + _extraMsg + "."});
      // Importierte Config auch ans Backend senden
      fetch("./api/import-config", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({yaml_text: text})
      }).catch(()=>{});
    } catch(e) {
      setImportStatus({ok:false, msg:"Fehler beim Parsen: " + e.message});
    }
  };

  return (
    <>
      {view==="files"&&<Sec title="Evaluationsmodus" accent="#D4A853">
        <Info>{cfg.dpRunName
          ? "Evaluationsmodus wurde durch DataPrep festgelegt und kann hier nicht geändert werden."
          : "Bestimmt, wie die Modelle nach dem Training evaluiert werden. Bei vorheriger DataPrep wird der Modus automatisch übernommen."
        }</Info>
        {(()=>{
          const locked = !!cfg.dpRunName;
          const currentMode = cfg.eval_mask_file && cfg.validateOn!=="external" ? "tmes" : cfg.validateOn==="external" ? "external" : "cross";
          const modes = [
            {k:"cross",label:"Cross-Validation",desc:"K-Fold auf den Trainingsdaten. Alle Daten werden effizient genutzt. Standard für explorative Analysen."},
            {k:"tmes",label:"Train Many, Evaluate Some",desc:"Training und Cross-Prediction auf allen Daten. Evaluation (Metriken, Plots) nur auf den ausgewählten Dateien."},
            {k:"external",label:"External Eval",desc:"Separater Datensatz für Evaluation. Preprocessor wird nur auf Trainingsdaten gefittet — kein Leakage."},
          ];
          return (<>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:10,marginTop:4}}>
              {modes.map(o => {
                const active = currentMode === o.k;
                const dis = locked || (o.k==="tmes" && !cfg.eval_mask_file);
                return (
                  <label key={o.k} onClick={e=>{
                    e.preventDefault();
                    if(dis) return;
                    if(o.k==="cross") set({...cfg, validateOn:"cross", eval_mask_file:""});
                    else if(o.k==="external") set({...cfg, validateOn:"external"});
                  }} style={{display:"flex",flexDirection:"column",padding:"14px 16px",borderRadius:10,border:active?"1.5px solid #D4A853":"1.5px solid "+C.border,background:active?"#fffbeb":"#fff",cursor:dis?"not-allowed":"pointer",opacity:(!active&&dis)?0.4:1,transition:"all 0.15s"}}>
                    <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
                      <input type="radio" name="eval_mode_files" checked={active} readOnly style={{accentColor:"#D4A853",pointerEvents:"none"}}/>
                      <span style={{fontSize:13,fontWeight:600,color:active?"#7a5a00":C.dark}}>{o.label}</span>
                      {active && locked && <span style={{fontSize:9,fontWeight:600,color:"#7a5a00",background:"rgba(212,168,83,0.15)",padding:"1px 6px",borderRadius:6}}>DataPrep</span>}
                    </div>
                    <div style={{fontSize:11,color:C.textMuted,lineHeight:1.4}}>{o.desc}</div>
                  </label>
                );
              })}
            </div>
            {currentMode==="tmes" && <div style={{fontSize:11,color:"#7a5a00",background:"#fffbeb",padding:"6px 12px",borderRadius:8,marginTop:8,lineHeight:1.4,border:"1px solid #fbbf24"}}><strong>TMES aktiv</strong> (aus DataPrep): Eval-Maske <span style={{fontFamily:MONO,fontSize:10}}>{(Array.isArray(cfg.eval_mask_file)?cfg.eval_mask_file[0]:cfg.eval_mask_file||"").split("/").pop()}</span></div>}
            {!cfg.eval_mask_file && !cfg.dpRunName && <div style={{fontSize:10.5,color:"#6b7280",marginTop:6}}>TMES erfordert DataPrep (Eval-Maske wird dort erzeugt).</div>}
          </>);
        })()}
      </Sec>}

      {view==="files"&&<>
        <Tabs tabs={["Dateipfade","Datei-Upload"]} active={tab} onSelect={setTab}/>

        <Sec title="Trainingsdaten">
          <Info>Die aufbereiteten Dateien für die Analyse-Pipeline. X (Features), T (Treatment), Y (Outcome) sind Pflicht. S (historischer Score) ist optional für Benchmark-Vergleiche.</Info>
          {tab==="Dateipfade" ? (
            <Row><Col>
              <Inp label="X-Datei (Features)" placeholder="runs/data/X.parquet" value={cfg.x_file} onChange={v=>set({...cfg,x_file:v})}/>
              <Inp label="T-Datei (Treatment)" placeholder="runs/data/T.parquet" value={cfg.t_file} onChange={v=>set({...cfg,t_file:v})}/>
            </Col><Col>
              <Inp label="Y-Datei (Outcome)" placeholder="runs/data/Y.parquet" value={cfg.y_file} onChange={v=>set({...cfg,y_file:v})}/>
              <Inp label="S-Datei (Hist. Score, optional)" placeholder="runs/data/S.parquet" value={cfg.s_file} onChange={v=>set({...cfg,s_file:v})}/>
            </Col></Row>
          ) : (
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
              {[{label:"X-Datei (Features)",key:"x_file"},{label:"T-Datei (Treatment)",key:"t_file"},{label:"Y-Datei (Outcome)",key:"y_file"},{label:"S-Datei (optional)",key:"s_file"}].map(f => {
                const uploadFile = async (file) => {
                  if(!file) return;
                  const fd = new FormData();
                  fd.append("file", file);
                  try {
                    const res = await fetch("./api/upload", {method:"POST", body: fd});
                    const data = await res.json();
                    if(data.status === "done" && data.path) {
                      set({...cfg, [f.key]: data.path});
                    } else {
                      alert("Upload fehlgeschlagen: " + (data.message || "Unbekannter Fehler"));
                    }
                  } catch(e) {
                    alert("Upload fehlgeschlagen: " + (e.message || "Server nicht erreichbar. Prüfe die Verbindung in der Sidebar."));
                  }
                };
                return (
                <div key={f.key}
                  onClick={()=>{const inp=document.createElement("input");inp.type="file";inp.accept=".csv,.parquet,.sas7bdat";inp.onchange=e=>uploadFile(e.target.files?.[0]);inp.click()}}
                  style={{border:"1.5px solid "+(cfg[f.key]?"#86efac":"#d4b5b8"),borderRadius:10,padding:"24px 16px",textAlign:"center",background:cfg[f.key]?"#f0fdf4":"#faf8f8",cursor:"pointer",transition:"all 0.15s"}} onMouseEnter={e=>{e.currentTarget.style.borderColor="#9B111E";e.currentTarget.style.background="#FDF2F3"}} onMouseLeave={e=>{e.currentTarget.style.borderColor=cfg[f.key]?"#86efac":"#d4b5b8";e.currentTarget.style.background=cfg[f.key]?"#f0fdf4":"#faf8f8"}}>
                  <div style={{fontWeight:600,color:cfg[f.key]?"#059669":"#6B0D15",fontSize:13}}>{cfg[f.key] ? "✓ " + cfg[f.key].split("/").pop() : f.label}</div>
                  <div style={{fontSize:11,color:"#999",marginTop:3}}>{cfg[f.key] ? "Klicken zum Ersetzen" : "Klicken zum Hochladen"}</div>
                </div>
              );})}
            </div>
          )}
        </Sec>

        {cfg.validateOn==="external" && <Sec title="Externe Evaluation" accent="#6366f1">
          <Info>Separater Holdout-Datensatz. Der Preprocessor wird leakage-frei nur auf den Trainingsdaten gefittet.</Info>
          {(cfg.eval_x_file && cfg.eval_t_file && cfg.eval_y_file)
            ? <div style={{fontSize:11,color:"#059669",background:"#f0fdf4",padding:"6px 12px",borderRadius:8,marginBottom:10,border:"1px solid #a7f3d0"}}><strong>Bereit</strong> — alle Pflichtdateien gesetzt.</div>
            : <div style={{fontSize:11,color:"#92400e",background:"#fef3c7",padding:"6px 12px",borderRadius:8,marginBottom:10,border:"1px solid #fbbf24"}}><strong>Dateien fehlen</strong> — mindestens X, T und Y benötigt.</div>
          }
          {tab==="Dateipfade" ? (
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>
              <Inp label="X (Features)" placeholder="runs/data/X_eval.parquet" value={cfg.eval_x_file||""} onChange={v=>set({...cfg,eval_x_file:v})}/>
              <Inp label="Y (Outcome)" placeholder="runs/data/Y_eval.parquet" value={cfg.eval_y_file||""} onChange={v=>set({...cfg,eval_y_file:v})}/>
              <Inp label="T (Treatment)" placeholder="runs/data/T_eval.parquet" value={cfg.eval_t_file||""} onChange={v=>set({...cfg,eval_t_file:v})}/>
              <Inp label="S (Score, optional)" placeholder="runs/data/S_eval.parquet" value={cfg.eval_s_file||""} onChange={v=>set({...cfg,eval_s_file:v})}/>
            </div>
          ) : (
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
              {[{label:"X (Features)",key:"eval_x_file"},{label:"Y (Outcome)",key:"eval_y_file"},{label:"T (Treatment)",key:"eval_t_file"},{label:"S (Score, optional)",key:"eval_s_file"}].map(f => {
                const uploadEval = async (file) => {
                  if(!file) return;
                  const fd = new FormData(); fd.append("file",file);
                  try{const r=await fetch("./api/upload",{method:"POST",body:fd});const d=await r.json();
                    if(d.status==="done"&&d.path) set({...cfg,[f.key]:d.path});
                    else alert("Upload fehlgeschlagen: "+(d.message||""));
                  }catch(e){alert("Upload fehlgeschlagen: "+(e.message||""));}
                };
                return (
                  <div key={f.key}
                    onClick={()=>{const inp=document.createElement("input");inp.type="file";inp.accept=".csv,.parquet,.sas7bdat";inp.onchange=e=>uploadEval(e.target.files?.[0]);inp.click()}}
                    style={{border:"1.5px solid "+(cfg[f.key]?"#86efac":"#c4b5fd"),borderRadius:10,padding:"20px 16px",textAlign:"center",background:cfg[f.key]?"#f0fdf4":"#f5f3ff",cursor:"pointer",transition:"all 0.15s"}}
                    onMouseEnter={e=>{e.currentTarget.style.borderColor="#6366f1";e.currentTarget.style.background="#ede9fe"}}
                    onMouseLeave={e=>{e.currentTarget.style.borderColor=cfg[f.key]?"#86efac":"#c4b5fd";e.currentTarget.style.background=cfg[f.key]?"#f0fdf4":"#f5f3ff"}}>
                    <div style={{fontWeight:600,color:cfg[f.key]?"#059669":"#6366f1",fontSize:12.5}}>{cfg[f.key] ? "✓ "+cfg[f.key].split("/").pop() : f.label}</div>
                    <div style={{fontSize:10.5,color:"#999",marginTop:3}}>{cfg[f.key] ? "Klicken zum Ersetzen" : "Klicken zum Hochladen"}</div>
                  </div>
                );
              })}
            </div>
          )}
        </Sec>}

        {cfg.s_file && <Sec title="Benchmark (optional)">
          <Info>Modelle werden gegen den historischen Score aus <code style={{background:"#f5f0f0",padding:"1px 5px",borderRadius:3,fontSize:11}}>{cfg.s_file.split("/").pop()}</code> verglichen. Läuft automatisch bei gesetzter S-Datei.</Info>
          <Row>
            <Col><Inp label="Score-Name" value={cfg.histScoreName||"historical_score"} onChange={v=>set({...cfg,histScoreName:v})} help="Bezeichnung im Report"/></Col>
            <Col><Inp label="Score-Spalte" value={cfg.histScoreCol||"S"} onChange={v=>set({...cfg,histScoreCol:v})} help="Spaltenname in der S-Datei"/></Col>
            <Col><Toggle label="Höher = besser" checked={cfg.histScoreHigher!==false} onChange={v=>set({...cfg,histScoreHigher:v})} help="Score-Richtung: höher = besser"/></Col>
          </Row>
        </Sec>}
      </>}

      {view==="template"&&(cfg.dpRunName ? (
        <div style={{display:"flex",alignItems:"center",gap:12,padding:"12px 18px",background:"#fdf8ee",border:"1px solid #e8d49c",borderRadius:10,marginBottom:22,fontSize:13}}>
          <span style={{fontSize:16,color:"#D4A853"}}>✓</span>
          <div style={{flex:1}}>
            <span style={{fontWeight:600,color:"#7a5a00"}}>Experiment „{cfg.expName}"</span>
            <span style={{color:"#57606a"}}> — DataPrep-Run: „{cfg.dpRunName}"</span>
          </div>
          <input type="text" value={cfg.expName||""} onChange={e=>set({...cfg,expName:e.target.value})} style={{width:180,height:32,padding:"0 10px",border:"1.5px solid #e8d49c",borderRadius:6,fontSize:13,fontWeight:600,background:"#fff",outline:"none",boxSizing:"border-box"}}/>
        </div>
      ) : (
        <div style={{background:"linear-gradient(135deg,#8a6d1b,#D4A853)",borderRadius:12,padding:"18px 22px",marginBottom:22,color:"#fff"}}>
        <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:10}}>
          <div style={{fontSize:11,textTransform:"uppercase",letterSpacing:1,opacity:.7}}>Experiment-Name (MLflow)</div>
          {(!cfg.expName || cfg.expName==="rubin") && <span style={{fontSize:10,background:"rgba(255,255,255,.2)",color:"#fff",padding:"2px 10px",borderRadius:10,fontWeight:600,border:"1px solid rgba(255,255,255,.3)"}}>Bitte benennen</span>}
        </div>
        <input type="text" value={cfg.expName||""} onChange={e=>set({...cfg,expName:e.target.value})} placeholder="z. B. churn_q2_2026" style={{width:"100%",height:42,padding:"0 14px",border:(!cfg.expName||cfg.expName==="rubin")?"2px solid rgba(255,255,255,.5)":"2px solid rgba(255,255,255,.25)",borderRadius:8,fontSize:17,fontWeight:700,background:"rgba(255,255,255,.15)",color:"#fff",outline:"none",boxSizing:"border-box",letterSpacing:.3}} onFocus={e=>{e.target.style.borderColor="rgba(255,255,255,.6)"}} onBlur={e=>{e.target.style.borderColor=(!cfg.expName||cfg.expName==="rubin")?"rgba(255,255,255,.5)":"rgba(255,255,255,.25)"}}/>
        <div style={{fontSize:11.5,opacity:.55,marginTop:6}}>Gemeinsamer Name für alle MLflow-Runs. Steht für das Produkt oder Projekt.</div>
      </div>
      ))}
      {view==="template"&&<Sec title="Config importieren" accent="#D4A853">
        <div style={{display:"flex",gap:10,alignItems:"center",flexWrap:"wrap"}}>
          <Btn small onClick={()=>fileInputRef.current?.click()}>Datei laden</Btn>
          <Btn small secondary onClick={()=>setImportMode(!importMode)}>{importMode?"Einfügen ausblenden":"YAML einfügen"}</Btn>
          <input ref={fileInputRef} type="file" accept=".yml,.yaml" style={{display:"none"}} onChange={e=>{const f=e.target.files?.[0];if(f){const r=new FileReader();r.onload=ev=>handleYamlImport(ev.target.result);r.onerror=()=>setImportStatus({ok:false,msg:"Datei konnte nicht gelesen werden."});r.readAsText(f)}}}/>
          <span style={{fontSize:11,color:C.textMuted}}>Bestehende config.yml laden oder einfügen</span>
        </div>
        {importMode && (
          <div style={{marginTop:10}}>
            <textarea
              placeholder="mlflow:\n  experiment_name: rubin\n..."
              rows={6}
              style={{width:"100%",fontFamily:MONO,fontSize:11.5,padding:"10px 14px",border:`1.5px solid ${C.border}`,borderRadius:8,resize:"vertical",background:"#1e1e2e",color:"#cdd6f4",outline:"none",boxSizing:"border-box",lineHeight:1.6}}
              onFocus={e=>{e.target.style.borderColor=C.gold}}
              onBlur={e=>{e.target.style.borderColor=C.border}}
              id="yaml-paste"
            />
            <div style={{marginTop:6}}>
              <Btn small onClick={()=>{const el=document.getElementById("yaml-paste");if(el?.value)handleYamlImport(el.value)}}>Config übernehmen</Btn>
            </div>
          </div>
        )}
        {importStatus && <div style={{marginTop:8}}><Info type={importStatus.ok?"success":"error"}>{importStatus.msg}</Info></div>}
      </Sec>}

      {view==="template"&&<Sec title="Treatment-Typ">
        <Info>Grundlegende Entscheidung: Binary Treatment (T in {"{0,1}"}) oder Multi-Treatment (T in {"{0,1,...,K-1}"}). Bestimmt, welche Modelle und Metriken verfügbar sind.</Info>
        <div style={{display:"flex",gap:10,marginTop:10}}>
          {[{k:"binary",l:"Binary Treatment",d:"T in {0,1} – 8 Modelle + Ensemble verfügbar"},{k:"multi",l:"Multi-Treatment",d:"T in {0,...,K-1} – 3 Modelle: ParamDML, DRLearner, CausalForestDML"}].map(o => {
            const active = tt === o.k;
            return (
              <button key={o.k} onClick={()=>switchTreatment(o.k)} style={{flex:1,padding:"20px 22px",borderRadius:12,border:active?"2px solid #9B111E":"1.5px solid "+C.border,background:active?"#FDF2F3":"#faf6f6",cursor:"pointer",textAlign:"left",transition:"all 0.2s"}}
                onMouseEnter={e=>{if(!active)e.currentTarget.style.borderColor="#c4343f"}}
                onMouseLeave={e=>{if(!active)e.currentTarget.style.borderColor=C.border}}>
                <div style={{display:"flex",alignItems:"center",gap:10}}>
                  <div style={{width:20,height:20,borderRadius:10,border:active?"5px solid #9B111E":"2px solid #ccc",background:active?"#fff":"#fff",flexShrink:0}}/>
                  <div>
                    <div style={{fontSize:15,fontWeight:700,color:active?"#6B0D15":"#333"}}>{o.l}</div>
                    <div style={{fontSize:12,color:"#888",marginTop:2}}>{o.d}</div>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
        {tt==="multi" && <div style={{marginTop:8}}><Row><Col><Inp label="Reference Group" type="number" value={cfg.refGroup} onChange={v=>set({...cfg,refGroup:v})} help="Control-Baseline (typisch 0)"/></Col></Row></div>}
      </Sec>}

      {view==="template"&&<Sec title="Studientyp">
        <Info>Grundlegende Entscheidung: randomisiertes Experiment (RCT) oder Beobachtungsdaten? Bestimmt, welche Modelle gegen Confounding abgesichert sind.</Info>
        <div style={{display:"flex",gap:10,marginTop:10}}>
          {[{k:"rct",l:"RCT (randomisiert)",d:"Treatment zufällig zugewiesen – alle Modelle valide"},{k:"observational",l:"Beobachtungsdaten",d:"Confounding möglich – nur DR/DML-Modelle empfohlen"}].map(o => {
            const active = (cfg.studyType||"rct") === o.k;
            return (
              <button key={o.k} onClick={()=>{const upd={...cfg,studyType:o.k};if(o.k==="rct"){upd.models=["NonParamDML","DRLearner","SLearner","TLearner","XLearner","ParamDML","CausalForestDML","CausalForest"]}else{upd.models=["NonParamDML","ParamDML","DRLearner","CausalForestDML"]}set(upd)}} style={{flex:1,padding:"20px 22px",borderRadius:12,border:active?"2px solid #9B111E":"1.5px solid "+C.border,background:active?"#FDF2F3":"#faf6f6",cursor:"pointer",textAlign:"left",transition:"all 0.2s"}}
                onMouseEnter={e=>{if(!active)e.currentTarget.style.borderColor="#c4343f"}}
                onMouseLeave={e=>{if(!active)e.currentTarget.style.borderColor=C.border}}>
                <div style={{display:"flex",alignItems:"center",gap:10}}>
                  <div style={{width:20,height:20,borderRadius:10,border:active?"5px solid #9B111E":"2px solid #ccc",background:"#fff",flexShrink:0}}/>
                  <div>
                    <div style={{fontSize:15,fontWeight:700,color:active?"#6B0D15":"#333"}}>{o.l}</div>
                    <div style={{fontSize:12,color:"#888",marginTop:2}}>{o.d}</div>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
        {(cfg.studyType||"rct") !== "rct" && (
          <div style={{marginTop:10}}><Info type="warn"><strong>Confounding-Korrektur essentiell:</strong> Es werden nur Modelle mit Orthogonalisierung (DML-Familie) oder Doubly-Robust-Eigenschaft (DRLearner) empfohlen: <strong>NonParamDML, ParamDML, DRLearner, CausalForestDML</strong>. SLearner, TLearner und CausalForest sind als Baseline auswählbar, aber nicht gegen Confounding geschützt.</Info></div>
        )}
      </Sec>}

      {view==="template"&&<PModelsOnly cfg={cfg} set={set}/>}

      {view==="template"&&<Sec title="Pipeline-Optionen">
        <Info>Optionale Funktionen, die die Analyse erweitern. Mehrfachauswahl möglich — jede Option ist unabhängig aktivierbar.</Info>
        {(() => {
          const R1_tuning = () => {
            const c="#9B111E",bg="#fef9f9",bd="#e8b4b8",abg="#fce8ea";
            const subgroups = [
              {group:"tuning_blt",label:"Base-Learner-Tuning"},
              {group:"tuning_fmt",label:"Final-Model-Tuning"},
              {group:"tuning_cft",label:"CausalForest-Tuning"},
            ];
            const allItems = subgroups.flatMap(sg=>ADDON_PRESETS.filter(p=>p.group===sg.group));
            const someOn = allItems.some(p=>activeAddons.has(p.key));
            const stdKeys = ["bl_tuning","fmt","grf_tuning"];
            const allStd = stdKeys.every(k=>activeAddons.has(k));
            const toggleAllStd = () => {
              const next = new Set(activeAddons);
              if(allStd){
                allItems.forEach(p=>{next.delete(p.key)});
                const reset = {};
                allItems.forEach(p=>{Object.keys(p.cfg).forEach(k=>{reset[k]=DEFAULT_CFG[k]});if(p.waves)reset[p.waves.field]=DEFAULT_CFG[p.waves.field]});
                setCfg(prev=>({...prev,...reset}));
              } else {
                // Deselect any non-standard variants first
                allItems.forEach(p=>{if(!stdKeys.includes(p.key))next.delete(p.key)});
                // Apply all 3 standard presets at once
                const merged = {};
                stdKeys.forEach(k=>{
                  const p=ADDON_PRESETS.find(x=>x.key===k);
                  if(p){next.add(k);Object.assign(merged,p.cfg);if(p.waves)merged[p.waves.field]=_wavesToTrials(p.waves.w, p.waves.stage||"blt")}
                });
                setCfg(prev=>({...prev,...merged}));
              }
              setActiveAddons(next);
            };
            return (
              <div style={{background:bg,borderRadius:12,padding:"14px 16px",border:`1.5px solid ${someOn?c:bd}`}}>
                <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12}}>
                  <div style={{fontSize:11,fontWeight:700,textTransform:"uppercase",letterSpacing:".5px",color:c}}>Tuning</div>
                  <button onClick={toggleAllStd} style={{fontSize:10,color:c,background:"transparent",border:`1px solid ${c}`,borderRadius:6,padding:"2px 8px",cursor:"pointer",fontWeight:600,opacity:0.8}}>{allStd?"Alle ab":"Alle an"}</button>
                </div>
                {subgroups.map(sg => {
                  const items = ADDON_PRESETS.filter(p=>p.group===sg.group);
                  return (
                    <div key={sg.group} style={{marginBottom:10}}>
                      <div style={{fontSize:10.5,fontWeight:600,color:"#666",marginBottom:5,letterSpacing:0.3}}>{sg.label}</div>
                      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8}}>
                        {items.map(p => {
                          const on = activeAddons.has(p.key);
                          return (
                            <button key={p.key} onClick={()=>toggleAddon(p)}
                              style={{display:"flex",alignItems:"flex-start",gap:10,padding:"10px 14px",background:on?abg:"#fff",border:on?`2px solid ${c}`:`1.5px solid ${bd}`,borderRadius:8,cursor:"pointer",textAlign:"left",transition:"all 0.15s"}}>
                              <div style={{width:16,height:16,borderRadius:4,border:on?`2px solid ${c}`:"2px solid #ccc",background:on?c:"#fff",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,marginTop:2}}>
                                {on && <span style={{color:"#fff",fontSize:10,fontWeight:700}}>✓</span>}
                              </div>
                              <div>
                                <div style={{fontSize:12.5,fontWeight:600,color:on?c:"#333"}}>{p.label}</div>
                                <div style={{fontSize:10.5,color:"#888",marginTop:1,lineHeight:1.4}}>{p.desc}</div>
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          };
          const R1 = [];
          const R2 = [
            {group:"regularization",label:"Tuning-Regularisierung",c:"#7c3aed",bg:"#f5f3ff",bd:"#c4b5fd",abg:"#ddd6fe"},
          ];
          const R3 = [
            {group:"feature",label:"Feature-Selektion",c:"#0d9488",bg:"#f0fdfa",bd:"#99f6e4",abg:"#ccfbf1"},
            {group:"stability",label:"Stabilität",c:"#2563eb",bg:"#eff6ff",bd:"#93c5fd",abg:"#dbeafe"},
          ];
          const R4 = [
            {group:"performance",label:"Speed",c:"#57606a",bg:"#fbfcfd",bd:"#d0d7de",abg:"#eaeef2"},
            {group:"exploration",label:"Exploration",c:"#e67e22",bg:"#fff7ed",bd:"#fed7aa",abg:"#ffedd5"},
          ];
          const R5 = [
            {group:"interpret",label:"Erklärbarkeit",c:"#D4A853",bg:"#fffdf5",bd:"#e8d49c",abg:"#fff8dc"},
            {group:"production",label:"Production",c:"#059669",bg:"#f3faf5",bd:"#a3d9b1",abg:"#d4edda"},
          ];
          const renderG = ({group,label,c,bg,bd,abg}) => {
            const items = ADDON_PRESETS.filter(p=>p.group===group);
            if(!items.length) return null;
            const allOn = items.every(p=>activeAddons.has(p.key));
            const someOn = items.some(p=>activeAddons.has(p.key));
            const toggleAll = () => {
              if(allOn){
                const next = new Set(activeAddons);
                items.forEach(p=>{next.delete(p.key);const r={};Object.keys(p.cfg).forEach(k=>{r[k]=DEFAULT_CFG[k]});setCfg(prev=>({...prev,...r}))});
                setActiveAddons(next);
              } else {
                // When selecting all: for mutex pairs, prefer the stronger variant (last in pair)
                const skip = new Set();
                MUTEX.forEach(pair=>{const inGroup=pair.filter(k=>items.some(p=>p.key===k));if(inGroup.length>1)skip.add(inGroup[0])});
                const next = new Set(activeAddons);
                items.forEach(p=>{
                  if(skip.has(p.key)){next.delete(p.key);return}
                  next.add(p.key);setCfg(prev=>({...prev,...p.cfg}));
                });
                setActiveAddons(next);
              }
            };
            return (
              <div key={group} style={{background:bg,borderRadius:12,padding:"14px 16px",border:`1.5px solid ${someOn?c:bd}`}}>
                <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:10}}>
                  <div style={{fontSize:11,fontWeight:700,textTransform:"uppercase",letterSpacing:".5px",color:c}}>{label}</div>
                  {items.length>1 && <button onClick={toggleAll} style={{fontSize:10,color:c,background:"transparent",border:`1px solid ${c}`,borderRadius:6,padding:"2px 8px",cursor:"pointer",fontWeight:600,opacity:0.8}}>{allOn?"Alle ab":"Alle an"}</button>}
                </div>
                <div style={{display:"grid",gridTemplateColumns:items.length===1?"1fr":"1fr 1fr",gap:8}}>
                  {items.map(p => {
                    const on = activeAddons.has(p.key);
                    return (
                      <button key={p.key} onClick={()=>toggleAddon(p)}
                        style={{display:"flex",alignItems:"flex-start",gap:10,padding:"10px 14px",background:on?abg:"#fff",border:on?`2px solid ${c}`:`1.5px solid ${bd}`,borderRadius:8,cursor:"pointer",textAlign:"left",transition:"all 0.15s"}}>
                        <div style={{width:16,height:16,borderRadius:4,border:on?`2px solid ${c}`:"2px solid #ccc",background:on?c:"#fff",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,marginTop:2}}>
                          {on && <span style={{color:"#fff",fontSize:10,fontWeight:700}}>✓</span>}
                        </div>
                        <div>
                          <div style={{fontSize:12.5,fontWeight:600,color:on?c:"#333"}}>{p.label}</div>
                          <div style={{fontSize:10.5,color:"#888",marginTop:1,lineHeight:1.4}}>{p.desc}</div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          };
          return (<>
            <div style={{marginTop:12}}>
              {R1_tuning()}
            </div>
            <div style={{marginTop:12}}>
              {R2.map(renderG)}
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,marginTop:12}}>
              {R3.map(renderG)}
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,marginTop:12}}>
              {R4.map(renderG)}
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,marginTop:12}}>
              {R5.map(renderG)}
            </div>
          </>);
        })()}
      </Sec>}

      {view==="files"&&<Sec title="Datenqualität">
        {cfg.hasNaN && (cfg.nanCols||[]).length > 0 && (
          <Info type="warn"><strong>Fehlende Werte in {(cfg.nanCols||[]).length} Spalten.</strong> CausalForestDML, CausalForest und die Feature-Selektionsmethode CausalForest werden automatisch übersprungen. Alle anderen Modelle sind davon nicht betroffen. In der Datenvorbereitung kann eine NaN-Auffüll-Methode gewählt werden.</Info>
        )}
        {cfg.hasNaN && (cfg.nanCols||[]).length === 0 && (
          <Info type="warn"><strong>Fehlende Werte in den Daten.</strong> CausalForestDML, CausalForest und die Feature-Selektionsmethode CausalForest werden automatisch übersprungen. Alle anderen Modelle sind davon nicht betroffen.</Info>
        )}
        {!cfg.hasNaN && (
          <Info type="success">Keine fehlenden Werte – alle Modelle sind verfügbar.</Info>
        )}
        <div style={{fontSize:11.5,color:"#888",marginTop:6}}>Der NaN-Status wird automatisch erkannt, wenn Spalten in der Datenvorbereitung geladen werden. Ohne Datenvorbereitung kannst du den Status manuell setzen:</div>
        <div style={{marginTop:6}}>
          <Toggle label="Fehlende Werte manuell markieren" checked={cfg.hasNaN||false} onChange={v=>set({...cfg,hasNaN:v})} help="Nur nötig, falls die Datenvorbereitung nicht genutzt wurde"/>
        </div>
      </Sec>}
    </>
  );
};

// Remaining pages (Config, Models, Selection, Explain) - compact