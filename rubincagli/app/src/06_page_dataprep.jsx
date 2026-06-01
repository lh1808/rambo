const PDataPrep = ({dp,setDp,cfg,setCfg,setPg}) => {
  const [simCols,setSimCols] = useState(null);
  const [dpRunning,setDpRunning] = useState(false);
  const [dpDone,setDpDone] = useState(false);
  const [dpProgress,setDpProgress] = useState(0);
  const [dpError,setDpError] = useState(null);
  const [detecting, setDetecting] = useState(false);
  const [detectError, setDetectError] = useState(null);
  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState(null);
  const [dpTab, setDpTab] = useState("Dateipfade");

  const files = dp.files || [""];
  const setFiles = (fn) => setDp(prev => ({...prev, files: typeof fn==="function" ? fn(prev.files||[""]) : fn, targetValues: [], treatValues: [], detectedCols: null, nanCols: [], colStats: {}}));
  const addFile = () => setFiles(prev => [...prev, ""]);
  const removeFile = (i) => setFiles(prev => prev.filter((_,j)=>j!==i));
  const updateFile = (i,v) => setFiles(prev => prev.map((f,j)=>j===i?v:f));

  // Column detection via Backend-API (/api/detect-columns)
  const detectColumns = async () => {
    setDetecting(true);
    setDetectError(null);
    const filePath = (files||[""])[0];
    if (!filePath || !filePath.trim()) {
      setDetectError("Bitte zuerst einen Dateipfad angeben.");
      setDetecting(false);
      return;
    }
    try {
      const res = await fetch("./api/detect-columns", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({path: filePath, target_column: (dp.targets||[""])[0], treatment_column: dp.treatment, delimiter: dp.delimiter || ","})
      });
      const data = await res.json();
      if (data.status === "done") {
        setSimCols(data.columns);
        const sel = {};
        data.columns.forEach(c => { sel[c] = true; });
        const tgtU = ((dp.targets||[""])[0]||"Y").toUpperCase();
        const trtU = (dp.treatment||"").toUpperCase();
        const scU = (dp.scoreName||"").toUpperCase();
        data.columns.forEach(c => {
          if(c.toUpperCase()===tgtU || c.toUpperCase()===trtU || (scU && c.toUpperCase()===scU)) sel[c]=false;
        });
        const defMap = {};
        (data.treat_values||[]).forEach((v,i) => { defMap[v] = i===0 ? 0 : i; });
        const nanDetected = (data.nan_cols||[]).length > 0;
        setCfg(prev => ({...prev, hasNaN: nanDetected, nanCols: data.nan_cols||[]}));
        setDp(prev => ({...prev, nRows: data.n_rows||0, detectedCols: data.columns, featureSelection: sel, treatValues: data.treat_values||[], treatMap: defMap, colTypes: data.dtypes||{}, targetValues: data.target_values||[], nanCols: data.nan_cols||[], colStats: data.col_stats||{}}));
        setDetecting(false);
        return;
      }
      setDetectError(data.message || "Spaltenerkennung fehlgeschlagen.");
    } catch(e) {
      setDetectError("Backend nicht erreichbar: " + (e.message || "Netzwerkfehler") + ". Prüfe die Verbindung in der Sidebar.");
    }
    setDetecting(false);
  };

  const toggleType = (col) => {
    setDp(prev => {
      const types = {...(prev.colTypes||{})};
      types[col] = types[col] === "cat" ? "num" : "cat";
      return {...prev, colTypes: types};
    });
  };

  const exportDict = async () => {
    setExporting(true); setExportResult(null);
    try {
      const res = await fetch("./api/export-feature-dict", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          columns: simCols || [],
          featureSelection: dp.featureSelection || {},
          colTypes: dp.colTypes || {},
          target: (dp.targets||[""]).filter(t=>t.trim()),
          target_display: (dp.targets||[""]).filter(t=>t.trim()).join(" + "),
          treatment: dp.treatment || "T",
          scoreName: dp.scoreName || "",
          colStats: dp.colStats || {},
        })
      });
      const data = await res.json();
      if (data.status === "done") {
        setExportResult(data);
        if (data.path) setDp(prev => ({...prev, featurePath: data.path}));
      } else {
        setExportResult({error: data.message || "Export fehlgeschlagen."});
      }
    } catch(e) {
      setExportResult({error: "Backend nicht erreichbar: " + (e.message || "")});
    }
    setExporting(false);
  };

  const toggleFeature = (col) => {
    setDp(prev => {
      const sel = {...(prev.featureSelection||{})};
      sel[col] = !sel[col];
      return {...prev, featureSelection: sel};
    });
  };
  const selectAll = (val) => {
    if(!simCols) return;
    const excl = new Set([...(dp.targets||[""]),dp.treatment||"",dp.scoreName||""].map(s=>(s||"").toUpperCase()));
    setDp(prev => {
      const sel = {...(prev.featureSelection||{})};
      simCols.forEach(c => { if(!excl.has(c)) sel[c] = val; });
      return {...prev, featureSelection: sel};
    });
  };

  const runDataPrep = async () => {
    setDpRunning(true); setDpDone(false); setDpProgress(0); setDpError(null);
    const outPath = dp.outputPath || "runs/data";
    try {
      // Config an Backend senden und DataPrep starten
      const yaml = buildDataPrepYaml(dp, cfg);
      const res = await fetch("./api/run-dataprep", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({yaml})
      });
      const startData = await res.json();
      if (startData.status === "error") {
        setDpError(startData.message || "Datenvorbereitung konnte nicht gestartet werden.");
        setDpRunning(false); setDpProgress(0);
        return;
      }
      // Polling: Fortschritt vom Backend abfragen
      const poll = async () => {
        try {
          const pr = await fetch("./api/progress");
          const state = await pr.json();
          setDpProgress(state.percent || 0);
          if (state.status === "done") {
            setDpRunning(false); setDpDone(true);
            let mlflowUpdate = {};
            try {
              const infoRes = await fetch(`./api/dataprep-info?output_path=${encodeURIComponent(outPath)}`);
              if (infoRes.ok) {
                const info = await infoRes.json();
                if (info.experiment_name) mlflowUpdate.expName = info.experiment_name;
                if (info.run_name) mlflowUpdate.dpRunName = info.run_name;
              }
            } catch(_e) {}
            // NaN-Spalten aus dem tatsächlichen Output (X.parquet) ermitteln,
            // nicht aus der Rohdatei. DataPrep kann Spalten entfernen, NaN füllen, etc.
            let postNanCols = [];
            try {
              const nanRes = await fetch("./api/detect-columns", {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({path: outPath+"/X.parquet"})
              });
              const nanData = await nanRes.json();
              if (nanData.status === "done") {
                postNanCols = nanData.nan_cols || [];
              }
            } catch(_e) {}
            const nanAfterPrep = (dp.fillNa||"(keine)") === "(keine)" && postNanCols.length > 0;
            setDp(prev => ({...prev, nanCols: postNanCols}));
            const _evalMode = dp.evalMode || "cross";
            const hasEval = _evalMode === "external" && dp.evalFiles && dp.evalFiles.filter(f=>f).length > 0;
            const hasTmes = _evalMode === "tmes" && ((dp.evalFileIdxs||[]).length > 0 || dp.evalFileIdx != null);
            setCfg(prev => ({...prev,
              x_file:outPath+"/X.parquet", t_file:outPath+"/T.parquet", y_file:outPath+"/Y.parquet",
              ...((dp.scoreName||"").trim() ? {s_file:outPath+"/S.parquet"} : {}),
              hasNaN:nanAfterPrep, nanCols:nanAfterPrep?postNanCols:[],
              ...(hasEval ? {eval_x_file:outPath+"/X_eval.parquet", eval_t_file:outPath+"/T_eval.parquet", eval_y_file:outPath+"/Y_eval.parquet", ...((dp.scoreName||"").trim() ? {eval_s_file:outPath+"/S_eval.parquet"} : {}), validateOn:"external"} : {}),
              eval_mask_file: hasTmes ? outPath+"/eval_mask.npy" : "",
              dpEvalMode: _evalMode,
              ...mlflowUpdate,
            }));
            return;
          }
          if (state.status === "error") {
            const detail = state.stderr_tail ? "\n\nDetails:\n" + state.stderr_tail : "";
            setDpError((state.message || "Datenvorbereitung fehlgeschlagen.") + detail);
            setDpRunning(false); setDpProgress(0);
            return;
          }
          setTimeout(poll, 10000);
        } catch(e) { setTimeout(poll, 10000); }
      };
      setTimeout(poll, 2000);
      return;
    } catch(e) {
      setDpError("Backend nicht erreichbar: " + (e.message || "Netzwerkfehler") + ". Prüfe die Verbindung in der Sidebar.");
      setDpRunning(false); setDpProgress(0);
    }
  };

  const allTargetU = new Set((dp.targets||[""]).map(t=>(t||"").toUpperCase()));
  const treatU = (dp.treatment||"").toUpperCase();
  const scoreU = (dp.scoreName||"").toUpperCase();
  const reservedCols = new Set([...allTargetU, treatU, ...(scoreU ? [scoreU] : [])]);
  const selectedCount = simCols ? simCols.filter(c => !reservedCols.has(c) && (dp.featureSelection||{})[c] !== false).length : 0;
  const totalAvail = simCols ? simCols.filter(c => !reservedCols.has(c)).length : 0;
  const types = dp.colTypes || {};
  const numCount = simCols ? simCols.filter(c => !reservedCols.has(c) && (dp.featureSelection||{})[c] !== false && types[c]==="num").length : 0;
  const catCount = simCols ? simCols.filter(c => !reservedCols.has(c) && (dp.featureSelection||{})[c] !== false && types[c]==="cat").length : 0;

  return (
    <>
      {/* Experiment-Name prominent ganz oben */}
      <div style={{background:"linear-gradient(135deg,#8a6d1b,#D4A853)",borderRadius:10,padding:"20px 24px",marginBottom:20,color:"#fff"}}>
        <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:12}}>
          <div style={{fontSize:11,textTransform:"uppercase",letterSpacing:1,opacity:.6}}>Experiment</div>
          {(!cfg.expName || cfg.expName==="rubin") && <span style={{fontSize:10,background:"rgba(255,255,255,.2)",color:"#fff",padding:"2px 10px",borderRadius:10,fontWeight:600,border:"1px solid rgba(255,255,255,.3)"}}>Bitte benennen</span>}
        </div>
        <input type="text" value={cfg.expName||""} onChange={e=>setCfg(prev=>({...prev,expName:e.target.value}))} placeholder="z. B. churn_q2_2026" style={{width:"100%",height:44,padding:"0 16px",border:(!cfg.expName||cfg.expName==="rubin")?"2px solid rgba(255,255,255,.5)":"2px solid rgba(255,255,255,.2)",borderRadius:8,fontSize:18,fontWeight:700,background:"rgba(255,255,255,.12)",color:"#fff",outline:"none",boxSizing:"border-box",letterSpacing:.3}} onFocus={e=>{e.target.style.borderColor="rgba(255,255,255,.5)"}} onBlur={e=>{e.target.style.borderColor=(!cfg.expName||cfg.expName==="rubin")?"rgba(255,255,255,.5)":"rgba(255,255,255,.2)"}}/>
        <div style={{fontSize:12,opacity:.5,marginTop:6}}>Name für das gesamte Projekt. DataPrep und Analyse landen als Runs unter diesem Experiment in MLflow.</div>
        <div style={{display:"flex",gap:16,marginTop:12,alignItems:"center"}}>
          <label style={{display:"inline-flex",alignItems:"center",gap:6,fontSize:12.5,cursor:"pointer"}}><input type="checkbox" checked={dp.dpMlflow} onChange={e=>setDp(prev=>({...prev,dpMlflow:e.target.checked}))} style={{accentColor:"#D4A853"}}/><span style={{opacity:.8}}>DataPrep in MLflow loggen</span></label>
          {dp.dpMlflow && <input type="text" value={dp.dpRunName||""} onChange={e=>setDp(prev=>({...prev,dpRunName:e.target.value}))} placeholder="Run-Name (leer = auto)" style={{height:30,padding:"0 10px",border:"1px solid rgba(255,255,255,.2)",borderRadius:6,fontSize:12,background:"rgba(255,255,255,.08)",color:"#fff",outline:"none",width:220,boxSizing:"border-box"}}/>}
        </div>
      </div>

      <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:16}}>
        <span style={{fontSize:9.5,fontWeight:600,textTransform:"uppercase",letterSpacing:1,background:"#fffbeb",color:"#7a5a00",padding:"3px 10px",borderRadius:10,border:"1px solid #d4a853"}}>Optional</span>
        <span style={{fontSize:13.5,color:"#888"}}>Nur nötig, falls Rohdaten noch nicht zu X/T/Y aufbereitet sind.</span>
      </div>

      {/* ── Evaluationsmodus ── */}
      <Sec title="Evaluationsmodus" accent="#D4A853">
        <Info>Bestimmt, wie die Modelle nach dem Training evaluiert werden. Diese Auswahl wird an die Analyse-Pipeline übertragen.</Info>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:10,marginTop:4}}>
          {[
            {k:"cross",label:"Cross-Validation",desc:"K-Fold auf den Trainingsdaten. Alle Daten werden effizient genutzt. Standard für explorative Analysen."},
            {k:"tmes",label:"Train Many, Evaluate Some",desc:"Training und Cross-Prediction auf allen Daten. Evaluation (Metriken, Plots) nur auf den ausgewählten Dateien."},
            {k:"external",label:"External Eval",desc:"Separater Datensatz für Evaluation. Preprocessor wird nur auf Trainingsdaten gefittet — kein Leakage."},
          ].map(o => {
            const active = (dp.evalMode||"cross") === o.k;
            return (
              <label key={o.k} onClick={()=>setDp(prev=>({...prev, evalMode:o.k}))} style={{display:"flex",flexDirection:"column",padding:"14px 16px",borderRadius:10,border:active?"1.5px solid #D4A853":"1.5px solid "+C.border,background:active?"#fffbeb":"#fff",cursor:"pointer",transition:"all 0.15s"}}>
                <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
                  <input type="radio" name="dp_evalMode" checked={active} readOnly style={{accentColor:"#D4A853",pointerEvents:"none"}}/>
                  <span style={{fontSize:13,fontWeight:600,color:active?"#7a5a00":C.dark}}>{o.label}</span>
                </div>
                <div style={{fontSize:11,color:C.textMuted,lineHeight:1.4}}>{o.desc}</div>
              </label>
            );
          })}
        </div>
      </Sec>

      <Tabs tabs={["Dateipfade","Datei-Upload"]} active={dpTab} onSelect={setDpTab}/>

      <Sec title="Rohdaten laden" accent="#D4A853">
        <Info>Eine oder mehrere Rohdateien laden (CSV, Parquet, SAS7BDAT).</Info>
        <div style={{fontSize:12.5,fontWeight:600,color:"#24292f",marginBottom:8}}>Dateien</div>
        {dpTab==="Dateipfade" ? (<>
          {files.map((f,i) => (
            <div key={i} style={{display:"flex",gap:10,alignItems:"flex-start",marginBottom:8}}>
              <span style={{fontSize:11.5,fontWeight:700,color:C.ruby,minWidth:22,paddingTop:8,fontFamily:MONO}}>{i+1}.</span>
              <div style={{flex:1}}>
                <input type="text" value={f} onChange={e=>updateFile(i,e.target.value)} placeholder={"data/raw/datei_"+(i+1)+".csv"} style={{width:"100%",height:38,padding:"0 12px",border:"1.5px solid "+C.border,borderRadius:8,fontSize:13.5,background:"#fff",outline:"none",boxSizing:"border-box"}} onFocus={e=>{e.target.style.borderColor="#D4A853"}} onBlur={e=>{e.target.style.borderColor=C.border}}/>
              </div>
              {files.length > 1 && <button onClick={()=>removeFile(i)} style={{height:38,width:38,border:"1.5px solid "+C.border,borderRadius:8,background:"#fff",cursor:"pointer",fontSize:14,color:"#cf222e",fontWeight:600,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,padding:0}}>x</button>}
            </div>
          ))}
          <Btn small secondary onClick={addFile}>+ Weitere Datei</Btn>
        </>) : (<>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
            {files.map((f,i) => (
              <div key={i}
                onClick={()=>{const inp=document.createElement("input");inp.type="file";inp.accept=".csv,.parquet,.sas7bdat";inp.onchange=async e=>{
                  const file=e.target.files?.[0];if(!file)return;
                  const fd=new FormData();fd.append("file",file);
                  try{const r=await fetch("./api/upload",{method:"POST",body:fd});const d=await r.json();
                    if(d.status==="done"&&d.path){updateFile(i,d.path)}
                    else alert("Upload fehlgeschlagen: "+(d.message||""))
                  }catch(err){alert("Upload fehlgeschlagen: "+(err.message||""))}
                };inp.click()}}
                style={{border:"1.5px solid "+(f.trim()?"#86efac":"#d4b5b8"),borderRadius:10,padding:"20px 16px",textAlign:"center",background:f.trim()?"#f0fdf4":"#faf8f8",cursor:"pointer",transition:"all 0.15s"}}
                onMouseEnter={e=>{e.currentTarget.style.borderColor="#D4A853";e.currentTarget.style.background="#fffbeb"}}
                onMouseLeave={e=>{e.currentTarget.style.borderColor=f.trim()?"#86efac":"#d4b5b8";e.currentTarget.style.background=f.trim()?"#f0fdf4":"#faf8f8"}}>
                <div style={{fontWeight:600,color:f.trim()?"#059669":"#6B0D15",fontSize:12.5}}>{f.trim() ? "✓ "+f.split("/").pop() : "Datei "+(i+1)+" hochladen"}</div>
                <div style={{fontSize:10.5,color:"#999",marginTop:3}}>{f.trim() ? "Klicken zum Ersetzen" : "Klicken zum Hochladen"}</div>
              </div>
            ))}
          </div>
          <div style={{display:"flex",gap:10,marginTop:10}}>
            <Btn small secondary onClick={()=>setFiles(prev=>[...prev,""])}>+ Weitere Datei</Btn>
          </div>
        </>)}

        {files.filter(f=>f.trim()).length > 1 && (<>
          <Divider/>
          <div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:8}}>Mehrdatei-Logik</div>
          <div style={{display:"flex",gap:10}}>
            {[
              {k:"merge",l:"Merge",d:"Alle Dateien vertikal zusammenführen (pd.concat). Fehlende Spalten → NaN. Typisch: mehrere Exporte desselben Schemas."},
              {k:"treatment_only",l:"Treatment Only",d:"Pro Datei nur Treatment-Zeilen (übernehmen). Control (T=0) aus genau einer Datei. Typisch: separate Kampagnendateien je Arm."},
            ].map(o => {
              const active = (dp.multiOpt||"merge") === o.k;
              return (
                <button key={o.k} onClick={()=>setDp(prev=>({...prev,multiOpt:o.k}))} style={{flex:1,padding:"14px 18px",borderRadius:10,border:active?"1.5px solid #D4A853":"1.5px solid "+C.border,background:active?"#fffbeb":"#faf6f6",cursor:"pointer",textAlign:"left"}}>
                  <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:8}}>
                    <div style={{width:14,height:14,borderRadius:7,border:active?"4px solid #D4A853":"2px solid #ccc",background:"#fff",flexShrink:0}}/>
                    <span style={{fontSize:13.5,fontWeight:600,color:active?"#7a5a00":"#333"}}>{o.l}</span>
                  </div>
                  <div style={{fontSize:11.5,color:"#888",lineHeight:1.5}}>{o.d}</div>
                </button>
              );
            })}
          </div>
          {(dp.multiOpt||"merge")==="treatment_only" && (
            <div style={{marginTop:12}}>
              <Sel label="Control-Datei" options={files.filter(f=>f.trim()).map((_,i)=>"Datei "+(i+1))} value={"Datei "+((dp.controlFileIndex||0)+1)} onChange={v=>setDp(prev=>({...prev,controlFileIndex:parseInt(v.replace("Datei ",""))-1}))} help="Aus dieser Datei wird die Control-Gruppe (T=0) entnommen."/>
            </div>
          )}
          <Divider/>
          {(dp.evalMode||"cross")==="tmes" && (<>
          <div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:8}}>Train Many, Evaluate Some</div>
          <Info>Wähle eine oder mehrere Dateien, auf denen die Evaluation (Uplift-Metriken, DRTester-Plots, Policy Values) durchgeführt wird. Alle Dateien werden für Training und Cross-Prediction genutzt — die Auswahl filtert nur die Evaluation.</Info>
          {files.filter(f=>f.trim()).length > 1 ? (
            <div style={{display:"flex",flexDirection:"column",gap:6,marginTop:8}}>
              {files.map((f,i) => {
                if (!f.trim()) return null;
                const evalIdxs = dp.evalFileIdxs || (dp.evalFileIdx != null ? [dp.evalFileIdx] : []);
                const checked = evalIdxs.includes(i);
                return (
                  <label key={i} style={{display:"flex",alignItems:"center",gap:8,padding:"8px 12px",borderRadius:8,border:checked?"1.5px solid "+C.ruby:"1.5px solid "+C.border,background:checked?"rgba(155,17,30,0.03)":"#fff",cursor:"pointer",transition:"all 0.15s"}}>
                    <input type="checkbox" checked={checked} style={{accentColor:C.ruby}} onChange={e=>{
                      const cur = [...(dp.evalFileIdxs || (dp.evalFileIdx != null ? [dp.evalFileIdx] : []))];
                      if(e.target.checked){cur.push(i)}else{const j=cur.indexOf(i);if(j>=0)cur.splice(j,1)}
                      setDp(prev=>({...prev, evalFileIdxs:cur.length?cur:[], evalFileIdx:cur.length===1?cur[0]:null}))
                    }}/>
                    <span style={{fontSize:12.5,fontWeight:checked?600:400,color:checked?C.ruby:C.dark}}>Datei {i+1}</span>
                    <span style={{fontSize:11,color:"#999",flex:1,textOverflow:"ellipsis",overflow:"hidden",whiteSpace:"nowrap"}}>{f.split("/").pop()}</span>
                  </label>
                );
              })}
            </div>
          ) : (
            <div style={{fontSize:11.5,color:"#6b7280",marginTop:6}}>Mehrere Dateien nötig — bei nur einer Datei ist TMES nicht verfügbar.</div>
          )}
          {(dp.evalFileIdxs||[]).length > 0 && <Info type="warn"><strong>Aktiv:</strong> Evaluation auf {(dp.evalFileIdxs||[]).length === 1 ? `Datei ${(dp.evalFileIdxs||[])[0]+1}` : `${(dp.evalFileIdxs||[]).length} Dateien (${(dp.evalFileIdxs||[]).map(i=>i+1).join(", ")})`}. Training und Cross-Prediction auf allen {files.filter(f=>f.trim()).length} Dateien.</Info>}
          </>)}
        </>)}
      </Sec>

      {(dp.evalMode||"cross")==="external" && <Sec title="Externer Evaluationsdatensatz" accent="#6366f1">
        <Info>Separater Datensatz für externe Validierung. Der Preprocessor wird <strong>nur auf den Trainingsdaten</strong> (oben) gefittet und dann auf die Eval-Daten angewendet – kein Data-Leakage.</Info>
        {(()=>{
          if(!(dp.evalFiles||[]).length) setDp(prev=>({...prev,evalFiles:[""]}));
          return null;
        })()}
          <Divider/>
          {dpTab==="Dateipfade" ? (<>
            {(dp.evalFiles||[""]).map((f,i) => (
              <div key={i} style={{display:"flex",gap:8,alignItems:"flex-end",marginBottom:6}}>
                <div style={{flex:1}}>
                  <Inp label={`Eval-Datei ${i+1}`} value={f} onChange={v=>{const nf=[...(dp.evalFiles||[""])];nf[i]=v;setDp(prev=>({...prev,evalFiles:nf}))}} placeholder="runs/data/eval_data.csv"/>
                </div>
                {(dp.evalFiles||[]).length > 1 && <button onClick={()=>{const nf=(dp.evalFiles||[]).filter((_,j)=>j!==i);setDp(prev=>({...prev,evalFiles:nf}))}} style={{border:"none",background:"none",cursor:"pointer",color:"#999",fontSize:16,marginBottom:4}}>✕</button>}
              </div>
            ))}
            <Btn small secondary onClick={()=>setDp(prev=>({...prev,evalFiles:[...(dp.evalFiles||[]),""]}))}>+ Weitere Eval-Datei</Btn>
          </>) : (<>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
              {(dp.evalFiles||[""]).map((f,i) => (
                <div key={i}
                  onClick={()=>{const inp=document.createElement("input");inp.type="file";inp.accept=".csv,.parquet,.sas7bdat";inp.onchange=async e=>{
                    const file=e.target.files?.[0];if(!file)return;
                    const fd=new FormData();fd.append("file",file);
                    try{const r=await fetch("./api/upload",{method:"POST",body:fd});const d=await r.json();
                      if(d.status==="done"&&d.path){const nf=[...(dp.evalFiles||[""])];nf[i]=d.path;setDp(prev=>({...prev,evalFiles:nf}))}
                      else alert("Upload fehlgeschlagen: "+(d.message||""))
                    }catch(err){alert("Upload fehlgeschlagen: "+(err.message||""))}
                  };inp.click()}}
                  style={{border:"1.5px solid "+(f.trim()?"#86efac":"#c4b5fd"),borderRadius:10,padding:"20px 16px",textAlign:"center",background:f.trim()?"#f0fdf4":"#f5f3ff",cursor:"pointer",transition:"all 0.15s"}}
                  onMouseEnter={e=>{e.currentTarget.style.borderColor="#6366f1";e.currentTarget.style.background="#ede9fe"}}
                  onMouseLeave={e=>{e.currentTarget.style.borderColor=f.trim()?"#86efac":"#c4b5fd";e.currentTarget.style.background=f.trim()?"#f0fdf4":"#f5f3ff"}}>
                  <div style={{fontWeight:600,color:f.trim()?"#059669":"#6366f1",fontSize:12.5}}>{f.trim() ? "✓ "+f.split("/").pop() : `Eval-Datei ${i+1} hochladen`}</div>
                  <div style={{fontSize:10.5,color:"#999",marginTop:3}}>{f.trim() ? "Klicken zum Ersetzen" : "Klicken zum Hochladen"}</div>
                </div>
              ))}
            </div>
            <div style={{marginTop:10}}><Btn small secondary onClick={()=>setDp(prev=>({...prev,evalFiles:[...(dp.evalFiles||[]),""]}))}>+ Weitere Eval-Datei</Btn></div>
          </>)}
      </Sec>}

      <Sec title="Pflicht-Spalten zuweisen">
        <Info>Definiere, welche Spalte Treatment (T), Outcome (Y) und optional den historischen Score (S) enthält. Diese Spalten werden nicht als Features genutzt.</Info>

        <div style={{marginTop:0}}>
          <div style={{fontSize:12,fontWeight:600,color:"#24292f",marginBottom:6}}>Target-Spalte (Y)</div>
          <div style={{display:"grid",gridTemplateColumns:(dp.targets||[""]).length>1?"1fr 1fr":"1fr",gap:6}}>
            {(dp.targets||[""]).map((t,i) => (
              <div key={i} style={{display:"flex",gap:6,alignItems:"center"}}>
                <input type="text" value={t} onChange={e=>{const nt=[...(dp.targets||[""])];nt[i]=e.target.value;setDp(prev=>({...prev,targets:nt}))}} placeholder="z.B. Y, OUTCOME, ..." style={{flex:1,height:34,padding:"0 10px",border:"1.5px solid "+C.border,borderRadius:6,fontSize:13,background:"#fff",outline:"none",boxSizing:"border-box"}}/>
                {(dp.targets||[""]).length > 1 && <button onClick={()=>{const nt=(dp.targets||[""]).filter((_,j)=>j!==i);setDp(prev=>({...prev,targets:nt}))}} style={{border:"none",background:"none",cursor:"pointer",color:"#999",fontSize:14,padding:2}}>✕</button>}
              </div>
            ))}
          </div>
          <div style={{display:"flex",alignItems:"center",gap:12,marginTop:6}}>
            <button onClick={()=>setDp(prev=>({...prev,targets:[...(prev.targets||[""]),""]}))} style={{fontSize:11,color:"#9B111E",background:"none",border:"none",cursor:"pointer",fontWeight:600,padding:"2px 0"}}>+ Weitere Target-Spalte</button>
            {(dp.targets||[""]).filter(t=>t.trim()).length > 1 && <span style={{fontSize:10.5,color:"#7a5a00",background:"#fffbeb",padding:"3px 10px",borderRadius:6,border:"1px solid #e8d49c"}}>Y = {(dp.targets||[]).filter(t=>t.trim()).join(" + ")}</span>}
          </div>
        </div>

        <div style={{marginTop:14}}>
          <Inp label="Treatment-Spalte (T)" value={dp.treatment||""} onChange={v=>setDp(prev=>({...prev,treatment:v}))} help="Name der Spalte mit dem Treatment-Indikator" placeholder="z.B. T, TREATMENT, ..."/>
        </div>

        <div style={{marginTop:14}}>
          <Inp label="Score-Spalte (S, optional)" value={dp.scoreName!=null?dp.scoreName:""} onChange={v=>setDp(prev=>({...prev,scoreName:v}))} help="Historischer Score für Benchmark-Vergleich"/>
        </div>

        {/* Validierung: Existieren die zugewiesenen Spalten in den erkannten Daten? */}
        {simCols && simCols.length > 0 && (() => {
          const colSet = new Set(simCols.map(c => c.toUpperCase()));
          const warns = [];
          (dp.targets||[""]).filter(t=>t.trim()).forEach(t => {
            if (!colSet.has(t.toUpperCase())) warns.push({col: t, role: "Target (Y)"});
          });
          const treat = (dp.treatment||"").trim();
          if (treat && !colSet.has(treat.toUpperCase())) warns.push({col: treat, role: "Treatment (T)"});
          const score = (dp.scoreName||"").trim();
          if (score && !colSet.has(score.toUpperCase())) warns.push({col: score, role: "Score (S)"});
          if (warns.length === 0) return null;
          return (<Info type="warn"><strong>{warns.length === 1 ? "Spalte" : warns.length + " Spalten"} nicht in den Daten gefunden:</strong>{" "}
            {warns.map((w, i) => <span key={i}>{i > 0 ? ", " : ""}<span style={{fontFamily:MONO,fontWeight:600}}>{w.col}</span> ({w.role})</span>)}
            . Prüfe die Schreibweise — die Spalten werden bei DataPrep automatisch in Großbuchstaben konvertiert.</Info>);
        })()}

        <Divider/>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
          <div>
            <Toggle label="Target binärisieren (Y > 0 → 1)" checked={dp.binaryTarget||false} onChange={v=>setDp(prev=>({...prev,binaryTarget:v}))} help="Wandelt alle Werte > 0 in 1 um."/>
            {(() => {
              const tv = dp.targetValues || [];
              const isBinary = tv.length > 0 && tv.every(v => v === 0 || v === 1);
              const hasData = tv.length > 0;
              return (<>
                {hasData && isBinary && !dp.binaryTarget && <Info type="success">Target enthält nur 0/1 – bereits binär.</Info>}
                {hasData && isBinary && dp.binaryTarget && <Info>Bereits binär – Binärisierung hat keinen Effekt.</Info>}
                {hasData && !isBinary && !dp.binaryTarget && <Info type="warn">Nicht-binäre Werte: <span style={{fontFamily:MONO,fontWeight:600}}>{tv.slice(0,8).join(", ")}{tv.length>8?"...":""}</span></Info>}
                {hasData && !isBinary && dp.binaryTarget && <Info type="success">Aktiv – Werte werden zu 0/1 (Y {">"} 0 → 1).</Info>}
              </>);
            })()}
          </div>
          <Toggle label="Score als Feature nutzen" checked={dp.scoreAsFeature||false} onChange={v=>setDp(prev=>({...prev,scoreAsFeature:v}))} help="Historischen Score zusätzlich in die Feature-Matrix X aufnehmen"/>
        </div>

        <div style={{marginTop:14}}><Btn small onClick={detectColumns} disabled={detecting}>{detecting ? "Spalten werden erkannt ..." : "Spalten aus Datei erkennen"}</Btn></div>
        {detectError && <Info type="error">{detectError}</Info>}

        {(dp.treatValues||[]).length > 0 && (<>
          <Divider/>
          <div style={{fontSize:13,fontWeight:600,color:C.dark,marginBottom:8}}>Treatment-Ausprägungen zuordnen</div>
          <Info>Jede Ausprägung in der Treatment-Spalte muss einem numerischen Wert zugeordnet werden: <strong>0 = Control</strong>, <strong>1, 2, ... = Treatment-Arme</strong>. Bei Binary Treatment nur 0 und 1. Bei Multi-Treatment beliebig viele Arme.</Info>
          <div style={{border:"1px solid #ede6e7",borderRadius:8,overflow:"hidden",marginTop:8}}>
            <div style={{display:"grid",gridTemplateColumns:"1fr 120px 1fr",fontSize:11.5,fontWeight:600,padding:"8px 14px",background:"#6B0D15",color:"#fff"}}>
              <span>Rohwert</span><span>Zuordnung</span><span>Bedeutung</span>
            </div>
            {(dp.treatValues||[]).map((val,i) => {
              const mapped = (dp.treatMap||{})[val];
              const numVal = mapped != null ? Number(mapped) : i;
              const isControl = numVal === 0;
              return (
                <div key={val} style={{display:"grid",gridTemplateColumns:"1fr 120px 1fr",padding:"9px 14px",borderBottom:"1px solid #f5f0f0",background:isControl?"#fffbeb":i%2===0?"#fff":"#fdfafa",alignItems:"center"}}>
                  <span style={{fontSize:13,fontFamily:MONO,fontWeight:500,color:"#333"}}>{val}</span>
                  <div>
                    <select value={numVal} onChange={e=>{
                      const v = Number(e.target.value);
                      setDp(prev => ({...prev, treatMap: {...(prev.treatMap||{}), [val]: v}}));
                    }} style={{width:"100%",padding:"5px 8px",border:"1.5px solid "+C.border,borderRadius:6,fontSize:13,background:"#fff",cursor:"pointer",fontFamily:MONO,fontWeight:600,color:isControl?"#D4A853":"#9B111E"}}>
                      <option value={0}>0</option>
                      {[1,2,3,4,5,6,7,8,9].map(n => <option key={n} value={n}>{n}</option>)}
                    </select>
                  </div>
                  <span style={{fontSize:12.5,fontWeight:500,color:isControl?"#7a5a00":"#6B0D15"}}>
                    {isControl ? "Control (Referenzgruppe)" : "Treatment " + numVal}
                  </span>
                </div>
              );
            })}
          </div>
          {(() => {
            const vals = Object.values(dp.treatMap||{});
            const hasControl = vals.includes(0);
            const unique = new Set(vals);
            const hasDups = unique.size < vals.length;
            const arms = vals.filter(v=>v>0).length;
            return (<>
              {!hasControl && <Info type="error">Keine Ausprägung als Control (0) zugeordnet.</Info>}
              {hasDups && <Info type="warn">Mehrere Ausprägungen haben denselben Wert – sie werden zusammengelegt.</Info>}
              <div style={{marginTop:8,display:"flex",gap:10}}>
                <div style={{padding:"5px 14px",borderRadius:8,border:"1.5px solid #D4A853",fontSize:12,fontWeight:600,color:"#7a5a00"}}>Control: {hasControl?1:0}</div>
                <div style={{padding:"5px 14px",borderRadius:8,border:"1.5px solid #9B111E",fontSize:12,fontWeight:600,color:"#6B0D15"}}>Treatment-Arme: {arms}</div>
                <div style={{padding:"5px 14px",borderRadius:8,border:"1.5px solid #888",fontSize:12,fontWeight:600,color:"#555"}}>{arms > 1 ? "Multi-Treatment" : arms === 1 ? "Binary Treatment" : "–"}</div>
              </div>
            </>);
          })()}
        </>)}
      </Sec>

      <Sec title="Feature Dictionary laden (optional)" accent="#6366f1">
        <Info>Lade optional ein bestehendes Feature Dictionary (Excel mit NAME, ROLE, LEVEL). Nur <code style={{background:"#f8f0f0",padding:"1px 5px",borderRadius:3}}>ROLE=INPUT</code> wird als Feature übernommen, <code style={{background:"#f8f0f0",padding:"1px 5px",borderRadius:3}}>LEVEL=NOMINAL</code> als kategorisch. Du kannst diesen Schritt überspringen und direkt unten manuell arbeiten.</Info>
        {dpTab==="Dateipfade" ? (
          <Inp label="Feature Dictionary Pfad" placeholder="runs/exports/feature_dictionary.xlsx" value={dp.featurePath} onChange={v=>setDp(prev=>({...prev,featurePath:v}))}/>
        ) : (
          <div style={{border:"1.5px solid "+(dp.featurePath?"#86efac":"#c4b5fd"),borderRadius:10,padding:"20px 16px",textAlign:"center",background:dp.featurePath?"#f0fdf4":"#f5f3ff",cursor:"pointer",transition:"all 0.15s"}}
            onClick={()=>{const inp=document.createElement("input");inp.type="file";inp.accept=".xlsx,.xls,.csv";inp.onchange=async e=>{
              const file=e.target.files?.[0];if(!file)return;
              const fd=new FormData();fd.append("file",file);
              try{const r=await fetch("./api/upload",{method:"POST",body:fd});const d=await r.json();
                if(d.status==="done"&&d.path){setDp(prev=>({...prev,featurePath:d.path}))}
                else alert("Upload fehlgeschlagen: "+(d.message||""))
              }catch(err){alert("Upload fehlgeschlagen: "+(err.message||""))}
            };inp.click()}}
            onMouseEnter={e=>{e.currentTarget.style.borderColor="#6366f1";e.currentTarget.style.background="#ede9fe"}}
            onMouseLeave={e=>{e.currentTarget.style.borderColor=dp.featurePath?"#86efac":"#c4b5fd";e.currentTarget.style.background=dp.featurePath?"#f0fdf4":"#f5f3ff"}}>
            <div style={{fontWeight:600,color:dp.featurePath?"#059669":"#6366f1",fontSize:12.5}}>{dp.featurePath ? "✓ "+dp.featurePath.split("/").pop() : "Dictionary hochladen (.xlsx / .csv)"}</div>
            <div style={{fontSize:10.5,color:"#999",marginTop:3}}>{dp.featurePath ? "Klicken zum Ersetzen" : "Klicken zum Hochladen"}</div>
          </div>
        )}
        {dp.featurePath && (
          <div style={{marginTop:10,display:"flex",gap:10,alignItems:"center",flexWrap:"wrap"}}>
            <Btn small onClick={async()=>{
              setDetecting(true); setDetectError(null);
              try {
                const res = await fetch("./api/apply-dictionary", {
                  method:"POST", headers:{"Content-Type":"application/json"},
                  body: JSON.stringify({path:dp.featurePath, data_path:(files||[""])[0]||"", delimiter:dp.delimiter||","})
                });
                const d = await res.json();
                if(d.status==="done") {
                  const allCols = d.all_data_columns && d.all_data_columns.length > 0 ? d.all_data_columns : (simCols || d.matched_features || []);
                  const newSel = {};
                  const matchedSet = new Set(d.matched_features || []);
                  allCols.forEach(c => { newSel[c] = matchedSet.has(c); });
                  const newTypes = {...(d.data_dtypes||{}), ...(dp.colTypes||{}), ...(d.col_types||{})};
                  const rsvd = new Set([...(dp.targets||[""]),(dp.treatment||""),(dp.scoreName||"")].filter(Boolean).map(s=>s.toUpperCase()));
                  allCols.forEach(k => { if(rsvd.has(k.toUpperCase())) newSel[k] = false; });
                  setDp(prev => ({...prev, featureSelection:newSel, colTypes:newTypes, dictResult:d}));
                  setSimCols(allCols);
                } else {
                  setDetectError(d.message || "Dictionary konnte nicht angewendet werden.");
                }
              } catch(e) {
                setDetectError("Backend nicht erreichbar: "+(e.message||"Netzwerkfehler"));
              }
              setDetecting(false);
            }} disabled={detecting}>{detecting ? "Wird angewendet ..." : "Dictionary anwenden"}</Btn>
            {dp.dictResult && (
              <span style={{fontSize:12,color:"#059669",fontWeight:500}}>
                ✓ {dp.dictResult.total_input} INPUT, {(dp.dictResult.matched_features||[]).length} in Daten
                {(dp.dictResult.missing_in_data||[]).length > 0 && <span style={{color:"#cf222e"}}> ({dp.dictResult.missing_in_data.length} fehlen)</span>}
              </span>
            )}
          </div>
        )}
        {detectError && <Info type="error">{detectError}</Info>}
      </Sec>

      <Sec title="Features manuell bearbeiten">
        {!simCols && <Info type="warn">Erst oben unter „Pflicht-Spalten" auf „Spalten erkennen" klicken, dann kannst du hier Features auswählen und Datentypen anpassen.</Info>}
        {simCols && (<>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
            <span style={{fontSize:13,fontWeight:600,color:C.dark}}>{selectedCount} / {totalAvail} Features ({numCount} num, {catCount} cat)</span>
            <div style={{display:"flex",gap:6}}>
              <Btn small secondary onClick={()=>selectAll(true)}>Alle</Btn>
              <Btn small secondary onClick={()=>selectAll(false)}>Keine</Btn>
            </div>
          </div>
          <div style={{maxHeight:420,overflowY:"auto",border:"1px solid #ede6e7",borderRadius:8}}>
            <div style={{display:"grid",gridTemplateColumns:"44px 1fr 60px 70px 70px 60px",fontSize:10.5,fontWeight:600,padding:"8px 14px",background:"#6B0D15",color:"#fff",position:"sticky",top:0,letterSpacing:0.3}}>
              <span></span><span>Spalte</span><span>Typ</span><span>Unique</span><span>NaN %</span><span>Rolle</span>
            </div>
            {simCols.map((col,i) => {
              const isReserved = reservedCols.has(col);
              const isSelected = !isReserved && (dp.featureSelection||{})[col] !== false;
              const roleLabel = col===treatU?"Treatment":allTargetU.has(col)?"Outcome":col===scoreU?"Score":null;
              const colType = (types[col]) || "num";
              const isNum = colType === "num";
              const st = (dp.colStats||{})[col] || {};
              return (
                <div key={col} style={{display:"grid",gridTemplateColumns:"44px 1fr 60px 70px 70px 60px",padding:"6px 14px",borderBottom:"1px solid #f5f0f0",background:isReserved?"#fffbeb":isSelected?"#fff":"#fafafa",alignItems:"center",opacity:isReserved?0.5:1}}>
                  <div>
                    {!isReserved && (
                      <div onClick={()=>toggleFeature(col)} style={{width:18,height:18,borderRadius:4,border:isSelected?"2px solid #9B111E":"2px solid #ccc",background:isSelected?"#9B111E":"#fff",cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center"}}>
                        {isSelected && <span style={{color:"#fff",fontSize:11,fontWeight:700}}>✓</span>}
                      </div>
                    )}
                    {isReserved && <span style={{fontSize:10,color:"#D4A853",fontWeight:700}}>–</span>}
                  </div>
                  <span style={{fontSize:12.5,fontWeight:isReserved?600:400,fontFamily:MONO,color:isReserved?"#7a5a00":"#333"}}>{col}</span>
                  <div>
                    {!isReserved && isSelected && (
                      <button onClick={()=>toggleType(col)} style={{padding:"2px 8px",borderRadius:10,fontSize:10,fontWeight:600,border:"1.5px solid "+(isNum?"#9B111E":"#6366f1"),background:isNum?"#FDF2F3":"#f0f0ff",color:isNum?"#9B111E":"#6366f1",cursor:"pointer"}} title={"Klick: umschalten auf "+(isNum?"kategorisch":"numerisch")}>
                        {isNum?"NUM":"CAT"}
                      </button>
                    )}
                  </div>
                  <span style={{fontSize:11,fontFamily:MONO,color:"#666"}}>{st.n_unique!=null?st.n_unique:"–"}</span>
                  <span style={{fontSize:11,fontFamily:MONO,color:st.null_pct>0?"#cf222e":"#888"}}>{st.null_pct!=null?(st.null_pct>0?st.null_pct+"%":"0%"):"–"}</span>
                  <span style={{fontSize:10.5,color:isReserved?"#D4A853":"#bbb"}}>{roleLabel||"Feature"}</span>
                </div>
              );
            })}
          </div>
          <Divider/>
          <div style={{display:"flex",gap:10,alignItems:"center",flexWrap:"wrap"}}>
            <Btn small onClick={exportDict} disabled={exporting}>{exporting ? "Exportiere ..." : "Auswahl als Feature Dictionary exportieren"}</Btn>
            {exportResult && !exportResult.error && (
              <span style={{fontSize:12,color:"#059669",fontWeight:500}}>
                ✓ {exportResult.path} ({exportResult.n_input} INPUT, {exportResult.n_exclude} EXCLUDE)
                {" "}<a href={"./api/download/"+exportResult.path} style={{color:"#9B111E",textDecoration:"underline",fontWeight:600}}>Herunterladen</a>
              </span>
            )}
            {exportResult && exportResult.error && <span style={{fontSize:12,color:"#cf222e"}}>{exportResult.error}</span>}
          </div>
        </>)}
      </Sec>
      <Sec title="Weitere Optionen">
        {(() => {
          const fileList = (dp.files||[""]).filter(f=>f);
          const hasCSV = fileList.some(f => /\.(csv|tsv|txt)$/i.test(f));
          const hasSAS = fileList.some(f => /\.sas7bdat$/i.test(f));
          const hasMulti = fileList.length > 1;
          return (<>

          <div style={{fontSize:12,fontWeight:600,textTransform:"uppercase",letterSpacing:".5px",color:"#999",marginBottom:8}}>Datenqualität</div>
          {(() => {
            // NaN-Spalten auf ausgewählte Features filtern (ohne Target, Treatment, Score, deselektierte)
            const sel = dp.featureSelection||{};
            const reserved = new Set([...((dp.targets||[""]).map(t=>(t||"").toUpperCase())), (dp.treatment||"").toUpperCase(), ...((dp.scoreName||"").trim() ? [(dp.scoreName||"").toUpperCase()] : [])]);
            const nanFeatures = (dp.nanCols||[]).filter(c => !reserved.has(c.toUpperCase()) && sel[c] !== false);
            return (<>
          <Sel label="NaN-Behandlung" options={["(keine)","zero","median","mean","mode","max"]} value={dp.fillNa||"(keine)"} onChange={v=>{setDp(prev=>({...prev,fillNa:v}));if(v!=="(keine)"){setCfg(prev=>({...prev,hasNaN:false,nanCols:[]}))}else if(nanFeatures.length>0){setCfg(prev=>({...prev,hasNaN:true,nanCols:nanFeatures}))}}} help="Fehlende Werte in Features auffüllen. Bei Auswahl einer Methode werden NaN vor der Analyse bereinigt."/>
          {nanFeatures.length > 0 && (dp.fillNa||"(keine)")==="(keine)" && (
            <Info type="warn"><strong>{nanFeatures.length} Feature-Spalten mit fehlenden Werten.</strong> Ohne NaN-Behandlung werden CausalForestDML, CausalForest und die Feature-Selektionsmethode CausalForest automatisch übersprungen. Alle anderen Modelle (LightGBM/CatBoost-basiert) sind davon nicht betroffen. Wähle oben eine Auffüll-Methode, um alle Modelle nutzen zu können.</Info>
          )}
          {nanFeatures.length > 0 && (dp.fillNa||"(keine)")!=="(keine)" && (
            <Info type="success"><strong>NaN werden bereinigt</strong> ({dp.fillNa}). Alle Modelle inkl. CausalForestDML und CausalForest sind verfügbar.</Info>
          )}
          {nanFeatures.length === 0 && simCols && (
            <Info type="success">Keine fehlenden Werte in Features erkannt. Alle Modelle sind verfügbar.</Info>
          )}
            </>);
          })()}

          <div style={{marginTop:14}}>
            
          </div>

          <Divider/>
          <div style={{fontSize:12,fontWeight:600,textTransform:"uppercase",letterSpacing:".5px",color:"#999",marginBottom:8}}>Bereinigung</div>
          <Row>
            <Col>
              <Toggle label="Deduplizierung" checked={dp.dedup||false} onChange={v=>setDp(prev=>({...prev,dedup:v}))} help="Auf einen Eintrag pro Kunde reduzieren (keep=first). Verhindert Leakage bei Cross-Prediction."/>
              {dp.dedup&&<Inp label="ID-Spalte" value={dp.dedupCol||""} onChange={v=>setDp(prev=>({...prev,dedupCol:v}))} placeholder="PARTNER_ID"/>}
            </Col>
            {hasMulti && <Col>
              <Toggle label="Treatment-Balance" checked={dp.balanceTreat||false} onChange={v=>setDp(prev=>({...prev,balanceTreat:v}))} help="Per Random-Downsampling die Treatment-Raten über alle Dateien angleichen"/>
            </Col>}
          </Row>
          {dp.dedup&&<Info type="warn"><strong>Deduplizierung aktiv:</strong> Reduktion auf 1 Sample pro Kunde verhindert Leakage, wenn mehrere Einträge desselben Kunden in verschiedene Folds fallen.</Info>}
          {dp.balanceTreat&&<Info type="warn"><strong>Treatment-Balance aktiv:</strong> Bei mehr als 5 Prozentpunkten Abweichung zwischen Dateien werden Zeilen per Random-Downsampling entfernt.</Info>}

          <Divider/>
          <div style={{fontSize:12,fontWeight:600,textTransform:"uppercase",letterSpacing:".5px",color:"#999",marginBottom:8}}>Ausgabe</div>
          <Inp label="Ausgabepfad" value={dp.outputPath||"runs/data"} onChange={v=>setDp(prev=>({...prev,outputPath:v}))} help="Zielordner für X/T/Y.parquet + Preprocessing-Artefakte"/>

          {(hasCSV || hasSAS) && <><Divider/>
          <div style={{fontSize:12,fontWeight:600,textTransform:"uppercase",letterSpacing:".5px",color:"#999",marginBottom:8}}>Dateiformat</div>
          <Row>
            {hasCSV && <Col><Sel label="Trennzeichen (CSV)" options={[",",";","|","\t"]} value={dp.delimiter||","} onChange={v=>setDp(prev=>({...prev,delimiter:v}))}/></Col>}
            {hasSAS && <Col><Sel label="SAS-Encoding" options={["utf-8","latin-1","cp1252","iso-8859-1"]} value={dp.sasEncoding||"utf-8"} onChange={v=>setDp(prev=>({...prev,sasEncoding:v}))} help="Zeichensatz für SAS7BDAT-Dateien"/></Col>}
          </Row></>}

          <Expander title="Erweiterte DataPrep-Einstellungen">
            <Inp label="Chunksize" type="number" value={dp.chunksize||300000} onChange={v=>setDp(prev=>({...prev,chunksize:Number(v)}))} help="Zeilenzahl pro Chunk beim Einlesen großer Dateien"/>
          </Expander>

          </>);
        })()}
      </Sec>

      <Sec title="Config-Vorschau">
        {(() => {
          const y = buildDataPrepYaml(dp, cfg);
          return (<>
            <div style={{background:"#1e1e2e",borderRadius:10,padding:"16px 20px",fontFamily:MONO,fontSize:11.5,whiteSpace:"pre-wrap",maxHeight:340,overflowY:"auto",lineHeight:1.65,color:"#cdd6f4",boxShadow:"inset 0 2px 8px rgba(0,0,0,0.15)",border:"1px solid #313244"}}>{y}</div>
            <div style={{marginTop:10,display:"flex",gap:8}}>
              <Btn small onClick={()=>navigator.clipboard?.writeText(y)}>Kopieren</Btn>
              <Btn small secondary onClick={()=>{const b=new Blob([y],{type:"text/yaml"});const u=URL.createObjectURL(b);const a=document.createElement("a");a.href=u;a.download="config_dataprep.yml";a.click();}}>YAML herunterladen</Btn>
            </div>
          </>);
        })()}
      </Sec>

      <Sec title="Datenvorbereitung starten" accent="#D4A853">
        {!dpRunning && !dpDone && !dpError && <Btn onClick={runDataPrep}>Datenvorbereitung starten</Btn>}
        {dpError && !dpRunning && (
          <>
            <Info type="error"><strong>Datenvorbereitung fehlgeschlagen:</strong> {dpError}</Info>
            <div style={{marginTop:12,display:"flex",gap:10}}>
              <Btn onClick={()=>{setDpError(null);runDataPrep()}}>Erneut versuchen</Btn>
              <Btn small secondary onClick={()=>setPg("datafiles")}>Überspringen</Btn>
            </div>
          </>
        )}
        {dpRunning && (
          <div>
            <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:8}}>
              <span style={{fontSize:13,fontWeight:600,color:"#D4A853"}}>Daten werden aufbereitet ...</span>
              <span style={{fontSize:12,fontFamily:MONO,color:"#888"}}>{dpProgress}%</span>
            </div>
            <div style={{height:5,background:"#ede6e7",borderRadius:3,overflow:"hidden"}}><div style={{height:"100%",background:"linear-gradient(90deg,#D4A853,#e8c36a)",borderRadius:3,transition:"width 0.3s",width:dpProgress+"%"}}/></div>
          </div>
        )}
        {dpDone && (<>
          <Info type="success"><strong>Datenvorbereitung abgeschlossen.</strong> X.parquet, T.parquet und Y.parquet wurden erzeugt. Pfade wurden automatisch auf der Daten-Seite eingetragen.{(dp.evalFiles||[]).some(f=>f) && " Eval-Dateien wurden mit dem Train-Preprocessor transformiert. Validierungsmodus wurde auf 'external' gesetzt."}{dp.dpMlflow && " MLflow-Logging wurde aktiviert."}</Info>
          {dp.dpMlflow && cfg.dpRunName && <div style={{fontSize:11.5,color:"#7a5a00",background:"#fffbeb",border:"1px solid #d4a853",borderRadius:8,padding:"6px 12px",marginTop:6}}>
            <strong>MLflow:</strong> Run „{cfg.dpRunName}“ im Experiment „{cfg.expName||"rubin"}“ angelegt.
          </div>}
          <div style={{marginTop:12,display:"flex",gap:10}}>
            <Btn onClick={()=>setPg("datafiles")}>Weiter zu Daten</Btn>
            <Btn small secondary onClick={()=>{setDpDone(false);setDpProgress(0)}}>Erneut starten</Btn>
          </div>
        </>)}
        {!dpRunning && !dpDone && <div style={{marginTop:12}}><Btn small secondary onClick={()=>setPg("datafiles")}>Überspringen – direkt zu Daten</Btn></div>}
      </Sec>
    </>
  );
};

// ── Data Page ──

// ── YAML Import: parses config.yml back to cfg state ──