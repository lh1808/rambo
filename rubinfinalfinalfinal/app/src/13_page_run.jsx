const PRun = ({cfg,setPg,sp,spFmt,onRunningChange,onDoneChange,totalFits}) => {
  const issues = validate(cfg);
  const [running,setRunning] = useState(false);
  useEffect(() => { if(onRunningChange) onRunningChange(running); }, [running]);
  useEffect(() => { if(onDoneChange) onDoneChange(done); }, [done]);
  const [step,setStep] = useState(null);
  const [error,setError] = useState(null);
  const [done,setDone] = useState(false);
  const [completed,setCompleted] = useState(new Set());
  const [elapsed,setElapsed] = useState(0);
  const [showReport,setShowReport] = useState(false);
  const [stepProgress,setStepProgress] = useState(0);
  const [stepDurations,setStepDurations] = useState({});
  const timerRef = useRef(null);
  const timeoutsRef = useRef([]);
  const intervalsRef = useRef([]);
  const mountedRef = useRef(true);
  const pollingRef = useRef(null);
  const [reportUrl, setReportUrl] = useState(null);
  const [resultFiles, setResultFiles] = useState([]);
  const [logText, setLogText] = useState("");
  const [showLogs, setShowLogs] = useState(true);

  const [disconnected, setDisconnected] = useState(false);
  const failCountRef = useRef(0);

  const enabledRef = useRef(new Set());
  enabledRef.current = new Set(["load","train","eval","report"]);
  if(cfg.fsEnabled) enabledRef.current.add("fs");
  if(cfg.tuningEnabled) enabledRef.current.add("tune");
  if(cfg.fmtEnabled) enabledRef.current.add("fmt");
  if(cfg.cfTune && (cfg.models||[]).some(m => m==="CausalForestDML" || m==="CausalForest")) enabledRef.current.add("grf");
  if(cfg.surrEnabled) enabledRef.current.add("surrogate");
  if(cfg.bundleEnabled) enabledRef.current.add("bundle");
  if(cfg.explEnabled) enabledRef.current.add("explain");

  useEffect(() => {
    mountedRef.current = true;

    // ── Recovery: Beim App-Start prüfen ob eine Analyse läuft ──
    // Nützlich nach Browser-Refresh oder wenn App neu geladen wird.
    (async () => {
      try {
        const pr = await fetch("./api/progress");
        const state = await pr.json();
        if (state.status === "running" && state.pid) {
          // Analyse läuft noch → State wiederherstellen + Polling starten
          setRunning(true); setDone(false); setError(null);
          if (state.step) setStep(_resolveStepId(state.step));
          setStepProgress(state.percent || 0);
          _startPolling();
        } else if (state.status === "done") {
          // Analyse war fertig → Ergebnisse laden
          setDone(true); setRunning(false);
          try { const r = await fetch("./api/report"); const d = await r.json(); if(d.report_url) setReportUrl(d.report_url); } catch(e) {}
          try { const r = await fetch("./api/results"); const d = await r.json(); if(d.files) setResultFiles(d.files); } catch(e) {}
          setShowReport(true);
          const allIds = STEPS.filter(s => enabledRef.current.has(s.id)).map(s => s.id);
          setCompleted(new Set(allIds));
        } else if (state.status === "error" && state.message && state.message !== "run_analysis gestartet...") {
          setError((state.message || "Analyse fehlgeschlagen") + (state.stderr_tail ? "\n\nDetails:\n" + state.stderr_tail : ""));
        }
      } catch(e) { /* Server nicht erreichbar beim Start — OK */ }
    })();

    return () => {
      mountedRef.current = false;
      if(timerRef.current) clearInterval(timerRef.current);
      timeoutsRef.current.forEach(t => clearTimeout(t));
      intervalsRef.current.forEach(t => clearInterval(t));
      stopPolling();
    };
  }, []);

  useEffect(() => {
    if(running) {
      timerRef.current = setInterval(() => {
        if(mountedRef.current) setElapsed(e => e + 1);
      }, 1000);
    }
    return () => { if(timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; } };
  }, [running]);

  const stopPolling = () => { if(pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; } };

  // ── Step-Name Mapping (shared between startAnalysis + recovery) ──
  const _stepNameMap = {};
  STEPS.forEach(s => {
    _stepNameMap[s.label.toLowerCase()] = s.id;
    s.label.toLowerCase().split(/[&\s]+/).filter(w=>w.length>3).forEach(w => { _stepNameMap[w] = s.id; });
  });
  Object.assign(_stepNameMap, {
    "daten laden": "load", "loading": "load", "preprocessing": "load",
    "feature-selektion": "fs", "feature selection": "fs", "feature_selection": "fs",
    "base-learner-tuning": "tune", "tuning": "tune", "base learner": "tune",
    "final-model-tuning": "fmt", "final model tuning": "fmt", "final-model": "fmt", "fmt": "fmt",
    "training": "train", "cross-predictions": "train", "cross predictions": "train",
    "training & predictions": "train", "predictions": "train",
    "grf-tuning": "grf", "grf tuning": "grf", "grf": "grf",
    "grf-tuning & training": "train",
    "evaluation": "eval", "metriken": "eval", "evaluation & metriken": "eval",
    "surrogate": "surrogate", "surrogate-tree": "surrogate",
    "bundle": "bundle", "bundle-export": "bundle", "export": "bundle",
    "explainability": "explain", "shap": "explain", 
    "report": "report", "html-report": "report",
  });

  const _resolveStepId = (stepName) => {
    if (!stepName) return null;
    const lower = stepName.toLowerCase().trim();
    if (_stepNameMap[lower]) return _stepNameMap[lower];
    for (const [key, id] of Object.entries(_stepNameMap)) {
      if (lower.includes(key) || key.includes(lower)) return id;
    }
    return null;
  };

  // ── Polling-Loop (wiederverwendbar für Start + Recovery) ──
  const _startPolling = () => {
    stopPolling();
    let lastStepIndex = 0;
    const stepStartTimes = {};

    pollingRef.current = setInterval(async () => {
      if(!mountedRef.current) { stopPolling(); return; }
      try {
        const pr = await fetch("./api/progress");
        const state = await pr.json();
        // Verbindung OK → Disconnect-Counter zurücksetzen
        if(failCountRef.current > 0) { failCountRef.current = 0; if(mountedRef.current) setDisconnected(false); }
        if (state.step && state.step_index > 0) {
          const stepId = _resolveStepId(state.step) || STEPS[Math.min((state.step_index||1)-1, STEPS.length-1)]?.id;
          if (stepId) {
            setStep(stepId);
            setStepProgress(state.percent || 0);
            if (state.step_index !== lastStepIndex) {
              // Mark ALL enabled steps BEFORE the current one as completed.
              // This handles: first poll missed early steps, fast steps that
              // completed within the 3s polling interval, etc.
              const enabledIds = STEPS.filter(s => enabledRef.current.has(s.id)).map(s => s.id);
              const currentIdx = enabledIds.indexOf(stepId);
              if (currentIdx > 0) {
                const priorIds = enabledIds.slice(0, currentIdx);
                setCompleted(prev => {
                  const n = new Set(prev);
                  priorIds.forEach(id => n.add(id));
                  return n;
                });
              }
              // Duration of previous step (if tracked)
              const prevId = stepStartTimes._lastStepId;
              if(prevId && stepStartTimes[prevId]) {
                setStepDurations(prev => ({...prev, [prevId]: (Date.now()-stepStartTimes[prevId])/1000}));
              }
              stepStartTimes[stepId] = Date.now();
              stepStartTimes._lastStepId = stepId;
              lastStepIndex = state.step_index;
            }
          }
        }
        // Live-Logs aktualisieren (stderr = Python-Logging, stdout = [rubin]-Progress)
        const parts = [];
        if (state.stdout_tail) parts.push(state.stdout_tail);
        if (state.stderr_tail) parts.push(state.stderr_tail);
        if (parts.length) setLogText(parts.join("\n"));
        if (state.status === "done") {
          stopPolling();
          const allIds = STEPS.filter(s => enabledRef.current.has(s.id)).map(s => s.id);
          setCompleted(new Set(allIds));
          setStep(null); setStepProgress(0);
          setRunning(false);
          try { const repRes = await fetch("./api/report"); const repData = await repRes.json(); if(repData.report_url) setReportUrl(repData.report_url); } catch(e) {}
          try { const resRes = await fetch("./api/results"); const resData = await resRes.json(); if(resData.files) setResultFiles(resData.files); } catch(e) {}
          setDone(true); setShowReport(true);
          return;
        }
        if (state.status === "error") {
          stopPolling();
          const detail = state.stderr_tail ? "\n\nDetails:\n" + state.stderr_tail : "";
          setError((state.message || "Analyse fehlgeschlagen") + detail);
          setRunning(false);
          return;
        }
      } catch(e) {
        // Disconnect-Tracking: Nach 3 aufeinanderfolgenden Fehlern → UI zeigt Warnung
        failCountRef.current++;
        if(failCountRef.current >= 3 && mountedRef.current) setDisconnected(true);
      }
    }, 3000);
  };

  const startAnalysis = async () => {
    timeoutsRef.current.forEach(t => clearTimeout(t));
    intervalsRef.current.forEach(t => clearInterval(t));
    timeoutsRef.current = []; intervalsRef.current = [];
    stopPolling();
    setRunning(true); setDone(false); setError(null);
    setCompleted(new Set()); setElapsed(0); setShowReport(false);
    setStep(null); setStepProgress(0); setStepDurations({}); setLogText("");
    setReportUrl(null); setResultFiles([]);

    // Config an Backend senden und Analyse starten
    try {
      const yaml = buildYaml(cfg, sp, spFmt);
      const res = await fetch("./api/run-analysis", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({yaml})
      });
      const startData = await res.json();
      if (startData.status === "error") {
        if(mountedRef.current) { setError(startData.message || "Fehler beim Starten"); setRunning(false); }
        return;
      }
      // Polling starten (wiederverwendbare Funktion)
      _startPolling();
      return;
    } catch(e) {
      if(mountedRef.current) {
        setError("Backend nicht erreichbar: " + (e.message || "Netzwerkfehler") + ". Prüfe die Verbindung in der Sidebar.");
        setRunning(false);
      }
    }
  };

  const resetAll = () => {
    timeoutsRef.current.forEach(t => clearTimeout(t));
    intervalsRef.current.forEach(t => clearInterval(t));
    timeoutsRef.current = []; intervalsRef.current = [];
    stopPolling();
    setDone(false); setCompleted(new Set()); setElapsed(0);
    setShowReport(false); setStep(null); setError(null);
    setRunning(false); setStepProgress(0); setStepDurations({});
    setReportUrl(null); setResultFiles([]);
    // Backend-State auch zurücksetzen
    fetch("./api/reset", {method:"POST"}).catch(()=>{});
  };

  const fmtTime = s => String(Math.floor(s/60)) + ":" + String(s%60).padStart(2,"0");

  return (
    <>
      <Sec title={done ? "Analyse abgeschlossen" : running ? "Analyse läuft …" : "Analyse starten"} accent={done ? "#059669" : running ? "#D4A853" : undefined}>
        {!running && !done && issues.length > 0 ? issues.map((issue,x) => <Info key={x} type="warn">{issue}</Info>) : null}
        {!running && !done && issues.length === 0 && (
          <>
            <Info type="success">Konfiguration ist valide – bereit zur Analyse.</Info>
            <div style={{height:12}}/>
            <Row gap={10}>
              <MC value={(cfg.models||[]).length} label="Modelle"/>
              <MC value={(cfg.baseLearner||"catboost")==="both"?"Both":cfg.baseLearner==="lgbm"?"LGBM":"CB"} label="Learner"/>
              <MC value={cfg.tuningEnabled?(cfg.tuningTrials||50)+"T":"Aus"} label="BLT"/>
              <MC value={cfg.fmtEnabled?(cfg.fmtTrials||50)+"T":"Aus"} label="FMT"/>
              <MC value={cfg.cfTune?(cfg.cfTrials||50)+"T":"Aus"} label="CFT"/>
              <MC value={cfg.validateOn==="external"?"Ext.":cfg.eval_mask_file?"TMES":"CV-"+(cfg.cvSplits||5)} label="Validierung"/>
            </Row>
          </>
        )}
        {(running || done || error) && (
          <>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
              <span style={{fontSize:14,fontWeight:700,color:"#6B0D15"}}>Pipeline-Fortschritt</span>
              <span style={{fontSize:12,fontFamily:MONO,color:"#888",background:"#faf6f6",padding:"3px 10px",borderRadius:6}}>{fmtTime(elapsed)}</span>
            </div>
            <ProgressTracker currentStep={step} error={error} steps={STEPS} enabled={enabledRef.current} completedSteps={completed} stepProgress={stepProgress} stepDurations={stepDurations} totalElapsed={elapsed}/>
            {disconnected && running && <div style={{fontSize:11.5,color:"#d97706",background:"#fffbeb",padding:"8px 14px",borderRadius:8,border:"1px solid #fbbf24",marginTop:6,display:"flex",alignItems:"center",gap:8}}><div style={{width:6,height:6,borderRadius:"50%",background:"#d97706",boxShadow:"0 0 6px rgba(217,119,6,0.3)",flexShrink:0}}/>Verbindung unterbrochen — versuche Reconnect. Die Analyse läuft im Hintergrund weiter.</div>}
            {error && !running && <Info type="error"><strong>Analyse fehlgeschlagen:</strong> <span style={{whiteSpace:"pre-wrap",fontFamily:error.includes("Details:")?MONO:"inherit",fontSize:error.includes("Details:")?11.5:"inherit"}}>{error}</span></Info>}
            {(running || (done && logText)) && logText && (
              <div style={{marginTop:14}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
                  <span style={{fontSize:12,fontWeight:600,color:"#57606a",letterSpacing:0.3,textTransform:"uppercase"}}>Pipeline-Logs</span>
                  <div style={{display:"flex",gap:8,alignItems:"center"}}>
                    <span style={{fontSize:10,color:"#999",fontFamily:MONO}}>{(logText.split("\n").length)} Zeilen</span>
                    <span onClick={()=>setShowLogs(p=>!p)} style={{fontSize:11,color:"#9B111E",cursor:"pointer",fontWeight:600}}>{showLogs?"Ausblenden":"Einblenden"}</span>
                  </div>
                </div>
                {showLogs && (
                  <div style={{background:"#1e1e2e",borderRadius:10,padding:"16px 20px",fontFamily:MONO,fontSize:11,whiteSpace:"pre-wrap",wordBreak:"break-all",height:280,overflowY:"auto",lineHeight:1.65,color:"#cdd6f4",boxShadow:"inset 0 2px 8px rgba(0,0,0,0.15)",border:"1px solid #313244"}} ref={el=>{if(el&&running){const atBottom=el.scrollHeight-el.scrollTop-el.clientHeight<40;if(atBottom)el.scrollTop=el.scrollHeight}}}>{logText}</div>
                )}
              </div>
            )}
          </>
        )}
        {!done && (
          <>
            <Divider/>
            <div style={{display:"flex",gap:10}}>
              <Btn disabled={issues.length > 0 || running} onClick={startAnalysis}>{running ? "Wird ausgeführt ..." : error ? "Erneut starten" : "Analyse starten"}</Btn>
              <Btn secondary onClick={()=>setPg("preview")}>Config-Vorschau</Btn>
            </div>
          </>
        )}
      </Sec>

      {done && showReport && (
        <Sec title="Analyse-Report" accent="#059669">
          <Info type="success"><strong>Analyse abgeschlossen</strong> in {fmtTime(elapsed)}.</Info>
          <div style={{border:"1px solid #ede6e7",borderRadius:10,overflow:"hidden",marginTop:12}}>
            {reportUrl ? (
              <iframe src={reportUrl} style={{width:"100%",height:"80vh",minHeight:700,border:"none"}} title="Analyse-Report"/>
            ) : (
              <div style={{padding:"40px 20px",textAlign:"center",color:"#888",fontSize:14}}>
                
                <div style={{fontWeight:600,color:"#57606a",marginBottom:6}}>Report wird geladen ...</div>
                <div>Falls der Report nicht erscheint: <strong onClick={()=>window.open("./api/download/output/analysis_report.html","_blank")} style={{color:"#9B111E",cursor:"pointer",textDecoration:"underline"}}>Direkt herunterladen</strong></div>
              </div>
            )}
          </div>
          <div style={{marginTop:12,display:"flex",gap:10}}>
            <Btn small onClick={()=>setShowReport(false)}>Report ausblenden</Btn>
            <Btn small secondary onClick={()=>window.open(reportUrl || "./api/download/output/analysis_report.html","_blank")}>Report herunterladen</Btn>
          </div>
        </Sec>
      )}

      {done && !error && (
        <Sec title="Ergebnisse herunterladen">
          {resultFiles.length > 0 ? (
            <>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
                {resultFiles.map(f => (
                  <FileCard key={f.name} name={f.name} desc={f.desc} onClick={()=>window.open("./api/download/" + (f.path||("output/"+f.name)),"_blank")}/>
                ))}
              </div>
              <div style={{marginTop:14,display:"flex",gap:10}}>
                <Btn onClick={()=>window.open("./api/results","_blank")}>Alle Ergebnisse als ZIP</Btn>
                <Btn secondary onClick={resetAll}>Neue Analyse</Btn>
              </div>
            </>
          ) : (
            <>
              <Info type="warn">Ergebnis-Dateien werden vom Backend geladen. Falls nichts erscheint, prüfe den Output-Ordner manuell oder lade die Seite neu.</Info>
              <div style={{marginTop:14,display:"flex",gap:10}}>
                <Btn small onClick={async ()=>{try{const r=await fetch("./api/results");const d=await r.json();if(d.files)setResultFiles(d.files)}catch(e){}}}>Ergebnisse neu laden</Btn>
                <Btn small secondary onClick={()=>window.open("./api/results","_blank")}>Ergebnisse direkt öffnen</Btn>
                <Btn secondary onClick={resetAll}>Neue Analyse</Btn>
              </div>
            </>
          )}
        </Sec>
      )}

      {!running && !done && (
        <Info>Config herunterladen und manuell starten: <code style={{background:"rgba(255,255,255,0.5)",padding:"1px 5px",borderRadius:3}}>pixi run analyze -- --config config.yml</code> (oder: <code style={{background:"rgba(255,255,255,0.5)",padding:"1px 5px",borderRadius:3}}>python run_analysis.py --config config.yml</code>)</Info>
      )}
    </>
  );
};

// ── Main ──// ── Main ──