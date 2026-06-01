const POverview = ({setPg}) => (
  <>
    <div style={{background:"linear-gradient(135deg,#6B0D15 0%,#9B111E 55%,#C4343F 100%)",borderRadius:16,padding:"48px 52px",color:"#fff",display:"flex",alignItems:"center",gap:32,marginBottom:26,boxShadow:"0 8px 32px rgba(107,13,21,0.2),inset 0 1px 0 rgba(255,255,255,0.06)",position:"relative",overflow:"hidden"}}>
      <GemAccent/><RubinLogo size={92} light/>
      <div style={{position:"relative"}}>
        <h1 style={{fontSize:34,fontWeight:700,margin:0,letterSpacing:0.5}}>rubin</h1>
        <p style={{fontSize:14.5,opacity:0.82,margin:"8px 0 0",lineHeight:1.6}}>Treatment-Effekte verstehen. Entscheidungen optimieren.</p>
      </div>
    </div>
    <div style={{display:"grid",gridTemplateColumns:"repeat(5, 1fr)",gap:10,marginBottom:26}}>
      {[
        {v:"8",l:"Kausale Modelle",d:"Meta-Learner, DML, DR, CausalForest",accent:"#9B111E"},
        {v:"2",l:"Base-Learner",d:"CatBoost & LightGBM (GBDT)",accent:"#C4343F"},
        {v:"6",l:"Uplift-Metriken",d:"Qini, AUUC, Uplift@k, Policy Value, TOC",accent:"#D4A853"},
        {v:"10",l:"Diagnose-Plots",d:"CATE, Calibration, Qini, TOC, Balance u.a.",accent:"#2d6a4f"},
        {v:"3×3",l:"Tuning-Stufen",d:"BLT, FMT, CFT — je Schnell, Standard, Intensiv",accent:"#6366f1"},
      ].map(c => (
        <div key={c.l} style={{background:"#fff",borderRadius:12,padding:"18px 14px",textAlign:"center",borderTop:`3px solid ${c.accent}`,boxShadow:"0 1px 4px rgba(0,0,0,0.04)",transition:"transform 0.2s",cursor:"default",display:"flex",flexDirection:"column",justifyContent:"flex-start"}} onMouseEnter={e=>{e.currentTarget.style.transform="translateY(-2px)"}} onMouseLeave={e=>{e.currentTarget.style.transform="none"}}>
          <div style={{fontSize:28,fontWeight:800,color:c.accent,letterSpacing:-0.5}}>{c.v}</div>
          <div style={{fontSize:10,fontWeight:600,textTransform:"uppercase",letterSpacing:0.8,color:C.textSec,marginTop:3}}>{c.l}</div>
          {c.d && <div style={{fontSize:10,color:C.textFaint,marginTop:4,minHeight:28}}>{c.d}</div>}
        </div>
      ))}
    </div>
    <Sec title="Analyse-Pipeline">
      <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
        {["Daten laden","Feature-Sel.","Tuning","Training","Evaluation","Surrogate","Bundle","Explain","Report"].map((s,i) => (
          <div key={s} style={{display:"flex",alignItems:"center",gap:4}}>
            <div style={{background:i<2?"#6B0D15":i<5?"#9B111E":i<7?"#C4343F":"#D4A853",color:"#fff",padding:"7px 14px",borderRadius:8,fontSize:11.5,fontWeight:500,whiteSpace:"nowrap",transition:"transform 0.15s"}} onMouseEnter={e=>{e.currentTarget.style.transform="translateY(-2px)"}} onMouseLeave={e=>{e.currentTarget.style.transform="none"}}>{s}</div>
            {i<8 && <span style={{color:"#ccc",fontSize:18,margin:"0 1px",lineHeight:1}}>›</span>}
          </div>
        ))}
      </div>
    </Sec>
    <Sec title="Einstieg">
      <Info><strong>Zwei Wege:</strong> Falls du Rohdaten aufbereiten musst (CSV/SAS → X/T/Y), starte mit der optionalen Datenvorbereitung. Falls du bereits vorbereitete Dateien hast, gehe direkt zu "Daten".</Info>
      <div style={{display:"flex",gap:10,marginTop:12}}>
        <Btn secondary onClick={()=>setPg("dataprep")}>Datenvorbereitung (optional)</Btn>
        <Btn onClick={()=>setPg("datafiles")}>Direkt zu Daten</Btn>
      </div>
    </Sec>
  </>
);

// ── DataPrep YAML Builder ──