
const FixedParamsEditor = ({params,defaults,onChange,label,help,accent}) => {
  const p = params || {};
  const d = defaults || {};
  const keys = Object.keys(d);
  const set = (k,v) => onChange({...p,[k]:v});
  // Draft-State pro Feld: erlaubt Zwischenzustände beim Tippen ("0," / "0.").
  const [drafts,setDrafts] = useState({});
  // Locale-tolerantes Parsen: Dezimal-KOMMA und -PUNKT sind gleichwertig ("0,002" == "0.002").
  // Rückgabe: null = leeres Feld (-> Default), undefined = (noch) ungültig (-> nicht committen), sonst Zahl.
  const parseNum = raw => {let t=String(raw).trim();if(t==="")return null;const c=(t.match(/,/g)||[]).length;if(c>0){if(c>1||t.includes("."))return undefined;t=t.replace(",",".");}const n=Number(t);return Number.isFinite(n)?n:undefined;};
  const isDefault = k => p[k] === undefined || p[k] === d[k];
  const modified = keys.filter(k=>!isDefault(k)).length;
  const ac = accent || C.ruby;
  const acBg = accent ? accent+"12" : C.rose;
  const acBorder = accent ? accent+"40" : "#e0b5b9";
  return (
    <div style={{background:"#faf8f8",borderRadius:10,border:`1px solid ${C.borderLight}`,padding:"16px 20px",marginTop:8}}>
      {label && <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:6}}>
        <span style={{fontSize:13,fontWeight:600,color:C.dark}}>{label}</span>
        {modified>0 && <span style={{fontSize:10,background:acBg,color:ac,padding:"2px 10px",borderRadius:10,fontWeight:600}}>{modified} angepasst</span>}
      </div>}
      {help && <div style={{fontSize:12,color:C.textMuted,marginBottom:10,lineHeight:1.5}}>{help}</div>}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"6px 12px"}}>
        {keys.map(k => {
          const val = p[k] !== undefined ? p[k] : d[k];
          const def = d[k];
          const isNum = typeof def === "number";
          const isBool = typeof def === "boolean";
          const changed = !isDefault(k);
          return (
            <div key={k} style={{display:"flex",alignItems:"center",gap:8,padding:"6px 12px",background:changed?acBg:"#fff",borderRadius:R,border:`1px solid ${changed?acBorder:C.borderLight}`,transition:"all 0.15s"}}>
              <code style={{fontSize:11,fontWeight:600,color:changed?ac:C.textSec,minWidth:120,fontFamily:MONO}}>{k}</code>
              <div style={{flex:1,display:"flex",alignItems:"center",justifyContent:"flex-end",gap:6}}>
                {isBool ? (
                  <div onClick={()=>set(k,!val)} style={{width:34,height:18,borderRadius:9,background:val?C.ruby:"#ccc",position:"relative",cursor:"pointer",flexShrink:0,transition:"background 0.2s"}}>
                    <div style={{width:12,height:12,borderRadius:6,background:"#fff",position:"absolute",top:3,left:val?19:3,transition:"left 0.2s",boxShadow:"0 1px 3px rgba(0,0,0,0.15)"}}/>
                  </div>
                ) : (
                  <input type="text" inputMode={isNum?"decimal":undefined} value={drafts[k]!==undefined?drafts[k]:(val===null?"":String(val))} onChange={e=>{const raw=e.target.value;setDrafts(dr=>({...dr,[k]:raw}));if(isNum){const n=parseNum(raw);if(n===null){set(k,def);}else if(n!==undefined){set(k,n);}}else{set(k,raw===""?def:raw);}}} style={{width:80,height:30,padding:"0 8px",border:`1.5px solid ${changed?acBorder:C.border}`,borderRadius:6,fontSize:12,fontFamily:MONO,fontWeight:600,background:"#fff",outline:"none",boxSizing:"border-box",textAlign:"right",color:changed?ac:C.text}} onFocus={e=>{e.target.style.borderColor=ac;e.target.style.boxShadow=`0 0 0 2px ${ac}14`}} onBlur={e=>{setDrafts(dr=>{const c={...dr};delete c[k];return c});e.target.style.borderColor=changed?acBorder:C.border;e.target.style.boxShadow="none"}}/>
                )}
                {changed && <span style={{fontSize:11,color:C.light,cursor:"pointer",opacity:0.7,lineHeight:1}} onClick={()=>{setDrafts(dr=>{const c={...dr};delete c[k];return c});const next={...p};delete next[k];onChange(next)}} title="Auf Default zurücksetzen">↺</span>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

const STEPS=[{id:"load",label:"Daten laden & Preprocessing"},{id:"fs",label:"Feature-Selektion"},{id:"tune",label:"Base-Learner-Tuning"},{id:"fmt",label:"Final-Model-Tuning"},{id:"grf",label:"CausalForest-Tuning"},{id:"train",label:"Training & Predictions"},{id:"eval",label:"Evaluation & Metriken"},{id:"surrogate",label:"Surrogate-Tree"},{id:"bundle",label:"Bundle-Export"},{id:"explain",label:"Explainability"},{id:"report",label:"HTML-Report"}];

const Sec = ({title,children,accent,sub}) => (
  <div style={{background:C.card,borderRadius:14,padding:sub?"20px 24px":"26px 30px",marginBottom:sub?14:22,boxShadow:sub?"none":"0 1px 4px rgba(155,17,30,0.05),0 8px 30px rgba(155,17,30,0.03)",borderTop:sub?"none":`3px solid ${accent||C.ruby}`,border:sub?`1px solid ${C.borderLight}`:"none"}}>
    <h3 style={{color:C.dark,fontSize:sub?15:17,fontWeight:700,marginBottom:sub?10:14,paddingBottom:sub?8:10,borderBottom:`1px solid ${C.borderLight}`,marginTop:0,letterSpacing:0.1}}>{title}</h3>
    {children}
  </div>
);
const Info = ({children,type}) => {
  const m = {warn:{bg:"#fffbeb",border:C.gold,text:"#7a5a00"},success:{bg:"#e8f5ec",border:C.green,text:C.green},error:{bg:"#fff0f0",border:"#cf222e",text:"#cf222e"}};
  const s = m[type]||{bg:C.rose,border:C.light,text:C.textSec};
  return <div style={{background:s.bg,borderLeft:`4px solid ${s.border}`,padding:"11px 18px",borderRadius:`0 ${R}px ${R}px 0`,margin:"12px 0",fontSize:13.5,color:s.text,lineHeight:1.65}}>{children}</div>;
};
const MC = ({value,label,desc,highlight,accent}) => {
  const v = String(value);
  const isLong = v.length > 4;
  return (
    <div style={{background:highlight?"linear-gradient(135deg,#6B0D15,#9B111E)":"#fffbeb",borderRadius:12,padding:"16px 14px 14px",textAlign:"center",flex:1,minWidth:0,transition:"transform 0.2s, box-shadow 0.2s",cursor:"default",borderLeft:accent?`3px solid ${accent}`:"none"}} onMouseEnter={e=>{e.currentTarget.style.transform="translateY(-3px)";e.currentTarget.style.boxShadow="0 8px 20px rgba(212,168,83,0.15)"}} onMouseLeave={e=>{e.currentTarget.style.transform="none";e.currentTarget.style.boxShadow="none"}}>
      <div style={{fontSize:9.5,fontWeight:600,textTransform:"uppercase",letterSpacing:0.8,color:highlight?"rgba(255,255,255,0.65)":C.textSec,marginBottom:5}}>{label}</div>
      <div style={{fontSize:isLong?16:26,fontWeight:800,color:highlight?"#fff":C.dark,letterSpacing:isLong?0:-0.5,whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis",lineHeight:1.2}}>{value}</div>
      {desc && <div style={{fontSize:10,color:highlight?"rgba(255,255,255,0.4)":C.textFaint,marginTop:4}}>{desc}</div>}
    </div>
  );
};
const Inp = ({label,placeholder,value,onChange,help,type="text",icon}) => (
  <div style={{marginBottom:16}}>
    {label && <label style={{fontSize:12,fontWeight:600,color:C.text,display:"block",marginBottom:5,letterSpacing:0.1}}>{label}</label>}
    <div style={{position:"relative"}}>
      {icon && <span style={{position:"absolute",left:11,top:"50%",transform:"translateY(-50%)",color:C.ruby,fontSize:13,fontWeight:700}}>{icon}</span>}
      <input type={type} value={value===0?0:(value||"")} onChange={e=>onChange?.(type==="number"?Number(e.target.value):e.target.value)} placeholder={placeholder} style={{width:"100%",height:H,padding:icon?"0 12px 0 32px":"0 12px",border:`1.5px solid ${C.border}`,borderRadius:R,fontSize:13.5,background:"#fff",outline:"none",transition:"border-color 0.2s, box-shadow 0.2s",boxSizing:"border-box"}} onFocus={e=>{e.target.style.borderColor=C.ruby;e.target.style.boxShadow="0 0 0 3px rgba(155,17,30,0.08)"}} onBlur={e=>{e.target.style.borderColor=C.border;e.target.style.boxShadow="none"}}/>
    </div>
    {help && <div style={{fontSize:10.5,color:C.textMuted,marginTop:4,lineHeight:1.4}}>{help}</div>}
  </div>
);
const Sel = ({label,options,value,onChange,help}) => (
  <div style={{marginBottom:16}}>
    {label && <label style={{fontSize:12,fontWeight:600,color:C.text,display:"block",marginBottom:5,letterSpacing:0.1}}>{label}</label>}
    <select value={value} onChange={e=>onChange?.(e.target.value)} style={{width:"100%",height:H,padding:"0 12px",border:`1.5px solid ${C.border}`,borderRadius:R,fontSize:13.5,background:"#fff",cursor:"pointer",outline:"none",boxSizing:"border-box",appearance:"auto"}}>{options.map(o=><option key={o}>{o}</option>)}</select>
    {help && <div style={{fontSize:10.5,color:C.textMuted,marginTop:4,lineHeight:1.4}}>{help}</div>}
  </div>
);
const Toggle = ({label,checked,onChange,help}) => (
  <div style={{marginBottom:12}}>
    <label style={{fontSize:13.5,color:C.text,cursor:"pointer",display:"flex",alignItems:"center",gap:10,minHeight:H}}>
      <div onClick={e=>{e.preventDefault();onChange?.(!checked)}} style={{width:40,height:22,borderRadius:11,background:checked?C.ruby:"#d0d0d0",position:"relative",transition:"background 0.2s",cursor:"pointer",flexShrink:0,boxShadow:"inset 0 1px 3px rgba(0,0,0,0.1)"}}>
        <div style={{width:16,height:16,borderRadius:8,background:"#fff",position:"absolute",top:3,left:checked?21:3,transition:"left 0.2s",boxShadow:"0 1px 4px rgba(0,0,0,0.18)"}}/>
      </div>
      <span style={{fontWeight:checked?600:400,lineHeight:1.3}}>{label}</span>
    </label>
    {help && <div style={{fontSize:10.5,color:C.textMuted,marginTop:2,marginLeft:50,lineHeight:1.4}}>{help}</div>}
  </div>
);
const Sld = ({label,min,max,step,value,onChange}) => (
  <div style={{marginBottom:16}}>
    <label style={{fontSize:12,fontWeight:600,color:C.text,display:"flex",justifyContent:"space-between",marginBottom:8}}>
      <span>{label}</span>
      <span style={{color:C.ruby,fontFamily:MONO,fontSize:12.5,background:C.rose,padding:"1px 8px",borderRadius:5,fontWeight:700}}>{value}</span>
    </label>
    <input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange?.(Number(e.target.value))} style={{width:"100%",accentColor:C.ruby}}/>
  </div>
);
const RSlider = ({label,min,max,step,low,high,onLow,onHigh,type="float",accent}) => {
  const ac = accent || "#9B111E";
  const acLight = accent ? accent+"18" : "#C4343F";
  const uid = (ac.slice(1)+label).replace(/[^a-z0-9]/gi,"_");
  const f = v => type==="int"?Math.round(v):v.toFixed(step<0.01?3:step<0.1?2:1);
  const pctLow = ((low-min)/(max-min))*100;
  const pctHigh = ((high-min)/(max-min))*100;
  return (
    <div style={{marginBottom:8,padding:"9px 14px",background:"#faf7f7",borderRadius:R,border:`1px solid ${C.borderLight}`}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
        <span style={{fontSize:11.5,fontWeight:600,color:"#444"}}>{label}</span>
        <span style={{fontSize:11,fontFamily:MONO,color:ac,background:"#fff",padding:"1px 8px",borderRadius:4,border:`1px solid ${C.borderLight}`,fontWeight:600}}>{f(low)} – {f(high)}</span>
      </div>
      <div style={{display:"flex",gap:6,alignItems:"center"}}>
        <span style={{fontSize:9.5,color:C.textFaint,minWidth:30}}>{f(min)}</span>
        <div style={{flex:1,position:"relative",height:22}}>
          <div style={{position:"absolute",top:9,left:0,right:0,height:4,borderRadius:2,background:"#ede6e7"}}/>
          <div style={{position:"absolute",top:9,left:`${pctLow}%`,right:`${100-pctHigh}%`,height:4,borderRadius:2,background:ac}}/>
          <input type="range" min={min} max={max} step={step} value={low} onChange={e=>{const v=Number(e.target.value);if(v<=high)onLow(v)}} style={{position:"absolute",width:"100%",top:2,pointerEvents:"none",WebkitAppearance:"none",MozAppearance:"none",appearance:"none",background:"transparent",height:18,margin:0,zIndex:3}} className={"rs_"+uid+"_l"}/>
          <input type="range" min={min} max={max} step={step} value={high} onChange={e=>{const v=Number(e.target.value);if(v>=low)onHigh(v)}} style={{position:"absolute",width:"100%",top:2,pointerEvents:"none",WebkitAppearance:"none",MozAppearance:"none",appearance:"none",background:"transparent",height:18,margin:0,zIndex:4}} className={"rs_"+uid+"_h"}/>
        </div>
        <span style={{fontSize:9.5,color:C.textFaint,minWidth:30,textAlign:"right"}}>{f(max)}</span>
      </div>
      <style>{`.rs_${uid}_l::-webkit-slider-thumb,.rs_${uid}_h::-webkit-slider-thumb{-webkit-appearance:none;pointer-events:auto;width:14px;height:14px;border-radius:7px;background:${ac};border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,0.2);cursor:pointer}
.rs_${uid}_l::-moz-range-thumb,.rs_${uid}_h::-moz-range-thumb{pointer-events:auto;width:10px;height:10px;border-radius:7px;background:${ac};border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,0.2);cursor:pointer}`}</style>
    </div>
  );
};
const Btn = ({children,onClick,disabled,secondary,small,full}) => (
  <button onClick={onClick} disabled={disabled} style={{
    background:disabled?"#e8e2e3":secondary?"#fff":"linear-gradient(135deg,#6B0D15,#9B111E)",
    color:disabled?"#aaa":secondary?C.dark:"#fff",
    border:secondary?`1.5px solid ${C.light}`:"1.5px solid transparent",
    borderRadius:R,height:small?34:H,padding:small?"0 18px":"0 28px",
    fontWeight:600,fontSize:small?12.5:14,fontFamily:FONT,
    cursor:disabled?"not-allowed":"pointer",transition:"all 0.2s",letterSpacing:0.2,
    display:"inline-flex",alignItems:"center",justifyContent:"center",
    width:full?"100%":undefined,whiteSpace:"nowrap",
  }} onMouseEnter={e=>{if(!disabled&&!secondary)e.currentTarget.style.boxShadow="0 6px 20px rgba(155,17,30,0.25)";if(!disabled&&secondary)e.currentTarget.style.background=C.rose}} onMouseLeave={e=>{e.currentTarget.style.boxShadow="none";if(secondary)e.currentTarget.style.background="#fff"}}>
    {children}
  </button>
);
const Tabs = ({tabs,active,onSelect}) => (
  <div style={{display:"flex",gap:2,background:"#f5f0f0",borderRadius:R,padding:3,marginBottom:16}}>
    {tabs.map(t=>(
      <button key={t} onClick={()=>onSelect(t)} style={{flex:1,padding:"8px 16px",border:"none",background:active===t?"#fff":"transparent",fontSize:12.5,fontWeight:600,cursor:"pointer",color:active===t?C.ruby:C.textMuted,borderRadius:6,transition:"all 0.15s",boxShadow:active===t?"0 1px 6px rgba(0,0,0,0.06)":"none",fontFamily:FONT}}>{t}</button>
    ))}
  </div>
);

const Expander = ({title,children,defaultOpen,accent}) => {
  const [open,setOpen] = useState(defaultOpen||false);
  return (
    <div style={{border:`1px solid ${C.borderLight}`,borderRadius:10,marginTop:16,overflow:"hidden",transition:"box-shadow 0.2s",boxShadow:open?"0 2px 8px rgba(0,0,0,0.03)":"none"}}>
      <button onClick={()=>setOpen(!open)} style={{display:"flex",alignItems:"center",justifyContent:"space-between",width:"100%",padding:"11px 18px",background:open?"#f8f5f5":"#fff",border:"none",cursor:"pointer",fontSize:12.5,fontWeight:600,color:open?C.dark:C.textSec,fontFamily:FONT,transition:"all 0.15s"}} onMouseEnter={e=>{if(!open)e.currentTarget.style.background="#faf8f8"}} onMouseLeave={e=>{if(!open)e.currentTarget.style.background="#fff"}}>
        <span style={{display:"flex",alignItems:"center",gap:7}}><span style={{fontSize:12,opacity:0.5,color:accent||undefined}}>⚙</span>{title}</span>
        <span style={{fontSize:10,color:C.textMuted,transform:open?"rotate(180deg)":"rotate(0deg)",transition:"transform 0.25s ease"}}>▾</span>
      </button>
      {open && <div style={{padding:"16px 20px",borderTop:`1px solid ${C.borderLight}`,background:"#faf8f8"}}>{children}</div>}
    </div>
  );
};
const Row = ({children,gap=16}) => <div style={{display:"flex",gap,flexWrap:"wrap",alignItems:"flex-start"}}>{children}</div>;
const Col = ({children,flex=1}) => <div style={{flex,minWidth:180}}>{children}</div>;
const Divider = () => <div style={{borderTop:`1px solid ${C.borderLight}`,margin:"18px 0"}}/>;
const FileCard = ({name,desc,onClick}) => (
  <button onClick={onClick} style={{display:"flex",alignItems:"center",gap:12,padding:"14px 18px",background:"#faf7f7",border:`1.5px solid ${C.border}`,borderRadius:R,cursor:"pointer",transition:"all 0.15s",textAlign:"left",width:"100%"}} onMouseEnter={e=>{e.currentTarget.style.borderColor=C.ruby;e.currentTarget.style.background=C.rose;e.currentTarget.style.transform="translateX(2px)"}} onMouseLeave={e=>{e.currentTarget.style.borderColor=C.border;e.currentTarget.style.background="#faf7f7";e.currentTarget.style.transform="none"}}>
    <div style={{width:36,height:36,borderRadius:8,background:C.rose,border:`1px solid ${C.borderLight}`,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0}}>
      <span style={{fontSize:14,color:C.ruby,fontWeight:700}}>{name.split(".").pop().substring(0,3).toUpperCase()}</span>
    </div>
    <div>
      <div style={{fontSize:13,fontWeight:600,color:C.dark}}>{name}</div>
      <div style={{fontSize:11,color:C.textMuted,marginTop:1}}>{desc}</div>
    </div>
  </button>
);

// ── Search Space ──
const SSEditor = ({bl,sp,setSp,fmt,accent}) => {
  const defs=bl==="catboost"?(fmt?CB_FMT:CB):(fmt?LGBM_FMT:LGBM);const cur=sp[bl]||{};
  const g=k=>({low:cur[k]?.low??(defs[k].dLow??defs[k].min),high:cur[k]?.high??(defs[k].dHigh??defs[k].max)});
  const s=(k,l,h)=>setSp(p=>({...p,[bl]:{...p[bl],[k]:{low:l,high:h}}}));
  return (
    <div>
      <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:12}}>
        <span style={{fontSize:13.5,fontWeight:700,color:accent||"#6B0D15"}}>{bl==="catboost"?"CatBoost":"LightGBM"} Suchraum</span>
        <span style={{fontSize:10.5,color:accent||"#888",opacity:0.7,background:(accent||"#9B111E")+"12",padding:"2px 10px",borderRadius:10,border:`1px solid ${(accent||"#9B111E")}30`}}>Optuna low – high</span>
      </div>
      {bl==="catboost" && <div style={{fontSize:10.5,color:"#6b7280",lineHeight:1.4,marginBottom:10,padding:"6px 10px",background:"#f9fafb",borderRadius:6,border:"1px solid #e5e7eb"}}>
        <strong style={{color:"#374151"}}>CPU-Optimierung:</strong> <code style={{fontSize:10,background:"#f3f4f6",padding:"1px 4px",borderRadius:3}}>max_ctr_complexity=1</code> ist fest gesetzt (keine automatischen kat. Feature-Kombinationen), um die Trainingszeit bei vielen kategorischen Features massiv zu reduzieren. Individuelle Ordered Target Statistics bleiben erhalten.
      </div>}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"0 14px"}}>
        {Object.entries(defs).map(([k,d])=>{const r=g(k);const lbl=d.log?`${k} (log)`:k;return (<RSlider key={k} label={lbl} min={d.min} max={d.max} step={d.step} low={r.low} high={r.high} type={d.type} onLow={v=>s(k,v,r.high)} onHigh={v=>s(k,r.low,v)} accent={accent}/>);})}
      </div>
    </div>
  );
};

// ── Progress Tracker ──
const _sidebarStyles = `@keyframes rubin-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } } @keyframes rubin-dot-breathe { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.75); } }`;

const ProgressTracker = ({currentStep,error,steps,enabled,completedSteps,stepProgress,stepDurations,totalElapsed}) => {
  const activeSteps = steps.filter(s=>enabled.has(s.id));
  const completedCount = activeSteps.filter(s=>completedSteps.has(s.id)).length;
  const totalCount = activeSteps.length;
  const allDone = completedCount === totalCount && totalCount > 0;
  const fmtDur = d => d >= 60 ? Math.floor(d/60)+"m "+Math.round(d%60)+"s" : d.toFixed(0)+"s";
  const activeIdx = activeSteps.findIndex(s => s.id === currentStep);

  return (
    <div style={{margin:"10px 0"}}>
      <style>{`
        @keyframes rubin-pulse-line {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
      `}</style>
      <div style={{position:"relative",paddingLeft:28}}>
        {/* Vertical timeline line */}
        <div style={{position:"absolute",left:9,top:4,bottom:4,width:2,background:"#eee",borderRadius:1}}/>
        {/* Completed segment of the line */}
        {activeIdx > 0 && <div style={{position:"absolute",left:9,top:4,height:`calc(${(activeIdx / Math.max(activeSteps.length-1, 1)) * 100}% - 4px)`,width:2,background:"linear-gradient(180deg, #059669, #059669)",borderRadius:1,transition:"height 0.6s ease"}}/>}
        {/* Active pulse segment */}
        {activeIdx >= 0 && !error && !allDone && <div style={{position:"absolute",left:9,top:`calc(${(activeIdx / Math.max(activeSteps.length-1, 1)) * 100}%)`,height:`calc(${(1 / Math.max(activeSteps.length-1, 1)) * 100}%)`,width:2,background:"linear-gradient(180deg, #059669, #9B111E)",borderRadius:1,animation:"rubin-pulse-line 2.4s ease-in-out infinite"}}/>}
        {/* All-done: full green line */}
        {allDone && <div style={{position:"absolute",left:9,top:4,bottom:4,width:2,background:"#059669",borderRadius:1,transition:"all 0.6s"}}/>}

        <div style={{display:"flex",flexDirection:"column",gap:0}}>
          {activeSteps.map((st, i) => {
            const done=completedSteps.has(st.id);
            const active=st.id===currentStep;
            const failed=active&&error;
            const dur = stepDurations[st.id];
            const pending = !done && !active;

            return (
              <div key={st.id} style={{position:"relative",display:"flex",alignItems:"center",gap:14,padding:"8px 0",transition:"opacity 0.4s",opacity:pending?0.4:1}}>
                {/* Node on timeline */}
                <div style={{position:"absolute",left:-28,top:"50%",transform:"translateY(-50%)",width:20,display:"flex",justifyContent:"center"}}>
                  {done ? (
                    <div style={{width:7,height:7,borderRadius:"50%",background:"#059669",boxShadow:"0 0 6px rgba(5,150,105,0.4)",transition:"all 0.3s"}}/>
                  ) : active&&!failed ? (
                    <div style={{width:7,height:7,borderRadius:"50%",background:"#9B111E",boxShadow:"0 0 8px rgba(155,17,30,0.4)",animation:"rubin-pulse-line 2.4s ease-in-out infinite",transition:"all 0.3s"}}/>
                  ) : failed ? (
                    <div style={{width:7,height:7,borderRadius:"50%",background:"#cf222e",boxShadow:"0 0 6px rgba(207,34,46,0.4)",transition:"all 0.3s"}}/>
                  ) : (
                    <div style={{width:7,height:7,borderRadius:"50%",background:"#d0d0d0",transition:"all 0.3s"}}/>
                  )}
                </div>
                {/* Content */}
                <div style={{flex:1,display:"flex",alignItems:"center",gap:10,minHeight:24}}>
                  <span style={{fontSize:13,fontWeight:active?600:done?500:400,color:failed?"#cf222e":active?"#24292f":done?"#4a5568":"#bbb",flex:1,transition:"color 0.3s"}}>{st.label}</span>
                  {done&&dur!=null&&<span style={{fontSize:10,fontFamily:MONO,color:"#b0b0b0",background:"#f7f7f7",padding:"2px 8px",borderRadius:10}}>{fmtDur(dur)}</span>}
                  {active&&!failed&&<span style={{fontSize:9.5,fontWeight:600,color:"#9B111E",letterSpacing:1,textTransform:"uppercase",background:"rgba(155,17,30,0.06)",padding:"2px 8px",borderRadius:10}}>aktiv</span>}
                  {failed&&<span style={{fontSize:9.5,fontWeight:600,color:"#cf222e",letterSpacing:1,textTransform:"uppercase",background:"rgba(207,34,46,0.06)",padding:"2px 8px",borderRadius:10}}>fehler</span>}
                </div>
              </div>
            );
          })}
        </div>
      </div>
      {/* Footer */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginTop:10,paddingTop:8,borderTop:"1px solid #f0eded"}}>
        <div style={{display:"flex",alignItems:"center",gap:6}}>
          {allDone ? (
            <div style={{width:6,height:6,borderRadius:"50%",background:"#059669",boxShadow:"0 0 6px rgba(5,150,105,0.4)"}}/>
          ) : (
            <div style={{width:6,height:6,borderRadius:"50%",background:"#9B111E",animation:currentStep?"rubin-pulse-line 2.4s ease-in-out infinite":"none"}}/>
          )}
          <span style={{fontSize:11.5,color:allDone?"#059669":"#666",fontWeight:500}}>{allDone?"Pipeline abgeschlossen":`${completedCount} von ${totalCount} Schritten`}</span>
        </div>
      </div>
    </div>
  );
};

// ── Column Role Picker ──

const DEFAULT_CFG = {studyType:"rct",expName:"rubin",seed:42,tuningSeed:18,models:["NonParamDML","DRLearner","SLearner","TLearner","XLearner","ParamDML","CausalForestDML","CausalForest"],baseLearner:"catboost",ensembleEnabled:true,tuningEnabled:false,tuningTrials:50,tuningSingleFold:false,tuningModels:[],fmtEnabled:false,fmtModels:[],fmtSingleFold:false,fmtTrials:50,fmtMaxRows:0,fmtOverfitPenalty:0.0,fmtOverfitTolerance:0.1,fmtOverfitMaxGap:1.0,fmtScorer:"auto",cfTune:false,cfTrials:50,cfSingleFold:false,cfScorer:"auto",cfTuneModels:[],cfOverfitPenalty:0.0,cfOverfitTolerance:0.1,cfOverfitMaxGap:1.0,bundleEnabled:false,bundleDir:"runs/bundles",bundleMlflow:true,surrEnabled:false,surrMinLeaf:50,surrLeaves:31,surrDepth:0,fsEnabled:false,fsMethods:["catboost_importance"],fsCorrThresh:0.9,fsMaxFeatures:77,downsample:false,dfFrac:0.1,reduceMem:true,validateOn:"cross",cvSplits:5,treatmentType:"binary",refGroup:0,selMetric:"qini",higherBetter:true,manualChamp:null,hasNaN:false,nanCols:[],explEnabled:false,explSampleSize:10000,explTopN:20,shapModels:[],shapBins:10,histScoreName:"historical_score",histScoreCol:"S",histScoreHigher:true,outputDir:"",blFixed:{},fmtFixed:{},cfFixed:{},tuningTimeout:0,tuningMaxRows:0,fmtTimeout:0,maxPredRows:0,parallelLevel:3,workDir:null,mcIters:null,mcAgg:"mean",dmlCrossfitFolds:5,overfitPenalty:0.0,overfitTolerance:0.2,overfitMaxGap:1.0};

