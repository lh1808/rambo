// ════════════════════════════════════════════════════════════════════
// REGION: React Hooks + SVG Assets
// ════════════════════════════════════════════════════════════════════
const { useState, useEffect, useRef, useMemo } = React;

const RubinLogo = ({ size = 48, light = false }) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="50 -5 120 125" width={size} height={size}>
    <polygon points="110,0 148,16 162,56 148,96 110,112 72,96 58,56 72,16"
      fill={light?"rgba(255,255,255,0.08)":"rgba(155,17,30,0.08)"}
      stroke={light?"rgba(255,255,255,0.7)":"#9B111E"} strokeWidth="2"/>
    <circle cx="88" cy="56" r="5" fill={light?"rgba(255,255,255,0.65)":"#9B111E"}/>
    <line x1="93" y1="51" x2="136" y2="26" stroke={light?"rgba(255,255,255,0.85)":"#9B111E"} strokeWidth="2.2"/>
    <polygon points="139,22 134,29 144,29" fill={light?"rgba(255,255,255,0.85)":"#9B111E"}/>
    <line x1="93" y1="61" x2="136" y2="86" stroke={light?"rgba(255,255,255,0.4)":"#c4343f"} strokeWidth="1.6" strokeDasharray="5 3"/>
    <polygon points="139,90 134,83 144,83" fill={light?"rgba(255,255,255,0.4)":"#c4343f"}/>
  </svg>
);
const GemAccent = () => (
  <svg width="180" height="180" style={{position:"absolute",right:-20,top:-20,opacity:0.07}}>
    <polygon points="90,10 160,45 170,110 130,165 60,165 20,110 30,45" fill="#fff" stroke="#fff" strokeWidth="1.5">
      <animateTransform attributeName="transform" type="rotate" from="0 90 90" to="360 90 90" dur="80s" repeatCount="indefinite"/>
    </polygon>
  </svg>
);


// ════════════════════════════════════════════════════════════════════
// REGION: Navigation + Modell-Metadaten
// ════════════════════════════════════════════════════════════════════
const pages = [
  {label:"Übersicht",key:"overview"},
  {label:"Datenvorbereitung",key:"dataprep",optional:true,group:"Daten"},
  {label:"Daten",key:"datafiles",group:"Daten"},
  {label:"Experiment-Setup",key:"template",group:"Experiment"},
  {label:"Learner & Tuning",key:"models"},
  {label:"Pipeline-Einstellungen",key:"config"},
  {label:"Champion & Production",key:"selection"},
  {label:"Explainability",key:"explain"},
  {label:"Config-Vorschau",key:"preview",group:"Abschluss"},
  {label:"Analyse starten",key:"run"},
];
const allModels=["SLearner","TLearner","XLearner","DRLearner","NonParamDML","ParamDML","CausalForestDML","CausalForest"];
const btOnly=new Set(["SLearner","TLearner","XLearner","CausalForest","NonParamDML"]);
const LGBM_LOGO=`data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 160"><polygon points="0,0 100,0 0,65" fill="#E44A2C"/><polygon points="100,60 100,100 48,80" fill="#71B74A"/><polygon points="0,80 0,115 48,97" fill="#2F9FD1"/><polygon points="0,160 100,95 100,160" fill="#F3AE19"/></svg>')}`;
const CB_LOGO="data:image/png;base64,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";

// ════════════════════════════════════════════════════════════════════
// REGION: BLT Search Spaces (Nuisance-Modelle)
// ════════════════════════════════════════════════════════════════════
const LGBM = {
  n_estimators                  : {min:50,max:1000,step:50,type:"int",dLow:200,dHigh:600},
  learning_rate                 : {min:0.001,max:0.3,step:0.001,type:"float",log:true,dLow:0.01,dHigh:0.15},
  num_leaves                    : {min:7,max:255,step:1,type:"int",dLow:15,dHigh:127},
  max_depth                     : {min:2,max:12,step:1,type:"int",dLow:3,dHigh:8},
  min_child_samples             : {min:5,max:500,step:5,type:"int",dLow:10,dHigh:200},
  min_child_weight              : {min:0.001,max:100,step:0.001,type:"float",dLow:0.01,dHigh:50},
  subsample                     : {min:0.3,max:1,step:0.05,type:"float",dLow:0.5,dHigh:1},
  colsample_bytree              : {min:0.1,max:1,step:0.05,type:"float",dLow:0.3,dHigh:0.9},
  min_split_gain                : {min:0,max:3,step:0.05,type:"float",dLow:0,dHigh:1},
  reg_alpha                     : {min:0,max:50,step:0.1,type:"float",dLow:0,dHigh:20},
  reg_lambda                    : {min:0,max:50,step:0.1,type:"float",dLow:0,dHigh:20},
  max_bin                       : {min:7,max:255,step:1,type:"int",dLow:15,dHigh:127},
  path_smooth                   : {min:0,max:20,step:0.5,type:"float",dLow:0,dHigh:10},
};
const CB = {
  iterations                    : {min:50,max:1000,step:50,type:"int",dLow:200,dHigh:600},
  learning_rate                 : {min:0.001,max:0.3,step:0.001,type:"float",log:true,dLow:0.01,dHigh:0.15},
  depth                         : {min:2,max:12,step:1,type:"int",dLow:4,dHigh:8},
  l2_leaf_reg                   : {min:0.1,max:80,step:0.1,type:"float",dLow:1,dHigh:30},
  random_strength               : {min:0.001,max:20,step:0.001,type:"float",dLow:0.01,dHigh:10},
  subsample                     : {min:0.3,max:1,step:0.05,type:"float",dLow:0.5,dHigh:1},
  rsm                           : {min:0.1,max:1,step:0.05,type:"float",dLow:0.3,dHigh:0.9},
  min_data_in_leaf              : {min:5,max:500,step:5,type:"int",dLow:10,dHigh:200},
  model_size_reg                : {min:0,max:20,step:0.5,type:"float",dLow:0,dHigh:10},
  leaf_estimation_iterations    : {min:1,max:15,step:1,type:"int",dLow:1,dHigh:10},
};

// ════════════════════════════════════════════════════════════════════
// REGION: FMT Search Spaces (CATE-Effektmodell)
// ════════════════════════════════════════════════════════════════════
const LGBM_FMT = {
  n_estimators                  : {min:50,max:800,step:10,type:"int",dLow:100,dHigh:400},
  learning_rate                 : {min:0.001,max:0.2,step:0.001,type:"float",log:true,dLow:0.005,dHigh:0.12},
  num_leaves                    : {min:3,max:127,step:1,type:"int",dLow:7,dHigh:63},
  max_depth                     : {min:1,max:10,step:1,type:"int",dLow:2,dHigh:6},
  min_child_samples             : {min:5,max:500,step:5,type:"int",dLow:20,dHigh:200},
  min_child_weight              : {min:0.1,max:50,step:0.1,type:"float",dLow:0.5,dHigh:20},
  subsample                     : {min:0.2,max:1,step:0.05,type:"float",dLow:0.4,dHigh:0.85},
  colsample_bytree              : {min:0.1,max:1,step:0.05,type:"float",dLow:0.3,dHigh:0.7},
  max_bin                       : {min:7,max:255,step:1,type:"int",dLow:15,dHigh:127},
  min_split_gain                : {min:0,max:2,step:0.05,type:"float",dLow:0,dHigh:0.5},
  reg_alpha                     : {min:0,max:30,step:0.5,type:"float",dLow:0,dHigh:10},
  reg_lambda                    : {min:0,max:30,step:0.5,type:"float",dLow:0,dHigh:10},
  path_smooth                   : {min:0,max:15,step:0.5,type:"float",dLow:0,dHigh:5},
};
const CB_FMT = {
  iterations                    : {min:50,max:800,step:10,type:"int",dLow:100,dHigh:400},
  learning_rate                 : {min:0.001,max:0.2,step:0.001,type:"float",log:true,dLow:0.005,dHigh:0.12},
  depth                         : {min:1,max:10,step:1,type:"int",dLow:2,dHigh:6},
  l2_leaf_reg                   : {min:0.5,max:60,step:0.5,type:"float",dLow:3,dHigh:30},
  random_strength               : {min:0.1,max:20,step:0.1,type:"float",dLow:0.5,dHigh:10},
  subsample                     : {min:0.2,max:1,step:0.05,type:"float",dLow:0.4,dHigh:0.85},
  rsm                           : {min:0.1,max:1,step:0.05,type:"float",dLow:0.3,dHigh:0.7},
  min_data_in_leaf              : {min:5,max:500,step:5,type:"int",dLow:20,dHigh:200},
  model_size_reg                : {min:0,max:30,step:0.1,type:"float",dLow:0.1,dHigh:10},
  leaf_estimation_iterations    : {min:1,max:10,step:1,type:"int",dLow:1,dHigh:5},
};

// ════════════════════════════════════════════════════════════════════
// REGION: Fixed Params Defaults (wenn Tuning AUS)
// ════════════════════════════════════════════════════════════════════
// Default fixed params per base learner (used when tuning is OFF)
const LGBM_DEFAULTS = {
  boosting_type:"gbdt",
  n_estimators:500,
  learning_rate:0.05,
  num_leaves:63,
  max_depth:6,
  min_child_samples:20,
  min_child_weight:1.0,
  subsample:0.8,
  colsample_bytree:0.6,
  min_split_gain:0.0,
  reg_alpha:0.1,
  reg_lambda:1.0,
  path_smooth:0.0,
};
const CB_DEFAULTS = {
  iterations:500,
  learning_rate:0.05,
  depth:6,
  l2_leaf_reg:5.0,
  min_data_in_leaf:20,
  random_strength:1.0,
  subsample:0.8,
  rsm:0.6,
  model_size_reg:0.0,
  leaf_estimation_iterations:1,
};
// Final model (regressor) – konservative Defaults für CATE-Schätzung auf verrauschten Residuen
const LGBM_FINAL_DEFAULTS = {
  boosting_type:"gbdt",
  n_estimators:200,
  learning_rate:0.03,
  num_leaves:20,
  max_depth:5,
  min_child_samples:100,
  min_child_weight:5.0,
  subsample:0.7,
  colsample_bytree:0.5,
  min_split_gain:0.5,
  reg_alpha:3.0,
  reg_lambda:15.0,
  path_smooth:5.0,
};
const CB_FINAL_DEFAULTS = {
  iterations:200,
  learning_rate:0.03,
  depth:4,
  l2_leaf_reg:10.0,
  min_data_in_leaf:100,
  random_strength:3.0,
  subsample:0.7,
  rsm:0.5,
  model_size_reg:5.0,
  leaf_estimation_iterations:1,
};

// ════════════════════════════════════════════════════════════════════
// REGION: CausalForest Defaults + Design System
// ════════════════════════════════════════════════════════════════════
// CausalForestDML + CausalForest Forest-Defaults (identisch mit _CFDML_DEFAULTS / _CF_DEFAULTS in model_registry.py)
const CF_FOREST_DEFAULTS = {n_estimators:500,max_depth:null,min_weight_fraction_leaf:0.0,min_var_fraction_leaf:null,max_samples:0.45,min_samples_leaf:5,min_samples_split:10,max_features:"auto",min_impurity_decrease:0.0,criterion:"mse"};

// CF Search Space Ranges (für RSlider-Editor)
const CF_SS = {
  min_weight_fraction_leaf:{min:0.0001,max:0.1,step:0.0001,type:"float",log:true},
  min_var_fraction_leaf:   {min:0.0001,max:0.1,step:0.0001,type:"float",log:true},
};
// ── Design System ──
const H = 38; // standard control height
const R = 8;  // standard border radius
const FONT = "'DM Sans',system-ui,-apple-system,sans-serif";
const MONO = "'JetBrains Mono',monospace";
const C = {ruby:"#9B111E",dark:"#6B0D15",light:"#C4343F",rose:"#FDF2F3",gold:"#D4A853",green:"#059669",purple:"#6366f1",bg:"#fef8f8",card:"#fff",border:"#e0d6d7",borderLight:"#ede6e7",borderFocus:"#9B111E",text:"#24292f",textSec:"#57606a",textMuted:"#888",textFaint:"#bbb"};