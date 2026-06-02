
const ADDON_PRESETS = [
  // ── Tuning (Pipeline-Reihenfolge: BL → FMT → GRF) ──
  {key:"bl_tuning_schnell",label:"Schnell",desc:"3 Wellen",group:"tuning_blt",
    cfg:{tuningEnabled:true},waves:{field:"tuningTrials",w:3}},
  {key:"bl_tuning",label:"Standard",desc:"5 Wellen",group:"tuning_blt",
    cfg:{tuningEnabled:true},waves:{field:"tuningTrials",w:5}},
  {key:"bl_tuning_intensiv",label:"Intensiv",desc:"8 Wellen",group:"tuning_blt",
    cfg:{tuningEnabled:true},waves:{field:"tuningTrials",w:8}},
  {key:"fmt_schnell",label:"Schnell",desc:"30 Trials",group:"tuning_fmt",
    cfg:{fmtEnabled:true,fmtModels:["NonParamDML","DRLearner"],fmtTrials:30}},
  {key:"fmt",label:"Standard",desc:"50 Trials",group:"tuning_fmt",
    cfg:{fmtEnabled:true,fmtModels:["NonParamDML","DRLearner"],fmtTrials:50}},
  {key:"fmt_intensiv",label:"Intensiv",desc:"100 Trials",group:"tuning_fmt",
    cfg:{fmtEnabled:true,fmtModels:["NonParamDML","DRLearner"],fmtTrials:100}},
  {key:"grf_tuning_schnell",label:"Schnell",desc:"30 Trials",group:"tuning_cft",
    cfg:{cfTune:true,cfTrials:30,cfTuneModels:["CausalForestDML","CausalForest"]}},
  {key:"grf_tuning",label:"Standard",desc:"50 Trials",group:"tuning_cft",
    cfg:{cfTune:true,cfTrials:50,cfTuneModels:["CausalForestDML","CausalForest"]}},
  {key:"grf_tuning_intensiv",label:"Intensiv",desc:"100 Trials",group:"tuning_cft",
    cfg:{cfTune:true,cfTrials:100,cfTuneModels:["CausalForestDML","CausalForest"]}},
  // ── Production & Export ──
  {key:"bundle",label:"Bundle & Surrogate",desc:"Production-Export mit Surrogate-Einzelbaum",group:"production",
    cfg:{bundleEnabled:true,refitChamp:true,surrEnabled:true}},
  // ── Feature-Selektion ──
  {key:"feature_reduction",label:"Feature-Reduktion",desc:"Max. 77 Features via Importance + Korrelationsfilter",group:"feature",
    cfg:{fsEnabled:true,fsMethods:["catboost_importance","causal_forest"],fsCorrThresh:0.9,fsMaxFeatures:77}},
  // ── Erklärbarkeit ──
  {key:"explainability",label:"Explainability",desc:"SHAP-Analyse des Champions",group:"interpret",
    cfg:{explEnabled:true,explSampleSize:10000,explTopN:20}},
  // ── Tuning-Regularisierung ──
  {key:"reg_moderate",label:"Moderat",desc:"BLT: 15% Gap toleriert (Nuisance darf moderat overfitten). FMT: 5% Gap (CATE-Signal ist schwächer).",group:"regularization",
    cfg:{overfitPenalty:0.2,overfitTolerance:0.15,fmtOverfitPenalty:0.3,fmtOverfitTolerance:0.05,cfOverfitPenalty:0.3,cfOverfitTolerance:0.05}},
  {key:"reg_strong",label:"Stark",desc:"BLT: 8% Gap toleriert. FMT: 3% Gap. Konservativ — bei kleinen Datensätzen oder instabilen Vorläufen.",group:"regularization",
    cfg:{overfitPenalty:0.4,overfitTolerance:0.08,fmtOverfitPenalty:0.6,fmtOverfitTolerance:0.03,cfOverfitPenalty:0.6,cfOverfitTolerance:0.03}},
  // ── Stabilität ──
  {key:"mc_iters",label:"MC-Iterationen",desc:"Cross-Fitting 3x wiederholen – stabilere Residuals",group:"stability",
    cfg:{mcIters:3}},
  // ── Exploration ──
  {key:"exploration",label:"10%-Sampling",desc:"10% Sampling + 3 CV-Folds – schnelles Iterieren",group:"exploration",
    cfg:{downsample:true,dfFrac:0.1,cvSplits:3}},
  // ── Performance ──
  {key:"speed",label:"Speed",desc:"Single-Fold-Tuning – bis zu 5× schneller",group:"performance",
    cfg:{tuningSingleFold:true,fmtSingleFold:true,cfSingleFold:true}},
];

// ── Pages ──