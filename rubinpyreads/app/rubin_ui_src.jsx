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
const CB_LOGO="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAwCAYAAAC8NUKEAAAOtklEQVR4nM2aeZBnVXXHP+fc+35bd9Pd0+MoMMgiI6aMEEUEjEFUEkNpoYhQhqDlFiyVQiELpaXpRseUYGncSgqUWCqW4ogmgnHDDRWEMmBc4ig4AuMwwGy9/rZ37zn54/1mpmemZ2wZsTxV9/d7v/fuu/d877lnvT8A4U+LBI6tw8RJsLq5+96yXgT3SYUph6k/AWBTDrBmYmLknu3bF4C87FcLDj0RGAaGgPCo8PdHIoHDJj551dF/97RjF15YTlsPcf1jM2FBKQG1AneXolXnC5/fft3aj6//rNukilxuyxlHAL5/wwmf/svnbDyfB9tQc/BHlfd9ySOog0VICiuc7900On3aa+9/nDt9keVxFAHRti7Y1jIz300uxF1Pd2qQD6590X3fv1Y6++m/+Pfix95DB/fMBO14fuapOvaONxz+HpFNF7/whYe2brppc/t3gVHARU015EAgBFnUGDRZ9L3oWhk0WdQO1F8HbdB353NVAlI1DR7IFEHm8pnPq7/xhCNH/+LEE5/Yn5zkd27/g9MPHbSdIpIDj2gCWatvl0H/pcSriM0nefpTtuk/vLz5nssv/26amto1wwHZ+QOQ4C74zu14gH7L1UcBpTOTX/Q8O+NvThr6e1XMJx9FMI5gKOYRlwKXiO9vSAdJQsgBSYIkDghMFKyjsvrxs37BWUNXuHMIUxxwuQ4CjIA7ng0sgWU8JSzbvtM5UAjSbIAoUkQYrkGEvB9AjpCLoMx27aV/Gw9/3dmPu1AEn5zcvy+M+3uwB1nAg2HiqIG4YmK04wqK8VcSR09EQoTuPP3tNxBmvk5NM2QlxUysFaSfjtL5Upe8MaEo4Ukt6i8aJR69Hbq5WlZTLIC6YQiiGSuR5vBWu+DssTdd/cUjP7z2nfd3Bxq3zzIsD4xY9aZHxA3D6IYn0Dz6A4RDXoAPRg4taI6fR/ehd9N54Ao0GrWiRucbDaav2M7Yw0YBJDJy6xzT32oz/M5hGsd36Oc+hQkmlctRCwgOYsp83049Ka9+20Xdd639sF/mjokMBLiIlrXNPDguiqYG4HQ0EFdfjB7yArIvIKmP9g1nnlKHKA67hDj8fESNdN8Ic+9ps+rhgkYhxJpQqwmNokHrXmPhXV3YOkKqg0WnsiJUQKwyd30VCTqTzz6tf+mTDm0drwE7dwnelwXGBp8ifdwdbzyZYuxluDlKDQ8Rj4JogVrGmSCufAU1O4T2153WgwkpIGeFUij6Ac996gW01pd0buvS1AJJTrABBs2gCTzgUSXPJJ52QtcveuXQWjce+4ZvP3sfQ7AsMGJarZRm3EGLIyGsQkiIB0ycpIYTCTgJpz98FOhKauvnqAsYPTKB4E4Oib46mhUXY+E+QVIAc8QFV3CpQgZXIyZFTAMLfT/rjHjmaU9unnP++b+qXXghxe8PBh1YqMrbuc4AvcoUa6XQ0QOC4CrUMgQrsdoMVlecgBNAEoaCBxTHxVCHKAVlEckDDXZADNwUEyVYQgL03Dhi9bRd+rqhV2/evHnkttseP7yLxb3BqEWMgA+8unsNrIHkiFoEK8iFEhY24flndMQxdyQnzI2SPo4jKsj8r1DfBk8NlK4gBbjiIog7NYuYKqUqQ3/u4CUSBLxJ8IIUKh7EB0GdOCqutDt+xil+4kVnP+acn/30/h2Tk8QB/r0lY1CxAy6IZAglHrv0+0qPTNZMbeHXyG8/TxNFvE4WI4kjVkMpwH5B2nwtZaeg/nyj/ZRI2U/EmCFmPDopCL0y48+uU5xUInM9cj9QZodUEi0iDkquQOEUBpTI0PC8n/OS1r84jE5NYUtKxoKhOJJrYAHI9DUzW1+DHTmJrPkQeug7yI3j6W95H3nT+1AeAK1Tl0DMQk4/p7vhMmrzt+I4eSwx9rYaMycW5ARaQi6FdkrYs5o031IwP9LAJi4jHPcR9Ni19EdOoZ/z0r5eUDo9e+bTF46cfP2qt4hg/rlzdddeu+3TJ3zslFPueU2aWUgxa3RCpQul066voXjiJyhap1JSpaLS/gnzG85B+78mNE8nj52K6ioaK85l5t43U59bR9MCJhlc0AJ80+Fse902Wsdl/M9qhGMi9ZNLyqGAr76a+sTLKAf7pUibWNhwPs3ZWwiyl38USCYeW4Xfcde4nPem6edu3NH7ztvfji6SzOAFE0wdC+BJKCZeTNE6FVIfvMSsi7SOpzXxchrm1Oe+TbHp38i/+UdS7+eMFofR7FaGoFdIZZEKpTc9TRk71N8qDF3So37mHNZcgNoZ1MdfAlYiZmTvIOFwGhNvxKy10xzsIRpUhLnSn/H0Ui5+xSFvNaOYmvLFKfLOjElwj5h4NczwapJXTrPwguiQcGgcVil0aKA6xFDKWPsufPxZlCIImSIL0qtXPuZWiCsa6EonbS2RaSF0lDR8DFl7YEY0oSZg0kGaR+BhaJ9d5u4IigUC5XR+6Qtqf33aU8bPVd0j369SRw+ZYJmYa6SGkxfuxUQw7YD1yRLAIO3YQKIC7KEHBfj22/Ghoyhbh2LiRFOsSNAt6P2kRzihQGrgQfBQkAug/WPcMi51XHukHJHcxPvrwWb20RtB0AwiTtl1efxRc37RBeFSd+IekhED1BFJaFnDa9DfehNx4TaCjECoIVoQpaSsBcrgqPaJaWBK5u8EzzTicbiB5zpaZHxzi/Z9wvDJJWRBRREtwQ2nSRRFBJAGMQSwe+k+9CmC9JeOKcURAQ+qzM76Gc8MT/vn1x7+6j0CzWACCh4dKzOxjKjeTf+Xr0JXvRobOQJp305qz9E6+s1Y2ccfuhKrR3JNobORMLuR3ujpePtb1HNC6tD/1QI5OvE4wfslISmllNghz2Ho2HfTeehD0FtP0Xoe2Xt0pz9Fbf5WgrIrVtu9f5wygKgQPUNXfHxVW05+6lGv3ytqroI7FyB2UAuELBT5l6QHLsMCxLISQtlvU3/iFXSDwwMfpOYF5j269mNqq5+PbRnFih4+1KL7P87IEY6OFrj3KC3TGz2f1pMm6Wy+Hr1vkkIdl+sQrYp4IQUIXoUCe6o/0cBdEHNouGybrsmNN228Oe4BZGcVxQSiV51TkxSNgBEtQxjCYw+d+ywLGzKtNVMkTZQbP0B9SPCf3sLMz7s0bzRsh8GKgN+VqZ8j0HLStLGw8nTGnvCvlBuvIf72vRQ1xSlAErXK7WMaEEn7uhoH9QwaIXtmYij895eKH37iK9um4mLMGSGoI7mOlFXUarHEYkKyEETIYRaXQFEL6JZ1JNtGXPN+cm2MuauupPvRHxA2fQ8I1DGgT5Q6veud6bLNyCUvZvTYK+hu/BjFpveiRSSpU0XlhpdN0D45CMFlSb/pAqUatXqQTRtG0tWfnP1398nOvtZMGER5g7taUmQnuFXxkQmFZUgQYh3Z/i3yb9eSbsh039FjdFNgRVRUG3ioEQJIKGnMGH5tZuZDTXzjZ7DN7yVEoVsE1JzoiWAgXlZmnR7KEoVMAVfIQqY1pl//ZnnjD+5c+Bz8n8Q9+3m1RaXcteXCHvpng0o7IBnIxJFA7/s3MvdPn2OlR3I0LBs1X9iVgUYMxBgPyparP8PMYwrGzx3G+rPUU1kxvXOekHat7ZIVDwFP4vUR1TvuaviV12670h0577x1B1edsQjkIXqfFcbaAmpgvpuXxeSQEMZE0esF3xogKrr8Gv+ucYK6qTXl1ju7b1//cOeHrEPXrduVQTwCcqAG+UFBfpQQEbwqQ7BnbXY3iSmqieY9Jek3XYqVBRbL6o1l1tMs4zoS9Du31rZccuXsF9xRkWo/PnLJeFXb8u09ajugHUOVhZpXOcjeQBAQIwUneaC/KUNwkniVJi9rSkELfGZhQq77sqyF/t27uTnIIqDkOkaiHzKtnMmilGFpC1TNaAM7k/EUIGSKvhDy8tCIuTHSlP/8dv0X135x63UhkBafEDxyMALkjI40aK9wojkmkYzuriUPyASSOqagJmSBYpWCGJIK8LB07mJCClAGcANqBZvuH5cbvjxzqQjbU6oK/38QMJ4T4dBIXNMgoaSQEHVAcar83UQRAuqKelU17xyuxDUB7zleDEzmkjpTaaGhiGumOabrvubfvfG7s181260rBw8GwCLenGP4rMj2wmjmgHrlALM6/WCkUBUNkzpRhI4belaDsCpjfShrJfjSJs0VihQouuIMKz/6idhHPj19WWWK95XlwekMgi9kas/twgVN5pJT5ABRiAg1h8JBgmPRmU6J3skNVpwbIffwMNCXJSyAA1kVyKhkNx0Kn/9q/9q7N3VuB2Tdun0Pbg/uFCCW1dbpJVa9JmKvqrM9GL1SyBk817AcyQmsB37aMGOTY8joPG4G6gMfuX+TIebGeI1b7qxv/uC1vSlVkP10f+R+ZsCDS4EmsNYs4xeP0DtpnPIr83TvzhRzmVwEesfUaT1nnPG/EtKKh7FeRgO4BSw4KvuGLbt8T8QX5leET96Q39+h8/CFr6W45hrK3wHm9/8LgFSVKaxWjW2pTXH6Ao3TFJ8psHmQIjI61ofGQ1jfkVTlIogTLGDBqmJgFsTqEHqYOGIBsmRGRW+5mTs+/oUtVz3jGce2rrnmnu7++BmA0cqWPoJTZpHe7pRDMsxXKKWRCK3qDIcMzFd1iGq5K+ul9FETXBQRA1dMwV0hNwitnj/0YFM+cM2Oq0SYu/3Me1TuWCr6XAzG4IDHx78P7TwKcnafJO3SzKVWy5EsEMCKNg5oLpDcN+qNuO4b4eav/W/vvwZhywGX+9H7A8OBDmD3IcMcslSJoVlyxkrW/6xV/sf19YtV2TE4oD0gmAggVYyecfJyp/+D0sA6mxVoqULRyzmuKK7/CtfddfeDvw4Kl1++/+21kyKAZ1oMSdCuhqqK8EcmF2KRiEmgJzBR129+bWzb1Ec3XKxKPxthwGvvQMNEgIVy+Pvb7p0YJTXnZ2YZ23UWt1NI5qCyW7f2vsar33tf206dWepakOBuSbQoamVjrN0uF6S+9QFf+dhjWltu/kH/SwgzOSMiCIw/FnbcfyAw/w+JNU1a8mcILQAAAABJRU5ErkJggg==";

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
    cfg:{bundleEnabled:true,surrEnabled:true}},
  // ── Feature-Selektion ──
  {key:"feature_reduction",label:"Feature-Reduktion",desc:"Max. 77 Features via Importance + Korrelationsfilter",group:"feature",
    cfg:{fsEnabled:true,fsMethods:["catboost_importance","causal_forest"],fsCorrThresh:0.9,fsMaxFeatures:77}},
  // ── Erklärbarkeit ──
  {key:"explainability",label:"Explainability",desc:"SHAP-Analyse des Champions",group:"interpret",
    cfg:{explEnabled:true,explSampleSize:10000,explTopN:20}},
  // ── Tuning-Regularisierung ──
  {key:"reg_moderate",label:"Moderat",desc:"Milde Penalty: Meta-BLT 20%, Final (FMT/CFT) 20%. Für moderate Stichproben.",group:"regularization",
    cfg:{overfitPenalty:0.2,overfitTolerance:0.2,fmtOverfitPenalty:0.2,fmtOverfitTolerance:0.1,cfOverfitPenalty:0.2,cfOverfitTolerance:0.1}},
  {key:"reg_strong",label:"Stark",desc:"Starke Penalty: Meta-BLT 30%, Final (FMT/CFT) 35%. Für kleine Stichproben.",group:"regularization",
    cfg:{overfitPenalty:0.3,overfitTolerance:0.2,fmtOverfitPenalty:0.35,fmtOverfitTolerance:0.1,cfOverfitPenalty:0.35,cfOverfitTolerance:0.1}},
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
const buildDataPrepYaml = (dp, cfg) => {
  const l=[],a=s=>l.push(s);
  const outPath = dp.outputPath||"runs/data";
  a("# rubin DataPrep – generiert von der Web-UI");
  a("");
  // data_files ist Pflicht in AnalysisConfig – Pfade zeigen auf DataPrep-Output
  a("data_files:");
  a(`  x_file: ${outPath}/X.parquet`);
  a(`  t_file: ${outPath}/T.parquet`);
  a(`  y_file: ${outPath}/Y.parquet`);
  if(dp.evalFiles && dp.evalFiles.filter(f=>f).length > 0) {
    a(`  eval_x_file: ${outPath}/X_eval.parquet`);
    a(`  eval_t_file: ${outPath}/T_eval.parquet`);
    a(`  eval_y_file: ${outPath}/Y_eval.parquet`);
  }
  if((dp.evalFileIdxs||[]).length > 0 || dp.evalFileIdx!=null) {
    a(`  eval_mask_file: ${outPath}/eval_mask.npy`);
  }
  a("");
  a("data_prep:");
  a("  data_path:");
  (dp.files||[""]).filter(f=>f).forEach(f => a(`    - "${f}"`));
  const _tgts = (dp.targets||[""]).filter(t=>t.trim());
  a(`  target: ${_tgts.length===1 ? _tgts[0] : "["+_tgts.join(", ")+"]"}`);
  a(`  treatment: ${dp.treatment||""}`);
  if(dp.scoreName) a(`  score_name: ${dp.scoreName}`);
  if(dp.featurePath) a(`  feature_path: "${dp.featurePath}"`);
  a(`  output_path: ${outPath}`);
  a(`  delimiter: "${dp.delimiter||","}"`);
  if(dp.chunksize) a(`  chunksize: ${dp.chunksize}`);
  a(`  sas_encoding: ${dp.sasEncoding||"utf-8"}`);
  if(dp.fillNa && dp.fillNa!=="(keine)") a(`  fill_na_method: ${dp.fillNa}`);
  if(dp.binaryTarget !== undefined) a(`  binary_target: ${dp.binaryTarget ? "true" : "false"}`);
  if(dp.dedup && dp.dedupCol) {
    a("  deduplicate: true");
    a(`  deduplicate_id_column: ${dp.dedupCol}`);
  }
  if(dp.scoreAsFeature) a("  score_as_feature: true");
  if(dp.multiOpt) a(`  multiple_files_option: ${dp.multiOpt}`);
  if(dp.controlFileIndex>0) a(`  control_file_index: ${dp.controlFileIndex}`);
  if(dp.balanceTreat) a("  balance_treatments: true");
  if((dp.evalFileIdxs||[]).length > 1) {
    a(`  eval_file_index:`);
    (dp.evalFileIdxs||[]).forEach(i => a(`    - ${i}`));
  } else if((dp.evalFileIdxs||[]).length === 1) {
    a(`  eval_file_index: ${dp.evalFileIdxs[0]}`);
  } else if(dp.evalFileIdx!=null) {
    a(`  eval_file_index: ${dp.evalFileIdx}`);
  }
  // Explizite Feature-Auswahl (manuell oder Dictionary)
  const fs = dp.featureSelection||{};
  const selectedFeatures = Object.entries(fs).filter(([k,v])=>v===true).map(([k])=>k);
  const deselectedExists = Object.values(fs).some(v=>v===false);
  if(deselectedExists && selectedFeatures.length > 0) {
    a("  features:");
    selectedFeatures.forEach(f => a(`    - "${f}"`));
  }
  // Explizite Datentypen (cat/num)
  const ct = dp.colTypes||{};
  const catCols = Object.entries(ct).filter(([k,v])=>v==="cat").map(([k])=>k);
  if(catCols.length > 0) {
    a("  categorical_columns:");
    catCols.forEach(c => a(`    - "${c}"`));
  }
  // Treatment-Mapping (replacement)
  const tm = dp.treatMap||{};
  if(Object.keys(tm).length>0) {
    a("  treatment_replacement:");
    Object.entries(tm).forEach(([k,v]) => a(`    "${k}": ${v}`));
  }
  // MLflow
  if(dp.dpMlflow) {
    a("  log_to_mlflow: true");
    a(`  mlflow_experiment_name: ${cfg.expName||"rubin"}`);
    if(dp.dpRunName) a(`  mlflow_run_name: ${dp.dpRunName}`);
  }
  // Eval-Dateien (separater Datensatz, Preprocessor wird nur auf Train gefittet)
  if(dp.evalFiles && dp.evalFiles.filter(f=>f).length > 0) {
    a("  eval_data_path:");
    dp.evalFiles.filter(f=>f).forEach(f => a(`    - "${f}"`));
  }
  return l.join("\n");
};

// ── DataPrep Page ──
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
            // Arme = Anzahl VERSCHIEDENER Werte > 0. Zusammengelegte Ausprägungen
            // (z. B. Rohwerte 1 und 2 beide → 1) zählen als EIN Arm — das frühere
            // vals.filter(v=>v>0).length zählte Duplikate mit und zeigte fälschlich
            // "Multi-Treatment" für binäre Zusammenlegungs-Mappings.
            const arms = new Set(vals.filter(v=>v>0)).size;
            const isBinaryCfg = (cfg.treatmentType||"binary")==="binary";
            return (<>
              {!hasControl && <Info type="error">Keine Ausprägung als Control (0) zugeordnet.</Info>}
              {hasDups && <Info type="warn">Mehrere Ausprägungen haben denselben Wert – sie werden zusammengelegt.</Info>}
              {arms > 1 && isBinaryCfg && <Info type="error">Das Mapping erzeugt {arms} Treatment-Arme, unter „Daten“ ist aber <strong>Binary Treatment</strong> gewählt. Entweder Arme zusammenlegen (mehreren Rohwerten denselben Wert 1 zuordnen) oder den Treatment-Typ auf Multi-Treatment umstellen – sonst bricht die Analyse mit einem Treatment-Arm-Fehler ab.</Info>}
              {arms === 1 && !isBinaryCfg && <Info type="warn">Das Mapping erzeugt nur 1 Treatment-Arm, unter „Daten“ ist aber <strong>Multi-Treatment</strong> gewählt. Binary Treatment ist hier die passende Wahl (mehr Modelle verfügbar).</Info>}
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
              <Toggle label="Treatment-Balance" checked={dp.balanceTreat||false} onChange={v=>setDp(prev=>({...prev,balanceTreat:v}))} help="Per Random-Downsampling die Treatment-Raten über alle Dateien angleichen (Auswahl der Ziel-Rate nach maximaler effektiver Stichprobe)"/>
            </Col>}
          </Row>
          {dp.dedup&&<Info type="warn"><strong>Deduplizierung aktiv:</strong> Reduktion auf 1 Sample pro Kunde verhindert Leakage, wenn mehrere Einträge desselben Kunden in verschiedene Folds fallen.</Info>}
          {dp.balanceTreat&&<Info type="warn"><strong>Treatment-Balance aktiv:</strong> Bei mehr als 5 Prozentpunkten Abweichung zwischen Dateien werden Zeilen per Random-Downsampling entfernt. Die gemeinsame Ziel-Rate wird nach maximaler effektiver Stichprobe (N·p·(1−p)) gewählt — so bleibt die Balance erhalten, statt nur die Zeilenzahl zu maximieren.</Info>}

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
const YAML_TO_CFG = {
  "study_type":"studyType","mlflow.experiment_name":"expName","constants.SEED":"seed","constants.tuning_seed":"tuningSeed","constants.parallel_level":"parallelLevel","constants.work_dir":"workDir",
  "data_files.x_file":"x_file","data_files.t_file":"t_file","data_files.y_file":"y_file","data_files.s_file":"s_file",
  "data_files.eval_x_file":"eval_x_file","data_files.eval_t_file":"eval_t_file","data_files.eval_y_file":"eval_y_file","data_files.eval_s_file":"eval_s_file","data_files.eval_mask_file":"eval_mask_file",
  "treatment.type":"treatmentType","treatment.reference_group":"refGroup",
  "historical_score.name":"histScoreName","historical_score.column":"histScoreCol","historical_score.higher_is_better":"histScoreHigher",
  "data_processing.validate_on":"validateOn","data_processing.cross_validation_splits":"cvSplits","data_processing.mc_iters":"mcIters","data_processing.mc_agg":"mcAgg","data_processing.dml_crossfit_folds":"dmlCrossfitFolds",
  "data_processing.df_frac":"dfFrac","data_processing.reduce_memory":"reduceMem",
  "feature_selection.enabled":"fsEnabled","feature_selection.methods":"fsMethods","feature_selection.correlation_threshold":"fsCorrThresh","feature_selection.max_features":"fsMaxFeatures",
  "models.models_to_train":"models","models.ensemble":"ensembleEnabled",
  "base_learner.type":"baseLearner","base_learner.fixed_params":"blFixed",
  "causal_forest.tune_enabled":"cfTune","causal_forest.search_space":"_cfSS","causal_forest.depth_choices":"_cfDepthChoices","causal_forest.criterion_choices":"_cfCriterionChoices","causal_forest.n_trials":"cfTrials","causal_forest.single_fold":"cfSingleFold","causal_forest.scorer":"cfScorer","causal_forest.overfit_penalty":"cfOverfitPenalty","causal_forest.overfit_tolerance":"cfOverfitTolerance","causal_forest.overfit_max_penalized_gap":"cfOverfitMaxGap","causal_forest.tune_models":"cfTuneModels",
  "tuning.enabled":"tuningEnabled","tuning.n_trials":"tuningTrials","tuning.cv_splits":"dmlCrossfitFolds","tuning.single_fold":"tuningSingleFold","tuning.overfit_penalty":"overfitPenalty","tuning.overfit_tolerance":"overfitTolerance","tuning.overfit_max_penalized_gap":"overfitMaxGap",
  "tuning.timeout_seconds":"tuningTimeout","tuning.max_tuning_rows":"tuningMaxRows","tuning.models":"tuningModels",
  "final_model_tuning.enabled":"fmtEnabled","final_model_tuning.models":"fmtModels","final_model_tuning.cv_splits":"dmlCrossfitFolds","final_model_tuning.single_fold":"fmtSingleFold","final_model_tuning.overfit_penalty":"fmtOverfitPenalty","final_model_tuning.overfit_tolerance":"fmtOverfitTolerance","final_model_tuning.overfit_max_penalized_gap":"fmtOverfitMaxGap","final_model_tuning.scorer":"fmtScorer","final_model_tuning.n_trials":"fmtTrials",
  "final_model_tuning.max_tuning_rows":"fmtMaxRows","final_model_tuning.timeout_seconds":"fmtTimeout","final_model_tuning.fixed_params":"fmtFixed",
  "shap_values.calculate_shap_values":"explEnabled","shap_values.n_shap_values":"explSampleSize","shap_values.top_n_features":"explTopN","shap_values.num_bins":"shapBins",
  "selection.metric":"selMetric","selection.higher_is_better":"higherBetter","selection.manual_champion":"manualChamp",
  "surrogate_tree.enabled":"surrEnabled","surrogate_tree.min_samples_leaf":"surrMinLeaf","surrogate_tree.num_leaves":"surrLeaves","surrogate_tree.max_depth":"surrDepth",
  "optional_output.output_dir":"outputDir","optional_output.save_predictions":"savePreds","optional_output.predictions_format":"predsFormat","optional_output.max_prediction_rows":"maxPredRows",
  "bundle.enabled":"bundleEnabled","bundle.base_dir":"bundleDir","bundle.log_to_mlflow":"bundleMlflow",
};

const parseYamlToCfg = (yamlText) => {
  const result = {...DEFAULT_CFG};
  let section = "";
  for(const raw of yamlText.split("\n")) {
    const line = raw.replace(/#.*$/,"").trimEnd(); // strip comments
    if(!line.trim()) continue;
    const secMatch = line.match(/^([a-z_]+):\s*$/);
    if(secMatch) { section = secMatch[1]; continue; }
    // Top-level key-value (kein Indent, hat Wert)
    const topKv = line.match(/^([a-z_]+):\s+(.+)$/);
    if(topKv) {
      const tlKey = topKv[1];
      const tlCfgKey = YAML_TO_CFG[tlKey];
      if(tlCfgKey) {
        let v = topKv[2].trim();
        if(v === "true" || v === "True") v = true;
        else if(v === "false" || v === "False") v = false;
        else if(v === "null" || v === "None") v = null;
        else if(!isNaN(Number(v)) && v !== "") v = Number(v);
        else v = v.replace(/^"|"$/g,"");
        result[tlCfgKey] = v;
      }
      continue;
    }
    const kvMatch = line.match(/^\s+([a-zA-Z0-9_]+):\s*(.+)$/);
    if(!kvMatch) continue;
    const [,key,rawVal] = kvMatch;
    const fullKey = section + "." + key;
    const cfgKey = YAML_TO_CFG[fullKey];
    if(!cfgKey) continue;
    // Parse value
    let val = rawVal.trim();
    if(val === "true" || val === "True") val = true;
    else if(val === "false" || val === "False") val = false;
    else if(val === "null" || val === "None") val = null;
    else if(val.startsWith("[") && val.endsWith("]")) {
      val = val.slice(1,-1).split(",").map(s=>{
        s=s.trim(); if(!s) return undefined;
        if(s==="true"||s==="True") return true;
        if(s==="false"||s==="False") return false;
        if(!isNaN(Number(s))&&s!=="") return Number(s);
        return s;
      }).filter(v=>v!==undefined);
    } else if(val.startsWith("{") && val.endsWith("}")) {
      try {
        const inner = val.slice(1,-1).trim();
        if(!inner) { val = {}; } else {
          const obj = {};
          inner.split(",").forEach(p => {
            const [k,...rest] = p.split(":");
            let v = rest.join(":").trim();
            if(v==="true") v=true; else if(v==="false") v=false; else if(v==="null") v=null;
            else if(!isNaN(Number(v))) v=Number(v); else v=v.replace(/^"|"$/g,"");
            obj[k.trim()] = v;
          });
          val = obj;
        }
      } catch(e) { val = {}; }
    } else if(!isNaN(Number(val)) && val !== "") val = Number(val);
    result[cfgKey] = val;
  }
  // Derive downsample from df_frac
  if(result.dfFrac && result.dfFrac < 1 && result.dfFrac > 0) result.downsample = true;
  // Parse methods if string
  if(typeof result.fsMethods === "string") result.fsMethods = [result.fsMethods];
  // Parse forest_fixed_params (nested YAML block)
  const cfLines = yamlText.split("\n");
  let inForest = false, forestParams = {};
  for(const line of cfLines) {
    if(line.match(/^\s+forest_fixed_params:\s*$/)) { inForest = true; continue; }
    if(inForest) {
      const m = line.match(/^\s{4,}(\w+):\s*(.+)$/);
      if(m) {
        let v = m[2].trim();
        if(v==="true") v=true; else if(v==="false") v=false; else if(v==="null") v=null; else if(!isNaN(Number(v))) v=Number(v);
        forestParams[m[1]] = v;
      } else { inForest = false; }
    }
  }
  if(Object.keys(forestParams).length > 0) result.cfFixed = forestParams;
  // Parse nested fixed_params for "both"-mode (base_learner + final_model_tuning)
  const parseNestedFixedParams = (sectionName, resultKey) => {
    const lines = yamlText.split("\n");
    let inSection = false, inFixed = false;
    const nested = {};
    for(const line of lines) {
      if(line.match(new RegExp("^" + sectionName + ":\\s*$"))) { inSection = true; inFixed = false; continue; }
      if(inSection && line.match(/^[a-z_]+:\s*$/)) { inSection = false; inFixed = false; continue; }
      if(inSection && line.match(/^\s+fixed_params:\s*$/)) { inFixed = true; continue; }
      if(inFixed) {
        const m = line.match(/^\s{4,}(lgbm|catboost):\s*\{(.+)\}\s*$/);
        if(m) {
          const obj = {};
          m[2].split(",").forEach(p => {
            const [k,...rest] = p.split(":");
            if(!k) return;
            let v = rest.join(":").trim();
            if(v==="true") v=true;
            else if(v==="false") v=false;
            else if(v==="null") v=null;
            else if(!isNaN(Number(v)) && v !== "") v=Number(v);
            else v = v.replace(/^"|"$/g,"");
            obj[k.trim()] = v;
          });
          nested[m[1]] = obj;
        } else if(line.match(/^\s+[a-z_]+:/) && !line.match(/^\s{4,}/)) {
          inFixed = false;
        }
      }
    }
    if(Object.keys(nested).length > 0) result[resultKey] = nested;
  };
  parseNestedFixedParams("base_learner", "blFixed");
  parseNestedFixedParams("final_model_tuning", "fmtFixed");
  // Parse feature_selection methods
  const fsMatch = yamlText.match(/methods:\s*\[([^\]]+)\]/);
  if(fsMatch) result.fsMethods = fsMatch[1].split(",").map(s=>s.trim()).filter(Boolean);

  // Parse search_space für BL (tuning.search_space) und FMT (final_model_tuning.search_space)
  // YAML-Struktur:
  //   tuning:
  //     search_space:
  //       catboost:
  //         iterations: {type: int, low: 100, high: 300}
  //       lgbm:
  //         n_estimators: {type: int, low: 200, high: 400}
  // → wird in React-State `sp`/`spFmt` Struktur überführt:
  //   {catboost: {iterations: {low: 100, high: 300}}, lgbm: {n_estimators: {low: 200, high: 400}}}
  const parseSearchSpace = (sectionName) => {
    const lines = yamlText.split("\n");
    const out = {catboost: {}, lgbm: {}};
    let inSection = false, inSS = false, curLearner = null;
    for (const raw of lines) {
      const line = raw.replace(/#.*$/, "").trimEnd();
      if (!line.trim()) continue;
      // Start der Ziel-Sektion
      if (line.match(new RegExp("^" + sectionName + ":\\s*$"))) {
        inSection = true; inSS = false; curLearner = null; continue;
      }
      // Ende Sektion (neue top-level Sektion)
      if (inSection && line.match(/^[a-z_]+:\s*$/)) {
        inSection = false; inSS = false; curLearner = null; continue;
      }
      if (!inSection) continue;
      // search_space: Block-Start (2-space indent)
      if (line.match(/^\s{2}search_space:\s*$/)) { inSS = true; curLearner = null; continue; }
      // andere 2-space-Keys beenden search_space
      if (inSS && line.match(/^\s{2}[a-z_]+:/)) { inSS = false; curLearner = null; continue; }
      if (!inSS) continue;
      // Learner-Header (4-space indent): "    catboost:" oder "    lgbm:"
      const learnerMatch = line.match(/^\s{4}(catboost|lgbm):\s*$/);
      if (learnerMatch) { curLearner = learnerMatch[1]; continue; }
      // Parameter-Zeile (6-space indent): "      iterations: {type: int, low: 100, high: 300}"
      if (curLearner) {
        const pMatch = line.match(/^\s{6}([a-z0-9_]+):\s*\{([^}]+)\}\s*$/);
        if (pMatch) {
          const [, pname, inner] = pMatch;
          const parts = inner.split(",").map(s => s.trim());
          const obj = {};
          parts.forEach(p => {
            const [k, ...rest] = p.split(":");
            if (!k) return;
            const v = rest.join(":").trim();
            const n = Number(v);
            obj[k.trim()] = isNaN(n) ? v.replace(/^"|"$/g, "") : n;
          });
          // Konvertiere backend-Format {type, low, high, [log]} → UI-Format {low, high}
          // Das log-Flag wird NICHT in sp gespeichert (es kommt aus UI-Defs), aber wir
          // lesen es, damit der Parser nicht stolpert.
          if (obj.low !== undefined && obj.high !== undefined) {
            out[curLearner][pname] = {low: obj.low, high: obj.high};
          }
        }
      }
    }
    // Nur zurückgeben wenn mindestens eine Änderung drin ist
    const hasAny = Object.values(out).some(d => Object.keys(d).length > 0);
    return hasAny ? out : null;
  };
  const _bl_sp = parseSearchSpace("tuning");
  if (_bl_sp) result.__sp = _bl_sp;
  const _fmt_sp = parseSearchSpace("final_model_tuning");
  if (_fmt_sp) result.__spFmt = _fmt_sp;

  return result;
};

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
        <input type="number" min={0} max={999} step={1} value={cfg.seed==null?42:cfg.seed} onChange={e=>{if(e.target.value==="")return;const n=Math.max(0,Math.min(999,Math.round(Number(e.target.value))));set({...cfg,seed:n})}} style={{color:C.ruby,fontFamily:MONO,fontSize:13,background:C.rose,padding:"2px 8px",borderRadius:6,fontWeight:700,width:64,textAlign:"center",border:"1px solid "+C.border,outline:"none"}}/>
      </div>
      <style>{`
        .random-seed-slider { width:100%; }
        .random-seed-slider::-webkit-slider-runnable-track { background: linear-gradient(to right, #9B111E 0%, #9B111E var(--pct), #d0d7de var(--pct), #d0d7de 100%); height:6px; border-radius:3px; }
        .random-seed-slider::-moz-range-track { background: #d0d7de; height:6px; border-radius:3px; }
        .random-seed-slider::-moz-range-progress { background: #9B111E; height:6px; border-radius:3px; }
        .random-seed-slider::-webkit-slider-thumb { -webkit-appearance:none; width:16px; height:16px; border-radius:50%; background:#9B111E; border:2px solid #fff; margin-top:-5px; cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,0.15); }
        .random-seed-slider::-moz-range-thumb { width:12px; height:12px; border-radius:50%; background:#9B111E; border:2px solid #fff; cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,0.15); }
      `}</style>
      <input type="range" min={0} max={999} step={1} value={cfg.seed==null?42:cfg.seed} onChange={e=>set({...cfg,seed:Number(e.target.value)})} className="random-seed-slider" style={{"--pct":`${(cfg.seed==null?42:cfg.seed)/999*100}%`,WebkitAppearance:"none",appearance:"none",background:"transparent"}}/>
      <div style={{fontSize:10.5,color:C.textMuted,marginTop:6,lineHeight:1.4}}>Cross-Prediction Seed: gleicher Seed = gleiche Fold-Zuordnung bei identischen Daten</div>
    </div>
    </Col><Col>
    <div>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:8}}>
        <span style={{fontSize:12,fontWeight:600,color:C.text}}>Tuning Seed</span>
        <input type="number" min={0} max={999} step={1} value={cfg.tuningSeed==null?18:cfg.tuningSeed} onChange={e=>{if(e.target.value==="")return;const n=Math.max(0,Math.min(999,Math.round(Number(e.target.value))));set({...cfg,tuningSeed:n})}} style={{color:"#D4A853",fontFamily:MONO,fontSize:13,background:"#fef9ee",padding:"2px 8px",borderRadius:6,fontWeight:700,width:64,textAlign:"center",border:"1px solid "+C.border,outline:"none"}}/>
      </div>
      <style>{`
        .tuning-seed-slider { width:100%; }
        .tuning-seed-slider::-webkit-slider-runnable-track { background: linear-gradient(to right, #D4A853 0%, #D4A853 var(--pct), #d0d7de var(--pct), #d0d7de 100%); height:6px; border-radius:3px; }
        .tuning-seed-slider::-moz-range-track { background: #d0d7de; height:6px; border-radius:3px; }
        .tuning-seed-slider::-moz-range-progress { background: #D4A853; height:6px; border-radius:3px; }
        .tuning-seed-slider::-webkit-slider-thumb { -webkit-appearance:none; width:16px; height:16px; border-radius:50%; background:#D4A853; border:2px solid #fff; margin-top:-5px; cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,0.15); }
        .tuning-seed-slider::-moz-range-thumb { width:12px; height:12px; border-radius:50%; background:#D4A853; border:2px solid #fff; cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,0.15); }
      `}</style>
      <input type="range" min={0} max={999} step={1} value={cfg.tuningSeed==null?18:cfg.tuningSeed} onChange={e=>set({...cfg,tuningSeed:Number(e.target.value)})} className="tuning-seed-slider" style={{"--pct":`${(cfg.tuningSeed==null?18:cfg.tuningSeed)/999*100}%`,WebkitAppearance:"none",appearance:"none",background:"transparent"}}/>
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
  // berechnet (fullCv Folds × 2 Modelle: Control + Treated) — nicht pro Trial, da
  // trial-unabhängig. Der seltene Degenerations-Fallback ersetzt übersprungene Folds
  // (kein additiver Term). Wird in die Fits des jeweiligen Pseudo-Effekt-Tasks
  // eingerechnet, damit die Summe der "Fits"-Spalte exakt der Headline entspricht.
  const pseudoNuisanceOnce = (sig) => sig.split("__")[0]==="pseudo_effect" ? (fullCv * 2) : 0;
  const rowFitsFor = (sig) => trialsForTask(sig) * fitsForTask(sig) + pseudoNuisanceOnce(sig);
  const totalFits = plan.reduce((a,t) => a + rowFitsFor(t.sig), 0);
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
    if (fam==="pseudo_effect") return nt+"T × "+nCv+"F×"+K+"G cate + "+(fullCv*2)+" nuis (einmalig)";
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
          const rowFits = rowFitsFor(t.sig);
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
          <strong style={{color:"#374151"}}>Pseudo-Effekt:</strong> Die Nuisance-Modelle (μ₀, μ₁) mit den <em>getunten</em> Params aus dem Grouped-Outcome-Task werden <em>einmalig vor den Trials</em> berechnet ({fullCv}CV × 2 Gruppen = {fullCv*2} Fits) — sie hängen nicht von den Trial-Params der CATE-Regressoren ab. Anschließend werden pro Trial nur die CATE-Regressoren auf den resultierenden Pseudo-Outcomes evaluiert ({nCv}F × {K}G = {nCv*K} Fits).
        </div>
      )}
    </div>
  );
};

// ── Final-Model Tuning Plan ──
const FinalTuningPlanPreview = ({models, fmtEnabled, fmtModels, fmtSingleFold, fmtTrials, fmtCv, outerCv: outerCvProp, mcIters, isBoth, fmtScorer, studyType, isMulti}) => {
  const bothMultiplier = isBoth ? 2 : 1;
  const nTrials = (fmtTrials || 50) * bothMultiplier;
  const internalCv = fmtCv || 5;  // Innere CV (Default=5, synchronisiert mit BLT und DML)
  const outerCv = outerCvProp || 5; // Äußere Score-Folds (= cross_validation_splits)
  const mc = mcIters || 1;
  const eligible = (models||[]).filter(m => ["NonParamDML","DRLearner"].includes(m) && (fmtModels === undefined || fmtModels === null || fmtModels.includes(m)));

  // cache_values-Architektur: Nuisance EINMALIG pro äußerem Fold, Trials nur model_final
  const nuisanceFitsPerDmlFold = mc * internalCv * 2 + 1; // model_y + model_t + initial model_final
  const nuisanceFitsPerDrFold = mc * internalCv * 2 + 1; // propensity + regression + initial model_final
  const _fmtScRes = isMulti ? "rscore" : ((fmtScorer||"auto")==="auto" ? ((studyType||"rct")==="rct"?"qini":"rscore") : fmtScorer);
  const scorerFitsPerFold = _fmtScRes==="qini" ? 0 : 2 * 2; // RScorer: cv=2 × (model_y + model_t); QiniScorer: 0
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
  return (<Sec title="Kausale Modelle">{mt&&<Info type="warn"><strong>Multi-Treatment:</strong> Verfügbar sind <strong>ParamDML, DRLearner und CausalForestDML</strong>. NonParamDML unterstützt nur Binary Treatment (EconML-Restriktion: der Reweighting-Trick des finalen Modells benötigt ein skalares Treatment-Residuum) – ebenso Meta-Learner (SLearner/TLearner/XLearner) und CausalForest.</Info>}{cfg.hasNaN&&<Info type="warn"><strong>Fehlende Werte:</strong> CausalForestDML und CausalForest werden automatisch übersprungen (GRF-basierte Modelle können keine fehlenden Werte verarbeiten).</Info>}
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

// ── YAML Builder + Validate ──
const buildYaml = (cfg, sp, spFmt) => {
  const l=[],a=s=>l.push(s),bl=cfg.baseLearner||"catboost";
  const jsonInline = (obj) => {
    if(!obj || Object.keys(obj).length===0) return "{}";
    return "{ "+Object.entries(obj).map(([k,v])=>k+": "+(v===null?"null":v===true?"true":v===false?"false":typeof v==="string"?"\""+v+"\"":v)).join(", ")+" }";
  };
  const emitSS = (ssObj, indent) => {
    const pad = " ".repeat(indent);
    // Bei "both" beide Learner emittieren; sonst nur den aktiven
    const emitForLearner = (lname, defs) => {
      const cur = (ssObj && ssObj[lname]) || {};
      const modified = Object.entries(defs).filter(([k, d]) => {
        const c = cur[k];
        if (!c) return false;
        const lo = c.low ?? d.min, hi = c.high ?? d.max;
        return lo !== d.min || hi !== d.max;
      });
      if (modified.length === 0) return false;
      a(pad + "  " + lname + ":");
      modified.forEach(([k, d]) => {
        const c = cur[k];
        let lo = c.low ?? d.min, hi = c.high ?? d.max;
        const t = d.type === "int" ? "int" : "float";
        // log=True erfordert low > 0 (Optuna-Constraint)
        if (d.log && lo <= 0) lo = 1e-6;
        const logSuffix = d.log ? ", log: true" : "";
        a(pad + `    ${k}: {type: ${t}, low: ${lo}, high: ${hi}${logSuffix}}`);
      });
      return true;
    };
    if (bl === "both") {
      const cbHas = Object.entries(CB).some(([k, d]) => {
        const c = (ssObj && ssObj.catboost || {})[k];
        return c && ((c.low ?? d.min) !== d.min || (c.high ?? d.max) !== d.max);
      });
      const lgHas = Object.entries(LGBM).some(([k, d]) => {
        const c = (ssObj && ssObj.lgbm || {})[k];
        return c && ((c.low ?? d.min) !== d.min || (c.high ?? d.max) !== d.max);
      });
      if (!cbHas && !lgHas) return;
      a(pad + "search_space:");
      emitForLearner("catboost", CB);
      emitForLearner("lgbm", LGBM);
    } else {
      const defs = bl === "catboost" ? CB : LGBM;
      const hasAny = Object.entries(defs).some(([k, d]) => {
        const c = (ssObj && ssObj[bl] || {})[k];
        return c && ((c.low ?? d.min) !== d.min || (c.high ?? d.max) !== d.max);
      });
      if (!hasAny) return;
      a(pad + "search_space:");
      emitForLearner(bl, defs);
    }
  };
  const emitSSFmt = (ssObj, indent) => {
    const pad = " ".repeat(indent);
    const emitForLearner = (lname, defs) => {
      const cur = (ssObj && ssObj[lname]) || {};
      const modified = Object.entries(defs).filter(([k, d]) => {
        const c = cur[k];
        if (!c) return false;
        const lo = c.low ?? d.min, hi = c.high ?? d.max;
        return lo !== d.min || hi !== d.max;
      });
      if (modified.length === 0) return false;
      a(pad + "  " + lname + ":");
      modified.forEach(([k, d]) => {
        const c = cur[k];
        let lo = c.low ?? d.min, hi = c.high ?? d.max;
        const t = d.type === "int" ? "int" : "float";
        if (d.log && lo <= 0) lo = 1e-6;
        const logSuffix = d.log ? ", log: true" : "";
        a(pad + `    ${k}: {type: ${t}, low: ${lo}, high: ${hi}${logSuffix}}`);
      });
      return true;
    };
    if (bl === "both") {
      const cbHas = Object.entries(CB_FMT).some(([k, d]) => {
        const c = (ssObj && ssObj.catboost || {})[k];
        return c && ((c.low ?? d.min) !== d.min || (c.high ?? d.max) !== d.max);
      });
      const lgHas = Object.entries(LGBM_FMT).some(([k, d]) => {
        const c = (ssObj && ssObj.lgbm || {})[k];
        return c && ((c.low ?? d.min) !== d.min || (c.high ?? d.max) !== d.max);
      });
      if (!cbHas && !lgHas) return;
      a(pad + "search_space:");
      emitForLearner("catboost", CB_FMT);
      emitForLearner("lgbm", LGBM_FMT);
    } else {
      const defs = bl === "catboost" ? CB_FMT : LGBM_FMT;
      const hasAny = Object.entries(defs).some(([k, d]) => {
        const c = (ssObj && ssObj[bl] || {})[k];
        return c && ((c.low ?? d.min) !== d.min || (c.high ?? d.max) !== d.max);
      });
      if (!hasAny) return;
      a(pad + "search_space:");
      emitForLearner(bl, defs);
    }
  };
  const mt = cfg.treatmentType==="multi";

  if((cfg.studyType||"rct") !== "rct") a(`study_type: ${cfg.studyType}`);
  a("");
  a("mlflow:");
  a(`  experiment_name: ${cfg.expName||"rubin"}`);
  a("");
  a("constants:");
  a(`  SEED: ${cfg.seed==null?42:cfg.seed}`);
  a(`  tuning_seed: ${cfg.tuningSeed==null?18:cfg.tuningSeed}`);
  if((cfg.parallelLevel||3) !== 3) a(`  parallel_level: ${cfg.parallelLevel}`);
  if(cfg.workDir) a(`  work_dir: "${cfg.workDir}"`);
  a("");
  a("data_files:");
  a(`  x_file: ${cfg.x_file||"runs/data/X.parquet"}`);
  a(`  t_file: ${cfg.t_file||"runs/data/T.parquet"}`);
  a(`  y_file: ${cfg.y_file||"runs/data/Y.parquet"}`);
  a(`  s_file: ${cfg.s_file||"null"}`);
  if(cfg.validateOn==="external") {
    if(cfg.eval_x_file) a(`  eval_x_file: ${cfg.eval_x_file}`);
    if(cfg.eval_t_file) a(`  eval_t_file: ${cfg.eval_t_file}`);
    if(cfg.eval_y_file) a(`  eval_y_file: ${cfg.eval_y_file}`);
    if(cfg.eval_s_file) a(`  eval_s_file: ${cfg.eval_s_file}`);
  }
  if(cfg.eval_mask_file) {
    if(Array.isArray(cfg.eval_mask_file)) {
      a(`  eval_mask_file:`);
      cfg.eval_mask_file.forEach(f => a(`    - ${f}`));
    } else {
      a(`  eval_mask_file: ${cfg.eval_mask_file}`);
    }
  }
  a("");
  a("treatment:");
  a(`  type: ${cfg.treatmentType||"binary"}`);
  a(`  reference_group: ${cfg.refGroup||0}`);
  a("");
  a("historical_score:");
  a(`  name: ${cfg.histScoreName||"historical_score"}`);
  a(`  column: ${cfg.histScoreCol||"S"}`);
  a(`  higher_is_better: ${cfg.histScoreHigher!==false}`);
  a("");
  a("data_processing:");
  a(`  validate_on: ${cfg.validateOn||"cross"}`);
  if(cfg.validateOn==="cross"||cfg.validateOn===undefined) a(`  cross_validation_splits: ${cfg.cvSplits||5}`);
  if(cfg.validateOn==="external") a(`  cross_validation_splits: ${cfg.cvSplits||5}`);
  if(cfg.downsample) a(`  df_frac: ${cfg.dfFrac||0.1}`);
  a(`  reduce_memory: ${cfg.reduceMem!==false}`);
  if((cfg.dmlCrossfitFolds||5) !== 5) a(`  dml_crossfit_folds: ${cfg.dmlCrossfitFolds}`);
  if(cfg.mcIters && cfg.mcIters > 0) {
    a(`  mc_iters: ${cfg.mcIters}`);
    if((cfg.mcAgg||"mean") !== "mean") a(`  mc_agg: ${cfg.mcAgg}`);
  }
  a("");
  a("feature_selection:");
  a(`  enabled: ${!!cfg.fsEnabled}`);
  if(cfg.fsEnabled) {
    a(`  methods: [${(cfg.fsMethods||["catboost_importance"]).join(", ")}]`);
    a(`  max_features: ${cfg.fsMaxFeatures||77}`);
    a(`  correlation_threshold: ${cfg.fsCorrThresh||0.9}`);
  }
  a("");
  a("models:");
  a(`  models_to_train: [${(cfg.models||["NonParamDML"]).join(", ")}]`);
  if(cfg.ensembleEnabled) a("  ensemble: true");
  a("");
  a("base_learner:");
  a(`  type: ${bl}`);
  const blF = cfg.blFixed||{};
  if(bl === "both") {
    const cbF = blF.catboost||{}, lgbmF = blF.lgbm||{};
    if(Object.keys(cbF).length > 0 || Object.keys(lgbmF).length > 0) {
      a("  fixed_params:");
      if(Object.keys(cbF).length > 0) a(`    catboost: ${jsonInline(cbF)}`);
      if(Object.keys(lgbmF).length > 0) a(`    lgbm: ${jsonInline(lgbmF)}`);
    }
  } else if(Object.keys(blF).length > 0) a(`  fixed_params: ${jsonInline(blF)}`);
  a("");
  if((cfg.models||[]).includes("CausalForestDML") || (cfg.models||[]).includes("CausalForest")) {
    a("causal_forest:");
    const cfF = cfg.cfFixed||{};
    if(Object.keys(cfF).length > 0) {
      a("  forest_fixed_params:");
      Object.entries(cfF).forEach(([k,v]) => a(`    ${k}: ${v===null?"null":v}`));
    } else {
      a("  forest_fixed_params: { n_jobs: -1 }");
    }
    a(`  tune_enabled: ${!!cfg.cfTune}`);
    a(`  n_trials: ${cfg.cfTrials||50}`);
    if(cfg.cfSingleFold) a("  single_fold: true");
    if(cfg.cfScorer && cfg.cfScorer !== "auto") a(`  scorer: ${cfg.cfScorer}`);
    if((cfg.cfOverfitPenalty||0)>0){a(`  overfit_penalty: ${cfg.cfOverfitPenalty}`);a(`  overfit_tolerance: ${cfg.cfOverfitTolerance===undefined?0.1:cfg.cfOverfitTolerance}`);a(`  overfit_max_penalized_gap: ${cfg.cfOverfitMaxGap===undefined?1.0:cfg.cfOverfitMaxGap}`);}
    if((cfg.cfTuneModels||[]).length > 0) a(`  tune_models: [${cfg.cfTuneModels.join(", ")}]`);
    if(cfg._cfDepthChoices && cfg._cfDepthChoices.length > 0) {
      a(`  depth_choices: [${cfg._cfDepthChoices.join(", ")}]`);
    }
    if(cfg._cfCriterionChoices && cfg._cfCriterionChoices.length > 0) {
      a(`  criterion_choices: [${cfg._cfCriterionChoices.join(", ")}]`);
    }
    const cfSS = cfg._cfSS||{};
    if(Object.keys(cfSS).length > 0) {
      a("  search_space:");
      Object.entries(cfSS).forEach(([k,v]) => {
        a(`    ${k}:`);
        if(v.low !== undefined) a(`      low: ${v.low}`);
        if(v.high !== undefined) a(`      high: ${v.high}`);
      });
    }
        a("");
  }
  a("tuning:");
  a(`  enabled: ${!!cfg.tuningEnabled}`);
  if(cfg.tuningEnabled) {
    a(`  n_trials: ${cfg.tuningTrials||50}`);
    if((cfg.dmlCrossfitFolds||5) !== 5) a(`  cv_splits: ${cfg.dmlCrossfitFolds}`);
    if(cfg.tuningSingleFold) a("  single_fold: true");if((cfg.overfitPenalty||0)>0){a(`  overfit_penalty: ${cfg.overfitPenalty}`);a(`  overfit_tolerance: ${cfg.overfitTolerance===undefined?0.2:cfg.overfitTolerance}`);a(`  overfit_max_penalized_gap: ${cfg.overfitMaxGap===undefined?1.0:cfg.overfitMaxGap}`);}
    if(cfg.tuningTimeout) a(`  timeout_seconds: ${cfg.tuningTimeout}`);
    if(cfg.tuningMaxRows) a(`  max_tuning_rows: ${cfg.tuningMaxRows}`);
    if((cfg.tuningModels||[]).length>0) a(`  models: [${cfg.tuningModels.join(", ")}]`);
      }
  if(sp) emitSS(sp, 2);
  a("");
  a("final_model_tuning:");
  a(`  enabled: ${!!cfg.fmtEnabled}`);
  if(cfg.fmtEnabled) {
    a(`  n_trials: ${cfg.fmtTrials||50}`);
    if((cfg.dmlCrossfitFolds||5) !== 5) a(`  cv_splits: ${cfg.dmlCrossfitFolds}`);
    if((cfg.fmtModels||[]).length > 0) a(`  models: [${cfg.fmtModels.join(", ")}]`);
    if(cfg.fmtSingleFold) a("  single_fold: true");
    if((cfg.fmtOverfitPenalty||0)>0){a(`  overfit_penalty: ${cfg.fmtOverfitPenalty}`);a(`  overfit_tolerance: ${cfg.fmtOverfitTolerance===undefined?0.1:cfg.fmtOverfitTolerance}`);a(`  overfit_max_penalized_gap: ${cfg.fmtOverfitMaxGap===undefined?1.0:cfg.fmtOverfitMaxGap}`);}
    if(cfg.fmtScorer && cfg.fmtScorer !== "auto") a(`  scorer: ${cfg.fmtScorer}`);
        if(cfg.fmtTimeout) a(`  timeout_seconds: ${cfg.fmtTimeout}`);
    if(cfg.fmtMaxRows) a(`  max_tuning_rows: ${cfg.fmtMaxRows}`);
  }
  const fmtF = cfg.fmtFixed||{};
  if(bl === "both") {
    const cbF2 = fmtF.catboost||{}, lgbmF2 = fmtF.lgbm||{};
    if(Object.keys(cbF2).length > 0 || Object.keys(lgbmF2).length > 0) {
      a("  fixed_params:");
      if(Object.keys(cbF2).length > 0) a(`    catboost: ${jsonInline(cbF2)}`);
      if(Object.keys(lgbmF2).length > 0) a(`    lgbm: ${jsonInline(lgbmF2)}`);
    }
  } else if(Object.keys(fmtF).length > 0) a(`  fixed_params: ${jsonInline(fmtF)}`);
  if(spFmt) emitSSFmt(spFmt, 2);
  a("");
  a("shap_values:");
  a(`  calculate_shap_values: ${!!cfg.explEnabled}`);
  if(cfg.explEnabled) {
    a(`  n_shap_values: ${cfg.explSampleSize||10000}`);
    a(`  top_n_features: ${cfg.explTopN||20}`);
    a(`  num_bins: ${cfg.shapBins||10}`);
  }
  a("");
  a("selection:");
  a(`  metric: ${cfg.selMetric||(mt?"policy_value":"qini")}`);
  a(`  higher_is_better: ${cfg.higherBetter!==false}`);
  if(cfg.manualChamp) a(`  manual_champion: ${cfg.manualChamp}`);
  a("");
  a("surrogate_tree:");
  a(`  enabled: ${!!cfg.surrEnabled}`);
  if(cfg.surrEnabled) {
    a(`  min_samples_leaf: ${cfg.surrMinLeaf||50}`);
    a(`  num_leaves: ${cfg.surrLeaves||31}`);
    if(cfg.surrDepth) a(`  max_depth: ${cfg.surrDepth}`);
  }
  a("");
  a("optional_output:");
  a(`  output_dir: ${cfg.outputDir||"null"}`);
  if(cfg.maxPredRows) a(`  max_prediction_rows: ${cfg.maxPredRows}`);
  a("");
  a("bundle:");
  a(`  enabled: ${!!cfg.bundleEnabled}`);
  if(cfg.bundleEnabled) {
    a(`  base_dir: ${cfg.bundleDir||"runs/bundles"}`);
    a(`  log_to_mlflow: ${cfg.bundleMlflow!==false}`);
  }
  return l.join("\n");
};
const validate = cfg => {
  const i=[];
  if(!cfg.expName)i.push("Experiment-Name fehlt.");
  if(!cfg.x_file)i.push("X-Datei nicht angegeben.");
  if(!cfg.t_file)i.push("T-Datei nicht angegeben.");
  if(!cfg.y_file)i.push("Y-Datei nicht angegeben.");
  if(cfg.validateOn==="external") {
    if(!cfg.eval_x_file)i.push("Eval X-Datei nicht angegeben (externe Validierung).");
    if(!cfg.eval_t_file)i.push("Eval T-Datei nicht angegeben (externe Validierung).");
    if(!cfg.eval_y_file)i.push("Eval Y-Datei nicht angegeben (externe Validierung).");
  }
  // Effektive Modelle: NaN-blockierte und MT-inkompatible herausfiltern
  const nanBlocked = cfg.hasNaN ? ["CausalForestDML","CausalForest"] : [];
  const effectiveModels = (cfg.models||[]).filter(m => !nanBlocked.includes(m));
  if(!effectiveModels.length)i.push("Keine Modelle ausgewählt.");
  if(cfg.treatmentType==="multi"){const b=effectiveModels.filter(m=>btOnly.has(m));if(b.length > 0)i.push(`MT nicht kompatibel mit: ${b.join(", ")}`);}
  // Manual Champion muss in Modellen sein (spiegelt Backend-Validator)
  if(cfg.manualChamp && !(cfg.models||[]).includes(cfg.manualChamp)){
    i.push(`Manueller Champion „${cfg.manualChamp}" ist nicht in der Modell-Liste enthalten.`);
  }
  // "both"-Modus: fixed_params muss verschachtelt sein (spiegelt Backend-Validator)
  if((cfg.baseLearner||"")==="both"){
    const checkNested = (name, fp) => {
      if(!fp || Object.keys(fp).length===0) return;
      const flatKeys = Object.entries(fp).filter(([k,v])=>!["lgbm","catboost"].includes(k) || (typeof v!=="object" || v===null)).map(([k])=>k);
      if(flatKeys.length > 0){
        i.push(`${name} bei „CatBoost & LGBM" muss verschachtelt sein (lgbm/catboost Sub-Dicts). Flache Keys: ${flatKeys.join(", ")}`);
      }
    };
    checkNested("base_learner.fixed_params", cfg.blFixed);
    checkNested("final_model_tuning.fixed_params", cfg.fmtFixed);
  }
  // Reference Group: muss gültig sein (bei binary nur 0 erlaubt)
  if(cfg.treatmentType==="binary" && cfg.refGroup!==0 && cfg.refGroup!==undefined && cfg.refGroup!==null){
    i.push(`treatment.reference_group=${cfg.refGroup}: Bei binary Treatment nur 0 erlaubt.`);
  }
  // MT-incompatible Selection-Metrics (spiegelt Backend-Validator _bt_only_metrics).
  // Hinweis: 'policy_value' (global IPW) ist bei Multi-Treatment die EMPFOHLENE
  // Metrik (siehe settings.py) und gehört nicht in diese Liste.
  if(cfg.treatmentType==="multi"){
    const btOnlyMetrics = new Set(["qini","auuc","uplift_at_10pct","uplift_at_20pct","uplift_at_50pct"]);
    if(btOnlyMetrics.has(cfg.selMetric)){
      i.push(`selection.metric='${cfg.selMetric}' existiert bei Multi-Treatment nicht. Empfohlen: 'policy_value' (global), 'policy_value_T1', 'qini_T1', 'qini_T2' etc.`);
    }
    // Qini-Scorer (FMT/CFT) ist binär-only — spiegelt Backend-Validator.
    if(cfg.fmtEnabled && cfg.fmtScorer==="qini"){
      i.push(`final_model_tuning.scorer='qini' ist bei Multi-Treatment nicht möglich (binär-only). Bitte 'rscore' oder 'auto' wählen.`);
    }
    if(cfg.cfTune && cfg.cfScorer==="qini"){
      i.push(`causal_forest.scorer='qini' ist bei Multi-Treatment nicht möglich (binär-only). Bitte 'rscore' oder 'auto' wählen.`);
    }
  }
  return i;
};

const PPreview = ({cfg,sp,spFmt,totalFits}) => {const y=buildYaml(cfg,sp,spFmt),issues=validate(cfg);const mt=cfg.treatmentType==="multi";return(<>
  <Row gap={10}>
    <MC value={(cfg.models||[]).length} label="Modelle"/>
    <MC value={(cfg.baseLearner||"catboost")==="both"?"Both":cfg.baseLearner==="lgbm"?"LGBM":"CB"} label="Learner"/>
    <MC value={cfg.tuningEnabled?(cfg.tuningTrials||50)+"T":"Aus"} label="BLT"/>
    <MC value={cfg.fmtEnabled?(cfg.fmtTrials||50)+"T":"Aus"} label="FMT"/>
    <MC value={cfg.cfTune?(cfg.cfTrials||50)+"T":"Aus"} label="CFT"/>
    <MC value={cfg.validateOn==="external"?"Ext.":cfg.eval_mask_file?"TMES":"CV-"+(cfg.cvSplits||5)} label="Validierung"/>
  </Row>
  <div style={{height:16}}/>
  <Sec title="Config-Vorschau"><div style={{background:"#1e1e2e",border:"none",borderRadius:10,padding:"20px 24px",fontFamily:MONO,fontSize:12,whiteSpace:"pre-wrap",maxHeight:520,overflowY:"auto",lineHeight:1.7,color:"#cdd6f4",boxShadow:"inset 0 2px 8px rgba(0,0,0,0.15)"}}>{y}</div><div style={{marginTop:14,display:"flex",gap:10}}><Btn small onClick={()=>navigator.clipboard?.writeText(y)}>Kopieren</Btn><Btn small secondary onClick={()=>{const b=new Blob([y],{type:"text/yaml"});const u=URL.createObjectURL(b);const a=document.createElement("a");a.href=u;a.download="config.yml";a.click();}}>YAML herunterladen</Btn><Btn small secondary onClick={()=>fetch("./api/save-config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({yaml:y,filename:"config.yml"})}).then(r=>r.json()).then(d=>{if(d.status==="done")alert("Config gespeichert: "+d.path)}).catch(()=>alert("Backend nicht erreichbar"))}>Auf Server speichern</Btn></div></Sec><Sec title="Validierung">{issues.length>0?issues.map((i,x)=><Info key={x} type="warn">{i}</Info>):<Info type="success">Konfiguration ist valide.</Info>}</Sec></>);};

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
function App() {
  // ── Session-Persistenz: State überlebt Browser-Refresh ──
  const _ssKey = "rubin_app_state";
  const _ssLoad = () => { try { return JSON.parse(sessionStorage.getItem(_ssKey)) || {}; } catch { return {}; } };
  const _ssSave = (patch) => {
    try {
      const cur = _ssLoad();
      sessionStorage.setItem(_ssKey, JSON.stringify({...cur, ...patch}));
    } catch {}
  };
  const _ss0 = _ssLoad();

  const [pg,setPg] = useState(_ss0.pg || "overview");

  // Scroll-to-top bei Seitenwechsel
  useEffect(() => {
    // Robust scroll-to-top: verschiedene Browser und Hosting-Umgebungen
    // (Iframe in Domino, verschiedene Scroll-Container) behandeln.
    window.scrollTo(0, 0);
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
    // Fallback: nach kurzer Verzögerung nochmal scrollen, falls der
    // neue Seiteninhalt das Layout erst nach dem Render aufbaut.
    const t = setTimeout(() => {
      window.scrollTo(0, 0);
      document.documentElement.scrollTop = 0;
      document.body.scrollTop = 0;
    }, 50);
    return () => clearTimeout(t);
  }, [pg]);
  const [sp,setSp] = useState(_ss0.sp || {lgbm:{},catboost:{}});
  const [activeBase,setActiveBase] = useState(_ss0.activeBase || null);
  const [activeAddons,setActiveAddons] = useState(new Set(_ss0.activeAddons || []));
  const [spFmt,setSpFmt] = useState(_ss0.spFmt || {lgbm:{},catboost:{}});
  const [analysisRunning,setAnalysisRunning] = useState(false);
  const [analysisDone,setAnalysisDone] = useState(false);
  const [resetKey,setResetKey] = useState(0);
  const [serverOk,setServerOk] = useState(null); // null=checking, true=ok, false=error
  const [serverError,setServerError] = useState("");
  const [sysInfo,setSysInfo] = useState(null); // {ram:{percent,used_mb,total_mb}, cpu:{percent,cores}, process:{status,pid,step,percent}}

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const r = await fetch("./api/health");
        if(!r.ok) throw new Error("HTTP " + r.status);
        const d = await r.json();
        if(mounted) {
          setServerOk(true); setServerError("");
          if(d.ram || d.process || d.cpu) setSysInfo({ram: d.ram || {}, process: d.process || {}, cpu: d.cpu || {}});
        }
      } catch(e) {
        if(mounted) { setServerOk(false); setServerError(e.message || "Nicht erreichbar"); setSysInfo(null); }
      }
    };
    check();
    const iv = setInterval(check, 15000);
    return () => { mounted = false; clearInterval(iv); };
  }, []);
  const INIT_DP = {files:[""],evalFiles:[],featurePath:"",targets:[""],treatment:"",scoreName:"",multiOpt:"merge",controlFileIndex:0,fillNa:"(keine)",binaryTarget:false,dedup:false,dedupCol:"",scoreAsFeature:false,outputPath:"runs/data",delimiter:",",detectedCols:null,featureSelection:{},treatValues:[],treatMap:{},colTypes:{},targetValues:[],nanCols:[],colStats:{},dictResult:null,chunksize:300000,sasEncoding:"utf-8",dpMlflow:true,dpMlflowExp:"",balanceTreat:false,evalFileIdx:null,nRows:0};
  const INIT_CFG = {studyType:"rct",expName:"rubin",seed:42,tuningSeed:18,parallelLevel:3,workDir:null,x_file:"",t_file:"",y_file:"",s_file:"",treatmentType:"binary",refGroup:0,hasNaN:false,nanCols:[],validateOn:"cross",cvSplits:5,downsample:false,dfFrac:0.1,reduceMem:true,fsEnabled:false,fsMethods:["catboost_importance"],fsCorrThresh:0.9,fsMaxFeatures:77,outputDir:"",models:["NonParamDML","DRLearner","SLearner","TLearner","XLearner","ParamDML","CausalForestDML","CausalForest"],baseLearner:"catboost",tuningEnabled:false,tuningTrials:50,tuningSingleFold:false,tuningModels:[],fmtEnabled:false,fmtModels:[],fmtSingleFold:false,fmtTrials:50,fmtMaxRows:0,fmtScorer:"auto",cfTune:false,cfTrials:50,cfSingleFold:false,cfScorer:"auto",cfTuneModels:[],cfOverfitPenalty:0.0,cfOverfitTolerance:0.1,selMetric:"qini",higherBetter:true,manualChamp:null,surrEnabled:false,surrMinLeaf:50,surrLeaves:31,surrDepth:0,bundleEnabled:false,bundleDir:"runs/bundles",bundleMlflow:true,explEnabled:false,explSampleSize:10000,explTopN:20,shapModels:[],shapBins:10,histScoreName:"historical_score",histScoreCol:"S",histScoreHigher:true,maxPredRows:0,tuningTimeout:0,tuningMaxRows:0,fmtTimeout:0,blFixed:{},fmtFixed:{},cfFixed:{},ensembleEnabled:true,mcIters:null,mcAgg:"mean",fmtOverfitPenalty:0.0,fmtOverfitTolerance:0.1,dmlCrossfitFolds:5,overfitPenalty:0.0,overfitTolerance:0.2};

  const [dp,setDp] = useState(_ss0.dp ? {...INIT_DP, ..._ss0.dp} : {...INIT_DP});
  const [cfg,set] = useState(_ss0.cfg ? {...INIT_CFG, ..._ss0.cfg} : {...INIT_CFG});

  // ── Session-Persistenz: State bei jeder Änderung in sessionStorage schreiben ──
  useEffect(() => { _ssSave({pg}); }, [pg]);
  useEffect(() => { _ssSave({dp}); }, [dp]);
  useEffect(() => { _ssSave({cfg}); }, [cfg]);
  useEffect(() => { _ssSave({sp}); }, [sp]);
  useEffect(() => { _ssSave({spFmt}); }, [spFmt]);
  useEffect(() => { _ssSave({activeBase}); }, [activeBase]);
  useEffect(() => { _ssSave({activeAddons: [...activeAddons]}); }, [activeAddons]);

  // ── Datenstatistiken (aus DataPrep detect-columns) für Single-Fold-Warnung ──
  const dataStats = useMemo(() => {
    const cs = dp.colStats; if (!cs || !dp.treatment) return null;
    const treatCol = dp.treatment.toUpperCase();
    const targetCol = ((dp.targets||[])[0]||"").toUpperCase();
    const tMean = cs[treatCol]?.mean; const yMean = cs[targetCol]?.mean;
    const n = dp.nRows || 0;
    if (!n || tMean == null || yMean == null) return null;
    const nTreated = Math.round(n * tMean);
    const nPositive = Math.round(n * yMean);
    return { n, nTreated, nPositive, minority: Math.min(nTreated, nPositive), treatRate: tMean, outcomeRate: yMean };
  }, [dp.colStats, dp.treatment, dp.targets, dp.nRows]);

  const resetApp = () => {
    if(!window.confirm("Gesamte App zurücksetzen? Alle Einstellungen und Daten gehen verloren.")) return;
    setDp({...INIT_DP});
    set({...INIT_CFG});
    setSp({lgbm:{},catboost:{}});
    setSpFmt({lgbm:{},catboost:{}});
    setActiveBase(null);
    setActiveAddons(new Set());
    setAnalysisRunning(false);
    setAnalysisDone(false);
    setResetKey(k => k + 1);
    setPg("overview");
    try { sessionStorage.removeItem(_ssKey); } catch {}
    // Server-State zurücksetzen (beendet ggf. laufenden Prozess)
    fetch("./api/reset", {method:"POST"}).catch(()=>{});
  };

  // ── Sidebar Status Computation ──
  const pageStatus = (() => {
    const s = {};
    // Datenvorbereitung (optional)
    if(dp.detectedCols) s.dataprep = {st:"done",detail:"Spalten erkannt"};
    else if(dp.files?.some(f=>f.trim())) s.dataprep = {st:"active",detail:"Dateien eingetragen"};
    else s.dataprep = {st:"open",detail:"Optional"};

    // Daten
    const hasFiles = !!(cfg.x_file && cfg.t_file && cfg.y_file);
    if(!hasFiles) s.datafiles = {st:"required",detail:"X/T/Y-Dateien fehlen"};
    else { const fParts=[cfg.x_file?"X":"",cfg.t_file?"T":"",cfg.y_file?"Y":"",cfg.s_file?"S":""].filter(Boolean); s.datafiles = {st:"done",detail:fParts.join(", ")+" geladen"}; }

    // Vorlage
    const hasModels = (cfg.models||[]).length > 0;
    const btConflict = cfg.treatmentType==="multi" && (cfg.models||[]).some(m=>btOnly.has(m));
    if(!hasModels) s.template = {st:"required",detail:"Keine Modelle ausgewählt"};
    else if(btConflict) s.template = {st:"required",detail:"MT-inkompatible Modelle"};
    else { const study=(cfg.studyType||"rct")==="rct"?"RCT":"Observational"; s.template = {st:"done",detail:(cfg.models||[]).length+" Modelle, "+study+", "+cfg.treatmentType.toUpperCase()}; }

    // Konfiguration
    if(!cfg.expName) s.config = {st:"required",detail:"Experiment-Name fehlt"};
    else {
      const valLabel = cfg.validateOn==="cross" ? cfg.cvSplits+"F Cross-Val" : cfg.validateOn==="external" ? "External Eval" : "TMES";
      const parts = [valLabel];
      if(cfg.downsample) parts.push(Math.round(cfg.dfFrac*100)+"%");
      if(cfg.fsEnabled) parts.push("FS");
      parts.push("Seed "+cfg.seed+"/"+cfg.tuningSeed);
      s.config = {st:"done",detail:parts.join(", ")};
    }

    // Modelle & Tuning
    const nanBlocked = cfg.hasNaN ? new Set(["CausalForestDML","CausalForest"]) : new Set();
    const effectiveModels = (cfg.models||[]).filter(m => !nanBlocked.has(m));
    if(!effectiveModels.length) s.models = {st:"required",detail:"Keine Modelle"};
    else {
      const bl = (cfg.baseLearner||"catboost")==="both"?"CB+LGBM":(cfg.baseLearner||"catboost")==="catboost"?"CB":"LGBM";
      const parts = [effectiveModels.length+"M", bl];
      if(cfg.tuningEnabled) parts.push(cfg.tuningTrials+"T");
      if(cfg.fmtEnabled) parts.push("FMT "+cfg.fmtTrials+"T");
      if(!cfg.tuningEnabled && !cfg.fmtEnabled) parts.push("kein Tuning");
      s.models = {st:"done",detail:parts.join(", ")};
    }

    // Auswahl & Export
    const selParts = [cfg.selMetric];
    if(cfg.manualChamp) selParts.push("Champ: "+cfg.manualChamp);
    if(cfg.surrEnabled) selParts.push("Surrogate");
    if(cfg.bundleEnabled) selParts.push("Bundle");
    s.selection = {st:"done",detail:selParts.join(", ")};

    // Explainability
    if(!cfg.explEnabled) s.explain = {st:"open",detail:"Deaktiviert"};
    else {
      s.explain = {st:"done",detail:"SHAP, "+cfg.explSampleSize+" Samples, Champion"};
    }

    // Config-Vorschau
    const issues = validate(cfg);
    if(issues.length > 0) s.preview = {st:"required",detail:issues.length+" Problem"+(issues.length>1?"e":"")};
    else s.preview = {st:"done",detail:"Valide"};

    // Analyse starten
    if(analysisDone) s.run = {st:"done",detail:"Analyse abgeschlossen"};
    else if(analysisRunning) s.run = {st:"active",detail:"Analyse läuft …"};
    else if(issues.length > 0) s.run = {st:"required",detail:"Nicht startbar"};
    else s.run = {st:"open",detail:"Bereit"};

    return s;
  })();

  const statusColor = {done:"#4ade80",active:"#60a5fa",open:"rgba(255,255,255,0.3)",required:"#ff6b6b"};

  const nRequired = Object.values(pageStatus).filter(s=>s.st==="required").length;
  // ── Geschätzte Modell-Fits (vollständig) ──
  const estimateFits = () => {
    const m = cfg.models || []; if (!m.length) return 0;
    const outerCv = cfg.cvSplits || 5;     // Äußere Cross-Predictions
    const innerCv = cfg.dmlCrossfitFolds || 5; // Innere CV (BLT, FMT, DML — synchronisiert)
    const mc = cfg.mcIters || 1;
    const K = 2; // Binary Treatment
    const bundle = !!cfg.bundleEnabled;
    let f = 0;

    // Hilfsfunktion: interne Fits pro model.fit() Aufruf
    // DML/DR: internes Cross-Fitting (Nuisance) + model_final
    // mc_iters wiederholt nur Nuisance, nicht model_final
    const fitsPerFit = (x) => {
      if (["NonParamDML","ParamDML","CausalForestDML"].includes(x)) return mc*innerCv*2+1;
      if (x==="DRLearner") return mc*innerCv*2+1; // propensity + regression pro Fold
      if (x==="TLearner") return K;
      if (x==="XLearner") return 1+2*K;
      return 1; // SLearner, CausalForest
    };

    // 1. Feature-Selektion
    if (cfg.fsEnabled) (cfg.fsMethods||[]).forEach(x => { if (x==="catboost_importance"||x==="lgbm_importance") f++; if (x==="causal_forest") f++; });

    // 2. BL-Tuning (Signatur-getrennte Tasks)
    const bothMult = (cfg.baseLearner||"catboost") === "both" ? 2 : 1;
    if (cfg.tuningEnabled) {
      const bltM = (cfg.tuningModels||[]).length > 0 ? cfg.tuningModels : m;
      const _isRct = (cfg.studyType||"rct") === "rct";
      const tr = (cfg.tuningTrials||50) * bothMult, tc = cfg.tuningSingleFold ? 1 : innerCv;
      const propTr = (_isRct ? Math.min(cfg.tuningTrials||50, 20) : (cfg.tuningTrials||50)) * bothMult;
      const hasDml = bltM.some(x=>["NonParamDML","ParamDML","CausalForestDML"].includes(x));
      const hasDr = bltM.includes("DRLearner"), hasSl = bltM.includes("SLearner");
      const hasTl = bltM.includes("TLearner"), hasXl = bltM.includes("XLearner");
      // Scopes sind immer getrennt (kein Scope-Merge, train_subsample_ratio verschieden)
      // Task 1: Outcome (Classifier, DML model_y)
      if (hasDml) f += tr * tc;
      // Task 2+3: Propensity — getrennt: DML/DR (scope=all) + XL (scope=all_direct)
      if (hasDml||hasDr) f += propTr * tc;
      if (hasXl) f += propTr * tc;
      // Task 4+5: Outcome Regression — getrennt: DR (scope=all) + SL (scope=all_direct)
      if (hasDr) f += tr * tc;
      if (hasSl) f += tr * tc;
      // Task 6: Grouped Outcome Regression (TL + XL models)
      if (hasTl||hasXl) f += tr * tc * K;
      // Task 7: Pseudo-Effekt (XL cate_models): Pseudo-Outcome-Nuisance (μ₀,μ₁) wird
      // EINMALIG vor den Trials berechnet (innerCv Folds × 2 Modelle: Control + Treated) —
      // sie hängt nur von den getunten Outcome-Nuisances ab, nicht von den Trial-Params.
      // Der seltene Degenerations-Fallback ersetzt übersprungene Folds (kein additiver Term).
      // Pro Trial laufen nur die K CATE-Regressoren über tc Folds.
      if (hasXl) f += tr * (tc * K) + (innerCv * 2);
    }

    // 3. FMT (Final-Model-Tuning) — cache_values: Nuisance EINMALIG + RScorer + Trials nur model_final
    const fmtFitsPerDmlFit = mc * innerCv * 2 + 1;   // NonParamDML: model_y + model_t + model_final
    const fmtFitsPerDrFit = mc * innerCv * 2 + 1;   // DRLearner: propensity + regression + model_final
    const _fmtScRes = cfg.treatmentType==="multi" ? "rscore" : ((cfg.fmtScorer||"auto")==="auto" ? ((cfg.studyType||"rct")==="rct"?"qini":"rscore") : cfg.fmtScorer);
    const fmtScorerFitsPerFold = _fmtScRes==="qini" ? 0 : 4;  // RScorer: cv=2 × (model_y + model_t); QiniScorer: 0
    if (cfg.fmtEnabled) {
      const fmtTr = (cfg.fmtTrials||50) * bothMult;
      const fmtOuterFolds = cfg.fmtSingleFold ? 1 : outerCv;
      (cfg.fmtModels||[]).forEach(x => {
        if (x==="NonParamDML") {
          // Pre-fit: fmtOuterFolds × (Nuisance + RScorer), Trials: fmtTr × fmtOuterFolds × 1 (model_final)
          f += fmtOuterFolds * (fmtFitsPerDmlFit + fmtScorerFitsPerFold) + fmtTr * fmtOuterFolds;
        } else if (x==="DRLearner") {
          f += fmtOuterFolds * (fmtFitsPerDrFit + fmtScorerFitsPerFold) + fmtTr * fmtOuterFolds;
        }
      });
    }

    // 4. CF Tuning — cache_values: Nuisance EINMALIG + RScorer + Trials nur Forest/GRF refit
    if (cfg.cfTune) {
      const cfTrials = cfg.cfTrials || 50;
      const cfFolds = cfg.cfSingleFold ? 1 : innerCv;
      const _cfScRes = cfg.treatmentType==="multi" ? "rscore" : ((cfg.cfScorer||"auto")==="auto" ? ((cfg.studyType||"rct")==="rct"?"qini":"rscore") : cfg.cfScorer);
      const cfScorerFitsPerFold = _cfScRes==="qini" ? 0 : 4; // RScorer: cv=2 × (model_y + model_t); QiniScorer: 0
      // CausalForestDML: Pre-fit cfFolds × (Nuisance + RScorer) + cfTrials × cfFolds × 1 (forest refit)
      if ((cfg.cfTuneModels||[]).includes("CausalForestDML") && m.includes("CausalForestDML")) f += cfFolds * (mc*innerCv*2+1+cfScorerFitsPerFold) + cfTrials * cfFolds;
      // CausalForest: RScorer-Setup (cfFolds × 4) + cfTrials × cfFolds × 1 Forest
      if ((cfg.cfTuneModels||[]).includes("CausalForest") && m.includes("CausalForest")) f += cfTrials * cfFolds + cfFolds * cfScorerFitsPerFold;
    }

    // 5. Training
    // Cross/TMES: K-Fold CV + Train-Predictions-Fit (+1 bei SHAP für keep-fold)
    // External: nur 1 Fit auf Trainingsdaten, Predictions via predict() auf Holdout
    // TMES: normales K-Fold CV (Training auf allen Daten), Evaluation nur auf Mask-Subset
    const isHoldoutMode = cfg.validateOn === "external";
    const isExternal = isHoldoutMode;
    m.forEach(x => {
      const pf = fitsPerFit(x);
      if (isExternal) {
        f += pf;                                         // 1 Fit auf Trainingsdaten
      } else {
        f += pf * outerCv;                                    // Externe CV-Folds
        f += pf;                                          // Train-Predictions (full-data fit)
        // keep_last_fold_model bei SHAP: kein extra Fit, nur Referenz auf letztes Fold-Modell
      }
    });

    // 6. DRTester Nuisance (val-only, beide Modi)
    // CustomDRTester.fit_nuisance berechnet nur noch dr_val_ — der dr_train_-Pfad
    // wurde als ungenutzt/redundant entfernt. Daher in BEIDEN Modi nur die Val-
    // Nuisance: ~2 Modelle (regression + propensity) × cv=5 ≈ 10 Fits, einmalig
    // (shared pre-fit, über alle Modelle wiederverwendet).
    f += 10;

    // 7. Surrogate tree (Cross-Validated + Final)
    if (cfg.surrEnabled) f += outerCv + 1;

    // 8. Bundle: alle Modelle brauchen Refit auf vollen Daten
    if (bundle) {
      m.forEach(x => { f += fitsPerFit(x); });
      if (cfg.surrEnabled) f += 1;
    }

    return f;
  };
  const totalFits = estimateFits();

  return (
    <div style={{display:"flex",minHeight:"100vh",fontFamily:FONT,fontSize:15,color:"#24292f",background:"#faf7f8"}}>
      <nav style={{width:280,minWidth:280,background:"linear-gradient(180deg,#4a080f 0%,#6B0D15 30%,#8a0f1a 100%)",color:"#fff",position:"sticky",top:0,height:"100vh",overflowY:"auto",flexShrink:0,display:"flex",flexDirection:"column",boxShadow:"0 8px 40px rgba(107,13,21,0.25), 0 2px 12px rgba(0,0,0,0.08)",borderRadius:16}}>
        <div style={{display:"flex",flexDirection:"column",alignItems:"center",padding:"26px 24px 14px"}}>
          <div style={{background:"rgba(255,255,255,0.05)",borderRadius:20,padding:"14px 22px 10px",border:"1px solid rgba(255,255,255,0.06)"}}>
            <RubinLogo size={76} light/>
            <h1 style={{fontSize:30,fontWeight:700,margin:"10px 0 0",letterSpacing:1.5,textAlign:"center"}}>rubin</h1>
          </div>
          <p style={{fontSize:11,opacity:0.4,margin:"3px 0 0",letterSpacing:0.8}}>Causal ML Framework</p>
        </div>
        <div style={{borderTop:"1px solid rgba(255,255,255,0.08)",margin:"4px 28px 10px"}}/>
        <div style={{flex:1,padding:"0 10px"}}>
          {pages.map((p,idx) => {
            const active = pg===p.key;
            const st = pageStatus[p.key];
            const isOverview = p.key==="overview";
            const iColor = isOverview ? (active?"#fff":"rgba(255,255,255,0.5)") : statusColor[st?.st||"open"];
            const prevGroup = idx > 0 ? pages[idx-1].group : null;
            const showGroup = p.group && p.group !== prevGroup;
            return (
              <div key={p.key}>
                {showGroup && <div style={{fontSize:9,fontWeight:600,textTransform:"uppercase",letterSpacing:"1px",color:"rgba(255,255,255,0.2)",padding:"8px 18px 3px",marginTop:2}}>{p.group}</div>}
                <button onClick={()=>setPg(p.key)} style={{display:"flex",alignItems:"center",gap:10,width:"100%",textAlign:"left",padding:"7px 18px",background:active?"rgba(255,255,255,0.12)":"transparent",color:active?"#fff":"rgba(255,255,255,0.6)",border:"none",borderLeft:active?"3px solid rgba(255,255,255,0.85)":"3px solid transparent",borderRadius:active?"0 6px 6px 0":"0",cursor:"pointer",fontSize:13,fontWeight:active?600:400,fontFamily:"inherit",transition:"all 0.15s",letterSpacing:0.2,marginBottom:0}}
                onMouseEnter={e=>{if(!active)e.currentTarget.style.background="rgba(255,255,255,0.05)"}}
                onMouseLeave={e=>{if(!active)e.currentTarget.style.background="transparent"}}>
                  <span style={{width:18,height:18,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,position:"relative"}}><div style={{width:7,height:7,borderRadius:"50%",background:analysisRunning&&p.key==="run"?"#D4A853":iColor,boxShadow:st?.st==="done"?"0 0 6px rgba(74,222,128,0.35)":active?"0 0 8px rgba(255,255,255,0.35)":st?.st==="required"?"0 0 6px rgba(255,107,107,0.35)":"none",transition:"all 0.3s",animation:analysisRunning&&p.key==="run"?"rubin-dot-breathe 2.4s ease-in-out infinite":"none"}}/>{analysisRunning && p.key==="run" && <svg width="22" height="22" viewBox="0 0 22 22" style={{position:"absolute",left:-2,top:-2,animation:"rubin-spin 3s linear infinite"}}><circle cx="11" cy="11" r="9.5" fill="none" stroke="#D4A853" strokeWidth="1.5" strokeDasharray="15 45" strokeLinecap="round"/></svg>}</span>
                  <span style={{flex:1}}>{p.label}</span>
                  {p.optional && st?.st==="open" && <span style={{fontSize:8.5,background:"rgba(255,255,255,0.1)",padding:"1px 6px",borderRadius:6,color:"rgba(255,255,255,0.35)"}}>opt</span>}
                </button>
                {st && st.st==="required" && (
                  <div style={{fontSize:9.5,color:"#ff8a8a",padding:"0 18px 3px 49px",lineHeight:1.3}}>{st.detail}</div>
                )}
                {st && st.st==="done" && !isOverview && (
                  <div style={{fontSize:9.5,color:"rgba(255,255,255,0.3)",padding:"0 18px 2px 49px",lineHeight:1.3}}>{st.detail}</div>
                )}
              </div>
            );
          })}
        </div>

        <div style={{borderTop:"1px solid rgba(255,255,255,0.08)",margin:"8px 22px 0"}}/>
        <style>{_sidebarStyles}</style>
        <div style={{padding:"8px 18px 6px"}}>
          {analysisRunning ? (
            <div style={{background:"rgba(212,168,83,0.08)",borderRadius:10,padding:"10px 10px 8px",border:"1px solid rgba(212,168,83,0.18)"}}>
              <svg viewBox="0 0 200 72" width="100%" height="72" style={{display:"block",marginBottom:6}}>
                <defs>
                  <radialGradient id="tp"><stop offset="0%" stopColor="#D4A853" stopOpacity="0.5"/><stop offset="100%" stopColor="#D4A853" stopOpacity="0"/></radialGradient>
                </defs>
                <style>{`
                  @keyframes ht-pulse { 0%,100% { r:3.5; opacity:.65 } 50% { r:5.5; opacity:1 } }
                  @keyframes ht-glow { 0%,100% { r:12; opacity:.12 } 50% { r:20; opacity:.22 } }
                  .ht-pulse { animation: ht-pulse 1.8s ease-in-out infinite; }
                  .ht-glow { animation: ht-glow 1.8s ease-in-out infinite; }
                  @keyframes hc0 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.45}90%{opacity:.35}100%{cx:84;cy:33;opacity:0} }
                  @keyframes hc1 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.4}90%{opacity:.3}100%{cx:84;cy:39;opacity:0} }
                  @keyframes hc2 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.5}90%{opacity:.4}100%{cx:84;cy:34;opacity:0} }
                  @keyframes hc3 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.35}90%{opacity:.3}100%{cx:84;cy:38;opacity:0} }
                  @keyframes hc4 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.45}90%{opacity:.35}100%{cx:84;cy:32;opacity:0} }
                  @keyframes hc5 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.4}90%{opacity:.3}100%{cx:84;cy:40;opacity:0} }
                  @keyframes hc6 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.5}90%{opacity:.4}100%{cx:84;cy:35;opacity:0} }
                  @keyframes hc7 { 0%{cx:6;cy:36;opacity:0}8%{opacity:.35}90%{opacity:.25}100%{cx:84;cy:37;opacity:0} }
                  @keyframes hu0 { 0%{cx:96;cy:36;opacity:0;r:1.4}6%{opacity:.85;r:2.2}100%{cx:194;cy:8;opacity:0;r:1} }
                  @keyframes hu1 { 0%{cx:96;cy:36;opacity:0;r:1.4}6%{opacity:.75;r:2}100%{cx:190;cy:14;opacity:0;r:1.1} }
                  @keyframes hu2 { 0%{cx:96;cy:36;opacity:0;r:1.4}6%{opacity:.8;r:1.9}100%{cx:188;cy:20;opacity:0;r:1.2} }
                  @keyframes hu3 { 0%{cx:96;cy:36;opacity:0;r:1.4}6%{opacity:.7;r:2.1}100%{cx:192;cy:11;opacity:0;r:0.9} }
                  @keyframes hu4 { 0%{cx:96;cy:36;opacity:0;r:1.4}6%{opacity:.85;r:1.8}100%{cx:186;cy:25;opacity:0;r:1.3} }
                  @keyframes hu5 { 0%{cx:96;cy:36;opacity:0;r:1.4}6%{opacity:.65;r:2.3}100%{cx:195;cy:6;opacity:0;r:0.8} }
                  @keyframes hu6 { 0%{cx:96;cy:36;opacity:0;r:1.4}6%{opacity:.75;r:1.7}100%{cx:185;cy:28;opacity:0;r:1.1} }
                  @keyframes hd0 { 0%{cx:96;cy:36;opacity:0;r:1.3}6%{opacity:.65;r:1.9}100%{cx:194;cy:64;opacity:0;r:0.9} }
                  @keyframes hd1 { 0%{cx:96;cy:36;opacity:0;r:1.3}6%{opacity:.55;r:1.7}100%{cx:190;cy:58;opacity:0;r:1} }
                  @keyframes hd2 { 0%{cx:96;cy:36;opacity:0;r:1.3}6%{opacity:.6;r:1.8}100%{cx:188;cy:52;opacity:0;r:1.1} }
                  @keyframes hd3 { 0%{cx:96;cy:36;opacity:0;r:1.3}6%{opacity:.5;r:2}100%{cx:192;cy:61;opacity:0;r:0.8} }
                  @keyframes hd4 { 0%{cx:96;cy:36;opacity:0;r:1.3}6%{opacity:.6;r:1.6}100%{cx:186;cy:48;opacity:0;r:1.2} }
                  @keyframes hd5 { 0%{cx:96;cy:36;opacity:0;r:1.3}6%{opacity:.45;r:2.1}100%{cx:195;cy:66;opacity:0;r:0.7} }
                `}</style>
                <line x1="8" y1="36" x2="86" y2="36" stroke="rgba(255,255,255,0.12)" strokeWidth="1" strokeDasharray="4 3"/>
                <path d="M96,36 Q140,32 194,10" fill="none" stroke="rgba(74,222,128,0.08)" strokeWidth="0.7"/>
                <path d="M96,36 Q140,34 190,22" fill="none" stroke="rgba(74,222,128,0.05)" strokeWidth="0.5"/>
                <path d="M96,36 Q140,40 194,62" fill="none" stroke="rgba(248,113,113,0.12)" strokeWidth="0.7"/>
                <path d="M96,36 Q140,38 190,50" fill="none" stroke="rgba(248,113,113,0.08)" strokeWidth="0.5"/>
                <circle cx="91" cy="36" fill="url(#tp)" className="ht-glow"/>
                <circle cx="91" cy="36" fill="#D4A853" className="ht-pulse"/>
                {[
                  {a:"hc0",d:0,dur:3.4},{a:"hc1",d:0.42,dur:3.7},{a:"hc2",d:0.85,dur:3.2},
                  {a:"hc3",d:1.3,dur:3.6},{a:"hc4",d:1.7,dur:3.3},{a:"hc5",d:2.15,dur:3.8},
                  {a:"hc6",d:2.6,dur:3.1},{a:"hc7",d:3.05,dur:3.5},
                ].map((p,i) => <circle key={`c${i}`} cx="6" cy="36" r="1.4" fill="rgba(255,255,255,0.35)"
                  style={{animation:`${p.a} ${p.dur}s ease-in-out infinite`,animationDelay:`${p.d}s`}}/>)}
                {[
                  {a:"hu0",d:0,dur:3.1},{a:"hu1",d:0.45,dur:3.4},{a:"hu2",d:0.9,dur:2.9},
                  {a:"hu3",d:1.35,dur:3.3},{a:"hu4",d:1.8,dur:3.0},{a:"hu5",d:2.25,dur:3.5},
                  {a:"hu6",d:2.7,dur:2.8},
                ].map((p,i) => <circle key={`u${i}`} cx="96" cy="36" r="2" fill="#4ade80"
                  style={{animation:`${p.a} ${p.dur}s ease-out infinite`,animationDelay:`${p.d}s`}}/>)}
                {[
                  {a:"hd0",d:0.2,dur:3.3},{a:"hd1",d:0.65,dur:3.0},{a:"hd2",d:1.1,dur:3.5},
                  {a:"hd3",d:1.55,dur:2.9},{a:"hd4",d:2.0,dur:3.2},{a:"hd5",d:2.5,dur:3.4},
                ].map((p,i) => <circle key={`d${i}`} cx="96" cy="36" r="1.8" fill="#f87171"
                  style={{animation:`${p.a} ${p.dur}s ease-out infinite`,animationDelay:`${p.d}s`}}/>)}
              </svg>
              <div style={{fontSize:10.5,fontWeight:600,color:"#D4A853",textAlign:"center",letterSpacing:0.3}}>Heterogenität wird aufgedeckt …</div>
            </div>
          ) : analysisDone ? (
            <div style={{background:"rgba(74,222,128,0.1)",borderRadius:8,padding:"9px 12px",border:"1px solid rgba(74,222,128,0.15)"}}>
              <div style={{display:"flex",alignItems:"center",gap:6,fontSize:11.5,fontWeight:600,color:"#4ade80"}}><div style={{width:6,height:6,borderRadius:"50%",background:"#4ade80",boxShadow:"0 0 6px rgba(74,222,128,0.3)"}}/> Analyse abgeschlossen</div>
              <div style={{fontSize:10,color:"rgba(255,255,255,0.4)",marginTop:2}}>Ergebnisse verfügbar</div>
            </div>
          ) : nRequired > 0 ? (
            <div style={{background:"rgba(255,100,100,0.12)",borderRadius:8,padding:"9px 12px",border:"1px solid rgba(255,100,100,0.2)"}}>
              <div style={{display:"flex",alignItems:"center",gap:6,fontSize:11.5,fontWeight:600,color:"#ff6b6b"}}><div style={{width:6,height:6,borderRadius:"50%",background:"#ff6b6b",boxShadow:"0 0 6px rgba(255,107,107,0.3)"}}/> {nRequired} Schritt{nRequired>1?"e":""} offen</div>
              <div style={{fontSize:10,color:"rgba(255,255,255,0.4)",marginTop:2}}>Pflichtangaben fehlen</div>
            </div>
          ) : (
            <div style={{background:"rgba(74,222,128,0.1)",borderRadius:8,padding:"9px 12px",border:"1px solid rgba(74,222,128,0.15)"}}>
              <div style={{display:"flex",alignItems:"center",gap:6,fontSize:11.5,fontWeight:600,color:"#4ade80"}}><div style={{width:6,height:6,borderRadius:"50%",background:"#4ade80",boxShadow:"0 0 6px rgba(74,222,128,0.3)"}}/> Bereit</div>
              <div style={{fontSize:10,color:"rgba(255,255,255,0.4)",marginTop:2}}>Analyse kann gestartet werden</div>
            </div>
          )}

        </div>
        {totalFits > 0 && <div style={{padding:"4px 18px 0",display:"flex",alignItems:"center",justifyContent:"space-between",fontSize:10,color:"rgba(255,255,255,0.3)"}}>
          <span>Modell-Fits</span>
          <span style={{fontFamily:MONO,fontWeight:600,color:"rgba(255,255,255,0.5)"}}>{totalFits.toLocaleString("de-DE")}</span>
        </div>}
        <div style={{fontSize:10,opacity:0.25,textAlign:"center",padding:"10px 0 6px"}}>v1.0</div>
        <div style={{padding:"0 16px 4px"}}>
          <button onClick={resetApp} style={{display:"flex",alignItems:"center",justifyContent:"center",gap:6,width:"100%",padding:"7px 0",background:"rgba(255,255,255,0.06)",border:"1px solid rgba(255,255,255,0.1)",borderRadius:6,color:"rgba(255,255,255,0.45)",fontSize:10.5,fontWeight:500,cursor:"pointer",fontFamily:"inherit",transition:"all 0.15s",letterSpacing:0.3}}
            onMouseEnter={e=>{e.currentTarget.style.background="rgba(255,100,100,0.15)";e.currentTarget.style.color="rgba(255,200,200,0.8)";e.currentTarget.style.borderColor="rgba(255,100,100,0.3)"}}
            onMouseLeave={e=>{e.currentTarget.style.background="rgba(255,255,255,0.06)";e.currentTarget.style.color="rgba(255,255,255,0.45)";e.currentTarget.style.borderColor="rgba(255,255,255,0.1)"}}>
            ↺ App zurücksetzen
          </button>
        </div>
        <div style={{padding:"0 16px 14px"}}>
          {/* Server-Verbindung */}
          <div style={{display:"flex",alignItems:"center",gap:5,fontSize:10,color:serverOk===true?"rgba(74,222,128,0.7)":serverOk===false?"rgba(255,100,100,0.8)":"rgba(255,255,255,0.3)",marginBottom:6}}>
            <div style={{width:6,height:6,borderRadius:3,background:serverOk===true?"#4ade80":serverOk===false?"#ff6b6b":"#888"}}/>
            {serverOk===true?"Server verbunden":serverOk===false?"Nicht erreichbar":"Prüfe..."}
          </div>
          {serverOk===false && serverError && <div style={{fontSize:9,color:"rgba(255,100,100,0.5)",marginBottom:6}}>{serverError}</div>}
          {/* Prozess-Status */}
          {sysInfo && sysInfo.process && (()=>{const ps=sysInfo.process;const stMap={idle:{c:"rgba(255,255,255,0.3)",bg:"#888",t:"Bereit"},running:{c:"rgba(74,222,128,0.7)",bg:"#4ade80",t:"Läuft"+(ps.step?" – "+ps.step:"")},done:{c:"rgba(74,222,128,0.7)",bg:"#22c55e",t:"Fertig"},error:{c:"rgba(255,100,100,0.8)",bg:"#ff6b6b",t:"Fehler"},crashed:{c:"rgba(255,100,100,0.8)",bg:"#ff6b6b",t:"Abgestürzt"}};const s=stMap[ps.status]||stMap.idle;return(<><div style={{display:"flex",alignItems:"center",gap:5,fontSize:10,color:s.c,marginBottom:4}}><div style={{width:6,height:6,borderRadius:3,background:s.bg}}/><span style={{flex:1,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{s.t}</span>{ps.pid&&<span style={{opacity:0.4,fontSize:9}}>PID {ps.pid}</span>}</div>{ps.status==="running"&&ps.percent>0&&(<div style={{height:2,background:"rgba(255,255,255,0.1)",borderRadius:1,overflow:"hidden",marginBottom:4}}><div style={{height:"100%",width:ps.percent+"%",background:"#4ade80",borderRadius:1,transition:"width 0.5s"}}/></div>)}{(ps.status==="error"||ps.status==="crashed")&&(<button onClick={async()=>{try{await fetch("./api/restart-process",{method:"POST"});setSysInfo(p=>p?{...p,process:{...p.process,status:"idle",pid:null}}:p)}catch(e){}}} style={{display:"block",width:"100%",padding:"4px 0",marginBottom:4,background:"rgba(255,100,100,0.15)",border:"1px solid rgba(255,100,100,0.3)",borderRadius:4,color:"rgba(255,100,100,0.8)",fontSize:10,fontWeight:600,cursor:"pointer",fontFamily:"inherit"}}>Prozess neu starten</button>)}</>)})()}
          {/* CPU + RAM Auslastung */}
          {sysInfo && sysInfo.ram && sysInfo.ram.total_mb>0 && (()=>{const r=sysInfo.ram;const rPct=r.percent||0;const rColor=rPct>85?"#ff6b6b":rPct>70?"#f59e0b":"rgba(74,222,128,0.6)";const c=sysInfo.cpu||{};const cPct=c.percent||0;const cColor=cPct>85?"#ff6b6b":cPct>70?"#f59e0b":"rgba(74,222,128,0.6)";return(<div style={{marginTop:2,display:"flex",gap:10}}><div style={{flex:1}}><div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"rgba(255,255,255,0.35)",marginBottom:2}}><span>CPU</span><span>{cPct}%{c.cores?" · "+Math.round(cPct*c.cores/100)+"/"+c.cores+"C":""}</span></div><div style={{height:3,background:"rgba(255,255,255,0.08)",borderRadius:2,overflow:"hidden"}}><div style={{height:"100%",width:cPct+"%",background:cColor,borderRadius:2,transition:"width 1s"}}/></div></div><div style={{flex:1}}><div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"rgba(255,255,255,0.35)",marginBottom:2}}><span>RAM</span><span>{rPct}%{r.used_mb?" · "+(r.used_mb/1024).toFixed(1)+"/"+(r.total_mb/1024).toFixed(1)+" GB":""}</span></div><div style={{height:3,background:"rgba(255,255,255,0.08)",borderRadius:2,overflow:"hidden"}}><div style={{height:"100%",width:rPct+"%",background:rColor,borderRadius:2,transition:"width 1s"}}/></div></div></div>)})()}
        </div>
      </nav>
      <main style={{flex:1,maxWidth:960,padding:"36px 52px 80px",margin:"0 auto"}}>
        {pg==="overview" && <POverview setPg={setPg}/>}
        <div key={"dp-"+resetKey} style={{display: pg==="dataprep" ? "block" : "none"}}><PDataPrep dp={dp} setDp={setDp} cfg={cfg} setCfg={set} setPg={setPg}/></div>
        {pg==="datafiles" && <PData cfg={cfg} set={set} setCfg={set} activeBase={activeBase} setActiveBase={setActiveBase} activeAddons={activeAddons} setActiveAddons={setActiveAddons} setSp={setSp} setSpFmt={setSpFmt} view="files" sysInfo={sysInfo}/>}
        {pg==="template" && <PData cfg={cfg} set={set} setCfg={set} activeBase={activeBase} setActiveBase={setActiveBase} activeAddons={activeAddons} setActiveAddons={setActiveAddons} setSp={setSp} setSpFmt={setSpFmt} view="template" sysInfo={sysInfo}/>}
        {pg==="config" && <PConfig cfg={cfg} set={set}/>}
        {pg==="models" && <PModels cfg={cfg} set={set} sp={sp} setSp={setSp} spFmt={spFmt} setSpFmt={setSpFmt} dataStats={dataStats} sysInfo={sysInfo}/>}
        {pg==="selection" && <PSelection cfg={cfg} set={set}/>}
        {pg==="explain" && <PExplain cfg={cfg} set={set}/>}
        {pg==="preview" && <PPreview cfg={cfg} sp={sp} spFmt={spFmt} totalFits={totalFits}/>}
        <div key={"run-"+resetKey} style={{display: pg==="run" ? "block" : "none"}}><PRun cfg={cfg} setPg={setPg} sp={sp} spFmt={spFmt} onRunningChange={setAnalysisRunning} onDoneChange={setAnalysisDone} totalFits={totalFits}/></div>
        <div style={{display:"flex",flexDirection:"column",alignItems:"center",color:"#ccc",fontSize:11,marginTop:24,paddingTop:16,borderTop:`1px solid ${C.borderLight}`,gap:10}}><RubinLogo size={24}/><span>rubin – Causal ML Framework</span></div>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);