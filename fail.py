# ── Alle generierten Artefakte (konsolidiert unter runs/) ──
# Struktur:
#   runs/
#   ├── mlruns/        MLflow Tracking
#   ├── data/          DataPrep-Ausgabe (X/T/Y/S.parquet, dtypes, eval_mask)
#   ├── configs/       Gespeicherte YAML-Konfigurationen
#   ├── exports/       Feature Dictionary (Excel)
#   ├── bundles/       Bundle-Export (PKL, Surrogate, Registry)
#   ├── uploads/       Server-Uploads
#   └── .rubin_cache/  Pipeline-Cache (Reports, Configs, Progress)
runs/

# ── Python ──
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
*.egg

# ── Environment ──
.pixi/
.venv/
pixi.lock

# ── IDE ──
.idea/
.vscode/
*.swp
*.swo
