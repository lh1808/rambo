#!/usr/bin/env bash
# ============================================================
# Domino App Deployment – rubin UI
# ============================================================
# Startet den rubin Flask-Server (React-UI + Python-Backend).
# Domino stellt die App ueber $DOMINO_APP_PORT bereit.
# Lokal: Standard-Port 8501.
# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Port-Konfiguration
export PORT="${DOMINO_APP_PORT:-8501}"

# ══════════════════════════════════════════════
# UI-Build: JSX-Quelldateien → rubin_ui_src.jsx + index.html
# ══════════════════════════════════════════════
if [ -f "app/build_ui.py" ]; then
    echo "[rubin] Build UI: app/src/*.jsx → index.html ..."
    python3 app/build_ui.py || echo "[rubin] WARNUNG: build_ui.py fehlgeschlagen"
fi

# Pixi-Pfad
if [ -f "${HOME}/.pixi/bin/pixi" ]; then
    export PATH="${HOME}/.pixi/bin:${PATH}"
fi

# ══════════════════════════════════════════════
# Variante 1: Bereits in pixi → direkt starten
# ══════════════════════════════════════════════
if [ -n "${PIXI_PROJECT_ROOT:-}" ]; then
    echo "[rubin] Pixi-Environment aktiv – starte Server auf Port ${PORT} ..."
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
    exec python -m app.server
fi

# ══════════════════════════════════════════════
# Variante 2: pixi verfügbar → Environment automatisch
# ══════════════════════════════════════════════
if command -v pixi &>/dev/null; then
    echo "[rubin] Pixi erkannt – starte App im 'app'-Environment ..."
    exec pixi run -e app app
fi

# ══════════════════════════════════════════════
# Variante 3: Fallback ohne pixi (pip)
# ══════════════════════════════════════════════
echo "[rubin] Kein pixi – Fallback auf pip ..."

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

if [ -f "app/requirements_app.txt" ]; then
    echo "[rubin] Installiere App-Abhaengigkeiten ..."
    pip install -q -r app/requirements_app.txt 2>/dev/null || true
fi

echo "[rubin] Starte Server auf Port ${PORT} ..."
exec python -m app.server
