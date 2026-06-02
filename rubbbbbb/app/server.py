#!/usr/bin/env python3
"""rubin – Backend-Server (Flask)

Architektur:
  GET  /                    → React-App (frontend/index.html)
  GET  /api/health          → Health-Check (Monitoring, Load-Balancer)
  POST /api/upload          → Datei-Upload (multipart)
  POST /api/detect-columns  → Spalten aus Datei erkennen
  POST /api/save-config     → YAML auf Disk speichern
  POST /api/import-config   → YAML importieren (Client-Sync)
  POST /api/run-analysis    → Analyse starten (Background)
  POST /api/run-dataprep    → Datenvorbereitung starten (Background)
  GET  /api/progress        → Fortschritts-Status (Polling)
  POST /api/reset           → State zurücksetzen + Prozess beenden
  GET  /api/results         → Liste der Ergebnis-Dateien
  GET  /api/download/<path> → Datei herunterladen (als Attachment)
  GET  /api/view/<path>     → Datei inline anzeigen (iframe-Einbettung)
  GET  /api/report          → HTML-Report + Metriken laden
  404                       → SPA-Routing (index.html)

Route-Module:
  app/routes/frontend.py  – SPA Index + 404
  app/routes/health.py    – Health-Check, System-Info, Restart
  app/routes/data.py      – Upload, Spalten-Erkennung, Feature Dictionary
  app/routes/config.py    – Config Save/Import
  app/routes/analysis.py  – Background-Tasks, Progress, Reset
  app/routes/results.py   – Ergebnisse, Download, Report
"""

from __future__ import annotations

import os
import shutil

from flask import Flask, send_from_directory

from app.state import log, __version__, FRONTEND, ROOT, WORK_DIR, UPLOAD_DIR, MAX_UPLOAD_MB


def _clean_pycache():
    for dirpath, dirnames, _ in os.walk(str(ROOT)):
        for d in list(dirnames):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(dirpath, d), ignore_errors=True)
                dirnames.remove(d)

_clean_pycache()


def create_app() -> Flask:
    """App-Factory: Erstellt und konfiguriert die Flask-App."""
    app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False

    # ── Blueprints registrieren ──
    from app.routes.frontend import bp as frontend_bp
    from app.routes.health import bp as health_bp
    from app.routes.data import bp as data_bp
    from app.routes.config import bp as config_bp
    from app.routes.analysis import bp as analysis_bp
    from app.routes.results import bp as results_bp

    app.register_blueprint(frontend_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(config_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(results_bp)

    # ── Globale Error-Handler (müssen auf App-Level, nicht Blueprint) ──
    @app.errorhandler(404)
    def not_found(e):
        """SPA-Routing: Unbekannte Pfade liefern index.html (React-Router)."""
        from flask import request, jsonify
        if request.path.startswith("/api/"):
            return jsonify({"status": "error", "message": "Endpoint nicht gefunden."}), 404
        return send_from_directory(str(FRONTEND), "index.html")

    @app.errorhandler(403)
    def forbidden(e):
        """Path-Traversal oder unerlaubter Zugriff → JSON-Fehler."""
        from flask import jsonify
        return jsonify({"status": "error", "message": "Zugriff verweigert."}), 403

    @app.errorhandler(500)
    def internal_error(e):
        """Unbehandelte Ausnahme → JSON-Fehler statt HTML."""
        from flask import jsonify
        log.error("Unbehandelte Ausnahme: %s", e)
        return jsonify({"status": "error", "message": f"Interner Serverfehler: {e}"}), 500

    @app.errorhandler(413)
    def request_too_large(e):
        """Upload zu groß (MAX_CONTENT_LENGTH überschritten)."""
        from flask import jsonify
        return jsonify({"status": "error", "message": f"Datei zu groß (max. {MAX_UPLOAD_MB} MB)."}), 413

    return app


# ── App-Instanz (für Gunicorn: `gunicorn app.server:app`) ──
app = create_app()


def main():
    port = int(os.environ.get("DOMINO_APP_PORT", os.environ.get("PORT", 8501)))
    host = os.environ.get("RUBIN_HOST", "0.0.0.0")
    is_domino = bool(os.environ.get("DOMINO_PROJECT_NAME"))

    log.info("Server v%s starten auf %s:%d", __version__, host, port)
    log.info("Frontend: %s", FRONTEND)
    log.info("Projekt:  %s", ROOT)
    if is_domino:
        log.info("Umgebung: Domino (%s)", os.environ.get("DOMINO_PROJECT_NAME"))
    else:
        log.info("Umgebung: Standalone")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
