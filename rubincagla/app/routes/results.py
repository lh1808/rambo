"""Ergebnis-Routen: Download, View, Report, Dateiliste."""
from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Blueprint, jsonify, send_file, abort, request

from app.state import log, ROOT, WORK_DIR

bp = Blueprint("results", __name__)

def _scan_result_files() -> list[dict]:
    """Scannt Ergebnis-Dateien in output/ und bundles/."""
    files = []
    seen = set()

    def _add(f: Path, desc: str):
        key = str(f)
        if key not in seen and f.is_file():
            seen.add(key)
            try:
                files.append({
                    "name": f.name,
                    "path": str(f.relative_to(ROOT)),
                    "desc": desc,
                    "size": f.stat().st_size,
                })
            except Exception:
                pass

    # Bekannte Ergebnis-Dateien (Cache zuerst, dann output/, dann ROOT)
    search_dirs = [WORK_DIR / ".rubin_cache", ROOT / "output", ROOT]
    known = [
        ("analysis_report.html", "HTML-Report"),
        ("uplift_eval_summary.json", "Evaluationsmetriken"),
        ("model_registry.json", "Champion & Challenger"),
    ]
    for name, desc in known:
        for d in search_dirs:
            match = d / name
            if match.exists():
                _add(match, desc)
                break
            # Maximal 1 Ebene tief suchen
            for f in d.glob(f"*/{name}"):
                _add(f, desc)
                break

    # Config (now in .rubin_cache)
    cfg_file = WORK_DIR / ".rubin_cache" / "config_ui.yml"
    if cfg_file.exists():
        _add(cfg_file, "Verwendete Konfiguration")

    # Modelle, CSVs, Parquets in output/
    output_dir = ROOT / "output"
    if output_dir.exists():
        for ext, desc in [("*.pkl", "Modell"), ("*.csv", "CSV"), ("*.parquet", "Parquet")]:
            for f in output_dir.rglob(ext):
                _add(f, desc)

    # Bundles
    bundle_dir = ROOT / "bundles"
    if bundle_dir.exists():
        for f in bundle_dir.glob("*.zip"):
            _add(f, "Bundle-Archiv")
    for f in (ROOT / "output").glob("bundle*.zip") if output_dir.exists() else []:
        _add(f, "Bundle-Archiv")

    # Nach Typ sortieren: Reports zuerst, dann Modelle, dann Daten
    type_order = {"HTML-Report": 0, "Evaluationsmetriken": 1, "Verwendete Konfiguration": 2,
                  "Champion & Challenger": 3, "Modell": 4, "Bundle-Archiv": 5, "CSV": 6, "Parquet": 7}
    files.sort(key=lambda f: (type_order.get(f["desc"], 99), f["name"]))
    return files


@bp.route("/api/results")
def list_results():
    return jsonify({"files": _scan_result_files()})


@bp.route("/api/download/<path:filepath>")
def download_file(filepath):
    full = (ROOT / filepath).resolve()
    # Path-Traversal-Schutz ZUERST (vor exists-Check, verhindert Info-Leak)
    if not str(full).startswith(str(ROOT.resolve())):
        log.warning("Path-Traversal-Versuch blockiert: %s", filepath)
        abort(403)
    if not full.exists() or not full.is_file():
        log.warning("Download angefragt, nicht gefunden: %s", filepath)
        abort(404)
    log.info("Download: %s", filepath)
    return send_file(str(full), as_attachment=True, download_name=full.name)


@bp.route("/api/view/<path:filepath>")
def view_file(filepath):
    """Liefert eine Datei inline (fuer iframe-Einbettung, kein Download)."""
    full = (ROOT / filepath).resolve()
    if not str(full).startswith(str(ROOT.resolve())):
        log.warning("Path-Traversal-Versuch blockiert (view): %s", filepath)
        abort(403)
    if not full.exists() or not full.is_file():
        abort(404)
    # Mimetype basierend auf Endung
    mimetype = "text/html" if full.suffix.lower() == ".html" else None
    resp = send_file(str(full), as_attachment=False, mimetype=mimetype)
    # Kein Browser-Caching für Reports (sonst wird nach erneutem Lauf der alte angezeigt)
    if full.suffix.lower() == ".html":
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


@bp.route("/api/report")
def get_report():
    """Laed den neuesten HTML-Report und die Metriken."""
    report_path = None
    metrics = None

    # 1. Priorität: .rubin_cache (vom letzten Analyselauf, immer aktuell)
    cache_dir = WORK_DIR / ".rubin_cache"
    cache_report = cache_dir / "analysis_report.html"
    cache_metrics = cache_dir / "uplift_eval_summary.json"
    if cache_report.is_file():
        report_path = cache_report
    if cache_metrics.is_file():
        try:
            metrics = json.loads(cache_metrics.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 2. Fallback: output/ und ROOT durchsuchen
    if report_path is None:
        search_dirs = [ROOT / "output", ROOT]
        for d in search_dirs:
            if not d.exists():
                continue
            candidates = list(d.glob("analysis_report.html")) + list(d.glob("*/analysis_report.html"))
            if candidates:
                report_path = max(candidates, key=lambda p: p.stat().st_mtime)
                break

    if metrics is None:
        search_dirs = [ROOT / "output", ROOT]
        for d in search_dirs:
            if not d.exists():
                continue
            for f in list(d.glob("uplift_eval_summary.json")) + list(d.glob("*/uplift_eval_summary.json")):
                try:
                    metrics = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    pass
                break
            if metrics:
                break

    result = {"status": "done" if report_path else "not_found"}
    if report_path:
        # Cache-Buster: Modification-Timestamp verhindert, dass der Browser einen alten Report cached
        import time
        ts = int(report_path.stat().st_mtime * 1000) if report_path.exists() else int(time.time() * 1000)
        result["report_url"] = f"./api/view/{report_path.relative_to(ROOT)}?t={ts}"
    if metrics:
        result["metrics"] = metrics

    return jsonify(result)


