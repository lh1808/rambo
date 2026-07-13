"""Config-Routen: YAML speichern und importieren."""
from flask import Blueprint, jsonify, request, abort
from werkzeug.utils import secure_filename

from app.state import log, WORK_DIR

bp = Blueprint("config", __name__)


@bp.route("/api/save-config", methods=["POST"])
def save_config():
    data = request.get_json(silent=True) or {}
    yaml_text = data.get("yaml", "")
    if not yaml_text or not yaml_text.strip():
        return jsonify({"status": "error", "message": "Keine Konfiguration gesendet."}), 400
    filename = data.get("filename", "config.yml")
    safe = secure_filename(filename)
    if not safe or not safe.endswith((".yml", ".yaml")):
        safe = "config.yml"
    out_dir = WORK_DIR / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / safe
    if not str(out.resolve()).startswith(str(out_dir.resolve())):
        log.warning("Path-Traversal-Versuch blockiert: %s", filename)
        abort(403)
    out.write_text(yaml_text, encoding="utf-8")
    log.info("Konfiguration gespeichert: %s", out)
    return jsonify({"status": "done", "message": f"Gespeichert: runs/configs/{safe}", "path": str(out)})


@bp.route("/api/import-config", methods=["POST"])
def import_config():
    data = request.get_json(silent=True) or {}
    yaml_text = data.get("yaml_text", "")
    if not yaml_text or not yaml_text.strip():
        return jsonify({"status": "error", "message": "Kein YAML-Text gesendet."}), 400
    _import_dir = WORK_DIR / ".rubin_cache"
    _import_dir.mkdir(parents=True, exist_ok=True)
    out = _import_dir / "config_imported.yml"
    out.write_text(yaml_text, encoding="utf-8")
    log.info("Konfiguration importiert: %s", out)
    return jsonify({"status": "done", "message": f"Importiert: {out.name}"})
