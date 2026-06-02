"""Frontend-Routen: SPA Index."""
from flask import Blueprint, send_from_directory

from app.state import FRONTEND

bp = Blueprint("frontend", __name__)


@bp.route("/")
def index():
    return send_from_directory(str(FRONTEND), "index.html")
