#!/usr/bin/env python3
"""Build-Script für die rubin-UI.

Konkateniert alle JSX-Quelldateien aus app/src/ (sortiert nach Dateiname)
zu app/rubin_ui_src.jsx und generiert app/frontend/index.html durch
Einsetzen in die HTML-Shell-Vorlage.

Nutzung:
    python3 app/build_ui.py          # aus dem Repo-Root
    python3 build_ui.py              # aus app/

Wird automatisch von `pixi run app` aufgerufen (siehe app.sh).
"""

from pathlib import Path
import sys

# Pfade relativ zum Script-Verzeichnis (app/)
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
SHELL_HTML = SRC_DIR / "shell.html"
OUT_JSX = SCRIPT_DIR / "rubin_ui_src.jsx"
OUT_HTML = SCRIPT_DIR / "frontend" / "index.html"
PLACEHOLDER = "{{JSX_CONTENT}}"


def build():
    """Hauptroutine: Concat + HTML-Generierung."""
    # 1. Alle .jsx-Dateien aus src/ sammeln (sortiert nach Dateiname)
    jsx_files = sorted(SRC_DIR.glob("*.jsx"))
    if not jsx_files:
        print(f"FEHLER: Keine .jsx-Dateien in {SRC_DIR} gefunden.", file=sys.stderr)
        sys.exit(1)

    # 2. Konkatenieren
    parts = []
    for f in jsx_files:
        content = f.read_text(encoding="utf-8")
        parts.append(content)

    combined = "\n".join(parts)

    # 3. rubin_ui_src.jsx schreiben
    OUT_JSX.write_text(combined, encoding="utf-8")

    # 4. HTML generieren
    if not SHELL_HTML.exists():
        print(f"FEHLER: HTML-Shell {SHELL_HTML} nicht gefunden.", file=sys.stderr)
        sys.exit(1)

    shell = SHELL_HTML.read_text(encoding="utf-8")
    if PLACEHOLDER not in shell:
        print(f"FEHLER: Placeholder '{PLACEHOLDER}' nicht in shell.html.", file=sys.stderr)
        sys.exit(1)

    html = shell.replace(PLACEHOLDER, combined)
    OUT_HTML.write_text(html, encoding="utf-8")

    # 5. Zusammenfassung
    n_files = len(jsx_files)
    n_lines = len(combined.split("\n"))
    print(f"✓ build_ui: {n_files} JSX-Dateien → {n_lines} Zeilen")
    print(f"  → {OUT_JSX.relative_to(SCRIPT_DIR.parent)}")
    print(f"  → {OUT_HTML.relative_to(SCRIPT_DIR.parent)}")
    for f in jsx_files:
        lines = len(f.read_text().split("\n"))
        print(f"    {f.name:35s} {lines:5d}L")


if __name__ == "__main__":
    build()
