#!/usr/bin/env python3
"""Prüft die Drei-Ebenen-Synchronität der rubin-UI.

Die UI existiert in drei per Konvention manuell synchron gehaltenen Ebenen
(siehe docs/app_build.md):

    app/src/*.jsx            (14 Fragmente, Entwicklungs-Quelle)
    app/rubin_ui_src.jsx     (== "\\n".join(sortierte Fragmente))
    app/frontend/index.html  (enthält das Bundle verbatim im babel-Script-Tag)

Drift zwischen den Ebenen ist eine reale Fehlerquelle (drei Kopien!). Dieses
Skript beweist Byte-Konsistenz und eignet sich als Gate vor Commits:

    pixi run ui-sync            # bzw. python scripts/check_ui_sync.py

Exit-Code 0 = synchron, 1 = Drift (mit erster Divergenz-Position).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "app" / "src"
BUNDLE = ROOT / "app" / "rubin_ui_src.jsx"
INDEX = ROOT / "app" / "frontend" / "index.html"


def first_divergence(a: str, b: str) -> str:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            line = a[:i].count("\n") + 1
            return f"Zeichen {i} (Zeile ~{line}): {a[max(0,i-40):i+40]!r} vs. {b[max(0,i-40):i+40]!r}"
    return f"Längen-Differenz: {len(a)} vs. {len(b)} Zeichen (gemeinsamer Präfix identisch)"


def check_transpiles() -> bool:
    """Ebene 0: Die JSX-Quelle muss mit dem ausgelieferten Babel transpilieren.
    Ein einziger JSX-Syntaxfehler (z. B. benachbarte Elemente in einem
    Ternary-Zweig ohne umschließendes Element) zerschießt im Browser das
    GESAMTE Bundle — der Byte-Sync der Ebenen unten fängt das nicht.
    Ohne node wird der Check mit Hinweis übersprungen."""
    import shutil
    import subprocess
    if not shutil.which("node"):
        print("~ Ebene 0 übersprungen (node fehlt) — Transpile-Check nur mit node möglich")
        return True
    js = (
        'const Babel=require(process.argv[1]);'
        'const src=require("fs").readFileSync(process.argv[2],"utf8");'
        'try{Babel.transform(src,{presets:["react"]});}'
        'catch(e){console.error(String(e.message||e));process.exit(1);}'
    )
    r = subprocess.run(
        ["node", "-e", js, str(ROOT / "app" / "frontend" / "lib" / "babel.min.js"), str(BUNDLE)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"✗ Ebene 0: JSX-Syntaxfehler im Bundle — {r.stderr.strip()}")
        return False
    print("✓ Ebene 0: Bundle transpiliert fehlerfrei (Babel preset react)")
    return True


def main() -> int:
    fragments = sorted(SRC_DIR.glob("*.jsx"))
    if not fragments:
        print(f"FEHLER: keine Fragmente unter {SRC_DIR}")
        return 1
    concat = "\n".join(f.read_text(encoding="utf-8") for f in fragments)
    bundle = BUNDLE.read_text(encoding="utf-8")

    ok = check_transpiles()
    if concat == bundle:
        print(f"✓ Ebene 1↔2: rubin_ui_src.jsx == '\\n'.join({len(fragments)} src-Fragmente) — byte-identisch")
    else:
        ok = False
        print("✗ Ebene 1↔2 DRIFT zwischen src/*.jsx und rubin_ui_src.jsx:")
        print("  ", first_divergence(concat, bundle))

    html = INDEX.read_text(encoding="utf-8")
    m = re.search(r'<script[^>]*type=["\']text/babel["\'][^>]*>([\s\S]*?)</script>', html)
    if not m:
        ok = False
        print("✗ Ebene 3: kein text/babel-Script in index.html gefunden")
    elif bundle.strip() in m.group(1):
        print("✓ Ebene 2↔3: index.html enthält das Bundle verbatim")
    else:
        ok = False
        print("✗ Ebene 2↔3 DRIFT: Bundle nicht verbatim in index.html:")
        print("  ", first_divergence(m.group(1).strip(), bundle.strip()))

    print("SYNCHRON ✓" if ok else "DRIFT ✗ — betroffene Ebene(n) angleichen (Quelle: app/src/*.jsx)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
