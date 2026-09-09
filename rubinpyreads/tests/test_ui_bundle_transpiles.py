"""Das UI-Bundle muss mit dem ausgelieferten Babel transpilieren — ein
einziger JSX-Syntaxfehler zerschießt im Browser die gesamte App, und weder
build_ui.py (konkateniert nur) noch der Byte-Sync-Check fangen das. Realer
Vorfall: eine Textarea wurde als zweites Element in einen Ternary-Zweig
eingefügt ("Adjacent JSX elements must be wrapped") — Frontend lud nicht mehr."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node nicht verfügbar")
def test_ui_bundle_transpiles_with_shipped_babel():
    js = (
        'const Babel=require(process.argv[1]);'
        'const src=require("fs").readFileSync(process.argv[2],"utf8");'
        'Babel.transform(src,{presets:["react"]});'
    )
    r = subprocess.run(
        ["node", "-e", js,
         str(ROOT / "app" / "frontend" / "lib" / "babel.min.js"),
         str(ROOT / "app" / "rubin_ui_src.jsx")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"JSX-Syntaxfehler im Bundle: {r.stderr.splitlines()[:3]}"
