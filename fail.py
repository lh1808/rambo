# ─────────────────────────────────────────────────────────────────────────────
# pre-commit-Gate für rubin
#
# Zweck: Bugs der Klasse "undefined name" (NameError) und Syntaxfehler werden
#        BEIM COMMIT abgefangen, bevor sie ins Repo gelangen — genau die Klasse,
#        die in der Vergangenheit zu stillen Laufzeitfehlern geführt hat
#        (fehlender Import, falscher Variablenname, f-string-Backslash auf <3.12).
#
# Einrichtung (einmalig pro Klon):
#     pixi run hooks-install        # oder:  pre-commit install
#
# Manuell über alle Dateien:
#     pixi run gate                 # oder:  pre-commit run --all-files
#
# Hinweis Firmennetz: Es wird bewusst KEIN externes Hook-Repo geladen
# (kein astral-sh/ruff-pre-commit). Der Hook nutzt das ruff aus der
# pixi-/venv-Umgebung (language: system) → kein CDN/GitHub-Zugriff nötig.
# ─────────────────────────────────────────────────────────────────────────────
repos:
  - repo: local
    hooks:
      - id: ruff-undefined-names
        name: ruff – undefined names & syntax (F821)
        entry: ruff check --force-exclude --select F821 --target-version py310
        language: system
        types_or: [python, pyi]
        require_serial: false
