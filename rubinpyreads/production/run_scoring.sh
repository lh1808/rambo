#!/usr/bin/env bash
# ============================================================================
# Domino-Job: rubin Production-Scoring (generisch — Use-Cases über YAML-Configs)
#
# Der Use-Case steckt NICHT im Skript, sondern in den Scoring-Configs
# (production/scoring_<usecase>.yml): SCORING_CONFIGS listet, was gefahren
# wird (Default: scoring_ph.yml). Jede Config deklariert ihren Transport
# selbst über den Top-Level-Key `runner:` (file = Datei-Flow/XPT, Default;
# saspy = SAS-Library-Flow); run_scoring.py liest den Key selbst und delegiert
# saspy-Configs intern — gemischte Jobs möglich, die Shell kennt EINEN Einstieg.
#
# Master dieser Datei liegt VERSIONIERT im Repo (production/run_scoring.sh).
# Die ausgeführte Kopie muss auf dem Domino File System liegen (Job-Einstieg),
# z. B. /mnt/Production/<usecase>/run_scoring.sh — nach Änderungen im
# Repo dorthin synchronisieren.
#
# Ablauf: Repo frisch clonen (pinnbarer Stand) → pixi-prod-Env aus dem Lockfile
# installieren (kein Re-Solve!) → run_scoring.py pro Use-Case-Config fahren.
# XPT wird am Zielpfad überschrieben, Monitoring-JSON versioniert daneben.
#
# MEHRERE KAUSALSCORES: SCORING_CONFIGS nimmt eine Leerzeichen-Liste von
# Configs; jeder Score läuft in einem EIGENEN Python-Prozess. Damit gibt das
# Betriebssystem zwischen den Scores garantiert sämtlichen Speicher frei
# (Modelle, Frames, Fragmentierung) — kein OOM-Risiko durch akkumulierte
# Zustände über mehrere Läufe hinweg. Unterschiedliche Feature-Teilmengen
# pro Score sind unproblematisch: jeder Lauf liest über sein Bundle nur die
# benötigten Spalten (input.pull_only_needed_columns, Default an).
#
# Service-User: Es wird KEINE Git-Identity gesetzt (nur nötig für Commits).
# Der Job braucht ausschließlich den SSH-Deploy-Key des Service-Users unter
# ~/.ssh/id_rsa mit Lese-Recht auf das Repo.
# ============================================================================
set -euo pipefail

# ── Parameter (per Env überschreibbar, sonst Defaults) ──────────────────────
GIT_URL="${GIT_URL:-ssh://tfs.lan.huk-coburg.de:22/web/DefaultCollection/GIT_Projects/_git/da-hf1-rubin}"
GIT_REF="${GIT_REF:-main}"                 # Für reproduzierbare Prod-Läufe pinnen:
                                           # Tag (empfohlen — Shallow-Clone, schnell)
                                           # oder Commit-SHA (Fallback unten: voller
                                           # Clone + Checkout, funktioniert, langsamer).
# Ein oder mehrere Configs (Leerzeichen-getrennt — Pfade dürfen daher selbst
# keine Leerzeichen enthalten). SCORING_CONFIG (Singular) bleibt als Alias
# für den Ein-Score-Fall nutzbar.
SCORING_CONFIGS="${SCORING_CONFIGS:-${SCORING_CONFIG:-production/scoring_ph.yml}}"
# Fehlerpolitik bei mehreren Scores: 0 = Abbruch beim ersten Fehler (Default),
# 1 = alle Scores versuchen, am Ende Exit ≠ 0 wenn mindestens einer fehlschlug.
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
WORKDIR="${WORKDIR:-/home/ubuntu/rubin_scoring}"
PIXI_ENV="${PIXI_ENV:-prod}"               # schlankes Scoring-Env (gleiche solve-group wie default)
# Transport pro Config: Der Top-Level-Key `runner:` in der jeweiligen YAML
# entscheidet (file = Datei-Flow/XPT, Default; saspy = SAS-Library-Flow).
# Das Routing übernimmt run_scoring.py selbst (robustes YAML statt
# Shell-Parsing) — die Shell kennt nur EINEN Einstieg.
# Trockenlauf/Test: SKIP_SETUP=1 überspringt SSH/Clone/Pixi-Install und führt
# die Scoring-Schleife im aktuellen Verzeichnis aus; RUN_CMD ersetzt dabei den
# Python-Aufruf (Default: pixi-Env).
SKIP_SETUP="${SKIP_SETUP:-0}"
RUN_CMD="${RUN_CMD:-}"

START_TS=$(date +%s)
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== rubin Production-Scoring: Start (ref=${GIT_REF}, env=${PIXI_ENV}) ==="

if [[ "${SKIP_SETUP}" != "1" ]]; then

# ── Reproduzierbarkeits-Guard ───────────────────────────────────────────────
# Branch-Refs ziehen bei jedem Lauf stillschweigend den neuesten Commit — der
# teuerste stille Fehler in Prod. Default: laute Warnung. In produktiven
# Job-Definitionen REQUIRE_PINNED_REF=1 setzen → harter Abbruch, bis GIT_REF
# auf ein annotiertes Tag (empfohlen) oder eine Commit-SHA gepinnt ist.
case "${GIT_REF}" in
  main|master|develop|HEAD)
    if [[ "${REQUIRE_PINNED_REF:-0}" == "1" ]]; then
      log "FEHLER: GIT_REF='${GIT_REF}' ist ein Branch (nicht reproduzierbar) und REQUIRE_PINNED_REF=1 ist gesetzt."
      log "        GIT_REF auf ein annotiertes Tag oder eine Commit-SHA pinnen."
      exit 2
    fi
    log "WARNUNG: GIT_REF='${GIT_REF}' ist ein Branch — jeder Lauf zieht den jeweils neuesten Commit."
    log "         Für reproduzierbare Prod-Läufe GIT_REF auf Tag/SHA pinnen (und REQUIRE_PINNED_REF=1 setzen)."
    ;;
esac

# ── SSH-Key des Service-Users ───────────────────────────────────────────────
log "--- SSH vorbereiten ---"
# Rechte für alle vorhandenen privaten Keys fixen (Service-User-Key kann auch
# id_ed25519 o. ä. heißen); fehlt jeder Key, liefert der Clone die klare Meldung.
found_key=0
for k in ~/.ssh/id_rsa ~/.ssh/id_ed25519 ~/.ssh/id_ecdsa; do
  [[ -f "$k" ]] && { chmod 600 "$k"; found_key=1; log "SSH-Key: $k"; }
done
[[ "$found_key" -eq 0 ]] && log "WARNUNG: kein SSH-Key unter ~/.ssh gefunden — Clone wird voraussichtlich scheitern."
git --version

# ── TLS für pixi/uv (PyPI-Deps: catboost, pyreadstat, editierbares rubin) ───
# Auf Job-Executors ist die .devboxrc nicht garantiert geladen → explizit setzen.
export PIXI_TLS_ROOT_CERTS="all"
export UV_NATIVE_TLS="true"

# ── Repo frisch clonen (kein Stale-State vom vorherigen Lauf) ───────────────
log "--- Git Clone (${GIT_REF}) ---"
rm -rf "${WORKDIR}"
git clone --depth 1 --branch "${GIT_REF}" "${GIT_URL}" "${WORKDIR}" \
  || { # --branch akzeptiert nur Branches/Tags; für Commit-SHAs: voller Clone + Checkout
       log "Shallow-Clone auf '${GIT_REF}' fehlgeschlagen — versuche vollen Clone + Checkout (Commit-SHA?)";
       rm -rf "${WORKDIR}";
       git clone "${GIT_URL}" "${WORKDIR}";
       git -C "${WORKDIR}" checkout --detach "${GIT_REF}"; }
cd "${WORKDIR}"
log "Repo-Stand: $(git rev-parse --short HEAD) ($(git log -1 --format=%cd --date=short))"

# ── pixi-Environment aus dem Lockfile (deterministisch, kein Solver) ────────
log "--- Pixi Install (-e ${PIXI_ENV}) ---"
pixi --version
if [[ -f pixi.lock ]]; then
  # Lockfile vorhanden → exakt diesen Stand installieren, kein Solver-Lauf.
  log "pixi.lock gefunden → deterministische Installation (--frozen)."
  pixi install --frozen -e "${PIXI_ENV}"
else
  # Kein Lockfile im Repo (aktueller Stand): pixi löst anhand der Version-Pins.
  # EMPFEHLUNG für volle Reproduzierbarkeit: pixi.lock committen — dann greift
  # automatisch der --frozen-Pfad oben.
  log "WARNUNG: kein pixi.lock im Repo — pixi löst die Versionen zur Laufzeit."
  pixi install -e "${PIXI_ENV}"
fi
# Hinweis: KEIN 'pixi add' auf dem Executor — alle Abhängigkeiten sind
# Bestandteil der pixi.toml (prod-Feature).

fi  # SKIP_SETUP

# ── Scoring (ein Prozess PRO Config → Speicher wird zwischen Scores frei) ───
read -r -a CONFIGS <<< "${SCORING_CONFIGS}"
N_CONFIGS=${#CONFIGS[@]}
log "--- Scoring: ${N_CONFIGS} Config(s): ${SCORING_CONFIGS} ---"

FAILED=()
for CFG in "${CONFIGS[@]}"; do
  SCORE_TS=$(date +%s)
  log "--- Score starten: ${CFG} ---"
  if ${RUN_CMD:-pixi run -e "${PIXI_ENV}" python} production/run_scoring.py --config "${CFG}"; then
    log "--- Score OK: ${CFG} ($(( $(date +%s) - SCORE_TS ))s) ---"
  else
    RC=$?
    log "--- Score FEHLGESCHLAGEN (rc=${RC}): ${CFG} ---"
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      FAILED+=("${CFG}")
    else
      exit "${RC}"
    fi
  fi
done

ELAPSED=$(( $(date +%s) - START_TS ))
if [[ ${#FAILED[@]} -gt 0 ]]; then
  log "=== Fertig in $((ELAPSED / 60))m $((ELAPSED % 60))s — ${#FAILED[@]}/${N_CONFIGS} fehlgeschlagen: ${FAILED[*]} ==="
  exit 1
fi
log "=== Fertig in $((ELAPSED / 60))m $((ELAPSED % 60))s — ${N_CONFIGS}/${N_CONFIGS} Scores OK ==="
# Exit-Code ≠ 0 bei jedem Fehler → Domino markiert den Job rot.
