#!/usr/bin/env bash
# ============================================================================
# Domino-Job: rubin Production-Scoring (generisch — Use-Cases über YAML-Configs)
#
# Der Use-Case steckt NICHT im Skript, sondern in den Scoring-Configs
# (production/scoring_<usecase>.yml): SCORING_CONFIGS listet, was gefahren
# wird (Default: scoring_ph.yml). Jede Config deklariert ihren Transport
# selbst über den Top-Level-Key `runner:` (file = Datei-Flow/XPT, Default;
# saspy = SAS-Library-Flow) — gemischte Jobs sind damit möglich.
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
GIT_REF="${GIT_REF:-main}"                 # Für reproduzierbare Prod-Läufe: Tag/Commit pinnen!
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
# wählt den Einstieg (file → run_scoring.py, saspy → run_scoring_saspy.py).
# RUNNER_SCRIPT (Env) erzwingt einen Einstieg für ALLE Configs (Override).
RUNNER_SCRIPT="${RUNNER_SCRIPT:-}"

runner_for_config() {
  # Liest den Top-Level-Key `runner:` (nur Spaltenanfang → keine Treffer in
  # verschachtelten Sektionen). Default: Datei-Flow.
  local cfg="$1"
  if [[ -n "${RUNNER_SCRIPT}" ]]; then echo "${RUNNER_SCRIPT}"; return; fi
  local val
  val=$(grep -E '^runner:' "${cfg}" 2>/dev/null | head -1 | sed -E 's/^runner:[[:space:]]*//; s/[[:space:]]*(#.*)?$//')
  case "${val}" in
    saspy) echo "production/run_scoring_saspy.py" ;;
    ""|file) echo "production/run_scoring.py" ;;
    *) echo "UNBEKANNT:${val}" ;;
  esac
}
# Optionale Overrides (leer = Werte aus der YAML-Config):
BUNDLE_OVERRIDE="${BUNDLE_OVERRIDE:-}"
INPUT_OVERRIDE="${INPUT_OVERRIDE:-}"
OUTPUT_OVERRIDE="${OUTPUT_OVERRIDE:-}"

# Trockenlauf/Test: SKIP_SETUP=1 überspringt SSH/Clone/Pixi-Install und führt
# die Scoring-Schleife im aktuellen Verzeichnis aus; RUN_CMD ersetzt dabei den
# Python-Aufruf (Default: pixi-Env).
SKIP_SETUP="${SKIP_SETUP:-0}"
RUN_CMD="${RUN_CMD:-}"

START_TS=$(date +%s)
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== rubin Production-Scoring: Start (ref=${GIT_REF}, env=${PIXI_ENV}) ==="

if [[ "${SKIP_SETUP}" != "1" ]]; then

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

# Overrides gelten nur im Ein-Score-Fall mit Datei-Flow — bei mehreren Configs
# wären sie mehrdeutig, und der saspy-Runner kennt --input/--output nicht.
if [[ ${N_CONFIGS} -gt 1 && ( -n "${BUNDLE_OVERRIDE}" || -n "${INPUT_OVERRIDE}" || -n "${OUTPUT_OVERRIDE}" ) ]]; then
  log "FEHLER: BUNDLE/INPUT/OUTPUT_OVERRIDE sind bei mehreren Configs nicht erlaubt."
  exit 2
fi

FAILED=()
for CFG in "${CONFIGS[@]}"; do
  SCORE_TS=$(date +%s)
  RUNNER=$(runner_for_config "${CFG}")
  if [[ "${RUNNER}" == UNBEKANNT:* ]]; then
    log "FEHLER: unbekannter runner '${RUNNER#UNBEKANNT:}' in ${CFG} (erlaubt: file, saspy)."
    exit 2
  fi
  if [[ "${RUNNER}" == *saspy* && ( -n "${INPUT_OVERRIDE}" || -n "${OUTPUT_OVERRIDE}" ) ]]; then
    log "FEHLER: INPUT/OUTPUT_OVERRIDE gelten nur für den Datei-Runner — der"
    log "        saspy-Runner adressiert Tabellen über die Config (bzw. --table-in/--table-out)."
    exit 2
  fi
  log "--- Score starten: ${CFG} (${RUNNER}) ---"
  ARGS=(--config "${CFG}")
  [[ -n "${BUNDLE_OVERRIDE}" ]] && ARGS+=(--bundle "${BUNDLE_OVERRIDE}")
  [[ -n "${INPUT_OVERRIDE}"  ]] && ARGS+=(--input  "${INPUT_OVERRIDE}")
  [[ -n "${OUTPUT_OVERRIDE}" ]] && ARGS+=(--output "${OUTPUT_OVERRIDE}")

  if ${RUN_CMD:-pixi run -e "${PIXI_ENV}" python} "${RUNNER}" "${ARGS[@]}"; then
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
