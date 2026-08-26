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
# Service-User: Der Job braucht den SSH-Deploy-Key des Service-Users unter
# ~/.ssh (id_rsa/id_ed25519/id_ecdsa) mit Lese-Recht auf das Repo. Eine
# Git-Identity ist fürs Klonen nicht nötig; wo das Umgebungs-Setup sie
# erwartet, kann sie per --git-email/--git-name gesetzt werden
# (führt git config --global aus — bewährtes TFS-Vorgehen).
# ============================================================================
set -euo pipefail

# ── Parameter-Defaults (Steuerung AUSSCHLIESSLICH über Argumente, s. unten) ─
GIT_URL="ssh://tfs.lan.huk-coburg.de:22/web/DefaultCollection/GIT_Projects/_git/da-hf1-rubin"
GIT_REF="main"                 # Für reproduzierbare Prod-Läufe pinnen:
                                           # Tag (empfohlen — Shallow-Clone, schnell)
                                           # oder Commit-SHA (Fallback unten: voller
                                           # Clone + Checkout, funktioniert, langsamer).
# Ein oder mehrere Configs (--configs "a b" Leerzeichen-getrennt — Pfade
# dürfen daher selbst keine Leerzeichen enthalten; alternativ --config je Datei).
SCORING_CONFIGS="production/scoring_ph.yml"
# Fehlerpolitik bei mehreren Scores: 0 = Abbruch beim ersten Fehler (Default),
# 1 = alle Scores versuchen, am Ende Exit ≠ 0 wenn mindestens einer fehlschlug.
CONTINUE_ON_ERROR=0
WORKDIR="/home/ubuntu/rubin_scoring"
PIXI_ENV="prod"               # schlankes Scoring-Env (gleiche solve-group wie default)
# Transport pro Config: Der Top-Level-Key `runner:` in der jeweiligen YAML
# entscheidet (file = Datei-Flow/XPT, Default; saspy = SAS-Library-Flow).
# Das Routing übernimmt run_scoring.py selbst (robustes YAML statt
# Shell-Parsing) — die Shell kennt nur EINEN Einstieg.
# Trockenlauf/Test: SKIP_SETUP=1 überspringt SSH/Clone/Pixi-Install und führt
# die Scoring-Schleife im aktuellen Verzeichnis aus; RUN_CMD ersetzt dabei den
# Python-Aufruf (Default: pixi-Env).
SKIP_SETUP=0
RUN_CMD=""
REQUIRE_PINNED_REF=0
# Optionale Git-Identity (leer = nicht setzen; fürs Klonen nicht erforderlich):
GIT_USER_EMAIL=""
GIT_USER_NAME=""

# ── Job-Datei = das EINZIGE Steuerungs-Interface (Env wird nicht gelesen) ───
# Aufruf: bash production/run_scoring.sh production/jobs/job_<uc>.conf
# Die beiden --Flags unten sind reines Testwerkzeug (Trockenlauf).
usage() {
  cat <<'USAGE'
Aufruf: bash production/run_scoring.sh <JOB_CONF> [--skip-setup] [--run-cmd "<cmd>"]

  JOB_CONF        Pflicht: Pfad zur Job-Datei (KEY="wert"-Zeilen; Vorlage:
                  production/jobs/job_ph.conf). SÄMTLICHE Job-Parameter
                  (SCORING_CONFIGS, GIT_REF, REQUIRE_PINNED_REF,
                  CONTINUE_ON_ERROR, optional GIT_USER_EMAIL/GIT_USER_NAME,
                  WORKDIR, GIT_URL, PIXI_ENV, SKIP_SETUP, RUN_CMD) werden
                  dort gesetzt — es gibt bewusst keinen zweiten Weg.
  --skip-setup    Test: Schleife ohne Clone/Pixi im aktuellen Verzeichnis
  --run-cmd       Test: Python-Aufruf ersetzen (z. B. "python3")
  -h | --help     Diese Hilfe

In der Job-Datei nicht gesetzte Schlüssel behalten die Skript-Defaults.
USAGE
}
# ── Job-Conf-Datei (primäres Interface) ─────────────────────────────────────
# Erstes Argument ohne führendes "--" = Pfad zu einer Job-Datei in flacher
# KEY="wert"-Syntax (Vorlage: production/jobs/job_ph.conf). Sie wird per
# source geladen — kein YAML-Parser nötig, Bash liest sie nativ. Erlaubt sind
# ausschließlich Zuweisungen der bekannten Parameter, Kommentare und
# Leerzeilen; alles andere bricht ab (Schutz vor Tippfehlern/Injection).
JOB_CONF=""
if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage; exit 0
fi
if [[ $# -eq 0 || "$1" == --* ]]; then
  echo "FEHLER: Job-Datei fehlt. Aufruf: bash production/run_scoring.sh <production/jobs/job_<uc>.conf> [--skip-setup --run-cmd \"python\"]"
  echo "        (--help zeigt Details; Vorlage: production/jobs/job_ph.conf)"
  exit 2
fi
JOB_CONF="$1"; shift
if [[ ! -f "${JOB_CONF}" ]]; then
  echo "FEHLER: Job-Conf nicht gefunden: ${JOB_CONF}"; exit 2
fi
# Absolute Pfade für den späteren Drift-Check sichern (vor jedem cd):
JOB_CONF="$(realpath "${JOB_CONF}")"
_SELF="$(realpath "${BASH_SOURCE[0]}")"
_bad=$(grep -Ev '^[[:space:]]*(#|$|(SCORING_CONFIGS|GIT_REF|GIT_URL|WORKDIR|PIXI_ENV|CONTINUE_ON_ERROR|REQUIRE_PINNED_REF|GIT_USER_EMAIL|GIT_USER_NAME|SKIP_SETUP|RUN_CMD)=)' "${JOB_CONF}" || true)
if [[ -n "${_bad}" ]]; then
  echo "FEHLER: Job-Conf ${JOB_CONF} enthält unzulässige Zeilen (erlaubt: KEY=\"wert\" der bekannten Parameter, Kommentare):"
  echo "${_bad}"
  exit 2
fi
# shellcheck disable=SC1090
source "${JOB_CONF}"

# ── Test-Flags (einzige Optionen; alles andere gehört in die Job-Datei) ─────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup)        SKIP_SETUP=1; shift ;;
    --run-cmd)           RUN_CMD="$2"; shift 2 ;;
    -h|--help)           usage; exit 0 ;;
    *) echo "Unbekannte Option: $1 — alle Job-Parameter gehören in die Job-Datei (siehe --help)."; exit 2 ;;
  esac
done

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
    if [[ "${REQUIRE_PINNED_REF}" == "1" ]]; then
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

# ── Git-Identity (optional — bewährtes TFS-Umgebungs-Muster) ────────────────
# Fürs Klonen nicht erforderlich; wird nur gesetzt, wenn per Argument/Env
# vorgegeben (--git-email/--git-name bzw. GIT_USER_EMAIL/GIT_USER_NAME).
if [[ -n "${GIT_USER_EMAIL}" || -n "${GIT_USER_NAME}" ]]; then
  log "--- Git-Identity setzen ---"
  [[ -n "${GIT_USER_EMAIL}" ]] && git config --global user.email "${GIT_USER_EMAIL}"
  [[ -n "${GIT_USER_NAME}"  ]] && git config --global user.name  "${GIT_USER_NAME}"
fi

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

# ── Drift-Check: FS-Kopien gegen den geklonten Repo-Master ──────────────────
# run_scoring.sh und job_<uc>.conf sind die EINZIGEN doppelt abliegenden
# Dateien (Bootstrap-Kopien am Startort; Master im Repo). Eine veraltete
# Kopie liefe sonst still mit alter Logik — hier wird laut gewarnt.
check_copy_drift() {
  local running="$1" master="$2" label="$3"
  [[ -f "${master}" ]] || return 0
  if ! cmp -s "${running}" "${master}"; then
    log "WARNUNG: ${label} weicht vom Repo-Master (${GIT_REF}) ab — FS-Kopie bitte synchronisieren:"
    log "         laufend: ${running}"
    log "         Master:  ${master}"
  fi
}
check_copy_drift "${_SELF}" "${WORKDIR}/production/run_scoring.sh" "Gestartetes Skript"
check_copy_drift "${JOB_CONF}" "${WORKDIR}/production/jobs/$(basename "${JOB_CONF}")" "Job-Datei $(basename "${JOB_CONF}")"
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
