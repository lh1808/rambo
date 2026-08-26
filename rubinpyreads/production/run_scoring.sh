#!/usr/bin/env bash
# ============================================================================
# Domino-Job: rubin Production-Scoring (generisch — Use-Cases über YAML-Configs)
#
# Aufruf:
#   bash production/run_scoring.sh production/jobs/job_<usecase>.conf
#   (+ optional --skip-setup --run-cmd "python" für lokale Trockenläufe)
#
# Der Use-Case steckt NICHT im Skript: Die Job-Datei (KEY="wert", einziges
# Steuerungs-Interface — Env wird nicht gelesen) sagt, WAS gefahren wird;
# jede Scoring-Config (production/scoring_<usecase>.yml) beschreibt ihren
# Score selbst, inklusive Transport über den Top-Level-Key `runner:`
# (file = Datei-Flow/XPT, Default; saspy = SAS-Library-Flow). Das Routing
# übernimmt run_scoring.py — die Shell kennt EINEN Einstieg, gemischte
# Jobs sind möglich.
#
# Ablauf: Job-Datei lesen → Repo frisch clonen (pinnbarer Stand) → pixi-Env
# aus dem Lockfile installieren (kein Re-Solve) → run_scoring.py pro Config
# in einem EIGENEN Python-Prozess fahren (das OS gibt zwischen den Scores
# garantiert allen Speicher frei — kein OOM durch akkumulierte Zustände).
# XPT wird am Zielpfad überschrieben, Monitoring-JSON versioniert daneben.
#
# Ablage: Master dieser Datei und der Job-Datei liegen VERSIONIERT im Repo;
# die ausgeführten Kopien liegen zusammen am Startort auf dem Domino File
# System (z. B. /mnt/Production/<usecase>/) — nach Repo-Änderungen
# synchronisieren. Ein Drift-Check nach dem Clone warnt bei Abweichungen.
#
# Service-User: braucht nur seinen SSH-Deploy-Key unter ~/.ssh
# (id_rsa/id_ed25519/id_ecdsa) mit Lese-Recht auf das Repo. Eine Git-Identity
# ist fürs Klonen nicht nötig; wo das Umgebungs-Setup sie erwartet, in der
# Job-Datei GIT_USER_EMAIL/GIT_USER_NAME setzen (→ git config --global,
# bewährtes TFS-Vorgehen).
# ============================================================================
set -euo pipefail

# ── Defaults (die Job-Datei überschreibt; nicht Gesetztes bleibt so) ────────
# Job-Parameter — die drei Angaben, die ein Job typischerweise setzt:
SCORING_CONFIGS="production/scoring_ph.yml"  # Leerzeichen-Liste (Pfade ohne Leerzeichen)
GIT_REF="main"          # Für Prod pinnen: Tag (empfohlen, Shallow-Clone) oder Commit-SHA
CONTINUE_ON_ERROR=0     # 1 = alle Scores versuchen, Exit 1 bei Teilfehlern (Default: Fail-Fast)
REQUIRE_PINNED_REF=0    # 1 = Abbruch statt Warnung, wenn GIT_REF ein Branch ist (Prod-Empfehlung)

# Umgebungskonstanten — selten anzufassen:
GIT_URL="ssh://tfs.lan.huk-coburg.de:22/web/DefaultCollection/GIT_Projects/_git/da-hf1-rubin"
WORKDIR="/home/ubuntu/rubin_scoring"
PIXI_ENV="prod"         # schlankes Scoring-Env (gleiche solve-group wie default)

# Optionale Git-Identity (leer = nicht setzen; fürs Klonen nicht erforderlich):
GIT_USER_EMAIL=""
GIT_USER_NAME=""

# Testwerkzeug (üblich als Flags --skip-setup / --run-cmd, s. usage):
SKIP_SETUP=0            # 1 = ohne SSH/Clone/Pixi im aktuellen Verzeichnis scoren
RUN_CMD=""              # ersetzt den Python-Aufruf, z. B. "python3" (leer = pixi-Env)

# ── Hilfsfunktionen ─────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

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

# Drift-Check: run_scoring.sh und job_<uc>.conf sind die EINZIGEN doppelt
# abliegenden Dateien (Bootstrap-Kopien am Startort; Master im Repo). Eine
# veraltete Kopie liefe sonst still mit alter Logik — hier wird laut gewarnt.
check_copy_drift() {
  local running="$1" master="$2" label="$3"
  [[ -f "${master}" ]] || return 0
  if ! cmp -s "${running}" "${master}"; then
    log "WARNUNG: ${label} weicht vom Repo-Master (${GIT_REF}) ab — FS-Kopie bitte synchronisieren:"
    log "         laufend: ${running}"
    log "         Master:  ${master}"
  fi
}

# ── Job-Datei (Pflicht) und Test-Flags einlesen ─────────────────────────────
# Die Job-Datei wird per source geladen — kein YAML-Parser nötig, Bash liest
# sie nativ. Erlaubt sind ausschließlich Zuweisungen der bekannten Parameter,
# Kommentare und Leerzeilen; alles andere bricht ab (Tippfehler-/Injection-Schutz).
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup)  SKIP_SETUP=1; shift ;;
    --run-cmd)     RUN_CMD="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unbekannte Option: $1 — alle Job-Parameter gehören in die Job-Datei (siehe --help)."; exit 2 ;;
  esac
done

# ── Setup: Guard → SSH → Identity → Clone → Drift-Check → pixi ──────────────
setup_environment() {
  # Reproduzierbarkeits-Guard: Branch-Refs ziehen bei jedem Lauf stillschweigend
  # den neuesten Commit — der teuerste stille Fehler in Prod. Default: laute
  # Warnung; mit REQUIRE_PINNED_REF=1 (Prod-Empfehlung) harter Abbruch.
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

  # SSH-Key des Service-Users: Rechte aller vorhandenen privaten Keys fixen
  # (kann auch id_ed25519 o. ä. heißen); fehlt jeder Key, liefert der Clone
  # die klare Meldung.
  log "--- SSH vorbereiten ---"
  local found_key=0 k
  for k in ~/.ssh/id_rsa ~/.ssh/id_ed25519 ~/.ssh/id_ecdsa; do
    [[ -f "$k" ]] && { chmod 600 "$k"; found_key=1; log "SSH-Key: $k"; }
  done
  [[ "$found_key" -eq 0 ]] && log "WARNUNG: kein SSH-Key unter ~/.ssh gefunden — Clone wird voraussichtlich scheitern."
  git --version

  # Optionale Git-Identity (nur wenn in der Job-Datei gesetzt):
  if [[ -n "${GIT_USER_EMAIL}" || -n "${GIT_USER_NAME}" ]]; then
    log "--- Git-Identity setzen ---"
    [[ -n "${GIT_USER_EMAIL}" ]] && git config --global user.email "${GIT_USER_EMAIL}"
    [[ -n "${GIT_USER_NAME}"  ]] && git config --global user.name  "${GIT_USER_NAME}"
  fi

  # TLS für pixi/uv (PyPI-Deps: catboost, pyreadstat, editierbares rubin) —
  # auf Job-Executors ist die .devboxrc nicht garantiert geladen:
  export PIXI_TLS_ROOT_CERTS="all"
  export UV_NATIVE_TLS="true"

  # Repo frisch clonen (kein Stale-State vom vorherigen Lauf):
  log "--- Git Clone (${GIT_REF}) ---"
  rm -rf "${WORKDIR}"
  git clone --depth 1 --branch "${GIT_REF}" "${GIT_URL}" "${WORKDIR}" \
    || { # --branch akzeptiert nur Branches/Tags; für Commit-SHAs: voller Clone + Checkout
         log "Shallow-Clone auf '${GIT_REF}' fehlgeschlagen — versuche vollen Clone + Checkout (Commit-SHA?)";
         rm -rf "${WORKDIR}";
         git clone "${GIT_URL}" "${WORKDIR}";
         git -C "${WORKDIR}" checkout --detach "${GIT_REF}"; }
  cd "${WORKDIR}"

  check_copy_drift "${_SELF}" "${WORKDIR}/production/run_scoring.sh" "Gestartetes Skript"
  check_copy_drift "${JOB_CONF}" "${WORKDIR}/production/jobs/$(basename "${JOB_CONF}")" "Job-Datei $(basename "${JOB_CONF}")"
  log "Repo-Stand: $(git rev-parse --short HEAD) ($(git log -1 --format=%cd --date=short))"

  # pixi-Environment aus dem Lockfile (deterministisch, kein Solver-Lauf).
  # KEIN 'pixi add' auf dem Executor — alle Abhängigkeiten stehen in der
  # pixi.toml (prod-Feature).
  log "--- Pixi Install (-e ${PIXI_ENV}) ---"
  pixi --version
  if [[ -f pixi.lock ]]; then
    log "pixi.lock gefunden → deterministische Installation (--frozen)."
    pixi install --frozen -e "${PIXI_ENV}"
  else
    log "WARNUNG: kein pixi.lock im Repo — pixi löst die Versionen zur Laufzeit."
    pixi install -e "${PIXI_ENV}"
  fi
}

# ── Ablauf ──────────────────────────────────────────────────────────────────
START_TS=$(date +%s)
log "=== rubin Production-Scoring: Start (ref=${GIT_REF}, env=${PIXI_ENV}) ==="

if [[ "${SKIP_SETUP}" != "1" ]]; then
  setup_environment
fi

# Ein Prozess PRO Config → Speicher wird zwischen den Scores vollständig frei.
# ${RUN_CMD:-…}: greift auch beim Leerstring (Default RUN_CMD="") → pixi-Env.
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
