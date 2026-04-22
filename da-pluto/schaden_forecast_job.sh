#!/bin/bash

# Beende bei jedem Fehler
set -e

# ============================================================================
# Config-Auswahl
# ============================================================================
# Standardmäßig wird config.production.yaml verwendet. Über die Env-Variable
# PLUTO_PROFILE kann ein anderes Profil gewählt werden:
#
#   PLUTO_PROFILE=smoke   ./schaden_forecast_job.sh   # Schnelltest
#   PLUTO_PROFILE=tuning  ./schaden_forecast_job.sh   # Hyperparametersuche
#   PLUTO_PROFILE=tuned   ./schaden_forecast_job.sh   # Eingefrorene Parameter
#   ./schaden_forecast_job.sh                          # Produktion (Default)
#
# Alternativ kann ein beliebiger Pfad direkt gesetzt werden:
#   PLUTO_CONFIG=/pfad/zu/config.yaml ./schaden_forecast_job.sh
# ============================================================================

# ============================================================================
# Persistentes Runs-Verzeichnis (Domino)
# ============================================================================
# In Domino Scheduled Runs werden alle Dateien unter /mnt/ nach Abschluss
# automatisch ins Projekt zurück synchronisiert. Das Runs-Verzeichnis
# liegt daher DIREKT unter /mnt/ (nicht im geklonten Repo, das bei jedem
# Lauf gelöscht wird).
#
# /mnt/
# ├── runs/                          ← PERSISTENT (wächst über Läufe)
# │   ├── metrics_history.csv
# │   ├── retrospective_accuracy.csv
# │   ├── tuned_h13_model.yaml
# │   ├── 2025-04-21T08-30_h13_model/
# │   └── ...
# ├── da-pluto-timeseries/           ← wird bei jedem Lauf gelöscht + neu geklont
# │   ├── forecasting/
# │   ├── pluto_forecast_job.py
# │   └── ...
# └── ferien-api-data-main/          ← Ferien-ICS-Dateien
# ============================================================================

PROFILE="${PLUTO_PROFILE:-production}"
export PLUTO_RUNS_DIR="${PLUTO_RUNS_DIR:-/mnt/runs}"

echo "Setup SSH..."
# Sichere Berechtigungen für SSH-Key
chmod 400 ~/.ssh/id_rsa
# Add TFS zu known_hosts, falls nicht vorhanden
if ! grep -q "tfs" ~/.ssh/known_hosts 2>/dev/null; then
  ssh-keyscan -p 22 tfs >> ~/.ssh/known_hosts
fi

# (Optional) Git SSL prüfen deaktivieren
git config --global http.sslVerify false || true

# Repository-Variablen
REPO_DIR="da-pluto-timeseries"
REPO_URL="ssh://tfs:22/web/DefaultCollection/GIT_Projects/_git/da-pluto-timeseries"
BRANCH="holidays_winsorizer"

echo "Lösche vorhandenes Repo-Verzeichnis und klone neu..."
delete_and_clone() {
  if [ -d "$REPO_DIR" ]; then
    echo "$REPO_DIR existiert – lösche es"
    rm -rf "$REPO_DIR"
  fi
  echo "Repository wird geklont"
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
}

delete_and_clone

echo "Aktuelle Dateien im Repository:"
ls -alh .

# >>> conda initialize >>>
echo "Conda initialize..."
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
  eval "$__conda_setup"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  . "/opt/conda/etc/profile.d/conda.sh"
else
  export PATH="/opt/conda/bin:$PATH"
fi
unset __conda_setup
# <<< conda initialize <<<

echo "Activate Conda environment..."
conda activate generic

echo "Installiere Python-Dependencies..."
pip install u8darts[torch] icalendar optuna pyyaml

# Config-Auflösung: PLUTO_CONFIG hat Vorrang, sonst Profil-basiert.
if [ -z "${PLUTO_CONFIG}" ]; then
  CONFIG_FILE="config.${PROFILE}.yaml"
  if [ -f "${CONFIG_FILE}" ]; then
    export PLUTO_CONFIG="${PWD}/${CONFIG_FILE}"
    echo "Verwende Config-Profil: ${PROFILE} (${CONFIG_FILE})"
  else
    echo "Config-Profil '${PROFILE}' nicht gefunden (${CONFIG_FILE}). Nutze Defaults."
  fi
else
  echo "Verwende PLUTO_CONFIG=${PLUTO_CONFIG}"
fi

# Runs-Verzeichnis anlegen (bleibt über Läufe hinweg erhalten)
mkdir -p "${PLUTO_RUNS_DIR}"
echo "Runs-Verzeichnis: ${PLUTO_RUNS_DIR}"

# Bisherige Runs anzeigen (für Nachvollziehbarkeit im Log)
RUN_COUNT=$(find "${PLUTO_RUNS_DIR}" -maxdepth 1 -type d -name "20*" 2>/dev/null | wc -l)
echo "Bisherige Runs im Verzeichnis: ${RUN_COUNT}"
if [ -f "${PLUTO_RUNS_DIR}/metrics_history.csv" ]; then
  HISTORY_LINES=$(wc -l < "${PLUTO_RUNS_DIR}/metrics_history.csv")
  echo "Metriken-Historie: ${HISTORY_LINES} Zeilen"
fi

echo "Starte Forecast Job..."
python pluto_forecast_job.py

echo "Job abgeschlossen."
echo "Domino synchronisiert ${PLUTO_RUNS_DIR} automatisch ins Projekt."
