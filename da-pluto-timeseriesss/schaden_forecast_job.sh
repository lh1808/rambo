#!/bin/bash

# Beende bei jedem Fehler
set -e

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

echo "Lösche vorhandenes Verzeichnis und klone neu..."
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

echo "Starte Forecast Job..."
python pluto_forecast_job.py

echo "Job abgeschlossen."
