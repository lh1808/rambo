28.08.2026 11:40 : Using curl at /opt/domino/bin/curl
28.08.2026 11:40 : Preparing working directory.
28.08.2026 11:40 : -- starting API proxy --
28.08.2026 11:40 : Starting periodic API token refresh.
28.08.2026 11:40 : ###############################################################################################################
28.08.2026 11:40 : #                                                                                                             #
28.08.2026 11:40 : # DEPRECATION WARNING:                                                                                        #
28.08.2026 11:40 : #                                                                                                             #
28.08.2026 11:40 : # Availability of $DOMINO_TOKEN_FILE is getting deprecated and will be removed in a future release.           #
28.08.2026 11:40 : #                                                                                                             #
28.08.2026 11:40 : # Please consider using the API Proxy:                                                                        #
28.08.2026 11:40 : # https://docs.dominodatalab.com/en/latest/user_guide/ddf8eb/use-the-api-proxy-for-domino-api-authentication/ #
28.08.2026 11:40 : #                                                                                                             #
28.08.2026 11:40 : ###############################################################################################################
28.08.2026 11:40 : /app/.venv/lib/python3.12/site-packages/tzlocal/unix.py:207: UserWarning: Can not find any timezone configuration, defaulting to UTC.
28.08.2026 11:40 :   warnings.warn("Can not find any timezone configuration, defaulting to UTC.")
28.08.2026 11:40 : Started API Proxy on port 8899 with 2 worker processes
28.08.2026 11:40 : ### SETUP PROCESS STARTED ###
28.08.2026 11:40 : Watching for changes in /var/lib/domino/launch/poison-pill
28.08.2026 11:40 : ### Executing /domino/launch/preSetupScript.sh ###
28.08.2026 11:40 : ### Completed /domino/launch/preSetupScript.sh ###
28.08.2026 11:40 : + echo '### Completed /domino/launch/preSetupScript.sh ###'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910047594238581
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.pre_setup_script.end&timestamp=1787910047594238581'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910047599114169
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pre_setup_script:end/trace?timestamp=1787910047599114169'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910047603537951
28.08.2026 11:40 : + export PIP_CONFIG_FILE=/mnt/pip.conf
28.08.2026 11:40 : + PIP_CONFIG_FILE=/mnt/pip.conf
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_custom_config:start/trace?timestamp=1787910047603537951'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910047607698789
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_custom_config:end/trace?timestamp=1787910047607698789'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910047611926220
28.08.2026 11:40 : + '[' -z '' ']'
28.08.2026 11:40 : + id domino
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.user_setup_section:start/trace?timestamp=1787910047611926220'
28.08.2026 11:40 : + USER=ubuntu
28.08.2026 11:40 : + '!id' ubuntu
28.08.2026 11:40 : + DOMINO_ID_UPDATED=false
28.08.2026 11:40 : + '[' -n 12574 ']'
28.08.2026 11:40 : ++ id -u ubuntu
28.08.2026 11:40 : + domino_previous_user_id=12574
28.08.2026 11:40 : + '[' 12574 '!=' 12574 ']'
28.08.2026 11:40 : + '[' -n 12574 ']'
28.08.2026 11:40 : ++ id -g ubuntu
28.08.2026 11:40 : + '[' 12574 '!=' 12574 ']'
28.08.2026 11:40 : + '[' false == true ']'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910047625692690
28.08.2026 11:40 : + echo '### Linking flyte config file to ~/.flyte/config.yaml' for user
28.08.2026 11:40 : ### Linking flyte config file to ~/.flyte/config.yaml for user
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.user_setup_section:end/trace?timestamp=1787910047625692690'
28.08.2026 11:40 : ++ cat /etc/passwd
28.08.2026 11:40 : ++ grep '^ubuntu:'
28.08.2026 11:40 : ++ cut -d: -f6
28.08.2026 11:40 : + user_home=/mnt
28.08.2026 11:40 : + mkdir -p /mnt/.flyte
28.08.2026 11:40 : + ln -s /domino/workflows/config.yaml /mnt/.flyte/config.yaml
28.08.2026 11:40 : ln: Already exists
28.08.2026 11:40 : + true
28.08.2026 11:40 : + echo '### Done linking flyte config file'
28.08.2026 11:40 : ### Done linking flyte config file
28.08.2026 11:40 : + [[ 1DOMINO_IS_WORKFLOW_JOB != true ]]
28.08.2026 11:40 : + [[ ! -d /workflow/inputs ]]
28.08.2026 11:40 : + [[ ! -d /workflow/outputs ]]
28.08.2026 11:40 : + cd /tmp
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910047641326272
28.08.2026 11:40 : + '[' -f /var/lib/domino/launch/.git-credentials ']'
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.git_creds_section:start/trace?timestamp=1787910047641326272'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910047645407810
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.git_creds_section:end/trace?timestamp=1787910047645407810'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910047649362467
28.08.2026 11:40 : + '[' -f /mnt/requirements.txt ']'
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_install_section:start/trace?timestamp=1787910047649362467'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910047653502450
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_install_section:end/trace?timestamp=1787910047653502450'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910047657531135
28.08.2026 11:40 : + '[' -f /domino/launch/postSetupScript.sh ']'
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.post_setup_script:start/trace?timestamp=1787910047657531135'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910047661690876
28.08.2026 11:40 : + cd /mnt
28.08.2026 11:40 : + echo '### SETUP PROCESS FINISHED ###\n'
28.08.2026 11:40 : ### SETUP PROCESS FINISHED ###\n
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.post_setup_script:end/trace?timestamp=1787910047661690876'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910047665995611
28.08.2026 11:40 : + chmod +x /var/lib/domino/launch/command.sh
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.run_command_section:start/trace?timestamp=1787910047665995611'
28.08.2026 11:40 : + declare -ir run_command_pid=179
28.08.2026 11:40 : + /var/lib/domino/launch/command.sh
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : Using curl at /opt/domino/bin/curl
28.08.2026 11:40 : + TIMESTAMP=1787910047672642745
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.run_command_section:end/trace?timestamp=1787910047672642745'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910047676714964
28.08.2026 11:40 : + wait 179
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script:end/trace?timestamp=1787910047676714964'
28.08.2026 11:40 : ### Executing /mnt/.domino/configure-spark-defaults.sh ###
28.08.2026 11:40 : ### Completed /mnt/.domino/configure-spark-defaults.sh ###
28.08.2026 11:40 : ++ conda init bash
28.08.2026 11:40 : ### Executing /domino/launch/preRunScript.sh ###
28.08.2026 11:40 : no change     /opt/conda/condabin/conda
28.08.2026 11:40 : no change     /opt/conda/bin/conda
28.08.2026 11:40 : no change     /opt/conda/bin/activate
28.08.2026 11:40 : no change     /opt/conda/bin/deactivate
28.08.2026 11:40 : no change     /opt/conda/etc/profile.d/conda.sh
28.08.2026 11:40 : no change     /opt/conda/etc/fish/conf.d/conda.fish
28.08.2026 11:40 : no change     /opt/conda/shell/condabin/Conda.psm1
28.08.2026 11:40 : no change     /opt/conda/shell/condabin/conda-hook.ps1
28.08.2026 11:40 : no change     /opt/conda/lib/python3.14/site-packages/xontrib/conda.xsh
28.08.2026 11:40 : no change     /opt/conda/etc/profile.d/conda.csh
28.08.2026 11:40 : no change     /mnt/.bashrc
28.08.2026 11:40 : No action taken.
28.08.2026 11:40 : ++ micromamba shell init --shell bash --root-prefix=/opt/conda
28.08.2026 11:40 : Modifying RC file "/mnt/.bashrc"
28.08.2026 11:40 : Generating config for root prefix [1m"/opt/conda"[0m
28.08.2026 11:40 : Setting mamba executable to: [1m"/usr/local/bin/micromamba"[0m
28.08.2026 11:40 : Adding (or replacing) the following in your "/mnt/.bashrc" file
28.08.2026 11:40 : # >>> mamba initialize >>>
28.08.2026 11:40 : # !! Contents within this block are managed by 'mamba init' !!
28.08.2026 11:40 : export MAMBA_EXE='/usr/local/bin/micromamba';
28.08.2026 11:40 : export MAMBA_ROOT_PREFIX='/opt/conda';
28.08.2026 11:40 : __mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
28.08.2026 11:40 : if [ $? -eq 0 ]; then
28.08.2026 11:40 :     eval "$__mamba_setup"
28.08.2026 11:40 : else
28.08.2026 11:40 :     alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
28.08.2026 11:40 : fi
28.08.2026 11:40 : unset __mamba_setup
28.08.2026 11:40 : # <<< mamba initialize <<<
28.08.2026 11:40 : + echo '### Completed /domino/launch/preRunScript.sh ###'
28.08.2026 11:40 : ### Completed /domino/launch/preRunScript.sh ###
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910048623520954
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.pre_run_script.end&timestamp=1787910048623520954'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910048627639923
28.08.2026 11:40 : + cd /mnt
28.08.2026 11:40 : + set +o errexit
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.command_launch_script.pre_run_script:end/trace?timestamp=1787910048627639923'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910048632472401
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.user_init:start/trace?timestamp=1787910048632472401'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910048636910777
28.08.2026 11:40 : + declare -ri run_command=226
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.user_init.wait_connectable:start/trace?timestamp=1787910048636910777'
28.08.2026 11:40 : + bash Production/DWA_production/run_scoring.sh Production/DWA_production/job_dwa.conf
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : ++ tee -a /mnt/results/stdout.txt
28.08.2026 11:40 : ++ tee -a /mnt/results/stderr.txt
28.08.2026 11:40 : + TIMESTAMP=1787910048641020361
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.final_run_command_issued&timestamp=1787910048641020361'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910048644996259
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.command_launch_script:end/trace?timestamp=1787910048644996259'
28.08.2026 11:40 : ++ date +%s%3N
28.08.2026 11:40 : + TIMESTAMP=1787910048648847146
28.08.2026 11:40 : + wait 226
28.08.2026 11:40 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container:end/trace?timestamp=1787910048648847146'
28.08.2026 11:40 : [2026-08-28 11:40:48] === rubin Production-Scoring: Start (ref=master, env=prod) ===
28.08.2026 11:40 : [2026-08-28 11:40:48] WARNUNG: GIT_REF='master' ist ein Branch — jeder Lauf zieht den jeweils neuesten Commit.
28.08.2026 11:40 : [2026-08-28 11:40:48]          Für reproduzierbare Prod-Läufe GIT_REF auf Tag/SHA pinnen (und REQUIRE_PINNED_REF=1 setzen).
28.08.2026 11:40 : [2026-08-28 11:40:48] --- SSH vorbereiten ---
28.08.2026 11:40 : [2026-08-28 11:40:48] SSH-Key: /mnt/.ssh/id_rsa
28.08.2026 11:40 : git version 2.53.0
28.08.2026 11:40 : [2026-08-28 11:40:48] --- Git Clone (master) ---
28.08.2026 11:40 : Cloning into '/home/ubuntu/rubin_scoring'...
28.08.2026 11:40 : ** WARNING: connection is not using a post-quantum key exchange algorithm.
28.08.2026 11:40 : ** This session may be vulnerable to "store now, decrypt later" attacks.
28.08.2026 11:40 : ** The server may need to be upgraded. See https://openssh.com/pq.html
28.08.2026 11:40 : [2026-08-28 11:40:50] WARNUNG: Gestartetes Skript weicht vom Repo-Master (master) ab — FS-Kopie bitte synchronisieren:
28.08.2026 11:40 : [2026-08-28 11:40:50]          laufend: /mnt/Production/DWA_production/run_scoring.sh
28.08.2026 11:40 : [2026-08-28 11:40:50]          Master:  /home/ubuntu/rubin_scoring/production/run_scoring.sh
28.08.2026 11:40 : [2026-08-28 11:40:50] WARNUNG: Job-Datei job_dwa.conf weicht vom Repo-Master (master) ab — FS-Kopie bitte synchronisieren:
28.08.2026 11:40 : [2026-08-28 11:40:50]          laufend: /mnt/Production/DWA_production/job_dwa.conf
28.08.2026 11:40 : [2026-08-28 11:40:50]          Master:  /home/ubuntu/rubin_scoring/production/jobs/job_dwa.conf
28.08.2026 11:40 : [2026-08-28 11:40:50] Repo-Stand: 1d5bb99 (2026-08-28)
28.08.2026 11:40 : [2026-08-28 11:40:50] --- Pixi Install (-e prod) ---
28.08.2026 11:40 : pixi 0.72.0
28.08.2026 11:40 : [2026-08-28 11:40:50] pixi.lock gefunden → deterministische Installation (--frozen).
28.08.2026 11:40 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
28.08.2026 11:40 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
28.08.2026 11:40 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
28.08.2026 11:40 : ✔ The prod environment has been installed.
28.08.2026 11:40 : [2026-08-28 11:40:58] --- Scoring: 2 Config(s): production/scoring_un.yml production/scoring_wg.yml ---
28.08.2026 11:40 : [2026-08-28 11:40:58] --- Score starten: production/scoring_un.yml ---
28.08.2026 11:40 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
28.08.2026 11:40 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
28.08.2026 11:40 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
28.08.2026 11:40 :  WARN the lock file is up-to-date but uses an older format (v6), run `pixi lock` to upgrade to v7 for improved reproducibility
28.08.2026 11:49 : 2026-08-28 11:49:41,825 INFO [rubin.scoring] Input: 11799629 Zeilen, 110 Spalten (spalten-gepruned, Union über alle Bundles) (/domino/edv/pvc-hf1kundehuk/ScoringdatensatzProduktiv/scoringdatensatz.sas7bdat) — 1 Score(s) gegen diese eine Ladung
28.08.2026 11:49 : 2026-08-28 11:49:42,074 INFO [rubin.scoring] Bundle-Modelle (per YAML frei wählbar, unabhängig vom Champion): ['CausalForest', 'CausalForestDML', 'DRLearner', 'Ensemble', 'NonParamDML', 'SurrogateTree', 'TLearner', 'XLearner'] | Champion: CausalForestDML
28.08.2026 11:49 : 2026-08-28 11:49:44,100 WARNING [rubin.scoring] Nicht-numerische Werte in numerisch trainierten Features durch NaN ersetzt (→ gelernte Imputation greift) — Sonderwert-/Drift-Signal: {'GINT_KFZ_JAHRESFAHRKILOMETER': 0.002651}
28.08.2026 11:51 : 2026-08-28 11:51:09,525 WARNING [rubin.scoring] Erhöhte -1-Raten (unbekannte Kategorien/Missings) — mögliches Drift-Signal: {'PLZ1': 0.067271, 'PLZ_2A': 0.065262, 'PLZ_3A': 0.065272, 'REGIO_GS_BEREICH_FS': 0.215724, 'AKQ_BERUF_STATUS': 0.05474, 'AKQ_FAMILIENSTAND_FS': 0.832616, 'GINT_KFZ_SCHLUESSEL_NR_HERST': 0.695025, 'ALTERSGRUPPE': 0.099971, 'BDL': 0.809953}
28.08.2026 11:51 : 2026-08-28 11:51:13,712 INFO [matplotlib.font_manager] generated new fontManager
28.08.2026 11:52 : 2026-08-28 11:52:15,644 INFO [rubin.scoring] SCORE_P: CausalForestDML
28.08.2026 11:52 : 2026-08-28 11:52:17,206 INFO [rubin.scoring] SCORE_B: SurrogateTree
28.08.2026 11:52 : 2026-08-28 11:52:19,790 WARNING [rubin.scoring] XPT V5 kürzt Variablennamen auf 8 Zeichen: {'PARTNER_ID_V': 'PARTNER_', 'GESELLSCHAFT_FS': 'GESELLSC', 'SCORE_TYP': 'SCORE_TY', 'SCORE_VERFAHREN': 'SCORE_VE', 'TIMESTAMP': 'TIMESTAM'}
28.08.2026 11:53 : 2026-08-28 11:53:28,030 INFO [rubin.scoring] XPT geschrieben: /domino/edv/pvc-hf1kundehuk/ScoringdatensatzProduktiv/kausalscore_un.xpt (11799629 Zeilen, V5, Tabelle kau_un)
28.08.2026 11:53 : 2026-08-28 11:53:28,046 INFO [rubin.scoring] Monitoring: /domino/edv/pvc-hf1kundehuk/ScoringdatensatzProduktiv/monitoring/kausalscore_un_2026-240_094102.json
28.08.2026 11:53 : [2026-08-28 11:53:32] --- Score OK: production/scoring_un.yml (754s) ---
28.08.2026 11:53 : [2026-08-28 11:53:32] --- Score starten: production/scoring_wg.yml ---
28.08.2026 11:53 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
28.08.2026 11:53 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
28.08.2026 11:53 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
28.08.2026 11:53 :  WARN the lock file is up-to-date but uses an older format (v6), run `pixi lock` to upgrade to v7 for improved reproducibility
28.08.2026 12:02 : 2026-08-28 12:02:10,532 INFO [rubin.scoring] Input: 11799629 Zeilen, 102 Spalten (spalten-gepruned, Union über alle Bundles) (/domino/edv/pvc-hf1kundehuk/ScoringdatensatzProduktiv/scoringdatensatz.sas7bdat) — 1 Score(s) gegen diese eine Ladung
28.08.2026 12:02 : 2026-08-28 12:02:10,771 INFO [rubin.scoring] Bundle-Modelle (per YAML frei wählbar, unabhängig vom Champion): ['CausalForestDML', 'Ensemble', 'SurrogateTree', 'XLearner'] | Champion: Ensemble
28.08.2026 12:03 : 2026-08-28 12:03:33,805 WARNING [rubin.scoring] Erhöhte -1-Raten (unbekannte Kategorien/Missings) — mögliches Drift-Signal: {'PLZ1': 0.374304, 'PLZ_2A': 0.084667, 'PLZ_3A': 0.284819, 'REGIO_GS_BEREICH_FS': 0.215724, 'AKQ_BERUF_STATUS_FS': 0.842202, 'GINT_KFZ_SCHLUESSEL_NR_HERST': 0.694712, 'GINT_KFZ_SCHLUESSEL_NR_TYP': 0.446111, 'GINT_KFZ_KATEGORIE': 0.409117, 'GINT_KFZ_AUFBAU': 0.409117, 'BDL': 0.809953}
28.08.2026 12:03 : Traceback (most recent call last):
28.08.2026 12:03 :   File "/home/ubuntu/rubin_scoring/production/run_scoring.py", line 706, in <module>
28.08.2026 12:03 :     main()
28.08.2026 12:03 :   File "/home/ubuntu/rubin_scoring/production/run_scoring.py", line 702, in main
28.08.2026 12:03 :     run_scoring(cfg)
28.08.2026 12:03 :   File "/home/ubuntu/rubin_scoring/production/run_scoring.py", line 623, in run_scoring
28.08.2026 12:03 :     out, pipe, core = score_dataframe(df, e_cfg, day_stamp, pipe=pipe)
28.08.2026 12:03 :                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
28.08.2026 12:03 :   File "/home/ubuntu/rubin_scoring/production/run_scoring.py", line 445, in score_dataframe
28.08.2026 12:03 :     for col, vals in _score_columns("SCORE_P", predict_in_batches(pipe.models[p_name], Xp, batch)).items():
28.08.2026 12:03 :                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
28.08.2026 12:03 :   File "/home/ubuntu/rubin_scoring/production/run_scoring.py", line 291, in predict_in_batches
28.08.2026 12:03 :     parts = [np.asarray(_predict_effect(model, X.iloc[i:i + batch_size]))
28.08.2026 12:03 :                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
28.08.2026 12:03 :   File "/home/ubuntu/rubin_scoring/rubin/training.py", line 58, in _predict_effect
28.08.2026 12:03 :     pred = model.const_marginal_effect(X)
28.08.2026 12:03 :            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
28.08.2026 12:03 :   File "/home/ubuntu/rubin_scoring/rubin/training.py", line 253, in const_marginal_effect
28.08.2026 12:03 :     preds = [np.asarray(_predict_effect(m, X), dtype=float) for m in self.cate_models]
28.08.2026 12:03 :                         ^^^^^^^^^^^^^^^^^^^^^
28.08.2026 12:03 :   File "/home/ubuntu/rubin_scoring/rubin/training.py", line 58, in _predict_effect
28.08.2026 12:03 :     pred = model.const_marginal_effect(X)
28.08.2026 12:03 :            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
28.08.2026 12:03 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/econml/metalearners/_metalearners.py", line 442, in const_marginal_effect
28.08.2026 12:03 :     tau_hat = propensity_scores * self.cate_controls_models[ind].predict(X).reshape(m, -1) \
28.08.2026 12:03 :                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
28.08.2026 12:03 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/catboost/core.py", line 6229, in predict
28.08.2026 12:03 :     return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, 'predict', task_type)
28.08.2026 12:03 :            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
28.08.2026 12:03 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/catboost/core.py", line 2926, in _predict
28.08.2026 12:03 :     data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
28.08.2026 12:03 :                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
28.08.2026 12:03 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/catboost/core.py", line 2906, in _process_predict_input_data
28.08.2026 12:03 :     data = Pool(
28.08.2026 12:03 :            ^^^^^
28.08.2026 12:03 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/catboost/core.py", line 805, in __init__
28.08.2026 12:03 :     raise CatBoostError(
28.08.2026 12:03 : _catboost.CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features
28.08.2026 12:03 : [2026-08-28 12:03:38] --- Score FEHLGESCHLAGEN (rc=1): production/scoring_wg.yml ---
28.08.2026 12:03 : + exitcode=1
28.08.2026 12:03 : + '[' 1 -eq 0 ']'
28.08.2026 12:03 : + sleep 2
28.08.2026 12:03 : Evaluating cleanup command on EXIT with exit code 1: $CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true
28.08.2026 12:03 : + exit 1
28.08.2026 12:03 : + exit_logging
28.08.2026 12:03 : + local -r code_at_exit=1
28.08.2026 12:03 : + '[' -n '$CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true' ']'
28.08.2026 12:03 : + local max_retries=10
28.08.2026 12:03 : + local retry_delay=3
28.08.2026 12:03 : + echo 'Evaluating cleanup command on EXIT with exit code 1: $CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true'
28.08.2026 12:03 : + local n=1
28.08.2026 12:03 : + true
28.08.2026 12:03 : + eval '$CURL' -sS -H '"X-Api-Token:' '$DOMINO_EXECUTOR_TERMINATION_TOKEN"' -X POST 'http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit' '||' true
28.08.2026 12:03 : ++ /opt/domino/bin/curl -sS -H 'X-Api-Token: b8f783b4a4d5f0c4c15259f3390da755608432417f95f61621d9b4bd6271c4c9' -X POST 'http://127.0.0.1:9000/executor/exit?exitCode=1'
28.08.2026 12:03 : + break
28.08.2026 12:03 : + [[ 1 =~ ^(0|137|143)$ ]]
28.08.2026 12:03 : ++ tee -a /mnt/results/stdout.txt
28.08.2026 12:03 : + sleep 0.5
28.08.2026 12:03 : + echo 'Failed with exit code: 1'
28.08.2026 12:03 : Failed with exit code: 1
28.08.2026 12:03 : + exit 1
28.08.2026 12:03 : Caught termination signal!
28.08.2026 12:03 : -- killed by pod termination --
