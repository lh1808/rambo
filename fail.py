27.08.2026 12:22 : Using curl at /opt/domino/bin/curl
27.08.2026 12:22 : Preparing working directory.
27.08.2026 12:22 : -- starting API proxy --
27.08.2026 12:22 : Starting periodic API token refresh.
27.08.2026 12:22 : ###############################################################################################################
27.08.2026 12:22 : #                                                                                                             #
27.08.2026 12:22 : # DEPRECATION WARNING:                                                                                        #
27.08.2026 12:22 : #                                                                                                             #
27.08.2026 12:22 : # Availability of $DOMINO_TOKEN_FILE is getting deprecated and will be removed in a future release.           #
27.08.2026 12:22 : #                                                                                                             #
27.08.2026 12:22 : # Please consider using the API Proxy:                                                                        #
27.08.2026 12:22 : # https://docs.dominodatalab.com/en/latest/user_guide/ddf8eb/use-the-api-proxy-for-domino-api-authentication/ #
27.08.2026 12:22 : #                                                                                                             #
27.08.2026 12:22 : ###############################################################################################################
27.08.2026 12:22 : /app/.venv/lib/python3.12/site-packages/tzlocal/unix.py:207: UserWarning: Can not find any timezone configuration, defaulting to UTC.
27.08.2026 12:22 :   warnings.warn("Can not find any timezone configuration, defaulting to UTC.")
27.08.2026 12:22 : Started API Proxy on port 8899 with 2 worker processes
27.08.2026 12:22 : ### SETUP PROCESS STARTED ###
27.08.2026 12:22 : Watching for changes in /var/lib/domino/launch/poison-pill
27.08.2026 12:22 : ### Executing /domino/launch/preSetupScript.sh ###
27.08.2026 12:22 : ### Completed /domino/launch/preSetupScript.sh ###
27.08.2026 12:22 : + echo '### Completed /domino/launch/preSetupScript.sh ###'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826141649348457
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.pre_setup_script.end&timestamp=1787826141649348457'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826141654202165
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pre_setup_script:end/trace?timestamp=1787826141654202165'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826141658647835
27.08.2026 12:22 : + export PIP_CONFIG_FILE=/mnt/pip.conf
27.08.2026 12:22 : + PIP_CONFIG_FILE=/mnt/pip.conf
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_custom_config:start/trace?timestamp=1787826141658647835'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826141663154478
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_custom_config:end/trace?timestamp=1787826141663154478'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826141667681424
27.08.2026 12:22 : + '[' -z '' ']'
27.08.2026 12:22 : + id domino
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.user_setup_section:start/trace?timestamp=1787826141667681424'
27.08.2026 12:22 : + USER=ubuntu
27.08.2026 12:22 : + '!id' ubuntu
27.08.2026 12:22 : + DOMINO_ID_UPDATED=false
27.08.2026 12:22 : + '[' -n 12574 ']'
27.08.2026 12:22 : ++ id -u ubuntu
27.08.2026 12:22 : + domino_previous_user_id=12574
27.08.2026 12:22 : + '[' 12574 '!=' 12574 ']'
27.08.2026 12:22 : + '[' -n 12574 ']'
27.08.2026 12:22 : ++ id -g ubuntu
27.08.2026 12:22 : + '[' 12574 '!=' 12574 ']'
27.08.2026 12:22 : + '[' false == true ']'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826141683124662
27.08.2026 12:22 : + echo '### Linking flyte config file to ~/.flyte/config.yaml' for user
27.08.2026 12:22 : ### Linking flyte config file to ~/.flyte/config.yaml for user
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.user_setup_section:end/trace?timestamp=1787826141683124662'
27.08.2026 12:22 : ++ cat /etc/passwd
27.08.2026 12:22 : ++ grep '^ubuntu:'
27.08.2026 12:22 : ++ cut -d: -f6
27.08.2026 12:22 : + user_home=/mnt
27.08.2026 12:22 : + mkdir -p /mnt/.flyte
27.08.2026 12:22 : + ln -s /domino/workflows/config.yaml /mnt/.flyte/config.yaml
27.08.2026 12:22 : ln: Already exists
27.08.2026 12:22 : + true
27.08.2026 12:22 : + echo '### Done linking flyte config file'
27.08.2026 12:22 : ### Done linking flyte config file
27.08.2026 12:22 : + [[ 1DOMINO_IS_WORKFLOW_JOB != true ]]
27.08.2026 12:22 : + [[ ! -d /workflow/inputs ]]
27.08.2026 12:22 : + [[ ! -d /workflow/outputs ]]
27.08.2026 12:22 : + cd /tmp
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826141697992186
27.08.2026 12:22 : + '[' -f /var/lib/domino/launch/.git-credentials ']'
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.git_creds_section:start/trace?timestamp=1787826141697992186'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826141701975385
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.git_creds_section:end/trace?timestamp=1787826141701975385'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826141705938491
27.08.2026 12:22 : + '[' -f /mnt/requirements.txt ']'
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_install_section:start/trace?timestamp=1787826141705938491'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826141710414461
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_install_section:end/trace?timestamp=1787826141710414461'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826141715540878
27.08.2026 12:22 : + '[' -f /domino/launch/postSetupScript.sh ']'
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.post_setup_script:start/trace?timestamp=1787826141715540878'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826141721319468
27.08.2026 12:22 : + cd /mnt
27.08.2026 12:22 : + echo '### SETUP PROCESS FINISHED ###\n'
27.08.2026 12:22 : ### SETUP PROCESS FINISHED ###\n
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.post_setup_script:end/trace?timestamp=1787826141721319468'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826141727137149
27.08.2026 12:22 : + chmod +x /var/lib/domino/launch/command.sh
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.run_command_section:start/trace?timestamp=1787826141727137149'
27.08.2026 12:22 : + declare -ir run_command_pid=191
27.08.2026 12:22 : + /var/lib/domino/launch/command.sh
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : Using curl at /opt/domino/bin/curl
27.08.2026 12:22 : + TIMESTAMP=1787826141735875807
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.run_command_section:end/trace?timestamp=1787826141735875807'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826141740263599
27.08.2026 12:22 : + wait 191
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script:end/trace?timestamp=1787826141740263599'
27.08.2026 12:22 : ### Executing /mnt/.domino/configure-spark-defaults.sh ###
27.08.2026 12:22 : ### Completed /mnt/.domino/configure-spark-defaults.sh ###
27.08.2026 12:22 : ### Executing /domino/launch/preRunScript.sh ###
27.08.2026 12:22 : ++ conda init bash
27.08.2026 12:22 : no change     /opt/conda/condabin/conda
27.08.2026 12:22 : no change     /opt/conda/bin/conda
27.08.2026 12:22 : no change     /opt/conda/bin/activate
27.08.2026 12:22 : no change     /opt/conda/bin/deactivate
27.08.2026 12:22 : no change     /opt/conda/etc/profile.d/conda.sh
27.08.2026 12:22 : no change     /opt/conda/etc/fish/conf.d/conda.fish
27.08.2026 12:22 : no change     /opt/conda/shell/condabin/Conda.psm1
27.08.2026 12:22 : no change     /opt/conda/shell/condabin/conda-hook.ps1
27.08.2026 12:22 : no change     /opt/conda/lib/python3.14/site-packages/xontrib/conda.xsh
27.08.2026 12:22 : no change     /opt/conda/etc/profile.d/conda.csh
27.08.2026 12:22 : no change     /mnt/.bashrc
27.08.2026 12:22 : No action taken.
27.08.2026 12:22 : ++ micromamba shell init --shell bash --root-prefix=/opt/conda
27.08.2026 12:22 : Modifying RC file "/mnt/.bashrc"
27.08.2026 12:22 : Generating config for root prefix [1m"/opt/conda"[0m
27.08.2026 12:22 : Setting mamba executable to: [1m"/usr/local/bin/micromamba"[0m
27.08.2026 12:22 : Adding (or replacing) the following in your "/mnt/.bashrc" file
27.08.2026 12:22 : # >>> mamba initialize >>>
27.08.2026 12:22 : # !! Contents within this block are managed by 'mamba init' !!
27.08.2026 12:22 : export MAMBA_EXE='/usr/local/bin/micromamba';
27.08.2026 12:22 : export MAMBA_ROOT_PREFIX='/opt/conda';
27.08.2026 12:22 : __mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
27.08.2026 12:22 : if [ $? -eq 0 ]; then
27.08.2026 12:22 :     eval "$__mamba_setup"
27.08.2026 12:22 : else
27.08.2026 12:22 :     alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
27.08.2026 12:22 : fi
27.08.2026 12:22 : unset __mamba_setup
27.08.2026 12:22 : # <<< mamba initialize <<<
27.08.2026 12:22 : + echo '### Completed /domino/launch/preRunScript.sh ###'
27.08.2026 12:22 : ### Completed /domino/launch/preRunScript.sh ###
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826142759027523
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.pre_run_script.end&timestamp=1787826142759027523'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826142763906612
27.08.2026 12:22 : + cd /mnt
27.08.2026 12:22 : + set +o errexit
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.command_launch_script.pre_run_script:end/trace?timestamp=1787826142763906612'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826142768768411
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.user_init:start/trace?timestamp=1787826142768768411'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826142773208624
27.08.2026 12:22 : + declare -ri run_command=238
27.08.2026 12:22 : + bash Production/DWA_production/run_scoring.sh Production/DWA_production/job_dwa.conf
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.user_init.wait_connectable:start/trace?timestamp=1787826142773208624'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : ++ tee -a /mnt/results/stdout.txt
27.08.2026 12:22 : ++ tee -a /mnt/results/stderr.txt
27.08.2026 12:22 : + TIMESTAMP=1787826142778190303
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.final_run_command_issued&timestamp=1787826142778190303'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826142783598457
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.command_launch_script:end/trace?timestamp=1787826142783598457'
27.08.2026 12:22 : ++ date +%s%3N
27.08.2026 12:22 : + TIMESTAMP=1787826142788076672
27.08.2026 12:22 : + wait 238
27.08.2026 12:22 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container:end/trace?timestamp=1787826142788076672'
27.08.2026 12:22 : [2026-08-27 12:22:22] === rubin Production-Scoring: Start (ref=master, env=prod) ===
27.08.2026 12:22 : [2026-08-27 12:22:22] WARNUNG: GIT_REF='master' ist ein Branch — jeder Lauf zieht den jeweils neuesten Commit.
27.08.2026 12:22 : [2026-08-27 12:22:22]          Für reproduzierbare Prod-Läufe GIT_REF auf Tag/SHA pinnen (und REQUIRE_PINNED_REF=1 setzen).
27.08.2026 12:22 : [2026-08-27 12:22:22] --- SSH vorbereiten ---
27.08.2026 12:22 : [2026-08-27 12:22:22] SSH-Key: /mnt/.ssh/id_rsa
27.08.2026 12:22 : git version 2.53.0
27.08.2026 12:22 : [2026-08-27 12:22:22] --- Git Clone (master) ---
27.08.2026 12:22 : Cloning into '/home/ubuntu/rubin_scoring'...
27.08.2026 12:22 : ** WARNING: connection is not using a post-quantum key exchange algorithm.
27.08.2026 12:22 : ** This session may be vulnerable to "store now, decrypt later" attacks.
27.08.2026 12:22 : ** The server may need to be upgraded. See https://openssh.com/pq.html
27.08.2026 12:22 : [2026-08-27 12:22:23] WARNUNG: Gestartetes Skript weicht vom Repo-Master (master) ab — FS-Kopie bitte synchronisieren:
27.08.2026 12:22 : [2026-08-27 12:22:23]          laufend: /mnt/Production/DWA_production/run_scoring.sh
27.08.2026 12:22 : [2026-08-27 12:22:23]          Master:  /home/ubuntu/rubin_scoring/production/run_scoring.sh
27.08.2026 12:22 : [2026-08-27 12:22:23] WARNUNG: Job-Datei job_dwa.conf weicht vom Repo-Master (master) ab — FS-Kopie bitte synchronisieren:
27.08.2026 12:22 : [2026-08-27 12:22:23]          laufend: /mnt/Production/DWA_production/job_dwa.conf
27.08.2026 12:22 : [2026-08-27 12:22:23]          Master:  /home/ubuntu/rubin_scoring/production/jobs/job_dwa.conf
27.08.2026 12:22 : [2026-08-27 12:22:23] Repo-Stand: 90d06b4 (2026-08-27)
27.08.2026 12:22 : [2026-08-27 12:22:23] --- Pixi Install (-e prod) ---
27.08.2026 12:22 : pixi 0.72.0
27.08.2026 12:22 : [2026-08-27 12:22:23] pixi.lock gefunden → deterministische Installation (--frozen).
27.08.2026 12:22 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
27.08.2026 12:22 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
27.08.2026 12:22 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
27.08.2026 12:23 : ✔ The prod environment has been installed.
27.08.2026 12:23 : [2026-08-27 12:23:55] --- Scoring: 2 Config(s): production/scoring_wg.yml production/scoring_un.yml ---
27.08.2026 12:23 : [2026-08-27 12:23:55] --- Score starten: production/scoring_wg.yml ---
27.08.2026 12:23 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
27.08.2026 12:23 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
27.08.2026 12:23 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
27.08.2026 12:23 :  WARN the lock file is up-to-date but uses an older format (v6), run `pixi lock` to upgrade to v7 for improved reproducibility
27.08.2026 12:24 : Traceback (most recent call last):
27.08.2026 12:24 :   File "/home/ubuntu/rubin_scoring/production/run_scoring.py", line 683, in <module>
27.08.2026 12:24 :     main()
27.08.2026 12:24 :   File "/home/ubuntu/rubin_scoring/production/run_scoring.py", line 679, in main
27.08.2026 12:24 :     run_scoring(cfg)
27.08.2026 12:24 :   File "/home/ubuntu/rubin_scoring/production/run_scoring.py", line 564, in run_scoring
27.08.2026 12:24 :     pipes[b] = ProductionPipeline(b)
27.08.2026 12:24 :                ^^^^^^^^^^^^^^^^^^^^^
27.08.2026 12:24 :   File "/home/ubuntu/rubin_scoring/rubin/pipelines/production_pipeline.py", line 144, in __init__
27.08.2026 12:24 :     raise ValueError(
27.08.2026 12:24 : ValueError: ml_package_versions fehlt in metadata.json (/domino/edv/pvc-hf1kundehuk/mlflow/284392851464958591/8a3509dafd2c463ca34d9ee0bafe29f6/artifacts/bundle_bundle_20260606_054005/metadata.json). Ohne Versions-Stempel ist die Pickle-Kompatibilität nicht prüfbar.
27.08.2026 12:24 : [2026-08-27 12:24:00] --- Score FEHLGESCHLAGEN (rc=1): production/scoring_wg.yml ---
27.08.2026 12:24 : + exitcode=1
27.08.2026 12:24 : + '[' 1 -eq 0 ']'
27.08.2026 12:24 : + sleep 2
27.08.2026 12:24 : + exit 1
27.08.2026 12:24 : Evaluating cleanup command on EXIT with exit code 1: $CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true
27.08.2026 12:24 : + exit_logging
27.08.2026 12:24 : + local -r code_at_exit=1
27.08.2026 12:24 : + '[' -n '$CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true' ']'
27.08.2026 12:24 : + local max_retries=10
27.08.2026 12:24 : + local retry_delay=3
27.08.2026 12:24 : + echo 'Evaluating cleanup command on EXIT with exit code 1: $CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true'
27.08.2026 12:24 : + local n=1
27.08.2026 12:24 : + true
27.08.2026 12:24 : + eval '$CURL' -sS -H '"X-Api-Token:' '$DOMINO_EXECUTOR_TERMINATION_TOKEN"' -X POST 'http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit' '||' true
27.08.2026 12:24 : ++ /opt/domino/bin/curl -sS -H 'X-Api-Token: 375fc36fd6855033c455f0f8ab35026ba77955e4ba2b5a1e56658d216f132a0c' -X POST 'http://127.0.0.1:9000/executor/exit?exitCode=1'
27.08.2026 12:24 : + break
27.08.2026 12:24 : + [[ 1 =~ ^(0|137|143)$ ]]
27.08.2026 12:24 : ++ tee -a /mnt/results/stdout.txt
27.08.2026 12:24 : + sleep 0.5
27.08.2026 12:24 : + echo 'Failed with exit code: 1'
27.08.2026 12:24 : Failed with exit code: 1
27.08.2026 12:24 : + exit 1
27.08.2026 12:24 : Caught termination signal!
27.08.2026 12:24 : -- killed by pod termination --
