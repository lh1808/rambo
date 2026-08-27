27.08.2026 12:57 : Using curl at /opt/domino/bin/curl
27.08.2026 12:57 : Preparing working directory.
27.08.2026 12:57 : -- starting API proxy --
27.08.2026 12:57 : Starting periodic API token refresh.
27.08.2026 12:57 : ###############################################################################################################
27.08.2026 12:57 : #                                                                                                             #
27.08.2026 12:57 : # DEPRECATION WARNING:                                                                                        #
27.08.2026 12:57 : #                                                                                                             #
27.08.2026 12:57 : # Availability of $DOMINO_TOKEN_FILE is getting deprecated and will be removed in a future release.           #
27.08.2026 12:57 : #                                                                                                             #
27.08.2026 12:57 : # Please consider using the API Proxy:                                                                        #
27.08.2026 12:57 : # https://docs.dominodatalab.com/en/latest/user_guide/ddf8eb/use-the-api-proxy-for-domino-api-authentication/ #
27.08.2026 12:57 : #                                                                                                             #
27.08.2026 12:57 : ###############################################################################################################
27.08.2026 12:57 : /app/.venv/lib/python3.12/site-packages/tzlocal/unix.py:207: UserWarning: Can not find any timezone configuration, defaulting to UTC.
27.08.2026 12:57 :   warnings.warn("Can not find any timezone configuration, defaulting to UTC.")
27.08.2026 12:57 : Started API Proxy on port 8899 with 2 worker processes
27.08.2026 12:57 : ### SETUP PROCESS STARTED ###
27.08.2026 12:57 : Watching for changes in /var/lib/domino/launch/poison-pill
27.08.2026 12:57 : ### Executing /domino/launch/preSetupScript.sh ###
27.08.2026 12:57 : + echo '### Completed /domino/launch/preSetupScript.sh ###'
27.08.2026 12:57 : ### Completed /domino/launch/preSetupScript.sh ###
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828248697227902
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.pre_setup_script.end&timestamp=1787828248697227902'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828248702035729
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pre_setup_script:end/trace?timestamp=1787828248702035729'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828248706698376
27.08.2026 12:57 : + export PIP_CONFIG_FILE=/mnt/pip.conf
27.08.2026 12:57 : + PIP_CONFIG_FILE=/mnt/pip.conf
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_custom_config:start/trace?timestamp=1787828248706698376'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828248711496360
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_custom_config:end/trace?timestamp=1787828248711496360'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828248716108954
27.08.2026 12:57 : + '[' -z '' ']'
27.08.2026 12:57 : + id domino
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.user_setup_section:start/trace?timestamp=1787828248716108954'
27.08.2026 12:57 : + USER=ubuntu
27.08.2026 12:57 : + '!id' ubuntu
27.08.2026 12:57 : + DOMINO_ID_UPDATED=false
27.08.2026 12:57 : + '[' -n 12574 ']'
27.08.2026 12:57 : ++ id -u ubuntu
27.08.2026 12:57 : + domino_previous_user_id=12574
27.08.2026 12:57 : + '[' 12574 '!=' 12574 ']'
27.08.2026 12:57 : + '[' -n 12574 ']'
27.08.2026 12:57 : ++ id -g ubuntu
27.08.2026 12:57 : + '[' 12574 '!=' 12574 ']'
27.08.2026 12:57 : + '[' false == true ']'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828248729882381
27.08.2026 12:57 : + echo '### Linking flyte config file to ~/.flyte/config.yaml' for user
27.08.2026 12:57 : ### Linking flyte config file to ~/.flyte/config.yaml for user
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.user_setup_section:end/trace?timestamp=1787828248729882381'
27.08.2026 12:57 : ++ cat /etc/passwd
27.08.2026 12:57 : ++ grep '^ubuntu:'
27.08.2026 12:57 : ++ cut -d: -f6
27.08.2026 12:57 : + user_home=/mnt
27.08.2026 12:57 : + mkdir -p /mnt/.flyte
27.08.2026 12:57 : + ln -s /domino/workflows/config.yaml /mnt/.flyte/config.yaml
27.08.2026 12:57 : ln: Already exists
27.08.2026 12:57 : + true
27.08.2026 12:57 : + echo '### Done linking flyte config file'
27.08.2026 12:57 : ### Done linking flyte config file
27.08.2026 12:57 : + [[ 1DOMINO_IS_WORKFLOW_JOB != true ]]
27.08.2026 12:57 : + [[ ! -d /workflow/inputs ]]
27.08.2026 12:57 : + [[ ! -d /workflow/outputs ]]
27.08.2026 12:57 : + cd /tmp
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828248744536558
27.08.2026 12:57 : + '[' -f /var/lib/domino/launch/.git-credentials ']'
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.git_creds_section:start/trace?timestamp=1787828248744536558'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828248749005810
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.git_creds_section:end/trace?timestamp=1787828248749005810'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828248753098313
27.08.2026 12:57 : + '[' -f /mnt/requirements.txt ']'
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_install_section:start/trace?timestamp=1787828248753098313'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828248757843303
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_install_section:end/trace?timestamp=1787828248757843303'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828248763250974
27.08.2026 12:57 : + '[' -f /domino/launch/postSetupScript.sh ']'
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.post_setup_script:start/trace?timestamp=1787828248763250974'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828248768976913
27.08.2026 12:57 : + cd /mnt
27.08.2026 12:57 : + echo '### SETUP PROCESS FINISHED ###\n'
27.08.2026 12:57 : ### SETUP PROCESS FINISHED ###\n
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.post_setup_script:end/trace?timestamp=1787828248768976913'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828248774343625
27.08.2026 12:57 : + chmod +x /var/lib/domino/launch/command.sh
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.run_command_section:start/trace?timestamp=1787828248774343625'
27.08.2026 12:57 : + declare -ir run_command_pid=185
27.08.2026 12:57 : + /var/lib/domino/launch/command.sh
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : Using curl at /opt/domino/bin/curl
27.08.2026 12:57 : + TIMESTAMP=1787828248783055325
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.run_command_section:end/trace?timestamp=1787828248783055325'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828248788359599
27.08.2026 12:57 : + wait 185
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script:end/trace?timestamp=1787828248788359599'
27.08.2026 12:57 : ### Executing /mnt/.domino/configure-spark-defaults.sh ###
27.08.2026 12:57 : ### Completed /mnt/.domino/configure-spark-defaults.sh ###
27.08.2026 12:57 : ### Executing /domino/launch/preRunScript.sh ###
27.08.2026 12:57 : ++ conda init bash
27.08.2026 12:57 : no change     /opt/conda/condabin/conda
27.08.2026 12:57 : no change     /opt/conda/bin/conda
27.08.2026 12:57 : no change     /opt/conda/bin/activate
27.08.2026 12:57 : no change     /opt/conda/bin/deactivate
27.08.2026 12:57 : no change     /opt/conda/etc/profile.d/conda.sh
27.08.2026 12:57 : no change     /opt/conda/etc/fish/conf.d/conda.fish
27.08.2026 12:57 : no change     /opt/conda/shell/condabin/Conda.psm1
27.08.2026 12:57 : no change     /opt/conda/shell/condabin/conda-hook.ps1
27.08.2026 12:57 : no change     /opt/conda/lib/python3.14/site-packages/xontrib/conda.xsh
27.08.2026 12:57 : no change     /opt/conda/etc/profile.d/conda.csh
27.08.2026 12:57 : no change     /mnt/.bashrc
27.08.2026 12:57 : No action taken.
27.08.2026 12:57 : ++ micromamba shell init --shell bash --root-prefix=/opt/conda
27.08.2026 12:57 : Modifying RC file "/mnt/.bashrc"
27.08.2026 12:57 : Generating config for root prefix [1m"/opt/conda"[0m
27.08.2026 12:57 : Setting mamba executable to: [1m"/usr/local/bin/micromamba"[0m
27.08.2026 12:57 : Adding (or replacing) the following in your "/mnt/.bashrc" file
27.08.2026 12:57 : # >>> mamba initialize >>>
27.08.2026 12:57 : # !! Contents within this block are managed by 'mamba init' !!
27.08.2026 12:57 : export MAMBA_EXE='/usr/local/bin/micromamba';
27.08.2026 12:57 : export MAMBA_ROOT_PREFIX='/opt/conda';
27.08.2026 12:57 : __mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
27.08.2026 12:57 : if [ $? -eq 0 ]; then
27.08.2026 12:57 :     eval "$__mamba_setup"
27.08.2026 12:57 : else
27.08.2026 12:57 :     alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
27.08.2026 12:57 : fi
27.08.2026 12:57 : unset __mamba_setup
27.08.2026 12:57 : # <<< mamba initialize <<<
27.08.2026 12:57 : + echo '### Completed /domino/launch/preRunScript.sh ###'
27.08.2026 12:57 : ### Completed /domino/launch/preRunScript.sh ###
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828249855714606
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.pre_run_script.end&timestamp=1787828249855714606'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828249860601322
27.08.2026 12:57 : + cd /mnt
27.08.2026 12:57 : + set +o errexit
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.command_launch_script.pre_run_script:end/trace?timestamp=1787828249860601322'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828249865589879
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.user_init:start/trace?timestamp=1787828249865589879'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828249870056316
27.08.2026 12:57 : + declare -ri run_command=232
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.user_init.wait_connectable:start/trace?timestamp=1787828249870056316'
27.08.2026 12:57 : + bash Production/DWA_production/run_scoring.sh Production/DWA_production/job_dwa.conf
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : ++ tee -a /mnt/results/stdout.txt
27.08.2026 12:57 : ++ tee -a /mnt/results/stderr.txt
27.08.2026 12:57 : + TIMESTAMP=1787828249874651114
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.final_run_command_issued&timestamp=1787828249874651114'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828249879556345
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.command_launch_script:end/trace?timestamp=1787828249879556345'
27.08.2026 12:57 : ++ date +%s%3N
27.08.2026 12:57 : + TIMESTAMP=1787828249885599619
27.08.2026 12:57 : + wait 232
27.08.2026 12:57 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container:end/trace?timestamp=1787828249885599619'
27.08.2026 12:57 : [2026-08-27 12:57:29] === rubin Production-Scoring: Start (ref=master, env=prod) ===
27.08.2026 12:57 : [2026-08-27 12:57:29] WARNUNG: GIT_REF='master' ist ein Branch — jeder Lauf zieht den jeweils neuesten Commit.
27.08.2026 12:57 : [2026-08-27 12:57:29]          Für reproduzierbare Prod-Läufe GIT_REF auf Tag/SHA pinnen (und REQUIRE_PINNED_REF=1 setzen).
27.08.2026 12:57 : [2026-08-27 12:57:29] --- SSH vorbereiten ---
27.08.2026 12:57 : [2026-08-27 12:57:29] SSH-Key: /mnt/.ssh/id_rsa
27.08.2026 12:57 : git version 2.53.0
27.08.2026 12:57 : [2026-08-27 12:57:29] --- Git Clone (master) ---
27.08.2026 12:57 : Cloning into '/home/ubuntu/rubin_scoring'...
27.08.2026 12:57 : ** WARNING: connection is not using a post-quantum key exchange algorithm.
27.08.2026 12:57 : ** This session may be vulnerable to "store now, decrypt later" attacks.
27.08.2026 12:57 : ** The server may need to be upgraded. See https://openssh.com/pq.html
27.08.2026 12:57 : [2026-08-27 12:57:30] WARNUNG: Gestartetes Skript weicht vom Repo-Master (master) ab — FS-Kopie bitte synchronisieren:
27.08.2026 12:57 : [2026-08-27 12:57:30]          laufend: /mnt/Production/DWA_production/run_scoring.sh
27.08.2026 12:57 : [2026-08-27 12:57:30]          Master:  /home/ubuntu/rubin_scoring/production/run_scoring.sh
27.08.2026 12:57 : [2026-08-27 12:57:30] WARNUNG: Job-Datei job_dwa.conf weicht vom Repo-Master (master) ab — FS-Kopie bitte synchronisieren:
27.08.2026 12:57 : [2026-08-27 12:57:30]          laufend: /mnt/Production/DWA_production/job_dwa.conf
27.08.2026 12:57 : [2026-08-27 12:57:30]          Master:  /home/ubuntu/rubin_scoring/production/jobs/job_dwa.conf
27.08.2026 12:57 : [2026-08-27 12:57:30] Repo-Stand: 90d06b4 (2026-08-27)
27.08.2026 12:57 : [2026-08-27 12:57:30] --- Pixi Install (-e prod) ---
27.08.2026 12:57 : pixi 0.72.0
27.08.2026 12:57 : [2026-08-27 12:57:30] pixi.lock gefunden → deterministische Installation (--frozen).
27.08.2026 12:57 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
27.08.2026 12:57 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
27.08.2026 12:57 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
27.08.2026 12:57 : ✔ The prod environment has been installed.
27.08.2026 12:57 : [2026-08-27 12:57:38] --- Scoring: 1 Config(s): production/scoring_un.yml ---
27.08.2026 12:57 : [2026-08-27 12:57:38] --- Score starten: production/scoring_un.yml ---
27.08.2026 12:57 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
27.08.2026 12:57 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
27.08.2026 12:57 :  WARN 'tls-root-certs = "all"' is deprecated: merging webpki and system roots is no longer supported. Pick one of 'webpki' or 'system', or set SSL_CERT_FILE / SSL_CERT_DIR. The value falls back to 'system' for now.
27.08.2026 12:57 :  WARN the lock file is up-to-date but uses an older format (v6), run `pixi lock` to upgrade to v7 for improved reproducibility
27.08.2026 13:06 : 2026-08-27 13:06:42,236 INFO [rubin.scoring] Input: 11799629 Zeilen, 110 Spalten (spalten-gepruned, Union über alle Bundles) (/domino/edv/pvc-hf1kundehuk/ScoringdatensatzProduktiv/scoringdatensatz.sas7bdat) — 1 Score(s) gegen diese eine Ladung
27.08.2026 13:06 : 2026-08-27 13:06:42,479 INFO [rubin.scoring] Bundle-Modelle (per YAML frei wählbar, unabhängig vom Champion): ['CausalForest', 'CausalForestDML', 'DRLearner', 'Ensemble', 'NonParamDML', 'SurrogateTree', 'TLearner', 'XLearner'] | Champion: CausalForestDML
27.08.2026 13:07 : 2026-08-27 13:07:56,718 WARNING [rubin.scoring] Erhöhte -1-Raten (unbekannte Kategorien/Missings) — mögliches Drift-Signal: {'PLZ1': 0.067271, 'PLZ_2A': 0.065262, 'PLZ_3A': 0.065272, 'REGIO_GS_BEREICH_FS': 0.215724, 'AKQ_BERUF_STATUS': 0.05474, 'AKQ_FAMILIENSTAND_FS': 0.832616, 'GINT_KFZ_SCHLUESSEL_NR_HERST': 0.695025, 'ALTERSGRUPPE': 0.099971, 'BDL': 0.809953}
27.08.2026 13:08 : 2026-08-27 13:08:00,633 INFO [matplotlib.font_manager] generated new fontManager
27.08.2026 13:08 : Traceback (most recent call last):
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/production/run_scoring.py", line 683, in <module>
27.08.2026 13:08 :     main()
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/production/run_scoring.py", line 679, in main
27.08.2026 13:08 :     run_scoring(cfg)
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/production/run_scoring.py", line 600, in run_scoring
27.08.2026 13:08 :     out, pipe, core = score_dataframe(df, e_cfg, day_stamp, pipe=pipe)
27.08.2026 13:08 :                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/production/run_scoring.py", line 423, in score_dataframe
27.08.2026 13:08 :     for col, vals in _score_columns("SCORE_P", predict_in_batches(pipe.models[p_name], Xp, batch)).items():
27.08.2026 13:08 :                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/production/run_scoring.py", line 291, in predict_in_batches
27.08.2026 13:08 :     parts = [np.asarray(_predict_effect(model, X.iloc[i:i + batch_size]))
27.08.2026 13:08 :                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/rubin/training.py", line 58, in _predict_effect
27.08.2026 13:08 :     pred = model.const_marginal_effect(X)
27.08.2026 13:08 :            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/econml/_ortho_learner.py", line 1002, in const_marginal_effect
27.08.2026 13:08 :     return self._ortho_learner_model_final.predict(X)
27.08.2026 13:08 :            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/econml/dml/_rlearner.py", line 106, in predict
27.08.2026 13:08 :     return self._model_final.predict(X)
27.08.2026 13:08 :            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/econml/dml/causal_forest.py", line 96, in predict
27.08.2026 13:08 :     return self._model.predict(self._combine(X, fitting=False)).reshape((-1,) + self._d_y + self._d_t)
27.08.2026 13:08 :            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/econml/grf/classes.py", line 47, in predict
27.08.2026 13:08 :     pred = [estimator.predict(X, interval=interval, alpha=alpha) for estimator in self.estimators_]
27.08.2026 13:08 :             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/econml/grf/_base_grf.py", line 857, in predict
27.08.2026 13:08 :     y_hat = self.predict_full(X, interval=False)
27.08.2026 13:08 :             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/econml/grf/_base_grf.py", line 825, in predict_full
27.08.2026 13:08 :     return self._predict_point_and_var(X, full=True, point=True, var=False)
27.08.2026 13:08 :            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/econml/grf/_base_grf.py", line 706, in _predict_point_and_var
27.08.2026 13:08 :     alpha, jac = self.predict_alpha_and_jac(X)
27.08.2026 13:08 :                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/econml/grf/_base_grf.py", line 640, in predict_alpha_and_jac
27.08.2026 13:08 :     X = self._validate_X_predict(X)
27.08.2026 13:08 :         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/econml/grf/_base_grf.py", line 472, in _validate_X_predict
27.08.2026 13:08 :     return self.estimators_[0]._validate_X_predict(X, check_input=True)
27.08.2026 13:08 :            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/econml/tree/_tree_classes.py", line 284, in _validate_X_predict
27.08.2026 13:08 :     X = check_array(X, dtype=DTYPE, accept_sparse=False, ensure_min_features=0)
27.08.2026 13:08 :         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/sklearn/utils/validation.py", line 1055, in check_array
27.08.2026 13:08 :     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
27.08.2026 13:08 :             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 :   File "/home/ubuntu/rubin_scoring/.pixi/envs/prod/lib/python3.12/site-packages/sklearn/utils/_array_api.py", line 839, in _asarray_with_order
27.08.2026 13:08 :     array = numpy.asarray(array, order=order, dtype=dtype)
27.08.2026 13:08 :             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27.08.2026 13:08 : ValueError: could not convert string to float: 'V'
27.08.2026 13:08 : [2026-08-27 13:08:07] --- Score FEHLGESCHLAGEN (rc=1): production/scoring_un.yml ---
27.08.2026 13:08 : + exitcode=1
27.08.2026 13:08 : + '[' 1 -eq 0 ']'
27.08.2026 13:08 : + sleep 2
27.08.2026 13:08 : Evaluating cleanup command on EXIT with exit code 1: $CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true
27.08.2026 13:08 : + exit 1
27.08.2026 13:08 : + exit_logging
27.08.2026 13:08 : + local -r code_at_exit=1
27.08.2026 13:08 : + '[' -n '$CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true' ']'
27.08.2026 13:08 : + local max_retries=10
27.08.2026 13:08 : + local retry_delay=3
27.08.2026 13:08 : + echo 'Evaluating cleanup command on EXIT with exit code 1: $CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true'
27.08.2026 13:08 : + local n=1
27.08.2026 13:08 : + true
27.08.2026 13:08 : + eval '$CURL' -sS -H '"X-Api-Token:' '$DOMINO_EXECUTOR_TERMINATION_TOKEN"' -X POST 'http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit' '||' true
27.08.2026 13:08 : ++ /opt/domino/bin/curl -sS -H 'X-Api-Token: e0a084c6c2e3bba20199fcbd01de7be294df3a878128c4d5366b2845bd0432bd' -X POST 'http://127.0.0.1:9000/executor/exit?exitCode=1'
27.08.2026 13:08 : + break
27.08.2026 13:08 : + [[ 1 =~ ^(0|137|143)$ ]]
27.08.2026 13:08 : ++ tee -a /mnt/results/stdout.txt
27.08.2026 13:08 : + sleep 0.5
27.08.2026 13:08 : + echo 'Failed with exit code: 1'
27.08.2026 13:08 : Failed with exit code: 1
27.08.2026 13:08 : + exit 1
27.08.2026 13:08 : Caught termination signal!
27.08.2026 13:08 : -- killed by pod termination --










#GINT_KFZ_JAHRESFAHRKILOMETER Vorverarbeitung
df_m['GINT_KFZ_JAHRESFAHRKILOMETER'] = df_m['GINT_KFZ_JAHRESFAHRKILOMETER'].replace(['V', 'nan'], '73')
df_m['GINT_KFZ_JAHRESFAHRKILOMETER'] = df_m['GINT_KFZ_JAHRESFAHRKILOMETER'].fillna('73')

#NANs mit median_values füllen
df_m = df_m.fillna(median_values)
    
#Über alle Spalten iterieren und Encoding anwenden
for spalte, codierung in encoding_brief.items():     
    if spalte in df_m.columns:         
        df_m[spalte] = df_m[spalte].replace(codierung)

#Gesellschaft soll in Ursprungsform bleiben, um die Information darüber zurückgeben zu können
for col in df_m.columns:
    if col != 'GESELLSCHAFT_FS':
        df_m[col] = pd.to_numeric(df_m[col], errors='coerce').fillna(0)
    else:
        pass

# Schränken Sie den df_m auf die ausgewählten Spalten ein und ändern Sie die Datentypen
df_m['GINT_ZULASSUNG_DT'] = df_m['GINT_ZULASSUNG_DT'].clip(lower=0) #Probleme, da unerwartet negative Werte auftauchen
df_m = df_m.astype(column_datatypes)

#Infinity Values entfernen
df_m['GINT_KFZ_JAHRESBEITRAG_KFZ'] = df_m['GINT_KFZ_JAHRESBEITRAG_KFZ'].replace([np.inf, -np.inf], median_values['GINT_KFZ_JAHRESBEITRAG_KFZ'])
