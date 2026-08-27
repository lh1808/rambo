27.08.2026 12:37 : Using curl at /opt/domino/bin/curl
27.08.2026 12:37 : Preparing working directory.
27.08.2026 12:37 : -- starting API proxy --
27.08.2026 12:37 : Starting periodic API token refresh.
27.08.2026 12:37 : ###############################################################################################################
27.08.2026 12:37 : #                                                                                                             #
27.08.2026 12:37 : # DEPRECATION WARNING:                                                                                        #
27.08.2026 12:37 : #                                                                                                             #
27.08.2026 12:37 : # Availability of $DOMINO_TOKEN_FILE is getting deprecated and will be removed in a future release.           #
27.08.2026 12:37 : #                                                                                                             #
27.08.2026 12:37 : # Please consider using the API Proxy:                                                                        #
27.08.2026 12:37 : # https://docs.dominodatalab.com/en/latest/user_guide/ddf8eb/use-the-api-proxy-for-domino-api-authentication/ #
27.08.2026 12:37 : #                                                                                                             #
27.08.2026 12:37 : ###############################################################################################################
27.08.2026 12:37 : /app/.venv/lib/python3.12/site-packages/tzlocal/unix.py:207: UserWarning: Can not find any timezone configuration, defaulting to UTC.
27.08.2026 12:37 :   warnings.warn("Can not find any timezone configuration, defaulting to UTC.")
27.08.2026 12:37 : Started API Proxy on port 8899 with 2 worker processes
27.08.2026 12:37 : ### SETUP PROCESS STARTED ###
27.08.2026 12:37 : Watching for changes in /var/lib/domino/launch/poison-pill
27.08.2026 12:37 : ### Executing /domino/launch/preSetupScript.sh ###
27.08.2026 12:37 : + echo '### Completed /domino/launch/preSetupScript.sh ###'
27.08.2026 12:37 : ### Completed /domino/launch/preSetupScript.sh ###
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827034585019748
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.pre_setup_script.end&timestamp=1787827034585019748'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827034589374311
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pre_setup_script:end/trace?timestamp=1787827034589374311'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827034594013920
27.08.2026 12:37 : + export PIP_CONFIG_FILE=/mnt/pip.conf
27.08.2026 12:37 : + PIP_CONFIG_FILE=/mnt/pip.conf
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_custom_config:start/trace?timestamp=1787827034594013920'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827034598386934
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_custom_config:end/trace?timestamp=1787827034598386934'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827034602690746
27.08.2026 12:37 : + '[' -z '' ']'
27.08.2026 12:37 : + id domino
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.user_setup_section:start/trace?timestamp=1787827034602690746'
27.08.2026 12:37 : + USER=ubuntu
27.08.2026 12:37 : + '!id' ubuntu
27.08.2026 12:37 : + DOMINO_ID_UPDATED=false
27.08.2026 12:37 : + '[' -n 12574 ']'
27.08.2026 12:37 : ++ id -u ubuntu
27.08.2026 12:37 : + domino_previous_user_id=12574
27.08.2026 12:37 : + '[' 12574 '!=' 12574 ']'
27.08.2026 12:37 : + '[' -n 12574 ']'
27.08.2026 12:37 : ++ id -g ubuntu
27.08.2026 12:37 : + '[' 12574 '!=' 12574 ']'
27.08.2026 12:37 : + '[' false == true ']'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827034616592309
27.08.2026 12:37 : + echo '### Linking flyte config file to ~/.flyte/config.yaml' for user
27.08.2026 12:37 : ### Linking flyte config file to ~/.flyte/config.yaml for user
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.user_setup_section:end/trace?timestamp=1787827034616592309'
27.08.2026 12:37 : ++ cat /etc/passwd
27.08.2026 12:37 : ++ grep '^ubuntu:'
27.08.2026 12:37 : ++ cut -d: -f6
27.08.2026 12:37 : + user_home=/mnt
27.08.2026 12:37 : + mkdir -p /mnt/.flyte
27.08.2026 12:37 : + ln -s /domino/workflows/config.yaml /mnt/.flyte/config.yaml
27.08.2026 12:37 : ln: Already exists
27.08.2026 12:37 : + true
27.08.2026 12:37 : ### Done linking flyte config file
27.08.2026 12:37 : + echo '### Done linking flyte config file'
27.08.2026 12:37 : + [[ 1DOMINO_IS_WORKFLOW_JOB != true ]]
27.08.2026 12:37 : + [[ ! -d /workflow/inputs ]]
27.08.2026 12:37 : + [[ ! -d /workflow/outputs ]]
27.08.2026 12:37 : + cd /tmp
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827034631706713
27.08.2026 12:37 : + '[' -f /var/lib/domino/launch/.git-credentials ']'
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.git_creds_section:start/trace?timestamp=1787827034631706713'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827034636226856
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.git_creds_section:end/trace?timestamp=1787827034636226856'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827034640813427
27.08.2026 12:37 : + '[' -f /mnt/requirements.txt ']'
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_install_section:start/trace?timestamp=1787827034640813427'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827034645553388
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_install_section:end/trace?timestamp=1787827034645553388'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827034650353707
27.08.2026 12:37 : + '[' -f /domino/launch/postSetupScript.sh ']'
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.post_setup_script:start/trace?timestamp=1787827034650353707'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827034655702710
27.08.2026 12:37 : + cd /mnt
27.08.2026 12:37 : + echo '### SETUP PROCESS FINISHED ###\n'
27.08.2026 12:37 : ### SETUP PROCESS FINISHED ###\n
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.post_setup_script:end/trace?timestamp=1787827034655702710'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827034660843072
27.08.2026 12:37 : + chmod +x /var/lib/domino/launch/command.sh
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.run_command_section:start/trace?timestamp=1787827034660843072'
27.08.2026 12:37 : + declare -ir run_command_pid=184
27.08.2026 12:37 : + /var/lib/domino/launch/command.sh
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : Using curl at /opt/domino/bin/curl
27.08.2026 12:37 : + TIMESTAMP=1787827034669797327
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.run_command_section:end/trace?timestamp=1787827034669797327'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827034675041955
27.08.2026 12:37 : + wait 184
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script:end/trace?timestamp=1787827034675041955'
27.08.2026 12:37 : ### Executing /mnt/.domino/configure-spark-defaults.sh ###
27.08.2026 12:37 : ### Completed /mnt/.domino/configure-spark-defaults.sh ###
27.08.2026 12:37 : ++ conda init bash
27.08.2026 12:37 : ### Executing /domino/launch/preRunScript.sh ###
27.08.2026 12:37 : no change     /opt/conda/condabin/conda
27.08.2026 12:37 : no change     /opt/conda/bin/conda
27.08.2026 12:37 : no change     /opt/conda/bin/activate
27.08.2026 12:37 : no change     /opt/conda/bin/deactivate
27.08.2026 12:37 : no change     /opt/conda/etc/profile.d/conda.sh
27.08.2026 12:37 : no change     /opt/conda/etc/fish/conf.d/conda.fish
27.08.2026 12:37 : no change     /opt/conda/shell/condabin/Conda.psm1
27.08.2026 12:37 : no change     /opt/conda/shell/condabin/conda-hook.ps1
27.08.2026 12:37 : no change     /opt/conda/lib/python3.14/site-packages/xontrib/conda.xsh
27.08.2026 12:37 : no change     /opt/conda/etc/profile.d/conda.csh
27.08.2026 12:37 : no change     /mnt/.bashrc
27.08.2026 12:37 : No action taken.
27.08.2026 12:37 : ++ micromamba shell init --shell bash --root-prefix=/opt/conda
27.08.2026 12:37 : Modifying RC file "/mnt/.bashrc"
27.08.2026 12:37 : Generating config for root prefix [1m"/opt/conda"[0m
27.08.2026 12:37 : Setting mamba executable to: [1m"/usr/local/bin/micromamba"[0m
27.08.2026 12:37 : Adding (or replacing) the following in your "/mnt/.bashrc" file
27.08.2026 12:37 : # >>> mamba initialize >>>
27.08.2026 12:37 : # !! Contents within this block are managed by 'mamba init' !!
27.08.2026 12:37 : export MAMBA_EXE='/usr/local/bin/micromamba';
27.08.2026 12:37 : export MAMBA_ROOT_PREFIX='/opt/conda';
27.08.2026 12:37 : __mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
27.08.2026 12:37 : if [ $? -eq 0 ]; then
27.08.2026 12:37 :     eval "$__mamba_setup"
27.08.2026 12:37 : else
27.08.2026 12:37 :     alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
27.08.2026 12:37 : fi
27.08.2026 12:37 : unset __mamba_setup
27.08.2026 12:37 : # <<< mamba initialize <<<
27.08.2026 12:37 : ### Completed /domino/launch/preRunScript.sh ###
27.08.2026 12:37 : + echo '### Completed /domino/launch/preRunScript.sh ###'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827035644381388
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.pre_run_script.end&timestamp=1787827035644381388'
27.08.2026 12:37 : + TIMESTAMP=1787827035651958316
27.08.2026 12:37 : + cd /mnt
27.08.2026 12:37 : + set +o errexit
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.command_launch_script.pre_run_script:end/trace?timestamp=1787827035651958316'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827035657123922
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.user_init:start/trace?timestamp=1787827035657123922'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827035662178409
27.08.2026 12:37 : + declare -ri run_command=231
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.user_init.wait_connectable:start/trace?timestamp=1787827035662178409'
27.08.2026 12:37 : + bash Production/DWA_production/run_scoring.sh Production/DWA_production/job_dwa.conf
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : ++ tee -a /mnt/results/stdout.txt
27.08.2026 12:37 : ++ tee -a /mnt/results/stderr.txt
27.08.2026 12:37 : + TIMESTAMP=1787827035667326099
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.final_run_command_issued&timestamp=1787827035667326099'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827035672151488
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.command_launch_script:end/trace?timestamp=1787827035672151488'
27.08.2026 12:37 : ++ date +%s%3N
27.08.2026 12:37 : + TIMESTAMP=1787827035676904002
27.08.2026 12:37 : + wait 231
27.08.2026 12:37 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container:end/trace?timestamp=1787827035676904002'
27.08.2026 12:37 : /mnt/Production/DWA_production/job_dwa.conf: line 13: production/scoring_un.yml: No such file or directory
27.08.2026 12:37 : + exitcode=127
27.08.2026 12:37 : + '[' 127 -eq 0 ']'
27.08.2026 12:37 : + sleep 2
27.08.2026 12:37 : + exit 127
27.08.2026 12:37 : + exit_logging
27.08.2026 12:37 : + local -r code_at_exit=127
27.08.2026 12:37 : + '[' -n '$CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true' ']'
27.08.2026 12:37 : + local max_retries=10
27.08.2026 12:37 : + local retry_delay=3
27.08.2026 12:37 : + echo 'Evaluating cleanup command on EXIT with exit code 127: $CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true'
27.08.2026 12:37 : + local n=1
27.08.2026 12:37 : + true
27.08.2026 12:37 : Evaluating cleanup command on EXIT with exit code 127: $CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true
27.08.2026 12:37 : + eval '$CURL' -sS -H '"X-Api-Token:' '$DOMINO_EXECUTOR_TERMINATION_TOKEN"' -X POST 'http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit' '||' true
27.08.2026 12:37 : ++ /opt/domino/bin/curl -sS -H 'X-Api-Token: 0e6272b56768b40597e7ae364636970065e5840459d98970c21bca117fb1c123' -X POST 'http://127.0.0.1:9000/executor/exit?exitCode=127'
27.08.2026 12:37 : + break
27.08.2026 12:37 : + [[ 127 =~ ^(0|137|143)$ ]]
27.08.2026 12:37 : ++ tee -a /mnt/results/stdout.txt
27.08.2026 12:37 : + sleep 0.5
27.08.2026 12:37 : + echo 'Failed with exit code: 127'
27.08.2026 12:37 : Failed with exit code: 127
27.08.2026 12:37 : + exit 1
27.08.2026 12:37 : Caught termination signal!
27.08.2026 12:37 : -- killed by pod termination --
