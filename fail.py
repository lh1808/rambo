27.08.2026 11:47 : Using curl at /opt/domino/bin/curl
27.08.2026 11:47 : Preparing working directory.
27.08.2026 11:47 : -- starting API proxy --
27.08.2026 11:47 : Starting periodic API token refresh.
27.08.2026 11:47 : ###############################################################################################################
27.08.2026 11:47 : #                                                                                                             #
27.08.2026 11:47 : # DEPRECATION WARNING:                                                                                        #
27.08.2026 11:47 : #                                                                                                             #
27.08.2026 11:47 : # Availability of $DOMINO_TOKEN_FILE is getting deprecated and will be removed in a future release.           #
27.08.2026 11:47 : #                                                                                                             #
27.08.2026 11:47 : # Please consider using the API Proxy:                                                                        #
27.08.2026 11:47 : # https://docs.dominodatalab.com/en/latest/user_guide/ddf8eb/use-the-api-proxy-for-domino-api-authentication/ #
27.08.2026 11:47 : #                                                                                                             #
27.08.2026 11:47 : ###############################################################################################################
27.08.2026 11:47 : /app/.venv/lib/python3.12/site-packages/tzlocal/unix.py:207: UserWarning: Can not find any timezone configuration, defaulting to UTC.
27.08.2026 11:47 :   warnings.warn("Can not find any timezone configuration, defaulting to UTC.")
27.08.2026 11:47 : Started API Proxy on port 8899 with 2 worker processes
27.08.2026 11:48 : ### SETUP PROCESS STARTED ###
27.08.2026 11:48 : Watching for changes in /var/lib/domino/launch/poison-pill
27.08.2026 11:48 : ### Executing /domino/launch/preSetupScript.sh ###
27.08.2026 11:48 : ### Completed /domino/launch/preSetupScript.sh ###
27.08.2026 11:48 : + echo '### Completed /domino/launch/preSetupScript.sh ###'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824086200730320
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.pre_setup_script.end&timestamp=1787824086200730320'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824086205272421
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pre_setup_script:end/trace?timestamp=1787824086205272421'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824086209723888
27.08.2026 11:48 : + export PIP_CONFIG_FILE=/mnt/pip.conf
27.08.2026 11:48 : + PIP_CONFIG_FILE=/mnt/pip.conf
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_custom_config:start/trace?timestamp=1787824086209723888'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824086214394963
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_custom_config:end/trace?timestamp=1787824086214394963'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824086219066365
27.08.2026 11:48 : + '[' -z '' ']'
27.08.2026 11:48 : + id domino
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.user_setup_section:start/trace?timestamp=1787824086219066365'
27.08.2026 11:48 : + USER=ubuntu
27.08.2026 11:48 : + '!id' ubuntu
27.08.2026 11:48 : + DOMINO_ID_UPDATED=false
27.08.2026 11:48 : + '[' -n 12574 ']'
27.08.2026 11:48 : ++ id -u ubuntu
27.08.2026 11:48 : + domino_previous_user_id=12574
27.08.2026 11:48 : + '[' 12574 '!=' 12574 ']'
27.08.2026 11:48 : + '[' -n 12574 ']'
27.08.2026 11:48 : ++ id -g ubuntu
27.08.2026 11:48 : + '[' 12574 '!=' 12574 ']'
27.08.2026 11:48 : + '[' false == true ']'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824086235058354
27.08.2026 11:48 : + echo '### Linking flyte config file to ~/.flyte/config.yaml' for user
27.08.2026 11:48 : ### Linking flyte config file to ~/.flyte/config.yaml for user
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.user_setup_section:end/trace?timestamp=1787824086235058354'
27.08.2026 11:48 : ++ cat /etc/passwd
27.08.2026 11:48 : ++ grep '^ubuntu:'
27.08.2026 11:48 : ++ cut -d: -f6
27.08.2026 11:48 : + user_home=/mnt
27.08.2026 11:48 : + mkdir -p /mnt/.flyte
27.08.2026 11:48 : + ln -s /domino/workflows/config.yaml /mnt/.flyte/config.yaml
27.08.2026 11:48 : ln: Already exists
27.08.2026 11:48 : + true
27.08.2026 11:48 : + echo '### Done linking flyte config file'
27.08.2026 11:48 : ### Done linking flyte config file
27.08.2026 11:48 : + [[ 1DOMINO_IS_WORKFLOW_JOB != true ]]
27.08.2026 11:48 : + [[ ! -d /workflow/inputs ]]
27.08.2026 11:48 : + [[ ! -d /workflow/outputs ]]
27.08.2026 11:48 : + cd /tmp
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824086251876805
27.08.2026 11:48 : + '[' -f /var/lib/domino/launch/.git-credentials ']'
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.git_creds_section:start/trace?timestamp=1787824086251876805'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824086257073740
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.git_creds_section:end/trace?timestamp=1787824086257073740'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824086262516673
27.08.2026 11:48 : + '[' -f /mnt/requirements.txt ']'
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_install_section:start/trace?timestamp=1787824086262516673'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824086268005736
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.pip_install_section:end/trace?timestamp=1787824086268005736'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824086273279035
27.08.2026 11:48 : + '[' -f /domino/launch/postSetupScript.sh ']'
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.post_setup_script:start/trace?timestamp=1787824086273279035'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824086278229283
27.08.2026 11:48 : + cd /mnt
27.08.2026 11:48 : + echo '### SETUP PROCESS FINISHED ###\n'
27.08.2026 11:48 : ### SETUP PROCESS FINISHED ###\n
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.post_setup_script:end/trace?timestamp=1787824086278229283'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824086283217777
27.08.2026 11:48 : + chmod +x /var/lib/domino/launch/command.sh
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.run_command_section:start/trace?timestamp=1787824086283217777'
27.08.2026 11:48 : + declare -ir run_command_pid=187
27.08.2026 11:48 : + /var/lib/domino/launch/command.sh
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : Using curl at /opt/domino/bin/curl
27.08.2026 11:48 : + TIMESTAMP=1787824086291328161
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script.run_command_section:end/trace?timestamp=1787824086291328161'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824086296405777
27.08.2026 11:48 : + wait 187
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.start_run_script:end/trace?timestamp=1787824086296405777'
27.08.2026 11:48 : ### Executing /mnt/.domino/configure-spark-defaults.sh ###
27.08.2026 11:48 : ### Completed /mnt/.domino/configure-spark-defaults.sh ###
27.08.2026 11:48 : ### Executing /domino/launch/preRunScript.sh ###
27.08.2026 11:48 : ++ conda init bash
27.08.2026 11:48 : no change     /opt/conda/condabin/conda
27.08.2026 11:48 : no change     /opt/conda/bin/conda
27.08.2026 11:48 : no change     /opt/conda/bin/activate
27.08.2026 11:48 : no change     /opt/conda/bin/deactivate
27.08.2026 11:48 : no change     /opt/conda/etc/profile.d/conda.sh
27.08.2026 11:48 : no change     /opt/conda/etc/fish/conf.d/conda.fish
27.08.2026 11:48 : no change     /opt/conda/shell/condabin/Conda.psm1
27.08.2026 11:48 : no change     /opt/conda/shell/condabin/conda-hook.ps1
27.08.2026 11:48 : no change     /opt/conda/lib/python3.14/site-packages/xontrib/conda.xsh
27.08.2026 11:48 : no change     /opt/conda/etc/profile.d/conda.csh
27.08.2026 11:48 : no change     /mnt/.bashrc
27.08.2026 11:48 : No action taken.
27.08.2026 11:48 : ++ micromamba shell init --shell bash --root-prefix=/opt/conda
27.08.2026 11:48 : Modifying RC file "/mnt/.bashrc"
27.08.2026 11:48 : Generating config for root prefix [1m"/opt/conda"[0m
27.08.2026 11:48 : Setting mamba executable to: [1m"/usr/local/bin/micromamba"[0m
27.08.2026 11:48 : Adding (or replacing) the following in your "/mnt/.bashrc" file
27.08.2026 11:48 : # >>> mamba initialize >>>
27.08.2026 11:48 : # !! Contents within this block are managed by 'mamba init' !!
27.08.2026 11:48 : export MAMBA_EXE='/usr/local/bin/micromamba';
27.08.2026 11:48 : export MAMBA_ROOT_PREFIX='/opt/conda';
27.08.2026 11:48 : __mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
27.08.2026 11:48 : if [ $? -eq 0 ]; then
27.08.2026 11:48 :     eval "$__mamba_setup"
27.08.2026 11:48 : else
27.08.2026 11:48 :     alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
27.08.2026 11:48 : fi
27.08.2026 11:48 : unset __mamba_setup
27.08.2026 11:48 : # <<< mamba initialize <<<
27.08.2026 11:48 : ### Completed /domino/launch/preRunScript.sh ###
27.08.2026 11:48 : + echo '### Completed /domino/launch/preRunScript.sh ###'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824087216212665
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.pre_run_script.end&timestamp=1787824087216212665'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824087221474115
27.08.2026 11:48 : + cd /mnt
27.08.2026 11:48 : + set +o errexit
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.command_launch_script.pre_run_script:end/trace?timestamp=1787824087221474115'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824087226172196
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.user_init:start/trace?timestamp=1787824087226172196'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824087231929109
27.08.2026 11:48 : + declare -ri run_command=234
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.user_init.wait_connectable:start/trace?timestamp=1787824087231929109'
27.08.2026 11:48 : + bash Production/DWA_production/run_scoring.sh Production/DWA_production/job_dwa.conf
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : ++ tee -a /mnt/results/stdout.txt
27.08.2026 11:48 : ++ tee -a /mnt/results/stderr.txt
27.08.2026 11:48 : + TIMESTAMP=1787824087237934047
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/runBootSequenceEvent?eventKey=run.boot_sequence.final_run_command_issued&timestamp=1787824087237934047'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : + TIMESTAMP=1787824087243357146
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container.command_launch_script:end/trace?timestamp=1787824087243357146'
27.08.2026 11:48 : ++ date +%s%3N
27.08.2026 11:48 : Production/DWA_production/run_scoring.sh: line 34: set: pipefail
: invalid option name
27.08.2026 11:48 : + TIMESTAMP=1787824087248858558
27.08.2026 11:48 : + wait 234
27.08.2026 11:48 : + /opt/domino/bin/curl -s -X POST 'http://127.0.0.1:9000/executor/metrics/launch.run_container:end/trace?timestamp=1787824087248858558'
27.08.2026 11:48 : + exitcode=2
27.08.2026 11:48 : + '[' 2 -eq 0 ']'
27.08.2026 11:48 : + sleep 2
27.08.2026 11:48 : + exit 2
27.08.2026 11:48 : + exit_logging
27.08.2026 11:48 : + local -r code_at_exit=2
27.08.2026 11:48 : + '[' -n '$CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true' ']'
27.08.2026 11:48 : + local max_retries=10
27.08.2026 11:48 : + local retry_delay=3
27.08.2026 11:48 : + echo 'Evaluating cleanup command on EXIT with exit code 2: $CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true'
27.08.2026 11:48 : + local n=1
27.08.2026 11:48 : + true
27.08.2026 11:48 : + eval '$CURL' -sS -H '"X-Api-Token:' '$DOMINO_EXECUTOR_TERMINATION_TOKEN"' -X POST 'http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit' '||' true
27.08.2026 11:48 : Evaluating cleanup command on EXIT with exit code 2: $CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true
27.08.2026 11:48 : ++ /opt/domino/bin/curl -sS -H 'X-Api-Token: e2cf8e5ac75f99603717eccc23511502c435c33e007630e79f6876a002ff1536' -X POST 'http://127.0.0.1:9000/executor/exit?exitCode=2'
27.08.2026 11:48 : + break
27.08.2026 11:48 : + [[ 2 =~ ^(0|137|143)$ ]]
27.08.2026 11:48 : ++ tee -a /mnt/results/stdout.txt
27.08.2026 11:48 : + sleep 0.5
27.08.2026 11:48 : + echo 'Failed with exit code: 2'
27.08.2026 11:48 : Failed with exit code: 2
27.08.2026 11:48 : + exit 1
27.08.2026 11:48 : Caught termination signal!
27.08.2026 11:48 : -- killed by pod termination --
