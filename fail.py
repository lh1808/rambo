23.10.2026 17:24 : Collecting pyreadstat>=1.2 (from rubin==1.0.0)
23.10.2026 17:24 :   Downloading pyreadstat-1.3.7.tar.gz (549 kB)
23.10.2026 17:24 :      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 549.7/549.7 kB 65.0 MB/s  0:00:00
23.10.2026 17:24 :   Preparing metadata (pyproject.toml): started
23.10.2026 17:24 :   Preparing metadata (pyproject.toml): finished with status 'error'
23.10.2026 17:24 :   error: subprocess-exited-with-error
23.10.2026 17:24 :   
23.10.2026 17:24 :   × Preparing metadata (pyproject.toml) did not run successfully.
23.10.2026 17:24 :   │ exit code: 1
23.10.2026 17:24 :   ╰─> [13 lines of output]
23.10.2026 17:24 :       Traceback (most recent call last):
23.10.2026 17:24 :         File "/opt/conda/envs/hf1/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
23.10.2026 17:24 :           main()
23.10.2026 17:24 :         File "/opt/conda/envs/hf1/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
23.10.2026 17:24 :           json_out["return_val"] = hook(**hook_input["kwargs"])
23.10.2026 17:24 :         File "/opt/conda/envs/hf1/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 175, in prepare_metadata_for_build_wheel
23.10.2026 17:24 :           return hook(metadata_directory, config_settings)
23.10.2026 17:24 :         File "/opt/conda/envs/hf1/lib/python3.10/site-packages/setuptools/build_meta.py", line 380, in prepare_metadata_for_build_wheel
23.10.2026 17:24 :           self.run_setup()
23.10.2026 17:24 :         File "/opt/conda/envs/hf1/lib/python3.10/site-packages/setuptools/build_meta.py", line 317, in run_setup
23.10.2026 17:24 :           exec(code, locals())  # noqa: S102 # exec is intentional here
23.10.2026 17:24 :         File "<string>", line 21, in <module>
23.10.2026 17:24 :       ModuleNotFoundError: No module named 'Cython'
23.10.2026 17:24 :       [end of output]
23.10.2026 17:24 :   
23.10.2026 17:24 :   note: This error originates from a subprocess, and is likely not a problem with pip.
23.10.2026 17:24 : error: metadata-generation-failed
23.10.2026 17:24 : × Encountered error while generating package metadata.
23.10.2026 17:24 : ╰─> pyreadstat
23.10.2026 17:24 : note: This is an issue with the package mentioned above, not pip.
23.10.2026 17:24 : hint: See above for details.
23.10.2026 17:24 : + exitcode=1
23.10.2026 17:24 : + '[' 1 -eq 0 ']'
23.10.2026 17:24 : + sleep 2
23.10.2026 17:24 : + exit 1
23.10.2026 17:24 : + exit_logging
23.10.2026 17:24 : + local -r code_at_exit=1
23.10.2026 17:24 : + '[' -n '$CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true' ']'
23.10.2026 17:24 : + local max_retries=10
23.10.2026 17:24 : + local retry_delay=3
23.10.2026 17:24 : + echo 'Evaluating cleanup command on EXIT with exit code 1: $CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true'
23.10.2026 17:24 : Evaluating cleanup command on EXIT with exit code 1: $CURL -sS -H "X-Api-Token: $DOMINO_EXECUTOR_TERMINATION_TOKEN" -X POST http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit || true
23.10.2026 17:24 : + local n=1
23.10.2026 17:24 : + true
23.10.2026 17:24 : + eval '$CURL' -sS -H '"X-Api-Token:' '$DOMINO_EXECUTOR_TERMINATION_TOKEN"' -X POST 'http://127.0.0.1:9000/executor/exit?exitCode=$code_at_exit' '||' true
23.10.2026 17:24 : ++ /opt/domino/bin/curl -sS -H 'X-Api-Token: 2d8e43a6554cd7108523dc05942cbeeadef3e576b8e2f8a6d5e26b0f04d21b11' -X POST 'http://127.0.0.1:9000/executor/exit?exitCode=1'
23.10.2026 17:24 : + break
23.10.2026 17:24 : + [[ 1 =~ ^(0|137|143)$ ]]
23.10.2026 17:24 : ++ tee -a /mnt/results/stdout.txt
23.10.2026 17:24 : + sleep 0.5
23.10.2026 17:24 : + echo 'Failed with exit code: 1'
23.10.2026 17:24 : Failed with exit code: 1
23.10.2026 17:24 : + exit 1
23.10.2026 17:24 : Caught termination signal!
23.10.2026 17:24 : -- killed by pod termination --
