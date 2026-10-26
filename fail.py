(generic) ubuntu@192.168.130.54 /home/ubuntu/da-hf1-rubin (master) $ pixi run analyze-quick
 WARN Using local manifest /home/ubuntu/da-hf1-rubin/pixi.toml rather than /opt/_global/code-server/pixi.toml from environment variable `PIXI_PROJECT_MANIFEST`
✨ Pixi task (analyze-quick in default): python run_analysis.py --config configs/config_quickstart.yml: (Analyse mit Quickstart-Config (Smoke-Test; Daten via 'pixi run demo-data' oder eigene Parquets unter data/processed/))
11:46:13 INFO [rubin.analysis] Arbeitsverzeichnis: /home/ubuntu/da-hf1-rubin/runs
11:46:13 WARNING [root] Malformed experiment '312895441071439399'. Detailed error Yaml file '/domino/edv/pvc-hf1kundehuk/mlflow/312895441071439399/meta.yaml' does not exist.
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 386, in search_experiments
    exp = self._get_experiment(exp_id, view_type)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 487, in _get_experiment
    meta = FileStore._read_yaml(experiment_dir, FileStore.META_DATA_FILE_NAME)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 1676, in _read_yaml
    return _read_helper(root, file_name, attempts_remaining=retries)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 1669, in _read_helper
    result = read_yaml(root, file_name)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/mlflow/utils/yaml_utils.py", line 104, in read_yaml
    raise MissingConfigException(f"Yaml file '{file_path}' does not exist.")
mlflow.exceptions.MissingConfigException: Yaml file '/domino/edv/pvc-hf1kundehuk/mlflow/312895441071439399/meta.yaml' does not exist.
11:46:13 WARNING [root] Malformed experiment '312895441071439399'. Detailed error Yaml file '/domino/edv/pvc-hf1kundehuk/mlflow/312895441071439399/meta.yaml' does not exist.
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 386, in search_experiments
    exp = self._get_experiment(exp_id, view_type)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 487, in _get_experiment
    meta = FileStore._read_yaml(experiment_dir, FileStore.META_DATA_FILE_NAME)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 1676, in _read_yaml
    return _read_helper(root, file_name, attempts_remaining=retries)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 1669, in _read_helper
    result = read_yaml(root, file_name)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/mlflow/utils/yaml_utils.py", line 104, in read_yaml
    raise MissingConfigException(f"Yaml file '{file_path}' does not exist.")
mlflow.exceptions.MissingConfigException: Yaml file '/domino/edv/pvc-hf1kundehuk/mlflow/312895441071439399/meta.yaml' does not exist.
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/run_analysis.py", line 132, in <module>
    main()
  File "/home/ubuntu/da-hf1-rubin/run_analysis.py", line 128, in main
    pipe.run(export_bundle=args.export_bundle, bundle_dir=args.bundle_dir, bundle_id=args.bundle_id)
  File "/home/ubuntu/da-hf1-rubin/rubin/pipelines/analysis_pipeline.py", line 3561, in run
    mlflow.set_experiment(cfg.mlflow.experiment_name)
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/mlflow/tracking/fluent.py", line 236, in set_experiment
    raise MlflowException(
mlflow.exceptions.MlflowException: Cannot set a deleted experiment 'rubin_quickstart' as the active experiment. You can restore the experiment, or permanently delete the experiment to create a new one.
