(generic) ubuntu@192.168.130.54 /home/ubuntu/da-hf1-rubin (master) $ pixi run analyze-quick
 WARN Using local manifest /home/ubuntu/da-hf1-rubin/pixi.toml rather than /opt/_global/code-server/pixi.toml from environment variable `PIXI_PROJECT_MANIFEST`
✨ Pixi task (analyze-quick in default): python run_analysis.py --config configs/config_quickstart.yml: (Analyse mit Quickstart-Config (Smoke-Test; Daten via 'pixi run demo-data' oder eigene Parquets unter data/processed/))
11:57:02 INFO [rubin.analysis] Arbeitsverzeichnis: /home/ubuntu/da-hf1-rubin/runs
11:57:02 WARNING [root] Malformed experiment '312895441071439399'. Detailed error Yaml file '/domino/edv/pvc-hf1kundehuk/mlflow/312895441071439399/meta.yaml' does not exist.
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
11:57:02 WARNING [root] Malformed experiment '312895441071439399'. Detailed error Yaml file '/domino/edv/pvc-hf1kundehuk/mlflow/312895441071439399/meta.yaml' does not exist.
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
11:57:02 WARNING [rubin.analysis] MLflow-Experiment 'rubin_quickstart' lag im Papierkorb (in der UI gelöscht) und hätte jeden Lauf blockiert — automatisch wiederhergestellt (ID 696568539089932779, alte Runs bleiben erhalten). Endgültig entfernen: in der MLflow-UI 'Permanently delete'.
11:57:02 WARNING [root] Malformed experiment '312895441071439399'. Detailed error Yaml file '/domino/edv/pvc-hf1kundehuk/mlflow/312895441071439399/meta.yaml' does not exist.
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/rubin/utils/mlflow_compat.py", line 39, in set_experiment_robust
    return mlflow_module.set_experiment(experiment_name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/mlflow/tracking/fluent.py", line 236, in set_experiment
    raise MlflowException(
mlflow.exceptions.MlflowException: Cannot set a deleted experiment 'rubin_quickstart' as the active experiment. You can restore the experiment, or permanently delete the experiment to create a new one.

During handling of the above exception, another exception occurred:

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
11:57:02 WARNING [root] Malformed experiment '312895441071439399'. Detailed error Yaml file '/domino/edv/pvc-hf1kundehuk/mlflow/312895441071439399/meta.yaml' does not exist.
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/rubin/utils/mlflow_compat.py", line 39, in set_experiment_robust
    return mlflow_module.set_experiment(experiment_name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/mlflow/tracking/fluent.py", line 236, in set_experiment
    raise MlflowException(
mlflow.exceptions.MlflowException: Cannot set a deleted experiment 'rubin_quickstart' as the active experiment. You can restore the experiment, or permanently delete the experiment to create a new one.

During handling of the above exception, another exception occurred:

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
11:57:03 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
11:57:03 INFO [rubin.analysis] rubin Pipeline Start
11:57:03 INFO [rubin.analysis] ════════════════════════════════════════════════════════════
11:57:03 INFO [rubin.analysis] Config: 2 Modelle (SLearner, TLearner), LGBM, 5-Fold CV, Parallel-Level 3
11:57:03 INFO [rubin.analysis] Nuisance Cross-Fitting (DML + DR): 5 interne Folds
11:57:03 INFO [rubin.analysis] Aktiv: Ensemble
11:57:03 INFO [rubin.analysis] Validierungsmodus: Cross-Validation (5 Folds, Seed=42, Tuning-Seed=18)
[rubin] Step 1/5: Daten laden & Preprocessing
Traceback (most recent call last):
  File "/home/ubuntu/da-hf1-rubin/run_analysis.py", line 132, in <module>
    main()
  File "/home/ubuntu/da-hf1-rubin/run_analysis.py", line 128, in main
    pipe.run(export_bundle=args.export_bundle, bundle_dir=args.bundle_dir, bundle_id=args.bundle_id)
  File "/home/ubuntu/da-hf1-rubin/rubin/pipelines/analysis_pipeline.py", line 3725, in run
    X, T, Y, S, eval_mask = self._load_inputs()
                            ^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/pipelines/analysis_pipeline.py", line 161, in _load_inputs
    X = self._read_table(self.cfg.data_files.x_file)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/pipelines/analysis_pipeline.py", line 157, in _read_table
    return read_table(path, use_index=use_index)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/rubin/utils/io_utils.py", line 18, in read_table
    df = pd.read_parquet(p)
         ^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/pandas/io/parquet.py", line 669, in read_parquet
    return impl.read(
           ^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/pandas/io/parquet.py", line 258, in read
    path_or_handle, handles, filesystem = _get_path_or_handle(
                                          ^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/pandas/io/parquet.py", line 141, in _get_path_or_handle
    handles = get_handle(
              ^^^^^^^^^^^
  File "/home/ubuntu/da-hf1-rubin/.pixi/envs/default/lib/python3.12/site-packages/pandas/io/common.py", line 882, in get_handle
    handle = open(handle, ioargs.mode)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'data/processed/X.parquet'
