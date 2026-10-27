10:39:45 INFO [rubin.analysis] Arbeitsverzeichnis: /home/ubuntu/da-hf1-rubin/runs
10:39:45 WARNING [root] Malformed experiment '312895441071439399'. Detailed error Yaml file '/domino/edv/pvc-hf1kundehuk/mlflow/312895441071439399/meta.yaml' does not exist.
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
10:39:45 WARNING [root] Malformed experiment '312895441071439399'. Detailed error Yaml file '/domino/edv/pvc-hf1kundehuk/mlflow/312895441071439399/meta.yaml' does not exist.
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
