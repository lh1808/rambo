Noch besser:

[project]
conda-pypi-map = {"https://nexus3.lan.huk-coburg.de/repository/conda-forge" = "https://nexus.lan.huk-coburg.de/repository/raw-githubusercontent/prefix-dev/parselmouth/refs/heads/main/files/compressed_mapping.json"}

[pypi-options]
index-url = "https://nexus.lan.huk-coburg.de/repository/pypi/simple"
und env variable

PIXI_TLS_ROOT_CERTS="all"
Dann

pixi add --pypi spaneval
