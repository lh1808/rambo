Managing pypi packages with pixi
Pixi is primarily a Conda package manager. But can be used to solve and install pypi packages together with Conda packages as well. Note that Conda package versions are preferred over pypi packages. Only use pypi package versions if the Conda version is not suitable or non-existent.

Managing pypi package works via integrated uv. Pypi packages can be installed via --pypi flag, e.g.

pixi add --pypi black
To make this work in lan we have to set the following environment variable

export PIXI_TLS_ROOT_CERTS="all"
and add the following properties to the pixi.toml file

[workspace] # former project
conda-pypi-map = {"https://nexus3.lan.huk-coburg.de/repository/conda-forge" = "https://nexus.lan.huk-coburg.de/repository/raw-githubusercontent/prefix-dev/parselmouth/refs/heads/main/files/compressed_mapping.json"}

[pypi-options]
index-url = "https://nexus.lan.huk-coburg.de/repository/pypi/simple"
