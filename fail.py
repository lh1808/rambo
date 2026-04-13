 WARN Encountered 1 warning while parsing the manifest:
  ⚠ The `project` field is deprecated. Use `workspace` instead.
    ╭─[/mnt/rubin/pixi.toml:1:1]
  1 │ ╭─▶ [project]
  2 │ │   name = "rubin"
  3 │ │   description = "Causal ML Framework – Analyse- und Production-Pipelines"
  4 │ │   channels = [
  5 │ │     "https://nexus3.lan.huk-coburg.de/repository/conda-qc-huk",
  6 │ │     "https://nexus3.lan.huk-coburg.de/repository/conda-forge",
  7 │ │   ]
  8 │ │   platforms = ["linux-64"]
  9 │ ├─▶ conda-pypi-map = {"https://nexus3.lan.huk-coburg.de/repository/conda-forge" = "https://nexus.lan.huk-coburg.de/repository/raw-githubusercontent/prefix-dev/parselmouth/refs/heads/main/files/compressed_mapping.json"}
    · ╰──── replace this with 'workspace'
 10 │     
    ╰────

  ⠈ default:linux-64     [00:00:02] resolving pyyaml==6.0.3                              Error:   × failed to solve the pypi requirements of environment 'default' for platform 'linux-
  │ 64'
  ├─▶ failed to resolve pypi dependencies
  ├─▶ Failed to fetch: `https://nexus.lan.huk-coburg.de/repository/pypi/simple/flaml/`
  ├─▶ Request failed after 3 retries
  ├─▶ error sending request for url (https://nexus.lan.huk-coburg.de/repository/pypi/
  │   simple/flaml/)
  ├─▶ client error (Connect)
  ╰─▶ invalid peer certificate: UnknownIssuer
