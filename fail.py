(generic) ubuntu@192.168.11.218 ~/rubin $ pixi install
 WARN Encountered 1 warning while parsing the manifest:
  ⚠ The feature 'flaml' is defined but not used in any environment. Dependencies
  │ of unused features are not resolved or checked, and use wildcard (*) version
  │ specifiers by default, disregarding any set `pinning-strategy`
    ╭─[/mnt/rubin/pixi.toml:52:10]
 51 │ # ── Feature: flaml (optional, alternatives AutoML-Backend) ──
 52 │ [feature.flaml.dependencies]
    ·          ─────
 53 │ flaml = ">=2.1"
    ╰────
  help: Remove the feature from the manifest or add it to an environment

 WARN Skipped running the post-link scripts because `run-post-link-scripts` = `false`
        - bin/.librsvg-pre-unlink.sh

To enable them, run:
        pixi config set --local run-post-link-scripts insecure

More info:
        https://pixi.sh/latest/reference/pixi_configuration/#run-post-link-scripts

✔ The default environment has been installed.
