from __future__ import annotations

"""Production-Scoring über saspy: SAS-Library → rubin-Bundle → SAS-Library.

Für Umgebungen, in denen Scoring-Input und -Output als SAS-Datasets in
Libraries liegen (statt als Dateien): Der Input wird per ``sd2df`` aus einer
SAS-Library gezogen, gegen das Bundle gescort (gemeinsamer Kern
``score_dataframe`` aus run_scoring) und per ``df2sd`` + ``PROC APPEND`` in
die Ziel-Library zurückgeschrieben. Monitoring-JSON wie beim Datei-Einstieg.

Aufruf (Config-getrieben, CLI nur für Overrides):

    python production/run_scoring_saspy.py --config production/scoring_<usecase>.yml

Design-Entscheidungen:
- saspy ist optionale Abhängigkeit (``pip install -e ".[saspy]"`` bzw.
  pixi-Environment ``prod``); der Import ist lazy mit klarer Fehlermeldung.
- Die SAS-Verbindung kommt vollständig aus der saspy-Konfiguration
  (``sascfg_personal.py`` / ``cfgname``) — Credentials gehören nicht in die
  Scoring-Config.
- Pull erfolgt gechunkt über ``firstobs``/``obs``-Dataset-Optionen, damit
  große Tabellen nicht in einem Stück durch die saspy-Transportschicht
  müssen. Bei gesetztem ``where`` zählen firstobs/obs auf den gefilterten
  Zeilen — sequentielle Fenster partitionieren die Selektion daher korrekt.
- Rückschreiben gechunkt über WORK-Zwischentabelle + ``PROC APPEND``
  (``force``): robust für beliebige Output-Größen; ``write_mode: replace``
  löscht die Zieltabelle vor dem ersten Chunk, ``append`` hängt an den
  Bestand an. SAS-Datasets erlauben 32-Zeichen-Namen — die V5-Kürzung des
  XPT-Pfads ist hier nicht nötig.
- Jeder ``submit`` wird auf ``ERROR:`` im SAS-Log geprüft und schlägt hart
  fehl — stilles Weiterlaufen nach einem SAS-Fehler würde unvollständige
  Score-Tabellen hinterlassen.
"""

import argparse
import gc
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_scoring import (  # noqa: E402
    _deep_merge,
    _normalize_scores,
    _reject_unknown_keys,
    score_dataframe,
    write_monitoring,
)

_logger = logging.getLogger("rubin.scoring.saspy")


# ──────────────────────────────────────────────────────────────────────────────
# Konfiguration
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "saspy": {
        "cfgname": None,      # Eintrag in sascfg_personal.py (None → saspy-Default)
        "cfgfile": None,      # expliziter Pfad zu einer sascfg-Datei (optional)
        "setup_code": None,   # optionaler SAS-Code nach Session-Start, z.B.
                              # Libname-Zuweisungen: "libname SCORING '/pfad';"
    },
    "input": {
        "libref": None,
        "table": None,
        "where": None,            # SAS-WHERE-Ausdruck, z.B. "STORNO_FLAG = 0"
        "chunk_size": 500000,     # Zeilen pro sd2df-Pull
        "uppercase_columns": True,
        # Nur die vom Bundle benötigten Feature-Spalten + IDs ziehen
        # (keep=-Dataset-Option). Mehrere Scores mit unterschiedlichen
        # Feature-Teilmengen können so gegen dieselbe breite Tabelle laufen,
        # ohne sie jeweils komplett durch die Transportschicht zu bewegen.
        "pull_only_needed_columns": True,
    },
    "id_columns": [],
    "preprocessing": {"replace_inf_with_nan": True},
    "scoring": {
        "score_p_model": "champion",
        "score_b_model": None,
        "extra_models": [],
        "batch_size": 100000,
        "round_decimals": 6,
    },
    "output": {
        "libref": None,
        "table": None,
        "write_mode": "replace",  # replace | append
        "write_chunk_size": 500000,
        "meta_columns": {},
        "column_order": None,
        "timestamp_format": "%Y-%j",
    },
    "monitoring": {"dir": None, "keep_latest_copy": True},
}


def load_saspy_scoring_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if raw.get("runner") not in (None, "saspy"):
        raise ValueError(
            f"{path} deklariert runner: {raw.get('runner')!r} — diese Config gehört zu "
            "production/run_scoring.py (run_scoring.sh routet automatisch anhand des runner-Keys)."
        )
    _reject_unknown_keys(raw, _DEFAULTS,
                         extra_top_level={"name", "runner", "bundle", "scores"},
                         free_form={"output.meta_columns"})
    cfg = _deep_merge(_DEFAULTS, raw)
    if not cfg.get("name"):
        raise ValueError(f"Scoring-Config: Pflichtfeld 'name' fehlt in {path}.")
    for field in ("libref", "table"):
        if not (cfg.get("input") or {}).get(field):
            raise ValueError(f"Scoring-Config: 'input.{field}' fehlt in {path}.")
    if not cfg.get("id_columns"):
        raise ValueError(
            f"Scoring-Config: 'id_columns' fehlt oder ist leer in {path} — ohne "
            "ID-Spalten wären die Score-Zeilen im Zielsystem nicht zuordenbar."
        )
    _normalize_scores(cfg, raw, path,
                      required_output_field="table",
                      output_field_label="output.table",
                      target_key_fields=("libref", "table"),
                      defaults=_DEFAULTS)
    for i, entry in enumerate(cfg["scores"]):
        if not entry["output"].get("libref"):
            raise ValueError(f"Scoring-Config: scores[{i}]: 'output.libref' fehlt in {path}.")
        if entry["output"]["write_mode"] not in ("replace", "append"):
            raise ValueError(f"Scoring-Config: scores[{i}]: output.write_mode muss 'replace' oder 'append' sein ({path}).")
    if not (cfg.get("monitoring") or {}).get("dir"):
        raise ValueError(
            f"Scoring-Config: 'monitoring.dir' fehlt in {path} — beim saspy-Einstieg "
            "gibt es keinen XPT-Pfad als Ablage-Default."
        )
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# saspy-Anbindung
# ──────────────────────────────────────────────────────────────────────────────
def _import_saspy():
    try:
        import saspy  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "saspy ist nicht installiert — der SAS-Library-Einstieg benötigt es. "
            "Installation: pip install -e \".[saspy]\" bzw. pixi install -e prod. "
            "Die SAS-Verbindung wird über sascfg_personal.py konfiguriert "
            "(https://sassoftware.github.io/saspy/configuration.html)."
        ) from e
    return saspy


def _open_session(saspy, scfg: Dict[str, Any]):
    kwargs = {k: v for k, v in (("cfgname", scfg.get("cfgname")),
                                ("cfgfile", scfg.get("cfgfile"))) if v}
    _logger.info("SAS-Session wird geöffnet (%s) …",
                 ", ".join(f"{k}={v}" for k, v in kwargs.items()) or "saspy-Default-Config")
    return saspy.SASsession(**kwargs)


def _submit_checked(sas, code: str, what: str) -> str:
    """submit() mit hartem Fehler bei ERROR: im SAS-Log."""
    result = sas.submit(code)
    log = (result or {}).get("LOG", "") or ""
    if "ERROR:" in log:
        err_lines = [ln for ln in log.splitlines() if "ERROR" in ln][:5]
        raise RuntimeError(f"SAS-Fehler bei {what}: {' | '.join(err_lines)}")
    return log


def _pull_table(sas, icfg: Dict[str, Any], keep_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Gechunkter Pull via sd2df mit firstobs/obs-Dataset-Optionen.

    ``keep_columns`` beschränkt den Pull per keep=-Option auf die benötigten
    Spalten (SAS matcht Spaltennamen case-insensitiv)."""
    libref, table = icfg["libref"], icfg["table"]
    chunk = int(icfg["chunk_size"])
    where = icfg.get("where")
    frames: List[pd.DataFrame] = []
    first = 1
    while True:
        dsopts: Dict[str, Any] = {"firstobs": first, "obs": first + chunk - 1}
        if where:
            dsopts["where"] = where
        if keep_columns:
            dsopts["keep"] = " ".join(keep_columns)
        part = sas.sd2df(table=table, libref=libref, dsopts=dsopts)
        n = 0 if part is None else len(part)
        if n == 0:
            break
        frames.append(part)
        _logger.info("Pull %s.%s: Zeilen %d–%d (%d).", libref, table, first, first + n - 1, n)
        if n < chunk:
            break
        first += chunk
    if not frames:
        raise ValueError(
            f"SAS-Tabelle {libref}.{table} lieferte 0 Zeilen"
            + (f" (where: {where})" if where else "") + " — Abbruch (nichts zu scoren)."
        )
    df = pd.concat(frames, ignore_index=True)
    frames.clear()
    if icfg.get("uppercase_columns"):
        df.columns = [c.upper() for c in df.columns]
    return df


def _write_back(sas, out: pd.DataFrame, ocfg: Dict[str, Any]) -> None:
    """Gechunktes Rückschreiben: df2sd → WORK-Zwischentabelle → PROC APPEND."""
    libref, table = ocfg["libref"], ocfg["table"]
    mode = ocfg["write_mode"]
    chunk = int(ocfg.get("write_chunk_size") or len(out) or 1)

    if mode == "replace":
        # nowarn: Zieltabelle darf beim Erstlauf fehlen.
        _submit_checked(
            sas,
            f"proc datasets lib={libref} nolist nowarn; delete {table}; quit;",
            f"Löschen von {libref}.{table} (write_mode=replace)",
        )

    tmp = "_rubin_scores_chunk"
    for i in range(0, len(out), chunk):
        part = out.iloc[i:i + chunk]
        sd = sas.df2sd(part, table=tmp, libref="WORK")
        if sd is None:
            raise RuntimeError(f"df2sd fehlgeschlagen (Chunk ab Zeile {i}).")
        _submit_checked(
            sas,
            f"proc append base={libref}.{table} data=work.{tmp} force; run;\n"
            f"proc datasets lib=work nolist nowarn; delete {tmp}; quit;",
            f"PROC APPEND nach {libref}.{table} (Chunk ab Zeile {i})",
        )
    _logger.info("Rückgeschrieben: %s.%s (%d Zeilen, write_mode=%s).",
                 libref, table, len(out), mode)


# ──────────────────────────────────────────────────────────────────────────────
# Hauptablauf
# ──────────────────────────────────────────────────────────────────────────────
def run_scoring_saspy(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    saspy = _import_saspy()
    now = datetime.now(timezone.utc)
    day_stamp = now.strftime(cfg["output"].get("timestamp_format") or "%Y-%j")
    run_stamp = now.strftime("%Y-%j_%H%M%S")

    # Bundles ZUERST: definieren die Feature-Teilmengen für den keep=-Pull
    # (Union über alle Einträge). Modelle werden lazy geladen (erst beim Scoren);
    # gleiche Bundle-Pfade werden nur einmal geladen.
    from rubin.pipelines.production_pipeline import ProductionPipeline
    entries = cfg["scores"]
    is_multi = len(entries) > 1 or entries[0].get("name") is not None
    pipes: Dict[str, ProductionPipeline] = {}
    for entry in entries:
        b = str(entry["bundle"])
        if b not in pipes:
            pipes[b] = ProductionPipeline(b)

    keep_cols: Optional[List[str]] = None
    if cfg["input"].get("pull_only_needed_columns", True):
        union: List[str] = []
        complete = True
        for pipe_i in pipes.values():
            feats = list(getattr(pipe_i.preprocessor, "feature_columns", []) or [])
            if not feats:
                complete = False
                break
            union.extend(c for c in feats if c not in union)
        if complete and union:
            keep_cols = union + [c for c in (cfg.get("id_columns") or []) if c not in union]

    sas = _open_session(saspy, cfg["saspy"])
    reports: List[Dict[str, Any]] = []
    try:
        if cfg["saspy"].get("setup_code"):
            _submit_checked(sas, cfg["saspy"]["setup_code"], "saspy.setup_code")

        # 1) EIN Pull für alle Scores --------------------------------------------
        df = _pull_table(sas, cfg["input"], keep_columns=keep_cols)
        _logger.info("Input: %d Zeilen, %d Spalten%s (%s.%s) — %d Score(s) gegen diese eine Ladung",
                     len(df), df.shape[1],
                     " (keep-gepruned, Union über alle Bundles)" if keep_cols else "",
                     cfg["input"]["libref"], cfg["input"]["table"], len(entries))
        input_rows, input_cols = int(len(df)), int(df.shape[1])

        for entry in entries:
            entry_t0 = time.time()
            e_name = f"{cfg['name']}_{entry['name']}" if is_multi else cfg["name"]
            if is_multi:
                _logger.info("--- Score '%s' (Bundle: %s) ---", entry["name"], entry["bundle"])
            e_cfg = dict(cfg)
            e_cfg.update(bundle=entry["bundle"], scoring=entry["scoring"], output=entry["output"])
            day_stamp = now.strftime(entry["output"].get("timestamp_format") or "%Y-%j")
            pipe = pipes[str(entry["bundle"])]

            # 2–4) Gemeinsamer Kern: Preprocessing, Scoren, Output-Frame ---------
            out, pipe, core = score_dataframe(df, e_cfg, day_stamp, pipe=pipe)

            # 5) Rückschreiben -----------------------------------------------------
            _write_back(sas, out, entry["output"])

            reports.append({
                "run": {
                    "name": e_name, "timestamp_utc": now.isoformat(),
                    "day_stamp": day_stamp,
                    "duration_seconds": round(time.time() - entry_t0, 2),
                    "total_duration_seconds": round(time.time() - t0, 2),
                },
                "input": {
                    "libref": cfg["input"]["libref"], "table": cfg["input"]["table"],
                    "where": cfg["input"].get("where"),
                    "n_rows": input_rows, "n_columns": input_cols,
                    "column_pruning": keep_cols is not None,
                },
                "bundle": {
                    "path": str(entry["bundle"]),
                    "champion": pipe.champion_model_name,
                    "created_at_utc": pipe.metadata.get("created_at_utc"),
                    "ml_package_versions": pipe.metadata.get("ml_package_versions"),
                    "version_mismatches": {k: {"bundle": v[0], "runtime": v[1]}
                                           for k, v in pipe.version_mismatches.items()},
                },
                "preprocessing": core["preprocessing"],
                "models": core["models"],
                "scores": core["scores"],
                "output": {
                    "libref": entry["output"]["libref"], "table": entry["output"]["table"],
                    "write_mode": entry["output"]["write_mode"],
                    "n_rows": int(len(out)), "columns": out.columns.tolist(),
                    "round_decimals": core["round_decimals"],
                },
            })
            mon_cfg = {"name": e_name, "monitoring": cfg["monitoring"], "output": entry["output"]}
            mon_file = write_monitoring(mon_cfg, reports[-1], run_stamp)
            _logger.info("Monitoring: %s", mon_file)
            del out
            gc.collect()

        # Eingabetabelle erst nach ALLEN Scores freigeben — genau das ist der
        # Zweck dieses Modus (ein teurer Pull, mehrere Verarbeitungen).
        del df
        gc.collect()
    finally:
        try:
            sas.endsas()
        except Exception:
            _logger.warning("SAS-Session konnte nicht sauber beendet werden.", exc_info=True)

    return reports[0] if not is_multi else reports


def main() -> None:
    parser = argparse.ArgumentParser(description="rubin Production-Scoring über saspy (SAS-Library → Bundle → SAS-Library)")
    parser.add_argument("--config", required=True, help="Pfad zur Scoring-Config (YAML)")
    parser.add_argument("--bundle", help="Override: Bundle-Verzeichnis")
    parser.add_argument("--table-in", help="Override: input.table")
    parser.add_argument("--table-out", help="Override: output.table")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    cfg = load_saspy_scoring_config(args.config)
    if args.bundle:
        cfg["bundle"] = args.bundle
    if args.table_in:
        cfg["input"]["table"] = args.table_in
    if args.table_out:
        cfg["output"]["table"] = args.table_out
    run_scoring_saspy(cfg)


if __name__ == "__main__":
    main()
