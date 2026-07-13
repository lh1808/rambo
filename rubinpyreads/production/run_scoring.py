from __future__ import annotations

"""Generisches Production-Scoring: rubin-Bundle → SAS (XPT) + Monitoring-JSON.

Das Bundle liefert Preprocessor + Modelle self-contained; dieses Skript
orchestriert ausschließlich Einlesen → Transform → Scoren → Rückschreiben →
Monitoring. Es hält keine eigenen Modell- oder Preprocessing-Artefakte.

Aufruf (Config-getrieben, CLI nur für Pfad-Overrides):

    python production/run_scoring.py --config production/scoring_ph.yml
    python production/run_scoring.py --config ... --input other.parquet --bundle <dir>

Design-Entscheidungen (abgestimmt):
- XPT-file_format_version kommt aus der Config (Default 5 — Vorgabe des
  empfangenden Systems), table_name ebenso.
- Score-Rundung konfigurierbar (Default 6 Nachkommastellen). KEINE Skalierung:
  Ein pro Batch gefitteter Scaler würde Scores zwischen Läufen unvergleichbar
  machen; die Surrogates sind ohnehin ähnlich skaliert wie die Learner.
- SCORE_B (Surrogate) ist optional: fehlt der SurrogateTree im Bundle oder ist
  score_b_model: null, entfällt die Spalte ohne Fehler.
- Monitoring-JSON pro Lauf: zeitgestempelt (versioniert) + latest-Kopie. Die
  XPT wird bewusst überschrieben (Historisierung übernimmt das Zielsystem).
"""

import argparse
import gc
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# Fallback für Aufruf aus einem reinen Checkout ohne editable-Install
# (unter pixi ist rubin installiert; dieser Eintrag ist dann wirkungslos):
import sys as _sys
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from rubin.pipelines.production_pipeline import ProductionPipeline
from rubin.training import _predict_effect

_logger = logging.getLogger("rubin.scoring")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "input": {
        "path": None,              # PFLICHT — Eingabedatei
        "format": None,            # parquet|csv|sas7bdat; None → aus Endung
        "sas_chunksize": 200_000,
        "sas_encoding": "ISO-8859-1",
        "csv_sep": ",",            # nur für format: csv
        "csv_encoding": "utf-8",   # nur für format: csv
        "uppercase_columns": False,
        # Nur Bundle-Features + IDs einlesen (parquet: columns=, csv: usecols=,
        # sas7bdat: Subset pro Chunk) — mehrere Scores mit unterschiedlichen
        # Feature-Teilmengen können so gegen dieselbe breite Tabelle laufen.
        "pull_only_needed_columns": True,
    },
    "id_columns": [],
    "preprocessing": {"replace_inf_with_nan": True},
    "scoring": {
        "score_p_model": "champion",     # "champion" (Alias) | konkreter Modellname
        "score_b_model": None,           # z. B. "SurrogateTree"; None → keine SCORE_B-Spalte.
                                         # Bewusst kein stiller Surrogate-Default: was scort,
                                         # steht explizit in der Use-Case-YAML.
        "extra_models": [],
        "batch_size": 100_000,
        "round_decimals": 6,
    },
    "output": {
        "xpt_path": None,          # PFLICHT — XPT-Ziel (wird pro Lauf überschrieben)
        "table_name": "SCORES",
        "file_format_version": 5,
        "csv_path": None,
        "column_order": None,
        "meta_columns": {},
        "timestamp_format": "%Y-%j",   # YYYY-DDD (Schnittstellen-Konvention)
    },
    "monitoring": {"dir": None, "keep_latest_copy": True},
}


def _reject_unknown_keys(raw: Dict, allowed: Dict, extra_top_level: set,
                         free_form: set, path: str = "") -> None:
    """Lehnt unbekannte Config-Schlüssel hart ab (mit Tippfehler-Vorschlag).

    Ohne diese Prüfung würde ein Schreibfehler (z. B. 'score_p_modell' oder
    'chunksize') still ignoriert und der Default griffe unbemerkt — in einer
    Operator-Config ist das ein stiller Fehlkonfigurations-Kanal. free_form
    benennt Sektionen mit frei wählbaren Schlüsseln (z. B. meta_columns)."""
    import difflib
    for key, val in (raw or {}).items():
        here = f"{path}.{key}" if path else key
        if path == "" and key in extra_top_level:
            continue
        if here in free_form:
            continue
        if key not in allowed:
            candidates = list(allowed) + (list(extra_top_level) if path == "" else [])
            hint = difflib.get_close_matches(key, candidates, n=1)
            raise ValueError(
                f"Unbekannter Config-Schlüssel '{here}'"
                + (f" — meinten Sie '{(f'{path}.' if path else '') + hint[0]}'?" if hint else "")
                + f" Erlaubt auf dieser Ebene: {sorted(candidates)}"
            )
        if isinstance(val, dict) and isinstance(allowed.get(key), dict):
            _reject_unknown_keys(val, allowed[key], extra_top_level, free_form, here)


def _normalize_scores(cfg: Dict[str, Any], raw: Dict[str, Any], path: str,
                      required_output_field: str, output_field_label: str,
                      target_key_fields: tuple = None,
                      defaults: Dict[str, Any] = None) -> None:
    """Normalisiert die Config auf eine Liste von Score-Einträgen (cfg['scores']).

    Zwei gleichwertige Schreibweisen:
    - EIN Score: bundle/scoring/output auf Top-Level (klassisch) → wird zu
      einer Ein-Element-Liste normalisiert.
    - MEHRERE Scores gegen DENSELBEN Input: Top-Level-Liste ``scores:`` mit je
      name/bundle(/scoring/output) — der Input wird dann nur EINMAL geladen
      (Spalten = Union aller Bundle-Features + IDs) und pro Bundle
      unterschiedlich preprocessed und gescort. Für teures Laden (große
      sas7bdat, saspy-Transport) der richtige Modus; der Preis ist, dass die
      Eingabetabelle über alle Scores hinweg im Speicher bleibt.

    Bei ``scores:`` sind Top-Level bundle/scoring/output verboten (mehrdeutig).
    Jeder Eintrag erbt scoring/output-Defaults; Pflicht je Eintrag: name,
    bundle, output.<Zielfeld>."""
    entries = raw.get("scores")
    if entries is None:
        for key in ("bundle",):
            if not cfg.get(key):
                raise ValueError(f"Scoring-Config: Pflichtfeld 'bundle' fehlt in {path}.")
        if not (cfg.get("output") or {}).get(required_output_field):
            raise ValueError(f"Scoring-Config: '{output_field_label}' fehlt in {path}.")
        cfg["scores"] = [{"name": None, "bundle": cfg["bundle"],
                          "scoring": cfg["scoring"], "output": cfg["output"]}]
        return

    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Scoring-Config: 'scores' muss eine nicht-leere Liste sein ({path}).")
    conflict = [k for k in ("bundle", "scoring", "output") if k in raw]
    if conflict:
        raise ValueError(
            f"Scoring-Config: bei 'scores:' dürfen {conflict} nicht zusätzlich auf "
            f"Top-Level stehen (mehrdeutig) — pro Eintrag definieren ({path})."
        )
    defaults = defaults if defaults is not None else _DEFAULTS
    allowed_entry = {"name": None, "bundle": None,
                     "scoring": defaults["scoring"], "output": defaults["output"]}
    seen = set()
    norm = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Scoring-Config: scores[{i}] ist kein Mapping ({path}).")
        _reject_unknown_keys(entry, allowed_entry, extra_top_level=set(),
                             free_form={"output.meta_columns"})
        if not entry.get("name"):
            raise ValueError(f"Scoring-Config: scores[{i}].name fehlt ({path}) — "
                             "wird für Log- und Monitoring-Namen gebraucht.")
        if entry["name"] in seen:
            raise ValueError(f"Scoring-Config: doppelter scores-Name '{entry['name']}' ({path}).")
        seen.add(entry["name"])
        if not entry.get("bundle"):
            raise ValueError(f"Scoring-Config: scores[{i}].bundle fehlt ({path}).")
        merged_scoring = _deep_merge(defaults["scoring"], entry.get("scoring") or {})
        merged_output = _deep_merge(defaults["output"], entry.get("output") or {})
        if not merged_output.get(required_output_field):
            raise ValueError(
                f"Scoring-Config: scores[{i}] ('{entry['name']}'): "
                f"'{output_field_label}' fehlt ({path})."
            )
        norm.append({"name": str(entry["name"]), "bundle": entry["bundle"],
                     "scoring": merged_scoring, "output": merged_output})
    key_fields = target_key_fields or (required_output_field,)
    outputs = [tuple(e["output"].get(f) for f in key_fields) for e in norm]
    dupes = {o for o in outputs if outputs.count(o) > 1}
    if dupes:
        raise ValueError(f"Scoring-Config: mehrere scores-Einträge schreiben auf dasselbe Ziel {sorted(dupes)} ({path}).")
    cfg["scores"] = norm


def _deep_merge(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _deep_merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_scoring_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if raw.get("runner") not in (None, "file"):
        raise ValueError(
            f"{path} deklariert runner: {raw.get('runner')!r} — diese Config gehört zu "
            "production/run_scoring_saspy.py (run_scoring.sh routet automatisch anhand des runner-Keys)."
        )
    _reject_unknown_keys(raw, _DEFAULTS,
                         extra_top_level={"name", "runner", "bundle", "scores"},
                         free_form={"output.meta_columns"})
    cfg = _deep_merge(_DEFAULTS, raw)
    if not cfg.get("name"):
        raise ValueError(f"Scoring-Config: Pflichtfeld 'name' fehlt in {path}.")
    if not (cfg.get("input") or {}).get("path"):
        raise ValueError(f"Scoring-Config: 'input.path' fehlt in {path}.")
    if not cfg.get("id_columns"):
        raise ValueError(
            f"Scoring-Config: 'id_columns' fehlt oder ist leer in {path} — ohne "
            "ID-Spalten wären die Score-Zeilen im Zielsystem nicht zuordenbar."
        )
    _normalize_scores(cfg, raw, path,
                      required_output_field="xpt_path",
                      output_field_label="output.xpt_path")
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Einlesen
# ──────────────────────────────────────────────────────────────────────────────
def read_input(icfg: Dict[str, Any], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Liest den Scoring-Input. ``columns`` (optional) beschränkt das Einlesen
    auf die benötigten Spalten (Feature-Teilmenge des Bundles + IDs) —
    parquet/csv lesen dann nur diese Spalten (I/O + Speicher), sas7bdat wird
    nach jedem Chunk subsetted (Speicher). Nicht vorhandene Spalten werden
    toleriert; sie erscheinen später im Monitoring als missing_expected."""
    path = str(icfg["path"])
    fmt = icfg.get("format") or Path(path).suffix.lstrip(".").lower()
    upper = bool(icfg.get("uppercase_columns"))
    # Spaltenvergleich in der Quell-Schreibweise (vor uppercase_columns):
    wanted = None
    if columns:
        wanted = {c.upper() for c in columns} if upper else set(columns)

    def _subset(frame: pd.DataFrame) -> pd.DataFrame:
        if wanted is None:
            return frame
        keep = [c for c in frame.columns
                if (c.upper() if upper else c) in wanted]
        return frame[keep]

    if fmt in ("parquet", "pq"):
        if wanted is not None:
            import pyarrow.parquet as pq
            # Datei-Reihenfolge beibehalten (Set-Iteration wäre nichtdeterministisch)
            avail_names = pq.ParquetFile(path).schema_arrow.names
            use = [c for c in avail_names if (c.upper() if upper else c) in wanted]
            df = pd.read_parquet(path, columns=use)
        else:
            df = pd.read_parquet(path)
    elif fmt == "csv":
        csv_kw = {"sep": icfg.get("csv_sep", ","), "encoding": icfg.get("csv_encoding", "utf-8")}
        if wanted is not None:
            df = pd.read_csv(path, usecols=lambda c: (c.upper() if upper else c) in wanted, **csv_kw)
        else:
            df = pd.read_csv(path, **csv_kw)
    elif fmt == "sas7bdat":
        chunks = []
        for chunk in pd.read_sas(path, chunksize=int(icfg["sas_chunksize"]),
                                 encoding=icfg["sas_encoding"], format="sas7bdat"):
            chunks.append(_subset(chunk))
        df = pd.concat(chunks, ignore_index=True)
        chunks.clear()
    else:
        raise ValueError(f"Unbekanntes Input-Format '{fmt}' für {path}.")
    if upper:
        df.columns = [c.upper() for c in df.columns]
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────
def predict_in_batches(model, X: pd.DataFrame, batch_size: int) -> np.ndarray:
    """Shape-sicheres Batch-Scoring über _predict_effect (BT: (n,), MT: (n, K-1))."""
    parts = [np.asarray(_predict_effect(model, X.iloc[i:i + batch_size]))
             for i in range(0, len(X), batch_size)]
    return np.concatenate(parts, axis=0)


def _score_columns(name_prefix: str, values: np.ndarray) -> Dict[str, np.ndarray]:
    """BT → {prefix: (n,)}; MT → {prefix1: …, prefix2: …} (V5-taugliche ≤8-Zeichen-Namen)."""
    if values.ndim == 1:
        return {name_prefix: values}
    return {f"{name_prefix}{k + 1}": values[:, k] for k in range(values.shape[1])}


def _resolve_model(pipe: ProductionPipeline, spec: Optional[str]) -> Optional[str]:
    if spec is None:
        return None
    if spec == "champion":
        if pipe.champion_model_name is None:
            raise ValueError("Bundle hat keinen Champion in der Registry — score_p_model explizit setzen.")
        return pipe.champion_model_name
    return spec


# ──────────────────────────────────────────────────────────────────────────────
# Monitoring
# ──────────────────────────────────────────────────────────────────────────────
def _minus1_rates(pipe: ProductionPipeline, Xp: pd.DataFrame) -> Dict[str, float]:
    """Anteil des -1-Sentinels (im Training unbekannte Ausprägung/Missing) pro
    kategorialer Spalte — das direkteste Kategorien-Drift-Signal im Scoring."""
    rates: Dict[str, float] = {}
    for col in getattr(pipe.preprocessor, "categorical_columns", []) or []:
        if col in Xp.columns:
            vals = pd.to_numeric(pd.Series(Xp[col].to_numpy()), errors="coerce")
            rates[col] = round(float((vals == -1).mean()), 6)
    return rates


def _score_stats(series: pd.Series) -> Dict[str, Any]:
    v = series.to_numpy(dtype=float)
    return {
        "n": int(v.size), "nan": int(np.isnan(v).sum()),
        "min": float(np.nanmin(v)), "p25": float(np.nanpercentile(v, 25)),
        "median": float(np.nanmedian(v)), "mean": float(np.nanmean(v)),
        "p75": float(np.nanpercentile(v, 75)), "max": float(np.nanmax(v)),
        "std": float(np.nanstd(v)),
    }


def write_monitoring(cfg: Dict[str, Any], report: Dict[str, Any], run_stamp: str) -> Path:
    mon_dir = cfg["monitoring"].get("dir") or str(Path(cfg["output"]["xpt_path"]).parent / "monitoring")
    mon_path = Path(mon_dir); mon_path.mkdir(parents=True, exist_ok=True)
    name = cfg["name"]
    versioned = mon_path / f"{name}_{run_stamp}.json"
    with open(versioned, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    if cfg["monitoring"].get("keep_latest_copy", True):
        with open(mon_path / f"{name}_latest.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    return versioned


# ──────────────────────────────────────────────────────────────────────────────
# Gemeinsamer Scoring-Kern
# ──────────────────────────────────────────────────────────────────────────────
def score_dataframe(df: pd.DataFrame, cfg: Dict[str, Any], day_stamp: str,
                    pipe: Optional[ProductionPipeline] = None):
    """Gemeinsamer Scoring-Kern: Bundle laden, preprocessen, scoren, Output-Frame bauen.

    Wird vom Datei-Einstieg (run_scoring, XPT) und vom saspy-Einstieg
    (run_scoring_saspy, SAS-Library) verwendet. Gibt (out, pipe, core) zurück;
    core enthält die Monitoring-Bausteine der Schritte 2–4. ``pipe`` kann
    vorab geladen übergeben werden (Spalten-Pruning braucht die
    feature_columns bereits vor dem Einlesen)."""
    id_cols: List[str] = list(cfg.get("id_columns") or [])
    missing_ids = [c for c in id_cols if c not in df.columns]
    if missing_ids:
        raise ValueError(f"ID-Spalten fehlen im Input: {missing_ids}")
    ids = df[id_cols].copy() if id_cols else None

    # 2) Bundle + Preprocessing ------------------------------------------------
    if pipe is None:
        pipe = ProductionPipeline(str(cfg["bundle"]))
    _logger.info(
        "Bundle-Modelle (per YAML frei wählbar, unabhängig vom Champion): %s | Champion: %s",
        sorted(pipe.models), pipe.champion_model_name,
    )
    feature_columns = list(getattr(pipe.preprocessor, "feature_columns", []) or [])
    id_feature_overlap = sorted(set(id_cols) & set(feature_columns))
    if id_feature_overlap:
        raise ValueError(
            f"ID-Spalten sind zugleich Modell-Features: {id_feature_overlap} — "
            "das ergäbe doppelte Spalten im Output. id_columns bereinigen."
        )
    missing_expected = [c for c in feature_columns if c not in df.columns]
    if feature_columns and len(missing_expected) == len(feature_columns):
        raise ValueError(
            "KEINE der vom Bundle erwarteten Feature-Spalten ist im Input vorhanden "
            f"(erwartet z. B.: {feature_columns[:5]}; Input hat: {list(df.columns)[:5]}). "
            "Häufigste Ursache: input.uppercase_columns passt nicht zur Schreibweise "
            "der Bundle-Features (DataPrep-Bundles erwarten Großschreibung → true). "
            "Ein Weiterlaufen würde reine NaN-Scores erzeugen — Abbruch."
        )
    if missing_expected:
        _logger.warning(
            "Erwartete Feature-Spalten fehlen im Input und werden als NaN ergänzt "
            "(gelernte Missing-Behandlung greift, prüfen ob beabsichtigt): %s",
            missing_expected,
        )
    extra_dropped = sorted(set(df.columns) - set(feature_columns) - set(id_cols))

    inf_replaced = 0
    if cfg["preprocessing"].get("replace_inf_with_nan", True):
        num = df.select_dtypes(include=[np.number])
        inf_mask = np.isinf(num.to_numpy(dtype=float, na_value=np.nan))
        inf_replaced = int(inf_mask.sum())
        if inf_replaced:
            df[num.columns] = num.replace([np.inf, -np.inf], np.nan)
            _logger.warning("Inf-Werte durch NaN ersetzt (→ gelernte Imputation greift): %d Zellen", inf_replaced)

    Xp = pipe.preprocessor.transform(df)
    minus1 = _minus1_rates(pipe, Xp)
    high_drift = {k: v for k, v in minus1.items() if v > 0.01}
    if high_drift:
        _logger.warning("Erhöhte -1-Raten (unbekannte Kategorien/Missings) — mögliches Drift-Signal: %s", high_drift)

    # 3) Scoren ---------------------------------------------------------------
    batch = int(cfg["scoring"]["batch_size"])
    round_dec = cfg["scoring"].get("round_decimals")
    scores = pd.DataFrame(index=Xp.index)

    p_name = _resolve_model(pipe, cfg["scoring"].get("score_p_model") or "champion")
    if p_name not in pipe.models:
        raise ValueError(f"score_p_model '{p_name}' nicht im Bundle. Vorhanden: {sorted(pipe.models)}")
    for col, vals in _score_columns("SCORE_P", predict_in_batches(pipe.models[p_name], Xp, batch)).items():
        scores[col] = vals
    _logger.info("SCORE_P: %s", p_name)

    b_spec = cfg["scoring"].get("score_b_model")
    b_name = _resolve_model(pipe, b_spec) if b_spec else None
    if b_name is not None:
        if b_name in pipe.models:
            for col, vals in _score_columns("SCORE_B", predict_in_batches(pipe.models[b_name], Xp, batch)).items():
                scores[col] = vals
            _logger.info("SCORE_B: %s", b_name)
        else:
            _logger.warning("score_b_model '%s' nicht im Bundle — SCORE_B entfällt (optional).", b_name)
            b_name = None

    for extra in cfg["scoring"].get("extra_models") or []:
        if extra not in pipe.models:
            raise ValueError(f"extra_models: '{extra}' nicht im Bundle.")
        for col, vals in _score_columns(f"CATE_{extra}", predict_in_batches(pipe.models[extra], Xp, batch)).items():
            scores[col] = vals

    if round_dec is not None:
        scores = scores.round(int(round_dec))

    _nan_scores = {c: int(scores[c].isna().sum()) for c in scores.columns if scores[c].isna().any()}
    if _nan_scores:
        _logger.warning(
            "NaN in Score-Spalten (Input-Missings, die der Bundle-Preprocessor nicht "
            "imputiert — z. B. Schema-Fallback ohne DataPrep): %s", _nan_scores,
        )

    # Xp freigeben, bevor der Output-Frame gebaut wird: die transformierte
    # Feature-Matrix ist der größte Zwischenstand und wird nicht mehr gebraucht.
    del Xp
    gc.collect()

    # 4) Output-Frame ----------------------------------------------------------
    out = pd.concat(([ids] if ids is not None else []) + [scores], axis=1)
    for k, v in (cfg["output"].get("meta_columns") or {}).items():
        out[k] = v
    out["TIMESTAMP"] = day_stamp

    order = cfg["output"].get("column_order")
    if order:
        expanded: List[str] = []
        for c in order:
            if c in out.columns:
                expanded.append(c)
            else:  # MT-Expansion (SCORE_P → SCORE_P1, SCORE_P2, …) bzw. optionale SCORE_B
                expanded.extend(sorted(x for x in out.columns if x.startswith(c) and x not in expanded))
        rest = [c for c in out.columns if c not in expanded]
        out = out[expanded + rest]


    core: Dict[str, Any] = {
        "preprocessing": {
            "expected_features": len(feature_columns),
            "missing_expected_columns": missing_expected,       # als NaN ergänzt
            "extra_input_columns_dropped": len(extra_dropped),
            "inf_cells_replaced_with_nan": inf_replaced,
            "minus1_rate_per_categorical": minus1,               # Drift-Signal
        },
        "models": {"score_p": p_name, "score_b": b_name,
                   "extra": list(cfg["scoring"].get("extra_models") or [])},
        "scores": {c: _score_stats(scores[c]) for c in scores.columns},
        "round_decimals": round_dec,
    }
    return out, pipe, core


# ──────────────────────────────────────────────────────────────────────────────
# Hauptablauf
# ──────────────────────────────────────────────────────────────────────────────
def _write_outputs(out: pd.DataFrame, ocfg: Dict[str, Any]):
    """Schritt 5: XPT (+ optional CSV) schreiben. Gibt (xpt_path, fmt, v5_map) zurück."""
    import pyreadstat
    xpt_path = str(ocfg["xpt_path"])
    fmt_version = int(ocfg["file_format_version"])

    # XPT V5 kappt Variablennamen auf 8 Zeichen (SAS-Transport-Standard);
    # das empfangende System arbeitet mit den gekürzten Namen. Explizit statt
    # still: Mapping loggen + ins Monitoring, Kollisionen hart ablehnen.
    v5_truncations: Dict[str, str] = {}
    if fmt_version == 5:
        v5_truncations = {c: c[:8] for c in out.columns if len(c) > 8}
        truncated_all = [c[:8] for c in out.columns]
        dupes = {n for n in truncated_all if truncated_all.count(n) > 1}
        if dupes:
            raise ValueError(
                f"XPT-V5-Namenskollision nach 8-Zeichen-Kürzung: {sorted(dupes)} — "
                "Spalten umbenennen oder file_format_version: 8 verwenden."
            )
        if v5_truncations:
            _logger.warning("XPT V5 kürzt Variablennamen auf 8 Zeichen: %s", v5_truncations)

    Path(xpt_path).parent.mkdir(parents=True, exist_ok=True)
    pyreadstat.write_xport(
        out, xpt_path,
        table_name=ocfg["table_name"],
        file_format_version=fmt_version,
        column_labels=out.columns.tolist(),  # Labels tragen die vollen Namen
    )
    _logger.info("XPT geschrieben: %s (%d Zeilen, V%s, Tabelle %s)",
                 xpt_path, len(out), fmt_version, ocfg["table_name"])
    if ocfg.get("csv_path"):
        out.to_csv(ocfg["csv_path"], index=False)
        _logger.info("CSV geschrieben: %s", ocfg["csv_path"])
    return xpt_path, fmt_version, v5_truncations


def run_scoring(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    # 0) Fail-Fast: XPT-Export braucht pyreadstat. Ohne diesen Check würde der
    # Lauf erst in Schritt 5 — NACH dem kompletten Scoring — mit
    # ModuleNotFoundError abbrechen. Import bleibt lazy (Modul soll auch ohne
    # pyreadstat importierbar sein, z.B. für Tests einzelner Funktionen).
    try:
        import pyreadstat  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pyreadstat ist nicht installiert — der SAS-XPT-Export in Schritt 5 "
            "würde fehlschlagen. Installation: pip install 'pyreadstat>=1.2' "
            "bzw. pixi install (in pixi.toml enthalten)."
        ) from e

    now = datetime.now(timezone.utc)
    day_stamp = now.strftime(cfg["output"].get("timestamp_format") or "%Y-%j")
    run_stamp = now.strftime("%Y-%j_%H%M%S")

    # 1) Bundles zuerst, dann EINMAL Einlesen ----------------------------------
    # Jeder scores-Eintrag hat sein eigenes Bundle mit eigener Feature-
    # Teilmenge. Die Quelle wird nur EINMAL gelesen — bei pull_only_needed_
    # columns (Default) beschränkt auf die UNION aller Bundle-Features + IDs.
    # Gleiche Bundle-Pfade werden nur einmal geladen (Modelle ohnehin lazy).
    entries = cfg["scores"]
    is_multi = len(entries) > 1 or entries[0].get("name") is not None
    pipes: Dict[str, ProductionPipeline] = {}
    for entry in entries:
        b = str(entry["bundle"])
        if b not in pipes:
            pipes[b] = ProductionPipeline(b)

    needed_cols: Optional[List[str]] = None
    if cfg["input"].get("pull_only_needed_columns", True):
        union: List[str] = []
        complete = True
        for pipe_i in pipes.values():
            feats = list(getattr(pipe_i.preprocessor, "feature_columns", []) or [])
            if not feats:
                complete = False  # Bundle ohne Feature-Liste → volle Breite lesen
                break
            union.extend(c for c in feats if c not in union)
        if complete and union:
            needed_cols = union + [c for c in (cfg.get("id_columns") or []) if c not in union]

    df = read_input(cfg["input"], columns=needed_cols)
    _logger.info("Input: %d Zeilen, %d Spalten%s (%s) — %d Score(s) gegen diese eine Ladung",
                 len(df), df.shape[1],
                 " (spalten-gepruned, Union über alle Bundles)" if needed_cols is not None else "",
                 cfg["input"]["path"], len(entries))
    if len(df) == 0:
        raise ValueError(f"Input '{cfg['input']['path']}' enthält 0 Zeilen — Abbruch (nichts zu scoren).")
    input_rows, input_cols = int(len(df)), int(df.shape[1])

    reports: List[Dict[str, Any]] = []
    for entry in entries:
        entry_t0 = time.time()
        e_name = f"{cfg['name']}_{entry['name']}" if is_multi else cfg["name"]
        if is_multi:
            _logger.info("--- Score '%s' (Bundle: %s) ---", entry["name"], entry["bundle"])
        e_cfg = dict(cfg)
        e_cfg.update(bundle=entry["bundle"], scoring=entry["scoring"], output=entry["output"])
        day_stamp = now.strftime(entry["output"].get("timestamp_format") or "%Y-%j")
        pipe = pipes[str(entry["bundle"])]

        # 2–4) Gemeinsamer Kern: Preprocessing, Scoren, Output-Frame -----------
        out, pipe, core = score_dataframe(df, e_cfg, day_stamp, pipe=pipe)

        xpt_path, fmt_version, v5_truncations = _write_outputs(out, entry["output"])

        reports.append({
            "run": {
                "name": e_name, "timestamp_utc": now.isoformat(),
                "day_stamp": day_stamp,
                "duration_seconds": round(time.time() - entry_t0, 2),
                "total_duration_seconds": round(time.time() - t0, 2),
            },
            "input": {
                "path": cfg["input"]["path"], "n_rows": input_rows, "n_columns": input_cols,
                "column_pruning": needed_cols is not None,
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
                "xpt_path": xpt_path, "table_name": entry["output"]["table_name"],
                "file_format_version": fmt_version,
                "n_rows": int(len(out)), "columns": out.columns.tolist(),
                "v5_name_truncations": v5_truncations,
                "round_decimals": core["round_decimals"],
            },
        })
        mon_cfg = {"name": e_name, "monitoring": cfg["monitoring"], "output": entry["output"]}
        mon_file = write_monitoring(mon_cfg, reports[-1], run_stamp)
        _logger.info("Monitoring: %s", mon_file)
        del out
        gc.collect()

    del df
    gc.collect()
    return reports[0] if not is_multi else reports


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    ap = argparse.ArgumentParser(description="rubin Production-Scoring → SAS XPT + Monitoring-JSON")
    ap.add_argument("--config", required=True, help="Pfad zur Scoring-YAML (z. B. production/scoring_ph.yml)")
    ap.add_argument("--input", default=None, help="Override: input.path")
    ap.add_argument("--bundle", default=None, help="Override: bundle")
    ap.add_argument("--output", default=None, help="Override: output.xpt_path")
    args = ap.parse_args()

    cfg = load_scoring_config(args.config)
    if args.input:
        cfg["input"]["path"] = args.input
    if args.bundle or args.output:
        if len(cfg["scores"]) > 1:
            raise SystemExit("--bundle/--output sind bei einer scores-Liste mehrdeutig — "
                             "Ziele pro Eintrag in der YAML definieren.")
        if args.bundle:
            cfg["scores"][0]["bundle"] = args.bundle
        if args.output:
            cfg["scores"][0]["output"]["xpt_path"] = args.output
    run_scoring(cfg)


if __name__ == "__main__":
    main()
