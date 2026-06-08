"""Daten-Routen: Upload, Spalten-Erkennung, Feature Dictionary."""
from __future__ import annotations

import math
import os
from pathlib import Path

from flask import Blueprint, jsonify, request, abort
from werkzeug.utils import secure_filename

from app.state import log, ROOT, WORK_DIR, UPLOAD_DIR, safe_str

bp = Blueprint("data", __name__)

@bp.route("/api/upload", methods=["POST"])
def upload_file():
    """Multipart file upload. Speichert in data/uploads/ oder target_dir."""
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "Keine Datei im Request."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"status": "error", "message": "Kein Dateiname."}), 400

    # Filename sanitieren (entfernt ../, absolute Pfade etc.)
    safe_name = secure_filename(f.filename)
    if not safe_name:
        return jsonify({"status": "error", "message": "Ungueltiger Dateiname."}), 400

    target_dir = request.form.get("target_dir", "")
    if target_dir:
        target = (ROOT / target_dir).resolve()
        # Path-Traversal-Schutz: Nur ROOT und WORK_DIR erlaubt.
        # is_relative_to (statt str.startswith) verhindert, dass ein Geschwister-
        # Verzeichnis mit gleichem Namens-Präfix (z. B. <root>_backup) passiert.
        _root_r, _work_r = ROOT.resolve(), WORK_DIR.resolve()
        if not (target.is_relative_to(_root_r) or target.is_relative_to(_work_r)):
            log.warning("Path-Traversal-Versuch blockiert (upload target_dir): %s", target_dir)
            abort(403)
    else:
        target = UPLOAD_DIR

    target.mkdir(parents=True, exist_ok=True)

    filepath = target / safe_name
    f.save(str(filepath))
    log.info("Datei hochgeladen: %s (%d bytes)", filepath, filepath.stat().st_size)

    # Anzeige-Pfad relativ zur passenden Basis (target kann unter ROOT ODER WORK_DIR
    # liegen). relative_to(ROOT) allein würde bei WORK_DIR-Zielen ValueError werfen.
    try:
        rel = filepath.relative_to(ROOT)
    except ValueError:
        try:
            rel = filepath.relative_to(WORK_DIR)
        except ValueError:
            rel = filepath

    return jsonify({
        "status": "done",
        "message": f"Gespeichert: {rel}",
        "path": str(rel),
        "filename": safe_name,
    })


# ══════════════════════════════════════════════════════
# COLUMN DETECTION
# ══════════════════════════════════════════════════════

@bp.route("/api/detect-columns", methods=["POST"])
def detect_columns():
    """Liest eine Datei und gibt Spalten, Typen, NaN-Spalten zurueck."""
    data = request.get_json(silent=True) or {}
    filepath = data.get("path", "")

    if not filepath:
        return jsonify({"status": "error", "message": "Kein Pfad angegeben."}), 400

    # Absolute Pfade (z.B. PVC: /mnt/data/...) direkt verwenden,
    # relative Pfade werden relativ zu ROOT aufgelöst + Traversal-Check
    if os.path.isabs(filepath):
        full = Path(filepath).resolve()
    else:
        full = (ROOT / filepath).resolve()
        if not str(full).startswith(str(ROOT.resolve())):
            log.warning("Path-Traversal-Versuch blockiert (detect-columns): %s", filepath)
            return jsonify({"status": "error", "message": "Ungueltiger Pfad."}), 403
    if not full.exists():
        return jsonify({"status": "error", "message": f"Nicht gefunden: {filepath}"}), 404

    try:
        import pandas as pd  # lazy import – nur bei Spalten-Erkennung nötig
        import math

        ext = full.suffix.lower()
        if ext == ".parquet":
            # Nur erste Zeilen lesen um OOM bei großen Dateien zu vermeiden
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(full)
                batch = next(pf.iter_batches(batch_size=5000))
                df = batch.to_pandas()
            except Exception:
                # Fallback: ganzes File lesen (kleine Dateien)
                df = pd.read_parquet(full, engine="pyarrow")
        elif ext == ".csv":
            delim = data.get("delimiter", ",")
            df = pd.read_csv(full, nrows=5000, sep=delim)
        elif ext in (".sas7bdat",):
            try:
                reader = pd.read_sas(full, encoding=data.get("encoding", "utf-8"),
                                     chunksize=5000)
                df = next(reader)
                reader.close()
            except (TypeError, StopIteration):
                # Fallback: manche SAS-Dateien unterstützen kein chunksize
                df = pd.read_sas(full, encoding=data.get("encoding", "utf-8"))
        else:
            df = pd.read_csv(full, nrows=5000)

        # Spaltennamen vereinheitlichen (UPPER), damit detect-columns dasselbe
        # zeigt wie DataPrep intern verwendet. Verhindert Case-Mismatches bei
        # Feature-Dictionary-Abgleich und NaN-Erkennung.
        df.columns = [str(c).upper() for c in df.columns]
        columns = list(df.columns)
        dtypes = {}
        nan_cols = []
        col_stats = {}
        for col in columns:
            is_obj = df[col].dtype.kind in ("O", "b")
            try:
                n_unique = int(df[col].nunique())
            except Exception:
                n_unique = 0
            try:
                null_pct = float(df[col].isnull().mean()) * 100
                if not math.isfinite(null_pct): null_pct = 0.0
            except Exception:
                null_pct = 0.0
            # Auto-Detection: Object/bool → cat, alles andere → num
            # Niedrige Kardinalität (n_unique ≤ 15) allein reicht NICHT als Kriterium,
            # da binäre Targets (0/1), Treatment-Indikatoren und niedrig-kardiale
            # numerische Features (z.B. Anzahl Claims 0–10) fälschlich als kategorisch
            # markiert würden. Nur der Datentyp entscheidet.
            if is_obj:
                dtypes[col] = "cat"
            else:
                dtypes[col] = "num"
            if df[col].isnull().any():
                nan_cols.append(col)
            stats = {"n_unique": n_unique, "null_pct": round(null_pct, 1)}
            if not is_obj:
                try:
                    v_min = float(df[col].min())
                    v_max = float(df[col].max())
                    v_mean = float(df[col].mean())
                    if math.isfinite(v_min): stats["min"] = v_min
                    if math.isfinite(v_max): stats["max"] = v_max
                    if math.isfinite(v_mean): stats["mean"] = round(v_mean, 4)
                except Exception:
                    pass
            col_stats[col] = stats

        # Sample values for treatment/target detection
        sample_values = {}
        for col in columns[:50]:
            try:
                uniq = df[col].dropna().unique()[:20]
                sample_values[col] = [safe_str(v) for v in uniq]
            except Exception:
                pass

        # Target values (for binary detection)
        target_col = data.get("target_column")
        target_values = []
        if target_col and target_col in df.columns:
            target_values = [safe_str(v) for v in sorted(df[target_col].dropna().unique()[:50])]

        # Treatment values
        treat_col = data.get("treatment_column")
        treat_values = []
        if treat_col and treat_col in df.columns:
            treat_values = [safe_str(v) for v in sorted(df[treat_col].dropna().unique()[:20])]

        return jsonify({
            "status": "done",
            "columns": columns,
            "dtypes": dtypes,
            "nan_cols": nan_cols,
            "col_stats": col_stats,
            "n_rows": int(len(df)),
            "n_cols": len(columns),
            "sample_values": sample_values,
            "target_values": target_values,
            "treat_values": treat_values,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/apply-dictionary", methods=["POST"])
def apply_dictionary():
    """Liest ein Feature Dictionary (Excel) und gibt die Spaltenauswahl + Typen zurück."""
    data = request.get_json(silent=True) or {}
    filepath = data.get("path", "")
    data_path = data.get("data_path", "")  # Rohdatei um vorhandene Spalten zu prüfen

    if not filepath:
        return jsonify({"status": "error", "message": "Kein Dictionary-Pfad angegeben."}), 400

    if os.path.isabs(filepath):
        full = Path(filepath).resolve()
    else:
        full = (ROOT / filepath).resolve()
        if not str(full).startswith(str(ROOT.resolve())):
            return jsonify({"status": "error", "message": "Ungültiger Pfad."}), 403
    if not full.exists():
        return jsonify({"status": "error", "message": f"Nicht gefunden: {filepath}"}), 404

    try:
        import pandas as pd

        ext = full.suffix.lower()
        dict_df = None
        # Versuche das passende Format — mit Fallback bei falsch benannten Dateien
        if ext == ".csv":
            dict_df = pd.read_csv(full)
        else:
            # .xlsx / .xls / unbekannt → Reihenfolge: openpyxl → xlrd → CSV
            for engine in ("openpyxl", "xlrd"):
                try:
                    dict_df = pd.read_excel(full, engine=engine)
                    break
                except Exception:
                    continue
            if dict_df is None:
                # Letzter Fallback: vielleicht ist es trotz Extension eine CSV
                try:
                    dict_df = pd.read_csv(full)
                except Exception:
                    return jsonify({"status": "error", "message": f"Datei konnte weder als Excel noch als CSV gelesen werden: {filepath}"}), 400

        # Spalten-Namen normalisieren (case-insensitive)
        col_map = {c.upper(): c for c in dict_df.columns}
        name_col = col_map.get("NAME") or col_map.get("VARIABLE") or col_map.get("FEATURE")
        role_col = col_map.get("ROLE") or col_map.get("ROLLE") or col_map.get("USE")
        level_col = col_map.get("LEVEL") or col_map.get("TYPE") or col_map.get("TYP") or col_map.get("DTYPE")

        if not name_col:
            return jsonify({"status": "error", "message": "Spalte 'NAME' (oder 'VARIABLE', 'FEATURE') nicht gefunden im Dictionary."}), 400
        if not role_col:
            return jsonify({"status": "error", "message": "Spalte 'ROLE' (oder 'ROLLE', 'USE') nicht gefunden im Dictionary."}), 400

        # Features filtern: ROLE=INPUT (case-insensitive)
        dict_df["_role_upper"] = dict_df[role_col].astype(str).str.strip().str.upper()
        dict_df["_name"] = dict_df[name_col].astype(str).str.strip()
        dict_df["_name_upper"] = dict_df["_name"].str.upper()
        input_mask = dict_df["_role_upper"] == "INPUT"
        input_features = dict_df.loc[input_mask, "_name_upper"].tolist()

        # Typen bestimmen: LEVEL=NOMINAL → cat, sonst num
        col_types = {}
        if level_col:
            dict_df["_level_upper"] = dict_df[level_col].astype(str).str.strip().str.upper()
            for _, row in dict_df[input_mask].iterrows():
                lev = row.get("_level_upper", "")
                col_types[row["_name_upper"]] = "cat" if lev in ("NOMINAL", "CATEGORICAL", "CAT", "BINARY") else "num"
        else:
            for f in input_features:
                col_types[f] = "num"

        # Alle Rollen aus dem Dictionary
        all_entries = []
        for _, row in dict_df.iterrows():
            entry = {"name": row["_name_upper"], "role": row["_role_upper"]}
            if level_col:
                entry["level"] = row.get("_level_upper", "")
            all_entries.append(entry)

        # Optional: Abgleich mit Rohdatei (case-insensitive: beide Seiten UPPER)
        matched = input_features
        missing_in_data = []
        all_data_columns = []
        data_dtypes = {}
        if data_path:
            dp = Path(data_path) if os.path.isabs(data_path) else (ROOT / data_path).resolve()
            if dp.exists():
                try:
                    ext2 = dp.suffix.lower()
                    if ext2 == ".parquet":
                        import pyarrow.parquet as pq
                        schema = pq.read_schema(dp)
                        all_data_columns = [str(c).upper() for c in schema.names]
                        data_cols = set(all_data_columns)
                    elif ext2 == ".csv":
                        delim = data.get("delimiter", ",")
                        df_head = pd.read_csv(dp, nrows=100, sep=delim)
                        all_data_columns = [str(c).upper() for c in df_head.columns]
                        data_cols = set(all_data_columns)
                        for col in df_head.columns:
                            is_obj = df_head[col].dtype.kind in ("O", "b")
                            data_dtypes[str(col).upper()] = "cat" if is_obj else "num"
                    else:
                        df_head = pd.read_csv(dp, nrows=100)
                        all_data_columns = [str(c).upper() for c in df_head.columns]
                        data_cols = set(all_data_columns)
                    missing_in_data = [f for f in input_features if f not in data_cols]
                    matched = [f for f in input_features if f in data_cols]
                except Exception:
                    pass

        return jsonify({
            "status": "done",
            "input_features": input_features,
            "matched_features": matched,
            "missing_in_data": missing_in_data,
            "col_types": col_types,
            "all_entries": all_entries,
            "all_data_columns": all_data_columns,
            "data_dtypes": data_dtypes,
            "total_in_dict": len(dict_df),
            "total_input": len(input_features),
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/export-feature-dict", methods=["POST"])
def export_feature_dict():
    """Exportiert die aktuelle Feature-Auswahl als Feature Dictionary (Excel).

    Erwartet JSON mit:
      columns: Liste aller Spaltennamen
      featureSelection: {col: true/false}
      colTypes: {col: "num"|"cat"}
      target: str, treatment: str, scoreName: str (optionale Rollen)
      colStats: {col: {n_unique, null_pct, ...}} (optional)
    """
    data = request.get_json(silent=True) or {}
    columns = data.get("columns", [])
    selection = data.get("featureSelection", {})
    col_types = data.get("colTypes", {})
    target_raw = data.get("target") or ""
    if isinstance(target_raw, list):
        target_set = {str(t).upper() for t in target_raw if t}
    else:
        target_set = {str(target_raw).upper()} if target_raw else set()
    treatment = (data.get("treatment") or "").upper()
    score_name = (data.get("scoreName") or "").upper()
    col_stats = data.get("colStats", {})

    if not columns:
        return jsonify({"status": "error", "message": "Keine Spalten übergeben."}), 400

    try:
        import pandas as pd

        rows = []
        reserved = target_set | {treatment}
        if score_name:
            reserved.add(score_name)

        for col in columns:
            col_upper = col.upper() if col else col
            if col_upper in target_set:
                role = "TARGET"
            elif col_upper == treatment:
                role = "TREATMENT"
            elif score_name and col_upper == score_name:
                role = "SCORE"
            elif selection.get(col, True) is False:
                role = "EXCLUDE"
            else:
                role = "INPUT"

            ct = col_types.get(col, "num")
            level = "NOMINAL" if ct == "cat" else "INTERVAL"

            row = {"NAME": col, "ROLE": role, "LEVEL": level}
            stats = col_stats.get(col, {})
            if stats:
                row["N_UNIQUE"] = stats.get("n_unique", "")
                row["NULL_PCT"] = stats.get("null_pct", "")
                if "min" in stats:
                    row["MIN"] = stats["min"]
                if "max" in stats:
                    row["MAX"] = stats["max"]
                if "mean" in stats:
                    row["MEAN"] = stats["mean"]
            rows.append(row)

        dict_df = pd.DataFrame(rows)
        out_dir = WORK_DIR / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "feature_dictionary.xlsx"
        dict_df.to_excel(str(out_path), index=False, engine="openpyxl")

        return jsonify({
            "status": "done",
            "path": str(out_path.relative_to(ROOT)) if str(out_path).startswith(str(ROOT.resolve())) else str(out_path),
            "n_input": len([r for r in rows if r["ROLE"] == "INPUT"]),
            "n_exclude": len([r for r in rows if r["ROLE"] == "EXCLUDE"]),
            "n_nominal": len([r for r in rows if r["LEVEL"] == "NOMINAL" and r["ROLE"] == "INPUT"]),
            "n_interval": len([r for r in rows if r["LEVEL"] == "INTERVAL" and r["ROLE"] == "INPUT"]),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

