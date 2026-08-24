import sys
!{sys.executable} -m pip install ibm_db


# ============================================================================
# ZELLE 1 - Funktionen (einmal ausführen)
# ============================================================================
import logging
import os
import re
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd
import ibm_db_dbi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger("db2")


class Db2WriteError(RuntimeError):
    """Basisklasse für alle Fehler des Writers."""


class SchemaMismatchError(Db2WriteError):
    """DataFrame-Spalten passen nicht zur Zieltabelle."""


class ValueValidationError(Db2WriteError):
    """Ein Wert verletzt die Constraints der Zielspalte."""


def add_versanddatum(df, start="2027-10-15", end="2027-11-26",
                     only_business_days=False, shuffle=True, random_state=42):
    """Verteilt Versanddatums gleichmäßig über den gesamten Bestand.

    Rückgabe: Kopie von df mit zusätzlicher Spalte 'versanddatum' (str, yyyy-mm-dd).
    """
    dates = pd.date_range(start, end, freq="B" if only_business_days else "D")
    if len(dates) == 0:
        raise ValueError("Kein gültiges Datum im Bereich %s bis %s." % (start, end))

    date_str = dates.strftime("%Y-%m-%d").to_numpy()
    values = np.resize(date_str, len(df))  # zyklisch -> balancierte Häufigkeiten

    if shuffle:
        np.random.default_rng(random_state).shuffle(values)

    out = df.copy()
    out["versanddatum"] = values
    return out


def _clean_identifier(name):
    """DataFrame-Spaltenname -> gültiger DB2-Bezeichner (unquoted, uppercase).

    Wird beim CREATE TABLE und beim Spaltenabgleich verwendet, damit z. B.
    've produkt' und 'VE_PRODUKT' zusammenfinden.
    """
    ident = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(name))
    ident = ident.upper().strip("_")
    if not ident:
        raise ValueError("Spaltenname %r ergibt keinen gültigen Bezeichner." % name)
    if not ident[0].isalpha():
        ident = "C_" + ident
    return ident[:128]


def build_dsn(prefix="DB2_"):
    """Baut den Connection-String aus Environment-Variablen."""
    parts = [
        "DATABASE=" + os.environ[prefix + "DATABASE"],
        "HOSTNAME=" + os.environ[prefix + "HOSTNAME"],
        "PORT=" + os.environ.get(prefix + "PORT", "50000"),
        "PROTOCOL=TCPIP",
        "UID=" + os.environ[prefix + "USER"],
        "PWD=" + os.environ[prefix + "PASSWORD"],
    ]
    if os.environ.get(prefix + "SSL", "").lower() in {"1", "true", "yes"}:
        parts.append("SECURITY=SSL")
        cert = os.environ.get(prefix + "SSL_CERT")
        if cert:
            parts.append("SSLServerCertificate=" + cert)
    return ";".join(parts) + ";"


@contextmanager
def db2_connect(autocommit=False, prefix="DB2_"):
    """Context-Manager für eine DB2-Verbindung mit garantiertem close()."""
    conn = ibm_db_dbi.connect(build_dsn(prefix), "", "")
    try:
        conn.set_autocommit(autocommit)
        log.info("Verbunden mit %s als %s",
                 os.environ[prefix + "DATABASE"], os.environ[prefix + "USER"])
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            log.warning("Verbindung konnte nicht sauber geschlossen werden.")


@contextmanager
def _passthrough(c):
    yield c


_SCHEMA_SQL = """
    SELECT COLNAME, COLNO, TYPENAME, LENGTH, SCALE, NULLS,
           IDENTITY, GENERATED, DEFAULT
      FROM SYSCAT.COLUMNS
     WHERE TABSCHEMA = ? AND TABNAME = ?
     ORDER BY COLNO
"""


def fetch_table_schema(conn, schema, table):
    """Liest die Spaltendefinition der Zieltabelle aus dem Systemkatalog."""
    cur = conn.cursor()
    try:
        cur.execute(_SCHEMA_SQL, (schema.upper(), table.upper()))
        rows = cur.fetchall()
    finally:
        cur.close()

    if not rows:
        raise SchemaMismatchError(
            "Tabelle %s.%s existiert nicht oder der technische User hat keine "
            "Leseberechtigung auf SYSCAT.COLUMNS."
            % (schema.upper(), table.upper())
        )

    meta = {}
    for (colname, colno, typename, length, scale, nulls,
         identity, generated, default) in rows:
        name = str(colname).strip()
        meta[name] = {
            "name": name,
            "colno": int(colno),
            "typename": str(typename).strip().upper(),
            "length": int(length or 0),
            "scale": int(scale or 0),
            "nullable": str(nulls).strip().upper() == "Y",
            "identity": str(identity).strip().upper() == "Y",
            "generated": str(generated or "").strip().upper() in {"A", "D"},
            "has_default": default is not None,
        }
    return meta


def schema_as_frame(meta):
    """Katalog-Metadaten als DataFrame, zum Anschauen im Notebook."""
    return (pd.DataFrame(list(meta.values()))
              .sort_values("colno")
              .reset_index(drop=True))


def table_exists(conn, schema, table):
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM SYSCAT.TABLES WHERE TABSCHEMA = ? AND TABNAME = ?",
            (schema.upper(), table.upper()),
        )
        return cur.fetchone() is not None
    finally:
        cur.close()


# ---------------------------------------------------------------------------
# DDL-Generator
# ---------------------------------------------------------------------------
_VARCHAR_STAGES = (10, 20, 30, 50, 100, 255, 500, 1000, 2000, 4000)
_MAX_VARCHAR = 32672
_ID_PATTERN = re.compile(r"(NR|NUMMER|ID|SCHLUESSEL|SCHL|KEY)$|(^|_)(ID|NR|KEY)(_|$)")


def _varchar_length(max_bytes):
    """Nächste sinnvolle Stufe mit 50 % Reserve."""
    needed = max(10, int(max_bytes * 1.5) + 1)
    for stage in _VARCHAR_STAGES:
        if needed <= stage:
            return stage
    return min(_MAX_VARCHAR, needed)


def infer_db2_type(s):
    """Leitet einen DB2-Typ aus einer pandas-Series ab."""
    non_null = s.dropna()

    if str(s.dtype) in {"bool", "boolean"}:
        return "SMALLINT"          # 0/1, robuster als BOOLEAN über ibm_db

    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        if non_null.empty:
            return "TIMESTAMP"
        return "DATE" if (non_null.dt.normalize() == non_null).all() else "TIMESTAMP"

    if pd.api.types.is_integer_dtype(s.dtype):
        if non_null.empty:
            return "INTEGER"
        lo, hi = int(non_null.min()), int(non_null.max())
        if -32768 <= lo and hi <= 32767:
            return "SMALLINT"
        if -2147483648 <= lo and hi <= 2147483647:
            return "INTEGER"
        return "BIGINT"

    if pd.api.types.is_float_dtype(s.dtype):
        if non_null.empty:
            return "DOUBLE"
        vals = non_null.to_numpy(dtype="float64")
        if not np.all(np.isfinite(vals)):
            return "DOUBLE"
        if np.allclose(vals, np.round(vals, 2), atol=1e-9):
            int_digits = max(1, len(str(int(np.max(np.abs(vals))))))
            return "DECIMAL(%d,2)" % min(31, int_digits + 2 + 3)   # 3 Stellen Reserve
        return "DOUBLE"

    # object / string / category
    text = non_null.astype(str)
    if text.empty:
        log.warning("Spalte enthält nur NULL-Werte -> VARCHAR(50) als Notbehelf.")
        return "VARCHAR(50)"
    if text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        return "DATE"
    if text.str.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}.*").all():
        return "TIMESTAMP"
    max_bytes = int(text.map(lambda v: len(v.encode("utf-8"))).max())
    return "VARCHAR(%d)" % _varchar_length(max_bytes)


def suggest_ddl(df, table, schema, not_null=(), primary_key=(),
                add_audit_columns=True, overrides=None, tablespace=None):
    """Erzeugt einen CREATE-TABLE-Vorschlag aus dem DataFrame.

    not_null          : Spalten, die NOT NULL werden sollen
    primary_key       : Spalten des Primärschlüssels (impliziert NOT NULL)
    add_audit_columns : LADE_TS mit DEFAULT CURRENT TIMESTAMP ergänzen
    overrides         : {"spalte": "DECIMAL(15,2)"} zum manuellen Übersteuern
    """
    overrides = {k.upper(): v for k, v in (overrides or {}).items()}
    nn = ({_clean_identifier(c) for c in not_null}
          | {_clean_identifier(c) for c in primary_key})
    pk = [_clean_identifier(c) for c in primary_key]

    lines, seen = [], set()
    for col in df.columns:
        ident = _clean_identifier(col)
        if ident in seen:
            raise ValueError("Spaltenname %r kollidiert nach Bereinigung mit einer "
                             "bereits vergebenen Spalte %r." % (col, ident))
        seen.add(ident)
        if ident in overrides:
            db_type = overrides[ident]
        else:
            db_type = infer_db2_type(df[col])
            # Schlüsselspalten nie auf SMALLINT festnageln: die Stichprobe sagt
            # nichts über künftige Wertebereiche aus.
            if db_type == "SMALLINT" and _ID_PATTERN.search(ident):
                log.info("Spalte '%s' sieht nach einem Schlüssel aus -> "
                         "INTEGER statt SMALLINT.", ident)
                db_type = "INTEGER"
        suffix = " NOT NULL" if ident in nn else ""
        lines.append('    "%s" %s%s' % (ident, db_type, suffix))

    if add_audit_columns:
        lines.append('    "LADE_TS" TIMESTAMP NOT NULL DEFAULT CURRENT TIMESTAMP')

    if pk:
        missing = [c for c in pk if c not in seen]
        if missing:
            raise ValueError("Primärschlüsselspalten nicht im DataFrame: %s" % missing)
        lines.append('    PRIMARY KEY (%s)' % ", ".join('"%s"' % c for c in pk))

    ddl = 'CREATE TABLE "%s"."%s" (\n%s\n)' % (schema.upper(), table.upper(),
                                               ",\n".join(lines))
    if tablespace:
        ddl += '\n  IN "%s"' % tablespace
    return ddl + ";"


def execute_ddl(ddl, conn=None):
    """Führt DDL aus und committet. Ohne 'conn' wird selbst verbunden."""
    ctx = _passthrough(conn) if conn is not None else db2_connect()
    with ctx as connection:
        connection.set_autocommit(False)
        cur = connection.cursor()
        try:
            for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
                log.info("Führe aus: %s ...", stmt.splitlines()[0])
                cur.execute(stmt)
            connection.commit()
            log.info("DDL erfolgreich ausgeführt.")
        except Exception:
            connection.rollback()
            log.error("DDL fehlgeschlagen, Rollback ausgeführt.")
            raise
        finally:
            cur.close()


# ---------------------------------------------------------------------------
# Typ-Coercion
# ---------------------------------------------------------------------------
_INT_TYPES = {"SMALLINT", "INTEGER", "BIGINT"}
_DEC_TYPES = {"DECIMAL", "NUMERIC"}
_FLOAT_TYPES = {"REAL", "DOUBLE", "FLOAT", "DECFLOAT"}
_STR_TYPES = {"CHARACTER", "CHAR", "VARCHAR", "LONG VARCHAR", "CLOB",
              "GRAPHIC", "VARGRAPHIC", "DBCLOB"}
_INT_RANGE = {
    "SMALLINT": (-32768, 32767),
    "INTEGER": (-2147483648, 2147483647),
    "BIGINT": (-9223372036854775808, 9223372036854775807),
}


def _fail(meta, idx, value, reason):
    raise ValueValidationError(
        "Spalte '%s' (%s), DataFrame-Index %r: %s (Wert: %r)"
        % (meta["name"], meta["typename"], idx, reason, value)
    )


def _coerce_series(s, meta, truncate_strings=False):
    """Wandelt eine Series in ein object-Array mit nativen Python-Typen und None."""
    na = pd.isna(s).to_numpy()

    if not meta["nullable"] and na.any():
        raise ValueValidationError(
            "Spalte '%s' ist in DB2 NOT NULL, enthält aber %d fehlende Werte. "
            "Erste betroffene Indizes: %s"
            % (meta["name"], int(na.sum()), s.index[na][:5].tolist())
        )

    out = np.empty(len(s), dtype=object)
    out[:] = None
    keep = ~na
    if not keep.any():
        return out

    t = meta["typename"]

    # ---- Ganzzahlen (inkl. boolean -> 0/1) --------------------------------
    if t in _INT_TYPES:
        vals = s[keep]
        if pd.api.types.is_bool_dtype(vals.dtype) or str(vals.dtype) == "boolean":
            nums = vals.astype("int64")
        else:
            nums = pd.to_numeric(vals, errors="coerce")
            if nums.isna().any():
                idx = nums.index[nums.isna()][0]
                _fail(meta, idx, s.loc[idx], "nicht in eine Zahl konvertierbar")
            frac = np.mod(nums.to_numpy(dtype="float64"), 1.0)
            if np.any(np.abs(frac) > 1e-9):
                idx = nums.index[np.abs(frac) > 1e-9][0]
                _fail(meta, idx, s.loc[idx],
                      "hat Nachkommastellen, Ziel ist ganzzahlig")
        lo, hi = _INT_RANGE[t]
        arr = nums.to_numpy(dtype="float64")
        over = (arr < lo) | (arr > hi)
        if over.any():
            idx = nums.index[over][0]
            _fail(meta, idx, s.loc[idx],
                  "außerhalb des %s-Bereichs [%d, %d]" % (t, lo, hi))
        out[keep] = [int(v) for v in nums.to_numpy()]
        return out

    # ---- Dezimalzahlen ----------------------------------------------------
    if t in _DEC_TYPES:
        precision, scale = meta["length"], meta["scale"]
        max_abs = Decimal(10) ** (precision - scale)
        result = []
        for idx, v in s[keep].items():
            try:
                d = Decimal(str(v)).quantize(Decimal(1).scaleb(-scale))
            except (InvalidOperation, ValueError, TypeError):
                _fail(meta, idx, v, "nicht in DECIMAL konvertierbar")
            if abs(d) >= max_abs:
                _fail(meta, idx, v,
                      "überschreitet DECIMAL(%d,%d)" % (precision, scale))
            result.append(d)
        out[keep] = result
        return out

    # ---- Gleitkomma -------------------------------------------------------
    if t in _FLOAT_TYPES:
        nums = pd.to_numeric(s[keep], errors="coerce")
        if nums.isna().any():
            idx = nums.index[nums.isna()][0]
            _fail(meta, idx, s.loc[idx], "nicht in eine Zahl konvertierbar")
        out[keep] = [float(v) for v in nums.to_numpy()]
        return out

    # ---- Echtes BOOLEAN ---------------------------------------------------
    if t == "BOOLEAN":
        out[keep] = [bool(v) for v in s[keep].to_numpy()]
        return out

    # ---- Datum / Zeit -----------------------------------------------------
    if t in {"DATE", "TIMESTAMP", "TIME"}:
        vals = s[keep]
        dt = None
        if vals.dtype == object:
            # Schnellpfad für ISO-Strings; sonst parst pandas jede Zelle einzeln
            try:
                dt = pd.to_datetime(vals, format="%Y-%m-%d", errors="raise")
            except Exception:
                dt = None
        if dt is None:
            try:
                dt = pd.to_datetime(vals, errors="raise")
            except Exception as exc:
                raise ValueValidationError(
                    "Spalte '%s': Werte lassen sich nicht als %s interpretieren (%s)"
                    % (meta["name"], t, exc)
                )
        if t == "DATE":
            out[keep] = [v.date() for v in dt]
        elif t == "TIME":
            out[keep] = [v.time() for v in dt]
        else:
            out[keep] = [v.to_pydatetime() for v in dt]
        return out

    # ---- Zeichenketten ----------------------------------------------------
    if t in _STR_TYPES:
        strs = s[keep].astype(str)
        limit = meta["length"]
        if limit > 0:
            blen = strs.map(lambda v: len(v.encode("utf-8")))
            over = blen > limit
            if over.any():
                if not truncate_strings:
                    idx = strs.index[over][0]
                    _fail(meta, idx, s.loc[idx],
                          "ist %d Byte lang, %s(%d) erlaubt maximal %d"
                          % (int(blen.loc[idx]), t, limit, limit))
                log.warning("Spalte '%s': %d Werte werden auf %d Byte gekürzt.",
                            meta["name"], int(over.sum()), limit)
                strs = strs.map(
                    lambda v: v.encode("utf-8")[:limit].decode("utf-8", "ignore")
                )
        out[keep] = strs.tolist()
        return out

    # ---- Unbekannter Typ --------------------------------------------------
    log.warning("Spalte '%s': unbekannter DB2-Typ '%s', Werte werden "
                "unverändert übergeben.", meta["name"], t)
    out[keep] = [v.item() if isinstance(v, np.generic) else v for v in s[keep]]
    return out


def _resolve_columns(df, table_meta, allow_extra_columns=False):
    """Mappt DataFrame-Spalten auf Katalog-Spalten."""
    lookup = {k.upper(): k for k in table_meta}
    mapping, unknown = {}, []

    for col in df.columns:
        key = str(col).strip().upper()
        if key in lookup:
            mapping[col] = lookup[key]
            continue
        # zweiter Versuch mit derselben Normalisierung wie beim CREATE TABLE
        try:
            key = _clean_identifier(col)
        except ValueError:
            key = None
        if key in lookup:
            mapping[col] = lookup[key]
        else:
            unknown.append(str(col))

    if unknown:
        msg = ("Diese DataFrame-Spalten existieren nicht in der Zieltabelle: %s. "
               "Vorhandene Spalten: %s" % (unknown, sorted(table_meta)))
        if not allow_extra_columns:
            raise SchemaMismatchError(msg)
        log.warning("%s -- Spalten werden ignoriert.", msg)

    targets = list(mapping.values())
    dupes = {v for v in targets if targets.count(v) > 1}
    if dupes:
        raise SchemaMismatchError(
            "Mehrere DataFrame-Spalten zeigen auf dieselbe DB2-Spalte: %s" % dupes
        )

    supplied = set(targets)
    missing_required = [
        m["name"] for m in table_meta.values()
        if m["name"] not in supplied and not m["nullable"]
        and not (m["identity"] or m["generated"] or m["has_default"])
    ]
    if missing_required:
        raise SchemaMismatchError(
            "Im DataFrame fehlen NOT-NULL-Spalten ohne Default: %s" % missing_required
        )

    not_filled = sorted(set(table_meta) - supplied)
    if not_filled:
        log.info("Nicht befüllte Spalten (nullable oder mit Default): %s", not_filled)

    if not mapping:
        raise SchemaMismatchError("Keine einzige DataFrame-Spalte passt zur Zieltabelle.")

    return mapping


def _build_rows(df, mapping, table_meta, truncate_strings=False):
    """Validiert und konvertiert den kompletten DataFrame."""
    db_cols = sorted(mapping.values(), key=lambda c: table_meta[c]["colno"])
    reverse = {v: k for k, v in mapping.items()}
    columns = [
        _coerce_series(df[reverse[c]], table_meta[c], truncate_strings)
        for c in db_cols
    ]
    rows = list(zip(*columns)) if columns else []
    return db_cols, rows


def _diagnose(conn, sql, chunk, offset, index):
    """Nach einem fehlgeschlagenen Chunk die konkrete Problemzeile finden."""
    try:
        conn.rollback()
        cur = conn.cursor()
        try:
            for i, row in enumerate(chunk):
                try:
                    cur.execute(sql, row)
                except Exception as exc:
                    label = index[offset + i] if offset + i < len(index) else offset + i
                    return ("Erste fehlerhafte Zeile: DataFrame-Index %r\n"
                            "Werte: %r\nDB2-Meldung: %s" % (label, row, exc))
        finally:
            cur.close()
            conn.rollback()
    except Exception as exc:
        return "Diagnose nicht möglich: %s" % exc
    return "Einzelne Zeilen liefen durch; der Fehler tritt nur im Batch auf."


def write_dataframe(df, table, schema, conn=None, chunksize=10000, limit=None,
                    dry_run=False, allow_extra_columns=False,
                    truncate_strings=False, diagnose_errors=True):
    """Schreibt einen DataFrame per INSERT in eine bestehende DB2-Tabelle.

    Es wird nichts angelegt und nichts gelöscht. Eine Transaktion, ein Commit
    am Ende. Bei jedem Fehler vollständiger Rollback, also keine Teil-Inserts.

    limit    : nur die ersten n Zeilen schreiben (Testlauf)
    dry_run  : vollständige Validierung, aber kein Insert
    conn     : bestehende Verbindung; ohne Angabe wird selbst eine geöffnet

    Rückgabe: Anzahl geschriebener Zeilen (0 bei dry_run).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df muss ein pandas.DataFrame sein.")

    work = df.head(limit) if limit is not None else df
    if work.empty:
        log.warning("DataFrame ist leer, es wird nichts geschrieben.")
        return 0

    ctx = _passthrough(conn) if conn is not None else db2_connect()

    with ctx as connection:
        connection.set_autocommit(False)

        table_meta = fetch_table_schema(connection, schema, table)
        mapping = _resolve_columns(work, table_meta, allow_extra_columns)
        db_cols, rows = _build_rows(work, mapping, table_meta, truncate_strings)

        qualified = '"%s"."%s"' % (schema.upper(), table.upper())
        collist = ", ".join('"%s"' % c for c in db_cols)
        placeholders = ", ".join("?" * len(db_cols))
        sql = "INSERT INTO %s (%s) VALUES (%s)" % (qualified, collist, placeholders)

        log.info("Validierung erfolgreich: %d Zeilen, %d Spalten -> %s",
                 len(rows), len(db_cols), qualified)

        if dry_run:
            log.info("dry_run=True -- es wurde nichts geschrieben.")
            return 0

        cur = connection.cursor()
        written = 0
        try:
            for start in range(0, len(rows), chunksize):
                chunk = rows[start:start + chunksize]
                try:
                    cur.executemany(sql, chunk)
                except Exception as exc:
                    detail = ""
                    if diagnose_errors:
                        cur.close()
                        detail = "\n" + _diagnose(connection, sql, chunk,
                                                  start, work.index)
                        cur = connection.cursor()
                    raise Db2WriteError(
                        "Insert fehlgeschlagen bei Zeilen %d-%d: %s%s"
                        % (start, start + len(chunk) - 1, exc, detail)
                    )
                written += len(chunk)
                log.info("... %d / %d Zeilen gepuffert", written, len(rows))

            connection.commit()
            log.info("Commit erfolgreich: %d Zeilen in %s geschrieben.",
                     written, qualified)
            return written

        except Exception:
            try:
                connection.rollback()
                log.error("Rollback ausgeführt -- es wurden keine Daten geschrieben.")
            except Exception:
                log.critical("Rollback fehlgeschlagen. Datenstand bitte prüfen!")
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass


print("Funktionen geladen.")
