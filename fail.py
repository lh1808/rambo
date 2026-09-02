"""
db2_export.py
=============

Verbindet sich mit einer DB2-Datenbank, zieht eine Tabelle (oder eine beliebige
Abfrage) komplett ab und speichert sie als Parquet-Datei.

Voraussetzung:  pip install ibm_db pandas pyarrow
(ibm_db_dbi ist Teil von ibm_db - NICHT separat installieren)

Nach einer Neuinstallation ggf. den Kernel neu starten.
"""

import os
from contextlib import contextmanager

import ibm_db_dbi
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ======================================================================================
# 1) Verbindungsdaten - hier ausfüllen
# ======================================================================================
CONFIG = {
    "host": os.environ.get("DB2_HOST", "..."),        # z. B. "SDTSDAHF1"
    "port": os.environ.get("DB2_PORT", "50000"),      # DEV war 55000
    "database": os.environ.get("DB2_DB_NAME", "..."),  # z. B. "DTSDAHF1"
    "user": os.environ.get("DB2_USER", "..."),
    "password": os.environ.get("DB2_PASSWORD", "..."),
    "security": None,                                  # "SSL" falls verschlüsselt
}

SCHEMA = os.environ.get("DB2_SCHEMA", "...")
TABELLE = "..."

# Alternativ eine eigene Abfrage setzen; None = ganze Tabelle
ABFRAGE = None

ZIEL = "export.parquet"
CHUNKSIZE = 200_000          # Zeilen je Block; None = alles auf einmal


# ======================================================================================
# 2) Verbindung
# ======================================================================================
def dsn(config: dict = CONFIG) -> str:
    """Baut den DB2-Connection-String."""
    teile = [
        f"DATABASE={config['database']}",
        f"HOSTNAME={config['host']}",
        f"PORT={config['port']}",
        "PROTOCOL=TCPIP",
        f"UID={config['user']}",
        f"PWD={config['password']}",
    ]
    if config.get("security"):
        teile.append(f"SECURITY={config['security']}")
    return ";".join(teile) + ";"


@contextmanager
def verbindung(config: dict = CONFIG):
    """Öffnet eine DB2-Verbindung und schließt sie in jedem Fall wieder."""
    fehlend = [k for k, v in config.items()
               if k != "security" and (v is None or str(v).startswith("..."))]
    if fehlend:
        raise ValueError(f"Verbindungsdaten unvollständig: {', '.join(fehlend)}")

    conn = ibm_db_dbi.connect(dsn(config), "", "")
    try:
        yield conn
    finally:
        conn.close()


def zeilenzahl(conn, schema: str = SCHEMA, tabelle: str = TABELLE) -> int:
    """Zeilen der Zieltabelle - für eine Erwartung vor dem Abzug."""
    cur = conn.cursor()
    cur.execute(f'SELECT COUNT(*) FROM "{schema.upper()}"."{tabelle.upper()}"')
    n = cur.fetchone()[0]
    cur.close()
    return int(n)


# ======================================================================================
# 3) Export
# ======================================================================================
def exportiere(ziel: str = ZIEL, abfrage: str | None = ABFRAGE,
               schema: str = SCHEMA, tabelle: str = TABELLE,
               chunksize: int | None = CHUNKSIZE, config: dict = CONFIG) -> str:
    """Zieht die Daten ab und schreibt sie nach Parquet."""
    sql = abfrage or f'SELECT * FROM "{schema.upper()}"."{tabelle.upper()}"'

    with verbindung(config) as conn:
        if abfrage is None:
            erwartet = zeilenzahl(conn, schema, tabelle)
            print(f"Erwartet: {erwartet:,} Zeilen")

        cur = conn.cursor()
        cur.execute(sql)
        spalten = [b[0] for b in cur.description]

        if not chunksize:
            df = pd.DataFrame(cur.fetchall(), columns=spalten)
            cur.close()
            df.to_parquet(ziel, engine="pyarrow", compression="snappy", index=False)
            print(f"{len(df):,} Zeilen, {len(df.columns)} Spalten -> {ziel}")
            return ziel

        writer, gesamt = None, 0
        try:
            while True:
                block = cur.fetchmany(chunksize)
                if not block:
                    break
                teil = pd.DataFrame(block, columns=spalten)
                tab = pa.Table.from_pandas(teil, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(ziel, tab.schema, compression="snappy")
                else:
                    # Schemaabweichung zwischen Blöcken abfangen
                    tab = tab.cast(writer.schema)
                writer.write_table(tab)
                gesamt += len(teil)
                print(f"  {gesamt:,} Zeilen geschrieben", end="\r")
        finally:
            if writer is not None:
                writer.close()
            cur.close()

    print(f"\n{gesamt:,} Zeilen, {len(spalten)} Spalten -> {ziel}")
    return ziel


if __name__ == "__main__":
    exportiere()

    # Gegenprobe
    df = pd.read_parquet(ZIEL)
    print(df.shape)
    print(df.dtypes)
    print(df.head())
