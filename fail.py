import os
from contextlib import contextmanager

import ibm_db_dbi
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --- Verbindungsdaten: hier ausfüllen (oder aus Domino-Umgebungsvariablen) ---
CONFIG = {
    "host":     os.environ.get("DB2_HOST", "..."),        # z. B. "SDTSDAHF1"
    "port":     os.environ.get("DB2_PORT", "50000"),      # DEV war 55000
    "database": os.environ.get("DB2_DB_NAME", "..."),     # z. B. "DTSDAHF1"
    "user":     os.environ.get("DB2_USER", "..."),
    "password": os.environ.get("DB2_PASSWORD", "..."),
    "security": None,                                     # "SSL" falls verschlüsselt
}

SCHEMA = os.environ.get("DB2_SCHEMA", "...")
TABELLE = "..."
ABFRAGE = None            # None = ganze Tabelle, sonst eigenes SELECT
ZIEL = "export.parquet"
CHUNKSIZE = 200_000       # None = alles auf einmal


def dsn(config=CONFIG):
    """Baut den DB2-Connection-String."""
    teile = [f"DATABASE={config['database']}", f"HOSTNAME={config['host']}",
             f"PORT={config['port']}", "PROTOCOL=TCPIP",
             f"UID={config['user']}", f"PWD={config['password']}"]
    if config.get("security"):
        teile.append(f"SECURITY={config['security']}")
    return ";".join(teile) + ";"


@contextmanager
def verbindung(config=CONFIG):
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


def zeilenzahl(conn, schema=None, tabelle=None):
    """Zeilen der Zieltabelle - als Erwartung vor dem Abzug."""
    cur = conn.cursor()
    cur.execute(f'SELECT COUNT(*) FROM "{(schema or SCHEMA).upper()}"'
                f'."{(tabelle or TABELLE).upper()}"')
    n = cur.fetchone()[0]
    cur.close()
    return int(n)


def exportiere(ziel=None, abfrage=None, schema=None, tabelle=None,
               chunksize=None, config=CONFIG):
    """Zieht die Daten ab und schreibt sie nach Parquet."""
    ziel = ziel or ZIEL
    schema, tabelle = schema or SCHEMA, tabelle or TABELLE
    abfrage = abfrage if abfrage is not None else ABFRAGE
    chunksize = CHUNKSIZE if chunksize is None else chunksize
    sql = abfrage or f'SELECT * FROM "{schema.upper()}"."{tabelle.upper()}"'

    with verbindung(config) as conn:
        if abfrage is None:
            print(f"Erwartet: {zeilenzahl(conn, schema, tabelle):,} Zeilen")

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
                tab = pa.Table.from_pandas(pd.DataFrame(block, columns=spalten),
                                           preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(ziel, tab.schema, compression="snappy")
                else:
                    tab = tab.cast(writer.schema)   # Schemaabweichung zwischen Blöcken
                writer.write_table(tab)
                gesamt += len(block)
                print(f"  {gesamt:,} Zeilen geschrieben", end="\r")
        finally:
            if writer is not None:
                writer.close()
            cur.close()

    print(f"\n{gesamt:,} Zeilen, {len(spalten)} Spalten -> {ziel}")
    return ziel





exportiere()

df = pd.read_parquet(ZIEL)
print(df.shape)
print(df.dtypes)
df.head()
