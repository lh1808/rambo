def tabellen(schema=None, config=CONFIG, mit_zeilenzahl=False):
    """
    Listet alle Tabellen und Views eines Schemas aus dem DB2-Katalog.
    mit_zeilenzahl=True zählt jede Tabelle einzeln aus.
    """
    schema = (schema or SCHEMA).upper()
    sql = """
        SELECT TABNAME, TYPE, COLCOUNT, CARD, CREATE_TIME, REMARKS
        FROM SYSCAT.TABLES
        WHERE TABSCHEMA = ?
        ORDER BY TABNAME
    """
    with verbindung(config) as conn:
        cur = conn.cursor()
        cur.execute(sql, (schema,))
        spalten = [b[0] for b in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=spalten)
        cur.close()

        df["TYPE"] = df["TYPE"].map({"T": "Tabelle", "V": "View", "A": "Alias",
                                     "N": "Nickname", "S": "MQT"}).fillna(df["TYPE"])
        df = df.rename(columns={"CARD": "ZEILEN_STATISTIK"})

        if mit_zeilenzahl:
            echte = []
            for name in df["TABNAME"]:
                try:
                    c = conn.cursor()
                    c.execute(f'SELECT COUNT(*) FROM "{schema}"."{name}"')
                    echte.append(int(c.fetchone()[0]))
                    c.close()
                except Exception:
                    echte.append(None)
            df["ZEILEN_GEZAEHLT"] = echte

    return df


def schemata(config=CONFIG):
    """Alle Schemata mit Tabellen, auf die du Leserechte hast."""
    sql = """
        SELECT TABSCHEMA, COUNT(*) AS TABELLEN
        FROM SYSCAT.TABLES
        WHERE TYPE IN ('T', 'V')
        GROUP BY TABSCHEMA
        ORDER BY TABSCHEMA
    """
    with verbindung(config) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        df = pd.DataFrame(cur.fetchall(), columns=[b[0] for b in cur.description])
        cur.close()
    return df


def suche_tabelle(muster, config=CONFIG):
    """Sucht schemaübergreifend nach Tabellennamen, z. B. suche_tabelle('%BESTAND%')."""
    sql = """
        SELECT TABSCHEMA, TABNAME, TYPE, COLCOUNT
        FROM SYSCAT.TABLES
        WHERE TABNAME LIKE ?
        ORDER BY TABSCHEMA, TABNAME
    """
    with verbindung(config) as conn:
        cur = conn.cursor()
        cur.execute(sql, (muster.upper(),))
        df = pd.DataFrame(cur.fetchall(), columns=[b[0] for b in cur.description])
        cur.close()
    return df
