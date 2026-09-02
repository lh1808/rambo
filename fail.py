def tabellen(schema=None, config=CONFIG, mit_zeilenzahl=False):
    """
    Listet alle Tabellen und Views eines Schemas aus dem DB2-Katalog.

    mit_zeilenzahl=True zählt jede Tabelle einzeln aus (dauert bei vielen
    oder großen Tabellen entsprechend).
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
            for name, typ in zip(df["TABNAME"], df["TYPE"]):
                try:
                    c = conn.cursor()
                    c.execute(f'SELECT COUNT(*) FROM "{schema}"."{name}"')
                    echte.append(int(c.fetchone()[0]))
                    c.close()
                except Exception:
                    echte.append(None)
            df["ZEILEN_GEZAEHLT"] = echte

    return df
