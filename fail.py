with db2_connect() as conn:
    schemata = pd.read_sql("""
        SELECT s.SCHEMANAME,
               s.OWNER,
               s.CREATE_TIME,
               (SELECT COUNT(*) FROM SYSCAT.TABLES t
                 WHERE t.TABSCHEMA = s.SCHEMANAME AND t.TYPE = 'T') AS TABELLEN,
               (SELECT COUNT(*) FROM SYSCAT.TABLES t
                 WHERE t.TABSCHEMA = s.SCHEMANAME AND t.TYPE = 'V') AS VIEWS
          FROM SYSCAT.SCHEMATA s
         WHERE s.SCHEMANAME NOT LIKE 'SYS%'
           AND s.SCHEMANAME NOT IN ('NULLID', 'SQLJ', 'DB2QP')
         ORDER BY TABELLEN DESC, s.SCHEMANAME
    """, conn)

schemata[schemata["TABELLEN"] > 0]



# ============================================================================
# ZELLE 2 - Anwendung
# ============================================================================

# --- Verbindung (DEV) ---
os.environ.setdefault("DB2_HOSTNAME", "SDTSDAHF1")
os.environ.setdefault("DB2_DATABASE", "DTSDAHF1")
os.environ.setdefault("DB2_PORT", "55000")
# DB2_USER und DB2_PASSWORD kommen aus den Domino Environment Variables

SCHEMA = "MEIN_SCHEMA"                  # siehe CURRENT SCHEMA unten
TABLE = "TA_TAGESINKASSO_VERSAND"

# --- Daten aufbereiten ---
df = add_versanddatum(df)               # gleichverteilt 15.10.2027 - 26.11.2027

# --- Schema des technischen Users ermitteln ---
with db2_connect() as conn:
    cur = conn.cursor()
    cur.execute("VALUES CURRENT SCHEMA")
    print("CURRENT SCHEMA:", cur.fetchone()[0].strip())
    cur.close()

# --- DDL erzeugen und ANSEHEN (führt noch nichts aus) ---
ddl = suggest_ddl(
    df,
    table=TABLE,
    schema=SCHEMA,
    not_null=["versanddatum"],   # anpassen
    primary_key=[],              # z. B. ["kundennr"]
    overrides={},                # z. B. {"beitrag": "DECIMAL(15,2)"}
    add_audit_columns=True,      # ergänzt LADE_TS
)
print(ddl)

# --- Tabelle anlegen, dann schreiben ---
with db2_connect() as conn:
    if not table_exists(conn, SCHEMA, TABLE):
        execute_ddl(ddl, conn=conn)

    write_dataframe(df, table=TABLE, schema=SCHEMA, conn=conn, dry_run=True)  # Probe
    n = write_dataframe(df, table=TABLE, schema=SCHEMA, conn=conn)            # echt

print("Geschrieben:", n)

with db2_connect() as conn:
    display(pd.read_sql('SELECT COUNT(*) AS N FROM "%s"."%s"' % (SCHEMA, TABLE), conn))
