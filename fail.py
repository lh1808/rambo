Der tägliche Lauf arbeitet in drei Schritten:

1. Geänderte Verträge ermitteln
(DWHDeltaPullChunked.__init__, SQL delta_hus_incremental_pure.sql)
Hier werden alle Verträge geholt, die sich seit dem letzten Lauf geändert haben. Die erloschenen Verträge sind hier noch korrekt dabei.

    SELECT DISTINCT VERTRAG_ID
    FROM VW_MF_AK_DA_DELTA_HUS
    WHERE operating_time >= TIMESTAMP('{{ last_delta_date }}')

2. Nur aktive Verträge behalten
(pull_delta_2(), SQL delta_hus_inc_blank.sql)
Zu den geänderten Verträgen wird die Vertragsakte (VERTRAGSAKTE_ID) nachgeschlagen. Dabei werden aber nur Verträge übernommen, die zum Stichtag aktiv und im Scope sind. Das Ergebnis landet in der Zwischentabelle SESSION.VERTRAG_DELTA_HUS.

    WHERE v.VERTRAG_ID IN ({{ delta_ids }})
      AND v.effective_from_date <= TIMESTAMP('{{ stichtag }}')  -- Vertrag hat begonnen
      AND v.effective_to_date   >  TIMESTAMP('{{ stichtag }}')  -- Vertrag läuft noch
      AND v.valid_from_date     <= TIMESTAMP('{{ stichtag }}')
      AND v.valid_to_date       >  TIMESTAMP('{{ stichtag }}')
      AND v.anz_vertraege       >  0
      AND v.dim_gesellschaft_id IN ('23', '24', '133', '134')

Ein erloschener Vertrag scheitert hier mindestens an effective_to_date oder anz_vertraege und fällt raus.

3. Löschen
(remove_changed_contracts())
Gelöscht wird nur, was in der Zwischentabelle aus Schritt 2 steht:

    SELECT VERTRAGSAKTE_ID FROM SESSION.VERTRAG_DELTA_HUS
    DELETE FROM <KW-Tabelle> WHERE VE_ID IN (<IDs aus Schritt 2>)

Die erloschenen Verträge gehen also in Schritt 2 verloren. Deshalb werden sie in Schritt 3 weder gelöscht noch neu berechnet und behalten ihre alten Werte. Dasselbe gilt für Verträge, die aus anderen Gründen aus dem Scope fallen (z. B. anz_vertraege = 0 oder Wechsel der Gesellschaft).

Lösung:
Wir trennen das Löschen vom Neuberechnen.
- Löschen: Neue SQL-Datei (z. B. delta_hus_inc_delete.sql), die zu allen geänderten Verträgen aus Schritt 1 die Vertragsakte nachschlägt, ohne Stichtag- und Aktiv-Filter:

    SELECT DISTINCT v.VERTRAGSAKTE_ID
    FROM {{ dwh_schema }}.VW_MF_AK_HUS_VERTRAG as v
    WHERE v.VERTRAG_ID IN ({{ delta_ids }})

  remove_changed_contracts() nimmt die IDs zum Löschen dann aus dieser Abfrage statt aus SESSION.VERTRAG_DELTA_HUS. Damit werden alle geänderten Verträge gelöscht, ob aktiv oder nicht.
- Neuberechnen: delta_hus_inc_blank.sql bleibt unverändert. Wie bisher werden nur die noch aktiven Verträge aus Schritt 2 neu berechnet und wieder eingespielt.

Für KFZ wird die Lösch-Abfrage analog angelegt.
