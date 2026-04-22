Der Termineingang für Erstellung der Prognose in PLUTO soll in folgender Granularität übertragen werden:

Die Prognose soll für die beiden Kennzahlen "Termineingang Schriftstücke" (TERM_EINGANG_SCHRIFTST) und "Termineingang Sonstige Termine" (TERM_EINGANG_SONST) ermittelt werden. 

Die beiden Kennzahlen werden nach folgenden Produkten unterschieden:

KFZ_Vollkasko
KFZ_Teilkasko
KFZ_Haftpflicht
KFZ_Rest
HUS_Haftpflicht
HUS_Wohngebäude
HUS_Hausrat
HUS_Rest
Für den Termineingang pro Produkt wird zusätzlich die Information zum Schadenstatus "Neuschaden" und "Folgebearbeitung" benötigt. 

 

SQL --> Connection:



SQL --> Abfrage:

SELECT DIM_ZEIT, DIM_KENNZAHL,DIM_PRODUKT, DIM_SCHADENSTATUS, KENNZAHLWERT FROM t7.TA_DA_PLUTO_SP_2025
WHERE DIM_KENNZAHL IN ('TERM_EINGANG_SCHRIFTST', 'TERM_EINGANG_SONST')

Aufbau Schreibtabelle TA_DA_PLUTO_SP_2025_PROGNOSE:

DIM_ZEIT als Integer (Datum)
DIM_KENNZAHL = 'TERM_EINGANG_SCHRIFTST' und 'TERM_EINGANG_SONST'
DIM_PRODUKT als VARCHAR(50)
DIM_SCHADENSTATUS als VARCHAR(50)
KENNZAHLWERT als DECIMAL(20,9)
LOAD_DATE als Date (Ladedatum)
