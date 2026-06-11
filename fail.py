Hi Luka, bitte nimm nochmal einen neuen Abzug in deine Prognoseengine vor. Beim Test ist aufgefallen, dass es nicht für jeden Tag ein Prognosewert gibt. Ist das nicht möglich?

Passe bitte auch das Abzugs-SQL an, so dass auch die DIM_ORGA als Ausprägung mit einfließt. Geht das? Ich würde dann die Schreibtabelle entsprechend erweitern.

SQL alt:

SELECT DIM_ZEIT, DIM_KENNZAHL,DIM_PRODUKT, DIM_SCHADENSTATUS, KENNZAHLWERT FROM t7.TA_DA_PLUTO_SP_2025
WHERE DIM_KENNZAHL IN ('TERM_EINGANG_SCHRIFTST', 'TERM_EINGANG_SONST')

SQL neu:

SELECT DIM_ZEIT, DIM_KENNZAHL,DIM_PRODUKT, DIM_SCHADENSTATUS, KENNZAHLWERT, DIM_ORGA FROM t7.TA_DA_PLUTO_SP_2025
WHERE DIM_KENNZAHL IN ('TERM_EINGANG_SCHRIFTST', 'TERM_EINGANG_SONST')

Bei Fragen gerne melden.
