    erg = analysiere_kundenwert(df)
    erstelle_pdf(erg, "Kundenwertanalyse.pdf",
                 titel="Kundenwertanalyse Bestand",
                 untertitel="Segmentierung nach aggregierter Profitabilität",
                 quelle="Bestandsdaten Stichtag TT.MM.JJJJ")
