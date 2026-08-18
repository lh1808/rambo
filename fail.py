---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[78], line 2301
   2297 from reportlab.platypus import PageBreak, Paragraph, Spacer, Table, TableStyle
   2299 import matplotlib.pyplot as plt
-> 2301 from kundenwert_analyse import (ID, VERTRAG, BEITRAG, PROFIT, BEREICH, BEREICH_GROB,
   2302                                 ZWEIG, ZWEIG_VOLL, KFZ_REST_PRODUKTE, bereinige,
   2303                                 ergaenze_ebenen, vertragsaggregat)
   2304 from kundenwert_report import (C_LINIE, C_PRIMAER, MPL_AKZENT, MPL_GRAU, MPL_NEGATIV,
   2305                                MPL_POSITIV, MPL_PRIMAER, _Doc, _bild, _eckwerte, _eur,
   2306                                _eur_kurz, _fmt, _hinweis, _pct, _styles, _tabelle,
   2307                                _zwei_bilder)
   2309 # Reihenfolge der Wechselgruppen - überall gleich verwendet

ModuleNotFoundError: No module named 'kundenwert_analyse'
