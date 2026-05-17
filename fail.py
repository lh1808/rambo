11:58:07 INFO [rubin.dataprep] Deduplizierung auf 'PARTNER_ID_V': 472873 → 472860 Zeilen (13 Duplikate entfernt).
11:58:07 INFO [rubin.dataprep] Datei 0: 64638 Zeilen, Treatment-Rate: 61.3%
11:58:07 INFO [rubin.dataprep] Datei 1: 78243 Zeilen, Treatment-Rate: 61.7%
11:58:07 INFO [rubin.dataprep] Datei 2: 149993 Zeilen, Treatment-Rate: 66.7%
11:58:07 INFO [rubin.dataprep] Datei 3: 179986 Zeilen, Treatment-Rate: 50.0%
11:58:07 WARNING [rubin.dataprep] Treatment-Imbalance erkannt! Treatment-Raten pro Datei: ['61.3%', '61.7%', '66.7%', '50.0%'] (Differenz: 16.7 Prozentpunkte). Dies kann zu verzerrten Uplift-Metriken führen.
11:58:34 INFO [rubin.dataprep] Treatment-Balance Downsampling: 472860 → 389983 Zeilen (18% entfernt). Ziel-Treatment-Rate: 50.0%.
