1.3.1 Ebene 1 — Modellerstellung (randomisierte Experimente)

Die kausalen Modelle wurden auf Basis randomisierter Experimente (Treatment-/Kontroll-Split) erstellt, um den Uplift bestmöglich vorherzusagen. Eine gute Kalibrierung ist hier entscheidend, da die spätere Nutzenmessung auf der Verlässlichkeit des CATE-Rankings beruht. Die Validierung erfolgt über DRTester, Qini-Kurven und Calibration-Plots im Rahmen der Modellentwicklung.

Wie in Abb. 4 aufgezeigt wird, ist vor allem eine gute Kalibrierung entscheidend, um Mehrwerte rein modellbasiert valide messen zu können. Bei der vorliegenden Kalibrierung wird out-of-sample gruppierte durchschnittliche CATEs, also kundenindividuelle Uplift-Vorhersagen mit ihren tatsächlichen gruppierten ATEs gegenübergestellt. Wenn dies gut funktioniert, ist eine Mehrwertmessung basierend auf diesem Modell valide.
