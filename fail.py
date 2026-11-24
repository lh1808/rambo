1.3.1 Ebene 1 — Modellerstellung (randomisierte Experimente)
Die kausalen Modelle wurden auf Basis randomisierter Experimente (Treatment-/Kontroll-Split) erstellt, um den Uplift bestmöglich vorherzusagen. Eine gute Kalibrierung ist hier entscheidend, da die spätere Nutzenmessung auf der Verlässlichkeit des CATE-Rankings beruht. Die Validierung erfolgt über DRTester, Qini-Kurven und Calibration-Plots im Rahmen der Modellentwicklung.


Abb. 4: Beispielhafte out-of-Sample Kalibrierung eines Uplift Modells

Wie in Abb. 4 aufgezeigt wird, ist vor allem eine gute Kalibrierung entscheidend, um Mehrwerte rein modellbasiert valide messen zu können. Bei der vorliegenden Kalibrierung wird out-of-sample gruppierte durchschnittliche CATEs, also kundenindividuelle Uplift-Vorhersagen mit ihren tatsächlichen gruppierten ATEs gegenübergestellt. Wenn dies gut funktioniert, ist eine Mehrwertmessung basierend auf diesem Modell valide.

Der Grund dafür ist, dass die modellbasierte Mehrwertmessung den Mehrwert einer Maßnahme als Summe der vorhergesagten CATEs der angesprochenen Kunden berechnet. Diese Summe entspricht nur dann dem tatsächlich erzielten inkrementellen Effekt, wenn die Vorhersagen nicht nur die richtige Reihenfolge der Kunden, sondern auch das richtige Niveau des Effekts treffen. Genau das prüft die Kalibrierung: Die tatsächlichen Gruppen-ATEs stammen aus dem randomisierten Experiment und sind daher unverzerrte Schätzungen des wahren kausalen Effekts. Stimmen die durchschnittlichen Vorhersagen je Gruppe out-of-sample mit diesen überein, über- oder unterschätzt das Modell den Effekt in keinem Bereich des Scores systematisch. Die aufsummierten Vorhersagen für eine anhand des Scores ausgewählte Kundengruppe sind dann ein erwartungstreuer Schätzer ihres tatsächlichen Uplifts, auch ohne eigene Kontrollgruppe im Betrieb.

Ein gutes Ranking allein (z. B. eine gute Qini-Kurve) würde dafür nicht ausreichen: Ein Modell kann Kunden korrekt sortieren und den Effekt dennoch durchgängig zu hoch oder zu niedrig ausweisen, was den gemessenen Mehrwert direkt verzerren würde.

1.3.2 Ebene 2 — Nutzenmessung (teilweise Simulation)
Initial wird für jedes Kausal-Modell, welches in Produktion überführt wird, ein Test gegen das vorherig im Einsatz befindliche Affinitätsmodell gemacht. Dabei wird das Potenzial für eine Kampagne zufällig zwischen den beiden Verfahren verteilt und dann mittels des jeweiligen Modells innerhalb dieser Potenziale sortiert. Zusätzlich werden jeweils Kontrollgruppen mitgezogen, um die jeweiligen Uplifts für die beiden Vorgehen berechnen zu können. Sofern sich die vorher angenommenen Uplift-Vorteile durch die Umstellung auf Kausalscore bestätigen wird in den folgenden Aktionen vollständig auf die Kausalsortierung umgestellt.

Die Nutzen- bzw. Mehrwerte je Kampagne werden anschließend rein durch Simulation bestimmt: Vergleich der Auswahl nach Kausal-Score gegen die frühere Auswahl nach Affinitäts-Score, bewertet mit demselben Modell (Abschnitt 1.2.4 c). Ein laufender randomisierter Holdout pro Kampagne wird weiterhin permanent mitgezogen. Dieser dient dazu, den Uplift der Aktionen hinsichtlich Abschlüsse auf tatsächlich vorliegenden Daten zu bestimmen. Er kann jedoch nicht dafür verwendet werden, den Mehrwert des Kausalscore-Einsatzes zu bestimmen, da dieser sich aus dem Delta zum Einsatz eines Affinitätsscore ergibt. Da in Produktion aber nicht mehr, wie oben beschrieben, auch nach Affinitätsscore ausgesendet wird, muss dieses Delta simulationsbasiert bestimmt werden. Die Erkenntnisse aus dem tatsächlich gemessenen Uplift können jedoch verwendet werden, um ggf. das Nötigwerden eines Re-Trainings des Modells zu bestimmen. Dies geschieht durch den Vergleich des tatsächlich gemessenen Uplifts (observed) mit dem vorhergesagten Uplift (predicted). Wenn dies zunehmen auseinanderdriftet, ist ein Re-Training notwendig.

Das Kampagnencontrolling der MA21 ist ein Standardprozess und stellt die korrekte Zuordnung der einzelnen Kampagnen sicher. 

1.3.3 Aktuelle vs. historische Daten
Das Modell τ̂ stützt sich auf historische randomisierte Experimentdaten. Die Nutzenmessung selbst nutzt den aktuellen Kundenbestand-Snapshot der jeweiligen Kampagne. Ein Re-Training des Modells auf neueren Experimentdaten erfolgt anlassbezogen; der jeweils verwendete Modellstand wird in der Einzelmessung dokumentiert.



Hi Luka, Eva und ich haben das DW-Konzept nochmal gelesen. Aufbereitung und Co. alles fein. Ein Wunsch wäre noch: Beim Erkennen des Drifts und der Rekalibrierung neigt das bisherige Vorgehen u.E. zur Überschätzung. Da gäbe es diese Heilungsmöglichkeiten:
Pauschaler Abschlag
Ergänzen eines Retraining-Vorgehens inkl. Drift-Grenzen
Ergänzung, dass diese Überschätzung methodisch bekannt ist und bei einer Weiterführung der Diskussion über die Deckungsbeiträge aufgegriffen werden kann (wenn im Deckungsbeitrag bereits Puffer sind muss das nicht 2x drin sein, wenn hier der Kundenwert reinkäme wäre das anders). 
Verständlich?
