Da in früheren Dokumenten teilweise die Begriffe Nutzen und Mehrwert uneinheitlich verwendet wurden anbei zur Erinnerung die gültige Definition:

Nutzen: Brutto, vor HF-Kosten. Berücksichtigt werden sollen unmittelbar dem Use Case zuzuordnenden Kosten (beispielsweise Beauftragungsgebühren für Dienstleiter oder Versandkosten bei Anpassungen im Briefversand)

Mehrwert: Brutto-Nutzen abzüglich der dem Handlungsfeld zuzuordnenden Kosten.



Seitens DA erwarten wir für Nutzenmesskonzepte folgende Struktur:

Nutzenkonzept
Verweis auf die Referenzimplementierung
Dokumentation der einzelnen Messdurchführungen / Instanziierungen inkl. Sonderfällen/Auffälligkeiten
Im Folgenden werden diese genauer erläutert.

1) Nutzenmesskonzept
Use Case-Definition

Beschreibung der Wirkhebel und Ziele des Use Cases
Kurze Beschreibung des Umfeldes (in welchem System, welche Nutzer, welche Veränderungszyklen)


Definition der zu messenden Größen (in Anlehnung an den vorherig beschriebenen Wirkhebel)

Gibt es bereits ein geeignetes Messkonzept, was wiederverwendet / adaptiert werden kann? Falls nicht, neues Messkonzept erstellen
Nutzenhebel definiert mit quantifizierbaren Metriken. Der Werthebel wird als Formel ausgedrückt mit einer definierten Baseline, gegen die gemessen wird. Dabei können auch mehrere Baselines möglich sein
Überschneidung mit anderen Use Cases / Nutzen markieren und erfassen, insb. im Hinblick auf Doppelzählung der Nutzen
Definition der quantitativen Metriken
Die Erfolgs-Metriken (KPIs) mit Definition
Die definierten Gruppen (Wo messe ich? An wem messe ich?)
Wie wird gemessen? (Experiment, Vorher/nachher, direkt zuordenbar, Simulation, etc.)
Welche Metriken müssen als konstant angesehen werden (zumindest innerhalb der Messung)
Markierung und Begründung für nicht messbare Parameter. Annahmen, die hier einfließen nennen und begründen (inkl. Vor- und Nachteile)
Welcher Effekt wird gemessen und weshalb und wie ist dieser konkret der Aktivitäten im DA-Handlungsfeld zuordbar?
Decken die Metriken alle relevanten Nutzeneffekte aus der DA-Aktivität ab?
Wie ergibt sich der ökonomische Nutzen durch DA inkl. der Berechnung?
Ist die Messung konsistent möglich, d.h. liefert ähnliche Ergebnisse bei mehrfacherer Messung oder sind zeitabhängige Effekte nicht separierbar enthalten?
Ersten Messtermin terminieren und dokumentieren


Experimentaufbau für die Datensammlung

Beschrieben werden sollen in diesem Abschnitt der Experimentaufbau zur Datensammlung inkl. Randomisierungskonzept sowie die entstehenden Daten, Anforderungen an deren Samplegröße (für die Randomisierung) sowie der Einbezug von aktuell gesammelten Daten vs. historischen Experimentdaten in die Mehrwertmessung



Datenauswertung und Entscheidungen

In diesem Abschnitt soll beschrieben werden, wie die gesammelten Daten ausgewertet werden und wie der endgültige Messwert bestimmt wird und . Relevant sind zusätzlich auch ggf. in diesem Schritt bei Mehrfachmöglichkeiten oder unterschiedlichen Ermittlungswegen zu treffende Entscheidungen und wie diese zu treffen sind (Beispiel: Mehrere Samples verfügbar → werden diese gemittelt oder nur 1 Sample aufgrund X genutzt).



2) Verweis auf die Referenzimplementierung als Code in Gitlab oder TFS
Das Messkonzept wird als Code so umgesetzt, dass ein nicht im Handlungsfeld beteiligter Mitarbeiter in DA01 Messkonzept und Implementierung vergleichen kann. Der Anspruch ist dabei nicht, dass dieser Code nie wieder geändert werden kann. Es ist zu erwarten, dass dort regelmäßig Anpassungen aufgrund Weiterentwicklungen bei Daten, Modellen oder Samples durchzuführen sind. Die Referenzimplementierung muss aber abnehmbar und versioniert sein.



3) Durchführung der Einzelmessungen
Die Einzelmessungen werden dann auf Einzelseiten dokumentiert, die alle Messgrößen und Auswertungen umfassen. Diese stellen dann eine Instanziierung einer Messung dar. Alternativ ist das z.B. auch in Excel machbar, wenn das aufgrund der Interaktion mit anderen Gruppen besser ist.
