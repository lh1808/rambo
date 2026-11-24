1.3.2 Ebene 2 — Nutzenmessung (teilweise Simulation)

Initial wird für jedes Kausal-Modell, das in Produktion überführt wird, ein Test gegen das zuvor eingesetzte Affinitätsmodell durchgeführt. Dabei wird das Potenzial einer Kampagne zufällig zwischen den beiden Verfahren aufgeteilt und innerhalb dieser Potenziale mit dem jeweiligen Modell sortiert. Zusätzlich werden für beide Verfahren Kontrollgruppen mitgezogen, damit sich die jeweiligen Uplifts berechnen lassen. Bestätigen sich die angenommenen Uplift-Vorteile des Kausal-Scores, wird in den folgenden Aktionen vollständig auf die Kausal-Sortierung umgestellt.

Die Nutzen- bzw. Mehrwerte je Kampagne werden danach rein durch Simulation bestimmt. Dabei wird die Auswahl nach Kausal-Score mit der früheren Auswahl nach Affinitäts-Score verglichen. Beide Auswahlen bewertet dasselbe Bewertungsmodell (Abschnitt 1.2.4 c). Dieses Bewertungsmodell ist bewusst ein anderes als das Modell, das die Scores für die Kundenselektion erzeugt (siehe Abschnitt 1.3.3).

Pro Kampagne wird zusätzlich dauerhaft ein randomisierter Holdout mitgezogen. Mit ihm wird der Uplift der Aktionen auf die Abschlüsse anhand tatsächlich vorliegender Daten bestimmt. Den Mehrwert des Kausal-Scores kann er jedoch nicht messen, denn dieser ergibt sich aus dem Delta zum Einsatz eines Affinitätsscores. Da in Produktion nicht mehr zusätzlich nach Affinitätsscore ausgesteuert wird, muss dieses Delta per Simulation bestimmt werden. Der tatsächlich gemessene Uplift (observed) wird stattdessen mit dem vorhergesagten Uplift (predicted) verglichen. So lassen sich Drift erkennen und der Bedarf für ein Re-Training ableiten (Vorgehen und Grenzen siehe Abschnitt 1.3.3).

Das Kampagnencontrolling der MA21 ist ein Standardprozess und stellt die korrekte Zuordnung der einzelnen Kampagnen sicher.

1.3.3 Umgang mit methodisch bedingter Überschätzung

Wir wissen, dass die modellbasierte Mehrwertmessung eher zur Überschätzung als zur Unterschätzung neigt. Dafür gibt es zwei Ursachen:

Selektionseffekt („Winner's Curse“): Kunden werden ausgewählt, weil ihr vorhergesagter Uplift hoch ist. Jede Vorhersage enthält neben dem wahren Effekt auch einen Schätzfehler. Wer die höchsten Werte auswählt, erwischt daher bevorzugt Kunden, deren Effekt das Modell überschätzt. Bewertet dasselbe Modell anschließend diese Auswahl, fließt der Fehler direkt in den ausgewiesenen Mehrwert ein. Die Out-of-sample-Kalibrierung aus Abschnitt 1.3.1 begrenzt diesen Effekt. Ausschließen kann sie ihn aber nur so fein, wie die Gruppen geschnitten sind, und nur innerhalb ihrer statistischen Unsicherheit.
Zeitlicher Drift: Das Modell beruht auf historischen Experimentdaten. Effekte von Direktwerbung können mit der Zeit nachlassen, etwa durch Gewöhnung, eine veränderte Bestandsstruktur oder ein bereits ausgeschöpftes Potenzial. Bis zum nächsten Re-Training gehen dann zu hohe Effekte in die Bewertung ein.

Primäre Maßnahme: Trennung von Selektions- und Bewertungsmodell

Hauptsächlich fangen wir die Überschätzung ab, indem die Nutzenmessung nicht mit dem Modell erfolgt, das die Scores für die Selektion erzeugt. Ein eigenständiges Bewertungsmodell übernimmt diese Aufgabe. Seine Schätzfehler hängen weitgehend nicht mit denen des Selektionsmodells zusammen. Wählt das Selektionsmodell also Kunden mit überschätztem Uplift aus, sind dieselben Kunden im Bewertungsmodell nicht systematisch ebenfalls überschätzt. Der Selektionseffekt geht deshalb nicht direkt in den gemessenen Mehrwert ein.

Als Bewertungsmodell dient ein Causal Forest. Er eignet sich für die Nutzenmessung besonders, weil er mit Honest Trees arbeitet. Die Daten werden dabei geteilt: Ein Teil bestimmt die Baumstruktur, also die Splits. Der andere, davon unabhängige Teil schätzt die Effekte in den Blättern. Die Effektschätzungen beruhen also nicht auf denselben Beobachtungen, mit denen die Struktur gesucht wurde. Das verringert die typische Überanpassung an zufällig hohe Effekte deutlich. Außerdem liefert der Causal Forest unter diesem Vorgehen statistisch fundierte Konfidenzintervalle, mit denen sich die Unsicherheit der Mehrwertschätzung beziffern lässt.

Ganz beseitigt ist das Risiko damit nicht. Beide Modelle beruhen auf denselben historischen Experimentdaten. Gemeinsame systematische Abweichungen, vor allem zeitlicher Drift, kann die Modelltrennung deshalb nicht auffangen. Hier setzen die folgenden ergänzenden Maßnahmen an.

Ergänzende Maßnahmen (denkbar, zu evaluieren)

Retraining-Vorgehen mit Drift-Grenzen: Der laufende Holdout aus Abschnitt 1.3.2 wird innerhalb der kausal selektierten Population gezogen. Damit misst er genau dort, wo eine Überschätzung auftreten würde. Als Kennzahl bietet sich das Verhältnis aus gemessenem und vorhergesagtem Uplift (observed/predicted) an. Zu definieren sind eine Warngrenze und eine Eingriffsgrenze. Ein Beispiel wäre, dass das Konfidenzintervall des gemessenen Uplifts den vorhergesagten Wert wiederholt nicht mehr enthält. Wird die Eingriffsgrenze überschritten, folgen ein Re-Training oder eine Rekalibrierung. Die Holdouts einzelner Kampagnen sind klein und ihre Messungen entsprechend unsicher. Deshalb sollte die Kennzahl über mehrere Kampagnen oder rollierend ausgewertet werden. Konkrete Schwellenwerte werden im Rahmen der Evaluation festgelegt.
Pauschaler Abschlag: Denkbar ist ein Abschlag auf den modellbasiert ermittelten Mehrwert. Ableiten ließe er sich zum Beispiel aus dem empirischen observed/predicted-Verhältnis der Holdouts oder aus der unteren Grenze des Konfidenzintervalls des Causal Forest. Ein fest gesetzter Abschlag ohne empirische Grundlage wäre allerdings willkürlich. Ob und in welcher Höhe ein Abschlag sinnvoll ist, muss daher evaluiert werden.
Abstimmung mit der Deckungsbeitragsbewertung: Die beschriebene Tendenz zur Überschätzung ist methodisch bekannt und kann aufgegriffen werden, wenn die Diskussion über die Deckungsbeiträge weitergeführt wird. Enthalten die Deckungsbeiträge bereits Sicherheitspuffer, sollte kein zusätzlicher Abschlag auf Modellebene erfolgen, damit die Unsicherheit nicht doppelt berücksichtigt wird. Wird der Mehrwert dagegen über den Kundenwert bewertet, der keinen solchen Puffer enthält, ist das neu zu beurteilen.

1.3.4 Aktuelle vs. historische Daten

Das Modell τ̂ stützt sich auf historische randomisierte Experimentdaten. Die Nutzenmessung selbst verwendet den aktuellen Kundenbestand-Snapshot der jeweiligen Kampagne. Ein Re-Training auf neueren Experimentdaten erfolgt anlassbezogen, insbesondere wenn die in Abschnitt 1.3.3 beschriebenen Drift-Grenzen überschritten werden. Der jeweils verwendete Modellstand wird in der Einzelmessung dokumentiert.

Was sich geändert hat:

„Dasselbe Modell“ präzisiert: In 1.3.2 stand „bewertet mit demselben Modell“. Gemeint ist jetzt ausdrücklich „dasselbe Bewertungsmodell für beide Auswahlen“. Sonst hätte man das so lesen können, als würde das Selektionsmodell sich selbst bewerten, was dem neuen Abschnitt widersprechen würde.
Holdout als Argument genutzt: Der Holdout liegt innerhalb der kausal selektierten Population und misst damit genau die Gruppe, in der eine Überschätzung entstehen würde. Das stützt das Drift-Monitoring gut.
Grenzen offen benannt: Der Text sagt ausdrücklich, dass die Modelltrennung den Drift nicht abfängt. Das macht die Argumentation gegenüber Eva und deinem Kollegen glaubwürdiger, als wenn sie das Problem vollständig gelöst nennen würde.
Kleinigkeiten: Ich habe nebenbei „zunehmen“ zu „zunehmend“ korrigiert (der Satz ist jetzt umformuliert) und ein Komma ergänzt.
