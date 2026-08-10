Ja, guter Punkt — das Clipping ist da wirklich die bequeme Lösung: Echte Überlappungsprobleme würden damit einfach übertüncht statt gemeldet.

Relevant wird das aber erst bei Beobachtungsdaten mit geschätzter Treatment-Wahrscheinlichkeit — dann müsste man Fälle nahe 0 oder 1 bewusst behandeln (trimmen oder Overlap-Gewichte) statt still zu clippen.

Aktuell betrifft es uns nicht: Wir sind im RCT und geben die bekannte, konstante Wahrscheinlichkeit (~1/3 bzw. 1/2) direkt rein statt sie zu schätzen. Damit ist alles weit weg von der 0.01-Grenze — der Clip greift bei uns nie.
