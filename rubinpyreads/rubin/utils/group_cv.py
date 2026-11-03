"""Gruppen-bewusste Cross-Validation (Duplikat-/Cluster-Leakage-Prävention).

Hintergrund: Enthält der Datensatz mehrere Zeilen derselben Einheit (z. B.
2–3 Leads pro Partner), landen bei zeilenweisem StratifiedKFold nahezu
identische Zwillinge in Train UND Validation. Modelle mit hoher Kapazität
erkennen den Zwilling wieder — die OOF-Metriken (Qini, R-Score) werden
systematisch optimistisch, und das Tuning selektiert Memorisierung statt
Generalisierung. Gruppen-Folds (StratifiedGroupKFold) halten alle Zeilen
einer Gruppe im selben Fold und schließen diesen Kanal.

``GroupAwareStratifiedKFold`` ist der Splitter für EconMLs internes
Cross-Fitting: EconML 0.17 wirft bei ``fit(..., groups=...)`` einen
TypeError, wenn der konfigurierte ``cv`` eine (Stratified)KFold-INSTANZ ist
("must initialize this object with a splitter that can handle groups").
Diese Klasse ist bewusst KEINE Subklasse von StratifiedKFold: Ohne groups
delegiert sie 1:1 an StratifiedKFold (identische Splits bei gleichem Seed —
voll rückwärtskompatibel), mit groups an StratifiedGroupKFold.
"""

from __future__ import annotations

import logging

import numpy as np

_logger = logging.getLogger("rubin.group_cv")


def n_unique_groups(groups) -> int:
    """Anzahl eindeutiger Gruppen (NaN-sicher, ohne pandas-Pflicht)."""
    arr = np.asarray(groups)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return len(np.unique(arr.astype(str)))


def effective_group_splits(n_splits: int, groups) -> int:
    """Begrenzt die Fold-Zahl auf die Anzahl eindeutiger Gruppen.

    StratifiedGroupKFold/GroupKFold verlangen n_splits <= n_groups.
    Rückgabe < 2 heißt: Gruppen-CV ist nicht möglich (alles eine Gruppe).
    """
    return min(int(n_splits), n_unique_groups(groups))


class GroupAwareStratifiedKFold:
    """StratifiedKFold-Ersatz, der ``groups`` respektiert, wenn sie übergeben werden.

    - ``split(X, y)`` ohne groups → exakt StratifiedKFold(n_splits, shuffle,
      random_state): identische Folds wie bisher (Seed-deterministisch).
    - ``split(X, y, groups)`` → StratifiedGroupKFold: keine Gruppe liegt in
      Train UND Validation. Die Fold-Zahl wird defensiv auf die Anzahl
      eindeutiger Gruppen gekappt (mit Warnung).

    Absichtlich keine Subklasse von sklearn-KFold-Klassen: EconML lehnt
    (Stratified)KFold-Instanzen in Kombination mit ``fit(groups=...)`` per
    TypeError ab — genau diese Prüfung läuft über isinstance.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state=None):
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if groups is not None:
            eff = effective_group_splits(self.n_splits, groups)
            if eff >= 2:
                return eff
        return self.n_splits

    def split(self, X, y=None, groups=None):
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

        if groups is None:
            cv = StratifiedKFold(
                n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
            )
            return cv.split(X, y)

        eff = effective_group_splits(self.n_splits, groups)
        if eff < 2:
            _logger.warning(
                "GroupAwareStratifiedKFold: nur %d eindeutige Gruppe(n) — Gruppen-CV "
                "unmöglich, Fallback auf zeilenweise StratifiedKFold.",
                n_unique_groups(groups),
            )
            cv = StratifiedKFold(
                n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
            )
            return cv.split(X, y)
        if eff < self.n_splits:
            _logger.warning(
                "GroupAwareStratifiedKFold: n_splits=%d > %d Gruppen — auf %d Folds gekappt.",
                self.n_splits, n_unique_groups(groups), eff,
            )
        cv = StratifiedGroupKFold(
            n_splits=eff, shuffle=self.shuffle, random_state=self.random_state
        )
        return cv.split(X, y, groups=np.asarray(groups))

    def __repr__(self) -> str:  # pragma: no cover - nur Logging-Kosmetik
        return (
            f"GroupAwareStratifiedKFold(n_splits={self.n_splits}, "
            f"shuffle={self.shuffle}, random_state={self.random_state})"
        )
