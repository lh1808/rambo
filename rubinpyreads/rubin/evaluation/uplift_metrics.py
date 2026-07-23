from __future__ import annotations

"""Uplift-Metriken für Binary Treatment / Binary Outcome und Multi Treatment / Binary Outcome.
Diese Metriken beantworten eine andere Frage als klassische ML-Metriken:
- Klassifikation (ROC-AUC, LogLoss) bewertet: "Kann ich Y vorhersagen?"
- Uplift/Causal bewertet: "Kann ich entscheiden, *wen* ich behandeln sollte, um inkrementellen Nutzen zu erzeugen?"
Die zentrale Idee:
- Sortiere Beobachtungen nach geschätztem Uplift (CATE) absteigend.
- Betrachte kumulativ, wie stark sich Outcome in Treatment vs Control unterscheidet.
- Daraus lassen sich Kurven und Kennzahlen wie Qini-Koeffizient und AUUC ableiten.
Voraussetzungen:
- binary outcome y in {0,1}
- binary treatment t in {0,1} ODER multi treatment t in {0, 1, ..., K-1}
Hinweis:
- Es gibt mehrere Definitionen in der Literatur. Die hier implementierten Varianten sind pragmatisch und
in Uplift-Modelling gängig."""

from dataclasses import dataclass
from typing import Optional
import numpy as np



@dataclass(frozen=True)
class UpliftCurve:
    """Kumulierte Größen entlang einer Sortierung nach score."""
    fraction: np.ndarray           # Anteil der Population (0..1)
    n_treat: np.ndarray            # kumulierte Anzahl T=1 (bzw. T=k bei MT)
    n_control: np.ndarray          # kumulierte Anzahl T=0
    y_treat: np.ndarray            # kumulierte Summe Y unter T=1 (bzw. T=k)
    y_control: np.ndarray          # kumulierte Summe Y unter T=0
    uplift: np.ndarray             # kumulierter inkrementeller Outcome (Treatment-Control, skaliert)


def _check_binary(a: np.ndarray, name: str) -> None:
    u = np.unique(a)
    if not set(u).issubset({0, 1}):
        raise ValueError(f"{name} muss binär (0/1) sein, gefunden: {u}")


def _check_discrete(a: np.ndarray, name: str) -> None:
    """Prüft, ob ein Array diskrete nicht-negative ganzzahlige Werte enthält."""
    u = np.unique(a)
    if not np.all(u == u.astype(int)):
        raise ValueError(f"{name} muss ganzzahlig sein, gefunden: {u}")
    if np.any(u < 0):
        raise ValueError(f"{name} darf keine negativen Werte enthalten, gefunden: {u}")


def tiebreak_jitter(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Skalen-relativer Jitter für deterministisches Tie-Breaking beim Sortieren.

    Ein absoluter Jitter würde bei Scores großer Magnitude von Float64
    absorbiert (Ties blieben reihenfolgeabhängig) und bei extrem kleinen
    Scores das Ranking dominieren.

    Präzisionsgrenze: Scores mit Abstand < Jitter-Amplitude (≈1e-9·max|score|)
    werden als effektiv gleich behandelt und deterministisch randomisiert
    geordnet. Gegenüber jitter-freien Referenzen (z. B. sklift) können
    Kurvenflächen dadurch um O(1/n²) abweichen — das Modell-Ranking bleibt
    davon unberührt. 1e-9 · max|score| liegt sicher über
    der Float64-Auflösung (~2.2e-16 relativ) und sicher unter jedem echten
    Score-Unterschied."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    scale = float(np.max(np.abs(scores))) if n > 0 else 0.0
    eps = (scale if scale > 0 else 1.0) * 1e-9
    return rng.uniform(-eps, eps, n)


def orient_historical_score(s, higher_is_better: bool = True) -> np.ndarray:
    """Orientiert einen historischen Score für die Bewertungs-/Plot-Kette.

    - higher_is_better=False → Vorzeichen invertieren (danach gilt überall:
      höher = besser).
    - Nicht-finite Werte (NaN/±inf) werden NACH der Orientierung auf
      (min_finite − 1) gesetzt — also ans ENDE der Rangliste. Die frühere
      Sanitization nan_to_num(0.0) VOR der Invertierung platzierte fehlende
      Scores bei lower-better-Skalen (z. B. Klassen 1..10 → −10..−1) fälschlich
      an die SPITZE der historischen Rangliste (0 > −1). Auch ±inf werden als
      Datenfehler konservativ ans Ende sortiert.
    Gibt immer eine Kopie zurück (mutiert den Input nicht).
    """
    s = np.asarray(s, dtype=float).copy()
    if not higher_is_better:
        s = -s
    finite = np.isfinite(s)
    if not finite.all():
        worst = float(s[finite].min()) - 1.0 if finite.any() else 0.0
        s[~finite] = worst
    return s


def uplift_curve_mt_argmax(
    y: np.ndarray,
    t: np.ndarray,
    scores_2d: np.ndarray,
) -> UpliftCurve:
    """Argmax-Uplift-Kurve für Multi-Treatment: EIN Ranking über alle Arme.

    Verallgemeinerung der binären Qini-Kurve auf K Arme für den Anwendungsfall
    "Personen-Priorisierung": Jede Person wird nach ihrem BESTEN geschätzten
    Effekt s(x) = max_k τ̂_k(x) gerankt (genau die Liste, die produktiv
    abgearbeitet wird), empfohlener Arm ist rec(x) = argmax_k. Gemessen wird
    der kumulative Uplift der Politik "gib jedem seinen empfohlenen Arm":

    - Evidenz-Subset: Controls (T=0) und "matched treated" (T == rec) —
      Beobachtungen mit T>0, aber T != rec tragen keine Information über die
      empfohlene Politik und werden ausgeschlossen (wie bei
      uplift_curve_mt_per_arm die Fremd-Arme).
    - Horvitz-Thompson-Gewichtung der matched treated mit w ∝ 1/P(T=rec),
      selbst-normalisiert (Mittel der Gewichte = 1): Ohne Gewichtung wären
      Empfehlungen in größere Arme im Treated-Mittel überrepräsentiert.
      P(T=k) sind die empirischen Arm-Priors der GESAMTEN Stichprobe
      (RCT-Annahme: Zuweisung unabhängig von X).

    Konsistenz-Eigenschaften (durch Konstruktion, siehe Tests):
    - K=2: rec ≡ 1, Subset = alle Zeilen, Gewichte ≡ 1 → exakt identisch
      zur binären ``uplift_curve`` (gleiches Tie-Breaking, Seed 42).
    - Gleichverteilte Arme: Gewichte ≡ 1.
    - Dominanter Arm (Modell empfiehlt überall Arm k): identisch zur
      One-vs-Control-Kurve ``uplift_curve_mt_per_arm(arm=k)``.

    Hinweis zur Skala: fraction bezieht sich auf das Evidenz-Subset
    (Controls + matched treated), analog zur per-Arm-Kurve. Bei K Armen mit
    Gleichverteilung sind das ≈ (2/K)·n + Controls der Zeilen; das
    Tuning-Signal ist entsprechend verrauschter als binär — dafür fließen
    ALLE Arme in ein gemeinsames Ranking ein.
    """
    y = np.asarray(y).astype(int)
    t = np.asarray(t).astype(int)
    scores_2d = np.asarray(scores_2d, dtype=float)
    _check_binary(y, "y")
    _check_discrete(t, "t")
    if scores_2d.ndim == 1:
        scores_2d = scores_2d.reshape(-1, 1)
    if scores_2d.ndim != 2 or scores_2d.shape[0] != len(y):
        raise ValueError(f"scores_2d muss Shape (n, K-1) haben, gefunden: {scores_2d.shape}")
    _max_arm = int(t.max())
    if _max_arm >= 1 and scores_2d.shape[1] < _max_arm:
        raise ValueError(
            f"scores_2d hat {scores_2d.shape[1]} Spalte(n), aber Treatment-Arme bis "
            f"T={_max_arm} beobachtet — erwartet mindestens {_max_arm} CATE-Spalten (Arm 1..K-1)."
        )

    # Bestes Ranking-Signal + empfohlener Arm pro Person
    s = scores_2d.max(axis=1)
    rec = scores_2d.argmax(axis=1) + 1  # Spalten 0..K-2 ↔ Arme 1..K-1

    # Empirische Arm-Priors der Gesamtstichprobe (RCT: unabhängig von X)
    arms, counts = np.unique(t, return_counts=True)
    priors = {int(a): c / len(t) for a, c in zip(arms, counts)}

    # Evidenz-Subset: Controls + matched treated
    mask = (t == 0) | (t == rec)
    y_sub = y[mask]
    t_sub = t[mask]
    s_sub = s[mask]
    rec_sub = rec[mask]
    n = len(y_sub)
    if n == 0:
        return UpliftCurve(
            fraction=np.array([]), n_treat=np.array([]), n_control=np.array([]),
            y_treat=np.array([]), y_control=np.array([]), uplift=np.array([]),
        )

    # HT-Gewichte der matched treated: w ∝ 1/P(T=rec), selbst-normalisiert.
    treated_mask = t_sub > 0
    w = np.ones(n, dtype=float)
    if treated_mask.any():
        inv_p = np.array([1.0 / max(priors.get(int(a), 0.0), 1e-12) for a in rec_sub[treated_mask]])
        if np.all(inv_p == inv_p[0]):
            # Uniforme Gewichte (K=2, dominanter Arm oder gleiche Priors):
            # exakt 1.0 statt c·(n/(n·c)) — vermeidet ±1-ULP-Rundungsrauschen
            # und garantiert BIT-exakte Äquivalenz zur binären uplift_curve.
            pass  # w bleibt 1.0
        else:
            w[treated_mask] = inv_p * (treated_mask.sum() / inv_p.sum())  # Mittel = 1

    # Sortierung wie in uplift_curve: absteigend nach Score mit skalen-
    # relativem Tie-Breaking-Jitter (Seed 42) für exakte Binär-Konsistenz.
    _rng = np.random.default_rng(42)
    order = np.argsort(-(s_sub + tiebreak_jitter(s_sub, _rng)))
    y_o = y_sub[order]
    w_o = w[order]
    treat_o = treated_mask[order].astype(float)
    ctrl_o = (~treated_mask[order]).astype(float)

    # Kumulative (gewichtete) Zähler — Struktur identisch zu uplift_curve;
    # bei w ≡ 1 numerisch exakt gleich.
    n_t = np.cumsum(w_o * treat_o)
    n_c = np.cumsum(ctrl_o)
    y_t = np.cumsum(y_o * w_o * treat_o)
    y_c = np.cumsum(y_o * ctrl_o)

    rate_t = y_t / np.maximum(n_t, 1.0)
    rate_c = y_c / np.maximum(n_c, 1.0)
    frac = np.arange(1, n + 1) / n
    uplift = (rate_t - rate_c) * frac

    return UpliftCurve(
        fraction=frac,
        n_treat=n_t,
        n_control=n_c,
        y_treat=y_t,
        y_control=y_c,
        uplift=uplift,
    )


def uplift_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> UpliftCurve:
    """Berechnet die Uplift-Kurve für Binary Treatment / Binary Outcome.
    Für Multi-Treatment siehe uplift_curve_mt_per_arm."""
    y = np.asarray(y).astype(int)
    t = np.asarray(t).astype(int)
    score = np.asarray(score).astype(float)
    _check_binary(y, "y")
    _check_binary(t, "t")

    n = len(y)
    # Tie-Breaking: Bei identischen Scores randomisiert argsort die Reihenfolge
    # nicht — sie hängt von der Datenreihenfolge ab. Skalen-relativer Jitter
    # (siehe tiebreak_jitter) verhindert artifizielle Treatment-Balance-
    # Schwankungen. Seed=42 für Reproduzierbarkeit.
    _rng = np.random.default_rng(42)
    order = np.argsort(-(score + tiebreak_jitter(score, _rng)))
    y = y[order]
    t = t[order]

    treat = (t == 1).astype(int)
    control = (t == 0).astype(int)

    n_t = np.cumsum(treat)
    n_c = np.cumsum(control)
    y_t = np.cumsum(y * treat)
    y_c = np.cumsum(y * control)

    # inkrementeller Outcome: (y_t / n_t) - (y_c / n_c), auf Gesamtpopulation skaliert
    rate_t = y_t / np.maximum(n_t, 1)
    rate_c = y_c / np.maximum(n_c, 1)
    inc_rate = rate_t - rate_c
    uplift = inc_rate * (np.arange(1, n + 1) / n)

    frac = (np.arange(1, n + 1) / n)

    return UpliftCurve(
        fraction=frac,
        n_treat=n_t,
        n_control=n_c,
        y_treat=y_t,
        y_control=y_c,
        uplift=uplift,
    )


def auuc(curve: UpliftCurve) -> float:
    """AUUC: Fläche zwischen der Uplift-Kurve und der Random-Baseline (Lift über Zufall).

    Konvention wie sklift.metrics.uplift_auc_score, pro Population normiert (die
    Uplift-Kurve ``curve.uplift`` ist bereits (rate_t − rate_c)·Anteil). Die
    Random-Baseline ist die Gerade vom Ursprung zum Endpunkt (= zufällige
    Behandlung). Ein Random-Modell liefert ≈ 0, gutes Ranking > 0.

    Hinweis: Da die Random-Fläche nur vom Endpunkt (≈ ATE) abhängt — der über alle
    Modelle auf demselben Eval-Set identisch ist — ändert die Baseline-Subtraktion
    das Modell-Ranking nicht gegenüber der rohen Fläche."""
    if len(curve.fraction) == 0:
        return 0.0
    # Integral inklusive Ursprung (0, 0): Die Kurvenpunkte beginnen bei k=1;
    # ohne den Ursprung fehlte das erste Trapez-Segment — ein O(1/n)-Rest,
    # der die exakte Übereinstimmung mit sklift.uplift_curve verfehlte.
    x = np.concatenate([[0.0], curve.fraction])
    y = np.concatenate([[0.0], curve.uplift])
    baseline = x * y[-1]
    return float(np.trapezoid(y - baseline, x))


def qini_coefficient(curve: UpliftCurve) -> float:
    """Qini-Koeffizient: Fläche zwischen der Qini-Kurve und der Random-Baseline.

    Die Qini-Kurve ist ``Q(k) = Y_t(k) − Y_c(k)·N_t(k)/N_c(k)`` (kumulative
    inkrementelle Outcome-Anzahl) — exakt die Kurve, die ``_native_qini_curve``
    zeichnet, und die Konvention von sklift.metrics.qini_auc_score. Das Ergebnis
    ist pro Population normiert (Division durch n), damit es datensatzgrößen-
    invariant ist und dem gezeichneten Kurvenverlauf entspricht.

    Random-Baseline: Gerade vom Ursprung zum Endpunkt der Qini-Kurve."""
    n = len(curve.fraction)
    if n == 0:
        return 0.0
    n_c_safe = np.maximum(curve.n_control, 1)
    y_qini = (curve.y_treat - curve.y_control * curve.n_treat / n_c_safe) / n
    # Integral inklusive Ursprung (0, 0) — exakt die sklift.qini_curve-Fläche
    # (siehe auuc; ohne Ursprung bliebe ein O(1/n)-Diskretisierungsrest).
    x = np.concatenate([[0.0], curve.fraction])
    y_qini = np.concatenate([[0.0], y_qini])
    baseline = x * y_qini[-1]
    return float(np.trapezoid(y_qini - baseline, x))


def uplift_at_k(curve: UpliftCurve, k_fraction: float = 0.1) -> float:
    """Inkrementelle Response-Rate (rate_treat − rate_control) über die Top-k% nach score.

    Entspricht sklift.metrics.uplift_at_k(strategy='overall') und dem ersten
    Balken des Uplift-by-Percentile-Plots: die Within-Group-Differenz der
    Response-Raten in der Top-k%-Teilmenge — NICHT mit dem Populationsanteil
    skaliert (das ist die Konvention des `uplift`-Kurvenfelds, das für AUUC/Qini
    gebraucht wird).

    Schnitt: ``n_size = int(k·n)`` (Floor) — identisch zu sklift
    (``int(n_samples * k)``) und zu den ``linspace``-Bin-Kanten von
    ``_native_uplift_by_percentile`` (``int(n / n_bins)``). Dadurch stimmt
    ``uplift_at_k(0.1)`` exakt mit dem ersten Perzentil-Balken überein und der
    Qini-Kurven-Punkt bei diesem Schnitt erfüllt ``Q = n_treat · (rate_t − rate_c)``,
    auch wenn die Eval-Set-Größe kein Vielfaches von ``n_bins`` ist."""
    k_fraction = float(k_fraction)
    if k_fraction <= 0:
        return 0.0
    n = len(curve.fraction)
    n_size = int(k_fraction * n)  # Floor — wie sklift und die Perzentil-Bin-Kanten
    if n_size < 1:
        # k·n < 1: Die Top-k%-Teilmenge ist leer → 0.0. Ein Clip auf das
        # Top-1-Element würde eine nicht angeforderte Auswertung liefern.
        return 0.0
    idx = int(np.clip(n_size - 1, 0, n - 1))
    n_t = curve.n_treat[idx]
    n_c = curve.n_control[idx]
    rate_t = curve.y_treat[idx] / n_t if n_t > 0 else 0.0
    rate_c = curve.y_control[idx] / n_c if n_c > 0 else 0.0
    return float(rate_t - rate_c)


def policy_value(y: np.ndarray, t: np.ndarray, score: np.ndarray, threshold: float = 0.0) -> float:
    """Policy-Value-Kennzahl für Binary Treatment über die gesamte Population.

    Policy: Behandle wenn score >= threshold, nicht behandeln wenn score < threshold.

    Wert: Gewichteter Gesamtnutzen der Policy über alle Personen.
    - CATE >= threshold (behandeln): Uplift = E[Y|T=1] - E[Y|T=0]
    - CATE < threshold (nicht behandeln): Vermiedener Schaden = E[Y|T=0] - E[Y|T=1]
    Gewichtet nach dem Anteil der jeweiligen Gruppe an der Gesamtpopulation.

    WICHTIG: Nutzt naive Differenz-in-Means (kein IPW/Propensity-Weighting).
    Nur bei randomisiertem Treatment (RCT) unverzerrt. Die Analyse-Pipeline
    ruft diese Funktion nur bei study_type='rct' auf. Für Beobachtungsdaten
    liefert der DRTester doubly-robust Policy Values (siehe drtester_core.py)."""
    y = np.asarray(y).astype(int)
    t = np.asarray(t).astype(int)
    score = np.asarray(score).astype(float)
    _check_binary(y, "y")
    _check_binary(t, "t")

    n = len(score)
    if n == 0:
        return 0.0

    value = 0.0

    # Gruppe 1: CATE >= threshold → behandeln → Uplift messen
    pos_mask = score >= float(threshold)
    if pos_mask.sum() > 0:
        y_p, t_p = y[pos_mask], t[pos_mask]
        n_t, n_c = (t_p == 1).sum(), (t_p == 0).sum()
        if n_t > 0 and n_c > 0:
            uplift = y_p[t_p == 1].mean() - y_p[t_p == 0].mean()
            value += uplift * pos_mask.sum() / n

    # Gruppe 2: CATE < threshold → nicht behandeln → vermiedener Schaden
    neg_mask = ~pos_mask
    if neg_mask.sum() > 0:
        y_n, t_n = y[neg_mask], t[neg_mask]
        n_t, n_c = (t_n == 1).sum(), (t_n == 0).sum()
        if n_t > 0 and n_c > 0:
            avoided_harm = y_n[t_n == 0].mean() - y_n[t_n == 1].mean()
            value += avoided_harm * neg_mask.sum() / n

    return float(value)


# ---------------------------------------------------------------------------
# Multi-Treatment Metriken
# ---------------------------------------------------------------------------

def uplift_curve_mt_per_arm(
    y: np.ndarray,
    t: np.ndarray,
    scores: np.ndarray,
    treatment_arm: int,
) -> UpliftCurve:
    """Uplift-Kurve für einen einzelnen Treatment-Arm vs. Control (T=0).
    Filtert auf Beobachtungen mit T in {0, treatment_arm} und berechnet
    die Standard-Uplift-Kurve auf diesem Subset.

    Parameters
    ----------
    y : Outcome-Vektor (binär)
    t : Treatment-Vektor (diskret, 0 = Control)
    scores : CATE-Schätzungen für diesen Treatment-Arm, Shape (n,)
    treatment_arm : welcher Arm (z. B. 1, 2, ...)
    """
    y = np.asarray(y).astype(int)
    t = np.asarray(t).astype(int)
    scores = np.asarray(scores).astype(float)
    _check_binary(y, "y")
    _check_discrete(t, "t")

    mask = (t == 0) | (t == treatment_arm)
    y_sub = y[mask]
    t_sub = (t[mask] == treatment_arm).astype(int)
    s_sub = scores[mask]

    n = len(y_sub)
    if n == 0:
        return UpliftCurve(
            fraction=np.array([]),
            n_treat=np.array([]),
            n_control=np.array([]),
            y_treat=np.array([]),
            y_control=np.array([]),
            uplift=np.array([]),
        )

    _rng_mt = np.random.default_rng(42)
    order = np.argsort(-(s_sub + tiebreak_jitter(s_sub, _rng_mt)))
    y_sub = y_sub[order]
    t_sub = t_sub[order]

    treat = t_sub.astype(int)
    control = (1 - t_sub).astype(int)

    n_t = np.cumsum(treat)
    n_c = np.cumsum(control)
    y_t = np.cumsum(y_sub * treat)
    y_c = np.cumsum(y_sub * control)

    rate_t = y_t / np.maximum(n_t, 1)
    rate_c = y_c / np.maximum(n_c, 1)
    inc_rate = rate_t - rate_c
    frac = np.arange(1, n + 1) / n
    uplift_vals = inc_rate * frac

    return UpliftCurve(
        fraction=frac,
        n_treat=n_t,
        n_control=n_c,
        y_treat=y_t,
        y_control=y_c,
        uplift=uplift_vals,
    )


def policy_value_mt(
    y: np.ndarray,
    t: np.ndarray,
    scores_2d: np.ndarray,
    propensity: Optional[np.ndarray] = None,
) -> float:
    """Policy-Value-Schätzer für Multi-Treatment via IPW.

    Die optimale Policy ist pi*(X) = argmax_k tau_k(X) (k=1..K-1).
    Wenn max_k tau_k(X) <= 0, wird "nicht behandeln" (T=0) empfohlen.

    Parameters
    ----------
    y : Outcome (binär)
    t : Beobachtete Treatment-Zuweisung (0..K-1)
    scores_2d : CATE-Schätzungen, Shape (n, K-1)
    propensity : Optional (n, K) Array mit P(T=k|X). Falls None, wird
                 die empirische Verteilung genutzt (Randomisierung angenommen).

    Returns
    -------
    Geschätzter Policy-Value (V_IPW(pi*) - Baseline-Rate)

    Hinweis
    -------
    Die Default-Propensity (empirische Verteilung) ist nur unter der Annahme
    einer Randomisierung (z. B. aus einem A/B-Test) unverzerrt. Bei
    observationalen Daten sollte eine geschätzte Propensity (z. B. aus einem
    Klassifikator) über den Parameter ``propensity`` übergeben werden, um
    Confounding-Bias zu reduzieren.
    """
    y = np.asarray(y).astype(float)
    t = np.asarray(t).astype(int)
    scores_2d = np.asarray(scores_2d).astype(float)
    _check_discrete(t, "t")

    n = len(y)
    K = int(t.max()) + 1  # Anzahl Treatment-Gruppen inkl. Control

    if scores_2d.ndim == 1:
        scores_2d = scores_2d.reshape(-1, 1)

    # Optimale Zuweisung: argmax über K-1 Arme, aber nur wenn > 0
    best_effect = np.max(scores_2d, axis=1)
    best_arm = np.argmax(scores_2d, axis=1) + 1  # 1-basiert
    # Wenn der beste Effekt <= 0 ist, weise Control (0) zu
    optimal_treatment = np.where(best_effect > 0, best_arm, 0)

    # IPW-Schätzer
    if propensity is None:
        # Empirische Propensity (unter Randomisierungsannahme)
        propensity = np.zeros((n, K), dtype=float)
        for k in range(K):
            propensity[:, k] = max((t == k).sum(), 1) / n

    # V_IPW(pi) = (1/n) * sum_i Y_i * 1[T_i == pi(X_i)] / P(T_i | X_i)
    match = (t == optimal_treatment).astype(float)
    prop_obs = propensity[np.arange(n), t]
    prop_obs = np.maximum(prop_obs, 1e-10)

    policy_val = np.mean(y * match / prop_obs)
    # Baseline: Rate unter Control
    control_mask = t == 0
    baseline = y[control_mask].mean() if control_mask.sum() > 0 else 0.0

    return float(policy_val - baseline)


def policy_value_per_arm(
    y: np.ndarray,
    t: np.ndarray,
    scores: np.ndarray,
    treatment_arm: int,
    threshold: float = 0.0,
) -> float:
    """Policy-Value-Kennzahl für einen einzelnen Treatment-Arm vs. Control.

    Analogon zur Binary-Treatment-Funktion ``policy_value()``, aber für
    Multi-Treatment: Es wird auf Beobachtungen mit T ∈ {0, treatment_arm}
    gefiltert und dann der gewichtete Gesamtnutzen der Policy berechnet.

    - CATE >= threshold: Uplift = E[Y|T=arm] - E[Y|T=0]
    - CATE < threshold: Vermiedener Schaden = E[Y|T=0] - E[Y|T=arm]
    Gewichtet nach Gruppenanteil.
    """
    y = np.asarray(y).astype(int)
    t = np.asarray(t).astype(int)
    scores = np.asarray(scores).astype(float)
    _check_binary(y, "y")
    _check_discrete(t, "t")

    # Filtere auf Control + gewählten Arm
    arm_mask = (t == 0) | (t == treatment_arm)
    y_sub = y[arm_mask]
    t_sub = (t[arm_mask] == treatment_arm).astype(int)
    s_sub = scores[arm_mask]

    n = len(s_sub)
    if n == 0:
        return 0.0

    value = 0.0

    # Gruppe 1: CATE >= threshold → behandeln
    pos_mask = s_sub >= float(threshold)
    if pos_mask.sum() > 0:
        y_p, t_p = y_sub[pos_mask], t_sub[pos_mask]
        n_t, n_c = (t_p == 1).sum(), (t_p == 0).sum()
        if n_t > 0 and n_c > 0:
            uplift = y_p[t_p == 1].mean() - y_p[t_p == 0].mean()
            value += uplift * pos_mask.sum() / n

    # Gruppe 2: CATE < threshold → nicht behandeln
    neg_mask = ~pos_mask
    if neg_mask.sum() > 0:
        y_n, t_n = y_sub[neg_mask], t_sub[neg_mask]
        n_t, n_c = (t_n == 1).sum(), (t_n == 0).sum()
        if n_t > 0 and n_c > 0:
            avoided_harm = y_n[t_n == 0].mean() - y_n[t_n == 1].mean()
            value += avoided_harm * neg_mask.sum() / n

    return float(value)


def optimal_treatment_assignment(scores_2d: np.ndarray) -> np.ndarray:
    """Bestimmt für jede Beobachtung das optimale Treatment.

    Parameters
    ----------
    scores_2d : CATE-Schätzungen, Shape (n, K-1)

    Returns
    -------
    Array mit optimalen Treatment-Zuweisungen (0 = Control, 1..K-1 = Treatment)
    """
    scores_2d = np.asarray(scores_2d).astype(float)
    if scores_2d.ndim == 1:
        scores_2d = scores_2d.reshape(-1, 1)

    best_effect = np.max(scores_2d, axis=1)
    best_arm = np.argmax(scores_2d, axis=1) + 1
    return np.where(best_effect > 0, best_arm, 0)


def mt_eval_summary(
    y: np.ndarray,
    t: np.ndarray,
    scores_2d: np.ndarray,
    propensity: Optional[np.ndarray] = None,
) -> dict:
    """Erzeugt ein Evaluations-Summary für Multi-Treatment.

    Enthält:
    - Pro Treatment-Arm: qini_T{k}, auuc_T{k}, uplift10_T{k}, uplift20_T{k},
      uplift50_T{k}, policy_value_T{k}
    - Global: policy_value, best_treatment_distribution

    Parameters
    ----------
    y : Outcome (binär)
    t : Treatment (diskret, 0..K-1)
    scores_2d : CATE-Schätzungen, Shape (n, K-1)
    propensity : Optional (n, K) Array mit P(T=k|X). Falls None, wird
                 die empirische Verteilung genutzt (Randomisierung angenommen).
    """
    y = np.asarray(y).astype(int)
    t = np.asarray(t).astype(int)
    scores_2d = np.asarray(scores_2d).astype(float)

    n_effects = scores_2d.shape[1] if scores_2d.ndim == 2 else 1
    if scores_2d.ndim == 1:
        scores_2d = scores_2d.reshape(-1, 1)

    result: dict = {}

    # Per-Arm Metriken (Ansatz A)
    # Key-Namen folgen der BT-Konvention (uplift10/uplift20/uplift50 statt uplift_at_10pct/
    # uplift_at_20pct/uplift_at_50pct), damit MLflow-Metriken konsistent benannt sind.
    for k in range(n_effects):
        arm = k + 1
        try:
            curve = uplift_curve_mt_per_arm(y, t, scores_2d[:, k], treatment_arm=arm)
            if len(curve.fraction) > 0:
                result[f"qini_T{arm}"] = float(qini_coefficient(curve))
                result[f"auuc_T{arm}"] = float(auuc(curve))
                result[f"uplift10_T{arm}"] = float(uplift_at_k(curve, 0.10))
                result[f"uplift20_T{arm}"] = float(uplift_at_k(curve, 0.20))
                result[f"uplift50_T{arm}"] = float(uplift_at_k(curve, 0.50))
            else:
                result[f"qini_T{arm}"] = 0.0
                result[f"auuc_T{arm}"] = 0.0
                result[f"uplift10_T{arm}"] = 0.0
                result[f"uplift20_T{arm}"] = 0.0
                result[f"uplift50_T{arm}"] = 0.0
        except Exception:
            result[f"qini_T{arm}"] = 0.0
            result[f"auuc_T{arm}"] = 0.0
            result[f"uplift10_T{arm}"] = 0.0
            result[f"uplift20_T{arm}"] = 0.0
            result[f"uplift50_T{arm}"] = 0.0

        # Per-Arm Policy Value (Ansatz A.2): Behandle Arm k, wenn CATE_k > 0
        try:
            result[f"policy_value_T{arm}"] = float(
                policy_value_per_arm(y, t, scores_2d[:, k], treatment_arm=arm, threshold=0.0)
            )
        except Exception:
            result[f"policy_value_T{arm}"] = 0.0

    # Globaler Policy Value (Ansatz B): Optimale Zuweisung über alle Arme via IPW
    try:
        result["policy_value"] = policy_value_mt(y, t, scores_2d, propensity=propensity)
    except Exception:
        result["policy_value"] = 0.0

    # Treatment-Verteilung der optimalen Zuweisung
    optimal = optimal_treatment_assignment(scores_2d)
    K = int(t.max()) + 1
    dist = {}
    for k in range(K):
        dist[f"T{k}"] = float((optimal == k).sum() / len(optimal))
    result["best_treatment_distribution"] = dist

    return result
