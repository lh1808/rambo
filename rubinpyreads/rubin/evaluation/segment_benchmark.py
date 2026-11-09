"""Segment-Benchmark: das ehrliche „Mini-Oracle" für die Qini-Einordnung.

Beantwortet die Frage: **Wie viel Qini wäre allein die Kenntnis deklarierter
Segmente wert?** (z. B. einer Missing-Flag oder der Lead-Art). Der Benchmark-
Score weist jeder Zeile den geschätzten Treatment-Effekt ihres Segments zu
und läuft dann durch dieselbe Evaluations-Strecke wie Modelle und
historischer Score. Drei-Stufen-Diagnose im Report:

1. Benchmark-Qini im Rauschband  → die deklarierte Moderation trägt
   ökonomisch nichts; kein Modell kann daraus mehr machen.
2. Benchmark über dem Band, Modell darunter → Lern-Problem des Modells.
3. Modell ≈ Benchmark → Modell hat die bekannte Moderation eingefangen.

Drei Ehrlichkeits-Regeln, ohne die der Benchmark denselben Verzerrungen
unterläge, die er aufdecken soll:

- **Nie aus den Eval-Daten geschätzt**: Segment-ATEs kommen aus den
  Trainingsdaten; im Cross-Prediction-Modus out-of-fold (GroupAware-KFold).
  Sonst holte man sich den Gewinner-Fluch zurück (max über verrauschte
  Zell-ATEs derselben Daten, auf denen gemessen wird).
- **Cluster-robuste Standardfehler**: Mit Gruppen (z. B. Partner) werden die
  SEs über Cluster gerechnet — zeilenweise SEs überverkaufen Signifikanz.
- **Empirical-Bayes-Shrinkage (DerSimonian-Laird)**: Erst wird die echte
  Zwischen-Segment-Varianz tau² geschätzt, dann jedes Segment präzisions-
  gewichtet Richtung Gesamt-ATE gezogen. Ist die Heterogenität statistisch
  nicht belegbar (tau² ≈ 0), kollabiert der Score auf eine Konstante und
  sein Qini ehrlich auf ~0 — kleine Zellen mit großen Zufalls-ATEs (der
  klassische „3,6pp ± 4,0"-Fall) können den Benchmark nicht kapern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_logger = logging.getLogger("rubin.segment_benchmark")

_MISSING = "⟨fehlend⟩"


def _cluster_mean_var(vals: np.ndarray, clusters: Optional[np.ndarray]) -> float:
    """Varianz des Mittelwerts von ``vals`` — cluster-robust, wenn Cluster da sind.

    Ohne Cluster: var(vals)/n. Mit Clustern (Summen S_g, Größen n_g):
    Var(ȳ) = G/(G−1) · Σ_g (S_g − n_g·ȳ)² / (Σ n_g)².
    """
    n = len(vals)
    if n <= 1:
        return float("nan")
    if clusters is None:
        return float(np.var(vals, ddof=1) / n)
    codes = np.unique(np.asarray(clusters).astype(str), return_inverse=True)[1]
    G = int(codes.max()) + 1
    if G <= 1:
        return float(np.var(vals, ddof=1) / n)
    s_g = np.bincount(codes, weights=vals, minlength=G)
    n_g = np.bincount(codes, minlength=G).astype(float)
    ybar = float(vals.mean())
    resid = s_g - n_g * ybar
    return float((G / (G - 1)) * np.sum(resid**2) / (n**2))


def _benjamini_hochberg(pvals: List[float]) -> List[float]:
    """BH-FDR-Adjustierung (monoton, gekappt bei 1). Eigene Mini-Implementierung,
    damit kein Import-Pfad die Benchmark-Diagnose reißen kann."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank_from_end, i in enumerate(reversed(order)):
        rank = m - rank_from_end
        val = min(prev, pvals[i] * m / rank)
        adj[i] = min(1.0, val)
        prev = adj[i]
    return adj


@dataclass
class SegmentBenchmark:
    """Gefittetes Segment-Benchmark (Mapping + Diagnose-Tabelle)."""

    columns: List[str]
    mapping: Dict[Tuple, float]
    global_ate: float
    tau2: float
    table: List[dict] = field(default_factory=list)
    skipped_columns: List[dict] = field(default_factory=list)
    n_segments: int = 0
    heterogeneity_detected: bool = False
    q_stat: float = 0.0
    q_pvalue: float = 1.0
    min_effect: float = 0.0


def _segment_keys(seg_df: pd.DataFrame, columns: List[str]) -> pd.Series:
    parts = []
    for c in columns:
        col = seg_df[c]
        s = col.astype(object).where(~pd.isna(col), _MISSING).astype(str)
        parts.append(s)
    if len(parts) == 1:
        return parts[0]
    out = parts[0]
    for p in parts[1:]:
        out = out.str.cat(p, sep=" | ")
    return out


def resolve_benchmark_columns(
    requested: List[str], available: List[str], max_levels: int, seg_source: pd.DataFrame
) -> Tuple[List[str], List[dict]]:
    """Case-insensitive Spalten-Auflösung + Kardinalitäts-Guard."""
    by_upper = {str(c).upper(): c for c in available}
    used, skipped = [], []
    for name in requested or []:
        col = by_upper.get(str(name).upper())
        if col is None:
            skipped.append({"column": str(name), "reason": "nicht in X gefunden"})
            continue
        n_lev = int(seg_source[col].astype(object).where(~pd.isna(seg_source[col]), _MISSING).nunique())
        if n_lev > int(max_levels):
            skipped.append({"column": str(col), "reason": f"{n_lev} Ausprägungen > max_levels={max_levels}"})
            continue
        used.append(col)
    return used, skipped


def fit_segment_benchmark(
    seg_df: pd.DataFrame,
    t,
    y,
    *,
    columns: Optional[List[str]] = None,
    groups=None,
    min_rows: int = 20,
    min_effect: float = 0.0,
) -> Optional[SegmentBenchmark]:
    """Schätzt Segment-ATEs (Diff-in-Means) mit Cluster-SEs und DL-Shrinkage.

    ``seg_df`` enthält nur die (bereits aufgelösten) Segment-Spalten.
    Segmente ohne beide Arme oder mit < min_rows Zeilen bekommen keinen
    eigenen ATE (Mapping → Gesamt-ATE, in der Tabelle markiert).
    """
    columns = list(columns if columns is not None else seg_df.columns)
    if not columns:
        return None
    t = np.asarray(t).ravel()
    y = np.asarray(y, dtype=float).ravel()
    groups_arr = None if groups is None else np.asarray(groups)
    keys = _segment_keys(seg_df, columns).to_numpy()

    mask_t, mask_c = (t == 1), (t == 0)
    y_t_mean = float(y[mask_t].mean()) if mask_t.any() else 0.0
    y_c_mean = float(y[mask_c].mean()) if mask_c.any() else 0.0
    # Gesamt-ATE als Fallback, wird unten durch fixed-effect-Schätzer ersetzt,
    # sobald gültige Segmente existieren.
    global_ate = y_t_mean - y_c_mean

    rows = []
    for seg in pd.unique(keys):
        m = keys == seg
        n_seg = int(m.sum())
        mt, mc = m & mask_t, m & mask_c
        n_t, n_c = int(mt.sum()), int(mc.sum())
        row = {
            "segment": str(seg), "n": n_seg, "n_treat": n_t,
            "t_rate": (n_t / n_seg) if n_seg else float("nan"),
            "ate": None, "se": None, "z": None, "ate_shrunk": None,
            "estimable": False,
            # Priorisierungs-/Depriorisierungs-Diagnose (gefüllt für estimable):
            "ci_lo": None, "ci_hi": None,
            "ate_rest": None, "delta_vs_rest": None, "se_contrast": None,
            "ci_contrast_lo": None, "ci_contrast_hi": None,
            "p_contrast": None, "p_contrast_adj": None,
            "direction": None,        # "above"/"below" (BH-belegt) sonst None
            "tendency": None,         # "above"/"below" (roh p<0.05, adj nicht) sonst None
            "verdict": "inconclusive",  # vs. min_effect: effective/below_threshold/inconclusive
        }
        if n_seg >= int(min_rows) and n_t >= 2 and n_c >= 2:
            ate = float(y[mt].mean() - y[mc].mean())
            var = _cluster_mean_var(y[mt], groups_arr[mt] if groups_arr is not None else None) + \
                _cluster_mean_var(y[mc], groups_arr[mc] if groups_arr is not None else None)
            se = float(np.sqrt(var)) if np.isfinite(var) and var > 0 else None
            row.update(ate=ate, se=se, estimable=se is not None)
            if se:
                row["z"] = ate / se
                row["ci_lo"] = ate - 1.96 * se
                row["ci_hi"] = ate + 1.96 * se
                # Absolute Einordnung gegen die Relevanzschwelle (Default 0):
                # NUR "below_threshold" rechtfertigt Depriorisieren/Abschalten —
                # "inconclusive" heißt: Datenlage reicht nicht, nicht "wirkt nicht".
                if row["ci_lo"] > float(min_effect):
                    row["verdict"] = "effective"
                elif row["ci_hi"] < float(min_effect):
                    row["verdict"] = "below_threshold"
                # Relativer Kontrast: Segment vs. alle übrigen Zeilen (Priorisierung).
                # Hinweis: Cluster, die über Segment-Grenzen spannen, erzeugen eine
                # (meist kleine) Kovarianz zwischen Segment- und Rest-Schätzer, die
                # hier vernachlässigt wird — die SEs beider Seiten sind einzeln
                # cluster-robust.
                r_mt, r_mc = (~m) & mask_t, (~m) & mask_c
                if r_mt.sum() >= 2 and r_mc.sum() >= 2:
                    ate_rest = float(y[r_mt].mean() - y[r_mc].mean())
                    var_rest = _cluster_mean_var(y[r_mt], groups_arr[r_mt] if groups_arr is not None else None) + \
                        _cluster_mean_var(y[r_mc], groups_arr[r_mc] if groups_arr is not None else None)
                    if np.isfinite(var_rest) and var_rest > 0:
                        se_c = float(np.sqrt(se**2 + var_rest))
                        delta = ate - ate_rest
                        row.update(ate_rest=ate_rest, delta_vs_rest=delta, se_contrast=se_c,
                                   ci_contrast_lo=delta - 1.96 * se_c,
                                   ci_contrast_hi=delta + 1.96 * se_c)
                        try:
                            from scipy.stats import norm
                            row["p_contrast"] = float(2.0 * norm.sf(abs(delta / se_c)))
                        except Exception:  # pragma: no cover
                            row["p_contrast"] = None
        rows.append(row)

    est = [r for r in rows if r["estimable"]]
    tau2, q_stat, q_pvalue = 0.0, 0.0, 1.0
    if est:
        w = np.array([1.0 / r["se"] ** 2 for r in est])
        ates = np.array([r["ate"] for r in est])
        ate_fe = float(np.sum(w * ates) / np.sum(w))
        global_ate = ate_fe
        if len(est) > 1:
            Q = float(np.sum(w * (ates - ate_fe) ** 2))
            c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
            tau2 = max(0.0, (Q - (len(est) - 1)) / c) if c > 0 else 0.0
            q_stat = Q
            try:
                from scipy.stats import chi2
                q_pvalue = float(chi2.sf(Q, len(est) - 1))
            except Exception:  # pragma: no cover - scipy ist reguläre Dependency
                q_pvalue = 0.0 if Q > 2.5 * (len(est) - 1) else 1.0
            # Pre-Test-Shrinkage: Verwirft der Homogenitätstest NICHT, wird
            # tau² auf 0 gesetzt — der Score kollabiert dann exakt auf den
            # Gesamt-ATE (Qini ehrlich ≈ 0), statt aus einem zufälligen
            # Mini-tau² doch noch ein Pseudo-Ranking zu bauen. Das macht die
            # dokumentierte Garantie "ohne belegbare Heterogenität ist aus
            # den Segmenten nichts zu holen" hart.
            if q_pvalue >= 0.05:
                tau2 = 0.0
        for r in est:
            if tau2 > 0:
                wi, wg = 1.0 / r["se"] ** 2, 1.0 / tau2
                r["ate_shrunk"] = float((wi * r["ate"] + wg * ate_fe) / (wi + wg))
            else:
                r["ate_shrunk"] = ate_fe
    for r in rows:
        if not r["estimable"]:
            r["ate_shrunk"] = global_ate

    # Benjamini-Hochberg über die vs.-Rest-Kontraste: belegt ist eine Richtung
    # erst nach FDR-Korrektur (11 Segmente = 55 implizite Paarvergleiche — ohne
    # Korrektur wäre fast immer irgendetwas "signifikant"). Roh-signifikante,
    # aber nicht korrigiert belegte Segmente werden als TENDENZ markiert — als
    # vorregistrierte Einzelhypothese in der nächsten Welle prüfbar.
    _ctr = [r for r in rows if r.get("p_contrast") is not None]
    if _ctr:
        adj = _benjamini_hochberg([r["p_contrast"] for r in _ctr])
        for r, pa in zip(_ctr, adj):
            r["p_contrast_adj"] = float(pa)
            _dir = "above" if r["delta_vs_rest"] > 0 else "below"
            if pa < 0.05:
                r["direction"] = _dir
            elif r["p_contrast"] < 0.05:
                r["tendency"] = _dir

    mapping = {r["segment"]: float(r["ate_shrunk"]) for r in rows}
    rows.sort(key=lambda r: -(r["ate_shrunk"] if r["ate_shrunk"] is not None else -np.inf))
    # "Belegbar" heißt: Der Homogenitätstest (Cochrans Q, chi² mit k−1 df)
    # verwirft. Bloßes tau² > 0 reicht NICHT — bei homogenen Daten liegt Q
    # zufällig in der Hälfte der Fälle über den Freiheitsgraden und erzeugt
    # ein Mini-tau². Das Shrinkage nutzt tau² weiterhin (auch ein kleines
    # tau² zieht fast alles auf den Gesamt-ATE); das FLAG für Report und
    # Diagnose folgt dem Test.
    return SegmentBenchmark(
        columns=columns, mapping=mapping, global_ate=float(global_ate),
        tau2=float(tau2), table=rows, n_segments=len(rows),
        heterogeneity_detected=bool(len(est) > 1 and q_pvalue < 0.05),
        q_stat=float(q_stat), q_pvalue=float(q_pvalue),
        min_effect=float(min_effect),
    )


def score_segment_benchmark(bench: SegmentBenchmark, seg_df: pd.DataFrame) -> np.ndarray:
    """Benchmark-Score: jede Zeile erhält den geshrunkenen ATE ihres Segments.

    Unbekannte Segmente (nur im Eval gesehen) → Gesamt-ATE.
    """
    keys = _segment_keys(seg_df, bench.columns).to_numpy()
    return np.array([bench.mapping.get(k, bench.global_ate) for k in keys], dtype=float)


def fit_score_segment_benchmark_oof(
    seg_df: pd.DataFrame,
    t,
    y,
    *,
    columns: Optional[List[str]] = None,
    groups=None,
    n_splits: int = 5,
    seed: int = 42,
    min_rows: int = 20,
    min_effect: float = 0.0,
) -> Tuple[Optional[np.ndarray], Optional[SegmentBenchmark]]:
    """Out-of-fold-Benchmark für den Cross-Prediction-Modus.

    Jede Zeile wird von einem Fit gescort, der sie NICHT gesehen hat
    (GroupAware-KFold, t-stratifiziert, gruppenrein wenn Gruppen da sind).
    Zusätzlich wird ein Full-Data-Fit zurückgegeben — NUR für die
    Diagnose-Tabelle im Report, nie für den Score.
    """
    from rubin.utils.group_cv import GroupAwareStratifiedKFold

    t = np.asarray(t).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    full = fit_segment_benchmark(seg_df, t, y, columns=columns, groups=groups, min_rows=min_rows, min_effect=min_effect)
    if full is None:
        return None, None
    scores = np.full(n, np.nan, dtype=float)
    cv = GroupAwareStratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    groups_arr = None if groups is None else np.asarray(groups)
    for tr_idx, va_idx in cv.split(np.zeros(n), t, groups=groups_arr):
        bench = fit_segment_benchmark(
            seg_df.iloc[tr_idx], t[tr_idx], y[tr_idx],
            columns=full.columns,
            groups=groups_arr[tr_idx] if groups_arr is not None else None,
            min_rows=min_rows,
        )
        if bench is None:
            continue
        scores[va_idx] = score_segment_benchmark(bench, seg_df.iloc[va_idx])
    if np.isnan(scores).any():  # Rest (degenerierte Folds) → Full-Fallback
        nan_m = np.isnan(scores)
        scores[nan_m] = score_segment_benchmark(full, seg_df.iloc[np.where(nan_m)[0]])
    return scores, full
