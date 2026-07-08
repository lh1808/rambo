"""Numerische Gegenprüfung der Uplift-Metriken gegen unabhängige Referenzen.

Zwei Ebenen: (1) sklift-frei — Hand-Trapez über die Kurvendefinition inkl.
Ursprung (pinnt die Integral-Konvention), policy_value gegen naive
Differenz-in-Means. (2) sklift-Kreuztest (skippt ohne installiertes
scikit-uplift): exakte Flächen-Übereinstimmung (Toleranz deckt die
dokumentierte O(1/n²)-Jitter-Präzisionsgrenze) und identisches
Modell-Ranking — das entscheidet die Champion-Wahl."""

from __future__ import annotations

import numpy as np
import pytest

from rubin.evaluation.uplift_metrics import (
    auuc,
    policy_value,
    qini_coefficient,
    uplift_at_k,
    uplift_curve,
)


def _make(nseed, n=1500):
    rng = np.random.default_rng(nseed)
    t = rng.integers(0, 2, n)
    tau = 0.15 * (rng.normal(size=n) > 0)
    y = rng.binomial(1, np.clip(0.15 + tau * t + 0.05 * rng.normal(size=n), 0.01, 0.99))
    s = tau + 0.05 * rng.normal(size=n)
    return y, t, s


def test_qini_and_auuc_match_hand_reference_with_origin():
    for seed in range(3):
        y, t, s = _make(seed)
        n = len(y)
        c = uplift_curve(y, t, s)
        # Hand-Referenz exakt über die Kurvenfelder, Integral inkl. (0, 0)
        n_c_safe = np.maximum(c.n_control, 1)
        yq = np.concatenate([[0.0], (c.y_treat - c.y_control * c.n_treat / n_c_safe) / n])
        x = np.concatenate([[0.0], c.fraction])
        ref_q = float(np.trapezoid(yq - x * yq[-1], x))
        assert qini_coefficient(c) == pytest.approx(ref_q, abs=1e-15)
        yu = np.concatenate([[0.0], c.uplift])
        ref_a = float(np.trapezoid(yu - x * yu[-1], x))
        assert auuc(c) == pytest.approx(ref_a, abs=1e-15)


def test_policy_value_matches_diff_in_means():
    for seed in range(3):
        y, t, s = _make(seed)
        pos = s >= 0.0
        ref = 0.0
        if pos.any():
            ref += (y[pos][t[pos] == 1].mean() - y[pos][t[pos] == 0].mean()) * pos.mean()
        if (~pos).any():
            ref += (y[~pos][t[~pos] == 0].mean() - y[~pos][t[~pos] == 1].mean()) * (~pos).mean()
        assert policy_value(y, t, s) == pytest.approx(ref, abs=1e-15)


def test_metrics_match_sklift_exactly_and_preserve_ranking():
    sklift_metrics = pytest.importorskip("sklift.metrics")
    from scipy.stats import kendalltau

    def area(xs, ys):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        return float(np.trapezoid(ys - xs * (ys[-1] / xs[-1]), xs))

    rng = np.random.default_rng(7)
    rank_ok = 0
    for trial in range(10):
        n = int(rng.integers(400, 2500))
        t = rng.integers(0, 2, n)
        tau = 0.15 * (rng.normal(size=n) > 0)
        y = rng.binomial(1, np.clip(0.15 + tau * t + 0.05 * rng.normal(size=n), 0.01, 0.99))
        scores = [tau + 0.02 * rng.normal(size=n), tau + 0.3 * rng.normal(size=n),
                  -tau + 0.05 * rng.normal(size=n), rng.normal(size=n)]
        rq, sq = [], []
        for s in scores:
            c = uplift_curve(y, t, s)
            ref_q = area(*sklift_metrics.qini_curve(y, s, t))
            ref_a = area(*sklift_metrics.uplift_curve(y, s, t))
            # rubin normiert pro Population (÷n auf beiden Achsen) → Faktor n².
            # Toleranz 1e-4 relativ deckt die dokumentierte O(1/n²)-Jitter-
            # Präzisionsgrenze bei Scores mit Abständen < 1e-9·max|score|.
            assert qini_coefficient(c) * n * n == pytest.approx(ref_q, rel=1e-4, abs=1e-6)
            assert auuc(c) * n * n == pytest.approx(ref_a, rel=1e-4, abs=1e-6)
            for k in (0.1, 0.3):
                assert uplift_at_k(c, k) == pytest.approx(
                    sklift_metrics.uplift_at_k(y, s, t, strategy="overall", k=k), abs=1e-12)
            rq.append(qini_coefficient(c))
            sq.append(sklift_metrics.qini_auc_score(y, s, t))
        rank_ok += kendalltau(rq, sq).statistic == 1.0
    assert rank_ok == 10, f"Modell-Ranking wich in {10 - rank_ok} Trials von sklift ab"


# ---------------------------------------------------------------------------
# Multi-Treatment: Referenzen
# ---------------------------------------------------------------------------

def _make_mt(seed, n=1500, K=3):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, K, n)
    x = rng.normal(size=n)
    taus = np.stack([0.12 * (x > rng.normal()) + 0.02 * k for k in range(1, K)], axis=1)
    p = np.clip(0.12 + np.take_along_axis(
        np.hstack([np.zeros((n, 1)), taus]), t[:, None], 1)[:, 0], 0.01, 0.99)
    y = rng.binomial(1, p)
    scores = taus + 0.1 * rng.normal(size=taus.shape)
    return y, t, scores


def test_mt_per_arm_curve_is_bt_curve_on_subset():
    """Die Arm-Kurve muss BITIDENTISCH zur BT-Kurve auf der binärisierten
    Teilmenge {T ∈ {0, arm}} sein — damit erben die MT-Metriken transitiv
    die extern (sklift) verifizierte BT-Maschinerie."""
    from rubin.evaluation.uplift_metrics import uplift_curve_mt_per_arm
    for seed in range(4):
        y, t, scores = _make_mt(seed, K=int(3 + seed % 2))
        for k in range(scores.shape[1]):
            arm = k + 1
            mask = (t == 0) | (t == arm)
            c_mt = uplift_curve_mt_per_arm(y, t, scores[:, k], arm)
            c_bt = uplift_curve(y[mask], (t[mask] == arm).astype(int), scores[mask, k])
            for field in ("fraction", "n_treat", "n_control", "y_treat", "y_control", "uplift"):
                np.testing.assert_array_equal(
                    getattr(c_mt, field), getattr(c_bt, field), err_msg=f"arm={arm} {field}")


def test_mt_policy_values_match_references():
    from rubin.evaluation.uplift_metrics import (
        policy_value_per_arm, policy_value_mt, optimal_treatment_assignment)
    for seed in range(4):
        y, t, scores = _make_mt(seed)
        n = len(y)
        K = int(t.max()) + 1
        # per Arm == BT-policy_value auf Teilmenge (exakt)
        for k in range(scores.shape[1]):
            arm = k + 1
            mask = (t == 0) | (t == arm)
            assert policy_value_per_arm(y, t, scores[:, k], arm) == pytest.approx(
                policy_value(y[mask], (t[mask] == arm).astype(int), scores[mask, k]), abs=1e-15)
        # Zuweisung == Hand-argmax mit Kontroll-Fallback
        ref_pi = np.where(scores.max(1) > 0, scores.argmax(1) + 1, 0)
        np.testing.assert_array_equal(optimal_treatment_assignment(scores), ref_pi)
        # global == Hand-IPW (empirische Propensity, Kontroll-Baseline)
        prop_obs = np.array([max((t == g).sum(), 1) / n for g in range(K)])[t]
        v_ref = float(np.mean(y * (t == ref_pi) / np.maximum(prop_obs, 1e-10)) - y[t == 0].mean())
        assert policy_value_mt(y, t, scores) == pytest.approx(v_ref, abs=1e-12)


def test_mt_eval_summary_consistent_with_component_functions():
    from rubin.evaluation.uplift_metrics import (
        mt_eval_summary, uplift_curve_mt_per_arm, policy_value_per_arm, policy_value_mt)
    y, t, scores = _make_mt(4)
    summ = mt_eval_summary(y, t, scores)
    for k in range(scores.shape[1]):
        arm = k + 1
        c = uplift_curve_mt_per_arm(y, t, scores[:, k], arm)
        assert summ[f"qini_T{arm}"] == pytest.approx(qini_coefficient(c), abs=1e-15)
        assert summ[f"auuc_T{arm}"] == pytest.approx(auuc(c), abs=1e-15)
        assert summ[f"uplift10_T{arm}"] == pytest.approx(uplift_at_k(c, 0.10), abs=1e-15)
        assert summ[f"policy_value_T{arm}"] == pytest.approx(
            policy_value_per_arm(y, t, scores[:, k], arm), abs=1e-15)
    assert summ["policy_value"] == pytest.approx(policy_value_mt(y, t, scores), abs=1e-12)
