"""Tests für Uplift-Metriken (BT und MT)."""

from __future__ import annotations

import numpy as np
import pytest

from rubin.evaluation.uplift_metrics import (
    uplift_curve,
    auuc,
    qini_coefficient,
    uplift_at_k,
    policy_value,
    _check_discrete,
    uplift_curve_mt_per_arm,
    policy_value_mt,
    policy_value_per_arm,
    optimal_treatment_assignment,
    mt_eval_summary,
)


@pytest.fixture
def perfect_data():
    """Perfekter Score: behandelte Responder haben höchsten Score."""
    np.random.seed(42)
    n = 1000
    t = np.array([1] * 500 + [0] * 500)
    y = np.zeros(n, dtype=int)
    y[:200] = 1
    score = np.zeros(n, dtype=float)
    score[:200] = 1.0
    score[200:500] = 0.5
    score[500:] = 0.0
    return y, t, score


@pytest.fixture
def random_data():
    """Zufälliger Score (kein Signal)."""
    np.random.seed(42)
    n = 1000
    t = np.random.binomial(1, 0.5, n)
    y = np.random.binomial(1, 0.3, n)
    score = np.random.uniform(0, 1, n)
    return y, t, score


@pytest.fixture
def mt_data():
    """Multi-Treatment-Daten mit 3 Armen (Control + 2 Treatments)."""
    np.random.seed(42)
    n = 600
    t = np.array([0] * 200 + [1] * 200 + [2] * 200)
    y = np.zeros(n, dtype=int)
    y[:50] = 1       # control responders
    y[200:300] = 1   # treatment 1 responders
    y[400:420] = 1   # treatment 2 responders
    scores_2d = np.random.randn(n, 2)
    scores_2d[:, 0] += 0.5  # T1 besser
    return y, t, scores_2d


class TestUpliftCurve:
    def test_shape(self, random_data):
        y, t, score = random_data
        curve = uplift_curve(y, t, score)
        assert len(curve.fraction) == len(y)
        assert curve.fraction[-1] == pytest.approx(1.0)

    def test_binary_check(self):
        with pytest.raises(ValueError, match="binär"):
            uplift_curve(np.array([0, 1, 2]), np.array([0, 1, 0]), np.array([0.1, 0.2, 0.3]))


class TestAuuc:
    def test_positive_for_good_model(self, perfect_data):
        y, t, score = perfect_data
        curve = uplift_curve(y, t, score)
        assert auuc(curve) > 0

    def test_near_zero_for_random(self, random_data):
        y, t, score = random_data
        curve = uplift_curve(y, t, score)
        assert abs(auuc(curve)) < 0.1


class TestQiniCoefficient:
    def test_positive_for_good_model(self, perfect_data):
        y, t, score = perfect_data
        curve = uplift_curve(y, t, score)
        assert qini_coefficient(curve) > 0


class TestUpliftAtK:
    def test_uplift_at_10pct(self, perfect_data):
        y, t, score = perfect_data
        curve = uplift_curve(y, t, score)
        val = uplift_at_k(curve, k_fraction=0.10)
        assert val > 0

    def test_zero_fraction(self, perfect_data):
        y, t, score = perfect_data
        curve = uplift_curve(y, t, score)
        assert uplift_at_k(curve, k_fraction=0.0) == 0.0


class TestPlotMetricConsistency:
    """Konsistenz zwischen uplift_at_k, dem Uplift-by-Percentile-Plot und der
    Qini-Kurve — alle drei müssen auf denselben Top-k%-Schnitt verweisen, auch
    wenn die Eval-Set-Größe kein Vielfaches von n_bins ist (Floor-Schnitt)."""

    def _first_bin_and_cum(self, y, t, score, k, n_bins=10):
        # Spiegelt das Binning von _native_uplift_by_percentile (strategy='overall'):
        # identische Sortierung (default_rng(42)-Jitter) und linspace-Bin-Kanten.
        rng = np.random.default_rng(42)
        order = np.argsort(-(score + rng.uniform(-1e-10, 1e-10, len(score))))
        y_s, t_s = y[order], t[order]
        n = len(y_s)
        edges = np.linspace(0, n, n_bins + 1, dtype=int)
        hi = edges[int(round(k * n_bins))]  # kumulative Top-k%-Kante
        yb, tb = y_s[:hi], t_s[:hi]
        rt = yb[tb == 1].mean() if (tb == 1).sum() > 0 else 0.0
        rc = yb[tb == 0].mean() if (tb == 0).sum() > 0 else 0.0
        return float(rt - rc)

    @pytest.mark.parametrize("n", [2000, 2003, 1997, 999])
    def test_uplift_at_k_matches_percentile_bins(self, n):
        rng = np.random.default_rng(7)
        score = rng.normal(size=n)
        tau = 0.3 + 0.5 * score
        t = rng.binomial(1, 0.5, n)
        p = 1 / (1 + np.exp(-(0.2 * rng.normal(size=n) + tau * t)))
        y = rng.binomial(1, np.clip(p, 0, 1)).astype(int)
        curve = uplift_curve(y, t, score)

        # uplift@10% == erster Perzentil-Balken (= kumulative Top-10%-Kante)
        assert uplift_at_k(curve, 0.10) == pytest.approx(
            self._first_bin_and_cum(y, t, score, 0.10), abs=1e-12
        )
        # uplift@20%/50% == kumulative Dezil-Kanten
        assert uplift_at_k(curve, 0.20) == pytest.approx(
            self._first_bin_and_cum(y, t, score, 0.20), abs=1e-12
        )
        assert uplift_at_k(curve, 0.50) == pytest.approx(
            self._first_bin_and_cum(y, t, score, 0.50), abs=1e-12
        )

    @pytest.mark.parametrize("n", [2000, 2003, 1997])
    def test_qini_curve_point_matches_uplift_at_k(self, n):
        # Qini-Kurve bei 10% muss n_treat(10%) · uplift@10% sein
        # (Q(k) = Y_t − Y_c·N_t/N_c = N_t·(rate_t − rate_c)).
        rng = np.random.default_rng(11)
        score = rng.normal(size=n)
        t = rng.binomial(1, 0.5, n)
        p = 1 / (1 + np.exp(-(0.4 * score * t)))
        y = rng.binomial(1, np.clip(p, 0, 1)).astype(int)
        curve = uplift_curve(y, t, score)

        idx = int(np.clip(int(0.10 * n) - 1, 0, n - 1))
        n_c_safe = max(int(curve.n_control[idx]), 1)
        q_point = curve.y_treat[idx] - curve.y_control[idx] * curve.n_treat[idx] / n_c_safe
        assert q_point == pytest.approx(curve.n_treat[idx] * uplift_at_k(curve, 0.10), abs=1e-9)


class TestPolicyValue:
    def test_binary_check(self):
        with pytest.raises(ValueError, match="binär"):
            policy_value(np.array([0, 2]), np.array([0, 1]), np.array([0.1, 0.9]))

    def test_all_below_threshold(self, random_data):
        y, t, score = random_data
        val = policy_value(y, t, score, threshold=999.0)
        # Niemand wird behandelt → PV reflektiert "nicht-behandeln"-Ergebnis
        assert isinstance(val, float)


# --- Multi-Treatment Tests ---

class TestCheckDiscrete:
    def test_valid_discrete(self):
        _check_discrete(np.array([0, 1, 2, 3]), "test")

    def test_non_integer_raises(self):
        with pytest.raises(ValueError, match="ganzzahlig"):
            _check_discrete(np.array([0, 1.5]), "test")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="negative"):
            _check_discrete(np.array([-1, 0, 1]), "test")


class TestUpliftCurveMtPerArm:
    def test_shape(self, mt_data):
        y, t, scores_2d = mt_data
        curve = uplift_curve_mt_per_arm(y, t, scores_2d[:, 0], treatment_arm=1)
        # Nur Control + Arm 1
        n_subset = ((t == 0) | (t == 1)).sum()
        assert len(curve.fraction) == n_subset

    def test_two_arms(self, mt_data):
        y, t, scores_2d = mt_data
        curve1 = uplift_curve_mt_per_arm(y, t, scores_2d[:, 0], treatment_arm=1)
        curve2 = uplift_curve_mt_per_arm(y, t, scores_2d[:, 1], treatment_arm=2)
        # Beide sollten nicht-leere Kurven sein
        assert len(curve1.fraction) > 0
        assert len(curve2.fraction) > 0


class TestPolicyValueMt:
    def test_returns_float(self, mt_data):
        y, t, scores_2d = mt_data
        pv = policy_value_mt(y, t, scores_2d)
        assert isinstance(pv, float)

    def test_with_uniform_propensity(self, mt_data):
        y, t, scores_2d = mt_data
        n = len(y)
        prop = np.ones((n, 3)) / 3.0
        pv = policy_value_mt(y, t, scores_2d, propensity=prop)
        assert isinstance(pv, float)


class TestPolicyValuePerArm:
    def test_returns_float(self, mt_data):
        y, t, scores_2d = mt_data
        pv = policy_value_per_arm(y, t, scores_2d[:, 0], treatment_arm=1)
        assert isinstance(pv, float)

    def test_all_below_threshold(self, mt_data):
        y, t, scores_2d = mt_data
        pv = policy_value_per_arm(y, t, scores_2d[:, 0], treatment_arm=1, threshold=999.0)
        # Niemand wird behandelt → PV reflektiert "nicht-behandeln"-Ergebnis
        assert isinstance(pv, float)

    def test_consistent_with_bt_policy_value(self):
        """Bei nur einem Treatment-Arm sollte policy_value_per_arm konsistent
        mit der BT-Funktion policy_value sein (gleiche Daten, gleicher Filter)."""
        np.random.seed(42)
        n = 500
        t = np.array([0] * 250 + [1] * 250)
        y = np.random.binomial(1, 0.3, n)
        score = np.random.uniform(-1, 1, n)

        pv_bt = policy_value(y, t, score, threshold=0.0)
        pv_mt = policy_value_per_arm(y, t, score, treatment_arm=1, threshold=0.0)
        assert pv_bt == pytest.approx(pv_mt, abs=1e-10)

    def test_arm2_ignores_arm1(self, mt_data):
        """policy_value_per_arm für Arm 2 sollte nur T=0 und T=2 nutzen
        und konsistent mit der BT-Funktion policy_value auf dem Subset sein."""
        y, t, scores_2d = mt_data
        pv = policy_value_per_arm(y, t, scores_2d[:, 1], treatment_arm=2)
        # Manuell: Nur T in {0, 2}, dann BT policy_value auf dem Subset
        mask = (t == 0) | (t == 2)
        y_sub = y[mask]
        t_sub = (t[mask] == 2).astype(int)
        s_sub = scores_2d[:, 1][mask]
        expected = policy_value(y_sub, t_sub, s_sub, threshold=0.0)
        assert pv == pytest.approx(expected, abs=1e-10)


class TestOptimalTreatmentAssignment:
    def test_shape(self, mt_data):
        _, _, scores_2d = mt_data
        opt = optimal_treatment_assignment(scores_2d)
        assert opt.shape == (len(scores_2d),)

    def test_values_in_range(self, mt_data):
        _, _, scores_2d = mt_data
        opt = optimal_treatment_assignment(scores_2d)
        assert np.all(opt >= 0)
        assert np.all(opt <= 2)

    def test_negative_effects_yield_control(self):
        scores = np.array([[-1.0, -2.0], [-0.5, -0.3]])
        opt = optimal_treatment_assignment(scores)
        np.testing.assert_array_equal(opt, [0, 0])


class TestMtEvalSummary:
    def test_keys(self, mt_data):
        y, t, scores_2d = mt_data
        summary = mt_eval_summary(y, t, scores_2d)
        # Per-Arm-Metriken (Konvention: Kurzform wie in BT)
        assert "qini_T1" in summary
        assert "qini_T2" in summary
        assert "auuc_T1" in summary
        assert "auuc_T2" in summary
        assert "uplift10_T1" in summary
        assert "uplift10_T2" in summary
        assert "uplift20_T1" in summary
        assert "uplift20_T2" in summary
        assert "uplift50_T1" in summary
        assert "uplift50_T2" in summary
        # Per-Arm Policy Values
        assert "policy_value_T1" in summary
        assert "policy_value_T2" in summary
        assert isinstance(summary["policy_value_T1"], float)
        assert isinstance(summary["policy_value_T2"], float)
        # Globale Metriken
        assert "policy_value" in summary
        assert "best_treatment_distribution" in summary
        # Keine BT-only Keys
        assert "qini" not in summary
        assert "uplift_at_10pct" not in summary

    def test_with_propensity(self, mt_data):
        y, t, scores_2d = mt_data
        n = len(y)
        prop = np.ones((n, 3)) / 3.0
        summary = mt_eval_summary(y, t, scores_2d, propensity=prop)
        assert "policy_value" in summary
        assert isinstance(summary["policy_value"], float)

    def test_distribution_sums_to_one(self, mt_data):
        y, t, scores_2d = mt_data
        summary = mt_eval_summary(y, t, scores_2d)
        dist = summary["best_treatment_distribution"]
        total = sum(dist.values())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestTiebreakJitter:
    """Skalen-relativer Jitter: wirksam auf jeder Score-Skala."""

    def test_large_magnitude_ties_are_perturbed(self):
        # Ein absoluter Jitter (z.B. ±1e-10) würde bei Magnitude 1e8 von Float64
        # absorbiert (eps(1e8) ≈ 1.5e-8) — der relative Jitter muss die Werte
        # real verändern.
        from rubin.evaluation.uplift_metrics import tiebreak_jitter
        rng = np.random.default_rng(0)
        scores = np.full(1000, 1e8)
        jittered = scores + tiebreak_jitter(scores, rng)
        assert np.any(jittered != scores), "Jitter wurde von Float64 absorbiert"
        # ... aber klein genug, um echte Unterschiede nie zu überstimmen
        assert np.max(np.abs(jittered - scores)) < 1e8 * 1e-8

    def test_tiny_magnitude_ranking_preserved(self):
        # Bei winzigen Scores darf der Jitter echte Unterschiede nicht kippen.
        from rubin.evaluation.uplift_metrics import tiebreak_jitter
        rng = np.random.default_rng(0)
        scores = np.array([3e-12, 2e-12, 1e-12])
        order = np.argsort(-(scores + tiebreak_jitter(scores, rng)))
        assert list(order) == [0, 1, 2]

    def test_all_zero_scores_no_crash(self):
        from rubin.evaluation.uplift_metrics import tiebreak_jitter
        rng = np.random.default_rng(0)
        j = tiebreak_jitter(np.zeros(10), rng)
        assert j.shape == (10,) and np.all(np.isfinite(j))

    def test_uplift_curve_deterministic_across_calls(self):
        # Seed=42 intern: identische Inputs → identische Kurve, auch mit Ties.
        rng = np.random.default_rng(1)
        n = 200
        y = rng.integers(0, 2, n)
        t = rng.integers(0, 2, n)
        score = np.repeat([0.3, 0.1, 0.2, 0.1], n // 4)  # viele Ties
        c1 = uplift_curve(y, t, score)
        c2 = uplift_curve(y, t, score)
        np.testing.assert_array_equal(c1.uplift, c2.uplift)


class TestUpliftAtKEdgeCases:
    def test_k_smaller_than_one_row_returns_zero(self):
        # k·n < 1 → leere Top-Menge → 0.0
        rng = np.random.default_rng(2)
        n = 50
        y = rng.integers(0, 2, n)
        t = rng.integers(0, 2, n)
        score = rng.normal(size=n)
        curve = uplift_curve(y, t, score)
        assert uplift_at_k(curve, k_fraction=0.001) == 0.0
        # Regulärer Fall unverändert funktionsfähig
        assert np.isfinite(uplift_at_k(curve, k_fraction=0.5))


class TestUpliftCurveMtArgmax:
    """Referenztests für die Argmax-Qini-Kurve (Multi-Treatment-Ranking).

    Konsistenz-Anforderungen aus der Konstruktion (siehe Docstring):
    K=2 == binäre Kurve (exakt), dominanter Arm == per-Arm-Kurve (exakt),
    Diskrimination auf synthetischen MT-Daten, HT-Gewichte selbst-normalisiert.
    """

    def _mt_data(self, n=2000, seed=7, p=(0.34, 0.33, 0.33)):
        rng = np.random.RandomState(seed)
        X = rng.normal(size=(n, 3))
        t = rng.choice([0, 1, 2], size=n, p=list(p))
        tau1 = 0.15 * (X[:, 0] > 0)
        tau2 = 0.25 * (X[:, 1] > 0)
        base = 0.3 + 0.05 * X[:, 2]
        prob = np.clip(base + (t == 1) * tau1 + (t == 2) * tau2, 0.01, 0.99)
        y = (rng.uniform(size=n) < prob).astype(int)
        return X, t, y, np.column_stack([tau1, tau2])

    def test_k2_exactly_equals_binary_curve(self):
        from rubin.evaluation.uplift_metrics import (
            uplift_curve, uplift_curve_mt_argmax, qini_coefficient)
        rng = np.random.RandomState(1)
        n = 800
        t = rng.choice([0, 1], size=n)
        y = rng.choice([0, 1], size=n)
        score = rng.normal(size=n)
        c_bin = uplift_curve(y, t, score)
        c_arg = uplift_curve_mt_argmax(y, t, score.reshape(-1, 1))
        np.testing.assert_allclose(c_arg.uplift, c_bin.uplift, rtol=0, atol=1e-12)
        np.testing.assert_allclose(c_arg.n_treat, c_bin.n_treat, atol=1e-12)
        assert qini_coefficient(c_arg) == pytest.approx(qini_coefficient(c_bin), abs=1e-12)

    def test_dominant_arm_equals_per_arm_curve(self):
        from rubin.evaluation.uplift_metrics import (
            uplift_curve_mt_argmax, uplift_curve_mt_per_arm, qini_coefficient)
        X, t, y, tau = self._mt_data()
        # Modell empfiehlt überall Arm 1: Spalte 0 strikt größer
        scores = np.column_stack([tau[:, 0] + 1.0, tau[:, 1] - 1.0])
        c_arg = uplift_curve_mt_argmax(y, t, scores)
        c_arm = uplift_curve_mt_per_arm(y, t, scores[:, 0], treatment_arm=1)
        assert qini_coefficient(c_arg) == pytest.approx(qini_coefficient(c_arm), abs=1e-12)

    def test_discriminates_true_from_random_scores(self):
        from rubin.evaluation.uplift_metrics import uplift_curve_mt_argmax, qini_coefficient
        X, t, y, tau = self._mt_data(n=4000)
        rng = np.random.RandomState(9)
        q_true = qini_coefficient(uplift_curve_mt_argmax(y, t, tau))
        q_rand = np.mean([
            qini_coefficient(uplift_curve_mt_argmax(y, t, rng.normal(size=tau.shape)))
            for _ in range(5)])
        assert q_true > q_rand + 0.005, (q_true, q_rand)

    def test_unequal_priors_weights_self_normalized(self):
        from rubin.evaluation.uplift_metrics import uplift_curve_mt_argmax, qini_coefficient
        X, t, y, tau = self._mt_data(n=3000, p=(0.5, 0.15, 0.35))
        c = uplift_curve_mt_argmax(y, t, tau)
        # Gewichtete Treated-Zähler bleiben in der Größenordnung der echten Anzahl
        n_matched = c.n_treat[-1]
        assert 0 < n_matched
        assert np.isfinite(qini_coefficient(c))

    def test_input_validation(self):
        from rubin.evaluation.uplift_metrics import uplift_curve_mt_argmax
        y = np.array([0, 1, 0]); t = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="CATE-Spalten"):
            uplift_curve_mt_argmax(y, t, np.zeros((3, 1)))  # nur 1 Spalte für Arm 2
        with pytest.raises(ValueError, match="Shape"):
            uplift_curve_mt_argmax(y, t, np.zeros((2, 2)))  # falsche Zeilenzahl

    def test_model_ordering_preserved_under_unequal_priors(self):
        """Tuning-relevante Eigenschaft: Die Rangfolge der Kandidatenmodelle
        (wahre Effektfläche > vertauschte Arme > Zufall) bleibt auch bei stark
        ungleichen Zuweisungs-Priors erhalten (HT-Kompositions-Korrektur).
        Deterministisch über feste Seeds."""
        from rubin.evaluation.uplift_metrics import uplift_curve_mt_argmax, qini_coefficient
        for seed in (0, 1, 2):
            rng = np.random.RandomState(seed)
            n = 8000
            X = rng.normal(size=(n, 2))
            t = rng.choice([0, 1, 2], size=n, p=[0.34, 0.06, 0.60])
            tau1 = 0.30 * (X[:, 0] > 0)
            tau2 = 0.10 * np.ones(n)
            prob = np.clip(0.3 + (t == 1) * tau1 + (t == 2) * tau2, 0.01, 0.99)
            y = (rng.uniform(size=n) < prob).astype(int)
            tau = np.column_stack([tau1, tau2])
            q_true = qini_coefficient(uplift_curve_mt_argmax(y, t, tau))
            q_swap = qini_coefficient(uplift_curve_mt_argmax(y, t, tau[:, ::-1]))
            q_rand = qini_coefficient(uplift_curve_mt_argmax(y, t, rng.normal(size=tau.shape)))
            assert q_true > q_swap, (seed, q_true, q_swap)
            assert q_true > q_rand, (seed, q_true, q_rand)


class TestOrientHistoricalScore:
    """Regression für den NaN-Platzierungs-Bug: nan_to_num(0.0) VOR der
    Invertierung setzte fehlende Scores bei lower-better-Skalen an die
    SPITZE der Rangliste (0 > -1). Der Helper platziert Nicht-Finites
    orientiert ans ENDE."""

    def test_lower_better_nan_goes_to_end_not_top(self):
        from rubin.evaluation.uplift_metrics import orient_historical_score
        s = np.array([1.0, 5.0, np.nan, 10.0])   # 1 = bester (lower-better)
        o = orient_historical_score(s, higher_is_better=False)
        assert o[0] == -1.0 and o[3] == -10.0
        assert o[2] == -11.0                      # NaN → min-1 = ans Ende
        assert np.argsort(-o).tolist()[0] == 0    # bester bleibt vorn
        assert np.argsort(-o).tolist()[-1] == 2   # NaN ganz hinten

    def test_higher_better_nan_and_inf_to_end(self):
        from rubin.evaluation.uplift_metrics import orient_historical_score
        s = np.array([3.0, np.nan, 7.0, np.inf, -np.inf])
        o = orient_historical_score(s, higher_is_better=True)
        assert o[2] == 7.0 and o[0] == 3.0
        assert o[1] == o[3] == o[4] == 2.0        # min(3,7)-1 — alle ans Ende

    def test_no_input_mutation_and_all_nan(self):
        from rubin.evaluation.uplift_metrics import orient_historical_score
        s = np.array([np.nan, np.nan])
        o = orient_historical_score(s, higher_is_better=False)
        assert np.all(o == 0.0) and np.isnan(s).all()  # Fallback + Input unverändert


class TestHistoricalScoreAnomalies:
    """Guards für reale Stolperfallen beim historischen Score.

    Realfall: BEA_SCORE_WERT_HIST — Rang-Klassen 1..10 (1 = beste, Y-Rate 20%
    fallend auf 8%) PLUS upstream imputierter Mittelwert 3.655734 auf 48% der
    Zeilen. Mit rubin-Default higher_is_better=true wird der Score verkehrt
    gemessen, und der Tie-Block deckelt den Qini strukturell. Beide Muster
    müssen diagnostizierbar sein.
    """

    def _realcase(self, seed=0):
        """Synthetisches Abbild des Realfalls: invers + 48%-Mittelwert-Block."""
        rng = np.random.default_rng(seed)
        n = 20_000
        # 48% Block auf 3.655734, Rest Klassen 1..10
        is_block = rng.random(n) < 0.48
        s_raw = np.where(is_block, 3.655734, rng.integers(1, 11, n).astype(float))
        # Y-Rate fällt mit der Klasse (1 = beste): p = 0.20 - 0.013*(klasse-1)
        klasse = np.where(is_block, 5.0, s_raw)
        p = np.clip(0.20 - 0.013 * (klasse - 1.0), 0.02, 0.98)
        y = (rng.random(n) < p).astype(int)
        t = (rng.random(n) < 0.75).astype(int)
        return s_raw, y, t

    def test_dominance_detected_on_real_pattern(self):
        from rubin.evaluation.uplift_metrics import (
            historical_score_anomalies, orient_historical_score)
        s_raw, y, t = self._realcase()
        s = orient_historical_score(s_raw, higher_is_better=False)
        d = historical_score_anomalies(s, y, t, higher_is_better=False)
        assert d["dominance_suspect"]
        assert abs(d["dominant_share"] - 0.48) < 0.02
        # orientierter Blockwert ist -3.655734
        assert abs(d["dominant_value"] - (-3.655734)) < 1e-9

    def test_orientation_suspect_fires_only_when_wrong(self):
        from rubin.evaluation.uplift_metrics import (
            historical_score_anomalies, orient_historical_score)
        s_raw, y, t = self._realcase()
        # FALSCH orientiert (rubin-Default true auf lower-better-Rang):
        wrong = orient_historical_score(s_raw, higher_is_better=True)
        d_wrong = historical_score_anomalies(wrong, y, t, higher_is_better=True)
        assert d_wrong["orientation_suspect"]
        assert d_wrong["control_corr"] < -0.02
        # KORREKT orientiert: kein Verdacht
        right = orient_historical_score(s_raw, higher_is_better=False)
        d_right = historical_score_anomalies(right, y, t, higher_is_better=False)
        assert not d_right["orientation_suspect"]
        assert d_right["control_corr"] > 0

    def test_no_false_alarm_on_clean_continuous_score(self):
        from rubin.evaluation.uplift_metrics import historical_score_anomalies
        rng = np.random.default_rng(1)
        n = 5_000
        s = rng.normal(size=n)
        y = (rng.random(n) < 0.1 + 0.05 * (s > 0)).astype(int)  # positiv gekoppelt
        t = (rng.random(n) < 0.5).astype(int)
        d = historical_score_anomalies(s, y, t)
        assert not d["dominance_suspect"] and not d["orientation_suspect"]

    def test_small_control_group_stays_silent(self):
        from rubin.evaluation.uplift_metrics import historical_score_anomalies
        rng = np.random.default_rng(2)
        n = 300
        s = rng.normal(size=n)
        y = rng.integers(0, 2, n)
        t = np.ones(n); t[:50] = 0  # nur 50 Kontrollzeilen < min_control
        d = historical_score_anomalies(s, y, t)
        assert d["control_corr"] is None and not d["orientation_suspect"]

    def test_constant_score_no_crash_no_orientation_flag(self):
        from rubin.evaluation.uplift_metrics import historical_score_anomalies
        n = 1_000
        s = np.full(n, 7.0)
        y = np.random.default_rng(3).integers(0, 2, n)
        t = np.random.default_rng(4).integers(0, 2, n)
        d = historical_score_anomalies(s, y, t)
        # Ein-Wert-Score: Dominanz zu 100%, aber len(vals)==1 → kein Verdacht
        # (kein Imputations-BLOCK, sondern gar kein Score-Signal) und keine
        # Orientierungsaussage möglich.
        assert not d["dominance_suspect"] and not d["orientation_suspect"]

    def test_nan_in_score_and_y_handled(self):
        from rubin.evaluation.uplift_metrics import historical_score_anomalies
        rng = np.random.default_rng(5)
        n = 2_000
        s = rng.normal(size=n); s[:100] = np.nan
        y = rng.integers(0, 2, n).astype(float); y[100:150] = np.nan
        t = rng.integers(0, 2, n)
        d = historical_score_anomalies(s, y, t)  # darf nicht werfen
        assert d["control_n"] <= int((t == 0).sum())


class TestHistScoreGuardsInPipeline:
    """Die Guards feuern in _evaluate_historical_score (Realfall-Muster)."""

    def test_both_warnings_logged(self, monkeypatch, caplog):
        import logging
        import pandas as pd
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        import rubin.pipelines.analysis_pipeline as ap

        rng = np.random.default_rng(7)
        n = 6_000
        is_block = rng.random(n) < 0.48
        s_raw = np.where(is_block, 3.655734, rng.integers(1, 11, n).astype(float))
        klasse = np.where(is_block, 5.0, s_raw)
        p = np.clip(0.20 - 0.013 * (klasse - 1.0), 0.02, 0.98)
        Y = (rng.random(n) < p).astype(int)
        T = (rng.random(n) < 0.75).astype(int)
        X = pd.DataFrame({"a": rng.normal(size=n)})

        pipe = ap.AnalysisPipeline.__new__(ap.AnalysisPipeline)
        pipe._logger = logging.getLogger("rubin.test_hist_guards")
        pipe._eval_scores_ctx = {}
        pipe._plot_hist_uplift_basics = MagicMock()
        pipe._plot_models_vs_hist_qini = MagicMock()
        monkeypatch.setattr(ap, "generate_cate_distribution_plot", lambda **k: None)
        monkeypatch.setattr(ap, "evaluate_cate_with_plots", MagicMock(return_value=MagicMock()))

        cfg = SimpleNamespace(
            historical_score=SimpleNamespace(name="hist", higher_is_better=True),
            study_type="rct",
            constants=SimpleNamespace(random_seed=42, parallel_level=1),
            base_learner=SimpleNamespace(fixed_params={}, type="lightgbm"),
        )
        with caplog.at_level(logging.WARNING, logger="rubin.test_hist_guards"):
            eval_summary, pv = pipe._evaluate_historical_score(
                cfg, X, T, Y, None, {}, s_raw, {}, {}, MagicMock())
        text = caplog.text
        # Rang-Score lower-better + Default true → Orientierungs-Verdacht
        assert "higher_is_better" in text and "verkehrte Orientierung" in text
        # 48%-Mittelwert-Block → Dominanz-Warnung
        assert "DECKELT" in text and "3.65573" in text
        # Kette lief weiter: Qini wurde trotz Warnungen berechnet
        assert "qini" in eval_summary["hist"]

    def test_silent_on_clean_correctly_oriented_score(self, monkeypatch, caplog):
        import logging
        import pandas as pd
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        import rubin.pipelines.analysis_pipeline as ap

        rng = np.random.default_rng(8)
        n = 6_000
        s_raw = rng.normal(size=n)
        Y = (rng.random(n) < 0.1 + 0.05 * (s_raw > 0)).astype(int)
        T = (rng.random(n) < 0.5).astype(int)
        X = pd.DataFrame({"a": rng.normal(size=n)})

        pipe = ap.AnalysisPipeline.__new__(ap.AnalysisPipeline)
        pipe._logger = logging.getLogger("rubin.test_hist_guards2")
        pipe._eval_scores_ctx = {}
        pipe._plot_hist_uplift_basics = MagicMock()
        pipe._plot_models_vs_hist_qini = MagicMock()
        monkeypatch.setattr(ap, "generate_cate_distribution_plot", lambda **k: None)
        monkeypatch.setattr(ap, "evaluate_cate_with_plots", MagicMock(return_value=MagicMock()))

        cfg = SimpleNamespace(
            historical_score=SimpleNamespace(name="hist", higher_is_better=True),
            study_type="rct",
            constants=SimpleNamespace(random_seed=42, parallel_level=1),
            base_learner=SimpleNamespace(fixed_params={}, type="lightgbm"),
        )
        with caplog.at_level(logging.WARNING, logger="rubin.test_hist_guards2"):
            pipe._evaluate_historical_score(cfg, X, T, Y, None, {}, s_raw, {}, {}, MagicMock())
        assert "DECKELT" not in caplog.text
        assert "verkehrte Orientierung" not in caplog.text
