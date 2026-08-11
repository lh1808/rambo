"""RCT-Konsistenz: Der DRTester erhält (wie die CATE-Estimatoren über die
model_registry) die konstante Design-Propensity als DummyClassifier(prior)
statt eines gelernten Classifiers.

Hintergrund: Ein gelernter Propensity-Classifier kann auf randomisiertem
Treatment nur Rauschen fitten; der Prior eliminiert diese Restvarianz aus den
DR-Scores und macht EconMLs Propensity-Clip (np.clip(p, .01, inf) in
econml/validate/utils.py) strukturell zum No-Op — es gibt keine Werte, die
geclippt werden könnten.
"""
import re

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyClassifier

from rubin.evaluation.drtester_core import fit_drtester_nuisance


class TestDRTesterWithPrior:
    def test_fit_and_dr_outcomes_with_dummy_prior(self):
        """Funktional: Dummy-Prior + CustomDRTester fitten sauber; die
        DR-Outcomes reproduzieren bei konstanter Propensity den
        Differenz-der-Mittelwerte-ATE (AIPW-Kernidentität im RCT)."""
        rng = np.random.RandomState(0)
        n = 1200
        X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("ABCD"))
        T = rng.choice([0, 1], n)
        Y = (rng.uniform(size=n) < 0.30 + 0.10 * T).astype(int)

        tester = fit_drtester_nuisance(
            model_regression=LGBMRegressor(n_estimators=30, verbose=-1),
            model_propensity=DummyClassifier(strategy="prior"),
            X_val=X, T_val=T, Y_val=Y, cv=3, seed=42,
        )
        dr = np.asarray(tester.dr_val_).reshape(-1)
        assert dr.shape[0] == n and np.isfinite(dr).all()
        dim = Y[T == 1].mean() - Y[T == 0].mean()
        se = np.std(dr) / np.sqrt(n)
        assert abs(dr.mean() - dim) < 3 * se   # AIPW-ATE ≈ Diff-in-Means

    def test_pipeline_wires_prior_at_all_three_sites(self):
        """Quelltext-Invariante: Alle drei DRTester-Propensity-Konstruktionen
        der Pipeline tragen den RCT-Zweig mit DummyClassifier(prior). Schlägt
        dieser Test an, wurde eine Stelle entfernt/umgebaut — dann bitte die
        RCT-Prior-Regel dort bewusst mitziehen (oder diesen Test anpassen)."""
        src = open("rubin/pipelines/analysis_pipeline.py", encoding="utf-8").read()
        pattern = re.compile(
            r'study_type", "rct"\) == "rct":\s*\n\s*from sklearn\.dummy import DummyClassifier\s*\n'
            r'\s*model_prop = DummyClassifier\(strategy="prior"\)'
        )
        assert len(pattern.findall(src)) == 3, (
            "Erwartet: 3 RCT-Prior-Zweige für den DRTester (Haupt-Pfad, "
            "MT-/Nuisance-Fallback, External-Fallback)."
        )
