"""Wellen-Design des Optuna-Tunings: Die Anzahl paralleler Trials ist hart
gecappt (Level 3: 4; Level 4: 8 LightGBM / 6 CatBoost), damit der
TPE-Sampler sequentielle Wellen mit abgeschlossenen Ergebnissen zum Lernen
hat. Gilt für das BLT (study.optimize mit n_jobs=Wellen-Breite); FMT und
CFT tunen sequentiell (n_jobs=1) und brauchen keinen Cap. Realer Vorfall: 250 Kerne ohne Cap → 62 parallele Trials bei
n_trials=50 — alle Trials starteten gleichzeitig, das Tuning war reine
Zufallssuche und die Modellgüte brach messbar ein."""
from rubin.tuning.base_learner import compute_tuning_n_jobs


class TestTuningParallelCaps:
    def test_large_machine_is_capped_to_waves(self):
        # 250 Kerne (der Vorfall): statt 62 → 4/8/6 parallele Trials
        assert compute_tuning_n_jobs(3, 250, 4) == 4
        assert compute_tuning_n_jobs(4, 250, 4, "lgbm") == 8
        assert compute_tuning_n_jobs(4, 250, 4, "catboost") == 6
        assert compute_tuning_n_jobs(4, 250, 4, "both") == 6   # enthält CatBoost-Studies

    def test_sequential_levels_stay_sequential(self):
        assert compute_tuning_n_jobs(1, 250, 4) == 1
        assert compute_tuning_n_jobs(2, 250, 4) == 1

    def test_small_machines_keep_formula(self):
        assert compute_tuning_n_jobs(4, 16, 4) == 4            # 16//4 < Cap
        assert compute_tuning_n_jobs(4, 4, 4) == 1             # nie unter 1

    def test_more_cores_never_hurt(self):
        """Die Kern-Garantie: Mit steigender Kernzahl bleibt die Trial-
        Parallelität ab dem Cap konstant (Tuning-Qualität identisch), während
        die Kerne pro Trial-Fit monoton wachsen (nur schneller). Vor dem Cap
        wuchs die Parallelität unbegrenzt mit — mehr Kerne machten die
        Ergebnisse SCHLECHTER."""
        prev_per_fit = 0
        for cpus in (8, 16, 32, 64, 128, 250):
            pj = compute_tuning_n_jobs(4, cpus, 4, "lgbm")
            assert pj <= 8                          # Qualität: nie über den Cap
            per_fit = cpus // pj
            assert per_fit >= prev_per_fit          # Speed: Kerne/Fit monoton
            prev_per_fit = per_fit
        assert compute_tuning_n_jobs(4, 64, 4, "lgbm") == compute_tuning_n_jobs(4, 250, 4, "lgbm")

    def test_waves_exist_within_default_budget(self):
        # Kern-Invariante: Bei n_trials=50 gibt es mit jedem Cap ≥ 5 Wellen —
        # TPE sieht ab Welle 2 abgeschlossene Trials.
        for pj in (compute_tuning_n_jobs(4, 250, 4, "lgbm"),
                   compute_tuning_n_jobs(4, 250, 4, "catboost"),
                   compute_tuning_n_jobs(3, 250, 4)):
            assert 50 // pj >= 5
