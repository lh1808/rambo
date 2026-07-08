"""Tests für die Treatment-Balance-Auswahl (_check_and_balance_treatments).

Kernverhalten: Die Ziel-Rate wird nach MAXIMALER EFFEKTIVER STICHPROBE
N_keep · p·(1−p) gewählt — nicht nach minimalem Zeilenverlust. Dadurch kann
eine balanciertere Rate gewinnen, obwohl sie mehr Zeilen verliert.
"""
import logging
import types

import pandas as pd
import pytest

from rubin.pipelines.data_prep_pipeline import DataPrepPipeline


def _make_pipeline():
    inst = object.__new__(DataPrepPipeline)
    inst._logger = logging.getLogger("rubin.test.balance")
    inst.cfg = types.SimpleNamespace(constants=types.SimpleNamespace(random_seed=42))
    return inst


def _make_df(rates_by_file, n_per_file=1000):
    parts = []
    for fid, rate in rates_by_file:
        nt = int(rate * n_per_file)
        parts.append(pd.DataFrame({
            "T": [1] * nt + [0] * (n_per_file - nt),
            "__file_source__": fid,
        }))
    return pd.concat(parts, ignore_index=True)


def _dp(**kw):
    base = dict(
        balance_treatments=True,
        balance_min_arm_abs=0,
        balance_min_arm_frac=0.0,
        balance_max_loss_frac=1.0,
        balance_target_grid_step=0.0,
        log_to_mlflow=False,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


# Kanonisches Szenario: min-loss und effektive N divergieren.
# Ziel 50%: 20% Verlust, effektive N = 3200·0.25 = 800
# Ziel 67%: 10.5% Verlust, effektive N ≈ 3581·0.221 ≈ 792
# → min-loss würde 67% wählen, effektive N wählt 50%.
_CANONICAL = [(0, 0.61), (1, 0.62), (2, 0.67), (3, 0.50)]


def test_effective_n_prefers_balance_over_fewest_rows():
    inst = _make_pipeline()
    out = inst._check_and_balance_treatments(_make_df(_CANONICAL), "T", _dp(), lambda *a, **k: None)
    rate = (out["T"] == 1).mean()
    # Effektive N wählt 50% (balancierter), NICHT 67% (weniger Verlust).
    assert abs(rate - 0.50) < 0.02, f"erwartet ~50%, erhalten {rate:.3f}"
    assert rate < 0.60, "darf nicht die unbalancierte 67%-Variante wählen"


def test_single_file_no_balancing():
    inst = _make_pipeline()
    df = _make_df([(0, 0.61)])
    out = inst._check_and_balance_treatments(df, "T", _dp(), lambda *a, **k: None)
    # Eine Datei → keine Balance, nichts entfernt.
    assert len(out) == len(df)


def test_balanced_files_untouched():
    inst = _make_pipeline()
    # Differenz < 5pp → kein Downsampling.
    df = _make_df([(0, 0.50), (1, 0.52)])
    out = inst._check_and_balance_treatments(df, "T", _dp(), lambda *a, **k: None)
    assert len(out) == len(df)


def test_arm_floor_too_high_warns_but_returns(caplog):
    inst = _make_pipeline()
    with caplog.at_level(logging.WARNING, logger="rubin.test.balance"):
        out = inst._check_and_balance_treatments(
            _make_df(_CANONICAL), "T", _dp(balance_min_arm_abs=10_000), lambda *a, **k: None
        )
    # Kein Kandidat erfüllt den Floor → Warnung, aber Ergebnis trotzdem geliefert.
    assert len(out) > 0
    assert any("Arm-Floor" in r.message for r in caplog.records)


def test_grid_can_beat_both_endpoints():
    inst = _make_pipeline()
    out = inst._check_and_balance_treatments(
        _make_df(_CANONICAL), "T", _dp(balance_target_grid_step=0.05), lambda *a, **k: None
    )
    rate = (out["T"] == 1).mean()
    # Gitter darf eine Zwischenrate Richtung 0.5 wählen (höhere effektive N).
    assert 0.49 <= rate <= 0.68
    # Konkret in diesem Szenario: 60% hat die höchste effektive N (859) → weniger Verlust als 50%.
    assert len(out) > 3200, "Gitter-Optimum sollte weniger Zeilen verlieren als das 50%-Ziel"


@pytest.mark.parametrize("rates,grid", [
    ([(0, 0.0), (1, 1.0)], 0.0),    # reine Control- + reine Treatment-Datei
    ([(0, 0.0), (1, 0.8)], 0.0),    # reine Control-Datei + 80%-Datei
    ([(0, 0.0), (1, 0.9)], 0.1),    # mit Gitter
])
def test_degenerate_pure_arm_files_no_crash(rates, grid):
    # Zielraten 0/1 würden im Downsampling durch 0 dividieren — müssen ausgeschlossen
    # werden (Fallback 0.5). Pure-Arm-Dateien können nicht balanciert werden → passieren
    # unverändert durch, ohne Crash.
    inst = _make_pipeline()
    out = inst._check_and_balance_treatments(
        _make_df(rates), "T", _dp(balance_target_grid_step=grid), lambda *a, **k: None
    )
    assert len(out) > 0


def test_max_loss_frac_zero_is_respected(caplog):
    # balance_max_loss_frac=0.0 (strengster Deckel) ist falsy — ein `or 1.0`
    # würde die 0.0 stillschweigend zu 1.0 machen (Deckel aus). 0.0 muss als
    # expliziter Wert wirken: jeder Kandidat verliert Zeilen → kein feasibler
    # → Verlust-Deckel-Warnung.
    inst = _make_pipeline()
    with caplog.at_level(logging.WARNING, logger="rubin.test.balance"):
        out = inst._check_and_balance_treatments(
            _make_df(_CANONICAL), "T", _dp(balance_max_loss_frac=0.0), lambda *a, **k: None
        )
    assert len(out) > 0
    assert any("Verlust-Deckel" in r.message for r in caplog.records)


def test_max_loss_frac_caps_candidates(caplog):
    # Funktional: Deckel 0.12 disqualifiziert das eff-N-Optimum 50% (20% Verlust) und
    # erzwingt 67% (10.5% Verlust) — obwohl 50% die höhere effektive N hätte. Da ein
    # feasibler Kandidat existiert, KEINE Deckel-Warnung.
    inst = _make_pipeline()
    with caplog.at_level(logging.WARNING, logger="rubin.test.balance"):
        out = inst._check_and_balance_treatments(
            _make_df(_CANONICAL), "T", _dp(balance_max_loss_frac=0.12), lambda *a, **k: None
        )
    rate = (out["T"] == 1).mean()
    assert abs(rate - 0.67) < 0.02, f"Deckel sollte 67% erzwingen, Rate={rate:.3f}"
    assert not any("Verlust-Deckel" in r.message for r in caplog.records)
