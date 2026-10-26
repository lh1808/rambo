"""Synthetische Demo-Daten für den rubin-Smoke-Test (``pixi run analyze-quick``).

Erzeugt X/T/Y.parquet im Format, das ``configs/config_quickstart.yml`` erwartet
(Spalte "T" in T.parquet, Spalte "Y" in Y.parquet, Features rein numerisch).
Die Daten sind ein kleines RCT mit eingebautem heterogenem Treatment-Effekt
(Uplift steigt mit AFFINITAET und fällt mit ALTER), damit der Report etwas
Sinnvolles zeigt. Deterministisch über --seed.

Aufruf:  python scripts/make_demo_data.py [--out data/processed] [--n 4000] [--seed 42]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_data(out_dir: str = "data/processed", n: int = 4000, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "ALTER": rng.integers(18, 80, n).astype(float),
        "BEITRAG": rng.normal(90.0, 25.0, n).round(2),
        "VERTRAEGE": rng.integers(1, 6, n).astype(float),
        "AFFINITAET": rng.random(n).round(4),
        "REGION_NORD": rng.integers(0, 2, n).astype(float),
        "ONLINE_KUNDE": rng.integers(0, 2, n).astype(float),
    })
    T = rng.integers(0, 2, n)  # RCT, 50/50
    base = 0.08 + 0.04 * X["ONLINE_KUNDE"].to_numpy()
    tau = 0.06 * X["AFFINITAET"].to_numpy() - 0.0008 * (X["ALTER"].to_numpy() - 18)
    p = np.clip(base + T * np.clip(tau, -0.02, None), 0.01, 0.95)
    Y = (rng.random(n) < p).astype(int)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X.to_parquet(out / "X.parquet")
    pd.DataFrame({"T": T}).to_parquet(out / "T.parquet")
    pd.DataFrame({"Y": Y}).to_parquet(out / "Y.parquet")
    print(f"Demo-Daten geschrieben: {out}/X.parquet|T.parquet|Y.parquet "
          f"({n} Zeilen, {X.shape[1]} Features, Treatment-Rate {T.mean():.0%}, Y-Rate {Y.mean():.1%})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    make_demo_data(a.out, a.n, a.seed)
