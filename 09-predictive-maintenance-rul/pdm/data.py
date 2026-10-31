"""Run-to-failure simulator for a fleet of turbofan-like engines (C-MAPSS style)."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_SENSORS = 14
N_REGIMES = 3


def simulate_fleet(n_engines: int = 200, seed: int = 13, truncate: bool = False,
                   physics_seed: int = 0) -> pd.DataFrame:
    """Each engine runs until failure. Sensor readings depend on

    * the **operating regime** (altitude / throttle setting) – a large, nuisance shift,
    * a hidden **health index** that degrades exponentially, and
    * engine-specific manufacturing offsets plus noise.

    With ``truncate=True`` each series is cut at a random point before failure, and the
    true remaining useful life (RUL) at the cut is returned – mimicking a test fleet.

    ``physics_seed`` fixes the engine design (regimes, sensor sensitivities), so train and
    test fleets share the same physics while ``seed`` varies the individual engines.
    """
    phys = np.random.default_rng(physics_seed)
    regime_settings = phys.uniform([0, 0.2, 60], [42, 0.85, 100], (N_REGIMES, 3))
    regime_base = phys.normal(0, 6, (N_REGIMES, N_SENSORS))
    sensitivity = phys.normal(0, 1, N_SENSORS) * (phys.random(N_SENSORS) < 0.65)   # ~1/3 sensors uninformative
    noise_sd = phys.uniform(0.15, 0.6, N_SENSORS)
    rng = np.random.default_rng(seed)
    frames = []
    for e in range(n_engines):
        life = int(rng.uniform(140, 360))
        t = np.arange(1, life + 1)
        onset = rng.uniform(0.2, 0.6) * life                     # degradation starts somewhere mid-life
        rate = rng.uniform(3.0, 5.0)
        damage = np.where(t > onset, np.expm1(rate * (t - onset) / (life - onset)) / np.expm1(rate), 0.0)
        regime = rng.integers(0, N_REGIMES, life)
        offsets = rng.normal(0, 0.3, N_SENSORS)
        sensors = (regime_base[regime] + offsets + 3.0 * damage[:, None] * sensitivity
                   + rng.normal(0, 1, (life, N_SENSORS)) * noise_sd)
        df = pd.DataFrame(sensors, columns=[f"s{j+1}" for j in range(N_SENSORS)])
        settings = regime_settings[regime] + rng.normal(0, 0.05, (life, 3))
        df.insert(0, "setting3", settings[:, 2]); df.insert(0, "setting2", settings[:, 1])
        df.insert(0, "setting1", settings[:, 0])
        df.insert(0, "cycle", t); df.insert(0, "engine_id", e)
        df["rul"] = life - t
        if truncate:
            cut = int(rng.uniform(0.3, 0.95) * life)
            df = df.iloc[:cut]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
