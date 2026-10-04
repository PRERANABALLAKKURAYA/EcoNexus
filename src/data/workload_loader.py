"""Workload loader (Sec. 4.1.1): public workload pattern -> utilisation series in [0, 1].

Drop a real extract of a public trace (e.g. the Google cluster-usage traces) at
data/workload_sample.csv with a numeric column (preferably `utilisation`). If the
file is missing, a trace-like pattern (daily cycle + bursts + noise) is generated
and saved there so the project runs offline.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def synthetic_trace(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    daily = 0.5 + 0.22 * np.sin(2 * np.pi * (t / 288 - 0.3)) + 0.06 * np.sin(2 * np.pi * t / 96)
    noise = pd.Series(rng.normal(0, 0.05, n)).rolling(3, min_periods=1).mean().to_numpy()
    bursts = np.zeros(n)
    for s in rng.integers(0, n - 6, 25):
        bursts[s:s + rng.integers(2, 6)] += rng.uniform(0.1, 0.25)
    return pd.DataFrame({"utilisation": np.clip(daily + noise + bursts, 0.05, 0.98)})


def resample_to(series, n):
    """Linear resampling of a 1-D array to n points."""
    x_old = np.linspace(0, 1, len(series))
    return np.interp(np.linspace(0, 1, n), x_old, series)


def load_workload(path, steps):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_trace().to_csv(path, index=False)
    df = pd.read_csv(path)
    num = df.select_dtypes("number")
    if num.empty:
        raise ValueError(f"{path}: no numeric column found for utilisation")
    col = "utilisation" if "utilisation" in num else num.columns[0]
    s = num[col].to_numpy(dtype=float)
    if s.max() > 1.0 or s.min() < 0.0:
        s = (s - s.min()) / max(s.max() - s.min(), 1e-9)
    return np.clip(resample_to(s, steps), 0.0, 1.0)
