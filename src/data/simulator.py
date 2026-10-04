"""Thermal simulator (Sec. 4.1.2-4.1.3): utilisation -> rack temperatures (Eq. 4.1-4.3)."""
import numpy as np
import pandas as pd


def simulate(twin, workload, sim):
    """Return the telemetry table of Table 3.2 (+ cooling factors and scenario label)."""
    rng = np.random.default_rng(sim["seed"])
    n = int(sim["steps"])
    ids = twin.rack_ids
    R = len(ids)

    # per-rack utilisation: scaled, time-shifted base pattern + individual variation
    util = np.zeros((n, R))
    for j in range(R):
        scale = rng.uniform(0.75, 1.0)
        shift = int(rng.integers(0, 120))
        var = pd.Series(rng.normal(0, 0.04, n)).rolling(4, min_periods=1).mean().to_numpy()
        util[:, j] = np.roll(workload, shift) * scale + var
    scenario = np.full((n, R), "normal", dtype=object)

    # short bursts: single-interval spikes that must NOT trigger alarms
    for _ in range(int(sim["n_bursts"])):
        t, j = int(rng.integers(5, n - 5)), int(rng.integers(0, R))
        util[t, j], scenario[t, j] = 1.0, "burst"
    # sustained surges
    for ev in sim["surges"]:
        if ev["start"] >= n:          # event lies beyond the simulated period
            continue
        j = ids.index(ev["rack"])
        s, e = ev["start"], min(n, ev["start"] + ev["length"])
        util[s:e, j] = ev["level"] + rng.normal(0, 0.01, e - s)
        scenario[s:e, j] = "surge"
    util = np.clip(util, 0.0, 1.0)

    # cooling capacity factor over time (1.0 healthy)
    units = list(twin.units)
    k = np.ones((n, len(units)))
    for ev in sim["cooling_faults"]:
        if ev["start"] >= n:
            continue
        k[ev["start"]:ev["start"] + ev["length"], units.index(ev["unit"])] = ev["factor"]
        for j, rid in enumerate(ids):
            if ev["unit"] in twin.units_serving(rid):
                seg = scenario[ev["start"]:ev["start"] + ev["length"], j]
                seg[seg == "normal"] = "cooling_fault"

    W = twin.coupling_matrix()
    p_idle = np.array([twin.racks[r].p_idle for r in ids])
    p_max = np.array([twin.racks[r].p_max for r in ids])
    t_sup = np.array([twin.supply_temp(r) for r in ids])
    power = p_idle + (p_max - p_idle) * util
    c = np.array([[twin.cooling_effectiveness(r, dict(zip(units, k[t]))) for r in ids]
                  for t in range(n)])

    a, b, g = sim["alpha"], sim["beta"], sim["gamma"]
    T = np.zeros((n, R))
    T[0] = t_sup + a * power[0] / (b * c[0])                       # steady-state start
    for t in range(n - 1):                                         # Eq. 4.2
        exchange = W @ T[t] - W.sum(1) * T[t]                      # sum_j w_ij (T_j - T_i)
        T[t + 1] = (T[t] + a * power[t] - b * c[t] * (T[t] - t_sup) + g * exchange
                    + rng.normal(0, sim["sigma"], R))
        T[t + 1] = np.clip(T[t + 1], t_sup, sim["t_max"])

    ts = pd.date_range("2026-01-01", periods=n, freq=f"{int(sim['interval_minutes'])}min")
    rows = pd.DataFrame({
        "timestamp": np.repeat(ts, R),
        "rack_id": np.tile(ids, n),
        "zone_id": np.tile([twin.racks[r].zone_id for r in ids], n),
        "utilisation": util.ravel(),
        "power_kw": power.ravel(),
        "temperature_c": T.ravel(),
        "cooling_effectiveness": c.ravel(),
        "scenario": scenario.ravel(),
    })
    for i, u in enumerate(units):
        rows[f"k_{u}"] = np.repeat(k[:, i], R)
    return rows
