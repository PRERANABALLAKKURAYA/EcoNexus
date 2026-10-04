"""Feature engineering (Table 4.2): own temperature, workload, cooling, neighbours, time."""
import numpy as np
import pandas as pd

FEATURES = ["temp", "temp_lag1", "temp_lag2", "temp_lag3", "temp_roll", "temp_slope",
            "util", "util_lag1", "util_lag2", "util_roll", "power", "cooling_eff",
            "supply_temp", "nb_mean", "nb_max", "nb_hot", "tod_sin", "tod_cos"]
CONTEXT = ["rack_id", "timestamp", "zone_id"]


def build_features(tel, twin, thr, horizon=0, k=None):
    """One row per (rack, timestamp). If horizon > 0, adds targets y_h{1..H} as the
    temperature *change* over h intervals (delta target; Sec. 4.3 implementation note)."""
    k = k or int(thr["window_k"])
    ids = twin.rack_ids
    tel = tel.sort_values(["rack_id", "timestamp"])
    T = tel.pivot(index="timestamp", columns="rack_id", values="temperature_c")[ids]
    W = twin.coupling_matrix()
    adj = (W > 0).astype(float)
    nb_mean = (T.to_numpy() @ W.T) / W.sum(1)
    hot = (T.to_numpy() >= thr["warning_temp"]).astype(float)
    nb_hot = hot @ adj.T
    nb_max = np.stack([np.where(adj[i] > 0, T.to_numpy(), -np.inf).max(1)
                       for i in range(len(ids))], axis=1)
    nb = {name: pd.DataFrame(arr, index=T.index, columns=ids).stack().rename(name)
          for name, arr in [("nb_mean", nb_mean), ("nb_max", nb_max), ("nb_hot", nb_hot)]}
    nbdf = pd.concat(nb.values(), axis=1).rename_axis(["timestamp", "rack_id"]).reset_index()

    parts = []
    for rid, g in tel.groupby("rack_id", sort=False):
        g = g.sort_values("timestamp")
        f = pd.DataFrame({"rack_id": rid, "timestamp": g["timestamp"].values,
                          "zone_id": g["zone_id"].values})
        t, u = g["temperature_c"].reset_index(drop=True), g["utilisation"].reset_index(drop=True)
        f["temp"], f["temp_lag1"] = t.values, t.shift(1).values
        f["temp_lag2"], f["temp_lag3"] = t.shift(2).values, t.shift(3).values
        f["temp_roll"] = t.rolling(k).mean().values
        f["temp_slope"] = ((t - t.shift(3)) / 3).values
        f["util"], f["util_lag1"], f["util_lag2"] = u.values, u.shift(1).values, u.shift(2).values
        f["util_roll"] = u.rolling(k).mean().values
        f["power"] = g["power_kw"].values
        f["cooling_eff"] = g["cooling_effectiveness"].values
        f["supply_temp"] = twin.supply_temp(rid)
        for h in range(1, horizon + 1):
            f[f"y_h{h}"] = (t.shift(-h) - t).values
        parts.append(f)
    df = pd.concat(parts, ignore_index=True).merge(nbdf, on=["timestamp", "rack_id"])
    tod = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute) / 1440.0
    df["tod_sin"], df["tod_cos"] = np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)
    need = FEATURES + [f"y_h{h}" for h in range(1, horizon + 1)]
    return df.dropna(subset=need).sort_values(["timestamp", "rack_id"]).reset_index(drop=True)
