"""Pre-computes the 'live' simulation: one frame per interval holding the real model output
(temperatures, forecasts, risk, explanations, tested recommendations). The browser plays the
frames back, so the twin moves smoothly without Streamlit reruns. Cached in data/playback.json."""
import hashlib
import json

import pandas as pd

from src.decision.explainer import explain, explanation_text
from src.decision.hotspot import LEVELS, analyse
from src.decision.recommender import recommend
from src.pipeline import WINDOW

LEAD_IN = 50       # intervals shown before the most interesting moment
MAX_FRAMES = 170   # ~14 simulated hours at 5-minute intervals
HISTORY = 36


def window_bounds(S):
    times = sorted(S.telemetry.timestamp.unique())
    lo = times.index(S.test_start) + WINDOW
    start = max(lo, times.index(S.default_time) - LEAD_IN)
    return times, start, min(MAX_FRAMES, len(times) - start)


def _signature(S):
    h = hashlib.sha1(b"playback-v3")
    for f in ("config/twin.yaml", "config/sim.yaml", "config/thresholds.yaml"):
        h.update((S.root / f).read_bytes())
    h.update(str((S.root / "models/predictor.joblib").stat().st_size).encode())
    h.update(str(len(S.telemetry)).encode())
    return h.hexdigest()[:16]


def build_frames(S, progress=None):
    times, start, n = window_bounds(S)
    twin_ids = S.twin.rack_ids
    tel = S.telemetry
    frames = []
    for j in range(n):
        ts = pd.Timestamp(times[start + j])
        st = S.state_at(ts)
        an = analyse(st, S.predictor, S.thr)
        racks = []
        for rid in twin_ids:
            a, f = an.assessments[rid], an.feats.loc[rid]
            risky = a.level != "Normal"
            rs = explain(rid, an, st.twin, S.thr) if risky else []
            recs = []
            for act, rep in (recommend(rid, an, st, S.predictor, S.thr) if risky else []):
                t = rep.table
                row = t[t.rack == rid].iloc[0]
                recs.append([act.describe(), act.rule, act.justification, float(row["change"]),
                             round(float(row["end_after"] - row["end_before"]), 1),
                             list(rep.newly_riskier), act.type,
                             [float(v) for v in t.peak_after], [LEVELS[v] for v in t.level_after]])
            racks.append({
                "T": round(a.current_temp, 1), "P": round(a.predicted_peak, 1),
                "F": [round(float(v), 1) for v in an.forecasts.loc[rid]],
                "L": LEVELS[a.level], "S": round(a.score, 2), "SL": round(a.slope, 2),
                "U": round(float(f["util"]), 2), "UR": round(float(f["util_roll"]), 2),
                "C": round(a.cooling_eff, 2), "N": round(float(f["nb_mean"]), 1),
                "W": explanation_text(rid, rs, a.level),
                "RS": [[r.rule, r.text, r.values] for r in rs], "RC": recs,
                "BP": round(a.predicted_peak, 1)})
        frames.append({"t": f"{ts:%Y-%m-%dT%H:%M}", "k": [round(u.k, 2) for u in st.twin.units.values()],
                       "r": racks})
        if progress:
            progress((j + 1) / n)
    hist0 = {}
    for rid in twin_ids:
        s = tel[(tel.rack_id == rid) & (tel.timestamp < times[start])].temperature_c.tail(HISTORY)
        hist0[rid] = [round(float(v), 1) for v in s]
    twin = S.twin
    static = {
        "racks": [{"id": r, "zone": twin.racks[r].zone_id, "pos": list(twin.racks[r].position)}
                  for r in twin_ids],
        "units": [{"id": u.unit_id, "zone": u.zone_id, "serves": u.serves}
                  for u in twin.units.values()],
        "links": [[a, b, d["weight"]] for a, b, d in twin.graph.edges(data=True)
                  if d["kind"] == "coupling"],
        "thr": {"warn": S.thr["warning_temp"], "crit": S.thr["critical_temp"]},
        "horizon_min": int(S.sim["interval_minutes"]) * int(S.sim["horizon"]),
        "interval_min": int(S.sim["interval_minutes"]), "hist0": hist0}
    return {**static, "frames": frames}


def load_frames(S, progress=None):
    path = S.root / "data/playback.json"
    sig = _signature(S)
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("sig") == sig:
            return cached
    data = build_frames(S, progress)
    data["sig"] = sig
    path.write_text(json.dumps(data, separators=(",", ":")))
    return data
