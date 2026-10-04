"""Hotspot detection (Sec. 4.4): forecasts -> risk level + score (Table 4.3, Eq. 4.6)."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.ml.features import build_features

LEVELS = {"Normal": 0, "Rising thermal risk": 1, "Potential hotspot": 2}
COLOURS = {"Normal": "#2e9e5b", "Rising thermal risk": "#e8a317", "Potential hotspot": "#d64545"}


@dataclass
class RiskAssessment:
    rack_id: str
    level: str
    score: float
    predicted_peak: float
    slope: float
    current_temp: float
    cooling_eff: float
    rules: list = field(default_factory=list)   # classification rule ids that fired


@dataclass
class Analysis:
    feats: pd.DataFrame        # latest feature row per rack (index rack_id)
    forecasts: pd.DataFrame    # rack_id x h1..hH
    assessments: dict          # rack_id -> RiskAssessment


def risk_score(peak, util_mean, c, nb_mean, thr, t_sup=18.0):
    """Eq. 4.6: R = w1*M + w2*S + w3*(1-c) + w4*N, in [0, 1]."""
    w1, w2, w3, w4 = thr["risk_weights"]
    m = np.clip((peak - thr["warning_temp"]) / (thr["critical_temp"] - thr["warning_temp"]), 0, 1)
    n = np.clip((nb_mean - t_sup) / (thr["critical_temp"] - t_sup), 0, 1)
    return float(w1 * m + w2 * np.clip(util_mean, 0, 1) + w3 * (1 - c) + w4 * n)


def classify(feats, forecasts, twin, thr):
    """Classify every rack. `feats` is indexed by rack_id."""
    warn, crit = thr["warning_temp"], thr["critical_temp"]
    out = {}
    for rid in feats.index:
        f = feats.loc[rid]
        fc = forecasts.loc[rid].to_numpy()
        peak = float(fc.max())
        slope = float((fc[-1] - f["temp"]) / len(fc))
        level, rules = "Normal", []
        if peak >= crit or f["temp"] >= crit:
            level, rules = "Potential hotspot", ["H1 peak>=critical"]
        else:
            if peak >= warn:
                rules.append("H2 peak>=warning")
            if f["temp"] >= warn:
                rules.append("H5 already above warning")
            if slope >= thr["rising_slope"] and f["temp"] >= warn - thr["warning_margin"]:
                rules.append("H3 fast rise near warning")
            if rules:
                level = "Rising thermal risk"
        out[rid] = RiskAssessment(rid, level, 0.0, peak, slope, float(f["temp"]),
                                  float(f["cooling_eff"]), rules)
    # twin rule: weak cooling next to a potential hotspot -> amber (second pass)
    for rid, a in out.items():
        if a.level == "Normal" and a.cooling_eff <= thr["cooling_deficit"]:
            hot_nb = [n for n in twin.neighbours(rid)
                      if n in out and out[n].level == "Potential hotspot"]
            if hot_nb:
                a.level, a.rules = "Rising thermal risk", [f"H4 weak cooling beside {','.join(hot_nb)}"]
    for rid, a in out.items():
        f = feats.loc[rid]
        a.score = risk_score(a.predicted_peak, f["util_roll"], a.cooling_eff, f["nb_mean"],
                             thr, twin.supply_temp(rid))
    return out


def analyse(state, predictor, twin_thr):
    """Features -> forecasts -> risk for the latest timestamp of a SystemState."""
    thr = twin_thr
    feats = build_features(state.window, state.twin, thr)
    feats = feats[feats.timestamp == feats.timestamp.max()].set_index("rack_id")
    feats = feats.loc[state.twin.rack_ids]
    fc = predictor.predict(feats.reset_index())
    return Analysis(feats, fc, classify(feats, fc, state.twin, thr))
