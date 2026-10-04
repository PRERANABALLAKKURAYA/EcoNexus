"""What-if simulator (Sec. 4.7): apply an action to a COPY of the state and re-predict."""
from dataclasses import dataclass, field

import pandas as pd

from .hotspot import LEVELS, analyse

ACTION_ROWS = 3   # number of most recent intervals the action is assumed to be in effect


@dataclass
class Action:
    type: str                 # shift_load | boost_cooling | defer_jobs | monitor
    rack_id: str = ""         # the risky (source) rack
    target: str = ""          # shift_load target
    share: float = 0.0        # fraction of the rack's workload moved / deferred
    unit_id: str = ""         # boost_cooling unit
    boost: float = 0.0        # added to the unit's capacity factor (capped at 1.0)
    rule: str = ""
    justification: str = ""

    def describe(self):
        if self.type == "shift_load":
            return f"Shift {self.share:.0%} of {self.rack_id}'s workload to {self.target}"
        if self.type == "boost_cooling":
            return f"Raise cooling unit {self.unit_id} output by {self.boost:.0%}"
        if self.type == "defer_jobs":
            return f"Defer {self.share:.0%} of low-priority jobs on {self.rack_id}"
        return f"Keep monitoring {self.rack_id}"


@dataclass
class WhatIfReport:
    action: Action
    table: pd.DataFrame          # rack, before/after peak and level
    newly_riskier: list = field(default_factory=list)
    peak_change: float = 0.0     # change in the highest predicted temperature
    target_peak_change: float = 0.0


def apply_action(state, action):
    """Return a modified copy; the input state is never changed."""
    s = state.copy()
    w, twin = s.window, s.twin
    last = w.timestamp.isin(sorted(w.timestamp.unique())[-ACTION_ROWS:])

    def set_util(rack, fn):
        m = last & (w.rack_id == rack)
        w.loc[m, "utilisation"] = fn(w.loc[m, "utilisation"])

    if action.type in ("shift_load", "defer_jobs"):
        src = last & (w.rack_id == action.rack_id)
        moved = w.loc[src, "utilisation"].to_numpy() * action.share
        set_util(action.rack_id, lambda u: u * (1 - action.share))
        if action.type == "shift_load":
            tgt = last & (w.rack_id == action.target)
            w.loc[tgt, "utilisation"] = (w.loc[tgt, "utilisation"].to_numpy() + moved).clip(0, 1)
    elif action.type == "boost_cooling":
        u = twin.units[action.unit_id]
        twin.set_cooling_factor(action.unit_id, min(1.0, u.k + action.boost))
        for rid in twin.rack_ids:
            w.loc[last & (w.rack_id == rid), "cooling_effectiveness"] = \
                twin.cooling_effectiveness(rid)
    for rid in twin.rack_ids:                      # recompute derived power
        m = last & (w.rack_id == rid)
        w.loc[m, "power_kw"] = twin.racks[rid].power(w.loc[m, "utilisation"])
    return s


def what_if(state, action, predictor, thr, before=None):
    """`before` = analysis of `state` if the caller already has it (saves one pass)."""
    before = before or analyse(state, predictor, thr)
    after = analyse(apply_action(state, action), predictor, thr)
    rows, worse = [], []
    for rid in state.twin.rack_ids:
        b, a = before.assessments[rid], after.assessments[rid]
        rows.append({"rack": rid, "temp_now": round(b.current_temp, 1),
                     "peak_before": round(b.predicted_peak, 1), "level_before": b.level,
                     "peak_after": round(a.predicted_peak, 1), "level_after": a.level,
                     "change": round(a.predicted_peak - b.predicted_peak, 1),
                     "end_before": round(float(before.forecasts.loc[rid].iloc[-1]), 1),
                     "end_after": round(float(after.forecasts.loc[rid].iloc[-1]), 1)})
        if LEVELS[a.level] > LEVELS[b.level]:
            worse.append(rid)
    table = pd.DataFrame(rows)
    tgt = table[table.rack == action.target]["change"]
    return WhatIfReport(action, table, worse,
                        round(table.peak_after.max() - table.peak_before.max(), 1),
                        float(tgt.iloc[0]) if len(tgt) else 0.0), after
