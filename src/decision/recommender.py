"""Recommendation engine (Sec. 4.6, Table 4.5): propose candidates, rank by what-if."""
from .explainer import explain
from .whatif import Action, what_if

SHARES = (0.25, 0.5)
BOOST = 0.3
SIDE_EFFECT_PENALTY = 5.0   # degC-equivalent penalty per rack made riskier


def candidates(rack_id, analysis, twin, thr, reasons):
    fired = {r.rule for r in reasons}
    out = []
    hot_nb = set(twin.neighbours(rack_id))
    if "E1" in fired:
        targets = [rid for rid, a in analysis.assessments.items()
                   if rid != rack_id
                   and a.predicted_peak <= thr["warning_temp"] - thr["safe_margin"]
                   and analysis.feats.loc[rid, "util"] <= thr["spare_util"]]
        targets.sort(key=lambda r: (r in hot_nb, analysis.assessments[r].predicted_peak))
        for t in targets[:2]:
            for sh in SHARES:
                out.append(Action("shift_load", rack_id, target=t, share=sh, rule="A1"))
        if not targets:
            out.append(Action("defer_jobs", rack_id, share=0.3, rule="A3"))
    if fired & {"E2", "E3"}:
        zone = twin.racks[rack_id].zone_id
        units = {twin.unit_for_zone(zone), *twin.units_serving(rack_id)}
        for uid in sorted(units):
            out.append(Action("boost_cooling", rack_id, unit_id=uid, boost=BOOST, rule="A2"))
    return out


def recommend(rack_id, analysis, state, predictor, thr, top=3):
    """Return ranked [(Action, WhatIfReport)] for a risky rack (empty list for Normal)."""
    level = analysis.assessments[rack_id].level
    if level == "Normal":
        return []
    reasons = explain(rack_id, analysis, state.twin, thr)
    ranked = []
    for act in candidates(rack_id, analysis, state.twin, thr, reasons):
        rep, _ = what_if(state, act, predictor, thr, before=analysis)
        row = rep.table[rep.table.rack == rack_id].iloc[0]
        gain = -row["change"] - SIDE_EFFECT_PENALTY * len(rep.newly_riskier)
        if row["change"] < 0:
            act.justification = (f"Expected to lower {rack_id}'s predicted peak by "
                                 f"{-row['change']:.1f} C"
                                 + (f"; but makes {', '.join(rep.newly_riskier)} riskier"
                                    if rep.newly_riskier else ""))
            ranked.append((gain, act, rep))
    ranked.sort(key=lambda x: -x[0])
    if not ranked:
        act = Action("monitor", rack_id, rule="A4",
                     justification="No action tested lowers the forecast; re-check next interval")
        return [(act, what_if(state, act, predictor, thr, before=analysis)[0])]
    return [(a, r) for _, a, r in ranked[:top]]
