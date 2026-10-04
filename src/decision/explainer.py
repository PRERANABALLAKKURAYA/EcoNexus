"""Rule-based explanation engine (Sec. 4.5, Table 4.4). Every message cites rule + values."""
from dataclasses import dataclass


@dataclass
class Reason:
    rule: str
    text: str
    values: str
    contribution: float


def explain(rack_id, analysis, twin, thr):
    f, a = analysis.feats.loc[rack_id], analysis.assessments[rack_id]
    w1, w2, w3, w4 = thr["risk_weights"]
    reasons = []
    if f["util_roll"] >= thr["high_load"]:
        reasons.append(Reason("E1", "high workload for a long time",
                              f"mean utilisation {f['util_roll']:.2f} >= {thr['high_load']}",
                              w2 * f["util_roll"]))
    if a.cooling_eff <= thr["cooling_deficit"]:
        reasons.append(Reason("E2", "insufficient cooling in its zone",
                              f"cooling effectiveness {a.cooling_eff:.2f} <= {thr['cooling_deficit']}",
                              w3 * (1 - a.cooling_eff)))
    if f["nb_mean"] >= thr["warning_temp"]:
        hot = [n for n in twin.neighbours(rack_id)
               if analysis.feats.loc[n, "temp"] >= thr["warning_temp"]]
        who = ", ".join(hot) if hot else "adjacent racks"
        reasons.append(Reason("E3", f"heat spilling over from adjacent racks {who}",
                              f"weighted neighbour temp {f['nb_mean']:.1f} C >= {thr['warning_temp']}",
                              w4 * min(1, f["nb_mean"] / thr["critical_temp"])))
    if a.slope >= thr["rising_slope"]:
        reasons.append(Reason("E4", f"temperature rising quickly (+{a.slope:.1f} C per interval)",
                              f"slope {a.slope:.2f} >= {thr['rising_slope']}", w1 * 0.5))
    if a.current_temp >= thr["warning_temp"]:
        reasons.append(Reason("E5", "already above the warning level",
                              f"{a.current_temp:.1f} C >= {thr['warning_temp']}", w1 * 0.5))
    reasons.sort(key=lambda r: -r.contribution)
    return reasons


def explanation_text(rack_id, reasons, level):
    if level == "Normal":
        return f"{rack_id} is operating normally."
    if not reasons:
        return f"{rack_id} is at risk but no specific cause was identified; keep monitoring."
    top = [r.text for r in reasons[:3]]
    joined = top[0] if len(top) == 1 else "; ".join(top[:-1]) + "; and " + top[-1]
    verb = "may overheat" if level == "Potential hotspot" else "is showing rising thermal risk"
    return f"{rack_id} {verb}: {joined}."
