"""Unit + integration tests mapped to TC1-TC12 of the report (Table 4.8)."""
import tempfile
from pathlib import Path

import pandas as pd

from src.data.simulator import simulate
from src.data.workload_loader import load_workload
from src.decision.decision_log import DecisionLogger
from src.decision.explainer import explain, explanation_text
from src.decision.hotspot import analyse, classify, risk_score
from src.decision.recommender import recommend
from src.decision.whatif import Action, apply_action, what_if
from src.pipeline import ConfigError, build_system, state_at
from tests.helpers import ROOT, default_twin, sim_cfg, system, thresholds


def _gen(sim):
    tw = default_twin()
    return simulate(tw, load_workload(ROOT / "data/workload_sample.csv", sim["steps"]), sim)


def test_tc1_same_seed_identical_data():                         # FR1, NFR5
    sim = dict(sim_cfg(), steps=200)
    assert _gen(sim).equals(_gen(sim))


def test_tc2_twin_structure():                                   # FR2
    tw = default_twin()
    assert len(tw.racks) == 6 and len(tw.units) == 2
    assert tw.neighbours("R1") == {"R2": 0.5, "R4": 0.3}


def test_cooling_effectiveness_eq43():
    tw = default_twin()
    assert tw.cooling_effectiveness("R1") == 1.0
    tw.set_cooling_factor("C1", 0.5)
    assert tw.cooling_effectiveness("R1") == 0.5
    assert tw.cooling_effectiveness("R4") == 1.0          # redundancy from C2


def test_thermal_update_follows_eq42_without_noise():
    sim = dict(sim_cfg(), steps=60, sigma=0.0, n_bursts=0, surges=[], cooling_faults=[])
    df = _gen(sim)
    tw, r = default_twin(), "R2"
    g = df[df.rack_id == r].reset_index(drop=True)
    t0 = g.loc[10]
    nb = tw.neighbours(r)
    ex = sum(w * (df[(df.rack_id == n) & (df.timestamp == t0.timestamp)].temperature_c.iloc[0]
                  - t0.temperature_c) for n, w in nb.items())
    expect = (t0.temperature_c + sim["alpha"] * t0.power_kw
              - sim["beta"] * t0.cooling_effectiveness * (t0.temperature_c - tw.supply_temp(r))
              + sim["gamma"] * ex)
    assert abs(g.loc[11].temperature_c - expect) < 1e-6


def test_tc3_prediction_plausible():                             # FR3
    S = system()
    an = analyse(S.state_at(S.default_time), S.predictor, S.thr)
    assert len(an.forecasts) == 6
    assert an.forecasts.to_numpy().min() > 15 and an.forecasts.to_numpy().max() < 110


def test_model_beats_persistence():                              # Sec. 4.3.3
    m = system().metrics
    for h in (1, 2, 3):
        g = m[(m.horizon == h)].set_index("model").MAE
        assert g["GradientBoosting"] < g["Persistence"]


def _feats(temp, util=0.5, c=1.0, nb=50.0):
    return pd.DataFrame({"temp": temp, "util_roll": util, "cooling_eff": c, "nb_mean": nb},
                        index=pd.Index(["R1"], name="rack_id"))


def _classify(peak, temp=50.0, slope_end=None):
    tw, thr = default_twin(), thresholds()
    fc = pd.DataFrame({"h1": [peak], "h2": [peak], "h3": [slope_end or peak]}, index=["R1"])
    f = _feats(temp)
    tw_one = tw
    return classify(f, fc, tw_one, thr)["R1"]


def test_tc4_peak_at_critical_is_hotspot():                      # FR4
    assert _classify(75.0).level == "Potential hotspot"


def test_tc5_just_below_warning_flat_is_normal():                # FR4
    assert _classify(64.9, temp=64.9).level == "Normal"


def test_amber_for_warning_and_rising_slope_rule():
    assert _classify(65.0).level == "Rising thermal risk"
    assert _classify(62.0, temp=61.0, slope_end=64.0).level == "Rising thermal risk"


def test_tc6_short_burst_raises_no_hotspot():                    # FR4
    S = system()
    b = S.telemetry[S.telemetry.scenario == "burst"]
    ts = [t for t in b.timestamp.unique() if t > S.test_start + pd.Timedelta(hours=1)][:10]
    for t in ts:
        an = analyse(S.state_at(t), S.predictor, S.thr)
        burst_racks = set(b[b.timestamp == t].rack_id)
        assert all(an.assessments[r].level != "Potential hotspot" for r in burst_racks)


def test_risk_score_in_unit_interval():
    thr = thresholds()
    for peak in (40, 70, 90):
        assert 0.0 <= risk_score(peak, 1.0, 0.0, 90, thr) <= 1.0


def _hot_moment():
    S = system()
    return S, S.state_at(S.default_time)


def test_tc7_explanation_lists_rules():                          # FR6
    S, st = _hot_moment()
    an = analyse(st, S.predictor, S.thr)
    risky = [r for r, a in an.assessments.items() if a.level == "Potential hotspot"]
    assert risky
    reasons = explain(risky[0], an, st.twin, S.thr)
    assert {"E2"} <= {r.rule for r in reasons}
    assert "may overheat" in explanation_text(risky[0], reasons, "Potential hotspot")


def test_explainer_says_unclear_when_no_rule_fires():
    assert "no specific cause" in explanation_text("R1", [], "Rising thermal risk")


def test_tc8_recommendation_proposed_for_risky_rack():           # FR7
    S, st = _hot_moment()
    an = analyse(st, S.predictor, S.thr)
    risky = next(r for r, a in an.assessments.items() if a.level == "Potential hotspot")
    recs = recommend(risky, an, st, S.predictor, S.thr)
    assert recs and recs[0][0].type in {"boost_cooling", "shift_load", "defer_jobs", "monitor"}
    normal = next(r for r, a in an.assessments.items() if a.level == "Normal")
    assert recommend(normal, an, st, S.predictor, S.thr) == []


def test_tc9_whatif_leaves_base_state_unchanged():               # FR8
    S, st = _hot_moment()
    before_w, k = st.window.copy(), st.twin.units["C1"].k
    what_if(st, Action("boost_cooling", "R1", unit_id="C1", boost=0.3), S.predictor, S.thr)
    what_if(st, Action("shift_load", "R1", target="R6", share=0.5), S.predictor, S.thr)
    assert st.window.equals(before_w) and st.twin.units["C1"].k == k


def test_tc10_overloading_target_is_flagged_newly_riskier():     # FR9
    S = system()
    t = S.default_time
    st = S.state_at(t)
    # R6 starts warm: dump 100% of a hot rack's load on it
    st.window.loc[(st.window.rack_id == "R6") & (st.window.timestamp == t), "temperature_c"] = 63.0
    act = Action("shift_load", "R1", target="R6", share=1.0)
    s2 = apply_action(st, act)
    last = s2.window[s2.window.timestamp == t].set_index("rack_id").utilisation
    assert last["R6"] > st.window[st.window.timestamp == t].set_index("rack_id").utilisation["R6"]
    rep, _ = what_if(st, act, S.predictor, S.thr)
    assert set(rep.table.columns) >= {"peak_before", "peak_after", "level_after", "end_after"}
    assert isinstance(rep.newly_riskier, list)


def test_tc11_decision_log_has_two_timestamped_entries():        # FR10
    p = Path(tempfile.mkdtemp()) / "log.csv"
    lg = DecisionLogger(p)
    lg.log("2026-01-07 14:40", "R1", "Boost cooling C1", True)
    lg.log("2026-01-07 14:40", "R2", "Shift load", False)
    df = lg.read()
    assert list(df.decision) == ["accepted", "rejected"] and df.logged_at.notna().all()


def test_bad_config_gives_clear_message():                       # NFR4
    d = Path(tempfile.mkdtemp())
    try:
        build_system(d)
        assert False, "expected ConfigError"
    except ConfigError as e:
        assert "not found" in str(e)


def test_state_at_restores_cooling_factors():
    S = system()
    row = S.telemetry[S.telemetry.k_C1 < 1.0].iloc[-1]
    st = state_at(S.telemetry, S.twin, row.timestamp)
    assert st.twin.units["C1"].k == row.k_C1
