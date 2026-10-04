"""Streamlit tab bodies. Each function gets a context dict `x` (system, state, analysis, moment)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.dashboard.theme import style_fig
from src.decision.decision_log import DecisionLogger
from src.decision.explainer import explain, explanation_text
from src.decision.recommender import recommend
from src.decision.whatif import Action, apply_action, what_if
from src.ml.features import build_features
from src.pipeline import build_system

NICE = {"Normal": "Normal", "Rising thermal risk": "Rising risk", "Potential hotspot": "Hotspot"}


@st.cache_resource
def _h1_predictions(_S):
    """Model's 5-minute-ahead prediction for every rack/time, aligned to the time it is *for*."""
    f = build_features(_S.telemetry, _S.twin, _S.thr)
    step = pd.Timedelta(minutes=_S.sim["interval_minutes"])
    return pd.DataFrame({"rack_id": f.rack_id, "timestamp": f.timestamp + step,
                         "pred": _S.predictor.predict(f)["h1"].to_numpy()})


def clock(ts):
    return f"{pd.Timestamp(ts):%a %d %b · %H:%M}"


def forecast(x):
    S, an, ts = x["S"], x["an"], x["ts"]
    st.markdown("<div class='eco-help'>Compares what the model <b>predicted</b> with what actually "
                "<b>happened</b> in the simulation, so you can judge how far to trust it.</div>", unsafe_allow_html=True)
    rack = st.selectbox("Rack", S.twin.rack_ids, key="fc_rack")
    H, step = S.sim["horizon"], pd.Timedelta(minutes=S.sim["interval_minutes"])
    tel = S.telemetry[S.telemetry.rack_id == rack].set_index("timestamp")
    span = tel.loc[ts - step * 60: ts + step * H]
    hist = span.loc[:ts]
    allp = _h1_predictions(S)
    pred = allp[allp.rack_id == rack].set_index("timestamp").pred
    pred = pred[pred.index.isin(hist.index)]
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.temperature_c, name="Measured", line=dict(color="#6fb1e6"))
    fig.add_scatter(x=pred.index, y=pred.values, name="Model's 5-minute-ahead prediction",
                    line=dict(color="#e0a63e", dash="dot"))
    fig.add_scatter(x=[ts] + [ts + step * (h + 1) for h in range(H)],
                    y=[an.assessments[rack].current_temp] + list(an.forecasts.loc[rack]),
                    name="Forecast from this moment", line=dict(color="#ff6a4d", width=3))
    fut = span.loc[ts:]
    fig.add_scatter(x=fut.index, y=fut.temperature_c, name="What actually happened",
                    line=dict(color="#8ba39c", dash="dash"))
    fig.add_hline(y=S.thr["warning_temp"], line_color="#e0a63e", annotation_text="warning")
    fig.add_hline(y=S.thr["critical_temp"], line_color="#d6472b", annotation_text="critical")
    fig.update_layout(yaxis_title="Temperature (°C)")
    st.plotly_chart(style_fig(fig, 440))
    with st.expander("How accurate is the model? (held-out test period)"):
        m = S.metrics.copy()
        m["horizon"] = m["horizon"].map(lambda h: f"{h * S.sim['interval_minutes']} min ahead")
        st.dataframe(m.rename(columns={"horizon": "Looking", "model": "Method", "MAE": "Avg error (°C)",
                                       "RMSE": "RMSE (°C)"}).round(2), hide_index=True)
        a = S.alarm
        st.write(f"Catching critical temperatures: Gradient Boosting recall **{a['recall']:.0%}** "
                 f"(precision {a['precision']:.0%}), Ridge {a['ridge_recall']:.0%}, "
                 f"'no change' baseline {a['persistence_recall']:.0%}.")
        st.bar_chart(S.predictor.feature_importances().head(8))


def fleet(x):
    S, an, base = x["S"], x["an"], x["base"]
    st.markdown(f"<div class='eco-help'>Every rack at <b>{clock(x['ts'])}</b>, riskiest first. "
                "The risk score blends forecast peak, workload, cooling shortfall and neighbour heat.</div>",
                unsafe_allow_html=True)
    rows = []
    for rid, a in sorted(an.assessments.items(), key=lambda kv: -kv[1].score):
        rs = explain(rid, an, base.twin, S.thr) if a.level != "Normal" else []
        rows.append({"Rack": rid, "Zone": base.twin.racks[rid].zone_id, "Status": NICE[a.level],
                     "Risk": a.score, "Now (°C)": round(a.current_temp, 1),
                     "Peak in 15 min (°C)": round(a.predicted_peak, 1),
                     "Cooling it gets": round(a.cooling_eff * 100),
                     "Why": explanation_text(rid, rs, a.level)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.2f"),
        "Cooling it gets": st.column_config.ProgressColumn("Cooling it gets", min_value=0, max_value=100,
                                                           format="%d%%")})


def whatif(x):
    S, base, an, ts = x["S"], x["base"], x["an"], x["ts"]
    log = DecisionLogger(S.root / "data/decision_log.csv")
    st.markdown(f"<div class='eco-help'>Try a fix on a <b>copy</b> of the hall at <b>{clock(ts)}</b> "
                "(change the moment in the sidebar). Nothing real is touched. Compare before and after, "
                "then accept or reject; your decision is logged.</div>", unsafe_allow_html=True)
    risky = [r for r, a in sorted(an.assessments.items(), key=lambda kv: -kv[1].score) if a.level != "Normal"]
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("##### ① Which rack has the problem?")
            ids = S.twin.rack_ids
            pend = st.session_state.get("pending")
            rack = st.selectbox("Rack", ids, index=ids.index(pend.rack_id) if pend else
                                (ids.index(risky[0]) if risky else 0), label_visibility="collapsed")
            a = an.assessments[rack]
            if a.level == "Normal":
                st.success(f"{rack} is Normal at this moment. Pick a moment with a hotspot in the sidebar, "
                           "or use the 'Jump to the hottest moment' button.")
            else:
                rs = explain(rack, an, base.twin, S.thr)
                st.warning(explanation_text(rack, rs, a.level))
        with st.container(border=True):
            st.markdown("##### ② Suggested fixes (already tested by the model)")
            if rack not in st.session_state["recs"]:
                st.session_state["recs"][rack] = recommend(rack, an, base, S.predictor, S.thr)
            recs = st.session_state["recs"][rack]
            if not recs:
                st.caption("No fix needed for a Normal rack.")
            for i, (act, rep) in enumerate(recs):
                st.markdown(f"**{i + 1}. {act.describe()}**  \n{act.justification}")
                if act.type != "monitor" and st.button("Load this fix", key=f"load{i}"):
                    st.session_state.update(pending=act, report=rep)
                    st.rerun()
        with st.container(border=True):
            st.markdown("##### ③ Or design your own fix")
            kinds = {"shift_load": "Move workload to another rack", "boost_cooling": "Raise a cooling unit's output",
                     "defer_jobs": "Pause low-priority jobs"}
            kind = st.selectbox("Action", list(kinds), format_func=kinds.get,
                                index=list(kinds).index(pend.type) if pend and pend.type in kinds else 0)
            share = st.slider("How much of the rack's workload", 10, 100,
                              int(pend.share * 100) if pend and pend.share else 50, 5, format="%d%%") / 100
            target = st.selectbox("Move it to", [r for r in ids if r != rack])
            unit = st.selectbox("Cooling unit", list(S.twin.units))
            boost = st.slider("Raise output by", 10, 50, 30, 5, format="%d%%") / 100
            act = Action(kind, rack, target=target if kind == "shift_load" else "", share=share,
                         unit_id=unit if kind == "boost_cooling" else "",
                         boost=boost if kind == "boost_cooling" else 0.0)
            if st.button("Run what-if", type="primary"):
                st.session_state["report"], _ = what_if(base, act, S.predictor, S.thr, before=an)
                st.session_state["pending"] = act
                st.rerun()
    with right:
        rep = st.session_state.get("report")
        with st.container(border=True):
            st.markdown("##### ④ Before vs after")
            if rep is None:
                st.info("Load a suggested fix or run your own to see its effect on every rack here.")
            else:
                t = rep.table
                st.markdown(f"**{rep.action.describe()}**")
                fig = go.Figure([go.Bar(x=t.rack, y=t.peak_before, name="Before", marker_color="#8da0b6"),
                                 go.Bar(x=t.rack, y=t.peak_after, name="After the fix", marker_color="#4fa88c")])
                fig.add_hline(y=S.thr["warning_temp"], line_color="#e0a63e", annotation_text="warning")
                fig.add_hline(y=S.thr["critical_temp"], line_color="#d6472b", annotation_text="critical")
                fig.update_layout(barmode="group", yaxis_title="Expected peak in 15 min (°C)")
                st.plotly_chart(style_fig(fig, 340))
                c1, c2 = st.columns(2)
                c1.metric("Hottest rack, expected peak", f"{t.peak_after.max():.1f} °C", f"{rep.peak_change:+.1f} °C",
                          delta_color="inverse")
                c2.metric("Racks made riskier", len(rep.newly_riskier))
                if rep.newly_riskier:
                    st.warning("Side effect: " + ", ".join(rep.newly_riskier) + " would become riskier.")
                show = t.rename(columns={"rack": "Rack", "peak_before": "Peak before", "peak_after": "Peak after",
                                         "level_before": "Status before", "level_after": "Status after",
                                         "change": "Change (°C)"})
                st.dataframe(show[["Rack", "Peak before", "Peak after", "Change (°C)", "Status before",
                                   "Status after"]], hide_index=True)
                a1, a2 = st.columns(2)
                if a1.button("✔ Accept this fix", type="primary"):
                    log.log(str(ts), rep.action.rack_id, rep.action.describe(), True)
                    st.session_state["base"] = apply_action(base, rep.action)
                    st.session_state.update(version=st.session_state["version"] + 1, report=None, pending=None)
                    st.rerun()
                if a2.button("✖ Reject"):
                    log.log(str(ts), rep.action.rack_id, rep.action.describe(), False)
                    st.session_state.update(report=None, pending=None)
                    st.rerun()
        with st.container(border=True):
            st.markdown("##### Decision log")
            st.dataframe(log.read(), hide_index=True)


def system(x):
    S = x["S"]
    st.markdown("<div class='eco-help'>How ExoNexus AI works, end to end, and the numbers behind it.</div>",
                unsafe_allow_html=True)
    stages = [
        ("1 · Workload", "Public-trace style CPU load per rack"),
        ("2 · Simulator", "Heat equation with cooling, neighbours, noise"),
        ("3 · Digital twin", f"{len(S.twin.racks)} racks, {len(S.twin.units)} cooling units, airflow graph"),
        ("4 · Predictor", "Gradient Boosting, 5/10/15 min ahead"),
        ("5 · Detector", "Normal / Rising risk / Hotspot rules"),
        ("6 · Explainer", "Cites the rule and the numbers"),
        ("7 · Recommender", "Candidate fixes, tested in what-if"),
        ("8 · Decision log", "Accepted and rejected advice"),
    ]
    cols = st.columns(4)
    for i, (a, b) in enumerate(stages):
        with cols[i % 4].container(border=True):
            st.markdown(f"**{a}**")
            st.caption(b)
    a = S.alarm
    c = st.columns(4)
    c[0].metric("Critical events caught", f"{a['recall']:.0%}")
    c[1].metric("False alarms (precision)", f"{a['precision']:.0%}")
    c[2].metric("Validation error", f"{a['validation_MAE_h1']:.2f} °C")
    c[3].metric("Simulated data", f"{S.sim['steps']} intervals")
    st.caption("All data are simulated; metrics are measured on a held-out simulated period and are not "
               "real-world guarantees.")


def settings(x):
    S = x["S"]
    st.markdown("<div class='eco-help'>Administrator settings. Changing thresholds takes effect immediately; "
                "changing the simulation regenerates the data and retrains the model (about 30 seconds).</div>",
                unsafe_allow_html=True)
    thr = dict(S.thr)
    with st.form("thr"), st.container():
        st.markdown("##### Alert thresholds")
        c = st.columns(3)
        thr["warning_temp"] = c[0].number_input("Warning temperature (°C)", value=float(S.thr["warning_temp"]))
        thr["critical_temp"] = c[1].number_input("Critical temperature (°C)", value=float(S.thr["critical_temp"]))
        thr["rising_slope"] = c[2].number_input("Heating-rate limit (°C per interval)", value=float(S.thr["rising_slope"]))
        thr["cooling_deficit"] = c[0].number_input("Weak-cooling limit (0-1)", value=float(S.thr["cooling_deficit"]))
        thr["high_load"] = c[1].number_input("High-workload limit (0-1)", value=float(S.thr["high_load"]))
        if st.form_submit_button("Save thresholds"):
            if thr["warning_temp"] >= thr["critical_temp"]:
                st.error("Warning temperature must be below the critical temperature.")
            else:
                yaml.safe_dump(thr, open(S.root / "config/thresholds.yaml", "w"))
                st.cache_resource.clear()
                st.session_state.clear()
                st.rerun()
    sim = dict(S.sim)
    with st.form("sim"):
        st.markdown("##### Simulation")
        c = st.columns(2)
        sim["seed"] = int(c[0].number_input("Random seed", value=int(S.sim["seed"]), step=1))
        sim["sigma"] = c[1].number_input("Sensor/process noise (°C)", value=float(S.sim["sigma"]))
        if st.form_submit_button("Regenerate data and retrain"):
            yaml.safe_dump(sim, open(S.root / "config/sim.yaml", "w"))
            with st.spinner("Generating data and training the model..."):
                build_system(S.root, regenerate=True)
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()
    st.caption("To change the room layout, edit config/twin.yaml, then regenerate.")
