"""ExoNexus AI entry point. Run: streamlit run src/app.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from src.dashboard import player, theme, views  # noqa: E402
from src.dashboard.playback import load_frames  # noqa: E402
from src.decision.hotspot import analyse  # noqa: E402
from src.pipeline import WINDOW, ConfigError, build_system  # noqa: E402

st.set_page_config(page_title="ExoNexus AI", page_icon="❄", layout="wide")
st.markdown(theme.CSS, unsafe_allow_html=True)

if not st.session_state.get("entered_dashboard", False):
    st.markdown(theme.LANDING, unsafe_allow_html=True)
    if st.button("Launch live twin", type="primary", key="launch_dashboard"):
        st.session_state["entered_dashboard"] = True
        st.rerun()
    st.stop()


@st.cache_resource(show_spinner="Loading ExoNexus AI (first run generates data and trains the model)...")
def get_system():
    return build_system(ROOT)


@st.cache_resource(show_spinner="Preparing the live simulation (first run only, under a minute)...")
def get_playback(_S):
    return load_frames(_S)


try:
    S = get_system()
except ConfigError as e:
    st.error(f"Configuration problem: {e}")
    st.stop()
frames = get_playback(S)

times = sorted(S.telemetry.timestamp.unique())
lo, hi, default_i = times.index(S.test_start) + WINDOW, len(times) - 1, times.index(S.default_time)
st.session_state.setdefault("moment", default_i)

# ------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ❄ ExoNexus AI")
    st.markdown("**Analysis moment**")
    st.caption("Used by What-if lab, Forecast and All racks. The Live twin plays on its own.")
    step = st.select_slider("Moment", options=list(range(lo, hi + 1)), key="moment",
                            format_func=lambda i: views.clock(times[i]), label_visibility="collapsed")
    st.button("Jump to the hottest moment", on_click=lambda: st.session_state.update(moment=default_i))
    st.divider()
    st.markdown("**The room**")
    st.caption(f"{len(S.twin.racks)} racks in {len({r.zone_id for r in S.twin.racks.values()})} zones, "
               f"{len(S.twin.units)} cooling units. Warning at {S.thr['warning_temp']:.0f} °C, "
               f"critical at {S.thr['critical_temp']:.0f} °C. Forecast looks "
               f"{frames['horizon_min']} minutes ahead.")
    st.markdown("**Colours**")
    st.markdown("🟢 Normal  \n🟡 Rising risk  \n🔴 Hotspot (predicted to hit critical)")
    st.caption("Simulated data only. ExoNexus AI gives advice and never controls equipment.")

ts = pd.Timestamp(times[step])
if st.session_state.get("base_step") != step:
    st.session_state.update(base_step=step, base=S.state_at(ts), version=0, report=None, pending=None, recs={})
base = st.session_state["base"]
key = (step, st.session_state["version"])
if st.session_state.get("an_key") != key:
    st.session_state.update(an_key=key, an=analyse(base, S.predictor, S.thr), recs={})
x = {"S": S, "base": base, "an": st.session_state["an"], "ts": ts}

# ------------------------------------------------------------ main
st.markdown(theme.HEADER, unsafe_allow_html=True)
t_live, t_what, t_fc, t_all, t_sys, t_set = st.tabs(
    ["Live twin", "What-if lab", "Forecast", "All racks", "How it works", "Settings"])
with t_live:
    st.markdown(theme.STEPS, unsafe_allow_html=True)
    components.html(player.render(frames, 860), height=875, scrolling=False)
with t_what:
    views.whatif(x)
with t_fc:
    views.forecast(x)
with t_all:
    views.fleet(x)
with t_sys:
    views.system(x)
with t_set:
    views.settings(x)
