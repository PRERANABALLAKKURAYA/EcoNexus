"""Runs src/app.py end-to-end against a stub Streamlit/Plotly, to catch runtime errors
without a browser:  python scripts/smoke_app.py   (skipped checks never mask real errors)"""
import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class State(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class Fake(MagicMock):
    pass


st = MagicMock(name="streamlit")
st.session_state = State()
st.cache_resource = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
st.cache_resource.clear = lambda: None
st.columns = lambda spec, **k: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
st.tabs = lambda names: [MagicMock() for _ in names]
st.selectbox = lambda label, options, index=0, **k: list(options)[index]
st.select_slider = lambda label, options=(), key=None, **k: st.session_state[key]
st.slider = lambda label, mn=0, mx=1, value=None, *a, **k: value if value is not None else mn
st.number_input = lambda label, value=0, **k: value
st.button = lambda *a, **k: False
st.form_submit_button = lambda *a, **k: False
st.radio = lambda label, options, **k: list(options)[0]
mods = {"streamlit": st, "streamlit.components": MagicMock(), "streamlit.components.v1": MagicMock(),
        "plotly": MagicMock(), "plotly.graph_objects": MagicMock()}
try:
    import streamlit  # noqa: F401  (real one installed: use the stub anyway for determinism)
except ImportError:
    pass
sys.modules.update(mods)
st.components = mods["streamlit.components"]
st.components.v1 = mods["streamlit.components.v1"]
runpy.run_path(str(ROOT / "src/app.py"), run_name="__main__")
html_calls = mods["streamlit.components.v1"].html.call_args
assert html_calls and "DIGITAL TWIN" in html_calls[0][0].upper(), "player HTML was not rendered"
print("app smoke test OK: all tabs executed; player HTML length", len(html_calls[0][0]))
