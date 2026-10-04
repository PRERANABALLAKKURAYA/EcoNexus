"""Look and feel: landing page, dark control-room theme and chart styling."""
from pathlib import Path

CSS = "<style>" + (Path(__file__).parent / "assets" / "theme.css").read_text(encoding="utf-8") \
    + "</style>"

HEADER = (
    '<div style="height:3.4rem"></div> '
    '<div class="eco-head"><div> '
    '<div class="eco-eyebrow">Digital twin &middot; thermal decision support</div> '
    '<div class="eco-title">ExoNexus <span>AI</span></div> '
    '<div class="eco-sub">A live digital twin of a six-rack server hall. '
    'It predicts which racks will overheat in the next 15 minutes, '
    'explains why in plain words, suggests fixes, and lets you test each fix before you decide.</div> '
    '</div><div class="eco-pill">SIMULATED DATA &middot; ADVICE ONLY</div></div> '
)

LANDING = (
    '<div class="landing-shell">'
    '<div class="landing-nav"><div class="brand-mark"><span>EXO</span>NEXUS <b>AI</b></div>'
    '<div class="landing-nav-note">THERMAL INTELLIGENCE FOR MODERN DATA CENTRES</div></div>'
    '<div class="landing-hero">'
    '<div class="landing-copy">'
    '<div class="landing-kicker"><i></i> DIGITAL TWIN ONLINE &middot; ADVICE ONLY</div>'
    '<h1>Know the heat<br><em>before it happens.</em></h1>'
    '<p>ExoNexus AI gives your server hall a living digital twin. Predict overheating, understand what is driving it, and test the safest fix before touching production.</p>'
    '<div class="landing-actions"><span class="landing-arrow">&rarr;</span>'
    '<div><div class="landing-action-label">READY WHEN YOU ARE</div>'
    '<div class="landing-action-sub">Launch the live twin to begin</div></div></div>'
    '</div>'
    '<div class="landing-orbit"><div class="orbit-ring ring-one"></div><div class="orbit-ring ring-two"></div>'
    '<div class="orbit-core"><span>AI</span><small>THERMAL<br>MODEL</small></div>'
    '<div class="orbit-node node-one">PREDICT</div><div class="orbit-node node-two">EXPLAIN</div><div class="orbit-node node-three">SIMULATE</div></div>'
    '</div>'
    '<div class="landing-grid">'
    '<div><b>01</b><strong>Predict</strong><span>15-minute thermal forecasts for every rack.</span></div>'
    '<div><b>02</b><strong>Explain</strong><span>Plain-language causes, not black-box alerts.</span></div>'
    '<div><b>03</b><strong>Decide</strong><span>Try fixes safely on a copy of the hall.</span></div>'
    '</div></div>'
)

STEPS = (
    '<div class="eco-steps"> '
    '<div class="eco-step"><b>1 &middot; WATCH</b>The hall plays like a live feed. '
    'Racks turn red when the model expects them to overheat.</div> '
    '<div class="eco-step"><b>2 &middot; CLICK A RACK</b>See its forecast, what is driving the heat '
    'and the fixes the model recommends.</div> '
    '<div class="eco-step"><b>3 &middot; TEST A FIX</b>Open the What-if lab to try an action on a copy of the hall, '
    'then accept or reject it.</div> '
    '</div> '
)


def style_fig(fig, height=420):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0e1b19",
                      height=height, margin=dict(l=10, r=10, t=30, b=10),
                      font=dict(family="IBM Plex Sans, sans-serif", color="#e8e4dc"), legend_orientation="h")
    fig.update_xaxes(gridcolor="#1d3733")
    fig.update_yaxes(gridcolor="#1d3733")
    return fig
