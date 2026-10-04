"""Headless demo of the whole pipeline (no Streamlit needed): python scripts/demo.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.decision.explainer import explain, explanation_text  # noqa: E402
from src.decision.hotspot import analyse  # noqa: E402
from src.decision.recommender import recommend  # noqa: E402
from src.pipeline import build_system  # noqa: E402

S = build_system(ROOT)
print(f"Model alarm stats: {S.alarm}\n")
st = S.state_at(S.default_time)
an = analyse(st, S.predictor, S.thr)
print(f"State at {S.default_time}")
for rid, a in an.assessments.items():
    print(f"  {rid}: {a.level:<20} now {a.current_temp:5.1f}  peak {a.predicted_peak:5.1f}  score {a.score:.2f}")
for rid, a in an.assessments.items():
    if a.level == "Normal":
        continue
    print("\n" + explanation_text(rid, explain(rid, an, st.twin, S.thr), a.level))
    for act, rep in recommend(rid, an, st, S.predictor, S.thr, top=2):
        print(f"  -> {act.describe()}  [{act.rule}] {act.justification}")
