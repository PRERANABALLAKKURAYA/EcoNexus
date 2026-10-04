"""Tests for the live-twin playback data and player page."""
import json

from src.dashboard.playback import HISTORY, load_frames
from src.dashboard.player import render
from src.decision.hotspot import analyse
from src.decision.whatif import Action, what_if
from tests.helpers import system


def test_playback_frames_are_complete_and_real():
    S = system()
    d = load_frames(S)
    n_racks = len(S.twin.racks)
    assert len(d["frames"]) > 100 and len(d["racks"]) == n_racks
    for f in d["frames"]:
        assert len(f["r"]) == n_racks and len(f["k"]) == len(S.twin.units)
        for r in f["r"]:
            assert r["L"] in (0, 1, 2) and len(r["F"]) == S.sim["horizon"]
            assert (r["L"] == 0) == (not r["RC"] and not r["RS"])      # advice only for racks at risk
    assert all(len(v) == HISTORY for v in d["hist0"].values())
    assert any(r["L"] == 2 for f in d["frames"] for r in f["r"]), "demo window must contain a hotspot"
    json.dumps(d)                                                      # serialisable


def test_playback_uses_cache_on_second_call():
    S = system()
    a = load_frames(S)
    assert load_frames(S)["sig"] == a["sig"]


def test_player_html_embeds_data_and_has_no_placeholders():
    S = system()
    html = render(load_frames(S), 800)
    assert "__DATA__" not in html and "__H__" not in html and "DIGITAL TWIN" in html.upper()
    assert "</script>" in html and html.count("</script>") == 1       # data cannot break the page


def test_whatif_with_precomputed_before_gives_same_result():
    S = system()
    st = S.state_at(S.default_time)
    act = Action("boost_cooling", "R1", unit_id="C1", boost=0.3)
    an = analyse(st, S.predictor, S.thr)
    a, _ = what_if(st, act, S.predictor, S.thr)
    b, _ = what_if(st, act, S.predictor, S.thr, before=an)
    assert a.table.equals(b.table)


def test_rack_above_warning_is_never_labelled_normal():
    S = system()
    an = analyse(S.state_at(S.default_time), S.predictor, S.thr)
    for a in an.assessments.values():
        if a.current_temp >= S.thr["warning_temp"]:
            assert a.level != "Normal"
