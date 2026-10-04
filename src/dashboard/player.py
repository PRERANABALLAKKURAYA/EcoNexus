"""The live digital twin: a self-contained HTML/SVG page (assets/player.html) that PLAYS the
pre-computed frames. Racks, cooling units and airflow links heat, cool and raise alerts in real time."""
import json
from pathlib import Path

TEMPLATE = (Path(__file__).parent / "assets" / "player.html").read_text(encoding="utf-8")


def render(data, height=860):
    blob = json.dumps(data).replace("</", "<\\/")      # keep the data from closing the <script>
    return TEMPLATE.replace("__DATA__", blob).replace("__H__", str(height))
