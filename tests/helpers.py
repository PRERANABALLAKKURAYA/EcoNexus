"""Shared fixtures (plain functions so tests run with pytest or a minimal runner)."""
import functools
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=1)
def system():
    """Small, fast system built in a temp dir (fewer steps) so tests stay quick."""
    from src.pipeline import build_system
    tmp = Path(tempfile.mkdtemp())
    for d in ("config", "data", "models"):
        (tmp / d).mkdir()
    for f in ("twin", "thresholds"):
        (tmp / f"config/{f}.yaml").write_text((ROOT / f"config/{f}.yaml").read_text())
    (tmp / "config/sim.yaml").write_text((ROOT / "config/sim.yaml").read_text())
    return build_system(tmp)


def default_twin():
    from src.twin.twin import DigitalTwin
    return DigitalTwin(yaml.safe_load(open(ROOT / "config/twin.yaml")))


def thresholds():
    return yaml.safe_load(open(ROOT / "config/thresholds.yaml"))


def sim_cfg():
    return yaml.safe_load(open(ROOT / "config/sim.yaml"))
