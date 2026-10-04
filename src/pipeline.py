"""Wires the layers together: config -> twin -> telemetry -> model -> SystemState."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from src.data.simulator import simulate
from src.data.workload_loader import load_workload
from src.ml.predictor import TemperaturePredictor, train_predictor
from src.twin.twin import DigitalTwin, SystemState

WINDOW = 12   # telemetry rows kept per rack in a SystemState


class ConfigError(Exception):
    """Raised with a readable message for bad/missing configuration (NFR4)."""


def load_yaml(path):
    try:
        with open(path) as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError(f"Configuration file not found: {path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")


@dataclass
class System:
    root: Path
    twin: DigitalTwin
    sim: dict
    thr: dict
    telemetry: pd.DataFrame
    predictor: TemperaturePredictor
    metrics: pd.DataFrame
    alarm: dict
    test_start: pd.Timestamp
    default_time: pd.Timestamp

    def state_at(self, ts):
        return state_at(self.telemetry, self.twin, ts)


def state_at(tel, twin, ts):
    """SystemState at timestamp `ts`: last WINDOW rows per rack + twin with the
    cooling factors that were in effect at `ts`."""
    ts = pd.Timestamp(ts)
    times = sorted(tel.timestamp.unique())
    i = times.index(ts)
    window = tel[tel.timestamp.isin(times[max(0, i - WINDOW + 1):i + 1])].copy()
    t = twin.copy()
    row = window[window.timestamp == ts].iloc[0]
    for uid in t.units:
        t.set_cooling_factor(uid, row[f"k_{uid}"])
    return SystemState(window, t, ts)


def build_system(root=".", regenerate=False):
    root = Path(root)
    twin_cfg = load_yaml(root / "config/twin.yaml")
    sim, thr = load_yaml(root / "config/sim.yaml"), load_yaml(root / "config/thresholds.yaml")
    for key in ("seed", "steps", "horizon", "alpha", "beta", "gamma", "sigma"):
        if key not in sim:
            raise ConfigError(f"sim.yaml is missing required key '{key}'")
    if abs(sum(thr["risk_weights"]) - 1.0) > 1e-6:
        raise ConfigError("thresholds.yaml: risk_weights must sum to 1")
    if thr["warning_temp"] >= thr["critical_temp"]:
        raise ConfigError("thresholds.yaml: warning_temp must be below critical_temp")
    twin = DigitalTwin(twin_cfg)

    tel_path, model_path = root / "data/telemetry.csv", root / "models/predictor.joblib"
    if regenerate or not tel_path.exists():
        tel = simulate(twin, load_workload(root / "data/workload_sample.csv", sim["steps"]), sim)
        tel.to_csv(tel_path, index=False)
    tel = pd.read_csv(tel_path, parse_dates=["timestamp"])

    metrics_path = root / "models/metrics.yaml"
    if regenerate or not model_path.exists() or not metrics_path.exists():
        pred, metrics, alarm = train_predictor(tel, twin, thr, sim)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        pred.save(model_path)
        yaml.safe_dump({"metrics": metrics.to_dict("records"),
                        "alarm": {k: float(v) for k, v in alarm.items()}},
                       open(metrics_path, "w"))
    pred = TemperaturePredictor.load(model_path)
    saved = load_yaml(metrics_path)
    metrics, alarm = pd.DataFrame(saved["metrics"]), saved["alarm"]

    times = sorted(tel.timestamp.unique())
    test_start = pd.Timestamp(times[int(len(times) * (sim["train_frac"] + sim["val_frac"]))])
    test = tel[tel.timestamp >= test_start + pd.Timedelta(minutes=sim["interval_minutes"] * WINDOW)]
    hottest = test.loc[test.temperature_c.idxmax(), "timestamp"]
    default = max(test_start + pd.Timedelta(minutes=sim["interval_minutes"] * WINDOW),
                  hottest - pd.Timedelta(minutes=sim["interval_minutes"] * 6))
    return System(root, twin, sim, thr, tel, pred, metrics, alarm, test_start, default)
