"""Digital twin: racks, cooling units and airflow links as a weighted graph (Sec. 4.2)."""
import copy
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class Rack:
    rack_id: str
    zone_id: str
    p_idle: float
    p_max: float
    position: tuple = (0, 0)

    def power(self, u):
        """Eq. 4.1: power grows linearly with utilisation."""
        return self.p_idle + (self.p_max - self.p_idle) * u


@dataclass
class CoolingUnit:
    unit_id: str
    zone_id: str
    capacity: float
    supply_temp: float
    serves: dict = field(default_factory=dict)  # rack_id -> service weight s_iu
    k: float = 1.0                              # current capacity factor


class DigitalTwin:
    def __init__(self, cfg):
        self.racks = {r["id"]: Rack(r["id"], r["zone"], r["p_idle"], r["p_max"],
                                    tuple(r.get("position", (0, 0)))) for r in cfg["racks"]}
        self.units = {u["id"]: CoolingUnit(u["id"], u["zone"], u["capacity"],
                                           u["supply_temp"], dict(u["serves"]))
                      for u in cfg["cooling_units"]}
        self.graph = nx.Graph()
        for r in self.racks.values():
            self.graph.add_node(r.rack_id, kind="rack", zone=r.zone_id)
        for a, b, w in cfg["coupling"]:
            if a not in self.racks or b not in self.racks:
                raise ValueError(f"Coupling refers to unknown rack: {a}-{b}")
            self.graph.add_edge(a, b, kind="coupling", weight=float(w))
        for u in self.units.values():
            self.graph.add_node(u.unit_id, kind="cooling", zone=u.zone_id)
            for rid, s in u.serves.items():
                if rid not in self.racks:
                    raise ValueError(f"Cooling unit {u.unit_id} serves unknown rack {rid}")
                self.graph.add_edge(u.unit_id, rid, kind="service", weight=float(s))

    @property
    def rack_ids(self):
        return list(self.racks)

    def neighbours(self, rack_id):
        """Neighbouring racks with coupling weights."""
        return {n: d["weight"] for n, d in self.graph[rack_id].items()
                if d["kind"] == "coupling"}

    def coupling_matrix(self):
        ids = self.rack_ids
        W = np.zeros((len(ids), len(ids)))
        for i, a in enumerate(ids):
            for b, w in self.neighbours(a).items():
                W[i, ids.index(b)] = w
        return W

    def cooling_effectiveness(self, rack_id, k=None):
        """Eq. 4.3: c_i = min(1, sum_u s_iu * k_u). `k` may override unit factors."""
        total = sum(u.serves.get(rack_id, 0.0) * (k[u.unit_id] if k else u.k)
                    for u in self.units.values())
        return float(min(1.0, total))

    def supply_temp(self, rack_id):
        pairs = [(u.serves[rack_id], u.supply_temp) for u in self.units.values()
                 if rack_id in u.serves]
        return sum(s * t for s, t in pairs) / sum(s for s, _ in pairs)

    def unit_for_zone(self, zone_id):
        return next(u.unit_id for u in self.units.values() if u.zone_id == zone_id)

    def units_serving(self, rack_id):
        return [u.unit_id for u in self.units.values() if rack_id in u.serves]

    def set_cooling_factor(self, unit_id, k):
        self.units[unit_id].k = float(k)

    def copy(self):
        """Independent copy so what-if experiments never touch the base state."""
        return copy.deepcopy(self)


@dataclass
class SystemState:
    """Snapshot used by the predictor and what-if simulator: a short telemetry
    window per rack (most recent rows last) plus the twin with current factors."""
    window: pd.DataFrame
    twin: DigitalTwin
    timestamp: pd.Timestamp

    def copy(self):
        return SystemState(self.window.copy(), self.twin.copy(), self.timestamp)
