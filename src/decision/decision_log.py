"""Decision logger (FR10): append accepted/rejected decisions with a time stamp."""
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

COLUMNS = ["logged_at", "sim_time", "rack_id", "action", "decision"]


class DecisionLogger:
    def __init__(self, path):
        self.path = Path(path)

    def log(self, sim_time, rack_id, action, accepted):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with open(self.path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(COLUMNS)
            w.writerow([datetime.now().isoformat(timespec="seconds"), sim_time, rack_id,
                        action, "accepted" if accepted else "rejected"])

    def read(self):
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame(columns=COLUMNS)
