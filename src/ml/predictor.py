"""Temperature predictor (Sec. 4.3): pooled Gradient Boosting + persistence/Ridge baselines."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .features import FEATURES, build_features


class TemperaturePredictor:
    def __init__(self, horizon=3, seed=42):
        self.horizon, self.seed = horizon, seed
        self.models, self.params = {}, {}

    def fit(self, train):
        X = train[FEATURES]
        grid = {"n_estimators": [100], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]}
        search = GridSearchCV(GradientBoostingRegressor(random_state=self.seed), grid,
                              cv=TimeSeriesSplit(3), scoring="neg_mean_absolute_error")
        search.fit(X, train["y_h1"])                      # tune once on h = 1
        self.params = search.best_params_
        for h in range(1, self.horizon + 1):
            m = GradientBoostingRegressor(random_state=self.seed, **self.params)
            self.models[h] = m.fit(X, train[f"y_h{h}"])
        return self

    def predict(self, feats):
        """Forecast table: index rack_id, columns h1..hH (absolute temperatures, degC)."""
        X = feats[FEATURES]
        out = {f"h{h}": feats["temp"].to_numpy() + self.models[h].predict(X)
               for h in self.models}
        return pd.DataFrame(out, index=feats["rack_id"].to_numpy())

    def feature_importances(self):
        return pd.Series(self.models[1].feature_importances_, index=FEATURES).sort_values(
            ascending=False)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def chronological_split(df, train_frac, val_frac):
    ts = np.sort(df["timestamp"].unique())
    a, b = ts[int(len(ts) * train_frac)], ts[int(len(ts) * (train_frac + val_frac))]
    return df[df.timestamp < a], df[(df.timestamp >= a) & (df.timestamp < b)], df[df.timestamp >= b]


def _errs(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p)))}


def evaluate(pred, ridge, test, thr):
    """MAE/RMSE of model vs baselines per horizon + alarm precision/recall (Eq. 4.4-4.5)."""
    rows = []
    for h in range(1, pred.horizon + 1):
        y = test[f"y_h{h}"]
        for name, p in [("GradientBoosting", pred.models[h].predict(test[FEATURES])),
                        ("Ridge", ridge[h].predict(test[FEATURES])),
                        ("Persistence", np.zeros(len(test)))]:
            rows.append({"horizon": h, "model": name, **_errs(y, p)})
    metrics = pd.DataFrame(rows)
    fc = pred.predict(test)
    peak_pred = fc.max(axis=1).to_numpy()
    peak_true = test["temp"].to_numpy() + test[[f"y_h{h}" for h in range(1, pred.horizon + 1)]
                                              ].max(axis=1).to_numpy()
    crit = thr["critical_temp"]
    truth, flag = peak_true >= crit, peak_pred >= crit
    tp = int((truth & flag).sum())
    alarm = {"truth_positives": int(truth.sum()), "flags": int(flag.sum()),
             "recall": tp / max(int(truth.sum()), 1), "precision": tp / max(int(flag.sum()), 1)}
    ridge_peak = np.max([test["temp"].to_numpy() + ridge[h].predict(test[FEATURES])
                         for h in range(1, pred.horizon + 1)], axis=0)
    for name, pk in [("ridge", ridge_peak), ("persistence", test["temp"].to_numpy())]:
        f = pk >= crit
        tp_b = int((truth & f).sum())
        alarm[f"{name}_recall"] = tp_b / max(int(truth.sum()), 1)
        alarm[f"{name}_precision"] = tp_b / max(int(f.sum()), 1)
    return metrics, alarm


def train_predictor(tel, twin, thr, sim):
    """Full procedure of Sec. 4.3.4. Returns (predictor, metrics, alarm_stats)."""
    H = int(sim["horizon"])
    df = build_features(tel, twin, thr, horizon=H)
    train, val, test = chronological_split(df, sim["train_frac"], sim["val_frac"])
    pred = TemperaturePredictor(H, sim["seed"]).fit(train)
    ridge = {h: Ridge(alpha=1.0).fit(train[FEATURES], train[f"y_h{h}"]) for h in range(1, H + 1)}
    metrics, alarm = evaluate(pred, ridge, test, thr)
    val_mae = _errs(val["y_h1"], pred.models[1].predict(val[FEATURES]))["MAE"]
    alarm["validation_MAE_h1"] = val_mae
    return pred, metrics, alarm
