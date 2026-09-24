from __future__ import annotations
import numpy as np
import pandas as pd
from .data import POLLUTANTS


def _slope(values):
    y = np.asarray(values, float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x = np.arange(len(y))[ok]
    return float(np.polyfit(x, y[ok], 1)[0])


def build_intraday_features(city_hour: pd.DataFrame, min_hours=12) -> pd.DataFrame:
    """Create leakage-safe daily summaries from hourly pollutant/AQI observations.

    Every feature for date t is computed only from hourly observations within date t.
    The next-day target is never used.
    """
    h = city_hour.copy()
    h["Date"] = h["Datetime"].dt.floor("D")
    h["hour"] = h["Datetime"].dt.hour
    use = POLLUTANTS + ["AQI"]
    rows = []
    for (city, date), g in h.groupby(["City", "Date"], sort=True):
        row = {"City": city, "Date": date}
        for p in use:
            if p not in g.columns:
                continue
            s = pd.to_numeric(g[p], errors="coerce")
            n = int(s.notna().sum())
            prefix = "H_" + p.replace(".", "_")
            row[prefix + "_n"] = n
            if n < min_hours:
                for name in ["mean","std","min","max","median","p10","p90","range","iqr","slope","night","morning","afternoon","evening","peak_hour"]:
                    row[prefix + "_" + name] = np.nan
                continue
            row[prefix + "_mean"] = s.mean()
            row[prefix + "_std"] = s.std(ddof=0)
            row[prefix + "_min"] = s.min()
            row[prefix + "_max"] = s.max()
            row[prefix + "_median"] = s.median()
            row[prefix + "_p10"] = s.quantile(.10)
            row[prefix + "_p90"] = s.quantile(.90)
            row[prefix + "_range"] = s.max() - s.min()
            row[prefix + "_iqr"] = s.quantile(.75) - s.quantile(.25)
            row[prefix + "_slope"] = _slope(s.values)
            for label, hours in {
                "night": [0,1,2,3,4,5],
                "morning": [6,7,8,9,10,11],
                "afternoon": [12,13,14,15,16,17],
                "evening": [18,19,20,21,22,23],
            }.items():
                row[prefix + "_" + label] = g.loc[g["hour"].isin(hours), p].mean()
            if s.notna().any():
                row[prefix + "_peak_hour"] = float(g.loc[s.idxmax(), "hour"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["City", "Date"]).reset_index(drop=True)


def merge_state_intraday(state: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
    return state.merge(intraday, on=["City", "Date"], how="left", validate="one_to_one")
