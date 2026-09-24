from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]
SEVEN_CITIES = ["Bengaluru", "Hyderabad", "Chennai", "Delhi", "Jaipur", "Lucknow", "Gurugram"]


def _clean_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Pollutant/AQI concentrations cannot be negative.
            df.loc[df[c] < 0, c] = np.nan
    return df


def normalize_city(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace({"Bangalore": "Bengaluru", "Gurgaon": "Gurugram"})


def read_rohan_rao(raw_dir: str | Path, cities=None):
    raw_dir = Path(raw_dir)
    cities = cities or SEVEN_CITIES
    files = {
        "city_day": raw_dir / "city_day.csv",
        "city_hour": raw_dir / "city_hour.csv",
        "station_day": raw_dir / "station_day.csv",
        "station_hour": raw_dir / "station_hour.csv",
        "stations": raw_dir / "stations.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Rohan Rao input files:\n" + "\n".join(missing))

    city_day = pd.read_csv(files["city_day"], parse_dates=["Date"])
    city_hour = pd.read_csv(files["city_hour"], parse_dates=["Datetime"])
    stations = pd.read_csv(files["stations"])

    city_day["City"] = normalize_city(city_day["City"])
    city_hour["City"] = normalize_city(city_hour["City"])
    stations["City"] = normalize_city(stations["City"])
    keep_stations = set(stations.loc[stations["City"].isin(cities), "StationId"].astype(str))

    city_day = city_day[city_day["City"].isin(cities)].copy()
    city_hour = city_hour[city_hour["City"].isin(cities)].copy()

    day_chunks=[]
    for ch in pd.read_csv(files["station_day"], parse_dates=["Date"], chunksize=100_000):
        z=ch[ch["StationId"].astype(str).isin(keep_stations)]
        if len(z): day_chunks.append(z)
    station_day=pd.concat(day_chunks,ignore_index=True) if day_chunks else pd.DataFrame()
    hour_chunks=[]
    for ch in pd.read_csv(files["station_hour"], parse_dates=["Datetime"], chunksize=200_000):
        z=ch[ch["StationId"].astype(str).isin(keep_stations)]
        if len(z): hour_chunks.append(z)
    station_hour=pd.concat(hour_chunks,ignore_index=True) if hour_chunks else pd.DataFrame()
    stations = stations[stations["City"].isin(cities)].copy()

    numeric = POLLUTANTS + ["AQI", "NO", "NOx", "Benzene", "Toluene", "Xylene"]
    for df in [city_day, city_hour, station_day, station_hour]:
        _clean_numeric(df, numeric)

    city_day = city_day.sort_values(["City", "Date"]).drop_duplicates(["City", "Date"], keep="last")
    city_hour = city_hour.sort_values(["City", "Datetime"]).drop_duplicates(["City", "Datetime"], keep="last")
    station_day = station_day.sort_values(["StationId", "Date"]).drop_duplicates(["StationId", "Date"], keep="last")
    station_hour = station_hour.sort_values(["StationId", "Datetime"]).drop_duplicates(["StationId", "Datetime"], keep="last")
    return city_day, city_hour, station_day, station_hour, stations


def add_daily_state_features(city_day: pd.DataFrame) -> pd.DataFrame:
    df = city_day.copy().sort_values(["City", "Date"])
    # Canonical names used by CTGE scripts.
    rename = {"PM2.5": "PM2_5"}
    df = df.rename(columns=rename)
    state = ["PM2_5", "PM10", "NO2", "SO2", "CO", "O3", "NH3", "AQI"]
    for c in state:
        df[f"{c}_t"] = df[c]
        df[f"{c}_missing"] = df[c].isna().astype(int)
        # Strictly historical lags. No backward filling.
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f"{c}_lag{lag}"] = df.groupby("City")[c].shift(lag)
    df["target_AQI_t1"] = df.groupby("City")["AQI"].shift(-1)
    df["dow"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["doy"] = df["Date"].dt.dayofyear
    df["sin_doy"] = np.sin(2*np.pi*df["doy"]/365.25)
    df["cos_doy"] = np.cos(2*np.pi*df["doy"]/365.25)
    return df


def chronological_masks(df, train=(2018, 2018), valid=(2019, 2019), test=(2020, 2020)):
    y = df["Date"].dt.year
    return {
        "train": y.between(*train),
        "valid": y.between(*valid),
        "test": y.between(*test),
    }
