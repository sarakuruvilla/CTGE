#!/usr/bin/env python
"""Optional ERA5 acquisition fallback using Open-Meteo's historical ERA5 endpoint."""
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path: _sys.path.insert(0,str(_ROOT))
from pathlib import Path
import time, requests, pandas as pd, numpy as np
from ctge.config import load_config
CITY={"Delhi":(28.6139,77.2090),"Bengaluru":(12.9716,77.5946),"Hyderabad":(17.3850,78.4867),"Chennai":(13.0827,80.2707),"Jaipur":(26.9124,75.7873),"Lucknow":(26.8467,80.9462),"Gurugram":(28.4595,77.0266)}
VARS=["temperature_2m","relative_humidity_2m","surface_pressure","precipitation","wind_speed_10m","wind_direction_10m","boundary_layer_height"]
def circ(s):
    a=np.deg2rad(s.dropna().to_numpy()); return np.nan if len(a)==0 else (np.rad2deg(np.arctan2(np.sin(a).mean(),np.cos(a).mean()))+360)%360
cfg=load_config(); out=Path(cfg['root'])/'data'/'external'; frames=[]
for city,(lat,lon) in CITY.items():
    r=requests.get('https://archive-api.open-meteo.com/v1/archive',params={'latitude':lat,'longitude':lon,'start_date':'2017-01-01','end_date':'2020-07-01','hourly':','.join(VARS),'models':'era5','timezone':'Asia/Kolkata','wind_speed_unit':'ms'},timeout=120); r.raise_for_status(); js=r.json(); h=pd.DataFrame(js['hourly']); h['time']=pd.to_datetime(h['time']); h.insert(0,'City',city); frames.append(h); time.sleep(1)
h=pd.concat(frames,ignore_index=True); h.to_csv(out/'era5_hourly_openmeteo.csv',index=False); h['Date']=h['time'].dt.floor('D')
d=h.groupby(['City','Date'],as_index=False).agg(temperature_c=('temperature_2m','mean'),relative_humidity_pct=('relative_humidity_2m','mean'),surface_pressure_hpa=('surface_pressure','mean'),precipitation_mm=('precipitation','sum'),wind_speed_ms=('wind_speed_10m','mean'),boundary_layer_height_m=('boundary_layer_height','mean'))
wd=h.groupby(['City','Date'])['wind_direction_10m'].apply(circ).rename('wind_direction_deg').reset_index(); d=d.merge(wd,on=['City','Date']); d.to_csv(out/'era5_daily_openmeteo.csv',index=False); print('done',d.shape)
