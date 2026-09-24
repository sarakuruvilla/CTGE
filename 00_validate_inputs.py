#!/usr/bin/env python
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path: _sys.path.insert(0,str(_ROOT))
from pathlib import Path
import pandas as pd
from ctge.config import load_config

cfg=load_config(); root=Path(cfg['root']); raw=root/'data'/'raw'
required=['city_day.csv','city_hour.csv','station_day.csv','station_hour.csv','stations.csv']
missing=[f for f in required if not (raw/f).exists()]
if missing:
    raise SystemExit('Missing raw files in data/raw/: '+', '.join(missing))

checks={
 'city_day.csv':['City','Date','PM2.5','PM10','NO2','NH3','CO','SO2','O3','AQI'],
 'city_hour.csv':['City','Datetime','PM2.5','PM10','NO2','NH3','CO','SO2','O3','AQI'],
 'station_day.csv':['StationId','Date'],
 'station_hour.csv':['StationId','Datetime'],
 'stations.csv':['StationId','City'],
}
for f,cols in checks.items():
    df=pd.read_csv(raw/f,nrows=2)
    miss=[c for c in cols if c not in df.columns]
    if miss: raise SystemExit(f'{f}: missing columns {miss}')
    print(f'OK {f}: {len(df.columns)} columns')
print('Input validation complete.')
