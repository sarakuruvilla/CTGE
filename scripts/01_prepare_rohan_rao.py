#!/usr/bin/env python
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path: _sys.path.insert(0,str(_ROOT))
from pathlib import Path
import numpy as np, pandas as pd
from ctge.config import load_config
from ctge.data import read_rohan_rao, add_daily_state_features

cfg=load_config(); root=Path(cfg['root'])
raw=root/'data'/'raw'; out=root/'data'/'processed'; ext=root/'data'/'external'
out.mkdir(parents=True,exist_ok=True); ext.mkdir(parents=True,exist_ok=True)

city_day,city_hour,station_day,station_hour,stations=read_rohan_rao(raw,cfg['cities'])
state=add_daily_state_features(city_day)
state.to_csv(out/'city_daily_state.csv',index=False)
city_hour.to_csv(out/'city_hour_clean.csv',index=False)
station_day.to_csv(out/'station_day_clean.csv',index=False)
station_hour.to_csv(out/'station_hour_clean.csv',index=False)
stations.to_csv(out/'stations_clean.csv',index=False)

# Create a transparent calendar/regime file if the user did not provide one.
cal_path=ext/'event_calendar.csv'
if not cal_path.exists():
    dates=pd.date_range(state['Date'].min(),state['Date'].max(),freq='D')
    cal=pd.DataFrame({'Date':dates})
    cal['EV_MONSOON']=cal['Date'].dt.month.isin([6,7,8,9]).astype(int)
    cal['EV_WINTER_SMOG']=cal['Date'].dt.month.isin([11,12,1,2]).astype(int)
    cal['EV_NEWYEAR']=((cal['Date'].dt.month==1)&(cal['Date'].dt.day==1)).astype(int)
    cal['EV_CHRISTMAS']=((cal['Date'].dt.month==12)&(cal['Date'].dt.day==25)).astype(int)
    # COVID structural-regime indicator. Exact event/festival dates can be supplied in an external file.
    cal['EV_COVID']=cal['Date'].between('2020-03-25','2020-06-30').astype(int)
    cal.to_csv(cal_path,index=False)
    print('Created minimal event calendar:',cal_path)
print('Prepared Rohan Rao data for:',', '.join(cfg['cities']))
