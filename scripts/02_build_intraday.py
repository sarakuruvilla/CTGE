#!/usr/bin/env python
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path: _sys.path.insert(0,str(_ROOT))
from pathlib import Path
import pandas as pd
from ctge.config import load_config
from ctge.features import build_intraday_features, merge_state_intraday

cfg=load_config(); root=Path(cfg['root']); p=root/'data'/'processed'; e=root/'data'/'external'
state=pd.read_csv(p/'city_daily_state.csv',parse_dates=['Date'])
hour=pd.read_csv(p/'city_hour_clean.csv',parse_dates=['Datetime'])
intra=build_intraday_features(hour,min_hours=12)
intra.to_csv(p/'intraday_features.csv',index=False)
master=merge_state_intraday(state,intra)
cal=e/'event_calendar.csv'
if cal.exists():
    ev=pd.read_csv(cal,parse_dates=['Date'])
    # Prefix non-date event columns if necessary.
    ev=ev.rename(columns={c:(c if c=='Date' or c.startswith('EV_') else 'EV_'+c) for c in ev.columns})
    master=master.merge(ev,on='Date',how='left')
    evcols=[c for c in master if c.startswith('EV_')]
    master[evcols]=master[evcols].fillna(0)
master.to_csv(p/'ctge_base_master.csv',index=False)
print('Intraday features:',len([c for c in intra if c.startswith('H_')]))
print('Wrote',p/'ctge_base_master.csv')
