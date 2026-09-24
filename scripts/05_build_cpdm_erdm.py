#!/usr/bin/env python
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path: _sys.path.insert(0,str(_ROOT))
from pathlib import Path
import pandas as pd
from ctge.config import load_config
from ctge.cpdm import build_cpdm
from ctge.erdm import build_erdm

cfg=load_config(); root=Path(cfg['root']); p=root/'data'/'processed'
base=pd.read_csv(p/'ctge_base_master.csv',parse_dates=['Date'])
all_city=[]; meta=[]
for city,g in base.groupby('City'):
    g=g.sort_values('Date').reset_index(drop=True)
    cp, m=build_cpdm(g,train_end=cfg['periods']['development_train'][1])
    er=build_erdm(cp)
    all_city.append(er); m.insert(0,'City',city); meta.append(m)
master=pd.concat(all_city,ignore_index=True)
master.to_csv(p/'ctge_feature_master.csv',index=False)
pd.concat(meta,ignore_index=True).to_csv(p/'cpdm_estimator_audit.csv',index=False)
print('Wrote',p/'ctge_feature_master.csv')
