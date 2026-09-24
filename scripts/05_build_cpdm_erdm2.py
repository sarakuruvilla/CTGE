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
# Manuscript Section 3.2 states CPDM's expected-value model conditions on six
# ERA5 meteorological covariates (temperature, relative humidity, surface
# pressure, precipitation, wind speed, boundary-layer height) in addition to
# calendar/seasonal/diurnal context. This previously silently loaded the
# non-ERA5 base master instead of the ERA5-merged master produced by
# 04_match_integrate_era5.py, so CPDM never actually received meteorological
# covariates despite the manuscript's explicit claim that it does. Fixed to
# load the ERA5-merged file when it exists; falls back to the base master
# with an explicit warning (not a silent substitution) if ERA5 has not been
# acquired/integrated, since the two are not equivalent runs.
era5_path=p/'ctge_master_with_era5.csv'
base_path=p/'ctge_base_master.csv'
if era5_path.exists():
    base=pd.read_csv(era5_path,parse_dates=['Date'])
    print(f'Loaded ERA5-integrated master: {era5_path}')
else:
    base=pd.read_csv(base_path,parse_dates=['Date'])
    print(f'WARNING: {era5_path} not found; falling back to {base_path}. '
          f'CPDM will NOT include ERA5 meteorological covariates in this run, '
          f'which does not match manuscript Section 3.2. Run '
          f'03_download_era5_cds.py (or 03b) and 04_match_integrate_era5.py '
          f'first to produce the ERA5-integrated master.')
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
