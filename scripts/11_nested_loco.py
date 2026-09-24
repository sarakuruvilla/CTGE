#!/usr/bin/env python
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path: _sys.path.insert(0,str(_ROOT))
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error
from ctge.config import load_config
from ctge.loco import pooled_fit_predict
from ctge.stats import exact_wilcoxon_greater

cfg=load_config(); root=Path(cfg['root']); df=pd.read_csv(root/'data'/'processed'/'ctge_feature_master.csv',parse_dates=['Date']); out=root/'outputs'
state=['PM2_5_t','PM10_t','NO2_t','SO2_t','CO_t','O3_t','NH3_t','AQI_t']+[f'{x}_missing' for x in ['PM2_5','PM10','NO2','SO2','CO','O3','NH3','AQI']]
intra=[c for c in df if c.startswith('H_')]; base=[c for c in state+intra if c in df]
context=[c for c in df if c.startswith('CPDM_') or c.startswith('ERDM_') or c.startswith('LCP_') or c.startswith('LER_') or c.startswith('EV_')]
seeds=cfg['forecast_seeds']; cand=cfg['loco']['shrinkage_candidates']; rows=[]
# Use 2018 source observations as fit, 2019 held-out observations as zero-shot target.
for held in cfg['cities']:
    src_cities=[c for c in cfg['cities'] if c!=held]
    # Inner source-city selection: choose largest scale positive in every inner held-source fold.
    feasible=[]
    for scale in cand:
        gains=[]
        for inner in src_cities:
            fitcities=[c for c in src_cities if c!=inner]
            s=df[df['City'].isin(fitcities)&df['Date'].dt.year.eq(2018)&df['target_AQI_t1'].notna()]
            t=df[(df['City']==inner)&df['Date'].dt.year.eq(2019)&df['target_AQI_t1'].notna()]
            if len(s)==0 or len(t)==0: continue
            p1,pct=pooled_fit_predict(s,t,base,context,42,scale); y=t['target_AQI_t1'].to_numpy(); r1=np.sqrt(np.mean((y-p1)**2)); rc=np.sqrt(np.mean((y-pct)**2)); gains.append(100*(r1-rc)/r1)
        if len(gains)==len(src_cities) and min(gains)>0: feasible.append(scale)
    scale=max(feasible) if feasible else min(cand)
    for seed in seeds:
        s=df[df['City'].isin(src_cities)&df['Date'].dt.year.eq(2018)&df['target_AQI_t1'].notna()]
        t=df[(df['City']==held)&df['Date'].dt.year.eq(2019)&df['target_AQI_t1'].notna()]
        p1,pct=pooled_fit_predict(s,t,base,context,seed,scale); y=t['target_AQI_t1'].to_numpy(); r1=np.sqrt(np.mean((y-p1)**2)); rc=np.sqrt(np.mean((y-pct)**2)); rows.append({'HeldOutCity':held,'Seed':seed,'Scale':scale,'C1_RMSE':r1,'CTGE_RMSE':rc,'Gain_pct':100*(r1-rc)/r1})
R=pd.DataFrame(rows); R.to_csv(out/'nested_loco_7seed.csv',index=False)
S=R.groupby('HeldOutCity').agg(C1_RMSE=('C1_RMSE','mean'),CTGE_RMSE=('CTGE_RMSE','mean'),Gain_pct=('Gain_pct','mean'),PositiveSeeds=('Gain_pct',lambda x:int((x>0).sum())),Scale=('Scale','first')).reset_index(); S.to_csv(out/'nested_loco_summary.csv',index=False)
w,pv=exact_wilcoxon_greater(S['C1_RMSE']-S['CTGE_RMSE']); print(S.to_string(index=False)); print('Cross-city W=',w,'p=',pv)
