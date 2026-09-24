#!/usr/bin/env python
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path: _sys.path.insert(0,str(_ROOT))
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ctge.config import load_config
from ctge.models import make_extratrees, fit_c2_residual, apply_c3_reliability, gate_alpha

cfg=load_config(); root=Path(cfg['root']); p=root/'data'/'processed'; out=root/'outputs'; out.mkdir(exist_ok=True)
df=pd.read_csv(p/'ctge_feature_master.csv',parse_dates=['Date'])
c2cfg=pd.read_csv(root/'configs'/'frozen_c2_config.csv'); c2cfg['City']=c2cfg['City'].str.lower()
c3cfg=pd.read_csv(root/'configs'/'frozen_c3_config.csv'); c3cfg['City']=c3cfg['City'].str.lower()

state=['PM2_5_t','PM10_t','NO2_t','SO2_t','CO_t','O3_t','NH3_t','AQI_t']+[f'{x}_missing' for x in ['PM2_5','PM10','NO2','SO2','CO','O3','NH3','AQI']]
intra=[c for c in df if c.startswith('H_')]
base=[c for c in state+intra if c in df]
context=[c for c in df if c.startswith('CPDM_') or c.startswith('ERDM_') or c.startswith('LCP_') or c.startswith('LER_') or c.startswith('EV_')]
rows=[]; preds=[]
for city0,g0 in df.groupby('City'):
    city=city0.lower(); g=g0.sort_values('Date').copy()
    tr=g[g['Date'].between(cfg['periods']['development_train'][0],cfg['periods']['development_train'][1]) & g['target_AQI_t1'].notna()].copy()
    va=g[g['Date'].between(cfg['periods']['development_validation'][0],cfg['periods']['development_validation'][1]) & g['target_AQI_t1'].notna()].copy()
    c2row=c2cfg[c2cfg['City']==city].iloc[0].to_dict(); c3row=c3cfg[c3cfg['City']==city].iloc[0].to_dict()
    for seed in cfg['forecast_seeds']:
        # C1 probe
        m1=make_extratrees(seed); m1.fit(tr[base],tr['target_AQI_t1']); p1tr=m1.predict(tr[base]); p1=m1.predict(va[base])
        # C2 contextual residual correction
        rmodel,p2,a2,rhat=fit_c2_residual(tr,va,p1tr,p1,context,float(c2row['Alpha']),float(c2row['Gamma']),c2row['Gate'],float(c2row['Q']))
        # rb30 history from training residual benefit, if C3 uses it.
        p2tr = p1tr + float(c2row['Gamma'])*gate_alpha(tr,c2row['Gate'],float(c2row['Q']))*rmodel.predict(tr[context])
        hist=((tr['target_AQI_t1'].to_numpy()-p1tr)**2-(tr['target_AQI_t1'].to_numpy()-p2tr)**2).tolist()
        va2=va.copy(); va2['abs_corr']=np.abs(p2-p1); va2['rb30']=0.0
        p3,rho=apply_c3_reliability(va2,p1,p2,c3row,hist)
        y=va['target_AQI_t1'].to_numpy()
        def met(p): return (np.sqrt(mean_squared_error(y,p)),mean_absolute_error(y,p),r2_score(y,p))
        r1,m1e,r21=met(p1); r2,m2e,r22=met(p2); r3,m3e,r23=met(p3)
        rows.append({'City':city,'Seed':seed,'C1_RMSE':r1,'C2_RMSE':r2,'C3_RMSE':r3,'C1_MAE':m1e,'C2_MAE':m2e,'C3_MAE':m3e,'C1_R2':r21,'C2_R2':r22,'C3_R2':r23,'C1_to_C2_pct':100*(r1-r2)/r1,'C2_to_C3_pct':100*(r2-r3)/r2,'C1_to_C3_pct':100*(r1-r3)/r1})
        for dt,yy,a,b,c,ga,rr in zip(va['Date'],y,p1,p2,p3,a2,rho):
            preds.append({'City':city,'Seed':seed,'Date':dt,'Actual':yy,'C1_Pred':a,'C2_Pred':b,'C3_Pred':c,'C2_GateAlpha':ga,'C3_Reliability':rr})
M=pd.DataFrame(rows); P=pd.DataFrame(preds)
M.to_csv(out/'ctge_7seed_metrics.csv',index=False); P.to_csv(out/'ctge_7seed_predictions.csv',index=False)
S=M.groupby('City').agg(C1_RMSE_mean=('C1_RMSE','mean'),C1_RMSE_sd=('C1_RMSE','std'),C2_RMSE_mean=('C2_RMSE','mean'),C2_RMSE_sd=('C2_RMSE','std'),C3_RMSE_mean=('C3_RMSE','mean'),C3_RMSE_sd=('C3_RMSE','std'),C3_MAE_mean=('C3_MAE','mean'),C3_MAE_sd=('C3_MAE','std'),C3_R2_mean=('C3_R2','mean'),C3_R2_sd=('C3_R2','std'),C1_to_C2_pct=('C1_to_C2_pct','mean'),C2_to_C3_pct=('C2_to_C3_pct','mean'),C1_to_C3_pct=('C1_to_C3_pct','mean')).reset_index()
S.to_csv(out/'ctge_7seed_summary.csv',index=False)
print(S.to_string(index=False))
