#!/usr/bin/env python
"""Same-data external comparator suite used by the manuscript.

Final empirical suite (TimeMixer intentionally absent): Persistence, Ridge,
HistGradientBoosting, Random Forest, XGBoost, LightGBM, iTransformer, CTGE.
All models use the same city split, one-day-ahead target, and common evaluation
calendar. iTransformer uses a 30-day lookback, so the first 29 validation days
are excluded for *all* comparators in the rank table.
"""
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path: _sys.path.insert(0,str(_ROOT))
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from ctge.config import load_config
from ctge.itransformer import fit_predict_itransformer

cfg=load_config(); root=Path(cfg['root']); p=root/'data'/'processed'; out=root/'outputs'; out.mkdir(exist_ok=True)
df=pd.read_csv(p/'ctge_feature_master.csv',parse_dates=['Date'])
ctge=pd.read_csv(out/'ctge_7seed_predictions.csv',parse_dates=['Date']) if (out/'ctge_7seed_predictions.csv').exists() else None
features=[c for c in df if c.endswith('_t') or c.endswith('_missing') or c.startswith('H_')]
LOOKBACK=30
rows=[]; pred_rows=[]

def rm(y,p):return float(np.sqrt(mean_squared_error(y,p)))
for city,g in df.groupby('City'):
    tr=g[g['Date'].between(cfg['periods']['development_train'][0],cfg['periods']['development_train'][1]) & g['target_AQI_t1'].notna()].copy().sort_values('Date')
    va=g[g['Date'].between(cfg['periods']['development_validation'][0],cfg['periods']['development_validation'][1]) & g['target_AQI_t1'].notna()].copy().sort_values('Date')
    if len(va)<LOOKBACK: continue
    va_eval=va.iloc[LOOKBACK-1:].copy(); y=va_eval['target_AQI_t1'].to_numpy(); dates=va_eval['Date'].to_numpy()

    # Persistence
    pp=va_eval['AQI_t'].to_numpy(); rows.append({'City':city,'Model':'Persistence','RMSE':rm(y,pp)})
    pred_rows += [{'City':city,'Model':'Persistence','Seed':0,'Date':d,'Actual':yy,'Prediction':ppp} for d,yy,ppp in zip(dates,y,pp)]

    models={
      'Ridge':Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',Ridge(alpha=10))]),
      'Random Forest':Pipeline([('imp',SimpleImputer(strategy='median')),('m',RandomForestRegressor(n_estimators=500,max_features=.7,min_samples_leaf=2,random_state=42,n_jobs=-1))]),
      'HistGradientBoosting':Pipeline([('imp',SimpleImputer(strategy='median')),('m',HistGradientBoostingRegressor(max_iter=400,learning_rate=.04,max_depth=6,random_state=42))]),
    }
    try:
        from xgboost import XGBRegressor
        models['XGBoost']=Pipeline([('imp',SimpleImputer(strategy='median')),('m',XGBRegressor(n_estimators=600,max_depth=4,learning_rate=.03,subsample=.9,colsample_bytree=.8,random_state=42,n_jobs=-1))])
    except Exception as e: print('XGBoost skipped:',e)
    try:
        from lightgbm import LGBMRegressor
        models['LightGBM']=Pipeline([('imp',SimpleImputer(strategy='median')),('m',LGBMRegressor(n_estimators=600,num_leaves=31,learning_rate=.03,subsample=.9,colsample_bytree=.8,random_state=42,verbosity=-1))])
    except Exception as e: print('LightGBM skipped:',e)
    for name,m in models.items():
        m.fit(tr[features],tr['target_AQI_t1']); pred=m.predict(va_eval[features]); rows.append({'City':city,'Model':name,'RMSE':rm(y,pred)})
        pred_rows += [{'City':city,'Model':name,'Seed':0,'Date':d,'Actual':yy,'Prediction':ppp} for d,yy,ppp in zip(dates,y,pred)]

    # iTransformer: same information, same common validation dates.
    imp=SimpleImputer(strategy='median'); Xtr=imp.fit_transform(tr[features]); Xva=imp.transform(va[features]); sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xva=sc.transform(Xva)
    if 'AQI_t' in features:
        ai=features.index('AQI_t'); order=[i for i in range(len(features)) if i!=ai]+[ai]; Xtr=Xtr[:,order]; Xva=Xva[:,order]
    rms=[]
    for seed in cfg['forecast_seeds']:
        pi=fit_predict_itransformer(Xtr,tr['target_AQI_t1'].to_numpy(),Xva,seed=seed,lookback=LOOKBACK,epochs=100,aqi_index=-1)
        rms.append(rm(y,pi))
        pred_rows += [{'City':city,'Model':'iTransformer','Seed':seed,'Date':d,'Actual':yy,'Prediction':ppp} for d,yy,ppp in zip(dates,y,pi)]
    rows.append({'City':city,'Model':'iTransformer','RMSE':float(np.mean(rms))})

    # Full CTGE: align to exactly the same dates; mean RMSE over seven seeds.
    if ctge is not None:
        q=ctge[ctge['City'].str.lower()==city.lower()].copy()
        rms=[]
        for seed,s in q.groupby('Seed'):
            s=s[s['Date'].isin(pd.to_datetime(dates))].sort_values('Date')
            if len(s)==len(y):
                rms.append(rm(s['Actual'],s['C3_Pred']))
                pred_rows += [{'City':city,'Model':'CTGE','Seed':int(seed),'Date':d,'Actual':yy,'Prediction':ppp} for d,yy,ppp in zip(s['Date'],s['Actual'],s['C3_Pred'])]
        if rms: rows.append({'City':city,'Model':'CTGE','RMSE':float(np.mean(rms))})
R=pd.DataFrame(rows); P=pd.DataFrame(pred_rows)
R.to_csv(out/'external_comparator_city_rmse.csv',index=False); P.to_csv(out/'external_comparator_predictions.csv',index=False)
print(R.pivot(index='City',columns='Model',values='RMSE').to_string())
