#!/usr/bin/env python
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path: _sys.path.insert(0,str(_ROOT))
from pathlib import Path
import math, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, precision_recall_fscore_support
from ctge.config import load_config
from ctge.cpcb import category,boundary_mask,transition_mask
from ctge.stats import exact_wilcoxon_greater

cfg=load_config(); root=Path(cfg['root']); out=root/'outputs'
pred=pd.read_csv(out/'ctge_7seed_predictions.csv',parse_dates=['Date'])
rows=[]
for (city,seed),g in pred.groupby(['City','Seed']):
    g=g.sort_values('Date'); y=g['Actual'].to_numpy(); p1=g['C1_Pred'].to_numpy(); p3=g['C3_Pred'].to_numpy()
    yc=category(y); c1=category(p1); c3=category(p3)
    b=boundary_mask(y,cfg['cpcb']['boundary_width']); t=transition_mask(y)
    def rm(mask,p): return math.sqrt(np.mean((y[mask]-p[mask])**2)) if mask.sum() else np.nan
    def ma(mask,p): return np.mean(np.abs(y[mask]-p[mask])) if mask.sum() else np.nan
    hi=y>300
    pr1=precision_recall_fscore_support(hi,p1>300,average='binary',zero_division=0)
    pr3=precision_recall_fscore_support(hi,p3>300,average='binary',zero_division=0)
    rows.append({'City':city,'Seed':seed,'CategoryAcc_C1':accuracy_score(yc,c1),'CategoryAcc_C3':accuracy_score(yc,c3),'MacroF1_C1':f1_score(yc,c1,average='macro',zero_division=0),'MacroF1_C3':f1_score(yc,c3,average='macro',zero_division=0),'WeightedKappa_C1':cohen_kappa_score(yc,c1,weights='quadratic'),'WeightedKappa_C3':cohen_kappa_score(yc,c3,weights='quadratic'),'HighRiskF1_C1':pr1[2],'HighRiskF1_C3':pr3[2],'Boundary_RMSE_C1':rm(b,p1),'Boundary_RMSE_C3':rm(b,p3),'Boundary_RMSE_gain_pct':100*(rm(b,p1)-rm(b,p3))/rm(b,p1),'Transition_RMSE_C1':rm(t,p1),'Transition_RMSE_C3':rm(t,p3),'Transition_RMSE_gain_pct':100*(rm(t,p1)-rm(t,p3))/rm(t,p1) if t.sum() else 0.0})
R=pd.DataFrame(rows); R.to_csv(out/'cpcb_7seed_metrics.csv',index=False)
S=R.groupby('City').mean(numeric_only=True).reset_index(); S.to_csv(out/'cpcb_city_summary.csv',index=False)
for metric in ['Boundary_RMSE_gain_pct','Transition_RMSE_gain_pct']:
    vals=S[metric].to_numpy(); stat,pv=exact_wilcoxon_greater(vals)
    print(metric,'positive cities=',int((vals>0).sum()),'W=',stat,'p=',pv,'mean=',vals.mean())
# Boundary-width sensitivity without changing model predictions.
sens=[]
for w in cfg['cpcb']['boundary_sensitivity']:
    for city,g in pred.groupby('City'):
        gains=[]
        for seed,s in g.groupby('Seed'):
            y=s['Actual'].to_numpy(); b=boundary_mask(y,w); p1=s['C1_Pred'].to_numpy(); p3=s['C3_Pred'].to_numpy()
            if b.sum():
                r1=np.sqrt(np.mean((y[b]-p1[b])**2)); r3=np.sqrt(np.mean((y[b]-p3[b])**2)); gains.append(100*(r1-r3)/r1)
        sens.append({'Width':w,'City':city,'MeanGainPct':np.mean(gains)})
pd.DataFrame(sens).to_csv(out/'cpcb_boundary_sensitivity.csv',index=False)
