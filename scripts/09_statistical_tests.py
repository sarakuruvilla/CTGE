#!/usr/bin/env python
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path: _sys.path.insert(0,str(_ROOT))
from pathlib import Path
import math, numpy as np, pandas as pd
from ctge.config import load_config
from ctge.stats import dm_hac,moving_block_delta_rmse_ci,holm,rank_biserial,exact_wilcoxon_greater,friedman_quade_nemenyi

cfg=load_config(); root=Path(cfg['root']); out=root/'outputs'
pred=pd.read_csv(out/'ctge_7seed_predictions.csv',parse_dates=['Date'])
comparisons=[('C1','C2'),('C2','C3'),('C1','C3')]; rows=[]
for base,new in comparisons:
    for city,g in pred.groupby('City'):
        # Average losses across seeds by date. Predictions are NOT ensembled.
        d=g.groupby('Date').apply(lambda x:pd.Series({'sq_b':np.mean((x['Actual']-x[f'{base}_Pred'])**2),'sq_n':np.mean((x['Actual']-x[f'{new}_Pred'])**2),'ae_b':np.mean(np.abs(x['Actual']-x[f'{base}_Pred'])),'ae_n':np.mean(np.abs(x['Actual']-x[f'{new}_Pred']))}),include_groups=False).reset_index()
        diff=d['sq_b']-d['sq_n']; stat,pv=dm_hac(diff,cfg['statistics']['hac_lag_days']); ci=moving_block_delta_rmse_ci(d['sq_b'],d['sq_n'],cfg['statistics']['bootstrap_reps'],cfg['statistics']['bootstrap_block_days'])
        rb=rank_biserial(d['ae_b']-d['ae_n']); rbm=np.sqrt(d['sq_b'].mean()); rnm=np.sqrt(d['sq_n'].mean())
        rows.append({'Comparison':f'{base}_vs_{new}','City':city,'N_days':len(d),'Delta_RMSE':rbm-rnm,'RMSE_gain_pct':100*(rbm-rnm)/rbm,'DM_HAC_stat':stat,'DM_HAC_p':pv,'Bootstrap_CI_low':ci[0],'Bootstrap_CI_median':ci[1],'Bootstrap_CI_high':ci[2],'Rank_biserial_abs_error':rb})
R=pd.DataFrame(rows); R['Holm_p']=np.nan
for comp,idx in R.groupby('Comparison').groups.items():R.loc[idx,'Holm_p']=holm(R.loc[idx,'DM_HAC_p'])
R['Holm_significant']=R['Holm_p']<.05; R.to_csv(out/'component_prediction_level_statistics.csv',index=False)
G=[]
for comp,x in R.groupby('Comparison'):
    w,p=exact_wilcoxon_greater(x['Delta_RMSE']); G.append({'Comparison':comp,'Cities_positive':int((x['Delta_RMSE']>0).sum()),'Wilcoxon_W':w,'Wilcoxon_p':p,'Mean_city_gain_pct':x['RMSE_gain_pct'].mean()})
pd.DataFrame(G).to_csv(out/'component_cross_city_statistics.csv',index=False)
# If comparator city-RMSE file exists, add Friedman/Quade/Nemenyi.
f=out/'external_comparator_city_rmse.csv'
if f.exists():
    cm=pd.read_csv(f).pivot(index='City',columns='Model',values='RMSE')
    res=friedman_quade_nemenyi(cm)
    pd.DataFrame([{'friedman_stat':res['friedman_stat'],'friedman_p':res['friedman_p'],'quade_F':res['quade_F'],'quade_p':res['quade_p'],'nemenyi_cd':res['nemenyi_cd']}]).to_csv(out/'multimodel_omnibus_statistics.csv',index=False)
    pd.DataFrame({'Model':list(res['average_ranks'].keys()),'AverageRank':list(res['average_ranks'].values())}).sort_values('AverageRank').to_csv(out/'multimodel_average_ranks.csv',index=False)
    res['pairwise'].to_csv(out/'nemenyi_pairwise.csv',index=False)
print(R.to_string(index=False))

# Same-data CTGE vs iTransformer paired prediction-level test, when available.
ep=out/'external_comparator_predictions.csv'
if ep.exists():
    E=pd.read_csv(ep,parse_dates=['Date']); comp=[]
    for city in E['City'].unique():
        c=E[E['City']==city]
        a=c[c['Model']=='CTGE']; b=c[c['Model']=='iTransformer']
        if len(a)==0 or len(b)==0: continue
        da=a.groupby('Date').apply(lambda x:pd.Series({'sq_ctge':np.mean((x['Actual']-x['Prediction'])**2),'ae_ctge':np.mean(np.abs(x['Actual']-x['Prediction']))}),include_groups=False)
        db=b.groupby('Date').apply(lambda x:pd.Series({'sq_itr':np.mean((x['Actual']-x['Prediction'])**2),'ae_itr':np.mean(np.abs(x['Actual']-x['Prediction']))}),include_groups=False)
        d=da.join(db,how='inner'); stat,pv=dm_hac(d['sq_itr']-d['sq_ctge'],cfg['statistics']['hac_lag_days']); ci=moving_block_delta_rmse_ci(d['sq_itr'],d['sq_ctge'],cfg['statistics']['bootstrap_reps'],cfg['statistics']['bootstrap_block_days']); rb=rank_biserial(d['ae_itr']-d['ae_ctge'])
        ri=np.sqrt(d['sq_itr'].mean()); rc=np.sqrt(d['sq_ctge'].mean())
        comp.append({'City':city,'iTransformer_RMSE':ri,'CTGE_RMSE':rc,'CTGE_gain_pct':100*(ri-rc)/ri,'DM_HAC_stat':stat,'DM_HAC_p':pv,'Bootstrap_CI_low':ci[0],'Bootstrap_CI_high':ci[2],'Rank_biserial':rb})
    C=pd.DataFrame(comp)
    if len(C):
        C['Holm_p']=holm(C['DM_HAC_p']); C['Holm_significant']=C['Holm_p']<.05; C.to_csv(out/'ctge_vs_itransformer_statistics.csv',index=False)
