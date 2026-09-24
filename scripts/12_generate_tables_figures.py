#!/usr/bin/env python
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path: _sys.path.insert(0,str(_ROOT))
from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from ctge.config import load_config
from ctge.stats import friedman_quade_nemenyi

cfg=load_config(); root=Path(cfg['root']); out=root/'outputs'; figs=out/'figures'; tabs=out/'tables'; figs.mkdir(exist_ok=True); tabs.mkdir(exist_ok=True)
# Component table
s=out/'ctge_7seed_summary.csv'
if s.exists():
    df=pd.read_csv(s); df.to_csv(tabs/'Table_component_C1_C2_C3.csv',index=False)
    x=np.arange(len(df)); fig,ax=plt.subplots(figsize=(10,5)); ax.errorbar(x,df['C1_RMSE_mean'],yerr=df['C1_RMSE_sd'],fmt='o-',label='C1'); ax.errorbar(x,df['C2_RMSE_mean'],yerr=df['C2_RMSE_sd'],fmt='o-',label='C2'); ax.errorbar(x,df['C3_RMSE_mean'],yerr=df['C3_RMSE_sd'],fmt='o-',label='C3'); ax.set_xticks(x,df['City'],rotation=35,ha='right'); ax.set_ylabel('RMSE'); ax.legend(); fig.tight_layout(); fig.savefig(figs/'component_rmse_mean_sd.png',dpi=220); plt.close(fig)
# Comparator ranks and simple CD visualization
f=out/'external_comparator_city_rmse.csv'
if f.exists():
    cm=pd.read_csv(f).pivot(index='City',columns='Model',values='RMSE'); res=friedman_quade_nemenyi(cm); ranks=pd.DataFrame({'Model':res['average_ranks'].keys(),'AverageRank':res['average_ranks'].values()}).sort_values('AverageRank'); ranks.to_csv(tabs/'Table_external_average_ranks.csv',index=False); res['pairwise'].to_csv(tabs/'Table_nemenyi_pairwise.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,3)); ax.scatter(ranks['AverageRank'],np.zeros(len(ranks))); 
    for _,r in ranks.iterrows(): ax.text(r['AverageRank'],0.02,r['Model'],rotation=45,ha='left',va='bottom',fontsize=8)
    ax.hlines(0,ranks['AverageRank'].min()-.3,ranks['AverageRank'].max()+.3); ax.set_yticks([]); ax.set_xlabel(f"Average RMSE rank (Nemenyi CD={res['nemenyi_cd']:.2f})"); fig.tight_layout(); fig.savefig(figs/'critical_difference_rank_axis.png',dpi=220); plt.close(fig)
# CPCB and LOCO tables
for src,name in [('cpcb_city_summary.csv','Table_CPCB_practical.csv'),('nested_loco_summary.csv','Table_LOCO.csv'),('component_prediction_level_statistics.csv','Table_component_statistics.csv')]:
    p=out/src
    if p.exists(): pd.read_csv(p).to_csv(tabs/name,index=False)
print('Generated tables in',tabs,'and figures in',figs)
