#!/usr/bin/env python
"""AFSA reliability-calibration search and control benchmarks.

Run after 07_run_seven_seed_ctge.py with --stage c2 if you want to re-search the
reliability configuration. For manuscript reproduction, the frozen configuration
in configs/frozen_c3_config.csv is used by default.
"""
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path: _sys.path.insert(0,str(_ROOT))
from pathlib import Path
import numpy as np, pandas as pd
from ctge.config import load_config
from ctge.afsa import discrete_afsa

cfg=load_config(); root=Path(cfg['root']); out=root/'outputs'; out.mkdir(exist_ok=True)
pred_path=out/'c2_validation_predictions.csv'
if not pred_path.exists(): raise SystemExit('Need outputs/c2_validation_predictions.csv first.')
pred=pd.read_csv(pred_path,parse_dates=['Date'])
signals=['AQI_t','abs_corr','LCP_SYS_absmean','LER_drift7_frob','LER_edge_absmean30','rb30']
qvals=np.array([.50,.60,.70,.75,.80,.85,.90,.95]); betas=np.array([.4,.6,.8,1.0,1.2,1.4,1.6,1.8])
rows=[]; seedrows=[]
for city,g in pred.groupby('City'):
    # Candidate scores use all seven forecast seeds and validation-only observations.
    grid=np.full((len(signals),len(qvals),len(betas),len(betas)),-np.inf)
    detail={}
    for si,sig in enumerate(signals):
        for qi,q in enumerate(qvals):
            vals=[]
            for seed,s in g.groupby('Seed'):
                x=s[sig].astype(float); thr=x.quantile(q)
                for bi,blo in enumerate(betas):
                    for bj,bhi in enumerate(betas):
                        c=s['C2_Pred']-s['C1_Pred']; rho=np.where(x>=thr,bhi,blo); p3=s['C1_Pred']+rho*c
                        r2=np.sqrt(np.mean((s['Actual']-s['C2_Pred'])**2)); r3=np.sqrt(np.mean((s['Actual']-p3)**2))
                        gain=100*(r2-r3)/r2
                        key=(si,qi,bi,bj); detail.setdefault(key,[]).append(gain)
    for key,vals in detail.items():
        # Robust fitness: mean gain - variability penalty; hard negative seed penalty.
        score=np.mean(vals)-0.25*np.std(vals)-2.0*max(0,-min(vals))
        grid[key]=score
    afsa_solutions=[]
    for aseed in cfg['afsa_seeds']:
        pos,score=discrete_afsa(grid,aseed); key=tuple(pos); vals=detail[key]
        rec={'City':city,'AFSASeed':aseed,'si':key[0],'qi':key[1],'bi':key[2],'bj':key[3],'Signal':signals[key[0]],'Q':qvals[key[1]],'BetaLow':betas[key[2]],'BetaHigh':betas[key[3]],'Score':score,'MeanGain':np.mean(vals),'MinSeedGain':min(vals)}
        seedrows.append(rec); afsa_solutions.append(rec)
    best=max(afsa_solutions,key=lambda r:r['Score']); rows.append(best)
pd.DataFrame(seedrows).to_csv(out/'afsa_seed_solutions.csv',index=False)
pd.DataFrame(rows).to_csv(out/'afsa_selected_config_research.csv',index=False)
print('Wrote AFSA search outputs. Frozen manuscript configs remain in configs/frozen_c3_config.csv.')
