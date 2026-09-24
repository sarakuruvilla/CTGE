import numpy as np, pandas as pd
from ctge.stats import dm_hac, holm, exact_wilcoxon_greater
from ctge.cpcb import category, boundary_mask, transition_mask
from ctge.afsa import discrete_afsa

def test_stats():
    stat,p=dm_hac(np.array([1,2,1,3,2,1,2,3,1,2.],float),lag=2)
    assert np.isfinite(stat) and 0<=p<=1
    assert len(holm([.01,.02,.2]))==3
    w,p=exact_wilcoxon_greater([1,2,3,4,5,6,7]); assert abs(p-1/128)<1e-9

def test_cpcb():
    y=np.array([45,55,110,205,305,405]); assert list(category(y))==[0,1,2,3,4,5]
    assert boundary_mask(y,10).sum()>=4
    assert transition_mask(y).sum()==5

def test_afsa():
    grid=np.zeros((3,4)); grid[2,3]=10
    pos,score=discrete_afsa(grid,seed=1,nfish=12,iterations=20)
    assert score>=0
