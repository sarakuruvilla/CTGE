from __future__ import annotations
import numpy as np

THRESHOLDS=np.array([50,100,200,300,400],float)

def category(x):
    x=np.asarray(x,float)
    return np.select([x<=50,x<=100,x<=200,x<=300,x<=400],[0,1,2,3,4],default=5)

def boundary_mask(actual,width=20):
    y=np.asarray(actual,float)
    return np.min(np.abs(y[:,None]-THRESHOLDS[None,:]),axis=1)<=width

def transition_mask(actual):
    c=category(actual); return np.r_[False,c[1:]!=c[:-1]]
