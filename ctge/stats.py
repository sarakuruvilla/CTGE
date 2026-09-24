from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, friedmanchisquare, f, studentized_range, wilcoxon


def dm_hac(loss_diff, lag=7):
    d=np.asarray(loss_diff,float); d=d[np.isfinite(d)]
    n=len(d); mu=d.mean(); x=d-mu
    gamma0=np.dot(x,x)/n; lrv=gamma0
    L=min(lag,n-1)
    for k in range(1,L+1):
        g=np.dot(x[k:],x[:-k])/n; w=1-k/(L+1); lrv += 2*w*g
    se=math.sqrt(max(lrv,1e-12)/n)
    stat=mu/se; p=2*(1-norm.cdf(abs(stat)))
    return stat,p


def moving_block_delta_rmse_ci(sq_base,sq_new,reps=5000,block=7,seed=20260823):
    rng=np.random.default_rng(seed); a=np.asarray(sq_base,float); b=np.asarray(sq_new,float)
    n=len(a); starts=np.arange(max(1,n-block+1)); out=np.empty(reps)
    for r in range(reps):
        idx=[]
        while len(idx)<n:
            s=int(rng.choice(starts)); idx.extend(range(s,min(s+block,n)))
        idx=np.asarray(idx[:n])
        out[r]=math.sqrt(np.mean(a[idx]))-math.sqrt(np.mean(b[idx]))
    return np.percentile(out,[2.5,50,97.5])


def holm(pvals):
    p=np.asarray(pvals,float); m=len(p); order=np.argsort(p); adj=np.empty(m); run=0.0
    for r,i in enumerate(order):
        run=max(run,(m-r)*p[i]); adj[i]=min(1.0,run)
    return adj


def rank_biserial(diff):
    d=np.asarray(diff,float); d=d[np.isfinite(d)&(d!=0)]
    if len(d)==0:return 0.0
    ranks=rankdata(np.abs(d)); wp=ranks[d>0].sum(); wm=ranks[d<0].sum()
    return float((wp-wm)/(wp+wm))


def exact_wilcoxon_greater(diffs):
    d=np.asarray(diffs,float); d=d[np.isfinite(d)&(d!=0)]
    if len(d)==0:return np.nan,np.nan
    w=wilcoxon(d,alternative="greater",zero_method="wilcox",method="exact")
    return float(w.statistic),float(w.pvalue)


def friedman_quade_nemenyi(matrix: pd.DataFrame, alpha=.05):
    """Rows=blocks/cities, columns=models, lower values are better."""
    X=matrix.to_numpy(float); n,k=X.shape
    fr=friedmanchisquare(*[X[:,j] for j in range(k)])
    ranks=np.vstack([rankdata(row,method="average") for row in X])
    avg=ranks.mean(axis=0)

    # Quade test: rank treatments within block; weight blocks by within-block range rank.
    block_ranges=X.max(axis=1)-X.min(axis=1)
    Q=rankdata(block_ranges,method="average")
    centered=ranks-(k+1)/2
    A=centered*Q[:,None]
    Abar=A.mean(axis=0)
    num=n*(n-1)*np.sum(Abar**2)
    den=np.sum((A-Abar)**2)
    qF=((k-1)*num/den) if den>0 else np.nan
    qp=1-f.cdf(qF,k-1,(k-1)*(n-1)) if np.isfinite(qF) else np.nan

    # Nemenyi critical difference using Studentized-range q_alpha / sqrt(2).
    qcrit=studentized_range.ppf(1-alpha,k,np.inf)/np.sqrt(2)
    cd=qcrit*np.sqrt(k*(k+1)/(6*n))
    pair=[]
    se=np.sqrt(k*(k+1)/(6*n))
    for i in range(k):
        for j in range(i+1,k):
            delta=abs(avg[i]-avg[j])
            q=delta/se*np.sqrt(2)
            p=float(studentized_range.sf(q,k,np.inf))
            pair.append({"model_a":matrix.columns[i],"model_b":matrix.columns[j],"rank_diff":delta,"p_nemenyi":p,"significant":delta>cd})
    return {
        "friedman_stat":float(fr.statistic),"friedman_p":float(fr.pvalue),
        "quade_F":float(qF),"quade_p":float(qp),"nemenyi_cd":float(cd),
        "average_ranks":dict(zip(matrix.columns,avg)),"pairwise":pd.DataFrame(pair)
    }
