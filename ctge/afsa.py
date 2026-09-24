from __future__ import annotations
import numpy as np


def discrete_afsa(score_grid, seed=101, nfish=20, iterations=35, visual=2):
    """Discrete AFSA used for auditable reliability-calibration search.

    `score_grid` is an N-dimensional array over a predeclared finite candidate space.
    Fish use prey, follow and swarm moves. No test outcomes enter the score grid.
    """
    rng = np.random.default_rng(seed)
    dims = np.array(score_grid.shape)
    def fit(x): return float(score_grid[tuple(x)])
    fish = np.column_stack([rng.integers(0, dims[i], nfish) for i in range(len(dims))])
    f = np.array([fit(x) for x in fish])
    best = fish[np.argmax(f)].copy(); best_f = float(np.max(f))
    for _ in range(iterations):
        top = fish[np.argsort(f)[-max(3,nfish//4):]]
        centroid = np.rint(top.mean(axis=0)).astype(int)
        centroid = np.clip(centroid, 0, dims-1)
        for i in range(nfish):
            cur = fish[i].copy(); cand=[]
            for _ in range(5):
                cand.append(np.clip(cur+rng.integers(-visual,visual+1,len(dims)),0,dims-1))
            cand.append(np.clip(cur+np.sign(best-cur)*rng.integers(0,2,len(dims)),0,dims-1))
            cand.append(np.clip(cur+np.sign(centroid-cur)*rng.integers(0,2,len(dims)),0,dims-1))
            vals=np.array([fit(x) for x in cand]); j=int(np.argmax(vals))
            if vals[j] > f[i]: fish[i],f[i]=cand[j],vals[j]
        j=int(np.argmax(f))
        if f[j] > best_f: best,best_f=fish[j].copy(),float(f[j])
    return best, best_f
