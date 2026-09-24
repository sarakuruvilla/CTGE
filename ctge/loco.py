from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from .models import make_extratrees


def pooled_fit_predict(source: pd.DataFrame,target: pd.DataFrame,base_features,context_features,seed,scale):
    """Zero-shot C1 + shrunken contextual residual transfer.

    No target-city labels are used in fitting. The scale is selected by inner source-city
    validation outside this function.
    """
    c1=make_extratrees(seed)
    c1.fit(source[base_features],source["target_AQI_t1"])
    c1_src=c1.predict(source[base_features]); c1_tgt=c1.predict(target[base_features])
    resid=source["target_AQI_t1"].to_numpy()-c1_src
    corr=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",Ridge(alpha=10.0))])
    corr.fit(source[context_features],resid)
    return c1_tgt, c1_tgt+float(scale)*corr.predict(target[context_features])
