from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor


def make_extratrees(seed=42, n_estimators=300):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", ExtraTreesRegressor(
            n_estimators=n_estimators, max_features=.7, min_samples_leaf=2,
            random_state=seed, n_jobs=-1
        ))
    ])


def make_ridge(alpha=10.0):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=alpha))
    ])


def make_rf(seed=42):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", RandomForestRegressor(n_estimators=500, max_features=.7, min_samples_leaf=2, random_state=seed, n_jobs=-1))
    ])


def make_hgb(seed=42):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", HistGradientBoostingRegressor(max_iter=400, learning_rate=.04, max_depth=6, random_state=seed))
    ])


def fit_c1(train, valid, features, seed):
    m = make_extratrees(seed)
    m.fit(train[features], train["target_AQI_t1"])
    return m, m.predict(valid[features])


def gate_alpha(df, scheme, q, fit_mask=None):
    """Regime gate used by C2.

    Quantile thresholds are estimated from fit_mask only. Signals are current AQI severity,
    CPDM abnormality pressure, and ERDM relational drift; all are available at forecast origin.
    """
    fit_mask = np.asarray(fit_mask if fit_mask is not None else np.ones(len(df), dtype=bool))
    sev = df["AQI_t"].astype(float)
    def _series(primary, secondary):
        if primary in df.columns: return df[primary].astype(float)
        if secondary in df.columns: return df[secondary].astype(float)
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    cp = _series("LCP_SYS_absmean", "CPDM_SYS_absmean")
    er = _series("LER_drift7_frob", "ERDM_drift7_frob")
    def soft(s):
        tr = s[fit_mask].dropna()
        thr = float(tr.quantile(q)) if len(tr) else 0.0
        scale = float(tr.quantile(.75)-tr.quantile(.25)) if len(tr) else 1.0
        scale = max(scale, 1e-6)
        return 1/(1+np.exp(-(s-thr)/scale))
    a_sev, a_cp, a_er = soft(sev), soft(cp), soft(er)
    if scheme == "severity":
        return a_sev.to_numpy()
    if scheme == "all_soft":
        return np.nanmean(np.vstack([a_sev,a_cp,a_er]), axis=0)
    if scheme == "sev_or_cp_or_er":
        return np.nanmax(np.vstack([a_sev,a_cp,a_er]), axis=0)
    raise ValueError(f"Unknown gate scheme: {scheme}")


def fit_c2_residual(train, valid, c1_train_pred, c1_valid_pred, context_features, alpha, gamma, gate_scheme, q):
    resid = train["target_AQI_t1"].to_numpy() - np.asarray(c1_train_pred)
    model = make_ridge(alpha=alpha)
    model.fit(train[context_features], resid)
    rhat = model.predict(valid[context_features])
    # Gate thresholds are learned on training only and then applied to validation.
    combo = pd.concat([train, valid], ignore_index=True)
    alpha_all = gate_alpha(combo, gate_scheme, q, fit_mask=np.arange(len(combo)) < len(train))
    a_valid = alpha_all[len(train):]
    pred = np.asarray(c1_valid_pred) + float(gamma)*a_valid*rhat
    return model, pred, a_valid, rhat


def apply_c3_reliability(df_valid, c1_pred, c2_pred, config, history_benefit=None):
    correction = np.asarray(c2_pred) - np.asarray(c1_pred)
    signal = config["Signal"]
    thr = float(config["Threshold"])
    blo = float(config["BetaLow"])
    bhi = float(config["BetaHigh"])
    if signal == "AQI_t":
        sig = df_valid["AQI_t"].to_numpy(float)
    elif signal == "abs_corr":
        sig = np.abs(correction)
    elif signal == "rb30":
        queue = list(history_benefit or [])
        sig = []
        y = df_valid["target_AQI_t1"].to_numpy(float)
        for i in range(len(df_valid)):
            sig.append(float(np.mean(queue[-30:])) if len(queue) >= 10 else 0.0)
            queue.append(float((y[i]-c1_pred[i])**2 - (y[i]-c2_pred[i])**2))
        sig = np.asarray(sig)
    elif signal in df_valid.columns:
        sig = df_valid[signal].to_numpy(float)
    else:
        raise ValueError(f"C3 reliability signal {signal} not available")
    rho = np.where(sig >= thr, bhi, blo)
    return np.asarray(c1_pred) + rho*correction, rho
