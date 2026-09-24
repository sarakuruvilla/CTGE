from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

POLS = ["PM2_5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]


def robust_scale(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x-med))
    s = 1.4826*mad
    if not np.isfinite(s) or s < 1e-6:
        s = np.nanstd(x)
    return max(float(s if np.isfinite(s) else 1.0), 1e-6)


def context_columns(df, pollutant):
    cols = ["sin_doy", "cos_doy", "dow", "month"]
    # Strictly lagged state/history and previous-day intraday summaries.
    for c in [pollutant, "AQI"]:
        for lag in [1,2,3,7,14,30]:
            name = f"{c}_lag{lag}"
            if name in df.columns:
                cols.append(name)
    hcols = [c for c in df.columns if c.startswith("H_")]
    # shift intraday by one day at use time, so these are previous-day summaries.
    cols += hcols
    event_cols = [c for c in df.columns if c.startswith("EV_")]
    cols += event_cols
    return list(dict.fromkeys(cols))


def build_cpdm(df: pd.DataFrame, train_end="2018-12-31", random_state=42):
    """Fit context-conditioned expected pollutant models and create deviation features.

    Estimator choice (Ridge vs HistGradientBoosting) is made on an inner chronological
    split within the pre-validation period. Final expected values for validation dates are
    produced by models fit only on earlier observations.
    """
    out = df.copy().sort_values("Date").reset_index(drop=True)
    hcols = [c for c in out.columns if c.startswith("H_")]
    # previous-day intraday context
    for c in hcols:
        out[c] = out[c].shift(1)

    inner_train = out["Date"] <= "2017-09-30"
    inner_val = out["Date"].between("2017-10-01", "2017-12-31")
    fit_mask = out["Date"] <= pd.Timestamp(train_end)

    meta = []
    for p in POLS:
        target = f"{p}_t"
        if target not in out.columns:
            continue
        cols = context_columns(out, p)
        cols = [c for c in cols if c in out.columns]

        ridge = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=10.0))
        ])
        hgb = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", HistGradientBoostingRegressor(max_depth=4, learning_rate=.05, max_iter=250, random_state=random_state))
        ])
        candidates = {"ridge": ridge, "hgb": hgb}
        best_name, best_rmse = None, np.inf
        y = out[target]
        if inner_train.sum() >= 60 and inner_val.sum() >= 30:
            for name, model in candidates.items():
                oktr = inner_train & y.notna()
                okva = inner_val & y.notna()
                model.fit(out.loc[oktr, cols], y.loc[oktr])
                pred = model.predict(out.loc[okva, cols])
                rmse = float(np.sqrt(np.mean((y.loc[okva].values-pred)**2)))
                if rmse < best_rmse:
                    best_name, best_rmse = name, rmse
        else:
            best_name = "ridge"

        model = candidates[best_name]
        okfit = fit_mask & y.notna()
        model.fit(out.loc[okfit, cols], y.loc[okfit])
        expected = model.predict(out[cols])
        # For dates in the training period these fitted values are diagnostics; downstream
        # validation/test comparisons use only dates after the fit period.
        dev = y.to_numpy() - expected
        scale = robust_scale(dev[okfit.to_numpy()])
        z = np.clip(dev/scale, -6, 6)
        prefix = f"CPDM_{p}"
        out[prefix+"_expected"] = expected
        out[prefix+"_dev"] = dev
        out[prefix+"_z"] = z
        zs = pd.Series(z, index=out.index)
        out[prefix+"_zmom1"] = zs.diff(1)
        out[prefix+"_zmom7"] = zs.diff(7)
        for w in [3,7,14,30]:
            out[prefix+f"_zmean{w}"] = zs.shift(1).rolling(w, min_periods=max(2,w//2)).mean()
            out[prefix+f"_zstd{w}"] = zs.shift(1).rolling(w, min_periods=max(2,w//2)).std()
        out[prefix+"_signpersist7"] = np.sign(zs).shift(1).rolling(7, min_periods=3).mean()
        out[prefix+"_persscore"] = out[prefix+"_zmean7"].abs() * out[prefix+"_signpersist7"].abs()
        meta.append({"pollutant": p, "estimator": best_name, "inner_rmse": best_rmse, "residual_scale": scale})

    zcols = [f"CPDM_{p}_z" for p in POLS if f"CPDM_{p}_z" in out.columns]
    if zcols:
        Z = out[zcols]
        out["CPDM_SYS_mean"] = Z.mean(axis=1)
        out["CPDM_SYS_absmean"] = Z.abs().mean(axis=1)
        out["CPDM_SYS_maxabs"] = Z.abs().max(axis=1)
        out["CPDM_SYS_dispersion"] = Z.std(axis=1)
        out["CPDM_SYS_posfrac"] = (Z > 0).mean(axis=1)
    return out, pd.DataFrame(meta)
