from __future__ import annotations
import numpy as np
import pandas as pd


def _corr_matrix(x: pd.DataFrame):
    c = x.corr(min_periods=max(4, len(x)//3)).fillna(0.0).to_numpy()
    np.fill_diagonal(c, 0.0)
    return c


def _graph_summary(A, threshold=.6):
    vals = np.abs(A[np.triu_indices_from(A, 1)])
    if len(vals) == 0:
        return dict(absmean=np.nan, std=np.nan, density=np.nan, maxabs=np.nan, spectral=np.nan)
    spectral = float(np.max(np.abs(np.linalg.eigvalsh((A+A.T)/2)))) if A.shape[0] > 1 else 0.0
    return {
        "absmean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "density": float(np.mean(vals >= threshold)),
        "maxabs": float(np.max(vals)),
        "spectral": spectral,
    }


def build_erdm(df: pd.DataFrame, windows=(7,14,30), drift_step=7):
    """Event, Regime, and Relational Dynamics from CPDM deviation states.

    All rolling windows are trailing and end at t. Drift compares the current trailing
    graph to a graph ending drift_step days earlier.
    """
    out = df.copy().sort_values("Date").reset_index(drop=True)
    zcols = [c for c in out.columns if c.startswith("CPDM_") and c.endswith("_z")]
    if len(zcols) < 2:
        raise ValueError("CPDM z-features are required before ERDM construction")

    for w in windows:
        absmean = np.full(len(out), np.nan)
        std = np.full(len(out), np.nan)
        density = np.full(len(out), np.nan)
        maxabs = np.full(len(out), np.nan)
        spectral = np.full(len(out), np.nan)
        drift = np.full(len(out), np.nan)
        edge_absmean = np.full(len(out), np.nan)
        for i in range(len(out)):
            if i+1 < w:
                continue
            A = _corr_matrix(out.loc[i-w+1:i, zcols])
            s = _graph_summary(A)
            absmean[i], std[i], density[i], maxabs[i], spectral[i] = s["absmean"], s["std"], s["density"], s["maxabs"], s["spectral"]
            if i-drift_step+1 >= w:
                B = _corr_matrix(out.loc[i-drift_step-w+1:i-drift_step, zcols])
                D = A-B
                drift[i] = float(np.linalg.norm(D, ord="fro")/np.sqrt(D.size))
                edge_absmean[i] = float(np.mean(np.abs(D[np.triu_indices_from(D, 1)])))
        out[f"ERDM_absmean{w}"] = absmean
        out[f"ERDM_std{w}"] = std
        out[f"ERDM_density{w}"] = density
        out[f"ERDM_maxabs{w}"] = maxabs
        out[f"ERDM_spectral{w}"] = spectral
        out[f"ERDM_drift{w}_frob"] = drift
        out[f"ERDM_edge_absmean{w}"] = edge_absmean

    # Alias names retained for frozen C3 configuration compatibility.
    if "ERDM_drift7_frob" in out:
        out["LER_drift7_frob"] = out["ERDM_drift7_frob"]
    if "ERDM_edge_absmean30" in out:
        out["LER_edge_absmean30"] = out["ERDM_edge_absmean30"]
    if "CPDM_SYS_absmean" in out:
        out["LCP_SYS_absmean"] = out["CPDM_SYS_absmean"]

    # Event/regime interactions; event file is optional but, if present, columns use EV_ prefix.
    ev = [c for c in out.columns if c.startswith("EV_")]
    base = [c for c in ["ERDM_drift7_frob", "ERDM_edge_absmean30", "CPDM_SYS_absmean"] if c in out.columns]
    for e in ev:
        for b in base:
            out[f"ERDM_INT_{e}_{b}"] = out[e].fillna(0) * out[b]
    return out
