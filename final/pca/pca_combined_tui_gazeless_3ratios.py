#!/usr/bin/env python3
"""
Variant: gaze_8emos
visual     → 6 gaze/head cols only.
visual_fer → only 8 emotion means (emo_angry … emo_surprise).
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--n-components", type=int, default=None)
    p.add_argument("--winsor", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()

_args = _parse_args()
ROOT = Path(_args.root) if _args.root else Path(__file__).resolve().parent.parent.parent

SESSION = "推介"

GROUPS = {
    "verbal": {
        "source": "analyze/output/verbal_sentiment.csv",
        "cols": [
            "ann_positive_ratio", "ann_negative_ratio",
            "social_positive_ratio", "social_negative_ratio",
        ],
    },
    "vocal": {
        "source": "analyze/output/vocal_features.csv",
        "cols": [
            "f0_cv", "f0_slope", "f0_range", "rms_dynamic_range", "rms_cv",
            "articulation_rate", "pause_rate",
        ],
        "derived": {"f0_cv": ("f0_std", "f0_mean")},
    },
    "visual": {
        "source": "analyze/output/visual_gaze.csv",
        "cols": [
            "gaze_at_camera_ratio_5","gaze_x_std", "gaze_y_std",
        ],
    },
    "visual_fer": {
        "source": "analyze/output/visual_fer.csv",
        "cols": [
            "positive_ratio", "negative_ratio", "neutral_ratio",
        ],
    },
}

def winsorize(arr):
    a = np.asarray(arr, dtype=float)
    lo = np.nanpercentile(a, 1)
    hi = np.nanpercentile(a, 99)
    return np.clip(a, lo, hi)

def load_and_agg(path, cols, session, derived=None):
    df = pd.read_csv(path)
    df["ipo_id"]   = df["file_stem"].str.split("_").str[0]
    df["_session"] = df["file_stem"].str.split("_").str[-1]
    df = df[df["_session"] == session]
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")]
    for dcol, (lhs, rhs) in (derived or {}).items():
        if lhs in df.columns and rhs in df.columns:
            df[dcol] = df[lhs] / df[rhs].replace(0, np.nan)
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  WARNING: missing columns skipped: {missing}")
    return df.groupby("ipo_id")[present].mean().reset_index()

merged = None
all_cols = []

for gname, gcfg in GROUPS.items():
    agg = load_and_agg(ROOT / gcfg["source"], gcfg["cols"], SESSION,
                       gcfg.get("derived", {}))
    present_cols = [c for c in gcfg["cols"] if c in agg.columns]
    rename = {c: f"{gname}_{c}" for c in present_cols}
    agg = agg.rename(columns=rename)
    prefixed = list(rename.values())
    all_cols.extend(prefixed)
    print(f"  {gname}: {len(agg)} rows, {len(prefixed)} cols")
    if merged is None:
        merged = agg
    else:
        merged = merged.merge(agg, on="ipo_id", how="inner")

print(f"\nMerged: {len(merged)} IPOs × {len(all_cols)} features")

X_raw = merged[all_cols].copy()
if _args.winsor:
    for c in all_cols:
        X_raw[c] = winsorize(X_raw[c].to_numpy())

valid_mask = X_raw.notna().all(axis=1)
ipo_ids    = merged.loc[valid_mask, "ipo_id"].values
X_fit      = X_raw.loc[valid_mask].to_numpy(dtype=float)
print(f"Valid rows after dropping NaN: {len(X_fit)}")

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        chi2, p_val = calculate_bartlett_sphericity(X_fit)
        _, kmo_model = calculate_kmo(X_fit)
        print(f"Bartlett's Test: chi2={chi2:.2f}, p-value={p_val:.4e}")
        print(f"KMO Test: overall_kmo={kmo_model:.4f}")
    except Exception as e:
        print(f"Test failed: {e}")

n_comp = _args.n_components or len(all_cols)
n_comp = min(n_comp, X_fit.shape[0] - 1, X_fit.shape[1])

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_fit)
pca      = PCA(n_components=n_comp, random_state=42)
pca.fit(X_scaled)
scores   = pca.transform(X_scaled)

ev = pca.explained_variance_ratio_
print(f"PCs retained: {n_comp}")
print("  " + " | ".join(f"PC{i+1}={ev[i]:.1%}" for i in range(min(5, len(ev)))))
print(f"  Cumulative (all {n_comp}): {ev.sum():.1%}")

out_dir  = ROOT / "final" / "pca"
pc_names = [f"pc{i+1}" for i in range(n_comp)]

loadings_df = pd.DataFrame(pca.components_.T, index=all_cols, columns=pc_names)
loadings_df.index.name = "variable"
loadings_df.to_csv(out_dir / "pca_loadings_combined_tui.csv", encoding="utf-8-sig")

ev_df = pd.DataFrame({
    "pc":                   range(1, n_comp + 1),
    "explained_var_ratio":  np.round(ev, 6),
    "cumulative_var_ratio": np.round(np.cumsum(ev), 6),
})
ev_df.to_csv(out_dir / "pca_explained_variance_combined_tui.csv", index=False, encoding="utf-8-sig")

scores_df = pd.DataFrame(scores, columns=pc_names)
scores_df.insert(0, "ipo_id", ipo_ids)
scores_df.insert(0, "file_stem", scores_df["ipo_id"].astype(str) + "_pca_combined_" + SESSION)
scores_df.to_csv(out_dir / "pca_scores_combined_tui.csv", index=False, encoding="utf-8-sig")
print(f"Scores → pca_scores_combined_tui.csv  ({len(scores_df)} rows, {n_comp} PCs)")
