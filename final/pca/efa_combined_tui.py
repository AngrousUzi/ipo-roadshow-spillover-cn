#!/usr/bin/env python3
"""
EFA across ALL dimensions combined (verbal + vocal + visual + visual_fer),
fit and scored on the 推介 session only.

Outputs (final/pca/):
  efa_loadings_combined_tui.csv      — rotated factor loadings
  efa_communalities_combined_tui.csv — communalities & uniquenesses
  efa_explained_variance_combined_tui.csv
  efa_scores_combined_tui.csv        — factor scores per ipo_id
"""

import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# factor_analyzer uses the deprecated `force_all_finite` kwarg removed in sklearn 1.6;
# patch check_array before importing factor_analyzer so it still works.
import sklearn.utils.validation as _skv
_orig_check_array = _skv.check_array
def _patched_check_array(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs.setdefault("ensure_all_finite", kwargs.pop("force_all_finite"))
    return _orig_check_array(*args, **kwargs)
_skv.check_array = _patched_check_array

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--n-factors", type=int, default=None,
                   help="Number of factors. If omitted, use parallel analysis.")
    p.add_argument("--rotation", type=str, default="varimax",
                   choices=["varimax", "oblimin", "promax", "none"],
                   help="Factor rotation method (default: varimax).")
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
            "gaze_at_camera_ratio_10", "gaze_x_mean", "gaze_x_std",
            "gaze_y_mean", "gaze_y_std", "head_frontal_ratio_10",
            "head_pitch_mean", "head_pitch_std", "head_yaw_mean", "head_yaw_std",
        ],
    },
    "visual_fer": {
        "source": "analyze/output/visual_fer.csv",
        "cols": [
            "positive_ratio", "negative_ratio", "neutral_ratio",
            "emo_happy", "emo_neutral", "emo_sad",
        ],
    },
}


def winsorize(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    lo = np.nanpercentile(a, 1)
    hi = np.nanpercentile(a, 99)
    return np.clip(a, lo, hi)


def load_and_agg(path: Path, cols: list[str], session: str,
                 derived: dict | None = None) -> pd.DataFrame:
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


def parallel_analysis(X: np.ndarray, n_iter: int = 100,
                      percentile: int = 95, seed: int = 42) -> int:
    """Return number of factors with eigenvalues exceeding random-data percentile."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    corr = np.corrcoef(X, rowvar=False)
    obs_eigs = np.linalg.eigvalsh(corr)[::-1]

    rand_eigs = np.zeros((n_iter, p))
    for i in range(n_iter):
        rand_data = rng.standard_normal((n, p))
        rand_data = (rand_data - rand_data.mean(0)) / rand_data.std(0)
        rand_eigs[i] = np.linalg.eigvalsh(np.corrcoef(rand_data, rowvar=False))[::-1]

    threshold = np.percentile(rand_eigs, percentile, axis=0)
    n_factors = int((obs_eigs > threshold).sum())
    return max(1, n_factors)


# ── Load & merge all groups ───────────────────────────────────────────────────
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

# ── Winsorize ─────────────────────────────────────────────────────────────────
X_raw = merged[all_cols].copy()
if _args.winsor:
    for c in all_cols:
        X_raw[c] = winsorize(X_raw[c].to_numpy())

# ── Drop rows with any NaN ────────────────────────────────────────────────────
valid_mask = X_raw.notna().all(axis=1)
ipo_ids    = merged.loc[valid_mask, "ipo_id"].values
X_fit      = X_raw.loc[valid_mask].to_numpy(dtype=float)
print(f"Valid rows after dropping NaN: {len(X_fit)}")

# ── Standardize ───────────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_fit)

# ── Suitability tests ─────────────────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        chi2, p_val = calculate_bartlett_sphericity(X_scaled)
        _, kmo_model = calculate_kmo(X_scaled)
        print(f"Bartlett's Test: chi2={chi2:.2f}, p-value={p_val:.4e}")
        print(f"KMO Test: overall_kmo={kmo_model:.4f}")
    except Exception as e:
        print(f"Suitability tests failed: {e}")

# ── Determine number of factors ───────────────────────────────────────────────
max_factors = min(X_fit.shape[0] - 1, X_fit.shape[1])

if _args.n_factors:
    n_factors = min(_args.n_factors, max_factors)
    print(f"\nUsing n_factors={n_factors} (user-specified)")
else:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_factors = parallel_analysis(X_scaled)
    n_factors = min(n_factors, max_factors)
    print(f"\nParallel analysis suggests n_factors={n_factors}")

# ── EFA ───────────────────────────────────────────────────────────────────────
rotation = None if _args.rotation == "none" else _args.rotation

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method="ml")
    fa.fit(X_scaled)

loadings      = fa.loadings_          # shape (n_vars, n_factors)
communalities = fa.get_communalities()
uniquenesses  = fa.get_uniquenesses()

ev_ss = fa.get_factor_variance()      # (SS Loadings, Proportion Var, Cumulative Var)
ss_loadings   = ev_ss[0]
prop_var      = ev_ss[1]
cum_var       = ev_ss[2]

print(f"\nRotation: {_args.rotation}  |  Factors: {n_factors}")
for i in range(n_factors):
    print(f"  F{i+1}: SS={ss_loadings[i]:.3f}, PropVar={prop_var[i]:.1%}, CumVar={cum_var[i]:.1%}")

# ── Factor scores ─────────────────────────────────────────────────────────────
scores = fa.transform(X_scaled)       # shape (n_obs, n_factors)

# ── Save outputs ──────────────────────────────────────────────────────────────
out_dir   = ROOT / "final" / "pca"
fac_names = [f"f{i+1}" for i in range(n_factors)]

# Rotated loadings
loadings_df = pd.DataFrame(loadings, index=all_cols, columns=fac_names)
loadings_df.index.name = "variable"
loadings_path = out_dir / "efa_loadings_combined_tui.csv"
loadings_df.to_csv(loadings_path, encoding="utf-8-sig")
print(f"\nLoadings → {loadings_path.name}")

# Communalities & uniquenesses
comm_df = pd.DataFrame({
    "variable":     all_cols,
    "communality":  np.round(communalities, 6),
    "uniqueness":   np.round(uniquenesses, 6),
})
comm_path = out_dir / "efa_communalities_combined_tui.csv"
comm_df.to_csv(comm_path, index=False, encoding="utf-8-sig")
print(f"Communalities → {comm_path.name}")

# Explained variance
ev_df = pd.DataFrame({
    "factor":           range(1, n_factors + 1),
    "ss_loadings":      np.round(ss_loadings, 6),
    "prop_var":         np.round(prop_var, 6),
    "cumulative_var":   np.round(cum_var, 6),
})
ev_path = out_dir / "efa_explained_variance_combined_tui.csv"
ev_df.to_csv(ev_path, index=False, encoding="utf-8-sig")
print(f"Explained variance → {ev_path.name}")

# Factor scores
scores_df = pd.DataFrame(scores, columns=fac_names)
scores_df.insert(0, "ipo_id", ipo_ids)
scores_df.insert(0, "file_stem", scores_df["ipo_id"].astype(str) + "_efa_combined_" + SESSION)
scores_path = out_dir / "efa_scores_combined_tui.csv"
scores_df.to_csv(scores_path, index=False, encoding="utf-8-sig")
print(f"Scores → {scores_path.name}  ({len(scores_df)} rows, {n_factors} factors)")
