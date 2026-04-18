#!/usr/bin/env python3
"""
PCA of multi-modal roadshow features across four variable groups.

Groups and their source columns:
  VERBAL    : ann_positive_ratio, ann_negative_ratio, ann_tone_score,
              social_positive_ratio, social_negative_ratio,
              policy_pos_ratio, policy_neg_ratio
  VOCAL     : f0_slope, f0_range, rms_dynamic_range, rms_cv,
              articulation_rate, speech_rate, pause_rate, mean_pause_duration
  VISUAL    : gaze_at_camera_ratio_10, gaze_x_mean, gaze_x_std,
              gaze_y_mean, gaze_y_std, head_frontal_ratio_10,
              head_pitch_mean, head_pitch_std, head_yaw_mean, head_yaw_std
  VISUAL_FER: positive_ratio, negative_ratio, neutral_ratio,
              emo_happy, emo_neutral, emo_sad, emo_surprise

PCA is fit on pooled data (both sessions combined) for each group,
then PC scores are extracted per session (推介 / 答谢).

Outputs (final/pca/):
  pca_loadings_{group}.csv       — variable loadings (pooled fit)
  pca_explained_variance.csv     — explained variance ratio per group and PC
  pca_scores_{session}.csv       — PC scores per ipo_id per session
                                   (includes file_stem column for regression use)
  pca_scores_all.csv             — combined scores, both sessions
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None,
                   help="Project root (default: 3 levels above this script)")
    p.add_argument("--n-components", type=int, default=None,
                   help="PCs to retain per group (default: all = n_features)")
    p.add_argument("--winsor", action=argparse.BooleanOptionalAction, default=True,
                   help="Winsorize variables at [1%%, 99%%] before PCA (default: on)")
    return p.parse_args()

_args = _parse_args()
ROOT = Path(_args.root) if _args.root else Path(__file__).resolve().parent.parent.parent

SESSION_TUI = "推介"
SESSION_DA  = "答谢"
SESSIONS    = [SESSION_TUI, SESSION_DA]

GROUPS = {
    "verbal": {
        "source": "analyze/output/verbal_sentiment.csv",
        "cols": [
            "ann_positive_ratio", "ann_negative_ratio", "ann_tone_score",
            "social_positive_ratio", "social_negative_ratio",
            "policy_pos_ratio", "policy_neg_ratio",
        ],
    },
    "vocal": {
        "source": "analyze/output/vocal_features.csv",
        "cols": [
            "f0_slope", "f0_range", "rms_dynamic_range", "rms_cv",
            "articulation_rate", "speech_rate", "pause_rate", "mean_pause_duration",
        ],
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
            "emo_happy", "emo_neutral", "emo_sad", "emo_surprise",
        ],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def winsorize(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    lo = np.nanpercentile(a, 1)
    hi = np.nanpercentile(a, 99)
    return np.clip(a, lo, hi)

def load_source(path: Path, session_filter: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ipo_id"]   = df["file_stem"].str.split("_").str[0]
    df["_session"] = df["file_stem"].str.split("_").str[-1]
    if session_filter is not None:
        df = df[df["_session"] == session_filter]
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")]
    return df

# ── Setup output ──────────────────────────────────────────────────────────────
out_dir = ROOT / "final" / "pca"
out_dir.mkdir(parents=True, exist_ok=True)

# ── Main PCA loop ─────────────────────────────────────────────────────────────
ev_records  = []          # explained variance per group × PC
sess_score_dfs = {s: None for s in SESSIONS}   # merged score DFs per session

for gname, gcfg in GROUPS.items():
    print(f"\n── {gname.upper()} ──")

    df_all = load_source(ROOT / gcfg["source"])
    cols   = [c for c in gcfg["cols"] if c in df_all.columns]
    missing = [c for c in gcfg["cols"] if c not in df_all.columns]
    if missing:
        print(f"  WARNING: missing columns skipped: {missing}")

    # Aggregate to (ipo_id, session) level
    agg_all = df_all.groupby(["ipo_id", "_session"])[cols].mean().reset_index()
    print(f"  Pooled: {len(agg_all)} (ipo_id, session) rows")

    # Winsorize
    X_raw = agg_all[cols].copy()
    if _args.winsor:
        for c in cols:
            X_raw[c] = winsorize(X_raw[c].to_numpy())

    # Drop rows with any NaN
    valid_mask = X_raw.notna().all(axis=1)
    X_fit      = X_raw.loc[valid_mask].to_numpy(dtype=float)
    print(f"  Valid rows for fitting: {len(X_fit)}")

    n_comp = _args.n_components or len(cols)
    n_comp = min(n_comp, X_fit.shape[0] - 1, X_fit.shape[1])

    # Fit PCA on pooled data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fit)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X_scaled)

    ev = pca.explained_variance_ratio_
    print(f"  PCs retained: {n_comp} | " +
          " | ".join(f"PC{i+1}={ev[i]:.1%}" for i in range(min(3, len(ev)))))
    if len(ev) > 3:
        print(f"  Cumulative (all {n_comp}): {ev.sum():.1%}")

    # Record explained variance
    for i, r in enumerate(ev):
        ev_records.append({
            "group":              gname,
            "pc":                 i + 1,
            "explained_var_ratio": round(r, 6),
            "cumulative_var_ratio": round(float(ev[: i + 1].sum()), 6),
        })

    # Save loadings (one file per group, pooled)
    pc_names    = [f"pc{i + 1}" for i in range(n_comp)]
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=cols,
        columns=pc_names,
    )
    loadings_df.index.name = "variable"
    loadings_path = out_dir / f"pca_loadings_{gname}.csv"
    loadings_df.to_csv(loadings_path, encoding="utf-8-sig")
    print(f"  Loadings → {loadings_path.name}")

    # Transform per session
    for sess in SESSIONS:
        sub = agg_all[agg_all["_session"] == sess].copy()
        X_sess = sub[cols].copy()
        if _args.winsor:
            for c in cols:
                X_sess[c] = winsorize(X_sess[c].to_numpy())

        valid_s = X_sess.notna().all(axis=1)
        ipo_ids = sub.loc[valid_s, "ipo_id"].values
        X_t     = scaler.transform(X_sess.loc[valid_s].to_numpy(dtype=float))
        scores  = pca.transform(X_t)
        print(f"  {sess}: {len(ipo_ids)} IPOs transformed")

        score_df = pd.DataFrame(
            scores,
            columns=[f"{gname}_{p}" for p in pc_names],
        )
        score_df.insert(0, "ipo_id", ipo_ids)

        # Merge into session-level accumulator
        if sess_score_dfs[sess] is None:
            sess_score_dfs[sess] = score_df
        else:
            sess_score_dfs[sess] = sess_score_dfs[sess].merge(
                score_df, on="ipo_id", how="outer"
            )

# ── Save explained variance ───────────────────────────────────────────────────
ev_df = pd.DataFrame(ev_records)
ev_path = out_dir / "pca_explained_variance.csv"
ev_df.to_csv(ev_path, index=False, encoding="utf-8-sig")
print(f"\nExplained variance → {ev_path.name}")

# ── Save per-session score CSVs ───────────────────────────────────────────────
combined_parts = []
for sess in SESSIONS:
    df = sess_score_dfs[sess]
    if df is None:
        continue
    # Add file_stem-compatible column for regression script compatibility
    df = df.copy()
    df["file_stem"] = df["ipo_id"].astype(str) + "_pca_" + sess
    # Re-order: file_stem first, then ipo_id, then scores
    pc_cols = [c for c in df.columns if c not in ("ipo_id", "file_stem")]
    df = df[["file_stem", "ipo_id"] + pc_cols]
    out_path = out_dir / f"pca_scores_{sess}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Scores ({sess}) → {out_path.name}  ({len(df)} rows, {len(pc_cols)} PCs)")

    part = df.drop(columns=["file_stem"]).copy()
    part["session"] = sess
    combined_parts.append(part)

combined_df = pd.concat(combined_parts, ignore_index=True)
combined_path = out_dir / "pca_scores_all.csv"
combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
print(f"Scores (all)    → {combined_path.name}  ({len(combined_df)} rows)")
