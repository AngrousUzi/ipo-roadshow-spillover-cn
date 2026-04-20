#!/usr/bin/env python3
"""
Calculate and plot the correlation matrix for the variables used in pca_combined_tui.py.
Outputs (final/pca/):
  pca_combined_tui_corr.xlsx
  pca_combined_tui_corr.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None)
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
            "policy_pos_ratio", "policy_neg_ratio",
        ],
    },
    "vocal": {
        "source": "analyze/output/vocal_features.csv",
        "cols": [
            "f0_cv", "f0_slope", "f0_range", "rms_dynamic_range", "rms_cv",
            "articulation_rate", "speech_rate", "pause_rate", "mean_pause_duration",
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
            "emo_angry", "emo_contempt", "emo_disgust", "emo_fear",
            "emo_happy", "emo_neutral", "emo_sad", "emo_surprise",
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

def main():
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
        if merged is None:
            merged = agg
        else:
            merged = merged.merge(agg, on="ipo_id", how="inner")
            
    X_raw = merged[all_cols].copy()
    if _args.winsor:
        for c in all_cols:
            X_raw[c] = winsorize(X_raw[c].to_numpy())
            
    valid_mask = X_raw.notna().all(axis=1)
    df_valid = X_raw.loc[valid_mask]
    
    print(f"Calculating correlation matrix for {df_valid.shape[0]} valid observations and {df_valid.shape[1]} variables...")
    corr = df_valid.corr()
    
    out_dir = Path(__file__).resolve().parent
    
    # 1. Output to Excel
    excel_path = out_dir / "pca_combined_tui_corr.xlsx"
    corr.to_excel(excel_path)
    print(f"Excel saved to: {excel_path}")
    
    # 2. Output to Image (Heatmap)
    image_path = out_dir / "pca_combined_tui_corr.png"
    plt.figure(figsize=(24, 20))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix of Combined PCA Variables (推介 Session)", fontsize=24, pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(image_path, dpi=300)
    print(f"Image saved to: {image_path}")

if __name__ == "__main__":
    main()
