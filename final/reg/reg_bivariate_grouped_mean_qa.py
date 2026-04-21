#!/usr/bin/env python3
"""
Bivariate grouped regression — QA variables as Y, roadshow features as X.
  Y = numeric columns in analyze/output/qa_analysis.csv (IPO-level mean)
  X = each numeric feature in analyze/output/{verbal_sentiment, vocal_features, visual_gaze}
  Groups: am (09:xx start) | pm (14:xx start)

  USE_CONTROLS : IPO-level controls (ipo_issue_price, ipo_pe_diluted, ipo_pb, ipo_proceeds_gross)
  USE_FE       : year fixed effects (from event_date in car_cav_windows_mean.csv)
  SE           : HC3 always — one obs per IPO, no clustering.

  Each run produces two regression columns per record:
    coef/se/tstat/pvalue/r2        — bivariate (+ FE if USE_FE)
    coef_ctrl/…/r2_ctrl            — with controls (+ FE if USE_FE)  [if USE_CONTROLS]
  Plus pval_{ctrl} / coef_{ctrl} per control variable.

Output:
  final/reg/reg_bivariate_grouped_mean_qa_ctrl_fe.csv
"""

import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Bivariate grouped regression — QA outcomes")
    p.add_argument("--mkt-mod", action=argparse.BooleanOptionalAction, default=False,
                   help="Add market moderation: ret_4w_sh000300 + X*mkt interaction")
    p.add_argument("--pca",    action=argparse.BooleanOptionalAction, default=False,
                   help="Use 推介 PCA scores (final/pca/pca_scores_推介.csv) as X instead of raw features")
    p.add_argument("--ife",    action=argparse.BooleanOptionalAction, default=False,
                   help="Add CSRC industry fixed effects (csrc3 dummies)")
    p.add_argument("--pltfe",  action=argparse.BooleanOptionalAction, default=False,
                   help="Add board and platform fixed effects (board_fe, platform_fe)")
    p.add_argument("--pls",      action=argparse.BooleanOptionalAction, default=False,
                   help="Cross-dimension PLS: combine all PLS_FEATURE_COLS sources into one X matrix, "
                        "fit PLSRegression, run cumulative OLS with HC3 SE.")
    p.add_argument("--pls-ncomp", type=int, default=1,
                   help="Number of PLS latent components for cumulative regression (default: 1).")
    return p.parse_args()

_args = _parse_args()

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Configuration ─────────────────────────────────────────────────────────────
USE_CONTROLS  = True
USE_FE        = True
IFE_COL       = "csrc3"
IND_FE        = _args.ife
PLT_FE        = _args.pltfe
PLT_FE_COLS   = ["board_fe", "platform_fe"]
CTRL_COLS     = ["ipo_log_size", "ipo_pe_diluted", "ipo_shares_issued", "ipo_price", "duration"]
WINSORIZE     = True
WINSOR_BOUNDS = (0.01, 0.99)
MKT_MOD       = _args.mkt_mod
MKT_COL       = "ret_4w_sh000300"
PCA_MODE      = _args.pca
PLS_MODE      = _args.pls
PLS_NCOMP     = _args.pls_ncomp

# ── PLS feature whitelist (mirrors pca_combined_tui.py GROUPS) ────────────────
PLS_FEATURE_COLS = {
    "verbal":     ["ann_positive_ratio", "ann_negative_ratio",
                   "social_positive_ratio", "social_negative_ratio"],
    "vocal":      ["f0_std", "f0_mean", "f0_slope", "f0_range",
                   "rms_dynamic_range", "rms_cv", "articulation_rate", "pause_rate"],
    "visual":     ["gaze_at_camera_ratio_10", "gaze_x_mean", "gaze_x_std",
                   "gaze_y_mean", "gaze_y_std", "head_frontal_ratio_10",
                   "head_pitch_mean", "head_pitch_std", "head_yaw_mean", "head_yaw_std"],
    "visual_fer": ["positive_ratio", "negative_ratio", "neutral_ratio",
                   "emo_happy", "emo_neutral", "emo_sad"],
}

SESSION_推介 = "推介"
SESSION_答谢 = "答谢"

_active_sessions = [SESSION_推介] if PCA_MODE else [SESSION_推介, SESSION_答谢, "mean"]

# ── 1. Load QA analysis → Y variables (IPO-level mean) ───────────────────────
qa_raw = pd.read_csv(ROOT / "analyze/output/qa_analysis.csv")
qa_raw["ipo_id"] = qa_raw["index2009"].str.strip()

# Drop non-numeric / meta columns; keep only numeric Y columns
META_QA = {"file_stem", "index2009", "ipo_id", "error", "q_method", "a_method"}
y_candidates = [
    c for c in qa_raw.columns
    if c not in META_QA
    and pd.api.types.is_numeric_dtype(qa_raw[c])
]
y_candidates = [c for c in y_candidates if c in {"qa_pairs", "n_unique_questioners"}]  # TEMP: restrict Y

# Drop rows with errors
if "error" in qa_raw.columns:
    qa_raw = qa_raw[qa_raw["error"].isna() | (qa_raw["error"].astype(str).str.strip() == "")]

qa_mean = qa_raw.groupby("ipo_id")[y_candidates].mean().reset_index()
print(f"QA Y: {len(qa_mean)} IPOs, {len(y_candidates)} Y cols")
print(f"Y cols: {y_candidates}")

# ── 2. Load event_year from CAR data (for FE) and controls ───────────────────
car = pd.read_csv(ROOT / "carv/output/car_cav_windows_mean.csv",
                  usecols=["ipo_id", "event_date"])
car["event_year"] = pd.to_datetime(car["event_date"]).dt.year
car_meta = car.drop_duplicates("ipo_id")[["ipo_id", "event_year"]]

ctrl_present = []
if USE_CONTROLS:
    _ind_load = [IFE_COL] if IND_FE else []
    _plt_load = PLT_FE_COLS if PLT_FE else []
    _load_cols = ["ipo_id"] + CTRL_COLS + ([MKT_COL] if MKT_MOD else []) + _ind_load + _plt_load
    ctrl_src = pd.read_csv(
        ROOT / "carv/output/car_cav_windows_controls.csv",
        usecols=lambda c: c in _load_cols,
    )
    ctrl_present = [c for c in CTRL_COLS if c in ctrl_src.columns]
    _plt_fe_present = [c for c in _plt_load if c in ctrl_src.columns]
    _agg_cols = ctrl_present + ([MKT_COL] if MKT_MOD and MKT_COL in ctrl_src.columns else []) + \
                ([IFE_COL] if IND_FE and IFE_COL in ctrl_src.columns else []) + \
                _plt_fe_present
    ipo_controls = ctrl_src.groupby("ipo_id")[_agg_cols].first().reset_index()
    print(f"IPO controls: {len(ipo_controls)} IPOs, cols: {ctrl_present}")
    car_meta = car_meta.merge(ipo_controls, on="ipo_id", how="left")
if MKT_MOD and MKT_COL not in car_meta.columns:
    print(f"  WARNING: '{MKT_COL}' not found — MKT_MOD disabled")
    MKT_MOD = False

qa_df = qa_mean.merge(car_meta, on="ipo_id", how="inner")
print(f"QA + event_year rows: {len(qa_df)}")

# ── 3. Roadshow start time → am/pm group ─────────────────────────────────────
import os as _os
_ann_dir = ROOT / "anns" if _os.name == "nt" else ROOT / "../cninf-ann-scraper"
idx = pd.read_excel(_ann_dir / "IPO_index.xlsx", usecols=["INDEX2009", "开始时间"])
idx_sub = idx[["INDEX2009", "开始时间"]].copy()
idx_sub.columns = ["ipo_id", "start_time"]
idx_sub["start_time"] = idx_sub["start_time"].astype(str).str.strip()
idx_sub["group"] = np.where(
    idx_sub["start_time"].str.startswith("09"), "am",
    np.where(idx_sub["start_time"].str.startswith("14"), "pm", None),
)
idx_sub = idx_sub[idx_sub["group"].notna()][["ipo_id", "group"]].drop_duplicates("ipo_id")
grp_counts = idx_sub["group"].value_counts()
print(f"IPOs in index: am={grp_counts.get('am',0)}, pm={grp_counts.get('pm',0)}")

qa_grp = qa_df.merge(idx_sub, on="ipo_id", how="inner")
print(f"IPO rows after group filter: {len(qa_grp)} "
      f"(am={(qa_grp['group']=='am').sum()}, pm={(qa_grp['group']=='pm').sum()})")

# ── 4. Load X features (verbal, vocal, visual) ───────────────────────────────
META_X = {"file_stem", "method", "error"}
sources = {
    "verbal":     "analyze/output/verbal_sentiment.csv",
    "vocal":      "analyze/output/vocal_features.csv",
    "visual":     "analyze/output/visual_gaze.csv",
    "visual_fer": "analyze/output/visual_fer.csv",
}
if PCA_MODE:
    sources = {"pca": "final/pca/pca_scores_combined_tui.csv"}

def load_agg(path, session_filter=None):
    df = pd.read_csv(path)
    df["ipo_id"]   = df["file_stem"].str.split("_").str[0]
    df["_session"] = df["file_stem"].str.split("_").str[-1]
    if session_filter is not None:
        df = df[df["_session"] == session_filter]
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")]
    xcols = [
        c for c in df.columns
        if c not in META_X and c not in ("ipo_id", "_session")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df.groupby("ipo_id")[xcols].mean().reset_index(), xcols

session_variants = {s: {} for s in _active_sessions}
for name, rel in sources.items():
    agg_tui, xcols = load_agg(ROOT / rel, session_filter=SESSION_推介)
    session_variants[SESSION_推介][name] = (agg_tui, xcols)
    if not PCA_MODE:
        agg_da,  _ = load_agg(ROOT / rel, session_filter=SESSION_答谢)
        agg_avg, _ = load_agg(ROOT / rel, session_filter=None)
        session_variants[SESSION_答谢][name] = (agg_da,  xcols)
        session_variants["mean"][name]       = (agg_avg, xcols)
        print(f"{name}: 推介={len(agg_tui)}, 答谢={len(agg_da)}, mean={len(agg_avg)}, {len(xcols)} X cols")
    else:
        print(f"{name}: 推介={len(agg_tui)}, {len(xcols)} X cols (PCA mode)")

# ── PLS: whitelist filter + cross-dimension combined build ──────────────────
if PLS_MODE:
    for sess_label in list(session_variants.keys()):
        for src_name in list(session_variants[sess_label].keys()):
            x_df, x_cols = session_variants[sess_label][src_name]
            whitelist = PLS_FEATURE_COLS.get(src_name)
            if whitelist is not None:
                x_cols_pls = [c for c in x_cols if c in whitelist]
                skipped    = [c for c in x_cols if c not in whitelist]
                if skipped:
                    print(f"  [PLS] {src_name}/{sess_label}: keeping {len(x_cols_pls)} PCA cols")
                session_variants[sess_label][src_name] = (x_df, x_cols_pls)

pls_combined_data = {}
if PLS_MODE:
    for sess_label in session_variants:
        comb_df = None; comb_cols = []
        for src_name, (x_df, x_cols) in session_variants[sess_label].items():
            if not x_cols: continue
            rename  = {c: f"{src_name}_{c}" for c in x_cols}
            df_pref = x_df[["ipo_id"] + x_cols].rename(columns=rename)
            comb_cols.extend(rename.values())
            comb_df = df_pref if comb_df is None else comb_df.merge(df_pref, on="ipo_id", how="inner")
        if comb_df is not None:
            pls_combined_data[sess_label] = (comb_df, comb_cols)
            print(f"  [PLS combined] {sess_label}: {len(comb_df)} IPOs, {len(comb_cols)} features")

# ── 5. Regression engine ──────────────────────────────────────────────────────
def run_ols_hc3(y_c, X_mat):
    return sm.OLS(y_c, X_mat).fit(cov_type="HC3")

def finite_mask(arr):
    return np.isfinite(np.asarray(arr, dtype=float))

def maybe_winsorize(arr):
    a = np.asarray(arr, dtype=float)
    lo = np.nanpercentile(a, WINSOR_BOUNDS[0] * 100)
    hi = np.nanpercentile(a, WINSOR_BOUNDS[1] * 100)
    return np.clip(a, lo, hi)

def _drop_const_cols(mat):
    """Return (cleaned_mat, kept_bool_mask) — removes zero-variance columns."""
    keep = np.std(mat, axis=0) > 0
    return mat[:, keep], keep

def _platform_fe_dummies(series):
    """Multi-hot encode comma-separated platform strings into per-platform binary columns."""
    all_plats = sorted({p.strip() for s in series.dropna() for p in str(s).split(",") if p.strip()})
    mat = np.zeros((len(series), len(all_plats)), dtype=float)
    for i, val in enumerate(series):
        if pd.notna(val):
            for p in str(val).split(","):
                p = p.strip()
                if p in all_plats:
                    mat[i, all_plats.index(p)] = 1.0
    return mat

def safe_add_constant(mat):
    """Prepend intercept after dropping zero-variance columns.

    statsmodels' add_constant silently skips adding the intercept when it
    detects an existing zero-variance column (e.g. a winsorised control that
    collapsed to a constant in a subgroup).  This shifts all param indices by
    -1, causing coef_ctrl to silently report a *control* coefficient instead
    of the X coefficient.  This helper avoids that by dropping such columns
    first and prepending ones explicitly.

    Returns (X_mat, kept_mask) where kept_mask is a boolean array over the
    *input* columns (before the prepended intercept).
    """
    clean, keep = _drop_const_cols(mat)
    X = np.column_stack([np.ones((clean.shape[0], 1)), clean])
    return X, keep

def run_regressions(qa_grp, y_cols, session_variants, ctrl_cols, use_fe,
                    mkt_mod=False, mkt_col=None, ind_fe=False, ind_col=None,
                    plt_fe_cols=None):
    records = []
    for sess_label, src_dict in session_variants.items():
        for src, (x_df, x_cols) in src_dict.items():
            merged = qa_grp.merge(x_df, on="ipo_id", how="inner").reset_index(drop=True)
            for grp in ("am", "pm", "all"):
                sub = (merged if grp == "all" else merged[merged["group"] == grp]).reset_index(drop=True)

                fe_parts = []
                if use_fe and "event_year" in sub.columns:
                    fe_parts.append(
                        pd.get_dummies(sub["event_year"], prefix="yr", drop_first=True).astype(float).values
                    )
                if ind_fe and ind_col and ind_col in sub.columns:
                    fe_parts.append(
                        pd.get_dummies(sub[ind_col], prefix="ind", drop_first=True).astype(float).values
                    )
                for _pfe in (plt_fe_cols or []):
                    if _pfe in sub.columns:
                        if _pfe == "platform_fe":
                            fe_parts.append(_platform_fe_dummies(sub[_pfe]))
                        else:
                            fe_parts.append(
                                pd.get_dummies(sub[_pfe], prefix=_pfe, drop_first=True).astype(float).values
                            )
                if grp == "all" and "group" in sub.columns:
                    fe_parts.append((sub["group"] == "am").astype(float).values.reshape(-1, 1))
                yr_arr = np.column_stack(fe_parts) if fe_parts else None

                mkt_arr_full = None
                if mkt_mod and mkt_col and mkt_col in sub.columns:
                    mkt_arr_full = maybe_winsorize(
                        sub[mkt_col].to_numpy(dtype=float, na_value=np.nan)
                    )

                for y_col in y_cols:
                    y_arr = maybe_winsorize(sub[y_col].to_numpy(dtype=float, na_value=np.nan))
                    for x_col in x_cols:
                        x_arr = maybe_winsorize(sub[x_col].to_numpy(dtype=float, na_value=np.nan))

                        base_ok = finite_mask(y_arr) & finite_mask(x_arr)
                        if mkt_mod and mkt_arr_full is not None:
                            base_ok &= finite_mask(mkt_arr_full)
                        if base_ok.sum() < 15:
                            continue

                        y_b = y_arr[base_ok]
                        x_b = x_arr[base_ok]

                        rec = {
                            "session":  sess_label,
                            "group":    grp,
                            "y_col":    y_col,
                            "x_source": src,
                            "x_col":    x_col,
                        }

                        # ── Bivariate (± FE ± mkt moderation) ────────────────
                        if mkt_mod and mkt_arr_full is not None:
                            mkt_b      = mkt_arr_full[base_ok]
                            interact_b = x_b * mkt_b
                            core_bi    = np.column_stack([x_b, mkt_b, interact_b])
                        else:
                            core_bi = x_b.reshape(-1, 1)

                        if use_fe and yr_arr is not None:
                            X_bi, _ = safe_add_constant(np.column_stack([core_bi, yr_arr[base_ok]]))
                        else:
                            X_bi, _ = safe_add_constant(core_bi)
                        try:
                            res_bi = run_ols_hc3(y_b, X_bi)
                            rec.update({
                                "n_obs":  int(base_ok.sum()),
                                "const":  res_bi.params[0],
                                "coef":   res_bi.params[1],
                                "se":     res_bi.bse[1],
                                "tstat":  res_bi.tvalues[1],
                                "pvalue": res_bi.pvalues[1],
                                "r2":     res_bi.rsquared,
                            })
                            if mkt_mod and mkt_arr_full is not None:
                                rec.update({
                                    "coef_mkt":        res_bi.params[2],
                                    "pval_mkt":        res_bi.pvalues[2],
                                    "coef_interact":   res_bi.params[3],
                                    "se_interact":     res_bi.bse[3],
                                    "tstat_interact":  res_bi.tvalues[3],
                                    "pvalue_interact": res_bi.pvalues[3],
                                })
                        except Exception as e:
                            rec["error_bi"] = str(e)

                        # ── With controls (± FE ± mkt moderation) ────────────
                        if ctrl_cols:
                            ctrl_arr = sub[ctrl_cols].to_numpy(dtype=float).copy()
                            for j in range(ctrl_arr.shape[1]):
                                ctrl_arr[:, j] = maybe_winsorize(ctrl_arr[:, j])
                            ctrl_ok = base_ok.copy()
                            for j in range(ctrl_arr.shape[1]):
                                ctrl_ok &= finite_mask(ctrl_arr[:, j])

                            y_c2  = y_arr[ctrl_ok]
                            x_c2  = x_arr[ctrl_ok]
                            ctrlv = ctrl_arr[ctrl_ok]

                            rec["n_obs_ctrl"] = int(ctrl_ok.sum())

                            if ctrl_ok.sum() >= 15:
                                if mkt_mod and mkt_arr_full is not None:
                                    mkt_c2      = mkt_arr_full[ctrl_ok]
                                    interact_c2 = x_c2 * mkt_c2
                                    core_ct     = np.column_stack([x_c2, mkt_c2, interact_c2])
                                    ctrl_offset = 4
                                else:
                                    core_ct     = x_c2.reshape(-1, 1)
                                    ctrl_offset = 2

                                if use_fe and yr_arr is not None:
                                    X_ctrl, keep_c = safe_add_constant(
                                        np.column_stack([core_ct, ctrlv, yr_arr[ctrl_ok]])
                                    )
                                else:
                                    X_ctrl, keep_c = safe_add_constant(np.column_stack([core_ct, ctrlv]))
                                try:
                                    res_ct = run_ols_hc3(y_c2, X_ctrl)
                                    rec.update({
                                        "coef_ctrl":   res_ct.params[1],
                                        "se_ctrl":     res_ct.bse[1],
                                        "tstat_ctrl":  res_ct.tvalues[1],
                                        "pvalue_ctrl": res_ct.pvalues[1],
                                        "r2_ctrl":     res_ct.rsquared,
                                    })
                                    if mkt_mod and mkt_arr_full is not None:
                                        rec.update({
                                            "coef_mkt_ctrl":        res_ct.params[2],
                                            "pval_mkt_ctrl":        res_ct.pvalues[2],
                                            "coef_interact_ctrl":   res_ct.params[3],
                                            "se_interact_ctrl":     res_ct.bse[3],
                                            "tstat_interact_ctrl":  res_ct.tvalues[3],
                                            "pvalue_interact_ctrl": res_ct.pvalues[3],
                                        })
                                    
                                    # control params are after core_ct variables
                                    # We use the keep_c mask to check which control variables survived dropping.
                                    # The core_ct variables have indices 0 to ctrl_offset-1 in the original matrix (without constant).
                                    c_keep = keep_c[ctrl_offset - 1 : ctrl_offset - 1 + len(ctrl_cols)]
                                    
                                    out_idx = ctrl_offset
                                    for i, cc in enumerate(ctrl_cols):
                                        if c_keep[i]:
                                            rec[f"coef_{cc}"] = res_ct.params[out_idx]
                                            rec[f"pval_{cc}"] = res_ct.pvalues[out_idx]
                                            out_idx += 1
                                        else:
                                            rec[f"coef_{cc}"] = np.nan
                                            rec[f"pval_{cc}"] = np.nan
                                            
                                except Exception as e:
                                    rec["error_ctrl"] = str(e)
                            else:
                                rec["error_ctrl"] = "too few obs after ctrl dropna"

                        records.append(rec)
        print(f"  session='{sess_label}' done")
    return pd.DataFrame(records)

def run_regressions_pca(qa_grp, y_cols, x_df, pc_cols, ctrl_cols, use_fe,
                        ind_fe=False, ind_col=None, plt_fe_cols=None):
    """PCA cumulative regression: Y ~ pc1, Y ~ pc1+pc2, ... One record per (group, y_col, n_pcs)."""
    records = []
    merged = qa_grp.merge(x_df, on="ipo_id", how="inner").reset_index(drop=True)

    for grp in ("am", "pm", "all"):
        sub = (merged if grp == "all" else merged[merged["group"] == grp]).reset_index(drop=True)

        fe_parts = []
        if use_fe and "event_year" in sub.columns:
            fe_parts.append(
                pd.get_dummies(sub["event_year"], prefix="yr", drop_first=True).astype(float).values
            )
        if ind_fe and ind_col and ind_col in sub.columns:
            fe_parts.append(
                pd.get_dummies(sub[ind_col], prefix="ind", drop_first=True).astype(float).values
            )
        for _pfe in (plt_fe_cols or []):
            if _pfe in sub.columns:
                if _pfe == "platform_fe":
                    fe_parts.append(_platform_fe_dummies(sub[_pfe]))
                else:
                    fe_parts.append(
                        pd.get_dummies(sub[_pfe], prefix=_pfe, drop_first=True).astype(float).values
                    )
        if grp == "all" and "group" in sub.columns:
            fe_parts.append((sub["group"] == "am").astype(float).values.reshape(-1, 1))
        yr_arr = np.column_stack(fe_parts) if fe_parts else None

        ctrl_arr = None
        if ctrl_cols:
            ctrl_arr = sub[ctrl_cols].to_numpy(dtype=float).copy()
            for j in range(ctrl_arr.shape[1]):
                ctrl_arr[:, j] = maybe_winsorize(ctrl_arr[:, j])

        for y_col in y_cols:
            y_arr = maybe_winsorize(sub[y_col].to_numpy(dtype=float, na_value=np.nan))
            X_all = np.column_stack([
                maybe_winsorize(sub[pc].to_numpy(dtype=float, na_value=np.nan))
                for pc in pc_cols
            ])

            for n in range(1, len(pc_cols) + 1):
                x_subset = pc_cols[:n]
                X_n = X_all[:, :n]

                base_ok = finite_mask(y_arr)
                for j in range(n):
                    base_ok &= finite_mask(X_n[:, j])
                if base_ok.sum() < 15:
                    continue

                y_b = y_arr[base_ok]
                X_b = X_n[base_ok] if n > 1 else X_n[base_ok].reshape(-1, 1)

                rec = {
                    "group":           grp,
                    "y_col":           y_col,
                    "n_pcs":           n,
                    "x_cols_included": ",".join(x_subset),
                }

                core = X_b
                if use_fe and yr_arr is not None:
                    X_bi, _ = safe_add_constant(np.column_stack([core, yr_arr[base_ok]]))
                else:
                    X_bi, _ = safe_add_constant(core)
                try:
                    res_bi = run_ols_hc3(y_b, X_bi)
                    rec["n_obs"] = int(base_ok.sum())
                    rec["r2"]    = res_bi.rsquared
                    for i, pc in enumerate(x_subset):
                        rec[f"coef_{pc}"]   = res_bi.params[1 + i]
                        rec[f"se_{pc}"]     = res_bi.bse[1 + i]
                        rec[f"tstat_{pc}"]  = res_bi.tvalues[1 + i]
                        rec[f"pvalue_{pc}"] = res_bi.pvalues[1 + i]
                except Exception as e:
                    rec["error_bi"] = str(e)

                if ctrl_arr is not None:
                    ctrl_ok = base_ok.copy()
                    for j in range(ctrl_arr.shape[1]):
                        ctrl_ok &= finite_mask(ctrl_arr[:, j])
                    rec["n_obs_ctrl"] = int(ctrl_ok.sum())
                    if ctrl_ok.sum() >= 15:
                        y_c   = y_arr[ctrl_ok]
                        X_c   = X_n[ctrl_ok] if n > 1 else X_n[ctrl_ok].reshape(-1, 1)
                        ctrlv = ctrl_arr[ctrl_ok]
                        if use_fe and yr_arr is not None:
                            _inner = np.column_stack([X_c, ctrlv, yr_arr[ctrl_ok]])
                        else:
                            _inner = np.column_stack([X_c, ctrlv])
                        X_ctrl, _keep = safe_add_constant(_inner)
                        _ctrl_keep = _keep[n: n + len(ctrl_cols)]
                        try:
                            res_ct = run_ols_hc3(y_c, X_ctrl)
                            rec["r2_ctrl"] = res_ct.rsquared
                            for i, pc in enumerate(x_subset):
                                rec[f"coef_{pc}_ctrl"]   = res_ct.params[1 + i]
                                rec[f"se_{pc}_ctrl"]     = res_ct.bse[1 + i]
                                rec[f"tstat_{pc}_ctrl"]  = res_ct.tvalues[1 + i]
                                rec[f"pvalue_{pc}_ctrl"] = res_ct.pvalues[1 + i]
                            _kept_i = 0
                            for j, cc in enumerate(ctrl_cols):
                                if _ctrl_keep[j]:
                                    rec[f"coef_{cc}_ctrl"] = res_ct.params[1 + n + _kept_i]
                                    rec[f"pval_{cc}_ctrl"] = res_ct.pvalues[1 + n + _kept_i]
                                    _kept_i += 1
                        except Exception as e:
                            rec["error_ctrl"] = str(e)
                    else:
                        rec["error_ctrl"] = "too few obs after ctrl dropna"

                records.append(rec)
        print(f"  group='{grp}' done")
    return pd.DataFrame(records)


def run_regressions_pls_combined(qa_grp, y_cols, pls_combined_data, ctrl_cols, use_fe,
                                  pls_ncomp=1, ind_fe=False, ind_col=None, plt_fe_cols=None):
    """Cross-dimension PLS cumulative regression for QA data (HC3 SE).
    Fits PLSRegression(n_components=pls_ncomp) per (session, group, y_col) subsample,
    then runs cumulative OLS: Y~pls1, Y~pls1+pls2, ... with HC3 SE.
    One record per (session, group, y_col, n_pls).
    """
    records = []
    for sess_label, (x_df_comb, all_pls_cols) in pls_combined_data.items():
        merged = qa_grp.merge(x_df_comb, on="ipo_id", how="inner").reset_index(drop=True)
        for grp in ("am", "pm", "all"):
            sub = (merged if grp == "all" else merged[merged["group"] == grp]).reset_index(drop=True)

            fe_parts = []
            if use_fe and "event_year" in sub.columns:
                fe_parts.append(pd.get_dummies(sub["event_year"], prefix="yr",
                                               drop_first=True).astype(float).values)
            if ind_fe and ind_col and ind_col in sub.columns:
                fe_parts.append(pd.get_dummies(sub[ind_col], prefix="ind",
                                               drop_first=True).astype(float).values)
            for _pfe in (plt_fe_cols or []):
                if _pfe in sub.columns:
                    if _pfe == "platform_fe":
                        fe_parts.append(_platform_fe_dummies(sub[_pfe]))
                    else:
                        fe_parts.append(pd.get_dummies(sub[_pfe], prefix=_pfe,
                                                       drop_first=True).astype(float).values)
            if grp == "all" and "group" in sub.columns:
                fe_parts.append((sub["group"] == "am").astype(float).values.reshape(-1, 1))
            yr_arr = np.column_stack(fe_parts) if fe_parts else None

            ctrl_arr = None
            if ctrl_cols:
                ctrl_arr = sub[ctrl_cols].to_numpy(dtype=float).copy()
                for j in range(ctrl_arr.shape[1]):
                    ctrl_arr[:, j] = maybe_winsorize(ctrl_arr[:, j])

            for y_col in y_cols:
                y_arr = maybe_winsorize(sub[y_col].to_numpy(dtype=float, na_value=np.nan))
                X_raw = np.column_stack([
                    maybe_winsorize(sub[c].to_numpy(dtype=float, na_value=np.nan))
                    for c in all_pls_cols
                ])
                base_ok = finite_mask(y_arr)
                for j in range(X_raw.shape[1]):
                    base_ok &= finite_mask(X_raw[:, j])
                if base_ok.sum() < 15:
                    continue

                y_b = y_arr[base_ok]
                X_b = X_raw[base_ok]
                actual_ncomp = min(pls_ncomp, X_b.shape[1], X_b.shape[0] - 1)
                try:
                    pls_model = PLSRegression(n_components=actual_ncomp, scale=True)
                    pls_model.fit(X_b, y_b)
                    scores_b = pls_model.transform(X_b)
                    if scores_b.ndim == 1:
                        scores_b = scores_b.reshape(-1, 1)
                except Exception:
                    continue

                pls_names   = [f"pls{i+1}" for i in range(actual_ncomp)]
                base_ok_pos = np.where(base_ok)[0]

                for n in range(1, actual_ncomp + 1):
                    x_subset = pls_names[:n]
                    S_b      = scores_b[:, :n]
                    rec = {
                        "session":    sess_label, "group":   grp,
                        "y_col":      y_col,      "n_pls":   n,
                        "n_x_cols":   len(all_pls_cols),
                        "x_cols_pls": ",".join(x_subset),
                    }

                    if use_fe and yr_arr is not None:
                        X_bi, _ = safe_add_constant(np.column_stack([S_b, yr_arr[base_ok]]))
                    else:
                        X_bi, _ = safe_add_constant(S_b)
                    try:
                        res_bi = run_ols_hc3(y_b, X_bi)
                        rec["n_obs"] = int(base_ok.sum())
                        rec["r2"]    = res_bi.rsquared
                        for i, pn in enumerate(x_subset):
                            rec[f"coef_{pn}"]   = res_bi.params[1 + i]
                            rec[f"se_{pn}"]     = res_bi.bse[1 + i]
                            rec[f"tstat_{pn}"]  = res_bi.tvalues[1 + i]
                            rec[f"pvalue_{pn}"] = res_bi.pvalues[1 + i]
                    except Exception as e:
                        rec["error_bi"] = str(e)

                    if ctrl_arr is not None:
                        ctrl_ok = base_ok.copy()
                        for j in range(ctrl_arr.shape[1]):
                            ctrl_ok &= finite_mask(ctrl_arr[:, j])
                        rec["n_obs_ctrl"] = int(ctrl_ok.sum())
                        if ctrl_ok.sum() >= 15:
                            in_base = np.isin(base_ok_pos, np.where(ctrl_ok)[0])
                            y_c     = y_arr[ctrl_ok]
                            S_c     = scores_b[in_base, :n]
                            ctrlv   = ctrl_arr[ctrl_ok]
                            if use_fe and yr_arr is not None:
                                _inner = np.column_stack([S_c, ctrlv, yr_arr[ctrl_ok]])
                            else:
                                _inner = np.column_stack([S_c, ctrlv])
                            X_ctrl, _keep = safe_add_constant(_inner)
                            _ctrl_keep = _keep[n: n + len(ctrl_cols)]
                            try:
                                res_ct = run_ols_hc3(y_c, X_ctrl)
                                rec["r2_ctrl"] = res_ct.rsquared
                                for i, pn in enumerate(x_subset):
                                    rec[f"coef_{pn}_ctrl"]   = res_ct.params[1 + i]
                                    rec[f"se_{pn}_ctrl"]     = res_ct.bse[1 + i]
                                    rec[f"tstat_{pn}_ctrl"]  = res_ct.tvalues[1 + i]
                                    rec[f"pvalue_{pn}_ctrl"] = res_ct.pvalues[1 + i]
                                _kept_i = 0
                                for j, cc in enumerate(ctrl_cols):
                                    if _ctrl_keep[j]:
                                        rec[f"coef_{cc}_ctrl"] = res_ct.params[1 + n + _kept_i]
                                        rec[f"pval_{cc}_ctrl"] = res_ct.pvalues[1 + n + _kept_i]
                                        _kept_i += 1
                            except Exception as e:
                                rec["error_ctrl"] = str(e)
                        else:
                            rec["error_ctrl"] = "too few obs after ctrl dropna"
                    records.append(rec)
            print(f"  group='{grp}' done (PLS combined, n_pls={actual_ncomp})")
    return pd.DataFrame(records)


def summarise(out, ctrl_cols, use_fe):
    fe_tag = "+FE" if use_fe else ""
    res_ok = out[out["pvalue"].notna()].copy()
    print(f"\nCompleted: {len(res_ok)}  SE: HC3")
    for sess in _active_sessions:
        sub = res_ok[res_ok["session"] == sess]
        print(f"  {sess:4s} (bivariate{fe_tag}): "
              f"p<0.01={(sub.pvalue<0.01).sum()}, "
              f"p<0.05={(sub.pvalue<0.05).sum()}, "
              f"p<0.10={(sub.pvalue<0.10).sum()}")
        if ctrl_cols and "pvalue_ctrl" in out.columns:
            sub_c = sub[sub["pvalue_ctrl"].notna()]
            print(f"  {sess:4s} (ctrl{fe_tag}):     "
                  f"p<0.01={(sub_c.pvalue_ctrl<0.01).sum()}, "
                  f"p<0.05={(sub_c.pvalue_ctrl<0.05).sum()}, "
                  f"p<0.10={(sub_c.pvalue_ctrl<0.10).sum()}")

# ── 6. Run & save ─────────────────────────────────────────────────────────────
print(f"\n=== Running QA regression ({len(y_candidates)} Y cols) ===")
_mkt_suffix   = "_mkt"   if MKT_MOD  else ""
_ife_suffix   = "_ife"   if IND_FE   else ""
_pca_suffix   = "_pca"   if PCA_MODE else ""
_pls_suffix   = f"_pls{PLS_NCOMP}" if PLS_MODE else ""
_pltfe_suffix = "_pltfe" if PLT_FE   else ""
out_path = ROOT / f"final/reg/reg_bivariate_grouped_mean_qa_ctrl_fe{_ife_suffix}{_mkt_suffix}{_pca_suffix}{_pls_suffix}{_pltfe_suffix}.csv"
if PCA_MODE and PLS_MODE:
    raise ValueError("--pca and --pls are mutually exclusive")
if PCA_MODE:
    _x_df, _pc_cols = session_variants[SESSION_推介]["pca"]
    out = run_regressions_pca(qa_grp, y_candidates, _x_df, _pc_cols, ctrl_present, USE_FE,
                              ind_fe=IND_FE, ind_col=IFE_COL,
                              plt_fe_cols=_plt_fe_present if PLT_FE else None)
elif PLS_MODE:
    out = run_regressions_pls_combined(
        qa_grp, y_candidates, pls_combined_data, ctrl_present, USE_FE,
        pls_ncomp=PLS_NCOMP, ind_fe=IND_FE, ind_col=IFE_COL,
        plt_fe_cols=_plt_fe_present if PLT_FE else None,
    )
else:
    out = run_regressions(qa_grp, y_candidates, session_variants, ctrl_present, USE_FE,
                          mkt_mod=MKT_MOD, mkt_col=MKT_COL, ind_fe=IND_FE, ind_col=IFE_COL,
                          plt_fe_cols=_plt_fe_present if PLT_FE else None)
    summarise(out, ctrl_present, USE_FE)
out.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"Saved {len(out)} rows \u2192 {out_path}")
