#!/usr/bin/env python3
"""
Bivariate grouped regression — IPO-level mean X, first-day premium Y:
  Y = Retnfstd (first-day return) from basicInfos/IPO_Ipoday.xlsx
  X = each numeric feature in analyze/output/{verbal_sentiment, vocal_features, visual_gaze}
      aggregated to IPO-level mean across all roadshow segments
  Groups: am (09:xx start) | pm (14:xx start)

  Controls (USE_IPO_CONTROL): ipo_log_size, ipo_pe_diluted, ipo_shares_issued, ipo_price
    — same set as USE_IPO_CONTROL in reg_bivariate_grouped_every.py
  FE: year fixed effects (event_year)
  SE: HC3 — one obs per IPO, no clustering.

  Each row in the output corresponds to one (session × group × x_source × x_col) cell.
  Columns:
    bivariate + year FE : coef / se / tstat / pvalue / r2
    with controls + FE  : coef_ctrl / se_ctrl / tstat_ctrl / pvalue_ctrl / r2_ctrl
                          coef_{ctrl} / pval_{ctrl} per control variable

Output:
  final/reg/reg_bivariate_grouped_mean_firstday_ic_yfe.csv
"""

import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
import os as _os
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Bivariate grouped regression — first-day outcomes")
    p.add_argument("--mkt-mod", action=argparse.BooleanOptionalAction, default=False,
                   help="Add market moderation: ret_4w_sh000300 + X*mkt interaction")
    return p.parse_args()

_args = _parse_args()

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Configuration ─────────────────────────────────────────────────────────────
IPO_CONTROL   = ["ipo_log_size", "ipo_pe_diluted", "ipo_shares_issued", "ipo_price", "duration"]
USE_FE        = True
WINSORIZE     = True
WINSOR_BOUNDS = (0.01, 0.99)   # winsorize all variables at [1 %, 99 %]
MKT_MOD       = _args.mkt_mod
MKT_COL       = "ret_4w_sh000300"

SESSION_推介 = "推介"
SESSION_答谢 = "答谢"

# ── 1. Load IPO-level base table from controls CSV ────────────────────────────
_raw = pd.read_csv(ROOT / "carv/output/car_cav_windows_controls.csv")

_want = ["ipo_id", "ipo_stkcd", "event_year"] + IPO_CONTROL
ctrl_present = [c for c in IPO_CONTROL if c in _raw.columns]
_missing = [c for c in IPO_CONTROL if c not in _raw.columns]
if _missing:
    print(f"  WARNING: control columns not in CSV, skipped: {_missing}")

_extra_y   = [c for c in ["ipo_online_subs_ratio"] if c in _raw.columns]
_mkt_extra = [MKT_COL] if MKT_MOD and MKT_COL in _raw.columns else []

ipo_base = (
    _raw[["ipo_id", "ipo_stkcd", "event_year"] + ctrl_present + _extra_y + _mkt_extra]
    .drop_duplicates("ipo_id")
    .reset_index(drop=True)
)
print(f"IPO base: {len(ipo_base)} unique IPOs, controls={ctrl_present}")
print("NaN counts:", ipo_base[ctrl_present].isna().sum().to_dict())
if MKT_MOD and MKT_COL not in ipo_base.columns:
    print(f"  WARNING: '{MKT_COL}' not found — MKT_MOD disabled")
    MKT_MOD = False

# ── 2. Load first-day premium from IPO_Ipoday.xlsx ───────────────────────────
_ipoday = pd.read_excel(
    ROOT / "basicInfos/IPO_Ipoday.xlsx",
    skiprows=2,
    header=None,
    names=["Stkcd", "Listdt", "Opnprc", "Hiprc", "Loprc", "Clsprc",
           "Nshrtrd", "Nvaltrd", "Retnfstd", "Retnadfstd",
           "Pefstd", "Pnafstd", "Tnovfstd", "Mretnfstd"],
)
_ipoday = _ipoday.iloc[1:].reset_index(drop=True)  # drop units row
_ipoday["Stkcd"]    = _ipoday["Stkcd"].astype(str).str.zfill(6)
_ipoday["Retnfstd"] = pd.to_numeric(_ipoday["Retnfstd"], errors="coerce")
ipoday = _ipoday[["Stkcd", "Retnfstd"]].drop_duplicates("Stkcd")
print(f"IPO_Ipoday: {len(ipoday)} rows, Retnfstd non-null={ipoday['Retnfstd'].notna().sum()}")

# Link via ipo_stkcd
ipo_base["_stkcd_str"] = ipo_base["ipo_stkcd"].astype(str).str.zfill(6)
ipo_base = ipo_base.merge(ipoday, left_on="_stkcd_str", right_on="Stkcd", how="left")
ipo_base = ipo_base.drop(columns=["_stkcd_str", "Stkcd"])
print(f"After linking Retnfstd: non-null={ipo_base['Retnfstd'].notna().sum()} of {len(ipo_base)}")

# ── 3. Roadshow start time → am / pm group ────────────────────────────────────
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
print(f"IPOs in index: am={grp_counts.get('am', 0)}, pm={grp_counts.get('pm', 0)}")

# ── 4. Load X features (3 sources × 3 session aggregations) ──────────────────
META = {"file_stem", "method", "error"}
sources = {
    "verbal":     "analyze/output/verbal_sentiment.csv",
    "vocal":      "analyze/output/vocal_features.csv",
    "visual":     "analyze/output/visual_gaze.csv",
    "visual_fer": "analyze/output/visual_fer.csv",
}

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
        if c not in META and c not in ("ipo_id", "_session")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df.groupby("ipo_id")[xcols].mean().reset_index(), xcols

session_variants = {SESSION_推介: {}, SESSION_答谢: {}, "mean": {}}
for name, rel in sources.items():
    agg_tui, xcols = load_agg(ROOT / rel, session_filter=SESSION_推介)
    agg_da,  _     = load_agg(ROOT / rel, session_filter=SESSION_答谢)
    agg_avg, _     = load_agg(ROOT / rel, session_filter=None)
    session_variants[SESSION_推介][name] = (agg_tui, xcols)
    session_variants[SESSION_答谢][name] = (agg_da,  xcols)
    session_variants["mean"][name]       = (agg_avg, xcols)
    print(f"{name}: 推介={len(agg_tui)}, 答谢={len(agg_da)}, mean={len(agg_avg)}, {len(xcols)} X cols")

# ── 5. Merge base table with group labels ─────────────────────────────────────
base_grp = ipo_base.merge(idx_sub, on="ipo_id", how="inner")
print(f"\nIPO rows after group filter: {len(base_grp)} "
      f"(am={(base_grp['group']=='am').sum()}, pm={(base_grp['group']=='pm').sum()})")

# ── 6. Regression engine ──────────────────────────────────────────────────────
def run_ols_hc3(y_c, X_mat):
    return sm.OLS(y_c, X_mat).fit(cov_type="HC3")

def finite_mask(arr):
    return np.isfinite(np.asarray(arr, dtype=float))

def maybe_winsorize(arr):
    a = np.asarray(arr, dtype=float)
    lo = np.nanpercentile(a, WINSOR_BOUNDS[0] * 100)
    hi = np.nanpercentile(a, WINSOR_BOUNDS[1] * 100)
    return np.clip(a, lo, hi)

def maybe_maybe_winsorize(arr):
    return maybe_winsorize(arr) if WINSORIZE else np.asarray(arr, dtype=float)

def _drop_const_cols(mat):
    """Return (cleaned_mat, kept_bool_mask) — removes zero-variance columns."""
    keep = np.std(mat, axis=0) > 0
    return mat[:, keep], keep

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

def run_regressions(base_grp, y_cols, session_variants, ctrl_cols, use_fe,
                    mkt_mod=False, mkt_col=None):
    records = []
    for sess_label, src_dict in session_variants.items():
        for src, (x_df, x_cols) in src_dict.items():
            merged = base_grp.merge(x_df, on="ipo_id", how="inner").reset_index(drop=True)
            for grp in ("am", "pm"):
                sub = merged[merged["group"] == grp].reset_index(drop=True)

                # Year FE dummies (shared across Y cols)
                if use_fe and "event_year" in sub.columns:
                    yr_dum = pd.get_dummies(
                        sub["event_year"], prefix="yr", drop_first=True
                    ).astype(float)
                    yr_arr = yr_dum.values
                else:
                    yr_arr = None

                mkt_arr_full = None
                if mkt_mod and mkt_col and mkt_col in sub.columns:
                    mkt_arr_full = maybe_winsorize(
                        sub[mkt_col].to_numpy(dtype=float, na_value=np.nan)
                    )

                for y_col in y_cols:
                    y_s = maybe_winsorize(sub[y_col].to_numpy(dtype=float, na_value=np.nan))

                    for x_col in x_cols:
                        x_arr = maybe_winsorize(sub[x_col].to_numpy(dtype=float, na_value=np.nan))

                        base_ok = finite_mask(y_s) & finite_mask(x_arr)
                        if mkt_mod and mkt_arr_full is not None:
                            base_ok &= finite_mask(mkt_arr_full)
                        if base_ok.sum() < 15:
                            continue

                        y_b = y_s[base_ok]
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
                            ctrl_ok  = base_ok.copy()
                            for j in range(ctrl_arr.shape[1]):
                                ctrl_ok &= finite_mask(ctrl_arr[:, j])

                            y_c2  = y_s[ctrl_ok]
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
                                    _inner = np.column_stack([core_ct, ctrlv, yr_arr[ctrl_ok]])
                                else:
                                    _inner = np.column_stack([core_ct, ctrlv])
                                # ctrl_cols occupy positions 0..len(ctrl_cols)-1 in ctrlv
                                # keep mask covers all inner columns; ctrl block starts at
                                # core_ct.shape[1] inside _inner (before dropping).
                                _n_core = core_ct.shape[1]
                                X_ctrl, _keep = safe_add_constant(_inner)
                                # Map kept ctrl indices back to original ctrl_cols names
                                _ctrl_keep = _keep[_n_core : _n_core + len(ctrl_cols)]
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
                                    _kept_i = 0
                                    for i, cc in enumerate(ctrl_cols):
                                        if _ctrl_keep[i]:
                                            rec[f"coef_{cc}"] = res_ct.params[ctrl_offset + _kept_i]
                                            rec[f"pval_{cc}"] = res_ct.pvalues[ctrl_offset + _kept_i]
                                            _kept_i += 1
                                except Exception as e:
                                    rec["error_ctrl"] = str(e)
                            else:
                                rec["error_ctrl"] = "too few obs after ctrl dropna"

                        records.append(rec)
        print(f"  session='{sess_label}' done")
    return pd.DataFrame(records)

def summarise(out, ctrl_cols):
    fe_tag = "+YFE" if USE_FE else ""
    res_ok = out[out["pvalue"].notna()].copy()
    print(f"\n[firstday_premium] Completed: {len(res_ok)}  SE: HC3")
    for sess in (SESSION_推介, SESSION_答谢, "mean"):
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
    if ctrl_cols:
        print(f"\n  Control significance (ctrl{fe_tag}, pooled):")
        for cc in ctrl_cols:
            col = f"pval_{cc}"
            if col in out.columns:
                ok = out[col].dropna()
                print(f"    {cc}: p<0.01={(ok<0.01).sum()}, "
                      f"p<0.05={(ok<0.05).sum()}, "
                      f"p<0.10={(ok<0.10).sum()} (of {len(ok)})")

# ── 7. Run and save ───────────────────────────────────────────────────────────
y_cols_all = ["Retnfstd"] + _extra_y
_mkt_suffix = "_mkt" if MKT_MOD else ""
OUTPUT_PATH = ROOT / f"final/reg/reg_bivariate_grouped_mean_firstday_ic_yfe{_mkt_suffix}.csv"
print(f"\n=== Running firstday regressions: {y_cols_all} ===")
out = run_regressions(base_grp, y_cols_all, session_variants, ctrl_present, USE_FE,
                      mkt_mod=MKT_MOD, mkt_col=MKT_COL)
out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"Saved {len(out)} rows → {OUTPUT_PATH}")
summarise(out, ctrl_present)
