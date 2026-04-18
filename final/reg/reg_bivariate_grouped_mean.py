#!/usr/bin/env python3
"""
Bivariate grouped regression — 3 session variants, IPO-level mean Y:
  Y = car_after_start_* / car_after_end_* from car_cav_windows_mean.csv
  X = each numeric feature in analyze/output/{verbal_sentiment, vocal_features, visual_gaze}
  Groups: am (09:xx start) | pm (14:xx start)

  USE_CONTROLS : IPO-level controls (ipo_log_size, ipo_pe_diluted, ipo_shares_issued, ipo_price)
  USE_FE       : year fixed effects (from event_date)
  SE           : HC3 always — one obs per IPO, no clustering.

  Moderation flags (--*-mod):
    --mkt-mod           : market moderation via ret_4w_sh000300
    --comp-verbal-mod   : moderation by competition_ratio from verbal_sentiment.csv (session-specific)
    --pros-verbal-mod   : moderation by prospect_ratio from verbal_sentiment.csv (session-specific)
    --q-comp-qa-mod     : moderation by q_competition_ratio from qa_analysis.csv (IPO-level)
    --a-comp-qa-mod     : moderation by a_competition_ratio from qa_analysis.csv (IPO-level)
    --q-pros-qa-mod     : moderation by q_prospect_ratio from qa_analysis.csv (IPO-level)
    --a-pros-qa-mod     : moderation by a_prospect_ratio from qa_analysis.csv (IPO-level)

  Each run produces two regression columns per record:
    coef/se/tstat/pvalue/r2        — bivariate (+ FE if USE_FE)
    coef_ctrl/…/r2_ctrl            — with controls (+ FE if USE_FE)  [if USE_CONTROLS]
  Plus pval_{ctrl} / coef_{ctrl} per control variable.
  For each active moderation (--*-mod), also outputs per moderator {name}:
    coef_x_mod_{name} / coef_mod_{name} / pval_mod_{name}
    coef_interact_{name} / se_interact_{name} / tstat_interact_{name} / pvalue_interact_{name}
    r2_mod_{name} / n_obs_mod_{name}
    [same with _ctrl suffix for controlled spec]

Outputs:
  final/reg/reg_bivariate_grouped_mean_after_start{suffix}.csv
  final/reg/reg_bivariate_grouped_mean_after_end{suffix}.csv
"""

import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Bivariate grouped regression — IPO-level mean Y")
    p.add_argument("--mkt-mod", action=argparse.BooleanOptionalAction, default=False,
                   help="Add market moderation: ret_4w_sh000300 + X*mkt interaction")
    # ── Moderation flags ──────────────────────────────────────────────────────
    p.add_argument("--comp-verbal-mod", action=argparse.BooleanOptionalAction, default=False,
                   help="Moderation by competition_ratio from verbal_sentiment.csv (session-specific)")
    p.add_argument("--pros-verbal-mod", action=argparse.BooleanOptionalAction, default=False,
                   help="Moderation by prospect_ratio from verbal_sentiment.csv (session-specific)")
    p.add_argument("--q-comp-qa-mod",   action=argparse.BooleanOptionalAction, default=False,
                   help="Moderation by q_competition_ratio from qa_analysis.csv (IPO-level)")
    p.add_argument("--a-comp-qa-mod",   action=argparse.BooleanOptionalAction, default=False,
                   help="Moderation by a_competition_ratio from qa_analysis.csv (IPO-level)")
    p.add_argument("--q-pros-qa-mod",   action=argparse.BooleanOptionalAction, default=False,
                   help="Moderation by q_prospect_ratio from qa_analysis.csv (IPO-level)")
    p.add_argument("--a-pros-qa-mod",   action=argparse.BooleanOptionalAction, default=False,
                   help="Moderation by a_prospect_ratio from qa_analysis.csv (IPO-level)")
    p.add_argument("--pca",             action=argparse.BooleanOptionalAction, default=False,
                   help="Use 推介 PCA scores (final/pca/pca_scores_推介.csv) as X instead of raw features")
    return p.parse_args()

_args = _parse_args()

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Configuration ─────────────────────────────────────────────────────────────
USE_CONTROLS  = True
USE_FE        = True
CTRL_COLS     = ["ipo_log_size", "ipo_pe_diluted", "ipo_shares_issued", "ipo_price", "duration"]
WINSORIZE     = True
WINSOR_BOUNDS = (0.01, 0.99)   # winsorize all variables at [1 %, 99 %]
MKT_MOD       = _args.mkt_mod
MKT_COL       = "ret_4w_sh000300"
PCA_MODE      = _args.pca          # use 推介 PCA scores as X features

SESSION_推介 = "推介"
SESSION_答谢 = "答谢"

# ── Moderation configuration ──────────────────────────────────────────────────
# Verbal moderators (session-specific, from verbal_sentiment.csv)
VERBAL_MOD_MAP = {
    "comp_verbal": "competition_ratio",
    "pros_verbal": "prospect_ratio",
}
# QA moderators (IPO-level, from qa_analysis.csv)
QA_MOD_MAP = {
    "q_comp_qa": "q_competition_ratio",
    "a_comp_qa": "a_competition_ratio",
    "q_pros_qa": "q_prospect_ratio",
    "a_pros_qa": "a_prospect_ratio",
}

active_verbal_mods = {k: v for k, v in VERBAL_MOD_MAP.items()
                      if getattr(_args, k + "_mod")}
active_qa_mods     = {k: v for k, v in QA_MOD_MAP.items()
                      if getattr(_args, k + "_mod")}

# ── 1. Load CARV IPO-level mean data ─────────────────────────────────────────
car = pd.read_csv(ROOT / "carv/output/car_cav_windows_mean.csv")
car["event_year"] = pd.to_datetime(car["event_date"]).dt.year

y_col_groups = {
    "after_start": [c for c in car.columns if c.startswith("car_after_start_")],
    "after_end":   [c for c in car.columns if c.startswith("car_after_end_")],
}
print(f"Y groups: { {k: len(v) for k, v in y_col_groups.items()} }")
print(f"car_cav_windows_mean rows: {len(car)}")
print(f"Years: {sorted(car['event_year'].dropna().unique().tolist())}")

# ── 1b. Load IPO-level controls ───────────────────────────────────────────────
ctrl_present = []
if USE_CONTROLS:
    _load_cols = ["ipo_id"] + CTRL_COLS + ([MKT_COL] if MKT_MOD else [])
    ctrl_src = pd.read_csv(
        ROOT / "carv/output/car_cav_windows_controls.csv",
        usecols=lambda c: c in _load_cols,
    )
    ctrl_present = [c for c in CTRL_COLS if c in ctrl_src.columns]
    _agg_cols = ctrl_present + ([MKT_COL] if MKT_MOD and MKT_COL in ctrl_src.columns else [])
    ipo_controls = ctrl_src.groupby("ipo_id")[_agg_cols].first().reset_index()
    print(f"IPO controls: {len(ipo_controls)} IPOs, cols: {ctrl_present}")
    print("NaN counts:", ipo_controls[ctrl_present].isna().sum().to_dict())
    car = car.merge(ipo_controls, on="ipo_id", how="left")
if MKT_MOD and MKT_COL not in car.columns:
    print(f"  WARNING: '{MKT_COL}' not found — MKT_MOD disabled")
    MKT_MOD = False

# ── 1c. Load QA moderator data (IPO-level) ────────────────────────────────────
if active_qa_mods:
    qa_df = pd.read_csv(ROOT / "analyze/output/qa_analysis.csv")
    qa_df["ipo_id"] = qa_df["index2009"]
    qa_src_cols = list(active_qa_mods.values())
    qa_src_cols_present = [c for c in qa_src_cols if c in qa_df.columns]
    _missing_qa = [c for c in qa_src_cols if c not in qa_df.columns]
    if _missing_qa:
        print(f"  WARNING: QA mod columns not found, skipped: {_missing_qa}")
        active_qa_mods = {k: v for k, v in active_qa_mods.items()
                          if v in qa_src_cols_present}
    if active_qa_mods:
        qa_mod_df = (qa_df.groupby("ipo_id")[qa_src_cols_present]
                     .first().reset_index())
        qa_mod_df = qa_mod_df.rename(
            columns={v: f"qmod_{v}" for v in qa_src_cols_present}
        )
        car = car.merge(qa_mod_df, on="ipo_id", how="left")
        print(f"QA mods merged: {qa_src_cols_present}")

# ── 2. Roadshow start time → group ───────────────────────────────────────────
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

# ── 3. Load X features ────────────────────────────────────────────────────────
META = {"file_stem", "method", "error"}
sources = {
    "verbal":     "analyze/output/verbal_sentiment.csv",
    "vocal":      "analyze/output/vocal_features.csv",
    "visual":     "analyze/output/visual_gaze.csv",
    "visual_fer": "analyze/output/visual_fer.csv",
}
# When --pca, replace sources with pre-computed combined-dimension 推介 PCA scores
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
        if c not in META and c not in ("ipo_id", "_session")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df.groupby("ipo_id")[xcols].mean().reset_index(), xcols

# When --pca, only 推介 session is available (PCA was fit on 推介 only)
_active_sessions = [SESSION_推介] if PCA_MODE else [SESSION_推介, SESSION_答谢]
session_variants = {s: {} for s in _active_sessions}
for name, rel in sources.items():
    agg_tui, xcols = load_agg(ROOT / rel, session_filter=SESSION_推介)
    session_variants[SESSION_推介][name] = (agg_tui, xcols)
    if not PCA_MODE:
        agg_da, _ = load_agg(ROOT / rel, session_filter=SESSION_答谢)
        session_variants[SESSION_答谢][name] = (agg_da, xcols)
        print(f"{name}: 推介={len(agg_tui)}, 答谢={len(agg_da)}, {len(xcols)} X cols")
    else:
        print(f"{name}: 推介={len(agg_tui)}, {len(xcols)} X cols (PCA mode)")

# ── 3b. Build verbal moderator aggregations (per session, from verbal source) ─
verbal_mod_aggs = {}
if active_verbal_mods:
    for sess_label in _active_sessions:
        v_agg, _ = session_variants[sess_label]["verbal"]
        cols_need = [c for c in active_verbal_mods.values() if c in v_agg.columns]
        cols_miss = [c for c in active_verbal_mods.values() if c not in v_agg.columns]
        if cols_miss:
            print(f"  WARNING: verbal mod cols missing in verbal source ({sess_label}): {cols_miss}")
        if cols_need:
            vmod = v_agg[["ipo_id"] + cols_need].copy()
            vmod = vmod.rename(columns={c: f"vmod_{c}" for c in cols_need})
            verbal_mod_aggs[sess_label] = vmod
    print(f"Verbal mod aggregations ready for sessions: {list(verbal_mod_aggs.keys())}, "
          f"cols: {list(active_verbal_mods.values())}")

# ── 4. Merge CAR with group labels ───────────────────────────────────────────
car_grp = car.merge(idx_sub, on="ipo_id", how="inner")
print(f"\nIPO rows after group filter: {len(car_grp)} "
      f"(am={(car_grp['group']=='am').sum()}, pm={(car_grp['group']=='pm').sum()})")

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
    collapsed to a constant in a subgroup), shifting all param indices by -1.
    This helper avoids that by dropping such columns first and prepending
    ones explicitly.

    Returns (X_mat, kept_mask) where kept_mask is a boolean array over the
    *input* columns (before the prepended intercept).
    """
    clean, keep = _drop_const_cols(mat)
    X = np.column_stack([np.ones((clean.shape[0], 1)), clean])
    return X, keep

def run_regressions(car_grp, y_cols, session_variants, ctrl_cols, use_fe,
                    mkt_mod=False, mkt_col=None,
                    verbal_mods=None, qa_mods=None, verbal_mod_aggs=None):
    """
    Parameters
    ----------
    verbal_mods : dict {mod_name: original_col}, optional
        Verbal moderators (session-specific). Their vmod_{col} columns are merged
        per session from verbal_mod_aggs.
    qa_mods : dict {mod_name: original_col}, optional
        QA moderators (IPO-level). Their qmod_{col} columns are already in car_grp.
    verbal_mod_aggs : dict {sess_label: DataFrame}, optional
        Pre-built verbal mod aggregations keyed by session label.
    """
    records = []
    for sess_label, src_dict in session_variants.items():
        # Verbal mod DF for this session (merged into each source merge below)
        vmod_df = None
        if verbal_mods and verbal_mod_aggs:
            vmod_df = verbal_mod_aggs.get(sess_label)

        for src, (x_df, x_cols) in src_dict.items():
            merged = car_grp.merge(x_df, on="ipo_id", how="inner").reset_index(drop=True)

            # Attach verbal mods for this session (session-specific aggregation)
            if vmod_df is not None:
                merged = merged.merge(vmod_df, on="ipo_id", how="left")

            for grp in ("am", "pm"):
                sub = merged[merged["group"] == grp].reset_index(drop=True)

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

                # Build moderator column list for this (session, src, grp)
                all_mods_cfg = []
                for mod_name, mod_col in (qa_mods or {}).items():
                    col = f"qmod_{mod_col}"
                    if col in sub.columns:
                        all_mods_cfg.append((mod_name, col))
                for mod_name, mod_col in (verbal_mods or {}).items():
                    col = f"vmod_{mod_col}"
                    if col in sub.columns:
                        all_mods_cfg.append((mod_name, col))

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

                        # Pre-compute ctrl arrays (reused by base ctrl spec and mod ctrl specs)
                        ctrl_arr = None
                        if ctrl_cols:
                            ctrl_arr = sub[ctrl_cols].to_numpy(dtype=float).copy()
                            for j in range(ctrl_arr.shape[1]):
                                ctrl_arr[:, j] = maybe_winsorize(ctrl_arr[:, j])

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
                        if ctrl_arr is not None:
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
                                    _inner = np.column_stack([core_ct, ctrlv, yr_arr[ctrl_ok]])
                                else:
                                    _inner = np.column_stack([core_ct, ctrlv])
                                _n_core = core_ct.shape[1]
                                X_ctrl, _keep = safe_add_constant(_inner)
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

                        # ── Additional moderations (competition / prospect) ────
                        for mod_name, mod_col in all_mods_cfg:
                            mod_arr_full = maybe_winsorize(
                                sub[mod_col].to_numpy(dtype=float, na_value=np.nan)
                            )
                            mod_ok = base_ok & finite_mask(mod_arr_full)

                            rec[f"n_obs_mod_{mod_name}"] = int(mod_ok.sum())
                            if mod_ok.sum() < 15:
                                continue

                            y_m = y_arr[mod_ok]; x_m = x_arr[mod_ok]
                            mval = mod_arr_full[mod_ok]
                            interact_m = x_m * mval

                            core_m = np.column_stack([x_m, mval, interact_m])
                            if use_fe and yr_arr is not None:
                                X_m, _ = safe_add_constant(
                                    np.column_stack([core_m, yr_arr[mod_ok]])
                                )
                            else:
                                X_m, _ = safe_add_constant(core_m)
                            try:
                                res_m = run_ols_hc3(y_m, X_m)
                                rec.update({
                                    f"coef_x_mod_{mod_name}":         res_m.params[1],
                                    f"coef_mod_{mod_name}":           res_m.params[2],
                                    f"pval_mod_{mod_name}":           res_m.pvalues[2],
                                    f"coef_interact_{mod_name}":      res_m.params[3],
                                    f"se_interact_{mod_name}":        res_m.bse[3],
                                    f"tstat_interact_{mod_name}":     res_m.tvalues[3],
                                    f"pvalue_interact_{mod_name}":    res_m.pvalues[3],
                                    f"r2_mod_{mod_name}":             res_m.rsquared,
                                })
                            except Exception as e:
                                rec[f"error_mod_{mod_name}"] = str(e)

                            # Moderated + controls spec
                            if ctrl_arr is not None:
                                ctrl_ok_m = mod_ok.copy()
                                for j in range(ctrl_arr.shape[1]):
                                    ctrl_ok_m &= finite_mask(ctrl_arr[:, j])

                                rec[f"n_obs_mod_{mod_name}_ctrl"] = int(ctrl_ok_m.sum())
                                if ctrl_ok_m.sum() >= 15:
                                    y_mc   = y_arr[ctrl_ok_m]; x_mc = x_arr[ctrl_ok_m]
                                    mval_c = mod_arr_full[ctrl_ok_m]
                                    interact_mc = x_mc * mval_c
                                    ctrlv_m = ctrl_arr[ctrl_ok_m]

                                    core_mc = np.column_stack([x_mc, mval_c, interact_mc])
                                    if use_fe and yr_arr is not None:
                                        _inner_mc = np.column_stack([core_mc, ctrlv_m, yr_arr[ctrl_ok_m]])
                                    else:
                                        _inner_mc = np.column_stack([core_mc, ctrlv_m])
                                    _n_core_mc = core_mc.shape[1]
                                    X_mc, _keep_mc = safe_add_constant(_inner_mc)
                                    _ctrl_keep_mc = _keep_mc[_n_core_mc : _n_core_mc + len(ctrl_cols)]
                                    try:
                                        res_mc = run_ols_hc3(y_mc, X_mc)
                                        # param layout: const(0) X(1) MOD(2) X*MOD(3) controls(4+)
                                        rec.update({
                                            f"coef_x_mod_{mod_name}_ctrl":      res_mc.params[1],
                                            f"coef_mod_{mod_name}_ctrl":        res_mc.params[2],
                                            f"pval_mod_{mod_name}_ctrl":        res_mc.pvalues[2],
                                            f"coef_interact_{mod_name}_ctrl":   res_mc.params[3],
                                            f"se_interact_{mod_name}_ctrl":     res_mc.bse[3],
                                            f"tstat_interact_{mod_name}_ctrl":  res_mc.tvalues[3],
                                            f"pvalue_interact_{mod_name}_ctrl": res_mc.pvalues[3],
                                            f"r2_mod_{mod_name}_ctrl":          res_mc.rsquared,
                                        })
                                        _kept_i_mc = 0
                                        for i, cc in enumerate(ctrl_cols):
                                            if _ctrl_keep_mc[i]:
                                                rec[f"coef_{cc}_mod_{mod_name}"] = res_mc.params[4 + _kept_i_mc]
                                                rec[f"pval_{cc}_mod_{mod_name}"] = res_mc.pvalues[4 + _kept_i_mc]
                                                _kept_i_mc += 1
                                    except Exception as e:
                                        rec[f"error_mod_{mod_name}_ctrl"] = str(e)
                                else:
                                    rec[f"error_mod_{mod_name}_ctrl"] = "too few obs after ctrl dropna"

                        records.append(rec)
        print(f"  session='{sess_label}' done")
    return pd.DataFrame(records)

def run_regressions_pca(car_grp, y_cols, x_df, pc_cols, ctrl_cols, use_fe):
    """
    PCA cumulative regression: for n=1..len(pc_cols), regress Y on [pc1..pcN].
    One record per (group, y_col, n_pcs).
    """
    records = []
    merged = car_grp.merge(x_df, on="ipo_id", how="inner").reset_index(drop=True)

    for grp in ("am", "pm"):
        sub = merged[merged["group"] == grp].reset_index(drop=True)

        yr_arr = None
        if use_fe and "event_year" in sub.columns:
            yr_dum = pd.get_dummies(sub["event_year"], prefix="yr", drop_first=True).astype(float)
            yr_arr = yr_dum.values

        ctrl_arr = None
        if ctrl_cols:
            ctrl_arr = sub[ctrl_cols].to_numpy(dtype=float).copy()
            for j in range(ctrl_arr.shape[1]):
                ctrl_arr[:, j] = maybe_winsorize(ctrl_arr[:, j])

        for y_col in y_cols:
            y_arr = maybe_winsorize(sub[y_col].to_numpy(dtype=float, na_value=np.nan))

            # Pre-winsorize all PC arrays
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
                X_b = X_n[base_ok]

                rec = {
                    "group":            grp,
                    "y_col":            y_col,
                    "n_pcs":            n,
                    "x_cols_included":  ",".join(x_subset),
                }

                # ── No-control spec ───────────────────────────────────────────
                core = X_b if n > 1 else X_b.reshape(-1, 1)
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

                # ── With controls ─────────────────────────────────────────────
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


def summarise(out, label, ctrl_cols, use_fe):
    fe_tag = "+FE" if use_fe else ""
    res_ok = out[out["pvalue"].notna()].copy()
    print(f"\n[{label}] Completed: {len(res_ok)}  SE: HC3")
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
    if ctrl_cols:
        print(f"\n  Control significance (ctrl{fe_tag}, pooled):")
        for cc in ctrl_cols:
            col = f"pval_{cc}"
            if col in out.columns:
                ok = out[col].dropna()
                print(f"    {cc}: p<0.01={(ok<0.01).sum()}, "
                      f"p<0.05={(ok<0.05).sum()}, "
                      f"p<0.10={(ok<0.10).sum()} (of {len(ok)})")

_mkt_suffix = "_mkt" if MKT_MOD else ""
_mod_parts = list(active_verbal_mods.keys()) + list(active_qa_mods.keys())
_mod_suffix = ("_" + "_".join(_mod_parts)) if _mod_parts else ""
_pca_suffix = "_pca" if PCA_MODE else ""

if PCA_MODE:
    # Extract the merged x_df and pc column list from session_variants (推介 only)
    _x_df, _pc_cols = session_variants[SESSION_推介]["pca"]
    for y_group, y_cols in y_col_groups.items():
        print(f"\n=== Running PCA cumulative {y_group} ({len(y_cols)} Y cols, {len(_pc_cols)} PCs) ===")
        out = run_regressions_pca(car_grp, y_cols, _x_df, _pc_cols, ctrl_present, USE_FE)
        out_path = ROOT / f"final/reg/reg_bivariate_grouped_mean_{y_group}_ctrl_fe{_pca_suffix}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(out)} rows → {out_path}")
else:
    for y_group, y_cols in y_col_groups.items():
        print(f"\n=== Running {y_group} ({len(y_cols)} Y cols) ===")
        out = run_regressions(car_grp, y_cols, session_variants, ctrl_present, USE_FE,
                              mkt_mod=MKT_MOD, mkt_col=MKT_COL,
                              verbal_mods=active_verbal_mods if active_verbal_mods else None,
                              qa_mods=active_qa_mods     if active_qa_mods     else None,
                              verbal_mod_aggs=verbal_mod_aggs if active_verbal_mods else None)
        out_path = ROOT / f"final/reg/reg_bivariate_grouped_mean_{y_group}_ctrl_fe{_mkt_suffix}{_mod_suffix}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(out)} rows → {out_path}")
        summarise(out, y_group, ctrl_present, USE_FE)
