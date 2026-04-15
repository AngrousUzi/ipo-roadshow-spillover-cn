#!/usr/bin/env python3
"""
Bivariate grouped regression — 3 session variants, peer-level Y:
  session = '推介' | '答谢' | 'mean'
  Y = car_after_start_* / car_after_end_* (12 variants each)
  X = each numeric feature in analyze/output/{verbal_sentiment, vocal_features, visual_gaze}
  Groups: am (09:xx start) | pm (14:xx start)

  Mode flags (set at top):
    USE_RIVAL_CONTROL_1 : Size, BM, ROA, Leverage
    USE_RIVAL_CONTROL_2 : PE, Is_SOE, Age_listed, Age_estab
    USE_IPO_CONTROL     : ipo_log_size, ipo_pe_diluted, ipo_shares_issued, ipo_issue_price
    YEAR_FE             : year fixed effects
    IND_FE              : industry fixed effects (requires IND_COL in controls CSV)
    MKT_MOD             : market moderation — adds ret_4w_sh000300 + X*mkt interaction

  Moderation flags (--*-mod):
    --comp-verbal-mod   : moderation by competition_ratio from verbal_sentiment.csv (session-specific)
    --pros-verbal-mod   : moderation by prospect_ratio from verbal_sentiment.csv (session-specific)
    --q-comp-qa-mod     : moderation by q_competition_ratio from qa_analysis.csv (IPO-level)
    --a-comp-qa-mod     : moderation by a_competition_ratio from qa_analysis.csv (IPO-level)
    --q-pros-qa-mod     : moderation by q_prospect_ratio from qa_analysis.csv (IPO-level)
    --a-pros-qa-mod     : moderation by a_prospect_ratio from qa_analysis.csv (IPO-level)

  Each run produces two regression columns per record:
    coef/se/tstat/pvalue/r2        — bivariate (+ active FEs)
    coef_ctrl/…/r2_ctrl            — with controls (+ active FEs)
  Plus pval_{ctrl} / coef_{ctrl} per control variable.
  When MKT_MOD is active, also outputs:
    coef_mkt / pval_mkt            — main effect of market proxy
    coef_interact / se_interact / tstat_interact / pvalue_interact
                                   — interaction X * mkt (bivariate spec)
    coef_interact_ctrl / …         — same in controlled spec
  For each active moderation (--*-mod), also outputs per moderator {name}:
    coef_x_mod_{name} / coef_mod_{name} / pval_mod_{name}
    coef_interact_{name} / se_interact_{name} / tstat_interact_{name} / pvalue_interact_{name}
    r2_mod_{name} / n_obs_mod_{name}
    [same with _ctrl suffix for controlled spec]
  Clustered SE at ipo_id level.

Outputs (suffix encodes active modes):
  final/reg/reg_bivariate_grouped_every_after_start_{suffix}.csv
  final/reg/reg_bivariate_grouped_every_after_end_{suffix}.csv
"""

import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CLI (used by SLURM array; defaults match interactive use) ─────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Bivariate grouped regression — peer-level")
    p.add_argument("--root",   type=str, default=None,
                   help="Project root dir (default: 3 levels above this script)")
    p.add_argument("--rc1",    action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rc2",    action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--rc3",    action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ic",     action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pc",     action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--year-fe",action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ind-fe", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--mkt-mod",action=argparse.BooleanOptionalAction, default=False,
                   help="Add market moderation: ret_4w_sh000300 main effect + X*mkt interaction")
    p.add_argument("--winsor", action=argparse.BooleanOptionalAction, default=True,
                   help="Winsorize all continuous variables at [1%%, 99%%] (default: on)")
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
    return p.parse_args()

_args = _parse_args()
ROOT = Path(_args.root) if _args.root else Path(__file__).resolve().parent.parent.parent

# ── Configuration ─────────────────────────────────────────────────────────────
USE_RIVAL_CONTROL_1 = _args.rc1     # log_size, bm, roa, leverage  (baseline)
USE_RIVAL_CONTROL_2 = _args.rc2     # age_listed, age_estab
USE_RIVAL_CONTROL_3 = _args.rc3     # pe_ttm, is_soe  (many missings — use with caution)
USE_IPO_CONTROL     = _args.ic      # ipo_log_size, ipo_pe_diluted, ipo_shares_issued, ipo_price
USE_PAIR_CONTROL    = _args.pc      # sim_mda  (pair-level MDA similarity)
YEAR_FE             = _args.year_fe # year fixed effects
IND_FE              = _args.ind_fe  # industry fixed effects
IND_COL             = "csrc3"       # column for IND_FE (CSRC 3-digit industry code)
MKT_MOD             = _args.mkt_mod # market moderation via ret_4w_sh000300
MKT_COL             = "ret_4w_sh000300"  # market proxy for moderation
WINSORIZE           = _args.winsor       # winsorize all continuous vars at [1 %, 99 %]
WINSOR_BOUNDS       = (0.01, 0.99)

RIVAL_CONTROL_1 = ["log_size", "bm", "roa", "leverage"]
RIVAL_CONTROL_2 = ["age_listed", "age_estab"]
RIVAL_CONTROL_3 = ["pe_ttm", "is_soe"]
IPO_CONTROL     = ["ipo_log_size", "ipo_pe_diluted", "ipo_shares_issued", "ipo_price"]
# NOTE: ipo_shares_issued is not in the CSV (closest: ipo_subs_ratio); skipped silently if absent.
PAIR_CONTROL    = ["sim_mda"]

# SE is always clustered at ipo_id level for peer-level data (no flag needed)

# ── Moderation configuration ──────────────────────────────────────────────────
# Verbal moderators (session-specific, sourced from verbal_sentiment.csv)
# Prefixed as vmod_{col} after merging into the regression dataframe
VERBAL_MOD_MAP = {
    "comp_verbal": "competition_ratio",
    "pros_verbal": "prospect_ratio",
}
# QA moderators (IPO-level, sourced from qa_analysis.csv)
# Prefixed as qmod_{col} after merging into the regression dataframe
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

_suffix_parts = []
if USE_RIVAL_CONTROL_1: _suffix_parts.append("rc1")
if USE_RIVAL_CONTROL_2: _suffix_parts.append("rc2")
if USE_RIVAL_CONTROL_3: _suffix_parts.append("rc3")
if USE_IPO_CONTROL:     _suffix_parts.append("ic")
if USE_PAIR_CONTROL:    _suffix_parts.append("pc")
if YEAR_FE:             _suffix_parts.append("yfe")
if IND_FE:              _suffix_parts.append("ife")
if MKT_MOD:             _suffix_parts.append("mkt")
if WINSORIZE:           _suffix_parts.append("w99")
for k in active_verbal_mods: _suffix_parts.append(k)
for k in active_qa_mods:     _suffix_parts.append(k)
OUTPUT_SUFFIX = ("_" + "_".join(_suffix_parts)) if _suffix_parts else "_base"

SESSION_推介 = "推介"
SESSION_答谢 = "答谢"

# ── 1. Load CARV peer-level data ──────────────────────────────────────────────
_raw = pd.read_csv(ROOT / "carv/output/car_cav_windows_controls.csv")

# Assemble active control columns
_want_ctrl = []
if USE_RIVAL_CONTROL_1: _want_ctrl += RIVAL_CONTROL_1
if USE_RIVAL_CONTROL_2: _want_ctrl += RIVAL_CONTROL_2
if USE_RIVAL_CONTROL_3: _want_ctrl += RIVAL_CONTROL_3
if USE_IPO_CONTROL:     _want_ctrl += IPO_CONTROL
if USE_PAIR_CONTROL:    _want_ctrl += PAIR_CONTROL
ctrl_present = [c for c in _want_ctrl if c in _raw.columns]
_missing = [c for c in _want_ctrl if c not in _raw.columns]
if _missing:
    print(f"  WARNING: control columns not in CSV, skipped: {_missing}")

# Assemble FE columns
fe_cols = []
if YEAR_FE:
    if "event_year" in _raw.columns:
        fe_cols.append("event_year")
    else:
        print("  WARNING: 'event_year' not found — YEAR_FE disabled")
if IND_FE:
    if IND_COL in _raw.columns:
        fe_cols.append(IND_COL)
    else:
        print(f"  WARNING: IND_COL='{IND_COL}' not found — IND_FE disabled")

car_cols = [c for c in _raw.columns if c.startswith(("car_", "cav_"))]
mkt_present = [MKT_COL] if MKT_MOD and MKT_COL in _raw.columns else []
if MKT_MOD and not mkt_present:
    print(f"  WARNING: MKT_COL='{MKT_COL}' not found in CSV — MKT_MOD disabled")
    MKT_MOD = False
car = _raw[["ipo_id"] + car_cols + ctrl_present + fe_cols + mkt_present].copy()
print(f"Loaded controls CSV: {len(car)} peer-rows, "
      f"controls={ctrl_present}, fe_cols={fe_cols}, mkt_mod={MKT_MOD}")

y_col_groups = {
    "after_start": [c for c in car.columns if c.startswith("car_after_start_")],
    "after_end":   [c for c in car.columns if c.startswith("car_after_end_")],
}
print(f"Y groups: { {k: len(v) for k, v in y_col_groups.items()} }")

# ── 1b. Load QA moderator data (IPO-level) ────────────────────────────────────
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
_ann_dir = ROOT / "anns" if _os.name == "nt" else ROOT / "../"
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

# ── 3. Load X features (3 sources × 3 session aggregations) ──────────────────
META = {"file_stem", "method", "error"}
sources = {
    "verbal": "analyze/output/verbal_sentiment.csv",
    "vocal":  "analyze/output/vocal_features.csv",
    "visual": "analyze/output/visual_gaze.csv",
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

# ── 3b. Build verbal moderator aggregations (per session, from verbal source) ─
verbal_mod_aggs = {}
if active_verbal_mods:
    for sess_label in [SESSION_推介, SESSION_答谢, "mean"]:
        v_agg, _ = session_variants[sess_label]["verbal"]
        cols_need  = [c for c in active_verbal_mods.values() if c in v_agg.columns]
        cols_miss  = [c for c in active_verbal_mods.values() if c not in v_agg.columns]
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
print(f"\nPeer rows after group filter: {len(car_grp)} "
      f"(am={(car_grp['group']=='am').sum()}, pm={(car_grp['group']=='pm').sum()})")

# ── 5. Regression engine ──────────────────────────────────────────────────────
def run_ols_clustered(y_c, X_mat, groups):
    return sm.OLS(y_c, X_mat).fit(cov_type="cluster", cov_kwds={"groups": groups})

def finite_mask(arr):
    """Boolean mask: not-null and finite for numeric arrays."""
    a = np.asarray(arr, dtype=float)
    return np.isfinite(a)

def winsorize(arr):
    a = np.asarray(arr, dtype=float)
    lo = np.nanpercentile(a, WINSOR_BOUNDS[0] * 100)
    hi = np.nanpercentile(a, WINSOR_BOUNDS[1] * 100)
    return np.clip(a, lo, hi)

def maybe_winsorize(arr):
    return winsorize(arr) if WINSORIZE else np.asarray(arr, dtype=float)

def run_regressions(car_grp, y_cols, session_variants, ctrl_cols, fe_cols,
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

                # Build combined FE array once per (session, source, group)
                fe_parts = []
                for fc in fe_cols:
                    if fc in sub.columns:
                        dum = pd.get_dummies(
                            sub[fc], prefix=fc, drop_first=True
                        ).astype(np.float32)
                        fe_parts.append(dum.values)
                fe_arr = np.column_stack(fe_parts).astype(np.float32) if fe_parts else None

                # Market proxy array (used when mkt_mod=True)
                mkt_arr_full = None
                if mkt_mod and mkt_col and mkt_col in sub.columns:
                    mkt_arr_full = maybe_winsorize(sub[mkt_col].to_numpy(dtype=float, na_value=np.nan))

                # Build moderator column list for this (session, src, grp)
                # Each entry: (mod_name, col_in_sub)
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
                        g_b = sub.loc[base_ok, "ipo_id"].values

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
                            ctrl_arr = sub[ctrl_cols].to_numpy(dtype=np.float32)
                            for j in range(ctrl_arr.shape[1]):
                                ctrl_arr[:, j] = maybe_winsorize(ctrl_arr[:, j])

                        # ── Bivariate (± FE ± mkt moderation) ────────────
                        if mkt_mod and mkt_arr_full is not None:
                            mkt_b      = mkt_arr_full[base_ok]
                            interact_b = x_b * mkt_b
                            core_bi    = np.column_stack([x_b, mkt_b, interact_b])
                        else:
                            core_bi = x_b.reshape(-1, 1)

                        if fe_arr is not None:
                            X_bi = sm.add_constant(np.column_stack([core_bi, fe_arr[base_ok]]))
                        else:
                            X_bi = sm.add_constant(core_bi)
                        try:
                            res_bi = run_ols_clustered(y_b, X_bi, g_b)
                            rec.update({
                                "n_obs":  int(base_ok.sum()),
                                "n_ipo":  int(pd.Series(g_b).nunique()),
                                "const":  res_bi.params[0],
                                "coef":   res_bi.params[1],
                                "se":     res_bi.bse[1],
                                "tstat":  res_bi.tvalues[1],
                                "pvalue": res_bi.pvalues[1],
                                "r2":     res_bi.rsquared,
                            })
                            if mkt_mod and mkt_arr_full is not None:
                                rec.update({
                                    "coef_mkt":         res_bi.params[2],
                                    "pval_mkt":         res_bi.pvalues[2],
                                    "coef_interact":    res_bi.params[3],
                                    "se_interact":      res_bi.bse[3],
                                    "tstat_interact":   res_bi.tvalues[3],
                                    "pvalue_interact":  res_bi.pvalues[3],
                                })
                        except Exception as e:
                            rec["error_bi"] = str(e)

                        # ── With controls (± FE ± mkt moderation) ─────────
                        if ctrl_arr is not None:
                            ctrl_ok = base_ok.copy()
                            for j in range(ctrl_arr.shape[1]):
                                ctrl_ok &= finite_mask(ctrl_arr[:, j])

                            y_c2    = y_arr[ctrl_ok]
                            x_c2    = x_arr[ctrl_ok]
                            ctrlv   = ctrl_arr[ctrl_ok]
                            g_c2    = sub.loc[ctrl_ok, "ipo_id"].values

                            rec["n_obs_ctrl"] = int(ctrl_ok.sum())
                            rec["n_ipo_ctrl"] = int(pd.Series(g_c2).nunique())

                            if ctrl_ok.sum() >= 15:
                                if mkt_mod and mkt_arr_full is not None:
                                    mkt_c2      = mkt_arr_full[ctrl_ok]
                                    interact_c2 = x_c2 * mkt_c2
                                    core_ct     = np.column_stack([x_c2, mkt_c2, interact_c2])
                                else:
                                    core_ct = x_c2.reshape(-1, 1)

                                if fe_arr is not None:
                                    X_ctrl = sm.add_constant(
                                        np.column_stack([core_ct, ctrlv, fe_arr[ctrl_ok]])
                                    )
                                else:
                                    X_ctrl = sm.add_constant(
                                        np.column_stack([core_ct, ctrlv])
                                    )
                                try:
                                    res_ct = run_ols_clustered(y_c2, X_ctrl, g_c2)
                                    rec.update({
                                        "coef_ctrl":   res_ct.params[1],
                                        "se_ctrl":     res_ct.bse[1],
                                        "tstat_ctrl":  res_ct.tvalues[1],
                                        "pvalue_ctrl": res_ct.pvalues[1],
                                        "r2_ctrl":     res_ct.rsquared,
                                    })
                                    if mkt_mod and mkt_arr_full is not None:
                                        # mkt at idx 2, interaction at idx 3, controls at 4+
                                        ctrl_offset = 4
                                        rec.update({
                                            "coef_mkt_ctrl":        res_ct.params[2],
                                            "pval_mkt_ctrl":        res_ct.pvalues[2],
                                            "coef_interact_ctrl":   res_ct.params[3],
                                            "se_interact_ctrl":     res_ct.bse[3],
                                            "tstat_interact_ctrl":  res_ct.tvalues[3],
                                            "pvalue_interact_ctrl": res_ct.pvalues[3],
                                        })
                                    else:
                                        ctrl_offset = 2
                                    for i, cc in enumerate(ctrl_cols):
                                        rec[f"coef_{cc}"] = res_ct.params[ctrl_offset + i]
                                        rec[f"pval_{cc}"] = res_ct.pvalues[ctrl_offset + i]
                                except Exception as e:
                                    rec["error_ctrl"] = str(e)
                            else:
                                rec["error_ctrl"] = "too few obs after ctrl dropna"

                        # ── Additional moderations (competition / prospect) ─
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
                            g_m = sub.loc[mod_ok, "ipo_id"].values

                            core_m = np.column_stack([x_m, mval, interact_m])
                            X_m = sm.add_constant(
                                np.column_stack([core_m, fe_arr[mod_ok]])
                                if fe_arr is not None else core_m
                            )
                            try:
                                res_m = run_ols_clustered(y_m, X_m, g_m)
                                rec.update({
                                    f"n_ipo_mod_{mod_name}":          int(pd.Series(g_m).nunique()),
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
                                    g_mc    = sub.loc[ctrl_ok_m, "ipo_id"].values

                                    core_mc = np.column_stack([x_mc, mval_c, interact_mc])
                                    X_mc = sm.add_constant(
                                        np.column_stack([core_mc, ctrlv_m, fe_arr[ctrl_ok_m]])
                                        if fe_arr is not None else
                                        np.column_stack([core_mc, ctrlv_m])
                                    )
                                    try:
                                        res_mc = run_ols_clustered(y_mc, X_mc, g_mc)
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
                                        for i, cc in enumerate(ctrl_cols):
                                            rec[f"coef_{cc}_mod_{mod_name}"] = res_mc.params[4 + i]
                                            rec[f"pval_{cc}_mod_{mod_name}"] = res_mc.pvalues[4 + i]
                                    except Exception as e:
                                        rec[f"error_mod_{mod_name}_ctrl"] = str(e)
                                else:
                                    rec[f"error_mod_{mod_name}_ctrl"] = "too few obs after ctrl dropna"

                        records.append(rec)
        print(f"  session='{sess_label}' done")
    return pd.DataFrame(records)

def summarise(out, label, ctrl_cols, fe_cols):
    fe_tag = ""
    if "event_year" in fe_cols: fe_tag += "+YFE"
    if IND_COL in fe_cols:      fe_tag += "+IFE"
    res_ok = out[out["pvalue"].notna()].copy()
    print(f"\n[{label}] Completed: {len(res_ok)}")
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

for y_group, y_cols in y_col_groups.items():
    print(f"\n=== Running {y_group} ({len(y_cols)} Y cols) ===")
    out = run_regressions(
        car_grp, y_cols, session_variants, ctrl_present, fe_cols,
        mkt_mod=MKT_MOD, mkt_col=MKT_COL,
        verbal_mods=active_verbal_mods if active_verbal_mods else None,
        qa_mods=active_qa_mods     if active_qa_mods     else None,
        verbal_mod_aggs=verbal_mod_aggs if active_verbal_mods else None,
    )
    out_path = ROOT / f"final/reg/reg_bivariate_grouped_every_{y_group}{OUTPUT_SUFFIX}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(out)} rows → {out_path}")
    summarise(out, y_group, ctrl_present, fe_cols)
