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
    --qa-pairs-mod      : moderation by qa_pairs (Q&A count) from qa_analysis.csv (IPO-level)

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
import concurrent.futures
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
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
    p.add_argument("--pltfe",  action=argparse.BooleanOptionalAction, default=False,
                   help="Add board and platform fixed effects (board_fe, platform_fe)")
    p.add_argument("--mkt-mod",action=argparse.BooleanOptionalAction, default=False,
                   help="Add market moderation: ret_4w_sh000300 main effect + X*mkt interaction")
    p.add_argument("--winsor", action=argparse.BooleanOptionalAction, default=True,
                   help="Winsorize all continuous variables at [1%%, 99%%] (default: on)")
    p.add_argument("--winsor-x", action=argparse.BooleanOptionalAction, default=True,
                   help="Winsorize X (main predictor) at [1%%, 99%%]; set --no-winsor-x to skip (e.g. for PCA scores)")
    p.add_argument("--top-rivals", type=int, default=None,
                   help="Keep only top-N rivals per IPO by sim_mda similarity; "
                        "drop IPO samples that have fewer than N rivals (e.g. 1, 3, 5)")
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
    p.add_argument("--qa-pairs-mod",    action=argparse.BooleanOptionalAction, default=False,
                   help="Moderation by qa_pairs (Q&A count) from qa_analysis.csv (IPO-level)")
    p.add_argument("--pca",             action=argparse.BooleanOptionalAction, default=False,
                   help="Use 推介 PCA scores (final/pca/pca_scores_推介.csv) as X instead of raw features")
    p.add_argument("--efa",             action=argparse.BooleanOptionalAction, default=False,
                   help="Use 推介 EFA scores (final/pca/efa_scores_combined_tui.csv) as X instead of raw features")
    p.add_argument("--max-pcs",    type=int, default=5,
                   help="PCA/EFA mode: max number of cumulative factors to regress (default: 5)")
    p.add_argument("--group",      type=str, default=None,
                   help="PCA mode only: PC column to split into quantile groups (e.g. pc1, pc2, pc3)")
    p.add_argument("--group-size", type=int, default=5,
                   help="Number of quantile groups when --group is active (default: 5)")
    p.add_argument("--pls",        action=argparse.BooleanOptionalAction, default=False,
                   help="Use PLS regression: project X onto latent components via PLSRegression, "
                        "then run OLS with clustered SE on the score(s). Works in both raw and PCA mode.")
    p.add_argument("--pls-ncomp",  type=int, default=1,
                   help="Number of PLS latent components to extract and use in cumulative regressions "
                        "(default: 1). Like PCA cumulative mode: Y~pls1, Y~pls1+pls2, etc.")
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
PLT_FE              = _args.pltfe   # board + platform fixed effects
PLT_FE_COLS         = ["board_fe", "platform_fe"]
MKT_MOD             = _args.mkt_mod # market moderation via ret_4w_sh000300
MKT_COL             = "ret_4w_sh000300"  # market proxy for moderation
WINSORIZE           = _args.winsor       # winsorize all continuous vars at [1 %, 99 %]
WINSORIZE_X         = _args.winsor_x and _args.winsor  # winsorize X (main predictor); requires --winsor
WINSOR_BOUNDS       = (0.01, 0.99)
TOP_RIVALS          = _args.top_rivals   # keep only top-N rivals per IPO by sim_mda (None = all)
PCA_MODE            = _args.pca          # use 推介 PCA scores as X features
EFA_MODE            = _args.efa          # use 推介 EFA scores as X features
MAX_PCS             = _args.max_pcs      # cap cumulative PC/factor loop in PCA/EFA mode
GROUP_COL           = _args.group if (_args.pca or _args.efa) else None
GROUP_SIZE          = _args.group_size   # number of quantile groups
PLS_MODE            = _args.pls          # use PLS latent score instead of raw X
PLS_NCOMP           = _args.pls_ncomp   # number of PLS components (cumulative regression)

# ── PLS feature whitelist (mirrors pca_combined_tui.py GROUPS) ────────────────
# These are the UN-prefixed column names in each source CSV that the PCA uses.
# When --pls is active, only these columns are fed to PLSRegression so that
# the latent score is conceptually comparable to the PCA scores.
PLS_FEATURE_COLS = {
    "verbal": [
        "ann_positive_ratio", "ann_negative_ratio",
        "social_positive_ratio", "social_negative_ratio",
    ],
    "vocal": [
        # f0_cv = f0_std / f0_mean (coefficient of variation) is derived in pca_combined_tui.py,
        # but the raw CSV only has f0_std and f0_mean separately. We pass them as-is;
        # PLSRegression(scale=True) finds the optimal linear combination, which is equivalent.
        "f0_std", "f0_mean",
        "f0_slope", "f0_range", "rms_dynamic_range", "rms_cv",
        "articulation_rate", "pause_rate",
    ],
    "visual": [
        "gaze_at_camera_ratio_10", "gaze_x_mean", "gaze_x_std",
        "gaze_y_mean", "gaze_y_std", "head_frontal_ratio_10",
        "head_pitch_mean", "head_pitch_std", "head_yaw_mean", "head_yaw_std",
    ],
    "visual_fer": [
        "positive_ratio", "negative_ratio", "neutral_ratio",
        "emo_happy", "emo_neutral", "emo_sad",
    ],
}

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
    "qa_pairs":  "qa_pairs",
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
if PLT_FE:              _suffix_parts.append("pltfe")
if MKT_MOD:             _suffix_parts.append("mkt")
if WINSORIZE:           _suffix_parts.append("w99")
if TOP_RIVALS is not None: _suffix_parts.append(f"top{TOP_RIVALS}")
if PCA_MODE:            _suffix_parts.append("pca")
if EFA_MODE:            _suffix_parts.append("efa")
if PLS_MODE:            _suffix_parts.append(f"pls{PLS_NCOMP}")
if GROUP_COL is not None: _suffix_parts.append(f"grp_{GROUP_COL}_{GROUP_SIZE}")
for k in active_verbal_mods: _suffix_parts.append(k)
for k in active_qa_mods:     _suffix_parts.append(k)
OUTPUT_SUFFIX = ("_" + "_".join(_suffix_parts)) if _suffix_parts else "_base"

SESSION_推介 = "推介"
SESSION_答谢 = "答谢"

# ── 1. Load CARV peer-level data ──────────────────────────────────────────────
_raw = pd.read_csv(ROOT / "carv/output/car_cav_windows_controls.csv")

# ── 1a. Top-N rivals filter (by sim_mda similarity) ──────────────────────────
if TOP_RIVALS is not None:
    if "sim_mda" not in _raw.columns:
        raise ValueError("--top-rivals requires 'sim_mda' column in car_cav_windows_controls.csv")
    _raw = _raw.sort_values("sim_mda", ascending=False)
    _raw = _raw.groupby("ipo_id", group_keys=False).head(TOP_RIVALS)
    _rival_counts = _raw.groupby("ipo_id").size()
    _valid_ipos   = _rival_counts[_rival_counts >= TOP_RIVALS].index
    _raw = _raw[_raw["ipo_id"].isin(_valid_ipos)].reset_index(drop=True)
    print(f"Top-{TOP_RIVALS} rivals filter: {len(_raw)} peer-rows, "
          f"{_raw['ipo_id'].nunique()} IPOs retained")

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
if PLT_FE:
    for _c in PLT_FE_COLS:
        if _c in _raw.columns:
            fe_cols.append(_c)
        else:
            print(f"  WARNING: PLT_FE col '{_c}' not found — skipped")

car_cols = [c for c in _raw.columns if c.startswith(("car_", "cav_"))]
mkt_present = [MKT_COL] if MKT_MOD and MKT_COL in _raw.columns else []
if MKT_MOD and not mkt_present:
    print(f"  WARNING: MKT_COL='{MKT_COL}' not found in CSV — MKT_MOD disabled")
    MKT_MOD = False
car = _raw[["ipo_id"] + car_cols + ctrl_present + fe_cols + mkt_present].copy()
print(f"Loaded controls CSV: {len(car)} peer-rows, "
      f"controls={ctrl_present}, fe_cols={fe_cols}, mkt_mod={MKT_MOD}")

y_col_groups = {
    "before_start": [c for c in car.columns if c.startswith("car_before_start_")],
    "after_start":  [c for c in car.columns if c.startswith("car_after_start_")],
    "after_end":    [c for c in car.columns if c.startswith("car_after_end_")],
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
    "verbal":     "analyze/output/verbal_sentiment.csv",
    "vocal":      "analyze/output/vocal_features.csv",
    "visual":     "analyze/output/visual_gaze.csv",
    "visual_fer": "analyze/output/visual_fer.csv",
}
if PCA_MODE:
    sources = {"pca": "final/pca/pca_scores_combined_tui.csv"}
elif EFA_MODE:
    sources = {"efa": "final/pca/efa_scores_combined_tui.csv"}

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

_active_sessions = [SESSION_推介] if (PCA_MODE or EFA_MODE) else [SESSION_推介, SESSION_答谢]
session_variants = {s: {} for s in _active_sessions}
for name, rel in sources.items():
    agg_tui, xcols = load_agg(ROOT / rel, session_filter=SESSION_推介)
    session_variants[SESSION_推介][name] = (agg_tui, xcols)
    if not PCA_MODE and not EFA_MODE:  # no second session in PCA/EFA mode
        agg_da, _ = load_agg(ROOT / rel, session_filter=SESSION_答谢)
        session_variants[SESSION_答谢][name] = (agg_da, xcols)
        print(f"{name}: 推介={len(agg_tui)}, 答谢={len(agg_da)}, {len(xcols)} X cols")
    else:
        mode_tag = "PCA" if PCA_MODE else "EFA"
        print(f"{name}: 推介={len(agg_tui)}, {len(xcols)} X cols ({mode_tag} mode)")

# ── 3c. When --pls: restrict x_cols to PCA feature whitelist ─────────────────
if PLS_MODE:
    for sess_label in session_variants:
        for src_name in list(session_variants[sess_label].keys()):
            x_df, x_cols = session_variants[sess_label][src_name]
            whitelist = PLS_FEATURE_COLS.get(src_name)
            if whitelist is not None:
                x_cols_pls = [c for c in x_cols if c in whitelist]
                skipped    = [c for c in x_cols if c not in whitelist]
                if skipped:
                    print(f"  [PLS] {src_name}/{sess_label}: dropped {len(skipped)} non-PCA cols, "
                          f"keeping {len(x_cols_pls)}: {x_cols_pls}")
                session_variants[sess_label][src_name] = (x_df, x_cols_pls)
            # If source not in PLS_FEATURE_COLS (e.g. 'pca' which can't happen
            # since --pls and --pca are mutually exclusive), leave untouched.

# ── 3d. When --pls: build combined cross-dimension feature DF per session ─────────
# Inner-join all 4 sources by ipo_id; prefix col names with src_ to avoid clashes.
# pls_combined_data: {sess_label: (combined_df, list_of_prefixed_col_names)}
pls_combined_data = {}
if PLS_MODE:
    for sess_label in session_variants:
        comb_df    = None
        comb_cols  = []
        for src_name, (x_df, x_cols) in session_variants[sess_label].items():
            if not x_cols:
                continue
            rename    = {c: f"{src_name}_{c}" for c in x_cols}
            df_pref   = x_df[["ipo_id"] + x_cols].rename(columns=rename)
            comb_cols.extend(rename[c] for c in x_cols)
            comb_df = df_pref if comb_df is None else comb_df.merge(df_pref, on="ipo_id", how="inner")
        if comb_df is not None:
            pls_combined_data[sess_label] = (comb_df, comb_cols)
            print(f"  [PLS combined] {sess_label}: {len(comb_df)} IPOs, "
                  f"{len(comb_cols)} features across all dimensions")

# ── 3b. Build verbal moderator aggregations (per session, from verbal source) ─
verbal_mod_aggs = {}
if active_verbal_mods:
    _verbal_path = ROOT / "analyze/output/verbal_sentiment.csv"
    _verbal_sessions = [SESSION_推介, SESSION_答谢] if not (PCA_MODE or EFA_MODE) else [SESSION_推介]
    _verbal_cache = {s: load_agg(_verbal_path, session_filter=s)[0] for s in _verbal_sessions}
    for sess_label in _verbal_sessions:
        v_agg = _verbal_cache[sess_label]
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

def _pls_score(y_b, X_b):
    """Fit PLSRegression(n_components=1) on (X_b, y_b) and return the X latent score (N,).

    The score is the projection of X onto the first PLS component direction that
    maximises covariance with y.  We then run OLS with clustered SE on this
    single score so that all downstream inference is identical to the OLS path.
    """
    pls = PLSRegression(n_components=1, scale=True)
    pls.fit(X_b, y_b)
    x_score = pls.transform(X_b).ravel()      # (N, 1) → (N,)
    return x_score

def run_reg(y_c, X_mat, groups):
    """Run OLS with clustered SE — the single entry-point for all regression calls.

    When PLS_MODE is active the caller is expected to have already replaced X_mat
    with [const, pls_score, (controls/FE)] via _make_pls_Xmat(); this function
    itself is always plain OLS so that clustered SE is preserved.
    """
    return run_ols_clustered(y_c, X_mat, groups)

def _make_pls_Xmat(y_b, x_raw_b, extra_b=None):
    """Build X matrix for PLS+OLS path.

    1. Fit PLS on (x_raw_b, y_b) to obtain a single latent score.
    2. Return safe_add_constant([score, extra_b]) where extra_b is
       optional (controls / FE columns already stacked).

    Parameters
    ----------
    y_b      : 1-D array of outcome values (used to supervise PLS).
    x_raw_b  : 2-D array (N, K) of raw predictors passed to PLS.
    extra_b  : 2-D array (N, M) of additional columns (ctrl + FE), or None.

    Returns
    -------
    X_mat : design matrix with intercept prepended, keep mask.
    """
    score = _pls_score(y_b, x_raw_b.reshape(len(y_b), -1))
    inner = score.reshape(-1, 1) if extra_b is None else np.column_stack([score, extra_b])
    return safe_add_constant(inner)

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

def maybe_winsorize_x(arr):
    return winsorize(arr) if WINSORIZE_X else np.asarray(arr, dtype=float)

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
    collapsed to a constant in a subgroup), shifting all param indices by -1.
    This helper avoids that by dropping such columns first and prepending
    ones explicitly.

    Returns (X_mat, kept_mask) where kept_mask is a boolean array over the
    *input* columns (before the prepended intercept).
    """
    clean, keep = _drop_const_cols(mat)
    X = np.column_stack([np.ones((clean.shape[0], 1)), clean])
    return X, keep

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

            for grp in ("am", "pm", "all"):
                sub = (merged if grp == "all" else merged[merged["group"] == grp]).reset_index(drop=True)

                # Build combined FE array once per (session, source, group)
                fe_parts = []
                for fc in fe_cols:
                    if fc in sub.columns:
                        if fc == "platform_fe":
                            fe_parts.append(_platform_fe_dummies(sub[fc]).astype(np.float32))
                        else:
                            dum = pd.get_dummies(
                                sub[fc], prefix=fc, drop_first=True
                            ).astype(np.float32)
                            fe_parts.append(dum.values)
                if grp == "all" and "group" in sub.columns:
                    fe_parts.append((sub["group"] == "am").astype(float).values.reshape(-1, 1))
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
                        x_arr = maybe_winsorize_x(sub[x_col].to_numpy(dtype=float, na_value=np.nan))

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
                            ctrl_arr = sub[ctrl_cols].to_numpy(dtype=np.float32).copy()
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
                            X_bi, _ = safe_add_constant(np.column_stack([core_bi, fe_arr[base_ok]]))
                        else:
                            X_bi, _ = safe_add_constant(core_bi)
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

                                _n_core_ct = core_ct.shape[1]
                                if fe_arr is not None:
                                    _inner_ct = np.column_stack([core_ct, ctrlv, fe_arr[ctrl_ok]])
                                else:
                                    _inner_ct = np.column_stack([core_ct, ctrlv])
                                X_ctrl, _keep_ct = safe_add_constant(_inner_ct)
                                _ctrl_keep_ct = _keep_ct[_n_core_ct : _n_core_ct + len(ctrl_cols)]
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
                                    _kept_i_ct = 0
                                    for i, cc in enumerate(ctrl_cols):
                                        if _ctrl_keep_ct[i]:
                                            rec[f"coef_{cc}"] = res_ct.params[ctrl_offset + _kept_i_ct]
                                            rec[f"pval_{cc}"] = res_ct.pvalues[ctrl_offset + _kept_i_ct]
                                            _kept_i_ct += 1
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
                            X_m, _ = safe_add_constant(
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
                                    _n_core_mc = core_mc.shape[1]
                                    if fe_arr is not None:
                                        _inner_mc = np.column_stack([core_mc, ctrlv_m, fe_arr[ctrl_ok_m]])
                                    else:
                                        _inner_mc = np.column_stack([core_mc, ctrlv_m])
                                    X_mc, _keep_mc = safe_add_constant(_inner_mc)
                                    _ctrl_keep_mc = _keep_mc[_n_core_mc : _n_core_mc + len(ctrl_cols)]
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


def run_regressions_pls(car_grp, y_cols, session_variants, ctrl_cols, fe_cols,
                        qa_mods=None, verbal_mods=None, verbal_mod_aggs=None):
    """PLS regression variant of run_regressions.

    For each (session, source, group, y_col) ALL x_cols in the source are fed
    simultaneously into PLSRegression(n_components=1, scale=True) which finds
    the single latent direction maximising cov(X, y).  The resulting score is
    then used as the sole predictor in OLS with ipo_id-clustered SE.

    Produces ONE record per (session, src, group, y_col), not one per x_col.
    Output column layout mirrors run_regressions:
      coef / se / tstat / pvalue / r2           (bivariate + FE)
      coef_ctrl / ... / r2_ctrl                 (with controls + FE)
      Per active moderator: same coef_x_mod_* / coef_interact_* / ... fields.
    Extra field: n_x_cols  — number of raw X columns fed to PLS.
    """
    records = []
    for sess_label, src_dict in session_variants.items():
        vmod_df = None
        if verbal_mods and verbal_mod_aggs:
            vmod_df = verbal_mod_aggs.get(sess_label)

        for src, (x_df, x_cols) in src_dict.items():
            merged = car_grp.merge(x_df, on="ipo_id", how="inner").reset_index(drop=True)
            if vmod_df is not None:
                merged = merged.merge(vmod_df, on="ipo_id", how="left")

            for grp in ("am", "pm", "all"):
                sub = (merged if grp == "all" else merged[merged["group"] == grp]).reset_index(drop=True)

                # Build FE array
                fe_parts = []
                for fc in fe_cols:
                    if fc in sub.columns:
                        if fc == "platform_fe":
                            fe_parts.append(_platform_fe_dummies(sub[fc]).astype(np.float32))
                        else:
                            dum = pd.get_dummies(sub[fc], prefix=fc, drop_first=True).astype(np.float32)
                            fe_parts.append(dum.values)
                if grp == "all" and "group" in sub.columns:
                    fe_parts.append((sub["group"] == "am").astype(float).values.reshape(-1, 1))
                fe_arr = np.column_stack(fe_parts).astype(np.float32) if fe_parts else None

                # Build ctrl array
                ctrl_arr = None
                if ctrl_cols:
                    ctrl_arr = sub[ctrl_cols].to_numpy(dtype=np.float32).copy()
                    for j in range(ctrl_arr.shape[1]):
                        ctrl_arr[:, j] = maybe_winsorize(ctrl_arr[:, j])

                # Moderator column list
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

                    # Stack ALL X columns into a (N, K) matrix
                    X_raw = np.column_stack([
                        maybe_winsorize_x(sub[xc].to_numpy(dtype=float, na_value=np.nan))
                        for xc in x_cols
                    ])

                    # Base validity mask: y finite AND every X column finite
                    base_ok = finite_mask(y_arr)
                    for j in range(X_raw.shape[1]):
                        base_ok &= finite_mask(X_raw[:, j])
                    if base_ok.sum() < 15:
                        continue

                    y_b = y_arr[base_ok]
                    X_b = X_raw[base_ok]          # (n, K)
                    g_b = sub.loc[base_ok, "ipo_id"].values

                    rec = {
                        "session":   sess_label,
                        "group":     grp,
                        "y_col":     y_col,
                        "x_source":  src,
                        "x_col":     "pls_score",
                        "n_x_cols":  len(x_cols),
                    }

                    # ── Bivariate PLS ──────────────────────────────────────────
                    extra_bi = fe_arr[base_ok] if fe_arr is not None else None
                    try:
                        X_bi, _ = _make_pls_Xmat(y_b, X_b, extra_b=extra_bi)
                        res_bi  = run_ols_clustered(y_b, X_bi, g_b)
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
                    except Exception as e:
                        rec["error_bi"] = str(e)

                    # ── With controls PLS ──────────────────────────────────────
                    if ctrl_arr is not None:
                        ctrl_ok = base_ok.copy()
                        for j in range(ctrl_arr.shape[1]):
                            ctrl_ok &= finite_mask(ctrl_arr[:, j])
                        y_c2  = y_arr[ctrl_ok]
                        X_c2  = X_raw[ctrl_ok]
                        ctrlv = ctrl_arr[ctrl_ok]
                        g_c2  = sub.loc[ctrl_ok, "ipo_id"].values
                        rec["n_obs_ctrl"] = int(ctrl_ok.sum())
                        rec["n_ipo_ctrl"] = int(pd.Series(g_c2).nunique())
                        if ctrl_ok.sum() >= 15:
                            extra_ct = (
                                np.column_stack([ctrlv, fe_arr[ctrl_ok]])
                                if fe_arr is not None else ctrlv
                            )
                            try:
                                X_ctrl, _keep_ct = _make_pls_Xmat(y_c2, X_c2, extra_b=extra_ct)
                                # Layout after safe_add_constant:
                                #   params[0]=intercept  params[1]=pls_score
                                #   params[2..]=extra_ct kept cols (ctrl first, then FE)
                                # _keep_ct covers [pls_score | ctrl | FE] before intercept
                                # _keep_ct[0]=score (always True), [1:1+K_ctrl]=ctrl
                                _ctrl_keep_ct = _keep_ct[1: 1 + len(ctrl_cols)]
                                res_ct = run_ols_clustered(y_c2, X_ctrl, g_c2)
                                rec.update({
                                    "coef_ctrl":   res_ct.params[1],
                                    "se_ctrl":     res_ct.bse[1],
                                    "tstat_ctrl":  res_ct.tvalues[1],
                                    "pvalue_ctrl": res_ct.pvalues[1],
                                    "r2_ctrl":     res_ct.rsquared,
                                })
                                _kept_i = 0
                                for i, cc in enumerate(ctrl_cols):
                                    if _ctrl_keep_ct[i]:
                                        rec[f"coef_{cc}"] = res_ct.params[2 + _kept_i]
                                        rec[f"pval_{cc}"] = res_ct.pvalues[2 + _kept_i]
                                        _kept_i += 1
                            except Exception as e:
                                rec["error_ctrl"] = str(e)
                        else:
                            rec["error_ctrl"] = "too few obs after ctrl dropna"

                    # ── Moderations ────────────────────────────────────────────
                    # PLS score is re-fitted per moderation sub-sample so that
                    # the latent direction is always supervised by y on that sample.
                    for mod_name, mod_col in all_mods_cfg:
                        mod_arr_full = maybe_winsorize(
                            sub[mod_col].to_numpy(dtype=float, na_value=np.nan)
                        )
                        mod_ok = base_ok & finite_mask(mod_arr_full)
                        rec[f"n_obs_mod_{mod_name}"] = int(mod_ok.sum())
                        if mod_ok.sum() < 15:
                            continue

                        y_m  = y_arr[mod_ok]
                        X_m  = X_raw[mod_ok]
                        mval = mod_arr_full[mod_ok]
                        g_m  = sub.loc[mod_ok, "ipo_id"].values

                        # Fit PLS on this sub-sample → score → interact with mod
                        try:
                            score_m    = _pls_score(y_m, X_m)   # (n_mod,)
                            interact_m = score_m * mval
                            core_m     = np.column_stack([score_m, mval, interact_m])
                            extra_mod  = (
                                np.column_stack([core_m, fe_arr[mod_ok]])
                                if fe_arr is not None else core_m
                            )
                            X_bm, _ = safe_add_constant(extra_mod)
                            res_bm  = run_ols_clustered(y_m, X_bm, g_m)
                            # params: [0]=const [1]=score [2]=mod [3]=score*mod [4+]=FE
                            rec.update({
                                f"n_ipo_mod_{mod_name}":          int(pd.Series(g_m).nunique()),
                                f"coef_x_mod_{mod_name}":         res_bm.params[1],
                                f"coef_mod_{mod_name}":           res_bm.params[2],
                                f"pval_mod_{mod_name}":           res_bm.pvalues[2],
                                f"coef_interact_{mod_name}":      res_bm.params[3],
                                f"se_interact_{mod_name}":        res_bm.bse[3],
                                f"tstat_interact_{mod_name}":     res_bm.tvalues[3],
                                f"pvalue_interact_{mod_name}":    res_bm.pvalues[3],
                                f"r2_mod_{mod_name}":             res_bm.rsquared,
                            })
                        except Exception as e:
                            rec[f"error_mod_{mod_name}"] = str(e)

                        # Moderation + controls
                        if ctrl_arr is not None:
                            ctrl_ok_m = mod_ok.copy()
                            for j in range(ctrl_arr.shape[1]):
                                ctrl_ok_m &= finite_mask(ctrl_arr[:, j])
                            rec[f"n_obs_mod_{mod_name}_ctrl"] = int(ctrl_ok_m.sum())
                            if ctrl_ok_m.sum() >= 15:
                                y_mc    = y_arr[ctrl_ok_m]
                                X_mc    = X_raw[ctrl_ok_m]
                                mval_c  = mod_arr_full[ctrl_ok_m]
                                ctrlv_m = ctrl_arr[ctrl_ok_m]
                                g_mc    = sub.loc[ctrl_ok_m, "ipo_id"].values
                                try:
                                    score_mc    = _pls_score(y_mc, X_mc)
                                    interact_mc = score_mc * mval_c
                                    core_mc     = np.column_stack([score_mc, mval_c, interact_mc])
                                    # extra_mc layout: [score(1) | mod(1) | interact(1) | ctrl(K) | FE(*)]
                                    extra_mc = (
                                        np.column_stack([core_mc, ctrlv_m, fe_arr[ctrl_ok_m]])
                                        if fe_arr is not None
                                        else np.column_stack([core_mc, ctrlv_m])
                                    )
                                    X_mc_full, _keep_mc = safe_add_constant(extra_mc)
                                    # _keep_mc[0..2]=score/mod/interact, [3:3+K_ctrl]=ctrl
                                    _ctrl_keep_mc = _keep_mc[3: 3 + len(ctrl_cols)]
                                    res_mc = run_ols_clustered(y_mc, X_mc_full, g_mc)
                                    rec.update({
                                        f"n_ipo_mod_{mod_name}_ctrl":       int(pd.Series(g_mc).nunique()),
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
                                    for j, cc in enumerate(ctrl_cols):
                                        if _ctrl_keep_mc[j]:
                                            rec[f"coef_{cc}_mod_{mod_name}"] = res_mc.params[4 + _kept_i_mc]
                                            rec[f"pval_{cc}_mod_{mod_name}"] = res_mc.pvalues[4 + _kept_i_mc]
                                            _kept_i_mc += 1
                                except Exception as e:
                                    rec[f"error_mod_{mod_name}_ctrl"] = str(e)
                            else:
                                rec[f"error_mod_{mod_name}_ctrl"] = "too few obs after ctrl dropna"

                    records.append(rec)
            print(f"  session='{sess_label}' done (PLS)")
    return pd.DataFrame(records)


def run_regressions_pls_combined(car_grp, y_cols, pls_combined_data, ctrl_cols, fe_cols,
                                  pls_ncomp=1, qa_mods=None, verbal_mods=None,
                                  verbal_mod_aggs=None):
    """Cross-dimension PLS regression (replaces per-source run_regressions_pls).

    For each (session, group, y_col):
      1. Combine ALL PLS_FEATURE_COLS from all sources into one (N, K) X matrix.
      2. Fit PLSRegression(n_components=pls_ncomp, scale=True) supervised by Y.
      3. Extract scores matrix S (N, pls_ncomp).
      4. Run CUMULATIVE OLS with clustered SE:
           n=1: Y ~ pls1          [+ ctrl + FE]
           n=2: Y ~ pls1 + pls2   [+ ctrl + FE]
           ...  up to pls_ncomp

    Output: one record per (session, group, y_col, n_pls).  Same column layout
    as run_regressions_pca (coef_plsN / se_plsN / tstat_plsN / pvalue_plsN).
    Moderators use PLS score 1 only (pls1 * MOD interaction) for interpretability.
    """
    records = []
    for sess_label, (x_df_comb, all_pls_cols) in pls_combined_data.items():
        merged = car_grp.merge(x_df_comb, on="ipo_id", how="inner").reset_index(drop=True)

        # Attach verbal mods
        if verbal_mods and verbal_mod_aggs:
            vmod_df = verbal_mod_aggs.get(sess_label)
            if vmod_df is not None:
                merged = merged.merge(vmod_df, on="ipo_id", how="left")

        for grp in ("am", "pm", "all"):
            sub = (merged if grp == "all" else merged[merged["group"] == grp]).reset_index(drop=True)

            # Build FE array
            fe_parts = []
            for fc in fe_cols:
                if fc in sub.columns:
                    if fc == "platform_fe":
                        fe_parts.append(_platform_fe_dummies(sub[fc]).astype(np.float32))
                    else:
                        dum = pd.get_dummies(sub[fc], prefix=fc, drop_first=True).astype(np.float32)
                        fe_parts.append(dum.values)
            if grp == "all" and "group" in sub.columns:
                fe_parts.append((sub["group"] == "am").astype(float).values.reshape(-1, 1))
            fe_arr = np.column_stack(fe_parts).astype(np.float32) if fe_parts else None

            # Build ctrl array
            ctrl_arr = None
            if ctrl_cols:
                ctrl_arr = sub[ctrl_cols].to_numpy(dtype=np.float32).copy()
                for j in range(ctrl_arr.shape[1]):
                    ctrl_arr[:, j] = maybe_winsorize(ctrl_arr[:, j])

            # Moderator list
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

                # Combined X matrix (N, K) across ALL dimensions
                X_raw = np.column_stack([
                    maybe_winsorize_x(sub[c].to_numpy(dtype=float, na_value=np.nan))
                    for c in all_pls_cols
                ])

                # Base validity mask
                base_ok = finite_mask(y_arr)
                for j in range(X_raw.shape[1]):
                    base_ok &= finite_mask(X_raw[:, j])
                if base_ok.sum() < 15:
                    continue

                y_b = y_arr[base_ok]
                X_b = X_raw[base_ok]              # (n_base, K)
                g_b = sub.loc[base_ok, "ipo_id"].values

                # Fit PLS ONCE on base subsample; get all pls_ncomp scores
                actual_ncomp = min(pls_ncomp, X_b.shape[1], X_b.shape[0] - 1)
                try:
                    pls_model = PLSRegression(n_components=actual_ncomp, scale=True)
                    pls_model.fit(X_b, y_b)
                    scores_b = pls_model.transform(X_b)   # (n_base, actual_ncomp)
                    scores_b = np.atleast_2d(scores_b) if scores_b.ndim == 1 else scores_b
                except Exception as e:
                    continue  # PLS fitting failed for this (group, y_col)

                pls_names = [f"pls{i+1}" for i in range(actual_ncomp)]

                # Pre-compute index helper: base_ok positions in sub
                base_ok_pos = np.where(base_ok)[0]

                # Cumulative regressions: n = 1, 2, ..., actual_ncomp
                for n in range(1, actual_ncomp + 1):
                    x_subset = pls_names[:n]
                    S_b = scores_b[:, :n]          # (n_base, n)

                    rec = {
                        "session":      sess_label,
                        "group":        grp,
                        "y_col":        y_col,
                        "n_pls":        n,
                        "n_x_cols":     len(all_pls_cols),
                        "x_cols_pls":   ",".join(x_subset),
                    }

                    # ── Bivariate (± FE) ──────────────────────────────────────────
                    if fe_arr is not None:
                        X_bi, _ = safe_add_constant(np.column_stack([S_b, fe_arr[base_ok]]))
                    else:
                        X_bi, _ = safe_add_constant(S_b)
                    try:
                        res_bi = run_ols_clustered(y_b, X_bi, g_b)
                        rec["n_obs"] = int(base_ok.sum())
                        rec["n_ipo"] = int(pd.Series(g_b).nunique())
                        rec["r2"]    = res_bi.rsquared
                        for i, pn in enumerate(x_subset):
                            rec[f"coef_{pn}"]   = res_bi.params[1 + i]
                            rec[f"se_{pn}"]     = res_bi.bse[1 + i]
                            rec[f"tstat_{pn}"]  = res_bi.tvalues[1 + i]
                            rec[f"pvalue_{pn}"] = res_bi.pvalues[1 + i]
                    except Exception as e:
                        rec["error_bi"] = str(e)

                    # ── With controls ──────────────────────────────────────────
                    if ctrl_arr is not None:
                        ctrl_ok = base_ok.copy()
                        for j in range(ctrl_arr.shape[1]):
                            ctrl_ok &= finite_mask(ctrl_arr[:, j])
                        g_c = sub.loc[ctrl_ok, "ipo_id"].values
                        rec["n_obs_ctrl"] = int(ctrl_ok.sum())
                        rec["n_ipo_ctrl"] = int(pd.Series(g_c).nunique())
                        if ctrl_ok.sum() >= 15:
                            # Map ctrl_ok to scores_b row indices
                            ctrl_ok_pos = np.where(ctrl_ok)[0]
                            in_base = np.isin(base_ok_pos, ctrl_ok_pos)
                            y_c   = y_arr[ctrl_ok]
                            S_c   = scores_b[in_base, :n]
                            ctrlv = ctrl_arr[ctrl_ok]
                            if fe_arr is not None:
                                _inner = np.column_stack([S_c, ctrlv, fe_arr[ctrl_ok]])
                            else:
                                _inner = np.column_stack([S_c, ctrlv])
                            X_ctrl, _keep = safe_add_constant(_inner)
                            _ctrl_keep = _keep[n: n + len(ctrl_cols)]
                            try:
                                res_ct = run_ols_clustered(y_c, X_ctrl, g_c)
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

                    # ── Moderation (pls1 × MOD, bivariate + ctrl) ───────────────────
                    # Re-fit PLS(n=n) on mod subsample so the latent direction is
                    # always supervised by y on that exact sample.
                    for mod_name, mod_col in all_mods_cfg:
                        mod_arr_full = maybe_winsorize(
                            sub[mod_col].to_numpy(dtype=float, na_value=np.nan)
                        )
                        mod_ok = base_ok & finite_mask(mod_arr_full)
                        rec[f"n_obs_mod_{mod_name}"] = int(mod_ok.sum())
                        if mod_ok.sum() < 15:
                            continue

                        y_m  = y_arr[mod_ok]
                        X_m  = X_raw[mod_ok]
                        mval = mod_arr_full[mod_ok]
                        g_m  = sub.loc[mod_ok, "ipo_id"].values
                        try:
                            pls_m = PLSRegression(n_components=min(n, X_m.shape[1], X_m.shape[0]-1),
                                                  scale=True)
                            pls_m.fit(X_m, y_m)
                            S_m = pls_m.transform(X_m)          # (n_mod, n)
                            if S_m.ndim == 1:
                                S_m = S_m.reshape(-1, 1)
                            S_m = S_m[:, :n]
                            # Use pls1 score for the interaction term
                            score_m1    = S_m[:, 0]
                            interact_m  = score_m1 * mval
                            core_m = np.column_stack([S_m, mval, interact_m])
                            # param layout: const(0) pls1..plsN(1..n) mod(n+1) pls1*mod(n+2) FE
                            extra_mod = (
                                np.column_stack([core_m, fe_arr[mod_ok]])
                                if fe_arr is not None else core_m
                            )
                            X_bm, _ = safe_add_constant(extra_mod)
                            res_bm  = run_ols_clustered(y_m, X_bm, g_m)
                            rec.update({
                                f"n_ipo_mod_{mod_name}":          int(pd.Series(g_m).nunique()),
                                f"coef_mod_{mod_name}":           res_bm.params[1 + n],
                                f"pval_mod_{mod_name}":           res_bm.pvalues[1 + n],
                                f"coef_interact_{mod_name}":      res_bm.params[2 + n],
                                f"se_interact_{mod_name}":        res_bm.bse[2 + n],
                                f"tstat_interact_{mod_name}":     res_bm.tvalues[2 + n],
                                f"pvalue_interact_{mod_name}":    res_bm.pvalues[2 + n],
                                f"r2_mod_{mod_name}":             res_bm.rsquared,
                            })
                            for i, pn in enumerate(x_subset):
                                rec[f"coef_{pn}_mod_{mod_name}"]  = res_bm.params[1 + i]
                                rec[f"tstat_{pn}_mod_{mod_name}"] = res_bm.tvalues[1 + i]
                                rec[f"pvalue_{pn}_mod_{mod_name}"]= res_bm.pvalues[1 + i]
                        except Exception as e:
                            rec[f"error_mod_{mod_name}"] = str(e)

                        # Moderation + controls
                        if ctrl_arr is not None:
                            ctrl_ok_m = mod_ok.copy()
                            for j in range(ctrl_arr.shape[1]):
                                ctrl_ok_m &= finite_mask(ctrl_arr[:, j])
                            rec[f"n_obs_mod_{mod_name}_ctrl"] = int(ctrl_ok_m.sum())
                            if ctrl_ok_m.sum() >= 15:
                                y_mc    = y_arr[ctrl_ok_m]
                                X_mc    = X_raw[ctrl_ok_m]
                                mval_c  = mod_arr_full[ctrl_ok_m]
                                ctrlv_m = ctrl_arr[ctrl_ok_m]
                                g_mc    = sub.loc[ctrl_ok_m, "ipo_id"].values
                                try:
                                    _nc_m = min(n, X_mc.shape[1], X_mc.shape[0]-1)
                                    pls_mc = PLSRegression(n_components=_nc_m, scale=True)
                                    pls_mc.fit(X_mc, y_mc)
                                    S_mc = pls_mc.transform(X_mc)[:, :n]
                                    if S_mc.ndim == 1:
                                        S_mc = S_mc.reshape(-1, 1)
                                    score_mc1   = S_mc[:, 0]
                                    interact_mc = score_mc1 * mval_c
                                    core_mc = np.column_stack([S_mc, mval_c, interact_mc])
                                    _n_core_mc = core_mc.shape[1]  # n + 2
                                    extra_mcc = (
                                        np.column_stack([core_mc, ctrlv_m, fe_arr[ctrl_ok_m]])
                                        if fe_arr is not None
                                        else np.column_stack([core_mc, ctrlv_m])
                                    )
                                    X_mc_full, _keep_mc = safe_add_constant(extra_mcc)
                                    _ctrl_keep_mc = _keep_mc[_n_core_mc: _n_core_mc + len(ctrl_cols)]
                                    res_mc = run_ols_clustered(y_mc, X_mc_full, g_mc)
                                    rec.update({
                                        f"n_ipo_mod_{mod_name}_ctrl":       int(pd.Series(g_mc).nunique()),
                                        f"coef_mod_{mod_name}_ctrl":        res_mc.params[1 + n],
                                        f"pval_mod_{mod_name}_ctrl":        res_mc.pvalues[1 + n],
                                        f"coef_interact_{mod_name}_ctrl":   res_mc.params[2 + n],
                                        f"se_interact_{mod_name}_ctrl":     res_mc.bse[2 + n],
                                        f"tstat_interact_{mod_name}_ctrl":  res_mc.tvalues[2 + n],
                                        f"pvalue_interact_{mod_name}_ctrl": res_mc.pvalues[2 + n],
                                        f"r2_mod_{mod_name}_ctrl":          res_mc.rsquared,
                                    })
                                    for i, pn in enumerate(x_subset):
                                        rec[f"coef_{pn}_mod_{mod_name}_ctrl"]  = res_mc.params[1 + i]
                                        rec[f"se_{pn}_mod_{mod_name}_ctrl"]    = res_mc.bse[1 + i]
                                        rec[f"tstat_{pn}_mod_{mod_name}_ctrl"] = res_mc.tvalues[1 + i]
                                        rec[f"pvalue_{pn}_mod_{mod_name}_ctrl"]= res_mc.pvalues[1 + i]
                                    _kept_i_mc = 0
                                    for j, cc in enumerate(ctrl_cols):
                                        if _ctrl_keep_mc[j]:
                                            rec[f"coef_{cc}_mod_{mod_name}_ctrl"] = res_mc.params[1 + _n_core_mc + _kept_i_mc]
                                            rec[f"pval_{cc}_mod_{mod_name}_ctrl"] = res_mc.pvalues[1 + _n_core_mc + _kept_i_mc]
                                            _kept_i_mc += 1
                                except Exception as e:
                                    rec[f"error_mod_{mod_name}_ctrl"] = str(e)
                            else:
                                rec[f"error_mod_{mod_name}_ctrl"] = "too few obs after ctrl dropna"

                    records.append(rec)
            print(f"  group='{grp}' done (PLS combined, n_pls={actual_ncomp})")
    return pd.DataFrame(records)


def run_regressions_pca(car_grp, y_cols, x_df, pc_cols, ctrl_cols, fe_cols,
                         mkt_mod=False, mkt_col=None, all_mods_cfg=None,
                         group_col="group", group_values=("am", "pm", "all"),
                         max_pcs=None):
    """PCA cumulative regression: Y ~ pc1, Y ~ pc1+pc2, ... One record per (group, y_col, n_pcs).
    SE clustered at ipo_id level (peer-level data).

    Moderation model (per active mod, unified for mkt and verbal/qa mods):
      Y ~ const + pc1..pcN + MOD + pc1*MOD..pcN*MOD [+ controls] [+ FE]
    Param layout: [0]=const [1..N]=PCs [N+1]=MOD [N+2..2N+1]=PC*MOD [2N+2..]=controls

    Stores per moderator {mod_name}:
      n_obs_mod_{mod_name}[_ctrl], n_ipo_mod_{mod_name}[_ctrl], r2_mod_{mod_name}[_ctrl]
      coef_mod_{mod_name}[_ctrl], pval_mod_{mod_name}[_ctrl]          (MOD main effect)
      coef_{pc}_mod_{mod_name}[_ctrl], tstat_{pc}_mod_{mod_name}[_ctrl], pvalue_{pc}_mod_{mod_name}[_ctrl]
      coef_interact_{mod_name}_{pc}[_ctrl], tstat_interact_{mod_name}_{pc}[_ctrl], pvalue_interact_{mod_name}_{pc}[_ctrl]
    """
    if all_mods_cfg is None:
        all_mods_cfg = []

    records = []
    merged = car_grp.merge(x_df, on="ipo_id", how="inner").reset_index(drop=True)

    for grp in group_values:
        sub = (merged if grp == "all" else merged[merged[group_col] == grp]).reset_index(drop=True)

        fe_parts = []
        for fc in fe_cols:
            if fc in sub.columns:
                if fc == "platform_fe":
                    fe_parts.append(_platform_fe_dummies(sub[fc]).astype(np.float32))
                else:
                    dum = pd.get_dummies(sub[fc], prefix=fc, drop_first=True).astype(np.float32)
                    fe_parts.append(dum.values)
        if grp == "all" and group_col in sub.columns:
            fe_parts.append((sub[group_col] == "am").astype(np.float32).values.reshape(-1, 1))
        fe_arr = np.column_stack(fe_parts).astype(np.float32) if fe_parts else None

        ctrl_arr = None
        if ctrl_cols:
            ctrl_arr = sub[ctrl_cols].to_numpy(dtype=np.float32).copy()
            for j in range(ctrl_arr.shape[1]):
                ctrl_arr[:, j] = maybe_winsorize(ctrl_arr[:, j])

        # Pre-build moderator arrays for this group
        # mkt treated uniformly as mod_name="mkt" alongside verbal/qa mods
        _mod_arrays = {}  # mod_name -> full-length array
        if mkt_mod and mkt_col and mkt_col in sub.columns:
            _mod_arrays["mkt"] = maybe_winsorize(sub[mkt_col].to_numpy(dtype=float, na_value=np.nan))
        for mod_name, col in all_mods_cfg:
            if col in sub.columns:
                _mod_arrays[mod_name] = maybe_winsorize(sub[col].to_numpy(dtype=float, na_value=np.nan))

        for y_col in y_cols:
            y_arr = maybe_winsorize(sub[y_col].to_numpy(dtype=float, na_value=np.nan))
            n_max = min(len(pc_cols), max_pcs) if max_pcs else len(pc_cols)
            X_all = np.column_stack([
                maybe_winsorize_x(sub[pc].to_numpy(dtype=float, na_value=np.nan))
                for pc in pc_cols[:n_max]
            ])

            for n in range(1, n_max + 1):
                x_subset = pc_cols[:n]
                X_n = X_all[:, :n]

                base_ok = finite_mask(y_arr)
                for j in range(n):
                    base_ok &= finite_mask(X_n[:, j])
                if base_ok.sum() < 15:
                    continue

                y_b = y_arr[base_ok]
                X_b = X_n[base_ok] if n > 1 else X_n[base_ok].reshape(-1, 1)
                g_b = sub.loc[base_ok, "ipo_id"].values

                rec = {
                    "group":           grp,
                    "y_col":           y_col,
                    "n_pcs":           n,
                    "x_cols_included": ",".join(x_subset),
                }

                # ── Bivariate (no controls) ───────────────────────────────────
                if fe_arr is not None:
                    X_bi, _ = safe_add_constant(np.column_stack([X_b, fe_arr[base_ok]]))
                else:
                    X_bi, _ = safe_add_constant(X_b)
                try:
                    res_bi = run_ols_clustered(y_b, X_bi, g_b)
                    rec["n_obs"]  = int(base_ok.sum())
                    rec["n_ipo"]  = int(pd.Series(g_b).nunique())
                    rec["r2"]     = res_bi.rsquared
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
                    g_c = sub.loc[ctrl_ok, "ipo_id"].values
                    rec["n_ipo_ctrl"] = int(pd.Series(g_c).nunique())
                    if ctrl_ok.sum() >= 15:
                        y_c   = y_arr[ctrl_ok]
                        X_c   = X_n[ctrl_ok] if n > 1 else X_n[ctrl_ok].reshape(-1, 1)
                        ctrlv = ctrl_arr[ctrl_ok]
                        if fe_arr is not None:
                            _inner = np.column_stack([X_c, ctrlv, fe_arr[ctrl_ok]])
                        else:
                            _inner = np.column_stack([X_c, ctrlv])
                        X_ctrl, _keep = safe_add_constant(_inner)
                        _ctrl_keep = _keep[n: n + len(ctrl_cols)]
                        try:
                            res_ct = run_ols_clustered(y_c, X_ctrl, g_c)
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

                # ── Moderation specs (bivariate + controlled) ─────────────────
                # Model: Y ~ const + pc1..pcN + MOD + pc1*MOD..pcN*MOD [+ ctrl] [+ FE]
                # Param layout: [0]=const [1..N]=PCs [N+1]=MOD [N+2..2N+1]=interactions [2N+2..]=ctrl
                for mod_name, mod_arr_full in _mod_arrays.items():
                    mod_ok = base_ok & finite_mask(mod_arr_full)
                    rec[f"n_obs_mod_{mod_name}"] = int(mod_ok.sum())
                    if mod_ok.sum() < 15:
                        continue

                    y_m   = y_arr[mod_ok]
                    X_m   = X_n[mod_ok] if n > 1 else X_n[mod_ok].reshape(-1, 1)
                    mval  = mod_arr_full[mod_ok]
                    g_m   = sub.loc[mod_ok, "ipo_id"].values
                    # Build interaction block: N columns (one per PC)
                    int_m = X_m * mval[:, None]
                    core_m = np.column_stack([X_m, mval[:, None], int_m])

                    if fe_arr is not None:
                        X_bm, _ = safe_add_constant(np.column_stack([core_m, fe_arr[mod_ok]]))
                    else:
                        X_bm, _ = safe_add_constant(core_m)
                    try:
                        res_bm = run_ols_clustered(y_m, X_bm, g_m)
                        rec[f"n_ipo_mod_{mod_name}"] = int(pd.Series(g_m).nunique())
                        rec[f"r2_mod_{mod_name}"]    = res_bm.rsquared
                        # MOD main effect at param index N+1
                        rec[f"coef_mod_{mod_name}"]  = res_bm.params[1 + n]
                        rec[f"pval_mod_{mod_name}"]  = res_bm.pvalues[1 + n]
                        for i, pc in enumerate(x_subset):
                            rec[f"coef_{pc}_mod_{mod_name}"]   = res_bm.params[1 + i]
                            rec[f"tstat_{pc}_mod_{mod_name}"]  = res_bm.tvalues[1 + i]
                            rec[f"pvalue_{pc}_mod_{mod_name}"] = res_bm.pvalues[1 + i]
                            # Interaction at param index N+2+i
                            rec[f"coef_interact_{mod_name}_{pc}"]   = res_bm.params[1 + n + 1 + i]
                            rec[f"tstat_interact_{mod_name}_{pc}"]  = res_bm.tvalues[1 + n + 1 + i]
                            rec[f"pvalue_interact_{mod_name}_{pc}"] = res_bm.pvalues[1 + n + 1 + i]
                    except Exception as e:
                        rec[f"error_mod_{mod_name}_bi"] = str(e)

                    # Moderation + controls spec
                    if ctrl_arr is not None:
                        ctrl_ok_m = mod_ok.copy()
                        for j in range(ctrl_arr.shape[1]):
                            ctrl_ok_m &= finite_mask(ctrl_arr[:, j])
                        rec[f"n_obs_mod_{mod_name}_ctrl"] = int(ctrl_ok_m.sum())
                        if ctrl_ok_m.sum() >= 15:
                            y_mc    = y_arr[ctrl_ok_m]
                            X_mc    = X_n[ctrl_ok_m] if n > 1 else X_n[ctrl_ok_m].reshape(-1, 1)
                            mval_c  = mod_arr_full[ctrl_ok_m]
                            ctrlv_m = ctrl_arr[ctrl_ok_m]
                            g_mc    = sub.loc[ctrl_ok_m, "ipo_id"].values
                            int_mc  = X_mc * mval_c[:, None]
                            core_mc = np.column_stack([X_mc, mval_c[:, None], int_mc])
                            _n_core_mc = core_mc.shape[1]   # = 2N+1
                            if fe_arr is not None:
                                _inner_mc = np.column_stack([core_mc, ctrlv_m, fe_arr[ctrl_ok_m]])
                            else:
                                _inner_mc = np.column_stack([core_mc, ctrlv_m])
                            X_mc_full, _keep_mc = safe_add_constant(_inner_mc)
                            _ctrl_keep_mc = _keep_mc[_n_core_mc: _n_core_mc + len(ctrl_cols)]
                            try:
                                res_mc = run_ols_clustered(y_mc, X_mc_full, g_mc)
                                rec[f"n_ipo_mod_{mod_name}_ctrl"] = int(pd.Series(g_mc).nunique())
                                rec[f"r2_mod_{mod_name}_ctrl"]    = res_mc.rsquared
                                # MOD main effect at index N+1
                                rec[f"coef_mod_{mod_name}_ctrl"]  = res_mc.params[1 + n]
                                rec[f"pval_mod_{mod_name}_ctrl"]  = res_mc.pvalues[1 + n]
                                for i, pc in enumerate(x_subset):
                                    rec[f"coef_{pc}_mod_{mod_name}_ctrl"]   = res_mc.params[1 + i]
                                    rec[f"se_{pc}_mod_{mod_name}_ctrl"]     = res_mc.bse[1 + i]
                                    rec[f"tstat_{pc}_mod_{mod_name}_ctrl"]  = res_mc.tvalues[1 + i]
                                    rec[f"pvalue_{pc}_mod_{mod_name}_ctrl"] = res_mc.pvalues[1 + i]
                                    # Interaction at index N+2+i
                                    rec[f"coef_interact_{mod_name}_{pc}_ctrl"]   = res_mc.params[1 + n + 1 + i]
                                    rec[f"se_interact_{mod_name}_{pc}_ctrl"]     = res_mc.bse[1 + n + 1 + i]
                                    rec[f"tstat_interact_{mod_name}_{pc}_ctrl"]  = res_mc.tvalues[1 + n + 1 + i]
                                    rec[f"pvalue_interact_{mod_name}_{pc}_ctrl"] = res_mc.pvalues[1 + n + 1 + i]
                                _kept_i_mc = 0
                                for j, cc in enumerate(ctrl_cols):
                                    if _ctrl_keep_mc[j]:
                                        rec[f"coef_{cc}_mod_{mod_name}_ctrl"] = res_mc.params[1 + _n_core_mc + _kept_i_mc]
                                        rec[f"pval_{cc}_mod_{mod_name}_ctrl"] = res_mc.pvalues[1 + _n_core_mc + _kept_i_mc]
                                        _kept_i_mc += 1
                            except Exception as e:
                                rec[f"error_mod_{mod_name}_ctrl"] = str(e)
                        else:
                            rec[f"error_mod_{mod_name}_ctrl"] = "too few obs after ctrl dropna"

                records.append(rec)
        print(f"  group='{grp}' done")
    return pd.DataFrame(records)


def summarise(out, label, ctrl_cols, fe_cols):
    fe_tag = ""
    if "event_year" in fe_cols: fe_tag += "+YFE"
    if IND_COL in fe_cols:      fe_tag += "+IFE"
    res_ok = out[out["pvalue"].notna()].copy()
    print(f"\n[{label}] Completed: {len(res_ok)}")
    for sess in (SESSION_推介, SESSION_答谢):
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

if PCA_MODE or EFA_MODE:
    _src_key = "pca" if PCA_MODE else "efa"
    _x_df, _pc_cols = session_variants[SESSION_推介][_src_key]

    # Merge 推介 verbal mods into car_grp for PCA mode (session-specific → use 推介 only)
    if active_verbal_mods and verbal_mod_aggs:
        _vmod_tui = verbal_mod_aggs.get(SESSION_推介)
        if _vmod_tui is not None:
            car_grp = car_grp.merge(_vmod_tui, on="ipo_id", how="left")

    # Build unified mods config list: (mod_name, col_in_car_grp_after_merge)
    # mkt is handled separately via mkt_mod flag; verbal/qa mods via all_mods_cfg
    _pca_mods_cfg = []
    for mod_name, mod_col in (active_qa_mods or {}).items():
        col = f"qmod_{mod_col}"
        if col in car_grp.columns:
            _pca_mods_cfg.append((mod_name, col))
        else:
            print(f"  WARNING: QA mod col '{col}' not in car_grp — skipping {mod_name}")
    for mod_name, mod_col in (active_verbal_mods or {}).items():
        col = f"vmod_{mod_col}"
        if col in car_grp.columns:
            _pca_mods_cfg.append((mod_name, col))
        else:
            print(f"  WARNING: verbal mod col '{col}' not in car_grp — skipping {mod_name}")

    # Compute quantile groups on the specified PC column when --group is active
    _group_col    = "group"
    _group_values = ("am", "pm", "all")
    _car_grp_pca  = car_grp
    if GROUP_COL is not None:
        # Use _x_df (already 1 row per ipo_id) to avoid a many-to-many merge explosion
        _ipo_grp = _x_df[["ipo_id", GROUP_COL]].copy()
        _ipo_grp["pc_group"] = pd.qcut(
            _ipo_grp[GROUP_COL], GROUP_SIZE,
            labels=[f"q{i+1}" for i in range(GROUP_SIZE)],
            duplicates="drop",
        )
        _car_grp_pca = car_grp.merge(
            _ipo_grp[["ipo_id", "pc_group"]], on="ipo_id", how="left"
        )
        _group_col    = "pc_group"
        _group_values = [f"q{i+1}" for i in range(GROUP_SIZE)]
        print(f"Group mode: {GROUP_COL} → {GROUP_SIZE} quantile groups "
              f"({_ipo_grp['pc_group'].value_counts().sort_index().to_dict()})")

    def _run_pca_group(y_group, y_cols):
        _active_mods = [m[0] for m in _pca_mods_cfg] + (["mkt"] if MKT_MOD else [])
        print(f"\n=== Running PCA cumulative {y_group} "
              f"({len(y_cols)} Y cols, {len(_pc_cols)} PCs, "
              f"groups={list(_group_values)}, mods={_active_mods}) ===")
        out = run_regressions_pca(
            _car_grp_pca, y_cols, _x_df, _pc_cols, ctrl_present, fe_cols,
            mkt_mod=MKT_MOD, mkt_col=MKT_COL,
            all_mods_cfg=_pca_mods_cfg,
            group_col=_group_col, group_values=_group_values,
            max_pcs=MAX_PCS,
        )
        out_path = ROOT / f"final/reg/reg_bivariate_grouped_every_{y_group}{OUTPUT_SUFFIX}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(out)} rows → {out_path}")

    with concurrent.futures.ThreadPoolExecutor() as _pool:
        _futs = [_pool.submit(_run_pca_group, yg, yc)
                 for yg, yc in y_col_groups.items()]
        for _f in concurrent.futures.as_completed(_futs):
            _f.result()
else:
    if sum([bool(PLS_MODE), bool(PCA_MODE), bool(EFA_MODE)]) > 1:
        raise ValueError("--pls, --pca, and --efa are mutually exclusive.")
    def _run_group(y_group, y_cols):
        _method = f"PLS (combined, ncomp={PLS_NCOMP})" if PLS_MODE else "OLS"
        print(f"\n=== Running {_method} {y_group} ({len(y_cols)} Y cols) ===")
        if PLS_MODE:
            out = run_regressions_pls_combined(
                car_grp, y_cols, pls_combined_data, ctrl_present, fe_cols,
                pls_ncomp=PLS_NCOMP,
                qa_mods=active_qa_mods     if active_qa_mods     else None,
                verbal_mods=active_verbal_mods if active_verbal_mods else None,
                verbal_mod_aggs=verbal_mod_aggs if active_verbal_mods else None,
            )
        else:
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
        if not PLS_MODE:
            summarise(out, y_group, ctrl_present, fe_cols)

    with concurrent.futures.ThreadPoolExecutor() as _pool:
        _futs = [_pool.submit(_run_group, yg, yc)
                 for yg, yc in y_col_groups.items()]
        for _f in concurrent.futures.as_completed(_futs):
            _f.result()
