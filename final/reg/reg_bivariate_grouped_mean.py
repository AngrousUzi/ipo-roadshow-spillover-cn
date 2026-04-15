#!/usr/bin/env python3
"""
Bivariate grouped regression — 3 session variants, IPO-level mean Y:
  Y = car_after_start_* / car_after_end_* from car_cav_windows_mean.csv
  X = each numeric feature in analyze/output/{verbal_sentiment, vocal_features, visual_gaze}
  Groups: am (09:xx start) | pm (14:xx start)

  USE_CONTROLS : IPO-level controls (ipo_issue_price, ipo_pe_diluted, ipo_pb, ipo_proceeds_gross)
  USE_FE       : year fixed effects (from event_date)
  SE           : HC3 always — one obs per IPO, no clustering.

  Each run produces two regression columns per record:
    coef/se/tstat/pvalue/r2        — bivariate (+ FE if USE_FE)
    coef_ctrl/…/r2_ctrl            — with controls (+ FE if USE_FE)  [if USE_CONTROLS]
  Plus pval_{ctrl} / coef_{ctrl} per control variable.

Outputs:
  final/reg/reg_bivariate_grouped_mean_after_start.csv
  final/reg/reg_bivariate_grouped_mean_after_end.csv
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Configuration ─────────────────────────────────────────────────────────────
USE_CONTROLS = True
USE_FE       = True
CTRL_COLS    = ["ipo_issue_price", "ipo_pe_diluted", "ipo_pb", "ipo_proceeds_gross"]

SESSION_推介 = "推介"
SESSION_答谢 = "答谢"

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
    ctrl_src = pd.read_csv(
        ROOT / "carv/output/car_cav_windows_controls.csv",
        usecols=["ipo_id"] + CTRL_COLS,
    )
    ctrl_present = [c for c in CTRL_COLS if c in ctrl_src.columns]
    ipo_controls = ctrl_src.groupby("ipo_id")[ctrl_present].first().reset_index()
    print(f"IPO controls: {len(ipo_controls)} IPOs, cols: {ctrl_present}")
    print("NaN counts:", ipo_controls[ctrl_present].isna().sum().to_dict())
    car = car.merge(ipo_controls, on="ipo_id", how="left")

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

# ── 4. Merge CAR with group labels ───────────────────────────────────────────
car_grp = car.merge(idx_sub, on="ipo_id", how="inner")
print(f"\nIPO rows after group filter: {len(car_grp)} "
      f"(am={(car_grp['group']=='am').sum()}, pm={(car_grp['group']=='pm').sum()})")

# ── 5. Regression engine ──────────────────────────────────────────────────────
def run_ols_hc3(y_c, X_mat):
    return sm.OLS(y_c, X_mat).fit(cov_type="HC3")

def finite_mask(arr):
    return np.isfinite(np.asarray(arr, dtype=float))

def run_regressions(car_grp, y_cols, session_variants, ctrl_cols, use_fe):
    records = []
    for sess_label, src_dict in session_variants.items():
        for src, (x_df, x_cols) in src_dict.items():
            merged = car_grp.merge(x_df, on="ipo_id", how="inner").reset_index(drop=True)
            for grp in ("am", "pm"):
                sub = merged[merged["group"] == grp].reset_index(drop=True)

                if use_fe and "event_year" in sub.columns:
                    yr_dum = pd.get_dummies(
                        sub["event_year"], prefix="yr", drop_first=True
                    ).astype(float)
                    yr_arr = yr_dum.values
                else:
                    yr_arr = None

                for y_col in y_cols:
                    y_arr = sub[y_col].to_numpy(dtype=float, na_value=np.nan)
                    for x_col in x_cols:
                        x_arr = sub[x_col].to_numpy(dtype=float, na_value=np.nan)

                        base_ok = finite_mask(y_arr) & finite_mask(x_arr)
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

                        # ── Bivariate (± FE) ──────────────────────────────
                        if use_fe and yr_arr is not None:
                            X_bi = sm.add_constant(np.column_stack([x_b, yr_arr[base_ok]]))
                        else:
                            X_bi = sm.add_constant(x_b)
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
                        except Exception as e:
                            rec["error_bi"] = str(e)

                        # ── With controls (± FE) ──────────────────────────
                        if ctrl_cols:
                            ctrl_arr = sub[ctrl_cols].to_numpy(dtype=float)
                            ctrl_ok  = base_ok.copy()
                            for j in range(ctrl_arr.shape[1]):
                                ctrl_ok &= finite_mask(ctrl_arr[:, j])

                            y_c2  = y_arr[ctrl_ok]
                            x_c2  = x_arr[ctrl_ok]
                            ctrlv = ctrl_arr[ctrl_ok]

                            rec["n_obs_ctrl"] = int(ctrl_ok.sum())

                            if ctrl_ok.sum() >= 15:
                                if use_fe and yr_arr is not None:
                                    X_ctrl = sm.add_constant(
                                        np.column_stack([x_c2, ctrlv, yr_arr[ctrl_ok]])
                                    )
                                else:
                                    X_ctrl = sm.add_constant(np.column_stack([x_c2, ctrlv]))
                                try:
                                    res_ct = run_ols_hc3(y_c2, X_ctrl)
                                    rec.update({
                                        "coef_ctrl":   res_ct.params[1],
                                        "se_ctrl":     res_ct.bse[1],
                                        "tstat_ctrl":  res_ct.tvalues[1],
                                        "pvalue_ctrl": res_ct.pvalues[1],
                                        "r2_ctrl":     res_ct.rsquared,
                                    })
                                    for i, cc in enumerate(ctrl_cols):
                                        rec[f"coef_{cc}"] = res_ct.params[i + 2]
                                        rec[f"pval_{cc}"] = res_ct.pvalues[i + 2]
                                except Exception as e:
                                    rec["error_ctrl"] = str(e)
                            else:
                                rec["error_ctrl"] = "too few obs after ctrl dropna"

                        records.append(rec)
        print(f"  session='{sess_label}' done")
    return pd.DataFrame(records)

def summarise(out, label, ctrl_cols, use_fe):
    fe_tag = "+FE" if use_fe else ""
    res_ok = out[out["pvalue"].notna()].copy()
    print(f"\n[{label}] Completed: {len(res_ok)}  SE: HC3")
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
    out = run_regressions(car_grp, y_cols, session_variants, ctrl_present, USE_FE)
    out_path = ROOT / f"final/reg/reg_bivariate_grouped_mean_{y_group}_ctrl_fe.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(out)} rows → {out_path}")
    summarise(out, y_group, ctrl_present, USE_FE)
