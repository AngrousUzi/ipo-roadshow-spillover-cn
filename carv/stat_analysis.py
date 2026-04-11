"""
Statistical analysis of intraday CAR/CAV event windows.

Event windows (each at 30-min and 1-hr granularity):
  pre      : before_start  – 30/60 min before roadshow starts
  during   : after_start   – first 30/60 min of the roadshow
  post     : after_end     – 30/60 min after roadshow ends

Stage split: Stage 1 = 2009-2015, Stage 2 = 2016-2024 (regulation change)

Outputs:
  carv/output/stat_summary.xlsx  -- multi-sheet workbook
  carv/output/stat_report.txt    -- plain-text report
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ---------------------------------------------------------------------------
# Windows: (period_label, duration, car_col, cav_col)
# Using with925 (includes 9:25 auction) and est1 as primary;
# est2 and est3 kept for robustness columns in the Excel output.
# ---------------------------------------------------------------------------
WINDOWS = [
    ("pre",    "30min", "car_before_start_30min_with925_est1", "cav_before_start_30min_with925_est1"),
    ("pre",    "1hr",   "car_before_start_1hr_with925_est1",   "cav_before_start_1hr_with925_est1"),
    ("during", "30min", "car_after_start_30min_with925_est1",  "cav_after_start_30min_with925_est1"),
    ("during", "1hr",   "car_after_start_1hr_with925_est1",    "cav_after_start_1hr_with925_est1"),
    ("post",   "30min", "car_after_end_30min_with925_est1",    "cav_after_end_30min_with925_est1"),
    ("post",   "1hr",   "car_after_end_1hr_with925_est1",      "cav_after_end_1hr_with925_est1"),
]

# All CAR/CAV columns needed (est1/2/3, with/no 925, all six periods)
PERIODS = ["before_start_30min", "before_start_1hr",
           "after_start_30min",  "after_start_1hr",
           "after_end_30min",    "after_end_1hr"]
ESTIMATORS = ["est1", "est2", "est3"]
VARIANTS   = ["with925", "no925"]

ALL_COLS = (
    [f"car_{p}_{v}_{e}" for p in PERIODS for v in VARIANTS for e in ESTIMATORS] +
    [f"cav_{p}_{v}_{e}" for p in PERIODS for v in VARIANTS for e in ESTIMATORS]
)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading car_cav_windows.csv ...")
carv = pd.read_csv(os.path.join(OUT_DIR, "car_cav_windows.csv"), encoding="utf-8-sig")
carv["event_date"] = pd.to_datetime(carv["event_date"])
carv["year"] = carv["event_date"].dt.year

print("Loading IPO index ...")
idx = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "IPO_roadshow_index_2009_with_trading_days.csv"),
    encoding="utf-8-sig",
    encoding_errors="replace",
    usecols=["Stkcd", "Listdt", "INDEX2009"],
)
idx.columns = ["stkcd_ipo", "listdt", "ipo_id"]

print("Loading ind similarity (LLM) ...")
ind = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "ind", "ind_all_sim_pairs_within_llm.csv"),
)
ind["sector"] = ind["csrc3"].str[0]

# ---------------------------------------------------------------------------
# 2. Merge index → carv
# ---------------------------------------------------------------------------
carv = carv.merge(idx[["ipo_id", "stkcd_ipo"]], on="ipo_id", how="left")
carv["stkcd_rival"] = carv["rival_fc"].str[2:].astype(int)

# ---------------------------------------------------------------------------
# 3. Aggregate to IPO level (mean across rivals per ipo_id)
# ---------------------------------------------------------------------------
print("Aggregating to IPO level ...")

# Rival count per IPO × year → yearly average
rival_count = (
    carv.groupby(["ipo_id", "year"])["rival_fc"]
    .nunique()
    .reset_index()
    .rename(columns={"rival_fc": "n_rivals"})
)
df_rival_yr = (
    rival_count.groupby("year")["n_rivals"]
    .agg(["mean", "median", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "avg_rivals", "median": "med_rivals",
                     "std": "std_rivals", "count": "n_ipos"})
)

agg_cols = [c for c in ALL_COLS if c in carv.columns]
ipo = (
    carv.groupby(["ipo_id", "year", "stkcd_ipo"])[agg_cols]
    .mean()
    .reset_index()
)
ipo["stage"] = ipo["year"].apply(
    lambda y: "Stage1 (2009-2015)" if y <= 2015 else "Stage2 (2016-2024)"
)

# Merge industry (ipo stkcd × year → sector)
ind_ipo = (
    ind.groupby(["stkcd_i", "year"])[["csrc3", "sector"]]
    .first()
    .reset_index()
    .rename(columns={"stkcd_i": "stkcd_ipo"})
)
ipo = ipo.merge(ind_ipo, on=["stkcd_ipo", "year"], how="left")
print(f"IPO obs: {len(ipo)}  with industry: {ipo['csrc3'].notna().sum()}")

# ---------------------------------------------------------------------------
# 4. Statistics helper
# ---------------------------------------------------------------------------
def desc(s: pd.Series) -> dict:
    s = s.dropna()
    n = len(s)
    if n < 2:
        return {"n": n, "mean": np.nan, "median": np.nan, "std": np.nan,
                "t": np.nan, "p_t": np.nan, "p_w": np.nan, "sig": ""}
    t, p_t = stats.ttest_1samp(s, 0)
    try:
        _, p_w = stats.wilcoxon(s)
    except Exception:
        p_w = np.nan
    sig = "***" if p_t < 0.01 else ("**" if p_t < 0.05 else ("*" if p_t < 0.1 else ""))
    return {"n": n, "mean": s.mean(), "median": s.median(), "std": s.std(),
            "t": t, "p_t": p_t, "p_w": p_w, "sig": sig}

def diff_test(a: pd.Series, b: pd.Series) -> dict:
    a, b = a.dropna(), b.dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
    return {"mean_s1": a.mean(), "mean_s2": b.mean(),
            "t_diff": t, "p_diff": p, "sig_diff": sig}

# ---------------------------------------------------------------------------
# 5. Build summary tables
# ---------------------------------------------------------------------------

def window_table(df, groupby=None):
    """Return a tidy table: one row per (window, metric[, group])."""
    rows = []
    for period, dur, car_col, cav_col in WINDOWS:
        for metric, col in [("CAR", car_col), ("CAV", cav_col)]:
            if col not in df.columns:
                continue
            if groupby is None:
                d = desc(df[col])
                rows.append({"period": period, "duration": dur, "metric": metric, **d})
            else:
                for grp, sub in df.groupby(groupby):
                    d = desc(sub[col])
                    rows.append({"period": period, "duration": dur, "metric": metric,
                                 groupby: grp, **d})
    return pd.DataFrame(rows)

print("Building tables ...")

# Overall
df_overall = window_table(ipo)

# By stage
df_by_stage = window_table(ipo, groupby="stage")

# Stage-difference test (Welch)
diff_rows = []
s1 = ipo[ipo["stage"] == "Stage1 (2009-2015)"]
s2 = ipo[ipo["stage"] == "Stage2 (2016-2024)"]
for period, dur, car_col, cav_col in WINDOWS:
    for metric, col in [("CAR", car_col), ("CAV", cav_col)]:
        if col not in ipo.columns:
            continue
        d = diff_test(s1[col], s2[col])
        diff_rows.append({"period": period, "duration": dur, "metric": metric, **d})
df_stage_diff = pd.DataFrame(diff_rows)

# By industry (all years)
df_by_ind = window_table(ipo[ipo["sector"].notna()], groupby="sector")

# Industry × Stage
df_ind_stage = window_table(ipo[ipo["sector"].notna()], groupby="sector")
# Need two-level group; rebuild manually
ind_stage_rows = []
for period, dur, car_col, cav_col in WINDOWS:
    for metric, col in [("CAR", car_col), ("CAV", cav_col)]:
        if col not in ipo.columns:
            continue
        sub = ipo[ipo["sector"].notna()]
        for (sector, stage), grp in sub.groupby(["sector", "stage"]):
            d = desc(grp[col])
            ind_stage_rows.append({"period": period, "duration": dur, "metric": metric,
                                   "sector": sector, "stage": stage, **d})
df_ind_stage = pd.DataFrame(ind_stage_rows)

# Year trend
df_year = window_table(ipo, groupby="year")

# ---------------------------------------------------------------------------
# 6. Write Excel
# ---------------------------------------------------------------------------
out_xlsx = os.path.join(OUT_DIR, "stat_summary.xlsx")
print(f"Writing {out_xlsx} ...")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    df_overall.to_excel(writer, sheet_name="Overall", index=False)
    df_by_stage.to_excel(writer, sheet_name="By_Stage", index=False)
    df_stage_diff.to_excel(writer, sheet_name="Stage_Diff", index=False)
    df_by_ind.to_excel(writer, sheet_name="By_Industry", index=False)
    df_ind_stage.to_excel(writer, sheet_name="Industry_x_Stage", index=False)
    df_year.to_excel(writer, sheet_name="By_Year", index=False)
    df_rival_yr.to_excel(writer, sheet_name="Rival_Count_by_Year", index=False)

# ---------------------------------------------------------------------------
# 7. Text report
# ---------------------------------------------------------------------------
out_txt = os.path.join(OUT_DIR, "stat_report.txt")
lines = []

W  = 72
def hdr(t): lines.extend(["", "=" * W, t, "=" * W])
def sub(t): lines.extend(["", "  -- " + t])

def fmt(d, metric):
    if d["n"] < 2:
        return f"  n={d['n']}"
    return (
        f"  n={int(d['n']):<5}  mean={d['mean']:>10.5f}{d['sig']:<3}"
        f"  median={d['median']:>10.5f}  std={d['std']:>8.5f}"
        f"  t={d['t']:>7.3f}  p_t={d['p_t']:.4f}  p_w={d['p_w']:.4f}"
    )

# ---- Overall ----
hdr("OVERALL  (IPO-level mean across rivals, with925, est1)")
for period in ["pre", "during", "post"]:
    for dur in ["30min", "1hr"]:
        sub(f"{period} {dur}")
        for _, row in df_overall[
            (df_overall["period"] == period) & (df_overall["duration"] == dur)
        ].iterrows():
            lines.append(f"    {row['metric']}: " + fmt(row, row["metric"]))

# ---- Stage ----
hdr("BY STAGE")
for stage in ["Stage1 (2009-2015)", "Stage2 (2016-2024)"]:
    lines.append(f"\n  {stage}")
    sdf = df_by_stage[df_by_stage["stage"] == stage]
    for period in ["pre", "during", "post"]:
        for dur in ["30min", "1hr"]:
            sub(f"{period} {dur}")
            for _, row in sdf[
                (sdf["period"] == period) & (sdf["duration"] == dur)
            ].iterrows():
                lines.append(f"    {row['metric']}: " + fmt(row, row["metric"]))

# ---- Stage diff ----
hdr("STAGE DIFFERENCE TEST  (Welch t-test, Stage1 vs Stage2)")
for period in ["pre", "during", "post"]:
    for dur in ["30min", "1hr"]:
        sub(f"{period} {dur}")
        for _, row in df_stage_diff[
            (df_stage_diff["period"] == period) & (df_stage_diff["duration"] == dur)
        ].iterrows():
            lines.append(
                f"    {row['metric']}: S1={row['mean_s1']:>10.5f}  S2={row['mean_s2']:>10.5f}"
                f"  t={row['t_diff']:>7.3f}  p={row['p_diff']:.4f}  {row['sig_diff']}"
            )

# ---- Industry (primary window: pre/during/post 30min) ----
hdr("BY INDUSTRY  (all years, 30-min windows)")
for period in ["pre", "during", "post"]:
    for metric in ["CAR", "CAV"]:
        sub(f"{period} 30min | {metric}")
        sdf = df_by_ind[
            (df_by_ind["period"] == period) & (df_by_ind["duration"] == "30min") &
            (df_by_ind["metric"] == metric)
        ].sort_values("sector")
        for _, row in sdf.iterrows():
            if row["n"] < 3:
                continue
            lines.append(
                f"    {row['sector']}: n={int(row['n']):<5}  mean={row['mean']:>10.5f}"
                f"{row['sig']:<3}  p_t={row['p_t']:.4f}"
            )

# ---- Industry × Stage (primary: pre/during/post 30min CAR) ----
hdr("INDUSTRY × STAGE  (30-min windows, CAR)")
for stage in ["Stage1 (2009-2015)", "Stage2 (2016-2024)"]:
    lines.append(f"\n  {stage}")
    for period in ["pre", "during", "post"]:
        sub(f"{period} 30min")
        sdf = df_ind_stage[
            (df_ind_stage["stage"] == stage) & (df_ind_stage["period"] == period) &
            (df_ind_stage["duration"] == "30min") & (df_ind_stage["metric"] == "CAR")
        ].sort_values("sector")
        for _, row in sdf.iterrows():
            if row["n"] < 3:
                continue
            lines.append(
                f"    {row['sector']}: n={int(row['n']):<4}  mean={row['mean']:>10.5f}"
                f"{row['sig']:<3}  p_t={row['p_t']:.4f}"
            )

hdr("INDUSTRY × STAGE  (30-min windows, CAV)")
for stage in ["Stage1 (2009-2015)", "Stage2 (2016-2024)"]:
    lines.append(f"\n  {stage}")
    for period in ["pre", "during", "post"]:
        sub(f"{period} 30min")
        sdf = df_ind_stage[
            (df_ind_stage["stage"] == stage) & (df_ind_stage["period"] == period) &
            (df_ind_stage["duration"] == "30min") & (df_ind_stage["metric"] == "CAV")
        ].sort_values("sector")
        for _, row in sdf.iterrows():
            if row["n"] < 3:
                continue
            lines.append(
                f"    {row['sector']}: n={int(row['n']):<4}  mean={row['mean']:>10.5f}"
                f"{row['sig']:<3}  p_t={row['p_t']:.4f}"
            )

# ---- Year trend (30min primary windows) ----
hdr("YEAR TREND  (30-min windows, est1)")
for metric in ["CAR", "CAV"]:
    for period in ["pre", "during", "post"]:
        sub(f"{period} 30min | {metric}")
        sdf = df_year[
            (df_year["period"] == period) & (df_year["duration"] == "30min") &
            (df_year["metric"] == metric)
        ].sort_values("year")
        for _, row in sdf.iterrows():
            lines.append(
                f"    {int(row['year'])}: n={int(row['n']):<4}  mean={row['mean']:>10.5f}"
                f"{row['sig']:<3}  p_t={row['p_t']:.4f}"
            )

report = "\n".join(lines)
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"\nOutputs: {out_xlsx}\n        {out_txt}")

# ---------------------------------------------------------------------------
# Rival count figure (baseline: all CSRC-3 matches)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig_dir = os.path.join(OUT_DIR, "figures")
os.makedirs(fig_dir, exist_ok=True)

years_rc = df_rival_yr["year"].values
avg_rc   = df_rival_yr["avg_rivals"].values
std_rc   = df_rival_yr["std_rivals"].values
n_ipos   = df_rival_yr["n_ipos"].values

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(years_rc, avg_rc,
       color=["#4878CF" if y <= 2015 else "#D65F5F" for y in years_rc],
       alpha=0.75, zorder=3, width=0.7)
ax.errorbar(years_rc, avg_rc, yerr=std_rc / np.sqrt(n_ipos),
            fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
for x, a, se in zip(years_rc, avg_rc, std_rc / np.sqrt(n_ipos)):
    ax.text(x, a + se + avg_rc.max() * 0.02,
            f"{a:.1f}", ha="center", va="bottom", fontsize=7)
ax.axvline(2015.5, color="gray", linewidth=1.2, linestyle="--", label="Stage break (2016)")
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Avg rivals per IPO", fontsize=10)
ax.set_title(
    "Yearly Average Competitive Firm Count per IPO\n(Baseline: all CSRC-3 matches)",
    fontsize=11, fontweight="bold",
)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax2 = ax.twinx()
ax2.plot(years_rc, n_ipos, "k--o", markersize=4, linewidth=1.2, label="# IPOs (right)")
ax2.set_ylabel("# IPOs in sample", fontsize=9)
ax2.legend(fontsize=8, loc="upper right")
fig.tight_layout()
out_fig = os.path.join(fig_dir, "fig_rival_count_by_year.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Rival count figure: {out_fig}")
