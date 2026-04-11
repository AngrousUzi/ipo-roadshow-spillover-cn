"""
Repeat stat + plot analysis using LLM-filtered competitive firm matching.

For each of 2 similarity dimensions (sim_scope, sim_main), keep only the top-3.8508% most similar ipo-rival pairs
(threshold = quantile at 1 - 0.038508), then re-run the full pipeline.

Output folders (relative to this script):
  output_llm/sim_scope/
  output_llm/sim_main/
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TARGET_RATE = 0.038508
SCRIPT_DIR  = os.path.dirname(__file__)
BASE_OUT    = os.path.join(SCRIPT_DIR, "output_llm")

DIMS = ["sim_scope", "sim_main"]

# ── Window definitions (same as main analysis) ───────────────────────────────
WIN_DEF = [
    ("pre_30min",    "Pre-roadshow 30 min",
     [f"car_before_start_30min_with925_est{e}" for e in [1,2,3]],
     [f"cav_before_start_30min_with925_est{e}" for e in [1,2,3]]),
    ("pre_1hr",      "Pre-roadshow 1 hr",
     [f"car_before_start_1hr_with925_est{e}"   for e in [1,2,3]],
     [f"cav_before_start_1hr_with925_est{e}"   for e in [1,2,3]]),
    ("during_30min", "During roadshow (first 30 min)",
     [f"car_after_start_30min_with925_est{e}"  for e in [1,2,3]],
     [f"cav_after_start_30min_with925_est{e}"  for e in [1,2,3]]),
    ("during_1hr",   "During roadshow (first 1 hr)",
     [f"car_after_start_1hr_with925_est{e}"    for e in [1,2,3]],
     [f"cav_after_start_1hr_with925_est{e}"    for e in [1,2,3]]),
    ("post_30min",   "Post-roadshow 30 min",
     [f"car_after_end_30min_with925_est{e}"    for e in [1,2,3]],
     [f"cav_after_end_30min_with925_est{e}"    for e in [1,2,3]]),
    ("post_1hr",     "Post-roadshow 1 hr",
     [f"car_after_end_1hr_with925_est{e}"      for e in [1,2,3]],
     [f"cav_after_end_1hr_with925_est{e}"      for e in [1,2,3]]),
]

ALL_OUTCOME_COLS = list({c for _, _, cc, cv in WIN_DEF for c in cc + cv})

STAGES     = ["Stage 1\n(2009–2015)", "Stage 2\n(2016–2024)"]
STAGE_CLR  = ["#4878CF", "#D65F5F"]
WIN_COLORS = ["#4878CF", "#1F4E8C", "#E8851A", "#A85C00", "#4DAF52", "#2A7A2E"]
WIN_STYLES_YEAR = [
    ("pre 30min",    "car_before_start_30min_with925_est{e}", "cav_before_start_30min_with925_est{e}", "-",  "o"),
    ("pre 1hr",      "car_before_start_1hr_with925_est{e}",   "cav_before_start_1hr_with925_est{e}",   "--", "o"),
    ("during 30min", "car_after_start_30min_with925_est{e}",  "cav_after_start_30min_with925_est{e}",  "-",  "s"),
    ("during 1hr",   "car_after_start_1hr_with925_est{e}",    "cav_after_start_1hr_with925_est{e}",    "--", "s"),
    ("post 30min",   "car_after_end_30min_with925_est{e}",    "cav_after_end_30min_with925_est{e}",    "-",  "^"),
    ("post 1hr",     "car_after_end_1hr_with925_est{e}",      "cav_after_end_1hr_with925_est{e}",      "--", "^"),
]

# ── Load common data once ─────────────────────────────────────────────────────
print("Loading base data ...")
carv_raw = pd.read_csv(
    os.path.join(SCRIPT_DIR, "output", "car_cav_windows.csv"),
    encoding="utf-8-sig",
)
carv_raw["event_date"] = pd.to_datetime(carv_raw["event_date"])
carv_raw["year"]       = carv_raw["event_date"].dt.year
carv_raw["stkcd_rival"]= carv_raw["rival_fc"].str[2:].astype(int)

idx = pd.read_csv(
    os.path.join(SCRIPT_DIR, "IPO_roadshow_index_2009_with_trading_days.csv"),
    encoding="utf-8-sig", encoding_errors="replace",
    usecols=["Stkcd", "INDEX2009"],
)
idx.columns = ["stkcd_ipo", "ipo_id"]
carv_raw = carv_raw.merge(idx, on="ipo_id", how="left")

sim_data = pd.read_csv(
    os.path.join(SCRIPT_DIR, "..", "ind", "ind_all_sim_pairs_within_llm.csv"),
)

ind_llm = pd.read_csv(
    os.path.join(SCRIPT_DIR, "..", "ind", "ind_all_sim_pairs_within_llm.csv"),
)
ind_llm["sector"] = ind_llm["csrc3"].str[0]
ind_ipo = (
    ind_llm.groupby(["stkcd_i", "year"])[["sector"]]
    .first().reset_index().rename(columns={"stkcd_i": "stkcd_ipo"})
)

print(f"Base carv rows: {len(carv_raw):,}   unique ipo×rival: {carv_raw[['ipo_id','rival_fc']].drop_duplicates().shape[0]:,}")

# ── Stat helpers ──────────────────────────────────────────────────────────────
def desc(s):
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

def diff_test(a, b):
    a, b = a.dropna(), b.dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
    return {"mean_s1": a.mean(), "mean_s2": b.mean(), "t_diff": t, "p_diff": p, "sig_diff": sig}

def window_table(df, groupby=None):
    rows = []
    for fn, lbl, car_cols, cav_cols in WIN_DEF:
        for metric, cols in [("CAR", car_cols), ("CAV", cav_cols)]:
            for ei, col in enumerate(cols, 1):
                if col not in df.columns:
                    continue
                if groupby is None:
                    d = desc(df[col])
                    rows.append({"window": fn, "est": f"est{ei}", "metric": metric, **d})
                else:
                    for grp, sub in df.groupby(groupby):
                        d = desc(sub[col])
                        rows.append({"window": fn, "est": f"est{ei}", "metric": metric,
                                     groupby: grp, **d})
    return pd.DataFrame(rows)

def ci95(s):
    s = s.dropna()
    n = len(s)
    if n < 2:
        return np.nan, np.nan, np.nan
    se = s.std() / np.sqrt(n)
    m  = s.mean()
    return m, m - 1.96 * se, m + 1.96 * se

def sig_star(s):
    s = s.dropna()
    if len(s) < 2:
        return ""
    _, p = stats.ttest_1samp(s, 0)
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))

# ── Plot: per-window bar charts (est1/2/3 × stage1/2) ────────────────────────
def draw_panel(ax, df, cols, ylabel):
    n_ests = len(cols)
    bw = 0.22
    offsets = np.linspace(-(n_ests - 1) / 2 * bw, (n_ests - 1) / 2 * bw, n_ests)
    x = np.arange(len(STAGES))
    for ei, (col, hatch) in enumerate(zip(cols, ["", "//", ".."])):
        if col not in df.columns:
            continue
        ms, los, his, ss = [], [], [], []
        for stage in STAGES:
            sub = df[df["stage"] == stage][col]
            m, lo, hi = ci95(sub)
            ms.append(m); los.append(lo); his.append(hi)
            ss.append(sig_star(sub))
        for si in range(len(STAGES)):
            xi = x[si] + offsets[ei]
            ax.bar(xi, ms[si], width=bw, color=STAGE_CLR[si], alpha=0.75,
                   hatch=hatch, edgecolor="white", zorder=3)
            if not np.isnan(ms[si]):
                ax.errorbar(xi, ms[si],
                            yerr=[[ms[si] - los[si]], [his[si] - ms[si]]],
                            fmt="none", color="black", capsize=3, linewidth=1, zorder=4)
                if ss[si]:
                    ypos = his[si] if ms[si] >= 0 else los[si]
                    shift = abs(ms[si]) * 0.08 + abs(his[si] - los[si]) * 0.05
                    ax.text(xi, ypos + shift if ms[si] >= 0 else ypos - shift,
                            ss[si], ha="center",
                            va="bottom" if ms[si] >= 0 else "top",
                            fontsize=8, color="#800000", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(STAGES, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    handles = [mpatches.Patch(facecolor="gray", hatch=h, edgecolor="black",
                               label=f"Est {i+1}", alpha=0.7)
               for i, h in enumerate(["", "//", ".."])]
    handles += [mpatches.Patch(facecolor=c, label=s.replace("\n", " "), alpha=0.8)
                for s, c in zip(STAGES, STAGE_CLR)]
    ax.legend(handles=handles, fontsize=7, ncol=2, framealpha=0.7)

def make_window_figs(ipo, fig_dir, dim_label):
    for fn, period_label, car_cols, cav_cols in WIN_DEF:
        fig, (ax_car, ax_cav) = plt.subplots(1, 2, figsize=(12, 5.5))
        fig.suptitle(
            f"Rival-Firm Abnormal Return & Volume — {period_label}\n"
            f"(llm filter: {dim_label}, with 9:25 auction, est1/2/3)",
            fontsize=11, fontweight="bold",
        )
        draw_panel(ax_car, ipo, car_cols, "CAR (cumul. abnormal return)")
        draw_panel(ax_cav, ipo, cav_cols, "CAV (cumul. abnormal volume)")
        ax_car.set_title("CAR", fontsize=10, fontweight="bold")
        ax_cav.set_title("CAV", fontsize=10, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"fig_{fn}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

def make_yearly_figs(ipo, fig_dir, dim_label):
    years = sorted(ipo["year"].unique())
    for est in ["est1", "est2", "est3"]:
        fig, (ax_car, ax_cav) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(
            f"Yearly Trend by Event Window — {est.upper()}\n"
            f"(llm filter: {dim_label}, with 9:25 auction)",
            fontsize=11, fontweight="bold",
        )
        for (label, car_tmpl, cav_tmpl, ls, mk), clr in zip(WIN_STYLES_YEAR, WIN_COLORS):
            e_num = est[-1]   # "1", "2", or "3"
            car_col = car_tmpl.format(e=e_num)
            cav_col = cav_tmpl.format(e=e_num)
            for ax, col in [(ax_car, car_col), (ax_cav, cav_col)]:
                if col not in ipo.columns:
                    continue
                ms = [ipo[ipo["year"] == y][col].mean() for y in years]
                ax.plot(years, ms, linestyle=ls, marker=mk, color=clr,
                        label=label, linewidth=1.6, markersize=5)
        for ax, metric in [(ax_car, "CAR"), (ax_cav, "CAV")]:
            ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
            ax.axvspan(2008.5, 2015.5, alpha=0.07, color="#4878CF")
            ax.axvspan(2015.5, 2024.5, alpha=0.07, color="#D65F5F")
            ax.set_ylabel(metric, fontsize=11)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.8)
        ax_cav.set_xticks(years)
        ax_cav.set_xticklabels(years, rotation=45, fontsize=8)
        ax_cav.set_xlabel("Year", fontsize=10)
        ax_car.set_title("CAR", fontsize=10, fontweight="bold")
        ax_cav.set_title("CAV", fontsize=10, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"fig_yearly_trend_{est}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

# ── Main loop over dimensions ─────────────────────────────────────────────────
for dim in DIMS:
    print(f"\n{'='*60}")
    print(f"Dimension: {dim}")

    # 1. Threshold & filter LLM pairs
    threshold = sim_data[dim].quantile(1 - TARGET_RATE)
    kept      = sim_data[sim_data[dim] >= threshold][["stkcd_i", "stkcd_j", "year"]].copy()
    kept.columns = ["stkcd_ipo", "stkcd_rival", "year"]
    print(f"  Threshold: {threshold:.6f}  |  kept pairs: {len(kept):,}  rate: {len(kept)/len(sim_data):.6f}")

    # 2. Filter carv to matching (stkcd_ipo, stkcd_rival, year) triples
    carv_f = carv_raw.merge(kept, on=["stkcd_ipo", "stkcd_rival", "year"], how="inner")
    print(f"  carv rows after filter: {len(carv_f):,}")
    print(f"  unique IPOs: {carv_f['ipo_id'].nunique():,}   unique rivals: {carv_f['rival_fc'].nunique():,}")

    # 3a. Rival count per IPO per year → yearly average
    rival_count = (
        carv_f.groupby(["ipo_id", "year"])["rival_fc"]
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

    # 3b. Aggregate to IPO level
    agg_cols = [c for c in ALL_OUTCOME_COLS if c in carv_f.columns]
    ipo = (
        carv_f.groupby(["ipo_id", "year", "stkcd_ipo"])[agg_cols]
        .mean().reset_index()
    )
    ipo["stage"] = ipo["year"].apply(
        lambda y: "Stage 1\n(2009–2015)" if y <= 2015 else "Stage 2\n(2016–2024)"
    )
    ipo = ipo.merge(ind_ipo, on=["stkcd_ipo", "year"], how="left")
    print(f"  IPO-level obs: {len(ipo):,}  with industry: {ipo['sector'].notna().sum():,}")

    # 4. Create output dirs
    out_dir = os.path.join(BASE_OUT, dim)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 5. Stat tables
    s1 = ipo[ipo["stage"] == "Stage 1\n(2009–2015)"]
    s2 = ipo[ipo["stage"] == "Stage 2\n(2016–2024)"]

    df_overall    = window_table(ipo)
    df_by_stage   = window_table(ipo, groupby="stage")

    diff_rows = []
    for fn, lbl, car_cols, cav_cols in WIN_DEF:
        for metric, cols in [("CAR", car_cols), ("CAV", cav_cols)]:
            for ei, col in enumerate(cols, 1):
                if col not in ipo.columns:
                    continue
                d = diff_test(s1[col], s2[col])
                diff_rows.append({"window": fn, "est": f"est{ei}", "metric": metric, **d})
    df_stage_diff = pd.DataFrame(diff_rows)

    df_by_ind = window_table(ipo[ipo["sector"].notna()], groupby="sector")

    ind_stage_rows = []
    for fn, lbl, car_cols, cav_cols in WIN_DEF:
        for metric, cols in [("CAR", car_cols), ("CAV", cav_cols)]:
            for ei, col in enumerate(cols, 1):
                if col not in ipo.columns:
                    continue
                sub = ipo[ipo["sector"].notna()]
                for (sector, stage), grp in sub.groupby(["sector", "stage"]):
                    d = desc(grp[col])
                    ind_stage_rows.append({"window": fn, "est": f"est{ei}", "metric": metric,
                                           "sector": sector, "stage": stage, **d})
    df_ind_stage = pd.DataFrame(ind_stage_rows)
    df_year      = window_table(ipo, groupby="year")

    # 6. Write Excel
    xlsx_path = os.path.join(out_dir, "stat_summary.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_overall.to_excel(writer,    sheet_name="Overall",          index=False)
        df_by_stage.to_excel(writer,   sheet_name="By_Stage",         index=False)
        df_stage_diff.to_excel(writer, sheet_name="Stage_Diff",       index=False)
        df_by_ind.to_excel(writer,     sheet_name="By_Industry",      index=False)
        df_ind_stage.to_excel(writer,  sheet_name="Industry_x_Stage", index=False)
        df_year.to_excel(writer,       sheet_name="By_Year",          index=False)
        df_rival_yr.to_excel(writer,   sheet_name="Rival_Count_by_Year", index=False)

    # 7. Text report
    lines = []
    def hdr(t): lines.extend(["", "=" * 68, t, "=" * 68])
    def fmt(d):
        if d["n"] < 2: return f"  n={d['n']}"
        return (f"  n={int(d['n']):<5}  mean={d['mean']:>10.5f}{d['sig']:<3}"
                f"  t={d['t']:>7.3f}  p_t={d['p_t']:.4f}  p_w={d['p_w']:.4f}")

    hdr(f"OVERALL  [{dim}  threshold={threshold:.4f}  rate={TARGET_RATE}]")
    for _, row in df_overall.iterrows():
        lines.append(f"  {row['window']:<14} {row['est']}  {row['metric']}: " + fmt(row))

    hdr("STAGE ANALYSIS")
    for stage in ["Stage 1\n(2009–2015)", "Stage 2\n(2016–2024)"]:
        lines.append(f"\n  {stage.replace(chr(10),' ')}")
        sub = df_by_stage[df_by_stage["stage"] == stage]
        for _, row in sub.iterrows():
            lines.append(f"    {row['window']:<14} {row['est']}  {row['metric']}: " + fmt(row))

    hdr("STAGE DIFFERENCE  (Welch t-test)")
    for _, row in df_stage_diff.iterrows():
        lines.append(
            f"  {row['window']:<14} {row['est']}  {row['metric']}:"
            f"  S1={row['mean_s1']:>9.5f}  S2={row['mean_s2']:>9.5f}"
            f"  t={row['t_diff']:>7.3f}  p={row['p_diff']:.4f}  {row['sig_diff']}"
        )

    txt_path = os.path.join(out_dir, "stat_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Stat output: {xlsx_path}")

    # 8. Figures
    make_window_figs(ipo, fig_dir, dim)
    make_yearly_figs(ipo, fig_dir, dim)

    # Rival count figure
    fig, ax = plt.subplots(figsize=(10, 4.5))
    years_rc = df_rival_yr["year"].values
    avg      = df_rival_yr["avg_rivals"].values
    std      = df_rival_yr["std_rivals"].values
    n_ipos   = df_rival_yr["n_ipos"].values
    ax.bar(years_rc, avg, color=["#4878CF" if y <= 2015 else "#D65F5F" for y in years_rc],
           alpha=0.75, zorder=3, width=0.7)
    ax.errorbar(years_rc, avg, yerr=std / np.sqrt(n_ipos),
                fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
    for x, a, n in zip(years_rc, avg, n_ipos):
        ax.text(x, a + std[list(years_rc).index(x)] / np.sqrt(n) + avg.max() * 0.02,
                f"{a:.1f}", ha="center", va="bottom", fontsize=7)
    ax.axvline(2015.5, color="gray", linewidth=1.2, linestyle="--", label="Stage break (2016)")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Avg rivals per IPO", fontsize=10)
    ax.set_title(
        f"Yearly Average Competitive Firm Count per IPO\n"
        f"(llm filter: {dim}, threshold={threshold:.4f})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    # secondary: n_ipos line
    ax2 = ax.twinx()
    ax2.plot(years_rc, n_ipos, "k--o", markersize=4, linewidth=1.2, label="# IPOs (right)")
    ax2.set_ylabel("# IPOs in sample", fontsize=9, color="black")
    ax2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_rival_count_by_year.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Figures:     {fig_dir}  ({len(os.listdir(fig_dir))} files)")

print("\nAll dimensions complete.")
