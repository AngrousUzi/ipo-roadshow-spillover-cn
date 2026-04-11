"""
For each of 6 intraday event windows, produce one figure showing:
  - Rows: CAR (top) and CAV (bottom)
  - Columns: Stage 1 (2009-2015) and Stage 2 (2016-2024)
  - Within each panel: est1 / est2 / est3 grouped bars with 95% CI

6 output figures:
  fig_pre_30min.png
  fig_pre_1hr.png
  fig_during_30min.png
  fig_during_1hr.png
  fig_post_30min.png
  fig_post_1hr.png
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Window definitions: (figure_name, period_label, car_cols, cav_cols)
# car_cols / cav_cols are [est1, est2, est3] for with925 variant
# ---------------------------------------------------------------------------
WINDOWS = [
    (
        "pre_30min", "Pre-roadshow 30 min",
        ["car_before_start_30min_with925_est1",
         "car_before_start_30min_with925_est2",
         "car_before_start_30min_with925_est3"],
        ["cav_before_start_30min_with925_est1",
         "cav_before_start_30min_with925_est2",
         "cav_before_start_30min_with925_est3"],
    ),
    (
        "pre_1hr", "Pre-roadshow 1 hr",
        ["car_before_start_1hr_with925_est1",
         "car_before_start_1hr_with925_est2",
         "car_before_start_1hr_with925_est3"],
        ["cav_before_start_1hr_with925_est1",
         "cav_before_start_1hr_with925_est2",
         "cav_before_start_1hr_with925_est3"],
    ),
    (
        "during_30min", "During roadshow (first 30 min)",
        ["car_after_start_30min_with925_est1",
         "car_after_start_30min_with925_est2",
         "car_after_start_30min_with925_est3"],
        ["cav_after_start_30min_with925_est1",
         "cav_after_start_30min_with925_est2",
         "cav_after_start_30min_with925_est3"],
    ),
    (
        "during_1hr", "During roadshow (first 1 hr)",
        ["car_after_start_1hr_with925_est1",
         "car_after_start_1hr_with925_est2",
         "car_after_start_1hr_with925_est3"],
        ["cav_after_start_1hr_with925_est1",
         "cav_after_start_1hr_with925_est2",
         "cav_after_start_1hr_with925_est3"],
    ),
    (
        "post_30min", "Post-roadshow 30 min",
        ["car_after_end_30min_with925_est1",
         "car_after_end_30min_with925_est2",
         "car_after_end_30min_with925_est3"],
        ["cav_after_end_30min_with925_est1",
         "cav_after_end_30min_with925_est2",
         "cav_after_end_30min_with925_est3"],
    ),
    (
        "post_1hr", "Post-roadshow 1 hr",
        ["car_after_end_1hr_with925_est1",
         "car_after_end_1hr_with925_est2",
         "car_after_end_1hr_with925_est3"],
        ["cav_after_end_1hr_with925_est1",
         "cav_after_end_1hr_with925_est2",
         "cav_after_end_1hr_with925_est3"],
    ),
]

# collect all needed columns
ALL_COLS = []
for _, _, car_cols, cav_cols in WINDOWS:
    ALL_COLS += car_cols + cav_cols

# ---------------------------------------------------------------------------
# Load and aggregate to IPO level
# ---------------------------------------------------------------------------
print("Loading car_cav_windows.csv ...")
carv = pd.read_csv(os.path.join(OUT_DIR, "car_cav_windows.csv"), encoding="utf-8-sig")
carv["event_date"] = pd.to_datetime(carv["event_date"])
carv["year"] = carv["event_date"].dt.year

idx = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "IPO_roadshow_index_2009_with_trading_days.csv"),
    encoding="utf-8-sig", encoding_errors="replace",
    usecols=["Stkcd", "INDEX2009"],
)
idx.columns = ["stkcd_ipo", "ipo_id"]
carv = carv.merge(idx, on="ipo_id", how="left")

agg_cols = [c for c in ALL_COLS if c in carv.columns]
ipo = (
    carv.groupby(["ipo_id", "year"])[agg_cols]
    .mean()
    .reset_index()
)
ipo["stage"] = ipo["year"].apply(
    lambda y: "Stage 1\n(2009–2015)" if y <= 2015 else "Stage 2\n(2016–2024)"
)

STAGES = ["Stage 1\n(2009–2015)", "Stage 2\n(2016–2024)"]
STAGE_COLORS = ["#4878CF", "#D65F5F"]
EST_LABELS = ["Est 1", "Est 2", "Est 3"]
EST_HATCHES = ["", "//", ".."]

# ---------------------------------------------------------------------------
# Helper: mean ± 95% CI + significance star
# ---------------------------------------------------------------------------
def summary(s):
    s = s.dropna()
    n = len(s)
    if n < 2:
        return np.nan, np.nan, np.nan, ""
    m  = s.mean()
    se = s.std() / np.sqrt(n)
    _, p = stats.ttest_1samp(s, 0)
    star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
    return m, m - 1.96 * se, m + 1.96 * se, star

# ---------------------------------------------------------------------------
# Draw one panel (ax) for a given metric and set of columns
# ---------------------------------------------------------------------------
def draw_panel(ax, df, cols, metric, ylabel):
    """
    Grouped bar chart: x-axis = Stage 1 / Stage 2
                       groups  = est1, est2, est3
    """
    n_stages = len(STAGES)
    n_ests   = len(cols)           # always 3
    bw       = 0.22
    offsets  = np.linspace(-(n_ests - 1) / 2 * bw, (n_ests - 1) / 2 * bw, n_ests)
    x        = np.arange(n_stages)

    for ei, (col, lbl, hatch) in enumerate(zip(cols, EST_LABELS, EST_HATCHES)):
        if col not in df.columns:
            continue
        means, los, his, stars = [], [], [], []
        for stage in STAGES:
            sub = df[df["stage"] == stage][col]
            m, lo, hi, star = summary(sub)
            means.append(m); los.append(lo); his.append(hi); stars.append(star)

        for si in range(n_stages):
            clr = STAGE_COLORS[si]
            xi  = x[si] + offsets[ei]
            ax.bar(xi, means[si], width=bw, color=clr, alpha=0.75,
                   hatch=hatch, edgecolor="white", label=f"{lbl} ({STAGES[si].replace(chr(10),' ')})" if ei == 0 else None,
                   zorder=3)
            if not np.isnan(means[si]):
                err_lo = means[si] - los[si]
                err_hi = his[si] - means[si]
                ax.errorbar(xi, means[si], yerr=[[err_lo], [err_hi]],
                            fmt="none", color="black", capsize=3, linewidth=1, zorder=4)
                if stars[si]:
                    ypos = his[si] if means[si] >= 0 else los[si]
                    vshift = abs(means[si]) * 0.08 + abs(his[si] - los[si]) * 0.05
                    ax.text(xi, ypos + vshift if means[si] >= 0 else ypos - vshift,
                            stars[si], ha="center",
                            va="bottom" if means[si] >= 0 else "top",
                            fontsize=8, color="#800000", fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Custom legend: est patterns + stage colors
    from matplotlib.patches import Patch
    legend_handles = []
    for ei, (lbl, hatch) in enumerate(zip(EST_LABELS, EST_HATCHES)):
        legend_handles.append(
            Patch(facecolor="gray", hatch=hatch, edgecolor="black", label=lbl, alpha=0.7)
        )
    for stage, clr in zip(STAGES, STAGE_COLORS):
        legend_handles.append(
            Patch(facecolor=clr, label=stage.replace("\n", " "), alpha=0.8)
        )
    ax.legend(handles=legend_handles, fontsize=7, ncol=2,
              loc="best", framealpha=0.7)

# ---------------------------------------------------------------------------
# Main loop: one figure per window
# ---------------------------------------------------------------------------
print("Drawing figures ...")
for fname, period_label, car_cols, cav_cols in WINDOWS:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        f"Rival-Firm Abnormal Return & Volume — {period_label}\n"
        f"(IPO-level mean across rivals, with 9:25 auction, est1/2/3)",
        fontsize=12, fontweight="bold",
    )

    draw_panel(axes[0], ipo, car_cols, "CAR",
               "CAR (cumulative abnormal return)")
    draw_panel(axes[1], ipo, cav_cols, "CAV",
               "CAV (cumulative abnormal volume)")

    axes[0].set_title("CAR", fontsize=11, fontweight="bold")
    axes[1].set_title("CAV", fontsize=11, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"fig_{fname}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

print("All done.")

# ---------------------------------------------------------------------------
# Yearly trend figures: one figure per estimator (est1 / est2 / est3)
# Each figure: 2 panels (CAR top, CAV bottom)
# Each panel: 6 lines, one per window (pre 30min, pre 1hr, during 30min,
#             during 1hr, post 30min, post 1hr)
# ---------------------------------------------------------------------------
WINDOW_STYLES = [
    # (label,           car_col_template,                  cav_col_template,                  linestyle, marker)
    ("pre 30min",    "car_before_start_30min_with925_{e}", "cav_before_start_30min_with925_{e}", "-",  "o"),
    ("pre 1hr",      "car_before_start_1hr_with925_{e}",   "cav_before_start_1hr_with925_{e}",   "--", "o"),
    ("during 30min", "car_after_start_30min_with925_{e}",  "cav_after_start_30min_with925_{e}",  "-",  "s"),
    ("during 1hr",   "car_after_start_1hr_with925_{e}",    "cav_after_start_1hr_with925_{e}",    "--", "s"),
    ("post 30min",   "car_after_end_30min_with925_{e}",    "cav_after_end_30min_with925_{e}",    "-",  "^"),
    ("post 1hr",     "car_after_end_1hr_with925_{e}",      "cav_after_end_1hr_with925_{e}",      "--", "^"),
]

WIN_COLORS = ["#4878CF", "#1F4E8C", "#E8851A", "#A85C00", "#4DAF52", "#2A7A2E"]

years = sorted(ipo["year"].unique())

print("Drawing yearly trend figures ...")
for est in ["est1", "est2", "est3"]:
    fig, (ax_car, ax_cav) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"Yearly Trend of Rival-Firm Abnormal Return & Volume by Event Window\n"
        f"({est.upper()}, with 9:25 auction, IPO-level mean across rivals)",
        fontsize=12, fontweight="bold",
    )

    for (label, car_tmpl, cav_tmpl, ls, mk), clr in zip(WINDOW_STYLES, WIN_COLORS):
        car_col = car_tmpl.format(e=est)
        cav_col = cav_tmpl.format(e=est)

        for ax, col, metric in [(ax_car, car_col, "CAR"), (ax_cav, cav_col, "CAV")]:
            if col not in ipo.columns:
                continue
            yr_means = [ipo[ipo["year"] == y][col].mean() for y in years]
            ax.plot(years, yr_means, linestyle=ls, marker=mk, color=clr,
                    label=label, linewidth=1.6, markersize=5)

    for ax, metric in [(ax_car, "CAR"), (ax_cav, "CAV")]:
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.axvspan(2008.5, 2015.5, alpha=0.07, color="#4878CF", label="_Stage 1")
        ax.axvspan(2015.5, 2024.5, alpha=0.07, color="#D65F5F", label="_Stage 2")
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.8)
        # Stage labels
        ax.text(2012, ax.get_ylim()[0], "Stage 1", ha="center",
                fontsize=8, color="#2255AA", alpha=0.7)
        ax.text(2020, ax.get_ylim()[0], "Stage 2", ha="center",
                fontsize=8, color="#AA2222", alpha=0.7)

    ax_cav.set_xticks(years)
    ax_cav.set_xticklabels(years, rotation=45, fontsize=8)
    ax_cav.set_xlabel("Year", fontsize=10)
    ax_car.set_title("CAR — Cumulative Abnormal Return", fontsize=10, fontweight="bold")
    ax_cav.set_title("CAV — Cumulative Abnormal Volume", fontsize=10, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"fig_yearly_trend_{est}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

print("All done.")
