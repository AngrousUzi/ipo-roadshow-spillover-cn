"""
For each of 6 intraday event windows, produce figures showing:
  - est1 / est2 / est3 grouped bars × Stage 1 / Stage 2

Figures are produced for three groups:
  all/        : all roadshows
  start_09/   : roadshows starting at 09:00
  start_14/   : roadshows starting at 14:00

Start time sourced from anns/IPO_index.xlsx (column '开始时间').

Output structure under carv/output/figures/:
  {group}/fig_{window}.png
  {group}/fig_yearly_trend_{est}.png
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
FIG_DIR = os.path.join(OUT_DIR, "figures")

WINDOWS = [
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

WINDOW_STYLES = [
    ("pre 30min",    "car_before_start_30min_with925_est{e}", "cav_before_start_30min_with925_est{e}", "-",  "o"),
    ("pre 1hr",      "car_before_start_1hr_with925_est{e}",   "cav_before_start_1hr_with925_est{e}",   "--", "o"),
    ("during 30min", "car_after_start_30min_with925_est{e}",  "cav_after_start_30min_with925_est{e}",  "-",  "s"),
    ("during 1hr",   "car_after_start_1hr_with925_est{e}",    "cav_after_start_1hr_with925_est{e}",    "--", "s"),
    ("post 30min",   "car_after_end_30min_with925_est{e}",    "cav_after_end_30min_with925_est{e}",    "-",  "^"),
    ("post 1hr",     "car_after_end_1hr_with925_est{e}",      "cav_after_end_1hr_with925_est{e}",      "--", "^"),
]

ALL_COLS = list({c for _, _, cc, cv in WINDOWS for c in cc + cv})

STAGES      = ["Stage 1\n(2009–2015)", "Stage 2\n(2016–2024)"]
STAGE_COLORS= ["#4878CF", "#D65F5F"]
WIN_COLORS  = ["#4878CF", "#1F4E8C", "#E8851A", "#A85C00", "#4DAF52", "#2A7A2E"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading car_cav_windows.csv ...")
carv = pd.read_csv(os.path.join(OUT_DIR, "car_cav_windows.csv"), encoding="utf-8-sig")
carv["event_date"] = pd.to_datetime(carv["event_date"])
carv["year"] = carv["event_date"].dt.year

print("Loading IPO_index.xlsx ...")
idx = pd.read_excel(
    os.path.join(os.path.dirname(__file__), "..", "anns", "IPO_index.xlsx"),
    usecols=[0, 1, 8],
)
idx.columns = ["ipo_id", "stkcd_ipo", "start_time"]
idx["start_time"] = idx["start_time"].astype(str).str.strip()

carv = carv.merge(idx[["ipo_id", "stkcd_ipo", "start_time"]], on="ipo_id", how="left")

agg_cols = [c for c in ALL_COLS if c in carv.columns]
ipo_base = (
    carv.groupby(["ipo_id", "year", "stkcd_ipo", "start_time"])[agg_cols]
    .mean()
    .reset_index()
)
ipo_base["stage"] = ipo_base["year"].apply(
    lambda y: "Stage 1\n(2009–2015)" if y <= 2015 else "Stage 2\n(2016–2024)"
)
print(f"IPO obs: {len(ipo_base)}  start_time breakdown:\n{ipo_base['start_time'].value_counts()}")

# ---------------------------------------------------------------------------
# Helpers
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

def draw_panel(ax, df, cols, ylabel, subtitle=""):
    bw = 0.22
    offsets = np.linspace(-bw, bw, 3)
    x = np.arange(len(STAGES))
    for ei, (col, hatch) in enumerate(zip(cols, ["", "//", ".."])):
        if col not in df.columns:
            continue
        ms, los, his, ss = [], [], [], []
        for stage in STAGES:
            sub = df[df["stage"] == stage][col]
            m, lo, hi, star = summary(sub)
            ms.append(m); los.append(lo); his.append(hi); ss.append(star)
        for si in range(len(STAGES)):
            xi = x[si] + offsets[ei]
            ax.bar(xi, ms[si], width=bw, color=STAGE_COLORS[si],
                   alpha=0.75, hatch=hatch, edgecolor="white", zorder=3)
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
    if subtitle:
        ax.set_title(subtitle, fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    handles = ([mpatches.Patch(facecolor="gray", hatch=h, edgecolor="black",
                               label=f"Est {i+1}", alpha=0.7)
                for i, h in enumerate(["", "//", ".."])] +
               [mpatches.Patch(facecolor=c, label=s.replace("\n", " "), alpha=0.8)
                for s, c in zip(STAGES, STAGE_COLORS)])
    ax.legend(handles=handles, fontsize=7, ncol=2, framealpha=0.7)

def make_window_figs(ipo, fig_dir, group_label):
    for fname, period_label, car_cols, cav_cols in WINDOWS:
        fig, (ax_car, ax_cav) = plt.subplots(1, 2, figsize=(12, 5.5))
        fig.suptitle(
            f"Rival-Firm Abnormal Return & Volume — {period_label}\n"
            f"({group_label}, with 9:25 auction, est1/2/3)",
            fontsize=11, fontweight="bold",
        )
        draw_panel(ax_car, ipo, car_cols, "CAR (cumul. abnormal return)", "CAR")
        draw_panel(ax_cav, ipo, cav_cols, "CAV (cumul. abnormal volume)", "CAV")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"fig_{fname}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

def make_yearly_figs(ipo, fig_dir, group_label):
    years = sorted(ipo["year"].unique())
    for est in ["est1", "est2", "est3"]:
        fig, (ax_car, ax_cav) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(
            f"Yearly Trend by Event Window — {est.upper()}\n({group_label}, with 9:25 auction)",
            fontsize=11, fontweight="bold",
        )
        for (label, car_tmpl, cav_tmpl, ls, mk), clr in zip(WINDOW_STYLES, WIN_COLORS):
            car_col = car_tmpl.format(e=est[-1])
            cav_col = cav_tmpl.format(e=est[-1])
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

# ---------------------------------------------------------------------------
# Run for each group: all, 09:00, 14:00
# ---------------------------------------------------------------------------
GROUPS = [
    ("all",       None,    "All roadshows"),
    ("start_09",  "09:00", "Roadshow start 09:00"),
    ("start_14",  "14:00", "Roadshow start 14:00"),
]

for folder, start_filter, group_label in GROUPS:
    fig_dir = os.path.join(FIG_DIR, folder)
    os.makedirs(fig_dir, exist_ok=True)

    ipo = ipo_base if start_filter is None else ipo_base[ipo_base["start_time"] == start_filter].copy()
    print(f"\n[{group_label}]  n={len(ipo)}")

    print(f"  Drawing window figures ...")
    make_window_figs(ipo, fig_dir, group_label)

    print(f"  Drawing yearly trend figures ...")
    make_yearly_figs(ipo, fig_dir, group_label)

    print(f"  Done → {fig_dir}  ({len(os.listdir(fig_dir))} files)")

print("\nAll done.")
