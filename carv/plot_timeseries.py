"""
Time-series plot of rival-firm CAR around IPO roadshow start (t = 0).

X-axis: trading bar index relative to roadshow start
  - bar 0  = first 5-min bar on event_date with timestamp >= start_ts
  - bar -1 = the bar immediately before (could be same day or previous trading day)
  - negative bars draw in from previous trading day naturally, no calendar gaps

Groups:
  - Start-time: 09:00 vs 14:00 (others dropped)
  - Period: 2009–2015 vs 2016–2024 (bold) + individual yearly averages (thin)

Input:  carv/output/ar_av_results.csv  (contains event_date ± 1 trading day)
Output: carv/output/car_timeseries.png
"""

import argparse
from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Paths ─────────────────────────────────────────────────────────
CARV_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = CARV_DIR / "output"
ANN_DIR    = CARV_DIR.parent / "anns"
IND_DIR    = CARV_DIR.parent / "ind"
if os.name=="nt":  # Windows
    INPUT_FILE  = OUTPUT_DIR / "ar_av_results.csv"
    INDEX_PATH  = ANN_DIR / "IPO_index.xlsx"
    TFIDF_PATH  = IND_DIR / "ind_all_sim_pairs_within_tfidf.csv"
    OUT_PLOT        = OUTPUT_DIR / "car_timeseries.png"
    OUT_CSV_PERIOD  = OUTPUT_DIR / "car_timeseries_period_avg.csv"
    OUT_CSV_YEAR    = OUTPUT_DIR / "car_timeseries_year_avg.csv"
else:
    INPUT_FILE  = OUTPUT_DIR / "ar_av_results.csv"
    INDEX_PATH  = Path("..") / "IPO_index.xlsx"
    TFIDF_PATH  = Path("..") / "ind_all_sim_pairs_within_tfidf.csv"
    OUT_PLOT        = OUTPUT_DIR / "car_timeseries.png"
    OUT_CSV_PERIOD  = OUTPUT_DIR / "car_timeseries_period_avg.csv"
    OUT_CSV_YEAR    = OUTPUT_DIR / "car_timeseries_year_avg.csv"
AR_COLS = ["ar_est1", "ar_est2", "ar_est3"]

# ±40 bars = ±200 trading minutes around roadshow start
BARS_BEFORE = 40
BARS_AFTER  = 40


# ── Data loading ──────────────────────────────────────────────────

def load_index() -> pd.DataFrame:
    df = pd.read_excel(INDEX_PATH, dtype=str)
    df["roadshow_date"] = pd.to_datetime(df["日期"], errors="coerce")
    df["start_str"]     = df["开始时间"].str.slice(0, 5)
    df = df[["INDEX2009", "roadshow_date", "start_str"]].dropna(
        subset=["roadshow_date", "start_str"]
    )
    df.rename(columns={"INDEX2009": "ipo_id"}, inplace=True)
    df = df.drop_duplicates("ipo_id")

    df["start_hour"] = pd.to_numeric(df["start_str"].str[:2], errors="coerce")
    df = df[df["start_hour"].isin([9, 14])].copy()
    df["start_type"] = df["start_hour"].map({9: "9am", 14: "2pm"})
    df["year"]   = df["roadshow_date"].dt.year
    df["period"] = df["year"].apply(lambda y: "2009–2015" if y <= 2015 else "2016–2024")
    return df[["ipo_id", "start_str", "start_type", "year", "period"]]


def build_top_rivals_set(top_n: int) -> set:
    """
    Return a set of (ipo_id, rival_fc) pairs keeping only the top-N rivals
    per IPO ranked by sim_mda descending within the same csrc3 industry.

    Stkcd in IPO_index.xlsx is numeric in Excel, so read without dtype=str
    then cast float→int→str to avoid "1391.0"-style strings.
    sim_mda is aggregated to max across years so each rival appears once.
    """
    idx_xl = pd.read_excel(INDEX_PATH, usecols=["Stkcd", "INDEX2009"], dtype=str)
    idx_xl = idx_xl.rename(columns={"Stkcd": "stkcd_ipo", "INDEX2009": "ipo_id"})
    idx_xl = idx_xl.dropna(subset=["stkcd_ipo", "ipo_id"])
    idx_xl["stkcd_ipo"] = idx_xl["stkcd_ipo"].str.zfill(6)
    idx_xl = idx_xl.drop_duplicates("ipo_id")

    tfidf = pd.read_csv(TFIDF_PATH)
    tfidf["stkcd_i"] = tfidf["stkcd_i"].astype(str).str.zfill(6)
    tfidf["stkcd_j"] = tfidf["stkcd_j"].astype(str).str.zfill(6)

    merged = tfidf.merge(idx_xl, left_on="stkcd_i", right_on="stkcd_ipo", how="inner")

    # Max sim_mda across years for each (ipo_id, csrc3, rival) → unique rivals
    best_sim = (
        merged.groupby(["ipo_id", "csrc3", "stkcd_j"], sort=False)["sim_mda"]
        .max()
        .reset_index()
    )
    top = (
        best_sim.sort_values("sim_mda", ascending=False)
        .groupby(["ipo_id", "csrc3"])
        .head(top_n)
    )

    def _to_fc(code: str) -> str:
        return ("SH" if code[0] in ("6", "9") else "SZ") + code

    top = top.copy()
    top["rival_fc"] = top["stkcd_j"].apply(_to_fc)
    return set(zip(top["ipo_id"], top["rival_fc"]))


def load_ar(index_df: pd.DataFrame,
            top_rivals: set | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        INPUT_FILE,
        usecols=["ipo_id", "rival_fc", "event_date", "timestamp"] + AR_COLS,
        low_memory=False,
    )
    df["timestamp"]  = pd.to_datetime(df["timestamp"],  errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df.dropna(subset=["timestamp", "event_date"], inplace=True)
    # Filter to relevant ipo_ids immediately to shed ~60-70% of rows
    df = df.merge(index_df, on="ipo_id", how="inner")
    if top_rivals is not None:
        mask = pd.Series(
            list(zip(df["ipo_id"], df["rival_fc"])), index=df.index
        ).isin(top_rivals)
        df = df[mask]
    return df


# ── Bar index assignment (vectorised) ─────────────────────────────

def assign_bar_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised: no Python loop over groups.

    1. Sort by (ipo_id, rival_fc, event_date, timestamp).
    2. cumcount within each (ipo_id, rival_fc, event_date) group → bar_rank.
    3. Flag ref candidates: bars on event_date with timestamp >= start_ts.
    4. For each group take the first ref candidate's bar_rank → ref_rank.
    5. bar_idx = bar_rank - ref_rank.
    """
    df = df.sort_values(["ipo_id", "rival_fc", "event_date", "timestamp"])

    # start_ts vectorised
    h = df["start_str"].str[:2].astype(int)
    m = df["start_str"].str[3:5].astype(int)
    df["start_ts"] = df["event_date"].dt.normalize() + pd.to_timedelta(h * 60 + m, unit="min")

    df["bar_rank"] = df.groupby(
        ["ipo_id", "rival_fc", "event_date"], sort=False
    ).cumcount()

    is_ref = (
        (df["timestamp"].dt.normalize() == df["event_date"].dt.normalize())
        & (df["timestamp"] >= df["start_ts"])
    )

    ref_rank = (
        df[is_ref]
        .groupby(["ipo_id", "rival_fc", "event_date"], sort=False)["bar_rank"]
        .first()
        .rename("ref_rank")
        .reset_index()
    )

    df = df.merge(ref_rank, on=["ipo_id", "rival_fc", "event_date"], how="inner")
    df["bar_idx"] = df["bar_rank"] - df["ref_rank"]
    df = df[(df["bar_idx"] >= -BARS_BEFORE) & (df["bar_idx"] <= BARS_AFTER)]
    return df.drop(columns=["bar_rank", "ref_rank", "start_ts"])





# ── CAR calculation ───────────────────────────────────────────────

def compute_car(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-event CAR for each estimator, normalised to 0 at bar_idx = 0.

    1. Average AR across rival firms within each (ipo_id, event_date, bar_idx).
    2. Cumsum over bar_idx within each event.
    3. Subtract cumsum value at bar_idx == -1 so CAR(-1) = 0 (bar 0 already reflects
       the roadshow effect).
    Returns long-format with columns: ..., est, car
    """
    grp_cols = ["ipo_id", "event_date", "bar_idx", "start_type", "year", "period"]
    ev = df.groupby(grp_cols, sort=False)[AR_COLS].mean().reset_index()
    ev.sort_values(["ipo_id", "event_date", "bar_idx"], inplace=True)

    # Cumsum and normalise for each estimator
    parts = []
    for col, est in zip(AR_COLS, [1, 2, 3]):
        sub = ev[grp_cols + [col]].copy()
        sub["car_raw"] = sub.groupby(["ipo_id", "event_date"])[col].cumsum()
        ref = (
            sub[sub["bar_idx"] == -1]
            .groupby(["ipo_id", "event_date"])["car_raw"]
            .first()
            .rename("car_ref")
            .reset_index()
        )
        sub = sub.merge(ref, on=["ipo_id", "event_date"], how="left")
        sub["car"] = sub["car_raw"] - sub["car_ref"]
        sub["est"] = est
        parts.append(sub[grp_cols + ["est", "car"]])

    return pd.concat(parts, ignore_index=True)


# ── Aggregation ───────────────────────────────────────────────────

def build_averages(ev: pd.DataFrame):
    period_avg = (
        ev.groupby(["start_type", "period", "est", "bar_idx"])["car"].mean().reset_index()
    )
    year_avg = (
        ev.groupby(["start_type", "year", "est", "bar_idx"])["car"].mean().reset_index()
    )
    return period_avg, year_avg


# ── Plotting ──────────────────────────────────────────────────────

PERIOD_COLORS = {"2009–2015": "steelblue", "2016–2024": "tomato"}

# Approximate bar offsets of session boundaries (for annotation lines)
# 9am start  → market opens at bar 0 (first bar is 9:25, ≈ 25 min after 9:00, but we
#              call that bar 0).  Lunch: 9:25+26 bars=11:30 → bar +26; resume 13:00 →
#              bar +39 (gap of 3 bars in index since no bars exist 11:30–13:00).
#              Wait — bars are continuous in index even across lunch, because we just
#              count sequential trading bars.  So session boundaries only appear as
#              natural breaks in data density, no explicit gaps in bar_idx.
# 2pm start  → bar 0 = 14:00.  Close = bar +13 (15:00).  Next-day open = bar +14.

SESSION_LINES = {
    # vertical lines at approximate bar indices of known session events
    # 9am: prev-day close is ~51 bars back; within ±40 window only "start" is visible
    "9am": [
        (0, "start"),
    ],
    # 2pm: today open ≈ bar -38 (9:25→14:00 = 38 bars); close ≈ bar +13 (14:00→15:00)
    "2pm": [
        (-38, "open"),
        (  0, "start"),
        ( 13, "close"),
    ],
}


EST_LINESTYLES = {1: "-", 2: "--", 3: ":"}

def _draw_panel(ax, st: str, est: int,
                period_avg: pd.DataFrame, year_avg: pd.DataFrame):
    st_year = year_avg[(year_avg["start_type"] == st) & (year_avg["est"] == est)]
    years   = sorted(st_year["year"].unique())
    cmap    = cm.get_cmap("tab20", max(len(years), 1))

    for i, yr in enumerate(years):
        sub = st_year[st_year["year"] == yr]
        ax.plot(sub["bar_idx"], sub["car"],
                color=cmap(i), alpha=0.35, lw=0.9, label=str(yr))

    st_period = period_avg[(period_avg["start_type"] == st) & (period_avg["est"] == est)]
    for period, color in PERIOD_COLORS.items():
        sub = st_period[st_period["period"] == period]
        if not sub.empty:
            ax.plot(sub["bar_idx"], sub["car"],
                    color=color, lw=2.5, zorder=5, label=period)

    for bar_pos, lbl in SESSION_LINES.get(st, []):
        ls  = "--" if lbl == "start" else ":"
        lw  = 1.2  if lbl == "start" else 0.8
        clr = "black" if lbl == "start" else "gray"
        ax.axvline(bar_pos, color=clr, lw=lw, linestyle=ls, alpha=0.8)

    ax.axhline(0, color="gray", lw=0.5, linestyle=":")
    ax.set_xlim(-BARS_BEFORE, BARS_AFTER)


def plot(period_avg: pd.DataFrame, year_avg: pd.DataFrame, out_path: Path):
    # Layout: 3 rows (est1/2/3) × 2 cols (9am / 2pm)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    start_configs = [("9am", "09:00 Start"), ("2pm", "14:00 Start")]

    for col_i, (st, st_title) in enumerate(start_configs):
        for row_i, est in enumerate([1, 2, 3]):
            ax = axes[row_i, col_i]
            _draw_panel(ax, st, est, period_avg, year_avg)
            ax.set_ylabel(f"CAR  (est{est}, base = bar −1)", fontsize=8)
            if row_i == 0:
                ax.set_title(st_title, fontsize=10)
            if row_i == 2:
                ax.set_xlabel("Trading bars from roadshow start  (1 bar = 5 min)")
            if row_i == 0 and col_i == 1:
                ax.legend(fontsize=7, ncol=2, loc="upper left")

    plt.suptitle("Rival-Firm CAR Around IPO Roadshow Start", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CAR time-series plot around IPO roadshow start")
    parser.add_argument(
        "--top", type=int, default=None, metavar="N",
        help="Keep only top-N rivals per IPO ranked by mda_tfidf (sim_mda) descending. "
             "Default: use all rivals.",
    )
    args = parser.parse_args()

    top_rivals = None
    if args.top is not None:
        print(f"Building top-{args.top} rival set by sim_mda...")
        top_rivals = build_top_rivals_set(args.top)
        print(f"  {len(top_rivals):,} (ipo_id, rival_fc) pairs kept")

    suffix = f"_top{args.top}" if args.top is not None else ""
    out_plot       = OUTPUT_DIR / f"car_timeseries{suffix}.png"
    out_csv_period = OUTPUT_DIR / f"car_timeseries_period_avg{suffix}.csv"
    out_csv_year   = OUTPUT_DIR / f"car_timeseries_year_avg{suffix}.csv"

    print("Loading roadshow index...")
    index_df = load_index()
    n9  = (index_df["start_type"] == "9am").sum()
    n14 = (index_df["start_type"] == "2pm").sum()
    print(f"  {len(index_df)} IPOs kept: {n9} × 09:00, {n14} × 14:00")

    print("Loading AR data (all ±1 day bars)...")
    df = load_ar(index_df, top_rivals=top_rivals)
    print(f"  {len(df):,} raw bars loaded")

    print("Assigning trading bar indices...")
    df = assign_bar_index(df)
    print(f"  {len(df):,} bars after windowing")

    print("Computing normalised CAR per event...")
    ev = compute_car(df)

    print("Building period & yearly averages...")
    period_avg, year_avg = build_averages(ev)

    print("Saving CSVs...")
    period_avg.to_csv(out_csv_period, index=False, encoding="utf-8-sig")
    year_avg.to_csv(out_csv_year,     index=False, encoding="utf-8-sig")
    print(f"  → {out_csv_period}  ({len(period_avg):,} rows)")
    print(f"  → {out_csv_year}  ({len(year_avg):,} rows)")

    print("Plotting...")
    plot(period_avg, year_avg, out_plot)
    print("Done.")


if __name__ == "__main__":
    main()
