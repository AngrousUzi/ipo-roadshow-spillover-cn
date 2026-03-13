"""
将 car_cav_results.csv（逐 bar AR/AV 序列）切分为多个时间窗口的 CAR/CAV
==========================================================================
输入：
  carv/output/car_cav_results.csv
    列：ipo_id, rival_fc, event_date, timestamp,
        ar_est1, ar_est2, ar_est3,
        av_est1, av_est2, av_est3
    timestamp 值为路演事件当日（含前后 1 交易日缓冲）的 5min bar 时间戳

  anns/IPO_index.xlsx
    需要 开始时间 / 结束时间 字段

输出：
  carv/output/car_cav_windows.csv        每竞争公司 × 每事件 × 各窗口的 CAR/CAV 标量
  carv/output/car_cav_windows_mean.csv   按 ipo_id 对竞争公司平均后的结果

窗口定义（每个窗口分 with925 / no925 两个变体）
-------------------------------------------------
full_day           : 路演当日 [09:25, 15:00]（with925） / [09:30, 15:00]（no925）
before_start_30min : 路演开始前最后 30 min（6 bars）
before_start_1hr   : 路演开始前最后  1 hr（12 bars）
after_start_30min  : 路演开始后前   30 min
after_start_1hr    : 路演开始后前    1 hr
after_end_30min    : 路演结束后前   30 min（可跨交易日）
after_end_1hr      : 路演结束后前    1 hr（可跨交易日）

CAR 采用路演当日的 AR 序列对所需 bar 求和；CAV 同理用 AV 序列。
含 09:25 变体保留 09:25 bar；不含变体剔除后再选 bar。
"""

import os
import datetime as dt
import re
from pathlib import Path

import pandas as pd
import numpy as np

# ── 路径 ──────────────────────────────────────────────────────────
if os.name == "nt":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    ANN_DIR      = PROJECT_ROOT / "anns"
else:
    ANN_DIR = Path("./anns").resolve()

CARV_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = CARV_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_FILE       = OUTPUT_DIR / "car_cav_results.csv"
INDEX_PATH       = ANN_DIR / "IPO_index.xlsx"
OUTPUT_WINDOWS   = OUTPUT_DIR / "car_cav_windows.csv"
OUTPUT_MEAN      = OUTPUT_DIR / "car_cav_windows_mean.csv"

N_BARS_30MIN = 6
N_BARS_1HR   = 12
T925         = dt.time(9, 25)
T930         = dt.time(9, 30)
T1500        = dt.time(15, 0)


# ── 帮助函数 ──────────────────────────────────────────────────────

def parse_time(time_str) -> "dt.time | None":
    """'HH:MM' 字符串 → dt.time；无效则 None。"""
    if pd.isna(time_str) or not isinstance(time_str, str) or len(time_str) < 5:
        return None
    try:
        return dt.time(int(time_str[:2]), int(time_str[3:5]))
    except Exception:
        return None


def _bars_before(ts_series: pd.Series, thresh_ts: "pd.Timestamp | None", n: int) -> pd.Series:
    """严格早于 thresh_ts 的最后 n 个 bar 的布尔掩码（ts_series 已排序）。"""
    if thresh_ts is None:
        return pd.Series(False, index=ts_series.index)
    mask = ts_series < thresh_ts
    idx  = ts_series[mask].iloc[-n:].index if mask.any() else pd.Index([])
    return ts_series.index.isin(idx)


def _bars_after(ts_series: pd.Series, thresh_ts: "pd.Timestamp | None", n: int) -> pd.Series:
    """不早于 thresh_ts 的前 n 个 bar 的布尔掩码（ts_series 已排序）。"""
    if thresh_ts is None:
        return pd.Series(False, index=ts_series.index)
    mask = ts_series >= thresh_ts
    idx  = ts_series[mask].iloc[:n].index if mask.any() else pd.Index([])
    return ts_series.index.isin(idx)


def _sum_window(df: pd.DataFrame, mask: "pd.Series | np.ndarray",
                ar_col: str, av_col: str,
                exclude_925: bool = False) -> tuple:
    """在指定 bar 掩码上对 AR/AV 列求和，返回 (car, cav)。"""
    sub = df[mask].copy()
    if exclude_925:
        sub = sub[sub["_time"] != T925]
    ar_vals = sub[ar_col].dropna()
    av_vals = sub[av_col].dropna()
    car = float(ar_vals.sum()) if not ar_vals.empty else np.nan
    cav = float(av_vals.sum()) if not av_vals.empty else np.nan
    return car, cav


def _parse_value_col(col: str) -> dict:
    """解析列名 car/cav_<window>_<with925|no925>_estN。"""
    m = re.match(r"^(car|cav)_(.+)_(with925|no925)_est(\d+)$", col)
    if not m:
        return {
            "metric": col.split("_")[0],
            "window": col,
            "variant": np.nan,
            "est": np.nan,
        }
    metric, window, variant, est = m.groups()
    return {
        "metric": metric,
        "window": window,
        "variant": variant,
        "est": int(est),
    }


# ── 主函数 ────────────────────────────────────────────────────────

def main():
    print("读取路演时间索引...")
    df_index = pd.read_excel(INDEX_PATH, dtype=str)
    df_index["roadshow_date"]  = pd.to_datetime(df_index["日期"], errors="coerce")
    df_index["roadshow_start"] = df_index["开始时间"].str.slice(0, 5)
    df_index["roadshow_end"]   = df_index["结束时间"].str.slice(0, 5)
    df_index = df_index[["INDEX2009", "roadshow_date", "roadshow_start", "roadshow_end"]].dropna(
        subset=["roadshow_date"]
    )
    df_index.rename(columns={"INDEX2009": "ipo_id"}, inplace=True)
    df_index = df_index.drop_duplicates("ipo_id")

    # 构建 ipo_id → (roadshow_start time, roadshow_end time) 映射
    def _times(row):
        return (
            parse_time(row["roadshow_start"]),
            parse_time(row["roadshow_end"]),
        )

    ipo_times = {row["ipo_id"]: _times(row) for _, row in df_index.iterrows()}

    print("读取逐 bar AR/AV 序列...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df["event_date"]  = pd.to_datetime(df["event_date"])
    df["_time"]       = df["timestamp"].dt.time

    ar_cols = ["ar_est1", "ar_est2", "ar_est3"]
    av_cols = ["av_est1", "av_est2", "av_est3"]
    est_nums = [1, 2, 3]

    windows = [
        "full_day",
        "before_start_30min",
        "before_start_1hr",
        "after_start_30min",
        "after_start_1hr",
        "after_end_30min",
        "after_end_1hr",
    ]

    all_rows = []

    grouped = df.groupby(["ipo_id", "rival_fc", "event_date"])
    total   = len(grouped)

    print(f"共 {total:,} 个 (ipo_id, rival_fc, event_date) 组合，开始切分窗口...")

    for idx_grp, ((ipo_id, rival_fc, event_date), grp) in enumerate(grouped):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        ts  = grp["timestamp"]

        # 路演开始 / 结束时间戳
        start_t, end_t = ipo_times.get(ipo_id, (None, None))
        d = event_date.date()
        rs_start = (
            pd.Timestamp(d.year, d.month, d.day, start_t.hour, start_t.minute)
            if start_t is not None else None
        )
        rs_end = (
            pd.Timestamp(d.year, d.month, d.day, end_t.hour, end_t.minute)
            if end_t is not None else None
        )

        # 路演日全天边界
        day_open_w  = pd.Timestamp(d.year, d.month, d.day, 9, 25)
        day_open_n  = pd.Timestamp(d.year, d.month, d.day, 9, 30)
        day_close   = pd.Timestamp(d.year, d.month, d.day, 15, 0)

        # 各窗口的 bar 掩码（with925 / no925 共用同一 bar 集，no925 在求和时剔除）
        masks_w = {
            "full_day":            (ts >= day_open_w)  & (ts <= day_close),
            "before_start_30min":  _bars_before(ts, rs_start, N_BARS_30MIN),
            "before_start_1hr":    _bars_before(ts, rs_start, N_BARS_1HR),
            "after_start_30min":   _bars_after( ts, rs_start, N_BARS_30MIN),
            "after_start_1hr":     _bars_after( ts, rs_start, N_BARS_1HR),
            "after_end_30min":     _bars_after( ts, rs_end,   N_BARS_30MIN),
            "after_end_1hr":       _bars_after( ts, rs_end,   N_BARS_1HR),
        }

        row = {"ipo_id": ipo_id, "rival_fc": rival_fc, "event_date": str(event_date.date())}

        for win in windows:
            mask = masks_w[win]
            for est, ar_col, av_col in zip(est_nums, ar_cols, av_cols):
                car_w, cav_w = _sum_window(grp, mask, ar_col, av_col, exclude_925=False)
                car_n, cav_n = _sum_window(grp, mask, ar_col, av_col, exclude_925=True)
                row[f"car_{win}_with925_est{est}"] = car_w
                row[f"cav_{win}_with925_est{est}"] = cav_w
                row[f"car_{win}_no925_est{est}"]   = car_n
                row[f"cav_{win}_no925_est{est}"]   = cav_n

        all_rows.append(row)

        if (idx_grp + 1) % 5000 == 0:
            pct = (idx_grp + 1) / total * 100
            print(f"  [{idx_grp+1:>7}/{total}] ({pct:.1f}%)")

    print("保存窗口结果...")
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(OUTPUT_WINDOWS, index=False, encoding="utf-8-sig")
    print(f"  → {OUTPUT_WINDOWS}")

    print("计算各 IPO 竞争公司均值...")
    value_cols = [c for c in df_out.columns if c.startswith(("car_", "cav_"))]
    df_mean = (
        df_out.groupby(["ipo_id", "event_date"])[value_cols]
        .mean()
        .reset_index()
    )
    df_mean.to_csv(OUTPUT_MEAN, index=False, encoding="utf-8-sig")
    print(f"  → {OUTPUT_MEAN}")


    df_mean["year"] = pd.to_datetime(df_mean["event_date"]).dt.year
    year_results = {}
    for year, group in df_mean.groupby("year"):
        year_results[year] = {}
        for col in value_cols:
            mean_val = group[col].mean()
            max_val = group[col].max()
            min_val = group[col].min()
            std_val = group[col].std()
            p95_val = group[col].quantile(0.95)
            p5_val = group[col].quantile(0.05)
            year_results[year][col] = {
                "mean": mean_val,
                "max": max_val,
                "min": min_val,
                "std": std_val,
                "p95": p95_val,
                "p5": p5_val,
            }
    df_year_stats = pd.DataFrame([
        {
            "year": year,
            **_parse_value_col(col),
            **stats,
        }
        for year, cols in year_results.items()
        for col, stats in cols.items()
    ])
    stats_output = OUTPUT_MEAN.with_name("car_cav_windows_year_stats.csv")
    df_year_stats.to_csv(stats_output, index=False, encoding="utf-8-sig")
    print(f"  → {stats_output}")
        
    print("完成。")


if __name__ == "__main__":
    main()
