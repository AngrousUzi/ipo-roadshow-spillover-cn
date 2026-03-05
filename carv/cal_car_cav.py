"""
计算竞争公司的 CAR / CAV
=========================
数据来源：
  - ind_pairs.csv         : IPO-竞争公司对
  - IPO_roadshow_index_2009.xlsx : 路演日期、时间
  - aggregated parquet    : 高频 tick 数据（5分钟，aggregated by_stock）
  
估计窗口（filepath.md）：
  EST1  : [-120, -30] 交易日
  EST2  : [-30,  -5]  交易日
  
事件窗口：
  路演当日完整交易时段 [09:25, 15:00]
  （路演开始时间通常 14:00，市场收盘 15:00；使用全天以捕获集合竞价到收盘的累积效应）

输出：
  carv/output/car_cav_results.csv
"""

import sys
import os
import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径设置 ──────────────────────────────────────────────────────
if os.name=="nt":
    PROJECT_ROOT    = Path(__file__).resolve().parent.parent          # PythonProject
    HIGH_FREQ_ROOT  = Path(r"D:\科研\stock_data\CN\high_freq")
    RESAMPLE_DIR    = HIGH_FREQ_ROOT / "resample"
    CALCULATE_DIR   = HIGH_FREQ_ROOT / "calculate"
    AGGREGATED_PATH = HIGH_FREQ_ROOT / "aggregated"
    ANN_DIR         = PROJECT_ROOT / "anns"
    IND_DIR         = PROJECT_ROOT / "ind"
    OUTPUT_DIR      = Path(__file__).resolve().parent / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    INDEX_PATH = ANN_DIR / "IPO_index.xlsx"
    IND_PATH = IND_DIR / "ind_pairs.csv"
else:
    BASE_DIR        = Path("./").resolve()
    PROJECT_ROOT    = Path("./ipo-roadshow-spillover-cn").resolve()           
    HIGH_FREQ_ROOT  = Path("../data").resolve()            
    RESAMPLE_DIR    = HIGH_FREQ_ROOT / "resample"
    CALCULATE_DIR   = HIGH_FREQ_ROOT / "calculate"
    AGGREGATED_PATH = HIGH_FREQ_ROOT / "aggregated"
    # ANN_DIR         = PROJECT_ROOT / "anns"
    # IND_DIR         = PROJECT_ROOT / "ind"
    OUTPUT_DIR      = Path(__file__).resolve().parent / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    INDEX_PATH = BASE_DIR / "IPO_index.xlsx"
    IND_PATH = BASE_DIR / "ind_pairs.csv"

for p in [str(RESAMPLE_DIR), str(CALCULATE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
# print(sys.path)
from cal_return import get_complete_return   # noqa: E402
from indicator_cal import calculate_car, calculate_cav  # noqa: E402

warnings.filterwarnings("ignore")

# ── 参数 ─────────────────────────────────────────────────────────
FREQ              = "5min"
MARKET_CODE       = "SH000300"
EST1_START_OFFSET = -120   # 包含
EST1_END_OFFSET   = -30    # 包含
EST2_START_OFFSET = -30    # 包含
EST2_END_OFFSET   = -5     # 包含
EVENT_START_OFFSET = 0
EVENT_END_OFFSET   = 0     # 只看路演当日
OUTPUT_FILE       = OUTPUT_DIR / "car_cav_results.csv"
ERROR_LOG         = OUTPUT_DIR / "errors.txt"

# ── 工具函数 ──────────────────────────────────────────────────────

def load_trading_dates() -> pd.DatetimeIndex:
    with open(AGGREGATED_PATH / "metadata" / "date_range.json", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DatetimeIndex(pd.to_datetime(data["workdays"], format="%Y%m%d")).sort_values()


def get_exchange_symbol(code: str) -> str:
    """根据证券代码判断交易所"""
    code=str(code).zfill(6)  # 确保是6位字符串
    if code.startswith("0") or code.startswith("3"):
        return "SZ"
    elif code.startswith("6"):
        return "SH"
    elif code.startswith("83") or code.startswith("87") or code.startswith("88") or code.startswith("92"):
        return "BJ"
    elif code.startswith("90"):
        return "SH"
    elif code.startswith("20"):
        return "BJ"    
    else:
        return "未知交易所"

def full_code(stkcd) -> str:
    code = str(int(stkcd)).zfill(6)
    return get_exchange_symbol(code) + code


def nth_trading_day(trading_dates: pd.DatetimeIndex, ref_date: pd.Timestamp, offset: int) -> pd.Timestamp:
    """
    在 trading_dates 中找到 ref_date 的位置，返回偏移 offset 个交易日后的日期。
    offset 为负时表示之前。
    """
    pos = trading_dates.searchsorted(ref_date)
    # searchsorted 返回插入点，若 ref_date 本身在列表内则 pos 正好是它的索引
    if pos >= len(trading_dates) or trading_dates[pos] != ref_date:
        raise ValueError(f"{ref_date.date()} 不在交易日列表中")
    new_pos = pos + offset
    if new_pos < 0 or new_pos >= len(trading_dates):
        raise IndexError(f"偏移后位置 {new_pos} 超出范围 (0, {len(trading_dates)-1})")
    return trading_dates[new_pos]


def log_error(msg: str):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(str(dt.datetime.now()) + " | " + msg + "\n")


def build_params(start: pd.Timestamp, end: pd.Timestamp) -> dict:
    return {
        "start": start.to_pydatetime(),
        "end":   end.to_pydatetime(),
        "freq":  FREQ,
        "base_dir": str(OUTPUT_DIR),
        "aggregated_base_path": str(AGGREGATED_PATH),
    }


# ── 数据加载 ──────────────────────────────────────────────────────

def load_roadshow_index() -> pd.DataFrame:
    """加载路演索引，返回含 INDEX2009、Stkcd、roadshow_date 三列。"""
    df = pd.read_excel(INDEX_PATH, dtype=str)
    df["roadshow_date"] = pd.to_datetime(df["日期"], errors="coerce")
    df["roadshow_start"] = df["开始时间"]    # 只保留必要的列，过滤掉缺失路演日期的行
    df = df[["INDEX2009", "Stkcd", "roadshow_date", "roadshow_start"]].dropna(subset=["roadshow_date"])
    df.rename(columns={"INDEX2009": "ipo_id", "Stkcd": "ipo_stkcd_raw"}, inplace=True)
    return df


def load_ind_pairs() -> pd.DataFrame:
    return pd.read_csv(
        IND_PATH,
        usecols=["ipo_id", "ipo_stkcd", "rival_stkcd"],
        dtype={"ipo_stkcd": str, "rival_stkcd": str},
    )


# ── 核心计算 ──────────────────────────────────────────────────────

def compute_car_cav_for_rival(
    rival_fc: str,
    market_return: pd.DataFrame,
    start_needed: pd.Timestamp,
    end_needed: pd.Timestamp,
    events: list[dict],
) -> list[pd.DataFrame]:
    """
    对单只竞争公司股票，批量计算所有关联事件的 CAR/CAV。

    Parameters
    ----------
    rival_fc     : full code, e.g. "SZ000089"
    market_return: SH000300 收益率 DataFrame，需覆盖所需区间，列为 ['Return','Volume']
    start_needed : 本竞争公司所有事件中最早的估计窗口开始日
    end_needed   : 本竞争公司所有事件中最晚的事件窗口结束日
    events       : list of dicts with keys:
                   ipo_id, event_date, est1_start, est1_end, est2_start, est2_end

    Returns
    -------
    list[pd.DataFrame]: 每个事件对应一个 DataFrame，列为
        [timestamp, car_est1, car_est2, cav_est1, cav_est2, ipo_id, rival_fc, event_date]
        行 = 事件窗口内的每个 5min 时间戳（累积值序列）
    """
    # ── 加载竞争公司 5分钟收益率 ──
    params = build_params(start_needed, end_needed)
    rival_data = get_complete_return(rival_fc, is_index=False, params=params)
    if rival_data is None or rival_data.empty:
        for ev in events:
            log_error(f"rival {rival_fc} | ipo_id {ev['ipo_id']} | no data")
        return []

    # cal_return returns columns ['Return', 'Volume']; rename to lowercase for indicator_cal
    rival_data = rival_data.rename(columns={"Return": "return"})
    # market_return: keep original columns, extract 'Return' Series for market arg
    mkt_series = market_return["Return"]  # DatetimeIndex Series

    results = []
    for ev in events:
        ev_date_str = str(ev["event_date"].date())

        try:
            # ── CAR （返回事件窗口内的 cumsum Series）──
            car_est1 = calculate_car(
                stock_data=rival_data,
                market_data=mkt_series,
                event_start=ev_date_str,
                event_end=ev_date_str,
                estimation_start=str(ev["est1_start"].date()),
                estimation_end=str(ev["est1_end"].date()),
            )
            car_est2 = calculate_car(
                stock_data=rival_data,
                market_data=mkt_series,
                event_start=ev_date_str,
                event_end=ev_date_str,
                estimation_start=str(ev["est2_start"].date()),
                estimation_end=str(ev["est2_end"].date()),
            )

            # ── CAV （返回事件窗口内的 cumsum Series）──
            cav_est1 = calculate_cav(
                stock_data=rival_data,
                event_start=ev_date_str,
                event_end=ev_date_str,
                estimation_start=str(ev["est1_start"].date()),
                estimation_end=str(ev["est1_end"].date()),
                volume_col="Volume",
            )
            cav_est2 = calculate_cav(
                stock_data=rival_data,
                event_start=ev_date_str,
                event_end=ev_date_str,
                estimation_start=str(ev["est2_start"].date()),
                estimation_end=str(ev["est2_end"].date()),
                volume_col="Volume",
            )

            # 将四个序列拼成一张宽表，行 = 事件窗口内的 5min 时间戳
            def _to_series(x, name):
                """scalar NaN 或 Series 都可处理"""
                if isinstance(x, pd.Series):
                    return x.rename(name)
                return pd.Series([np.nan], name=name)

            df_ev = pd.concat(
                [_to_series(car_est1, "car_est1"),
                 _to_series(car_est2, "car_est2"),
                 _to_series(cav_est1, "cav_est1"),
                 _to_series(cav_est2, "cav_est2")],
                axis=1,
            )

        except Exception as e:
            log_error(f"rival {rival_fc} | ipo_id {ev['ipo_id']} | event {ev_date_str} | {e}")
            df_ev = pd.DataFrame(
                [{"car_est1": np.nan, "car_est2": np.nan,
                  "cav_est1": np.nan, "cav_est2": np.nan}]
            )

        df_ev.index.name = "timestamp"
        df_ev = df_ev.reset_index()
        df_ev["ipo_id"]     = ev["ipo_id"]
        df_ev["rival_fc"]   = rival_fc
        df_ev["event_date"] = ev_date_str
        results.append(df_ev)

    return results


# ── 主流程 ────────────────────────────────────────────────────────

def main():
    print("加载交易日历...")
    trading_dates = load_trading_dates()

    print("加载路演索引...")
    roadshow_idx = load_roadshow_index()

    print("加载竞争公司对...")
    pairs = load_ind_pairs()

    # 合并路演日期到 pairs
    pairs = pairs.merge(
        roadshow_idx[["ipo_id", "roadshow_date"]],
        on="ipo_id", how="left",
    )
    pairs = pairs.dropna(subset=["roadshow_date"])
    print(f"有效配对数：{len(pairs):,}")

    # ── 计算每个事件的交易日偏移 ──────────────────────────────────
    def safe_offset(ref_date, offset):
        try:
            return nth_trading_day(trading_dates, ref_date, offset)
        except (ValueError, IndexError):
            return pd.NaT

    print("计算交易日偏移（可能需要数分钟）...")
    unique_events = roadshow_idx[["ipo_id", "roadshow_date"]].drop_duplicates("ipo_id").copy()
    unique_events["est1_start"] = unique_events["roadshow_date"].apply(
        lambda d: safe_offset(d, EST1_START_OFFSET))
    unique_events["est1_end"]   = unique_events["roadshow_date"].apply(
        lambda d: safe_offset(d, EST1_END_OFFSET))
    unique_events["est2_start"] = unique_events["roadshow_date"].apply(
        lambda d: safe_offset(d, EST2_START_OFFSET))
    unique_events["est2_end"]   = unique_events["roadshow_date"].apply(
        lambda d: safe_offset(d, EST2_END_OFFSET))
    unique_events["event_end"]  = unique_events["roadshow_date"].apply(
        lambda d: safe_offset(d, EVENT_END_OFFSET))

    # 过滤掉偏移计算失败的行
    unique_events = unique_events.dropna(
        subset=["est1_start", "est1_end", "est2_start", "est2_end"])

    # 合并回 pairs
    pairs = pairs.merge(unique_events.drop(columns=["roadshow_date"]), on="ipo_id", how="inner")
    print(f"计算偏移后有效配对数：{len(pairs):,}")

    # ── 预加载市场指数（全跨度一次性加载）────────────────────────
    global_start = pairs["est1_start"].min()
    global_end   = pairs["event_end"].max()
    print(f"加载市场指数 {MARKET_CODE}  {global_start.date()} ~ {global_end.date()} ...")
    market_data = get_complete_return(
        MARKET_CODE,
        is_index=True,
        params=build_params(global_start, global_end),
    )
    if market_data is None:
        raise RuntimeError(f"无法加载市场指数 {MARKET_CODE}")
    print(f"市场指数共 {len(market_data):,} 条 5分钟数据")

    # ── 按竞争公司分组，逐股计算 ──────────────────────────────────
    # 断点续算：读取已有结果
    if OUTPUT_FILE.exists():
        # existing = pd.read_csv(OUTPUT_FILE, usecols=["ipo_id", "rival_fc"])
        # done_keys = set(zip(existing["ipo_id"], existing["rival_fc"]))
        # print(f"已有 {len(done_keys):,} 条结果，将跳过已完成的配对")
        OUTPUT_FILE.unlink()  # TEMP:删除已有文件，重新计算全部配对
        done_keys = set()
    else:
        done_keys = set()

    # 筛掉已完成的
    pairs["rival_fc_col"] = pairs["rival_stkcd"].apply(full_code)
    pairs_todo = pairs[
        ~pairs.apply(lambda r: (r["ipo_id"], r["rival_fc_col"]) in done_keys, axis=1)
    ]
    print(f"待计算配对：{len(pairs_todo):,}")

    # 按竞争公司分组
    grouped = pairs_todo.groupby("rival_fc_col")
    total_rivals = len(grouped)
    print(f"待加载竞争公司数：{total_rivals:,}")

    all_results = []
    write_header = not OUTPUT_FILE.exists()

    for i, (rival_fc, grp) in enumerate(grouped):
        # 本竞争公司所需的最宽日期范围
        print(f"\n[{i+1}/{total_rivals}] 处理竞争公司 {rival_fc} ...")
        r_start = grp["est1_start"].min()
        r_end   = grp["event_end"].max()

        # 构建 events list
        events = grp[[
            "ipo_id", "roadshow_date",
            "est1_start", "est1_end",
            "est2_start", "est2_end",
            "event_end",
        ]].rename(columns={"roadshow_date": "event_date"}).to_dict("records")

        records = compute_car_cav_for_rival(
            rival_fc=rival_fc,
            market_return=market_data.copy(),
            start_needed=r_start,
            end_needed=r_end,
            events=events,
        )
        all_results.extend(records)

        # 每 100 只股票写出一次
        if (i + 1) % 100 == 0 or (i + 1) == total_rivals:
            if all_results:
                df_out = pd.concat(all_results, ignore_index=True)
                # 调整列顺序：标识德列在前
                cols_order = ["ipo_id", "rival_fc", "event_date", "timestamp",
                              "car_est1", "car_est2", "cav_est1", "cav_est2"]
                df_out = df_out[[c for c in cols_order if c in df_out.columns]]
                df_out.to_csv(
                    OUTPUT_FILE,
                    mode="a",
                    header=write_header,
                    index=False,
                    encoding="utf-8-sig",
                )
                write_header = False
                all_results = []
            pct = (i + 1) / total_rivals * 100
            print(f"  [{i+1:>5}/{total_rivals}] ({pct:.1f}%)  最近: {rival_fc}")

    print(f"\n完成！结果保存于：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
