"""
analyze/collect_qa.py
======================
Aggregate per-platform Q&A CSVs into one unified CSV per roadshow.

Source  : {platform}路演问答/{code}_{name}_{date}.csv  (5 platforms)
Output  : 路演问答/{index2009}_{code}_{date}.csv

Unified schema (one row per Q&A record or speech):
    index2009, platform, source_file,
    序号, 消息类型, 提问人, 提问内容, 提问时间,
    回答人, 回答内容, 回答时间, 路演ID

Platform-specific normalization:
    全景      all rows → 消息类型=问答; A fields from 回复* columns
    上证      drop 消息类型=1 (duplicate question-only rows);
              type=2 → 问答 (提问内容+内容); type=3/4 → 发言 (内容 → 提问内容)
    中证      same structure as 全景 (回复* columns)
    中国证券网 already clean 问答/发言; no time fields available
    IR        消息类型 "提问"→问答 "发言"→发言; 提问人 ← 用户名

断点续算：跳过已存在的输出文件。
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROJECT_ROOT, INDEX_QA_DIR, get_qa_dir

# ─── Index path (mirrors orchestrator.py) ─────────────────────────────────────

if os.name == "nt":
    INDEX_PATH = PROJECT_ROOT / "anns" / "IPO_index_selected_platforms.xlsx"
else:
    INDEX_PATH = PROJECT_ROOT / ".." / "IPO_index_selected_platforms.xlsx"

OUTPUT_DIR = INDEX_QA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Unified output columns ────────────────────────────────────────────────────

UNIFIED_COLUMNS = [
    "index2009",
    "platform",
    "source_file",
    "序号",
    "消息类型",
    "提问人",
    "questioner_id",   # reliable per-person identifier (UUID for 全景/IR; pseudonym for others)
    "提问内容",
    "提问时间",
    "回答人",
    "回答内容",
    "回答时间",
    "路演ID",
]


# ─── Helper ───────────────────────────────────────────────────────────────────

def _s(val) -> str:
    """Return stripped string; treat None / NaN as empty string."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    return str(val).strip()


# ─── Per-platform normalizers ──────────────────────────────────────────────────

def _normalize_quanjing(df: pd.DataFrame) -> pd.DataFrame:
    """
    全景路演问答
    Each row is a question with an optional reply already joined in the same row.
    No speech type — all rows are treated as 问答.
    """
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "序号":         _s(row.get("序号")),
            "消息类型":      "问答",
            "提问人":        _s(row.get("提问人")),
            "questioner_id": _s(row.get("用户ID")),   # UUID; unique even for 匿名用户 display names
            "提问内容":      _s(row.get("提问内容")),
            "提问时间":      _s(row.get("提问时间")),
            "回答人":        _s(row.get("回复人")),
            "回答内容":      _s(row.get("回复内容")),
            "回答时间":      _s(row.get("回复时间")),
            "路演ID":        _s(row.get("路演ID")),
        })
    return pd.DataFrame(rows, columns=list(rows[0].keys()) if rows else [])


def _normalize_sseinfo(df: pd.DataFrame) -> pd.DataFrame:
    """
    上证路演问答
    消息类型=1  原始提问（仅问题，type=2 的重复）→ 丢弃
    消息类型=2  问答对 → 问答  (提问内容 + 内容=答复)
    消息类型=3  嘉宾发言 → 发言
    消息类型=4  主持人发言 → 发言
    """
    rows = []
    for _, row in df.iterrows():
        msg_type = _s(row.get("消息类型"))
        if msg_type == "1":
            continue  # duplicate question-only entry, drop

        if msg_type == "2":
            rows.append({
                "序号":         _s(row.get("记录ID")),
                "消息类型":      "问答",
                "提问人":        _s(row.get("提问人")),
                "questioner_id": _s(row.get("提问人")),  # 用户ID is always NaN; pseudonym is stable
                "提问内容":      _s(row.get("提问内容")),
                "提问时间":      _s(row.get("提问时间")),
                "回答人":        _s(row.get("嘉宾姓名")),
                "回答内容":      _s(row.get("内容")),
                "回答时间":      _s(row.get("内容创建时间")),
                "路演ID":        _s(row.get("路演ID")),
            })
        else:  # type=3 (嘉宾发言) / type=4 (主持人发言)
            rows.append({
                "序号":         _s(row.get("记录ID")),
                "消息类型":      "发言",
                "提问人":        _s(row.get("提问人")),
                "questioner_id": "",
                "提问内容":      _s(row.get("内容")),
                "提问时间":      _s(row.get("内容创建时间")),
                "回答人":        "",
                "回答内容":      "",
                "回答时间":      "",
                "路演ID":        _s(row.get("路演ID")),
            })
    return pd.DataFrame(rows, columns=list(rows[0].keys()) if rows else [])


def _normalize_cscom(df: pd.DataFrame) -> pd.DataFrame:
    """
    中证路演问答
    Same denormalized Q+A structure as 全景 (回复* columns hold the answer).
    """
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "序号":         _s(row.get("序号")),
            "消息类型":      "问答",
            "提问人":        _s(row.get("提问人")),
            "questioner_id": _s(row.get("提问人")),  # 参考XXXXX style anonymous IDs
            "提问内容":      _s(row.get("提问内容")),
            "提问时间":      _s(row.get("提问时间")),
            "回答人":        _s(row.get("回复人")),
            "回答内容":      _s(row.get("回复内容")),
            "回答时间":      _s(row.get("回复时间")),
            "路演ID":        _s(row.get("路演ID")),
        })
    return pd.DataFrame(rows, columns=list(rows[0].keys()) if rows else [])


def _normalize_cnstock(df: pd.DataFrame) -> pd.DataFrame:
    """
    中国证券网路演问答
    Already clean 问答/发言 structure; no time fields in the source CSV.
    """
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "序号":         _s(row.get("序号")),
            "消息类型":      _s(row.get("消息类型")),
            "提问人":        _s(row.get("提问人")),
            "questioner_id": _s(row.get("提问人")),  # 游客XXXXX style anonymous IDs
            "提问内容":      _s(row.get("提问内容")),
            "提问时间":      "",
            "回答人":        _s(row.get("回答人")),
            "回答内容":      _s(row.get("回答内容")),
            "回答时间":      "",
            "路演ID":        _s(row.get("路演ID")),
        })
    return pd.DataFrame(rows, columns=list(rows[0].keys()) if rows else [])


def _normalize_ir(df: pd.DataFrame) -> pd.DataFrame:
    """
    IR路演问答
    消息类型: "提问" → 问答, "发言" → 发言
    提问人 ← 用户名 (嘉宾/主持人的 username field)
    """
    rows = []
    for _, row in df.iterrows():
        src_type    = _s(row.get("消息类型"))
        unified_type = "问答" if src_type == "提问" else "发言"
        rows.append({
            "序号":         _s(row.get("序号")),
            "消息类型":      unified_type,
            "提问人":        _s(row.get("用户名")),
            "questioner_id": _s(row.get("用户ID")),   # numeric user ID; more stable than masked phone
            "提问内容":      _s(row.get("内容")),
            "提问时间":      _s(row.get("发布时间")),
            "回答人":        _s(row.get("回答用户名")),
            "回答内容":      _s(row.get("回答内容")),
            "回答时间":      _s(row.get("回答发布时间")),
            "路演ID":        _s(row.get("路演ID")),
        })
    return pd.DataFrame(rows, columns=list(rows[0].keys()) if rows else [])


NORMALIZERS = {
    "全景":       _normalize_quanjing,
    "上证":       _normalize_sseinfo,
    "中证":       _normalize_cscom,
    "中国证券网":  _normalize_cnstock,
    "IR":         _normalize_ir,
}


# ─── Source file lookup ────────────────────────────────────────────────────────

def find_qa_csv(platform: str, code: str, date: str) -> Path | None:
    """
    Glob {platform}路演问答/{code}_*_{date}.csv and return the first match.
    Warns if multiple files match (shouldn't happen in practice).
    """
    qa_dir = get_qa_dir(platform)
    if not qa_dir.exists():
        return None
    matches = sorted(qa_dir.glob(f"{code}_*_{date}.csv"))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[WARN] 多个匹配文件，取第一个: {[m.name for m in matches]}")
    return matches[0]


# ─── Per-roadshow processor ────────────────────────────────────────────────────

def process_one(index2009: str, platform: str, code: str, date: str) -> tuple[bool, str]:
    """
    Find, normalize, and write one roadshow's unified Q&A CSV.
    Returns (success, message).
    """
    output_path = OUTPUT_DIR / f"{index2009}_{code}_{date}.csv"
    if output_path.exists():
        return True, "already done"

    src = find_qa_csv(platform, code, date)
    if src is None:
        return False, f"未找到源文件 ({get_qa_dir(platform).name}/)"

    normalizer = NORMALIZERS.get(platform)
    if normalizer is None:
        return False, f"未知平台 '{platform}'"

    try:
        df_raw = pd.read_csv(src, dtype=str, encoding="utf-8-sig")
    except Exception as e:
        return False, f"读取失败: {e}"

    if df_raw.empty:
        return False, "源文件为空"

    try:
        df_norm = normalizer(df_raw)
    except Exception as e:
        return False, f"标准化失败: {e}"

    if df_norm.empty:
        return False, "标准化后无记录"

    n_before = len(df_norm)

    # ── Deduplication ────────────────────────────────────────────────────────
    # 问答: same (questioner_id, 提问内容) → keep row with longest 回答内容
    #       (handles both exact duplicates and multiple answer versions)
    # 发言: drop exact 提问内容 duplicates
    qa_mask = df_norm["消息类型"] == "问答"
    df_qa = df_norm[qa_mask].copy()
    df_sp = df_norm[~qa_mask].copy()

    if not df_qa.empty:
        # Keep only answered pairs; sort descending by answer length so
        # dedup retains the longest answer when the same question appears twice
        df_qa["_a_len"] = df_qa["回答内容"].fillna("").str.len()
        df_qa = df_qa[df_qa["_a_len"] > 0].copy()   # drop unanswered
        id_col = "questioner_id" if "questioner_id" in df_qa.columns else "提问人"
        df_qa = (df_qa
                 .sort_values("_a_len", ascending=False)
                 .drop_duplicates(subset=[id_col, "提问内容"], keep="first")
                 .drop(columns=["_a_len"])
                 .reset_index(drop=True))

    if not df_sp.empty:
        df_sp = (df_sp
                 .drop_duplicates(subset=["提问内容"], keep="first")
                 .reset_index(drop=True))

    df_norm = pd.concat([df_qa, df_sp], ignore_index=True)
    n_dropped = n_before - len(df_norm)
    if n_dropped > 0:
        print(f"  [CLEAN] {src.name}: {n_before} → {len(df_norm)} rows "
              f"(dropped {n_dropped} unanswered/duplicate)")

    df_norm.insert(0, "index2009",   index2009)
    df_norm.insert(1, "platform",    platform)
    df_norm.insert(2, "source_file", src.name)

    df_out = df_norm.reindex(columns=UNIFIED_COLUMNS)
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    return True, f"{len(df_out)} 条 → {output_path.name}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("  QA Collect")
    print("═" * 60)
    print(f"IPO 索引: {INDEX_PATH}")
    print(f"输出目录: {OUTPUT_DIR}\n")

    df_index = pd.read_excel(INDEX_PATH, dtype=str)
    df_index = df_index[df_index["采用视频平台"].notna()].copy()
    print(f"路演总数: {len(df_index)}\n")

    n_ok = n_skip = n_fail = 0

    for _, row in df_index.iterrows():
        platform  = str(row["采用视频平台"]).strip()
        index2009 = str(row.get("INDEX2009", "")).strip()
        code      = str(row.get(f"{platform}_去重代码", "")).strip()
        date      = str(row.get(f"{platform}_日期",     "")).strip()

        if not all([platform, index2009, code, date]):
            print(f"[SKIP] 缺少必要字段  index={index2009} platform={platform}")
            n_skip += 1
            continue

        ok, msg = process_one(index2009, platform, code, date)

        if ok:
            if msg == "already done":
                n_skip += 1
            else:
                print(f"[OK]   {index2009}_{code}_{date}  {msg}")
                n_ok += 1
        else:
            print(f"[FAIL] {index2009}_{code}_{date}  {msg}")
            n_fail += 1

    print()
    print(f"完成: 成功={n_ok}  跳过={n_skip}  失败={n_fail}")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
