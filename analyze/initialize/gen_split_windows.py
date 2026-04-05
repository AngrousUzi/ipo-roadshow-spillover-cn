"""
analyze/initialize/gen_split_windows.py
=========================================
Detect video split windows (推介致辞 / 答谢致辞 boundaries) for every
roadshow that requires transcript-based splitting (platform: 全景, IR,
中证 with needs_split=True).

Data sources tried in order per roadshow
-----------------------------------------
1. Node CSV  (全景 only) — ``videos/全景路演视频节点/{code}_*_{date}.csv``
2. Transcription JSON    — ``路演转录_去幻觉 / IR路演转录 / 中证路演转录``

Output
------
``tmp/video_split_windows.csv`` — one row per roadshow to be split.

Columns
-------
  index2009, code, date, platform, video_path,
  v1_start_sec, v2_start_sec, v2_end_sec,
  detection_source, needs_manual, notes

  * v1_start_sec  — start of 推介致辞 clip (skip intro / 宣传片)
  * v2_start_sec  — start of 答谢致辞 clip (= split boundary)
  * v2_end_sec    — end of 答谢致辞 clip; empty = clip to EOF
  * needs_manual  — 1 when either boundary could not be auto-detected

Review & edit the CSV, then run the full initialize pipeline to apply cuts.

Usage
-----
  python -m analyze.initialize.gen_split_windows   # from PROJECT_ROOT
  python analyze/initialize/gen_split_windows.py   # equivalent
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Make analyze/ importable when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT, DATA_ROOT, get_video_dir

# ── Paths ──────────────────────────────────────────────────────────────────────

import os

if os.name == "nt":
    INDEX_PATH = PROJECT_ROOT / "anns" / "IPO_index_selected_platforms.xlsx"
else:
    INDEX_PATH = PROJECT_ROOT / ".." / "IPO_index_selected_platforms.xlsx"

NODE_DIR = DATA_ROOT / "videos" / "全景路演视频节点"

# Transcription directories per platform.
# 全景 uses the de-hallucinated directory only; IR / 中证 have no separate
# 去幻觉 variant so we use their base directory.
TRANS_DIRS: dict[str, list[Path]] = {
    "全景": [DATA_ROOT / "路演转录_去幻觉"],
    "IR":   [DATA_ROOT / "IR路演转录_去幻觉"],
    "中证": [DATA_ROOT / "中证路演转录_去幻觉"],
}

OUT_PATH = PROJECT_ROOT / "tmp" / "video_split_windows.csv"

_FIELDNAMES = [
    "index2009", "code", "date", "platform", "video_path",
    "v1_start_sec", "v2_start_sec", "v2_end_sec",
    "detection_source", "needs_manual", "notes",
]


# ── Node-CSV helpers ───────────────────────────────────────────────────────────

def _parse_point(s: str) -> int:
    """Convert 'M:SS', 'MM:SS', 'H:MM:SS', 'HH:MM:SS' to integer seconds."""
    parts = [int(x) for x in s.strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0


def _is_v2_node(title: str) -> bool:
    """True when the node marks the start of 答谢致辞 / closing speech."""
    return bool(re.search(
        r"结束致辞|结束致词|路演结束|总结致辞|总结发言|总结致词|答谢致辞",
        title,
    ))


def _is_v1_node(title: str) -> bool:
    """True when the node marks the start of 推介致辞 / presentation speech."""
    if _is_v2_node(title):
        return False
    # Exclude housekeeping / Q&A sections that do not contain speech
    for excl in (
        "发行概况", "发行信息", "宣传片", "嘉宾介绍",
        "全景交流", "网上交流", "现场答题", "全景互动",
        "现场报道", "我在现场",
    ):
        if excl in title:
            return False
    return bool(re.search(r"致辞|致词|推介|发言", title))


def _load_node_csv(csv_path: Path) -> list[dict]:
    """
    Parse a 全景路演视频节点 CSV.

    Handles two formats:
    - 3-col  ``orderNo, point, title`` (older files, no seconds column)
    - 12-col ``路演ID, 公司代码, …, orderNo, point, seconds, title, …``

    Returns a list of ``{'seconds': int, 'title': str}`` sorted by seconds.
    """
    rows: list[dict] = []
    try:
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = row.get("title", "").strip()
                if not title:
                    continue
                sec_raw = row.get("seconds", "").strip()
                if sec_raw.isdigit():
                    sec = int(sec_raw)
                else:
                    pt = row.get("point", "").strip()
                    if not pt:
                        continue
                    sec = _parse_point(pt)
                rows.append({"seconds": sec, "title": title})
    except Exception as e:
        print(f"[WARN] node CSV 读取失败: {csv_path} — {e}")
    return sorted(rows, key=lambda r: r["seconds"])


def detect_from_nodes(csv_path: Path) -> dict:
    """
    Derive v1_start and v2_start from a node CSV.

    Returns a dict with keys:
      v1_start_sec, v2_start_sec, v2_end_sec (always None from nodes),
      notes (str).
    """
    nodes = _load_node_csv(csv_path)
    if not nodes:
        return {"v1_start_sec": None, "v2_start_sec": None,
                "v2_end_sec": None, "notes": "node_csv空"}

    v1_start: float | None = None
    v2_start: float | None = None
    v1_title = v2_title = ""

    for n in nodes:
        if v1_start is None and _is_v1_node(n["title"]):
            v1_start = float(n["seconds"])
            v1_title = n["title"]
        if _is_v2_node(n["title"]):
            v2_start = float(n["seconds"])
            v2_title = n["title"]
            break   # first 答谢 marker wins

    parts: list[str] = []
    if v1_start is not None:
        parts.append(f"v1←node「{v1_title}」@{v1_start:.0f}s")
    if v2_start is not None:
        parts.append(f"v2←node「{v2_title}」@{v2_start:.0f}s")
    if not parts:
        parts.append("node_csv无匹配")

    return {
        "v1_start_sec": v1_start,
        "v2_start_sec": v2_start,
        "v2_end_sec":   None,
        "notes":        "; ".join(parts),
    }


# ── Transcription helpers ──────────────────────────────────────────────────────

# Patterns marking the start of 推介致辞 (presentation)
_TRANS_V1_RE = re.compile(
    r"推介致辞|推荐致辞|推介之词|推荐之词"
    r"|全景致词环节|全景致辞环节"
    r"|有请.{2,40}(做.{0,4})?(致辞|致词|之词|发言)"
    r"|有请.{5,45}做推[荐介]"
    r"|做推介|做推荐致辞|做推荐致词|做推卸致辞"
)

# Patterns marking the start of 答谢致辞 (closing thanks)
_TRANS_V2_RE = re.compile(
    r"(接|临)近尾声|答谢致辞|总结发言|路演即将结束|告一段落"
    r"|路演.{0,15}到此.{0,5}全部结束"
    r"|感谢.{2,10}的(精彩)?发言.{0,60}全部结束"
)

# Patterns marking the end of 答谢致辞
_TRANS_V2_END_RE = re.compile(
    r"[路录]演.{0,15}到此.{0,5}全部结束"
    r"|感谢.{2,10}的(精彩)?发言.{0,60}全部结束"
)

# Minimum position (fraction of total) where v2 is allowed to start.
# Set to 0.0 to match analyze_quanjing_structure.py (no position filter).
# A small positive value (e.g. 0.30) reduces false positives from early
# host phrases that echo 答谢致辞 keywords.
_V2_SEARCH_FROM = 0.0


def find_trans_json(
    code: str, date: str, platform: str, index2009: str = ""
) -> Path | None:
    """
    Search transcription directories for a matching JSON file.

    For 全景 files are named ``{index2009}_{code}_{date}.json``.
    For IR / 中证 they follow ``{code}_{company}_{date}.json``.
    """
    dirs = TRANS_DIRS.get(platform, [])
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob(f"*{code}*{date}*.json")):
            # Skip per-segment transcriptions for 中证
            if platform == "中证" and re.search(r"_视频\d+[_.]", p.name):
                continue
            return p
    return None


def detect_from_trans(trans_path: Path) -> dict:
    """
    Derive v1_start, v2_start, v2_end from a WhisperX transcription JSON.

    Uses the same segment-scanning logic as analyze_quanjing_structure.py
    but returns timestamps rather than segment indices.
    """
    try:
        with open(trans_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"v1_start_sec": None, "v2_start_sec": None,
                "v2_end_sec": None, "notes": f"转录读取失败:{e}"}

    segs = data.get("segments", [])
    if not segs:
        return {"v1_start_sec": None, "v2_start_sec": None,
                "v2_end_sec": None, "notes": "转录无segments"}

    total = segs[-1].get("end", 0)
    v2_search_from = total * _V2_SEARCH_FROM

    v1_start: float | None = None
    v2_start: float | None = None
    v2_end:   float | None = None
    v1_text = v2_text = v2_end_text = ""

    for seg in segs:
        text = seg.get("text", "")
        ts   = seg.get("start", 0.0)
        te   = seg.get("end",   0.0)

        if v1_start is None and _TRANS_V1_RE.search(text):
            v1_start = float(ts)
            v1_text  = text[:60]

        if v2_start is None and ts >= v2_search_from and _TRANS_V2_RE.search(text):
            v2_start = float(ts)
            v2_text  = text[:60]

    # Scan forward from v2_start to find explicit closing line
    if v2_start is not None:
        for seg in segs:
            if seg.get("start", 0) >= v2_start and _TRANS_V2_END_RE.search(seg.get("text", "")):
                v2_end     = float(seg["end"])
                v2_end_text = seg["text"][:40]
                break

    parts: list[str] = []
    if v1_start is not None:
        parts.append(f"v1←trans@{v1_start:.0f}s「{v1_text}」")
    if v2_start is not None:
        parts.append(f"v2←trans@{v2_start:.0f}s「{v2_text}」")
    if v2_end is not None:
        parts.append(f"v2_end←trans@{v2_end:.0f}s「{v2_end_text}」")
    if not parts:
        parts.append("转录无匹配")

    return {
        "v1_start_sec": v1_start,
        "v2_start_sec": v2_start,
        "v2_end_sec":   v2_end,
        "notes":        "; ".join(parts),
    }


# ── Node CSV file lookup ───────────────────────────────────────────────────────

def find_node_csv(code: str, date: str) -> Path | None:
    """Find 全景 node CSV by stock code and date. Returns None if absent."""
    if not NODE_DIR.exists():
        return None
    for p in NODE_DIR.glob(f"{code}_*_{date}.csv"):
        return p
    return None


# ── Per-roadshow detection ─────────────────────────────────────────────────────

def _detect_one(
    index2009: str, code: str, date: str, platform: str
) -> dict:
    """
    Return detection result dict for one roadshow.
    Keys: v1_start_sec, v2_start_sec, v2_end_sec, detection_source, notes.
    """
    result: dict = {
        "v1_start_sec":     None,
        "v2_start_sec":     None,
        "v2_end_sec":       None,
        "detection_source": "missing",
        "notes":            "",
    }

    note_parts: list[str] = []

    # ── Step 1: node CSV for 全景 ──────────────────────────────────────────
    if platform == "全景":
        node_csv = find_node_csv(code, date)
        if node_csv is not None:
            nd = detect_from_nodes(node_csv)
            note_parts.append(f"[节点]{nd['notes']}")
            result["v1_start_sec"] = nd["v1_start_sec"]
            result["v2_start_sec"] = nd["v2_start_sec"]
            if nd["v1_start_sec"] is not None or nd["v2_start_sec"] is not None:
                result["detection_source"] = "node_csv"
        else:
            note_parts.append("[节点]无节点CSV")

    # ── Step 2: transcription (full fallback or v2 supplement) ────────────
    need_v1 = result["v1_start_sec"] is None
    need_v2 = result["v2_start_sec"] is None

    if need_v1 or need_v2:
        trans_path = find_trans_json(code, date, platform, index2009)
        if trans_path is not None:
            td = detect_from_trans(trans_path)
            note_parts.append(f"[转录]{td['notes']}")
            if need_v1 and td["v1_start_sec"] is not None:
                result["v1_start_sec"] = td["v1_start_sec"]
            if need_v2 and td["v2_start_sec"] is not None:
                result["v2_start_sec"] = td["v2_start_sec"]
                result["v2_end_sec"]   = td["v2_end_sec"]
            # Update source label
            src = result["detection_source"]
            has_trans = (
                td["v1_start_sec"] is not None or td["v2_start_sec"] is not None
            )
            if has_trans:
                result["detection_source"] = (
                    "node_csv+transcription" if src == "node_csv" else "transcription"
                )
        else:
            note_parts.append("[转录]无转录文件")

    result["notes"] = " | ".join(note_parts)
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def generate() -> Path:
    """
    Build the split-windows table and write it to ``OUT_PATH``.

    Only roadshows that require splitting are included:
    - platform in ("全景", "IR")
    - platform == "中证" AND cscom_plan.needs_split == True

    Returns the output Path.
    """
    # Lazy import to avoid circular dep when used as a module
    from .cscom_plan import load_cscom_plans

    df_index = pd.read_excel(INDEX_PATH, dtype=str)
    df_index = df_index[df_index["采用视频平台"].notna()].copy()
    cscom_plans = load_cscom_plans()

    rows: list[dict] = []

    for _, row in df_index.iterrows():
        platform  = str(row["采用视频平台"]).strip()
        index2009 = str(row.get("INDEX2009", "")).strip()
        code      = str(row.get(f"{platform}_去重代码", "")).strip()
        date      = str(row.get(f"{platform}_日期",     "")).strip()

        # ── Filter: only roadshows that need transcript-based splitting ──
        if platform in ("上证", "中国证券网"):
            continue
        if platform == "中证":
            cp = cscom_plans.get(code, {})
            if not cp.get("needs_split"):
                continue

        # ── Locate source video ──────────────────────────────────────────
        video_dir = get_video_dir(platform)
        candidates = [
            vf for vf in video_dir.glob(f"{code}_*_{date}*.mp4")
            if "宣传片" not in vf.name
        ]
        video_path = str(candidates[0]) if candidates else ""
        if not video_path:
            print(f"[WARN] 未找到视频: index={index2009} code={code} date={date} platform={platform}")

        # ── Detect boundaries ────────────────────────────────────────────
        det = _detect_one(index2009, code, date, platform)

        needs_manual = int(
            det["v1_start_sec"] is None or det["v2_start_sec"] is None
        )

        rows.append({
            "index2009":        index2009,
            "code":             code,
            "date":             date,
            "platform":         platform,
            "video_path":       video_path,
            "v1_start_sec":     "" if det["v1_start_sec"] is None else det["v1_start_sec"],
            "v2_start_sec":     "" if det["v2_start_sec"] is None else det["v2_start_sec"],
            "v2_end_sec":       "" if det["v2_end_sec"]   is None else det["v2_end_sec"],
            "detection_source": det["detection_source"],
            "needs_manual":     needs_manual,
            "notes":            det["notes"],
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    ok    = sum(1 for r in rows if not r["needs_manual"])
    print(f"写入 {total} 条记录 → {OUT_PATH}")
    print(f"  自动识别完整: {ok}/{total}")
    print(f"  需人工审核:   {total - ok}/{total}")
    src_ctr: dict[str, int] = {}
    for r in rows:
        src_ctr[r["detection_source"]] = src_ctr.get(r["detection_source"], 0) + 1
    for src, cnt in sorted(src_ctr.items()):
        print(f"    {src}: {cnt}")
    return OUT_PATH


if __name__ == "__main__":
    generate()
