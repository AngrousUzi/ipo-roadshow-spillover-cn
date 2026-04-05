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
  video_duration_sec,
  v1_start_sec, v2_start_sec, v2_end_sec,
  v1_duration_sec, v2_duration_sec,
  detection_source, needs_manual, notes

  * v1_start_sec     — start of 推介致辞 clip (skip intro / 宣传片)
  * v2_start_sec     — start of 答谢致辞 clip (= split boundary)
  * v2_end_sec       — end of 答谢致辞 clip; empty = clip to EOF
  * v1_duration_sec  — duration of 推介致辞 clip (for manual verification)
  * v2_duration_sec  — duration of 答谢致辞 clip (for manual verification)
  * needs_manual     — 1 when either boundary could not be auto-detected

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
    NODE_DIR = DATA_ROOT / "videos" / "全景路演视频节点"
else:
    INDEX_PATH = PROJECT_ROOT / ".." / "IPO_index_selected_platforms.xlsx"
    NODE_DIR = DATA_ROOT / "全景路演视频节点"

NODE_DIR       = DATA_ROOT / "videos" / "全景路演视频节点"
NODE_DIR_CSCOM = DATA_ROOT / "videos" / "中证路演视频节点"

# Transcription directories per platform.
# 全景 uses the de-hallucinated directory only; IR / 中证 have no separate
# 去幻觉 variant so we use their base directory.
TRANS_DIRS: dict[str, list[Path]] = {
    "全景": [DATA_ROOT / "全景路演转录_去幻觉"],
    "IR":   [DATA_ROOT / "IR路演转录_去幻觉"],
    "中证": [DATA_ROOT / "中证路演转录_去幻觉"],
}

OUT_PATH = PROJECT_ROOT / "analyze" / "initialize" / "video_split_windows.csv"

_FIELDNAMES = [
    "index2009", "code", "date", "platform", "video_path",
    "video_duration_sec",
    "v1_start_sec", "v1_end_sec",    # 推介致辞: start → end (= QA start)
    "v2_start_sec", "v2_end_sec",    # 答谢致辞: start → end (None = EOF)
    "v1_duration_sec", "v2_duration_sec",
    "detection_source", "needs_manual", "notes",
]

GAP_THRESHOLD = 300  # seconds; silent Q&A gaps are typically 600-900s


# ── Video duration helper ──────────────────────────────────────────────────────

def _video_duration(video_path: str) -> float | None:
    """Return video duration in seconds via ffprobe, or None on failure."""
    if not video_path:
        return None
    import subprocess, json as _json
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            capture_output=True, text=True, encoding="utf-8", errors="ignore",
        )
        d = _json.loads(r.stdout).get("format", {}).get("duration")
        return round(float(d), 1) if d else None
    except Exception:
        return None


# ── Node-CSV helpers ───────────────────────────────────────────────────────────

def _parse_point(s: str) -> int:
    """Convert 'M:SS', 'MM:SS', 'H:MM:SS', 'HH:MM:SS' to integer seconds."""
    parts = [int(x) for x in s.strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0


def _is_qa_node(title: str) -> bool:
    """True when the node marks the start of the silent Q&A section (v1 end)."""
    return bool(re.search(r"全景交流|网上交流|全景互动|现场答题", title))


def _is_v2_node(title: str) -> bool:
    """True when the node marks the start of 答谢致辞 / closing speech."""
    return bool(re.search(
        r"结束致辞|结束致词|路演结束|总结致辞|总结发言|总结致词|答谢致辞",
        title,
    ))


def _is_v1_node(title: str) -> bool:
    """
    True when the node marks the start of 推介致辞 / presentation speech.

    Matches 全景致辞 / 全景致词 style labels only.
    Explicitly excludes 全景推介 (company IR pitch section within v1) and
    all housekeeping / Q&A labels.
    """
    if _is_v2_node(title):
        return False
    # Exclude housekeeping / Q&A / IR-pitch sections
    for excl in (
        "发行概况", "发行信息", "宣传片", "嘉宾介绍",
        "全景交流", "网上交流", "现场答题", "全景互动",
        "现场报道", "我在现场", "推介",
    ):
        if excl in title:
            return False
    return bool(re.search(r"致辞|致词|发言", title))


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
    Derive v1_start, v1_end, v2_start from a node CSV.

    v1_start : first 全景致辞 / 致辞 / 致词 node (excludes 推介 labels)
    v1_end   : first Q&A node (全景交流/网上交流/全景互动/现场答题)
    v2_start : first 结束致辞/路演结束/总结发言 node
    """
    nodes = _load_node_csv(csv_path)
    if not nodes:
        return {"v1_start_sec": None, "v1_end_sec": None,
                "v2_start_sec": None, "v2_end_sec": None, "notes": "node_csv空"}

    v1_start: float | None = None
    v1_end:   float | None = None
    v2_start: float | None = None
    v1_title = v1_end_title = v2_title = ""

    for n in nodes:
        if v1_start is None and _is_v1_node(n["title"]):
            v1_start = float(n["seconds"])
            v1_title = n["title"]
        if v1_end is None and _is_qa_node(n["title"]):
            v1_end       = float(n["seconds"])
            v1_end_title = n["title"]
        if _is_v2_node(n["title"]):
            v2_start = float(n["seconds"])
            v2_title = n["title"]
            break

    parts: list[str] = []
    if v1_start is not None:
        parts.append(f"v1←node「{v1_title}」@{v1_start:.0f}s")
    if v1_end is not None:
        parts.append(f"v1_end←node「{v1_end_title}」@{v1_end:.0f}s")
    if v2_start is not None:
        parts.append(f"v2←node「{v2_title}」@{v2_start:.0f}s")
    if not parts:
        parts.append("node_csv无匹配")

    return {
        "v1_start_sec": v1_start,
        "v1_end_sec":   v1_end,
        "v2_start_sec": v2_start,
        "v2_end_sec":   None,
        "notes":        "; ".join(parts),
    }


# ── Transcription helpers ──────────────────────────────────────────────────────

# Patterns marking the start of 推介致辞 (v1 start)
_TRANS_V1_RE = re.compile(
    r"推介致辞|推荐致辞|推介之词|推荐之词"
    r"|全景致词环节|全景致辞环节"
    r"|有请.{2,40}(做.{0,4})?(致辞|致词|之词|发言)"
    r"|有请.{5,45}做推[荐介]"
    r"|做推介|做推荐致辞|做推荐致词|做推卸致辞"
)

# Patterns marking the start of silent Q&A (v1 end)
# Exact logic from analyze_quanjing_structure.py detect_sections()
def _is_qa_trans(text: str) -> bool:
    if "网上交流环节" in text and not re.search(r"期待.{0,40}网上交流环节", text):
        return True
    if "往上交流环节" in text:
        return True
    if "现场报道就到这里" in text:
        return True
    if "以下方式" in text and "交流" in text:
        return True
    if re.search(r"进入.{0,15}[网往在]上.{0,5}交流|本次.{0,30}将.{0,15}进行.{0,10}[网往在]上交流", text):
        return True
    return False

# Patterns marking the start of 答谢致辞 (v2 start).
# NOTE: do NOT include pure closing lines ("路演到此结束" etc.) here —
# those appear at the very end and belong only to _TRANS_V2_END_RE.
_TRANS_V2_RE = re.compile(
    r"(接|临)近尾声|答谢致辞|总结发言|路演即将结束|告一段落"
)

# Patterns marking the end of 答谢致辞 (v2 end)
_TRANS_V2_END_RE = re.compile(
    r"[路录]演.{0,15}到此.{0,5}全部结束"
    r"|感谢.{2,10}的(精彩)?发言.{0,60}全部结束"
)


def _find_qa_by_gap(segs: list[dict]) -> float | None:
    """
    Gap-based fallback for silent Q&A detection (from analyze_quanjing_structure.py).
    Returns the end timestamp of the last segment before the largest gap,
    or None if no gap exceeds GAP_THRESHOLD.
    """
    best_gap = 0.0
    best_end: float | None = None
    for i in range(len(segs) - 1):
        gap = segs[i + 1]["start"] - segs[i]["end"]
        if gap > best_gap:
            best_gap = gap
            best_end = float(segs[i]["end"])
    return best_end if best_gap >= GAP_THRESHOLD else None


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
    Derive v1_start, v1_end, v2_start, v2_end from a WhisperX JSON.

    v1_start : 推介致辞 start  (same pattern as analyze_quanjing_structure.py)
    v1_end   : silent Q&A start (keyword detection → gap fallback)
    v2_start : 答谢致辞 start
    v2_end   : explicit closing line end (optional)
    """
    try:
        with open(trans_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"v1_start_sec": None, "v1_end_sec": None,
                "v2_start_sec": None, "v2_end_sec": None,
                "notes": f"转录读取失败:{e}"}

    segs = data.get("segments", [])
    if not segs:
        return {"v1_start_sec": None, "v1_end_sec": None,
                "v2_start_sec": None, "v2_end_sec": None,
                "notes": "转录无segments"}

    v1_start: float | None = None
    v1_end:   float | None = None
    v2_start: float | None = None
    v2_end:   float | None = None
    v1_text = v1_end_text = v2_text = v2_end_text = ""

    for seg in segs:
        text = seg.get("text", "")
        ts   = seg.get("start", 0.0)

        if v1_start is None and _TRANS_V1_RE.search(text):
            v1_start = float(ts)
            v1_text  = text[:60]

        if v1_end is None and _is_qa_trans(text):
            v1_end      = float(ts)
            v1_end_text = text[:60]

        if v2_start is None and _TRANS_V2_RE.search(text):
            v2_start = float(ts)
            v2_text  = text[:60]

    # Gap-based fallback for v1_end when keyword detection fails
    if v1_end is None:
        gap_end = _find_qa_by_gap(segs)
        if gap_end is not None:
            v1_end      = gap_end
            v1_end_text = f"gap>{GAP_THRESHOLD}s"

    # Explicit end of 答谢致辞
    if v2_start is not None:
        for seg in segs:
            if seg.get("start", 0) >= v2_start and _TRANS_V2_END_RE.search(seg.get("text", "")):
                v2_end      = float(seg["end"])
                v2_end_text = seg["text"][:40]
                break

    parts: list[str] = []
    if v1_start is not None:
        parts.append(f"v1←trans@{v1_start:.0f}s「{v1_text}」")
    if v1_end is not None:
        parts.append(f"v1_end←trans@{v1_end:.0f}s「{v1_end_text}」")
    if v2_start is not None:
        parts.append(f"v2←trans@{v2_start:.0f}s「{v2_text}」")
    if v2_end is not None:
        parts.append(f"v2_end←trans@{v2_end:.0f}s「{v2_end_text}」")
    if not parts:
        parts.append("转录无匹配")

    return {
        "v1_start_sec": v1_start,
        "v1_end_sec":   v1_end,
        "v2_start_sec": v2_start,
        "v2_end_sec":   v2_end,
        "notes":        "; ".join(parts),
    }


# ── Node CSV file lookup ───────────────────────────────────────────────────────

def find_node_csv(code: str, date: str, platform: str = "全景") -> Path | None:
    """
    Find a node CSV by stock code and date.
    For 全景: looks in NODE_DIR        → pattern ``{code}_*_{date}.csv``
    For 中证: looks in NODE_DIR_CSCOM  → pattern ``{code}_*_{date}_*.csv``
    Returns None if absent.
    """
    node_dir = NODE_DIR_CSCOM if platform == "中证" else NODE_DIR
    if not node_dir.exists():
        return None
    for p in node_dir.glob(f"{code}_*_{date}*.csv"):
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
        "v1_end_sec":       None,
        "v2_start_sec":     None,
        "v2_end_sec":       None,
        "detection_source": "missing",
        "notes":            "",
    }

    note_parts: list[str] = []

    # ── Node CSV (全景 and 中证): takes full priority when present ──────────
    if platform in ("全景", "中证"):
        node_csv = find_node_csv(code, date, platform)
        if node_csv is not None:
            nd = detect_from_nodes(node_csv)
            note_parts.append(f"[节点]{nd['notes']}")
            result["v1_start_sec"]     = nd["v1_start_sec"]
            result["v1_end_sec"]       = nd["v1_end_sec"]
            result["v2_start_sec"]     = nd["v2_start_sec"]
            result["v2_end_sec"]       = nd["v2_end_sec"]
            result["detection_source"] = "node_csv"
            result["notes"] = " | ".join(note_parts)
            return result
        note_parts.append("[节点]无节点CSV")

    # ── Transcription (全景/中证 without node CSV, IR) ────────────────────
    trans_path = find_trans_json(code, date, platform, index2009)
    if trans_path is not None:
        td = detect_from_trans(trans_path)
        note_parts.append(f"[转录]{td['notes']}")
        result["v1_start_sec"]     = td["v1_start_sec"]
        result["v1_end_sec"]       = td["v1_end_sec"]
        result["v2_start_sec"]     = td["v2_start_sec"]
        result["v2_end_sec"]       = td["v2_end_sec"]
        result["detection_source"] = "transcription"
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

        v1   = det["v1_start_sec"]  # float or None
        v1e  = det["v1_end_sec"]    # float or None  (QA start = v1 clip end)
        v2   = det["v2_start_sec"]  # float or None
        ve   = det["v2_end_sec"]    # float or None

        # ── Video duration (for open-ended clip length calculation) ──────
        vid_dur = _video_duration(video_path)

        # ── Derived durations ────────────────────────────────────────────
        # v1 clip: [v1_start, v1_end]
        v1_eff_end = v1e if v1e is not None else vid_dur
        v1_dur = round(v1_eff_end - (v1 or 0.0), 1) if v1_eff_end is not None else ""

        # v2 clip: [v2_start, v2_end or EOF]
        if v2 is not None:
            v2_end_eff = ve if ve is not None else vid_dur
            v2_dur = round(v2_end_eff - v2, 1) if v2_end_eff is not None else ""
        else:
            v2_dur = ""

        needs_manual = int(v1 is None or v1e is None or v2 is None)

        rows.append({
            "index2009":          index2009,
            "code":               code,
            "date":               date,
            "platform":           platform,
            "video_path":         video_path,
            "video_duration_sec": vid_dur if vid_dur is not None else "",
            "v1_start_sec":       "" if v1  is None else v1,
            "v1_end_sec":         "" if v1e is None else v1e,
            "v2_start_sec":       "" if v2  is None else v2,
            "v2_end_sec":         "" if ve  is None else ve,
            "v1_duration_sec":    v1_dur,
            "v2_duration_sec":    v2_dur,
            "detection_source":   det["detection_source"],
            "needs_manual":       needs_manual,
            "notes":              det["notes"],
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
