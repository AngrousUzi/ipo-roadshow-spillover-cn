"""
analyze/initialize/orchestrator.py
=====================================
Multiprocessing worker and main collection loop.

Each roadshow produces two output videos:
    路演视频/{index2009}_{code}_{date}_推介.mp4   — 推介致辞
    路演视频/{index2009}_{code}_{date}_答谢.mp4   — 答谢致辞

Worker input  : a serialisable dict describing one roadshow
Worker output : VideoTaskResult (as a plain dict for IPC)

Public API
----------
process_video_row(row_data)  → dict      (multiprocessing worker)
collect_video_tasks(...)     → list[dict]
collect_audio_tasks()        → list[tuple[str, Path | None]]
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

# Make analyze/ importable from inside the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_audio_dir, get_video_dir, get_trans_dir, PROJECT_ROOT

from .ffmpeg_utils import (
    check_video_quality,
    clip_video,
    concat_videos_reencode,
    cut_video,
)
from .platform_handlers import VideoSplitPlan, load_split_windows, resolve_split_plan
from .cscom_plan import load_cscom_plans

# ── Paths ──────────────────────────────────────────────────────────────────────

VIDEO_OUTPUT_DIR = PROJECT_ROOT / "路演视频"
AUDIO_OUTPUT_DIR = PROJECT_ROOT / "路演音频"
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if os.name == "nt":
    INDEX_PATH = PROJECT_ROOT / "anns" / "IPO_index_selected_platforms.xlsx"
else:
    INDEX_PATH = PROJECT_ROOT / ".." / "IPO_index_selected_platforms.xlsx"

PARALLEL_PROCESSES = int(os.getenv("SLURM_CPUS_PER_TASK", "16"))
FFMPEG_THREADS     = max(1, PARALLEL_PROCESSES // max(1, PARALLEL_PROCESSES))


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class VideoTaskResult:
    index2009:    str
    v1_path:      str  # empty string means failed / missing
    v2_path:      str
    v1_success:   bool
    v2_success:   bool
    v1_error:     str
    v2_error:     str
    v1_quality:   dict = field(default_factory=dict)
    v2_quality:   dict = field(default_factory=dict)
    v1_start_sec: float        = 0.0   # actual start used for 推介 clip
    split_time:   float | None = None  # actual split boundary used
    v2_end_sec:   float | None = None  # actual end used for 答谢 clip


# ── Internal helpers ───────────────────────────────────────────────────────────

def _output_paths(index2009: str, code: str, date: str) -> tuple[Path, Path]:
    stem = f"{index2009}_{code}_{date}"
    return (
        VIDEO_OUTPUT_DIR / f"{stem}_推介.mp4",
        VIDEO_OUTPUT_DIR / f"{stem}_答谢.mp4",
    )


def _produce_from_sources(
    sources: list[Path],
    output_path: Path,
    full_decode: bool,
) -> tuple[bool, str, dict]:
    """
    Produce a single output file from one or more source paths.
    Returns (success, error_str, quality_dict).
    """
    if not sources:
        return False, "无源文件", {}

    # Check for already-complete output
    if output_path.exists():
        q = check_video_quality(output_path, full_decode=full_decode)
        if not q["integrity_error"]:
            return True, "", q
        print(f"[WARN] 已有输出文件损坏，重新生成: {output_path.name}")
        output_path.unlink(missing_ok=True)

    missing = [p for p in sources if not p.exists()]
    if missing:
        return False, f"源文件不存在: {[p.name for p in missing]}", {}

    if len(sources) == 1:
        # Single source: stream-copy (fast)
        _, err = clip_video(sources[0], output_path)
        if output_path.exists():
            q = check_video_quality(output_path, full_decode=full_decode)
            return q["integrity_error"] == "", err, q
        return False, err, {}
    else:
        # Multiple sources: re-encode concat
        ok, err = concat_videos_reencode(
            sources, output_path, threads=FFMPEG_THREADS
        )
        if ok:
            q = check_video_quality(output_path, full_decode=full_decode)
            return True, err, q
        return False, err, {}


def _produce_from_split(
    plan: VideoSplitPlan,
    v1_output: Path,
    v2_output: Path,
    full_decode: bool,
) -> tuple[bool, str, dict, bool, str, dict, float, float | None, float | None]:
    """
    Cut *plan.full_source* into two clips at a validated (or auto-detected)
    boundary.

    Returns
    -------
    (v1_ok, v1_err, v1_q, v2_ok, v2_err, v2_q,
     v1_start_sec, split_time, v2_end_sec)
    """
    assert plan.full_source is not None

    # ── Resolve split boundary ─────────────────────────────────────────────
    split_time   = plan.split_time   # v2_start_sec from table; None = no 答谢致辞
    v1_start_sec = plan.v1_start_sec # v1_start_sec from table; default 0.0
    v2_end_sec   = plan.v2_end_sec   # v2_end_sec from table; None = EOF

    # Determine what needs to be produced.
    # v1 is meaningful when its start is non-zero OR there is a split boundary.
    # v2 only exists when split_time is set.
    has_v1 = v1_start_sec > 0 or split_time is not None
    has_v2 = split_time is not None

    if not has_v1:
        msg = (
            f"index={plan.index2009} code={plan.code}: "
            f"split_windows 中无有效切分点，跳过"
        )
        print(f"[SKIP] {msg}")
        return False, msg, {}, False, msg, {}, v1_start_sec, None, None

    # ── Check existing outputs ─────────────────────────────────────────────
    def _check_existing(p: Path) -> tuple[bool, dict] | None:
        if p.exists():
            q = check_video_quality(p, full_decode=full_decode)
            if not q["integrity_error"]:
                return True, q
            p.unlink(missing_ok=True)
        return None

    src = plan.full_source
    if not src.exists():
        msg = f"完整视频不存在: {src}"
        print(f"[ERROR] {msg}")
        return False, msg, {}, False, msg, {}, v1_start_sec, split_time, v2_end_sec

    # ── Cut v1 (推介致辞): [v1_start_sec, split_time or EOF] ──────────────
    v1_ok, v1_err, v1_q = True, "", {}
    v1_done = _check_existing(v1_output)
    if not v1_done:
        v1_ok, v1_err = cut_video(
            src, v1_output, start_sec=v1_start_sec, end_sec=split_time
        )
        if v1_ok:
            v1_q = check_video_quality(v1_output, full_decode=full_decode)
    else:
        v1_ok, v1_q = v1_done

    # ── Cut v2 (答谢致辞): [split_time, v2_end_sec or EOF] ────────────────
    v2_ok, v2_err, v2_q = False, "无答谢致辞", {}
    if has_v2:
        v2_done = _check_existing(v2_output)
        if not v2_done:
            v2_ok, v2_err = cut_video(
                src, v2_output, start_sec=split_time, end_sec=v2_end_sec
            )
            if v2_ok:
                v2_q = check_video_quality(v2_output, full_decode=full_decode)
        else:
            v2_ok, v2_q = v2_done
            v2_err = ""

    return (v1_ok, v1_err, v1_q, v2_ok, v2_err, v2_q,
            v1_start_sec, split_time, v2_end_sec)


# ── Multiprocessing worker ─────────────────────────────────────────────────────

def process_video_row(row_data: dict) -> dict:
    """
    Process one roadshow row and return a serialisable VideoTaskResult dict.
    Must be a module-level function for multiprocessing.Pool compatibility.
    """
    index2009  = row_data["index2009"]
    code       = row_data["code"]
    date       = row_data["date"]
    platform   = row_data["platform"]
    full_decode = row_data.get("full_decode", True)

    v1_output, v2_output = _output_paths(index2009, code, date)

    # Build the split plan
    cscom_plans   = row_data.get("cscom_plans", {})
    split_windows = row_data.get("split_windows", {})
    plan = resolve_split_plan(row_data, cscom_plans, split_windows)

    warnings = plan.validate()
    for w in warnings:
        print(f"[WARN] {w}")

    result = VideoTaskResult(
        index2009=index2009,
        v1_path="", v2_path="",
        v1_success=False, v2_success=False,
        v1_error="", v2_error="",
    )

    if plan.is_split_mode:
        (
            result.v1_success, result.v1_error, result.v1_quality,
            result.v2_success, result.v2_error, result.v2_quality,
            result.v1_start_sec, result.split_time, result.v2_end_sec,
        ) = _produce_from_split(plan, v1_output, v2_output, full_decode)
    else:
        result.v1_success, result.v1_error, result.v1_quality = (
            _produce_from_sources(plan.v1_sources, v1_output, full_decode)
        )
        result.v2_success, result.v2_error, result.v2_quality = (
            _produce_from_sources(plan.v2_sources, v2_output, full_decode)
        )

    if result.v1_success:
        result.v1_path = str(v1_output)
    if result.v2_success:
        result.v2_path = str(v2_output)

    return asdict(result)


# ── Main collection function ───────────────────────────────────────────────────

def collect_video_tasks(full_decode: bool = True) -> list[dict]:
    """
    Read the IPO index, process every roadshow in parallel, and write a
    quality report to ``PROJECT_ROOT/ipo_index_video_preprocess_v2.xlsx``.

    Returns a list of VideoTaskResult dicts (one per roadshow).
    """
    df_index = pd.read_excel(INDEX_PATH, dtype=str)
    df_index = df_index[df_index["采用视频平台"].notna()].copy()

    # Pre-load plans and split-window boundaries
    cscom_plans   = load_cscom_plans()
    split_windows = load_split_windows()
    if split_windows:
        n_sw = sum(1 for w in split_windows.values() if w.get("v2_start_sec", "").strip())
        print(f"split_windows 表已加载: {n_sw}/{len(split_windows)} 条有效分割点")
    else:
        print("[INFO] tmp/video_split_windows.csv 不存在，"
              "将对全景/IR/中证完整视频使用转录文本实时检测分割点")

    row_data_list: list[dict] = []
    for _, row in df_index.iterrows():
        platform  = str(row["采用视频平台"]).strip()
        index2009 = str(row.get("INDEX2009", "")).strip()
        code      = str(row.get(f"{platform}_去重代码", "")).strip()
        date      = str(row.get(f"{platform}_日期",     "")).strip()

        row_data_list.append({
            "platform":      platform,
            "index2009":     index2009,
            "code":          code,
            "date":          date,
            "full_decode":   full_decode,
            "cscom_plans":   cscom_plans,
            "split_windows": split_windows,
        })

    num_workers = max(1, PARALLEL_PROCESSES)
    with multiprocessing.Pool(processes=num_workers) as pool:
        raw_results: list[dict] = pool.map(process_video_row, row_data_list)

    v1_ok = sum(1 for r in raw_results if r["v1_success"])
    v2_ok = sum(1 for r in raw_results if r["v2_success"])
    n     = len(raw_results)
    print(
        f"完成: {n} 场路演 | "
        f"推介致辞 {v1_ok}/{n} | 答谢致辞 {v2_ok}/{n}"
    )

    # ── Write quality report ───────────────────────────────────────────────
    rows = []
    for r in raw_results:
        base = {
            "index2009":    r["index2009"],
            "v1_path":      r["v1_path"],
            "v2_path":      r["v2_path"],
            "v1_success":   r["v1_success"],
            "v2_success":   r["v2_success"],
            "v1_error":     r["v1_error"],
            "v2_error":     r["v2_error"],
            "v1_start_sec": r.get("v1_start_sec", 0.0),
            "split_time":   r.get("split_time"),
            "v2_end_sec":   r.get("v2_end_sec"),
        }
        for prefix, key in (("v1_", "v1_quality"), ("v2_", "v2_quality")):
            for k, v in (r.get(key) or {}).items():
                base[f"{prefix}{k}"] = v
        rows.append(base)

    df_quality = pd.DataFrame(rows)
    xlsx_path  = PROJECT_ROOT / "ipo_index_video_preprocess_v2.xlsx"
    df_quality.to_excel(xlsx_path, index=False)
    print(f"质量报告已保存: {xlsx_path}")

    return raw_results


# ── Audio task collection (unchanged logic) ────────────────────────────────────

def collect_audio_tasks() -> list[tuple[str, Path | None, Path | None]]:
    """
    Build an audio path list for every roadshow, now referencing the two
    separated clips (推介 / 答谢).

    Returns a list of (index2009, v1_audio_path | None, v2_audio_path | None).
    """
    df_index = pd.read_excel(INDEX_PATH, dtype=str)
    df_index = df_index[df_index["采用视频平台"].notna()].copy()

    results: list[tuple[str, Path | None, Path | None]] = []

    for _, row in df_index.iterrows():
        platform  = str(row["采用视频平台"]).strip()
        index2009 = str(row.get("INDEX2009", "")).strip()
        code      = str(row.get(f"{platform}_去重代码", "")).strip()
        date      = str(row.get(f"{platform}_日期",     "")).strip()
        stem      = f"{index2009}_{code}_{date}"

        v1_audio = AUDIO_OUTPUT_DIR / f"{stem}_推介.wav"
        v2_audio = AUDIO_OUTPUT_DIR / f"{stem}_答谢.wav"

        results.append((
            index2009,
            v1_audio if v1_audio.exists() else None,
            v2_audio if v2_audio.exists() else None,
        ))

    v1_ok = sum(1 for _, a, _ in results if a)
    v2_ok = sum(1 for _, _, b in results if b)
    n     = len(results)
    print(f"音频检查: {n} 场路演 | 推介 {v1_ok}/{n} | 答谢 {v2_ok}/{n}")
    return results
