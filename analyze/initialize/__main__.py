"""
analyze/initialize/__main__.py
================================
Entry point for ``python -m analyze.initialize``.

Pipeline
--------
1. collect_video_tasks()   — split / concat source videos into 推介 + 答谢 clips
2. Extract audio from each produced clip
3. collect_audio_tasks()   — verify audio presence and write a manifest CSV
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make analyze/ importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT

from .orchestrator import (
    AUDIO_OUTPUT_DIR,
    collect_audio_tasks,
    collect_video_tasks,
)

import os
import pandas as pd

if os.name == "nt":
    _VIDEO_OPS = PROJECT_ROOT / "videos"
    sys.path.insert(0, str(_VIDEO_OPS.resolve()))
else:
    _VIDEO_OPS = PROJECT_ROOT / ".." / "roadshow-cn"
    sys.path.insert(0, str(_VIDEO_OPS.resolve()))

from audio_extract import extract_task  # type: ignore[import]


def main() -> None:
    # ── 1. Process video splits / merges ──────────────────────────────────
    results = collect_video_tasks()

    # ── 2. Extract audio for every produced clip ───────────────────────────
    for r in results:
        for key in ("v1_path", "v2_path"):
            vp_str = r.get(key, "")
            if not vp_str:
                continue
            vp = Path(vp_str)
            audio_out = AUDIO_OUTPUT_DIR / (vp.stem + ".wav")
            if audio_out.exists():
                print(f"音频已存在，跳过: {audio_out.name}")
            else:
                extract_task((vp, audio_out))

    # ── 3. Verify audio presence and write manifest ────────────────────────
    audio_tasks = collect_audio_tasks()
    rows = [
        {
            "index2009":   idx,
            "v1_audio":    str(a1) if a1 else "",
            "v2_audio":    str(a2) if a2 else "",
        }
        for idx, a1, a2 in audio_tasks
    ]
    df = pd.DataFrame(rows)
    csv_path = PROJECT_ROOT / "audio_tasks_v2.csv"
    df.to_csv(csv_path, encoding="utf-8-sig", index=False)
    print(f"音频任务清单已保存: {csv_path}")


if __name__ == "__main__":
    main()
