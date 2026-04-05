"""
analyze/initialize/
====================
Video initialisation pipeline: splits each roadshow into two clips —

    推介致辞  (v1)  —  pitch / presentation speech
    答谢致辞  (v2)  —  closing thank-you speech

Platform-specific strategies
-----------------------------
上证 / 中国证券网
    Files are already pre-labelled ``_视频1_`` (推介) and ``_视频2_``
    (答谢); they are stream-copied directly.

中证
    Driven by ``initialize/中证视频分类.xlsx``:
    • Multiple labelled segments → re-encoded concatenation per type.
    • Single 完整视频 (needs_split=True) → transcript-based cut.

全景 / IR
    Single complete-roadshow file → transcript-based cut.

Usage (as a script)
-------------------
    python -m analyze.initialize          # full pipeline
    python analyze/initialize/__main__.py # equivalent

Public API
----------
collect_video_tasks(full_decode=True) -> list[dict]
collect_audio_tasks()                 -> list[tuple]
"""

from .orchestrator import collect_audio_tasks, collect_video_tasks
from .platform_handlers import VideoSplitPlan, resolve_split_plan, load_split_windows
from .cscom_plan import CscomCompanyPlan, load_cscom_plans
from .gen_split_windows import generate as generate_split_windows

__all__ = [
    "collect_video_tasks",
    "collect_audio_tasks",
    "VideoSplitPlan",
    "resolve_split_plan",
    "load_split_windows",
    "CscomCompanyPlan",
    "load_cscom_plans",
    "generate_split_windows",
]
