"""
analyze/initialize/__main__.py
================================
Entry point for ``python -m analyze.initialize``.

Pipeline
--------
1. collect_video_tasks()   — split / concat source videos into 推介 + 答谢 clips
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make analyze/ importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .orchestrator import collect_video_tasks


def main() -> None:
    collect_video_tasks()


if __name__ == "__main__":
    main()
