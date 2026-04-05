"""
analyze/initialize/cscom_plan.py
=======================================
Loads ``initialize/中证视频分类.xlsx`` and builds a per-company plan that
describes which source video files should become 推介致辞 (v1) and which
should become 答谢致辞 (v2).

Public API
----------
load_cscom_plans() -> dict[str, dict]
    Returns a mapping  code -> serialisable plan dict.
    Use :func:`plan_from_dict` to reconstruct a typed object.

Design notes
------------
- Videos with 采用 == 'False' are excluded.
- Within each type, files are ordered by their 视频N number (ascending),
  which reflects the original recording sequence.
- A company whose only adopted video is typed '完整视频' AND '需要切分' is
  flagged as needs_split=True; the single complete file will be split at a
  transcript-derived timestamp at runtime.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Make the parent analyze/ directory importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

_EXCEL_PATH = PROJECT_ROOT / "initialize" / "中证视频分类.xlsx"


@dataclass
class CscomCompanyPlan:
    """
    Processing plan for one 中证 company's roadshow.

    Attributes
    ----------
    code : str
        Stock code (e.g. "001217").
    v1_filenames : list[str]
        Source video filenames for 推介致辞, in concatenation order.
    v2_filenames : list[str]
        Source video filenames for 答谢致辞, in concatenation order.
    needs_split : bool
        True when there is a single 完整视频 that must be split with a
        transcript-based boundary.
    full_filename : str
        Filename of the complete video when ``needs_split`` is True.
    """

    code: str
    v1_filenames: list[str] = field(default_factory=list)
    v2_filenames: list[str] = field(default_factory=list)
    needs_split: bool = False
    full_filename: str = ""

    def to_dict(self) -> dict:
        return {
            "code":           self.code,
            "v1_filenames":   self.v1_filenames,
            "v2_filenames":   self.v2_filenames,
            "needs_split":    self.needs_split,
            "full_filename":  self.full_filename,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CscomCompanyPlan":
        return cls(
            code          = d["code"],
            v1_filenames  = d.get("v1_filenames", []),
            v2_filenames  = d.get("v2_filenames", []),
            needs_split   = d.get("needs_split", False),
            full_filename = d.get("full_filename", ""),
        )


def _video_number(filename: str) -> int:
    """Extract the integer after '_视频' from a filename for sort ordering."""
    import re
    m = re.search(r"_视频(\d+)", filename)
    return int(m.group(1)) if m else 999


def load_cscom_plans() -> dict[str, dict]:
    """
    Parse ``initialize/中证视频分类.xlsx`` and return a dict mapping each
    company code to its :class:`CscomCompanyPlan` serialised as a plain
    dict (suitable for multiprocessing pickling).

    Only rows where 采用 == 'True' are considered.
    """
    df = pd.read_excel(_EXCEL_PATH, dtype=str)

    # Normalise boolean-like strings
    df["采用"]     = df["采用"].str.strip()
    df["视频类型"] = df["视频类型"].fillna("").str.strip()
    df["需要切分"] = df["需要切分"].fillna("").str.strip()

    adopted = df[df["采用"] == "True"].copy()

    plans: dict[str, CscomCompanyPlan] = {}

    for code, group in adopted.groupby("公司代码", sort=False):
        code = str(code).strip()
        plan = CscomCompanyPlan(code=code)

        split_rows = group[group["需要切分"].str.lower() == "true"]

        if not split_rows.empty:
            # Split mode: one complete video to be cut at a transcript boundary
            plan.needs_split   = True
            plan.full_filename = str(split_rows.iloc[0]["视频名称"]).strip()
        else:
            # Merge mode: only 推介致辞 (v1) and 答谢致辞 (v2) rows are used
            for _, row in group.iterrows():
                vtype    = str(row["视频类型"]).strip()
                filename = str(row["视频名称"]).strip()
                if vtype == "推介致辞":
                    plan.v1_filenames.append(filename)
                elif vtype == "答谢致辞":
                    plan.v2_filenames.append(filename)
                # 完整视频 without split marker, 宣传片, empty → excluded

        plans[code] = plan

    # Sort by video number within each group
    for plan in plans.values():
        plan.v1_filenames.sort(key=_video_number)
        plan.v2_filenames.sort(key=_video_number)

    return {code: p.to_dict() for code, p in plans.items()}
