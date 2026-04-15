"""
analyze/qa_analysis.py
======================
Per-roadshow Q&A indicators from unified QA CSVs (路演问答/).

Structural indicators:
    qa_pairs             — count of 消息类型=问答 rows
    speech_count         — count of 消息类型=发言 rows
    avg_q_len            — mean char length of 提问内容 (QA rows)
    avg_a_len            — mean char length of 回答内容 (QA rows)
    a_q_len_ratio        — avg_a_len / avg_q_len
    num_ratio_in_answer  — digit chars / total answer chars
    n_unique_questioners — distinct questioner_id on QA rows
                           (falls back to 提问人 if questioner_id column absent)

Verbal sentiment — lexicon-based, same method as verbal_sentiment.py:
    q_* prefix — computed on concatenated 提问内容 text (all QA rows)
    a_* prefix — computed on concatenated 回答内容 text (all QA rows)
    Metrics per prefix:
        ann_positive_ratio, ann_negative_ratio, ann_tone_score,
        social_positive_ratio, social_negative_ratio, social_tone_score,
        competition_ratio, prospect_ratio, policy_pos_ratio, policy_neg_ratio,
        total_words, method
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from verbal_sentiment import load_lexicons, analyze_with_lexicons


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _num_ratio(text: str) -> float:
    """Digit characters / all non-whitespace characters in text."""
    chars = text.replace(" ", "")
    if not chars:
        return float("nan")
    return sum(1 for c in chars if c.isdigit()) / len(chars)


def _mean_len(series: pd.Series) -> float:
    """Mean character length of a string Series, ignoring empty/NaN."""
    lengths = series.dropna().map(lambda x: len(str(x).strip()))
    lengths = lengths[lengths > 0]
    return float(lengths.mean()) if len(lengths) > 0 else float("nan")


def _concat_text(series: pd.Series) -> str:
    """Concatenate non-empty strings from a Series into one string."""
    return " ".join(str(v).strip() for v in series.dropna() if str(v).strip())


def _nan_verbal(prefix: str) -> dict:
    """Return all-NaN verbal metrics for a given prefix."""
    nan = float("nan")
    return {
        f"{prefix}ann_positive_ratio":    nan,
        f"{prefix}ann_negative_ratio":    nan,
        f"{prefix}ann_tone_score":        nan,
        f"{prefix}social_positive_ratio": nan,
        f"{prefix}social_negative_ratio": nan,
        f"{prefix}social_tone_score":     nan,
        f"{prefix}competition_ratio":     nan,
        f"{prefix}prospect_ratio":        nan,
        f"{prefix}policy_pos_ratio":      nan,
        f"{prefix}policy_neg_ratio":      nan,
        f"{prefix}total_words":           0,
        f"{prefix}method":                "",
    }


def _score_text(text: str, lexicons: dict, prefix: str) -> dict:
    """Run lexicon analysis on text and prefix all output keys."""
    if not text.strip():
        return _nan_verbal(prefix)
    scores = analyze_with_lexicons(text, lexicons)
    return {f"{prefix}{k}": v for k, v in scores.items()}


# ─── Main analysis function ────────────────────────────────────────────────────

def analyze_one_qa(csv_path: Path, lexicons: dict) -> dict:
    """
    Compute all Q&A indicators for one roadshow.

    Parameters
    ----------
    csv_path : unified QA CSV (路演问答/{stem}.csv)
    lexicons : loaded lexicon dict from load_lexicons()

    Returns
    -------
    dict with keys: file_stem, index2009, structural indicators,
                    q_* verbal metrics, a_* verbal metrics, error
    """
    nan = float("nan")
    stem = csv_path.stem

    result: dict = {
        "file_stem":             stem,
        "index2009":             "",
        "qa_pairs":              0,
        "speech_count":          0,
        "avg_q_len":             nan,
        "avg_a_len":             nan,
        "a_q_len_ratio":         nan,
        "num_ratio_in_answer":   nan,
        "n_unique_questioners":  0,
        "error":                 "",
    }
    result.update(_nan_verbal("q_"))
    result.update(_nan_verbal("a_"))

    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
        if df.empty:
            result["error"] = "empty file"
            return result

        result["index2009"] = df["index2009"].iloc[0] if "index2009" in df.columns else ""

        qa = df[df["消息类型"] == "问答"].copy()
        sp = df[df["消息类型"] == "发言"].copy()

        result["qa_pairs"]    = len(qa)
        result["speech_count"] = len(sp)

        if qa.empty:
            result["error"] = "no QA rows"
            return result

        # ── Structural ────────────────────────────────────────────────────────
        result["avg_q_len"] = _mean_len(qa["提问内容"])
        result["avg_a_len"] = _mean_len(qa["回答内容"])

        avg_q = result["avg_q_len"]
        avg_a = result["avg_a_len"]
        if pd.notna(avg_q) and avg_q > 0 and pd.notna(avg_a):
            result["a_q_len_ratio"] = avg_a / avg_q

        a_text_all = _concat_text(qa["回答内容"])
        result["num_ratio_in_answer"] = _num_ratio(a_text_all)

        # questioner_id preferred; fall back to 提问人 for files pre-dating schema update
        id_col = "questioner_id" if "questioner_id" in df.columns else "提问人"
        result["n_unique_questioners"] = int(
            qa[id_col].replace("", pd.NA).dropna().nunique()
        )

        # ── Verbal sentiment ──────────────────────────────────────────────────
        q_text = _concat_text(qa["提问内容"])
        a_text = _concat_text(qa["回答内容"])

        result.update(_score_text(q_text, lexicons, "q_"))
        result.update(_score_text(a_text, lexicons, "a_"))

    except Exception as e:
        result["error"] = str(e)

    return result


# ─── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import INDEX_QA_DIR, LEXICON_DIR

    lexicons = load_lexicons(LEXICON_DIR)
    path = next(INDEX_QA_DIR.glob("*.csv"), None)
    if path is None:
        print("No CSV found in", INDEX_QA_DIR)
        sys.exit(1)

    print(f"Testing on: {path.name}\n")
    res = analyze_one_qa(path, lexicons)
    for k, v in res.items():
        print(f"  {k}: {v}")
