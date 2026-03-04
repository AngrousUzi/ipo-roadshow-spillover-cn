"""
analyze/verbal_sentiment.py
============================
从 WhisperX 转录 JSON 提取文本，计算情绪指标（Verbal Component）。

方法：
  1. 主力方法 — 中文金融情绪词典（姚加权等 2021）
       正面词比率 = 正面词数 / 总词数
       负面词比率 = 负面词数 / 总词数
       不确定词比率 = 不确定词数 / 总词数
       tone_score = (正面词数 - 负面词数) / 总词数

  2. 备用方法 — SnowNLP 句子极性分（当词典文件不存在时自动回退）
       对每个转录片段计算极性分 [0,1]，取均值后映射为 tone_score

词典格式：
  lexicons/ 目录下放置以下 txt 文件（每行一个词）：
    positive.txt   — 正面词
    negative.txt   — 负面词
    uncertainty.txt— 不确定词

  若找不到词典文件，则自动使用 SnowNLP 备用方法。

依赖：
  pip install jieba snownlp
"""

import json
import re
from pathlib import Path
from typing import Optional

try:
    import jieba
    _JIEBA_OK = True
except ImportError:
    _JIEBA_OK = False
    print("[WARNING] jieba 未安装，将使用字符级分析。pip install jieba")

try:
    from snownlp import SnowNLP
    _SNOW_OK = True
except ImportError:
    _SNOW_OK = False


# ─── 词典加载 ─────────────────────────────────────────────────────────

def _load_lexicon(txt_path: Path) -> set:
    """加载词典文件，返回词集合。"""
    if not txt_path.exists():
        return set()
    words = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w and not w.startswith("#"):
                words.add(w)
    return words


def load_lexicons(lexicon_dir: Path) -> dict[str, set]:
    """
    从 lexicon_dir 加载三类词典，返回 dict:
        {"positive": set, "negative": set, "uncertainty": set}
    """
    return {
        "positive":   _load_lexicon(lexicon_dir / "positive.txt"),
        "negative":   _load_lexicon(lexicon_dir / "negative.txt"),
        "uncertainty": _load_lexicon(lexicon_dir / "uncertainty.txt"),
    }


# ─── 文本预处理 ────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """去除标点符号、数字，保留汉字和字母。"""
    return re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", " ", text)


def _tokenize(text: str) -> list[str]:
    """分词：优先使用 jieba，否则按字分割。"""
    cleaned = _clean_text(text)
    if _JIEBA_OK:
        return [w for w in jieba.cut(cleaned) if w.strip()]
    else:
        # 退回到按字符拆分
        return [c for c in cleaned if c.strip()]


# ─── 主要分析方法 ──────────────────────────────────────────────────────

def _analyze_with_lexicon(text: str, lexicons: dict[str, set]) -> dict:
    """使用词典方法计算情绪指标。"""
    tokens = _tokenize(text)
    total = len(tokens)
    if total == 0:
        return {
            "positive_ratio":   0.0,
            "negative_ratio":   0.0,
            "uncertain_ratio":  0.0,
            "tone_score":       0.0,
            "total_words":      0,
            "method":          "lexicon",
        }

    pos_count  = sum(1 for t in tokens if t in lexicons["positive"])
    neg_count  = sum(1 for t in tokens if t in lexicons["negative"])
    unc_count  = sum(1 for t in tokens if t in lexicons["uncertainty"])

    return {
        "positive_ratio":  pos_count / total,
        "negative_ratio":  neg_count / total,
        "uncertain_ratio": unc_count / total,
        "tone_score":      (pos_count - neg_count) / total,
        "total_words":     total,
        "method":          "lexicon",
    }


def _analyze_with_snownlp(segments: list[dict]) -> dict:
    """使用 SnowNLP 计算段落极性均值，映射为 tone_score ∈ [-1, 1]。"""
    if not _SNOW_OK:
        raise ImportError("snownlp 未安装，且词典文件不存在，无法计算情绪。pip install snownlp")

    scores = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            try:
                s = SnowNLP(text).sentiments  # [0, 1]
                scores.append(s)
            except Exception:
                pass

    if not scores:
        sentiment_avg = 0.5
    else:
        sentiment_avg = sum(scores) / len(scores)

    # 映射：0.5 → tone=0, 1.0 → tone=1.0, 0.0 → tone=-1.0
    tone_score = (sentiment_avg - 0.5) * 2

    return {
        "positive_ratio":  max(0.0, tone_score),
        "negative_ratio":  max(0.0, -tone_score),
        "uncertain_ratio": float("nan"),
        "tone_score":      tone_score,
        "total_words":     len(scores),
        "method":          "snownlp",
    }


# ─── 主接口 ───────────────────────────────────────────────────────────

def analyze_verbal_sentiment(
    transcript_json_path: Path,
    lexicon_dir: Optional[Path] = None,
) -> dict:
    """
    从 WhisperX 转录 JSON 计算情绪指标。

    参数
    ----
    transcript_json_path : .json 文件路径（WhisperX 输出格式）
    lexicon_dir          : 词典目录；若为 None 则使用默认的 analyze/lexicons/

    返回
    ----
    dict with keys:
        file_stem        : JSON 文件名（无扩展名）
        positive_ratio   : 正面词比率
        negative_ratio   : 负面词比率
        uncertain_ratio  : 不确定词比率
        tone_score       : (pos - neg) / total，范围近似 [-1, 1]
        total_words      : 分词总数
        total_chars      : 总字符数（转录全文）
        method           : "lexicon" 或 "snownlp"
        error            : 错误信息
    """
    result = {
        "file_stem":       transcript_json_path.stem,
        "positive_ratio":  float("nan"),
        "negative_ratio":  float("nan"),
        "uncertain_ratio": float("nan"),
        "tone_score":      float("nan"),
        "total_words":     0,
        "total_chars":     0,
        "method":          "",
        "error":           "",
    }

    try:
        if not transcript_json_path.exists():
            raise FileNotFoundError(f"转录 JSON 不存在: {transcript_json_path}")

        with open(transcript_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])
        full_text = " ".join(seg.get("text", "").strip() for seg in segments)
        result["total_chars"] = len(full_text.replace(" ", ""))

        if not full_text.strip():
            result["error"] = "转录文本为空"
            return result

        # 确定词典目录
        if lexicon_dir is None:
            lexicon_dir = Path(__file__).resolve().parent / "lexicons"

        # 尝试加载词典
        lexicons = load_lexicons(lexicon_dir)
        has_lexicon = any(len(v) > 0 for v in lexicons.values())

        if has_lexicon:
            scores = _analyze_with_lexicon(full_text, lexicons)
        else:
            # 无词典文件时使用 SnowNLP
            scores = _analyze_with_snownlp(segments)

        result.update(scores)

    except Exception as e:
        result["error"] = str(e)

    return result
