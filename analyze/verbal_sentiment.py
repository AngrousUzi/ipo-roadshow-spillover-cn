"""
analyze/verbal_sentiment.py
============================
从 WhisperX 转录 JSON 提取文本，计算情绪与信息内容指标（Verbal Component）。

方法：
  1. 主力方法 — 中文金融情绪词典（姚加权等 2021）
       分别对年报词典（ann）和社媒词典（social）计算：
         positive_ratio = 正面词数 / 总词数
         negative_ratio = 负面词数 / 总词数
         tone_score     = (正面词数 - 负面词数) / 总词数

  2. 信息内容词典（IPO 路演溢出效应研究专用，人工构建）
       competition_ratio  — 竞争威胁信息比率（竞争效应/负溢出渠道）
       prospect_ratio     — 行业前景信息比率（认证效应/正溢出渠道）
       policy_pos_ratio   — 政策正向信息比率（政策支持/利好，正向溢出）
       policy_neg_ratio   — 政策负向信息比率（监管收紧/风险，负向溢出）

  3. 备用方法 — SnowNLP 句子极性分（当情绪词典文件均不存在时自动回退）

词典格式（lexicons/ 目录）：
  情绪词典：ann_positive.txt, ann_negative.txt,
            social_positive.txt, social_negative.txt
  信息词典：competition_threat.txt, industry_prospect.txt,
            policy_positive.txt, policy_negative.txt

依赖：
  pip install jieba snownlp
"""

import json
import re
from pathlib import Path
import time
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

def load_lexicon(txt_path: Path) -> set:
    """
    加载词典文件，返回词集合。
    同时将所有词条注入 jieba 用户词典，防止多字词组被切碎
    （如"市场规模"→["市场","规模"]）。
    """
    if not txt_path.exists():
        return set()
    words = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w and not w.startswith("#"):
                words.add(w)
                if _JIEBA_OK:
                    jieba.add_word(w)   # 注入用户词典，保证分词时作为整体保留
    return words


def load_lexicons(lexicon_dir: Path) -> dict:
    """
    从 lexicon_dir 加载情绪词典和信息内容词典，返回：
        {
            "ann":         {"positive": set, "negative": set},
            "social":      {"positive": set, "negative": set},
            "competition": set,   # 竞争威胁信息
            "prospect":    set,   # 行业前景信息
            "policy":      set,   # 政策监管信息
        }
    """
    return {
        "ann": {
            "positive": load_lexicon(lexicon_dir / "ann_positive.txt"),
            "negative": load_lexicon(lexicon_dir / "ann_negative.txt"),
        },
        "social": {
            "positive": load_lexicon(lexicon_dir / "social_positive.txt"),
            "negative": load_lexicon(lexicon_dir / "social_negative.txt"),
        },
        "competition": load_lexicon(lexicon_dir / "competition_threat.txt"),
        "prospect":    load_lexicon(lexicon_dir / "industry_prospect.txt"),
        "policy_pos":  load_lexicon(lexicon_dir / "policy_positive.txt"),
        "policy_neg":  load_lexicon(lexicon_dir / "policy_negative.txt"),
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
        return [c for c in cleaned if c.strip()]


# ─── 主要分析方法 ──────────────────────────────────────────────────────

def _score_one_lexicon(tokens: list[str], lexicon: dict[str, set], prefix: str) -> dict:
    """
    对已分词的 tokens 按一套词典（positive/negative）计算情绪指标。
    输出 key 均带 prefix（"ann_" 或 "social_"）。
    """
    total = len(tokens)
    if total == 0:
        return {
            f"{prefix}positive_ratio": 0.0,
            f"{prefix}negative_ratio": 0.0,
            f"{prefix}tone_score":     0.0,
        }

    pos_count = sum(1 for t in tokens if t in lexicon["positive"])
    neg_count = sum(1 for t in tokens if t in lexicon["negative"])

    return {
        f"{prefix}positive_ratio": pos_count / total,
        f"{prefix}negative_ratio": neg_count / total,
        f"{prefix}tone_score":     (pos_count - neg_count) / total,
    }


def _score_info_lexicon(tokens: list[str], word_set: set, key: str) -> dict:
    """计算单个信息内容词典的命中比率。"""
    total = len(tokens)
    if total == 0:
        return {key: 0.0}
    return {key: sum(1 for t in tokens if t in word_set) / total}


def analyze_with_lexicons(text: str, lexicons: dict) -> dict:
    """
    分词一次，分别对情绪词典和信息内容词典计算指标。

    返回 keys：
        ann_positive_ratio, ann_negative_ratio, ann_tone_score,
        social_positive_ratio, social_negative_ratio, social_tone_score,
        competition_ratio, prospect_ratio, policy_pos_ratio, policy_neg_ratio,
        total_words, method
    """
    tokens = _tokenize(text)
    result = {"total_words": len(tokens), "method": "lexicon"}
    result.update(_score_one_lexicon(tokens, lexicons["ann"],    "ann_"))
    result.update(_score_one_lexicon(tokens, lexicons["social"], "social_"))
    result.update(_score_info_lexicon(tokens, lexicons["competition"], "competition_ratio"))
    result.update(_score_info_lexicon(tokens, lexicons["prospect"],    "prospect_ratio"))
    result.update(_score_info_lexicon(tokens, lexicons["policy_pos"],  "policy_pos_ratio"))
    result.update(_score_info_lexicon(tokens, lexicons["policy_neg"],  "policy_neg_ratio"))
    return result


def analyze_with_snownlp(segments: list[dict]) -> dict:
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

    sentiment_avg = sum(scores) / len(scores) if scores else 0.5
    tone_score = (sentiment_avg - 0.5) * 2  # 映射到 [-1, 1]

    return {
        "ann_positive_ratio":    float("nan"),
        "ann_negative_ratio":    float("nan"),
        "ann_tone_score":        tone_score,
        "social_positive_ratio": float("nan"),
        "social_negative_ratio": float("nan"),
        "social_tone_score":     tone_score,
        "competition_ratio":     float("nan"),
        "prospect_ratio":        float("nan"),
        "policy_pos_ratio":      float("nan"),
        "policy_neg_ratio":      float("nan"),
        "total_words":           len(scores),
        "method":                "snownlp",
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
        file_stem             : JSON 文件名（无扩展名）
        ann_positive_ratio    : 年报正面词比率
        ann_negative_ratio    : 年报负面词比率
        ann_tone_score        : 年报 (pos-neg)/total
        social_positive_ratio : 社媒正面词比率
        social_negative_ratio : 社媒负面词比率
        social_tone_score     : 社媒 (pos-neg)/total
        total_words           : 分词总数
        total_chars           : 总字符数（转录全文）
        method                : "lexicon" 或 "snownlp"
        error                 : 错误信息
    """
    nan = float("nan")
    result = {
        "file_stem":             transcript_json_path.stem,
        "ann_positive_ratio":    nan,
        "ann_negative_ratio":    nan,
        "ann_tone_score":        nan,
        "social_positive_ratio": nan,
        "social_negative_ratio": nan,
        "social_tone_score":     nan,
        "competition_ratio":     nan,
        "prospect_ratio":        nan,
        "policy_pos_ratio":      nan,
        "policy_neg_ratio":      nan,
        "total_words":           0,
        "total_chars":           0,
        "method":                "",
        "error":                 "",
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

        if lexicon_dir is None:
            lexicon_dir = Path(__file__).resolve().parent / "lexicons"

        lexicons = load_lexicons(lexicon_dir)
        has_lexicon = any(
            len(v) > 0
            for suite in lexicons.values()
            for v in (suite.values() if isinstance(suite, dict) else [suite])
        )

        if has_lexicon:
            scores = analyze_with_lexicons(full_text, lexicons)
        else:
            scores = analyze_with_snownlp(segments)

        result.update(scores)

    except Exception as e:
        result["error"] = str(e)

    return result


if __name__ == "__main__":
    json_path = Path("IR路演转录/003030_祖名股份_2020-12-22.json")
    start_time = time.time()
    res = analyze_verbal_sentiment(json_path)
    end_time = time.time()
    print(res)
    print(f"分析耗时: {end_time - start_time:.2f} 秒")
