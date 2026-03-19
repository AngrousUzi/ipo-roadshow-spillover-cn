"""
analyze/vocal_features.py
==========================
从 .wav 音频文件中提取声学特征（Vocal Component of Pitch Factor）。

特征一览
--------
【音频层（librosa）】
  f0_mean / f0_std       : 基频均值与标准差 (Hz)，仅含有声帧
  f0_range               : 有声帧基频 IQR (P75 - P25)，捕捉典型音调变化幅度（原 max-min 对长录音饱和，已改为 IQR）
  f0_slope               : 有声帧基频线性趋势 (Hz/s)，正值=语调上扬，负值=下沉
  voiced_fraction        : 有声帧占总帧数比例（0~1），低值可能代表停顿多
  rms_mean / rms_std     : RMS 能量均值与标准差
  rms_cv                 : RMS 变异系数 (std/mean)，与录音音量无关的归一化波动
  rms_dynamic_range      : RMS p95 - p5，能量动态范围
  rms_snr_proxy          : RMS p50 / p05，粗略信噪比代理（越高越干净）

【转录层（WhisperX JSON）—— 与音频质量完全无关】
  speech_rate            : 语速 (字符/秒)，= 总字符 / 有效片段时长之和
  articulation_rate      : 发声速率 (字符/秒)，= 总字符 / 纯发声时长（字级时间戳之和）
                           与 speech_rate 的差反映停顿密集程度
  pause_rate             : 停顿比例 = (总时长 - 有效片段时长之和) / 总时长
  mean_pause_duration    : 片段间停顿 (>0.5s) 均值 (s)
  n_pauses_per_min       : 片段间停顿 (>0.5s) 每分钟次数
  asr_logprob_mean       : 片段级 avg_logprob 均值（ASR 置信度，可用作质量控制变量）

设计说明
--------
- HNR / Jitter / Shimmer 已放弃：极度依赖干净录音，10 年跨度的路演视频
  录制条件差异过大，这类指标会被背景噪声严重污染，缺乏跨样本可比性。
- MFCC 未纳入：主成分在不同录音设备/房间条件下语义不稳定，难以比较。
- rms_snr_proxy 和 asr_logprob_mean 可作为分析中的音质控制变量。

依赖：
  pip install librosa soundfile numpy
"""

import json
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# librosa 加载时会有一些 numba 警告，可以抑制
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False
    print("[WARNING] librosa 未安装，vocal 模块不可用。请运行: pip install librosa")


# ─── 常量 ─────────────────────────────────────────────────────────────

# 片段间停顿判定阈值（秒）：短于此值视为正常句间停顿，不计入停顿统计
PAUSE_THRESHOLD = 0.5

# 判断是否为汉字字符（排除标点）
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')


def _is_speech_char(word: str) -> bool:
    """判断一个 word_segments 词条是否为实际汉字（排除标点、空白）。"""
    return bool(_CJK_RE.search(word))


# ─── 音频特征提取 ──────────────────────────────────────────────────────

def load_audio(wav_path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """加载 wav 文件，返回 (waveform, sample_rate)。"""
    if not _LIBROSA_OK:
        raise ImportError("librosa 未安装")
    y, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
    return y, sr


def extract_f0_features(
    y: np.ndarray, sr: int,
    fmin: float = 50.0, fmax: float = 600.0,
    hop_length: int = 512,
) -> dict:
    """
    使用 pYIN 算法提取基频，返回均值、标准差、IQR、趋势斜率及有声帧比例。

    pYIN 相比 YIN 的优势：
      - 返回 voiced_flag 布尔数组，有声/无声判断准确（无声帧 f0=NaN）
      - voiced_fraction 不再被无声段的伪估计值虚高

    f0_range 改为 IQR（P75 - P25）：
      - 原 max-min 对长录音永远饱和（≈ fmax - fmin），无区分度
      - IQR 只反映典型音调变化幅度，对极端帧鲁棒

    返回 keys: f0_mean, f0_std, f0_range, f0_slope, voiced_fraction
    """
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )

    total_frames = len(f0)
    voiced_fraction = float(np.sum(voiced_flag)) / total_frames if total_frames > 0 else 0.0

    voiced = f0[voiced_flag]   # pYIN 在无声帧返回 NaN，voiced_flag=True 的帧均有效

    if len(voiced) == 0:
        return dict(f0_mean=float("nan"), f0_std=float("nan"),
                    f0_range=float("nan"), f0_slope=float("nan"),
                    voiced_fraction=voiced_fraction)

    f0_mean  = float(np.mean(voiced))
    f0_std   = float(np.std(voiced))
    f0_range = float(np.percentile(voiced, 75) - np.percentile(voiced, 25))  # IQR

    # 线性趋势：时间轴（秒）→ F0 (Hz)，斜率单位 Hz/s
    if len(voiced) >= 10:
        voiced_indices = np.where(voiced_flag)[0]
        times = voiced_indices * hop_length / sr
        slope = float(np.polyfit(times, voiced, 1)[0])
    else:
        slope = float("nan")

    return dict(f0_mean=f0_mean, f0_std=f0_std, f0_range=f0_range,
                f0_slope=slope, voiced_fraction=voiced_fraction)


def extract_rms_features(y: np.ndarray, hop_length: int = 512) -> dict:
    """
    提取帧级 RMS 特征。

    返回 keys: rms_mean, rms_std, rms_cv, rms_dynamic_range, rms_snr_proxy
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    rms_mean = float(np.mean(rms))
    rms_std  = float(np.std(rms))
    rms_cv   = float(rms_std / rms_mean) if rms_mean > 0 else float("nan")

    p05 = float(np.percentile(rms, 5))
    p50 = float(np.percentile(rms, 50))
    p95 = float(np.percentile(rms, 95))

    rms_dynamic_range = p95 - p05
    rms_snr_proxy     = (p50 / p05) if p05 > 0 else float("nan")

    return dict(rms_mean=rms_mean, rms_std=rms_std, rms_cv=rms_cv,
                rms_dynamic_range=rms_dynamic_range, rms_snr_proxy=rms_snr_proxy)


# ─── 转录特征提取 ──────────────────────────────────────────────────────

def extract_transcript_features(
    transcript_json_path: Path,
    audio_duration_s: float,
) -> dict:
    """
    从 WhisperX JSON 提取语速、停顿和 ASR 置信度特征。
    与音频质量完全无关，仅依赖文本时间戳。

    参数
    ----
    transcript_json_path : WhisperX 输出的 JSON 文件路径
    audio_duration_s     : 音频总时长（秒），用于计算 pause_rate

    返回 keys:
        speech_rate, articulation_rate,
        pause_rate, mean_pause_duration, n_pauses_per_min,
        asr_logprob_mean
    """
    nan = float("nan")
    empty = dict(speech_rate=nan, articulation_rate=nan,
                 pause_rate=nan, mean_pause_duration=nan,
                 n_pauses_per_min=nan, asr_logprob_mean=nan)

    if transcript_json_path is None or not transcript_json_path.exists():
        return empty

    with open(transcript_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        return empty

    # ── 1. speech_rate：总字符 / 片段有效时长之和 ──────────────────────
    total_chars    = sum(len(seg.get("text", "").strip()) for seg in segments)
    speaking_time  = sum(
        max(0.0, seg.get("end", 0) - seg.get("start", 0)) for seg in segments
    )
    speech_rate = total_chars / speaking_time if speaking_time > 0 else nan

    # ── 2. articulation_rate：总字符 / 纯发声时长（字级时间戳） ─────────
    word_segments = data.get("word_segments", [])
    # 回退：若顶层没有 word_segments，从各 segment.words 合并
    if not word_segments:
        for seg in segments:
            word_segments.extend(seg.get("words", []))

    speech_words = [
        w for w in word_segments
        if _is_speech_char(w.get("word", ""))
        and w.get("end") is not None and w.get("start") is not None
    ]
    phonation_time = sum(
        max(0.0, w["end"] - w["start"]) for w in speech_words
    )
    n_chars = len(speech_words)
    articulation_rate = n_chars / phonation_time if phonation_time > 0 else nan

    # ── 3. pause_rate：停顿占总时长比例 ──────────────────────────────────
    if audio_duration_s > 0 and speaking_time > 0:
        pause_rate = max(0.0, (audio_duration_s - speaking_time) / audio_duration_s)
    else:
        pause_rate = nan

    # ── 4. 片段间长停顿统计 ───────────────────────────────────────────────
    # 仅统计相邻片段之间的间隙，不含片头/片尾静音
    inter_gaps = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1].get("start", 0) - segments[i].get("end", 0)
        if gap > PAUSE_THRESHOLD:
            inter_gaps.append(gap)

    mean_pause_duration = float(np.mean(inter_gaps)) if inter_gaps else nan
    duration_min        = audio_duration_s / 60.0 if audio_duration_s > 0 else nan
    n_pauses_per_min    = (len(inter_gaps) / duration_min
                           if duration_min and duration_min > 0 else nan)

    # ── 5. ASR 置信度（avg_logprob 均值）────────────────────────────────
    logprobs = [seg["avg_logprob"] for seg in segments
                if "avg_logprob" in seg and seg["avg_logprob"] is not None]
    asr_logprob_mean = float(np.mean(logprobs)) if logprobs else nan

    return dict(
        speech_rate=speech_rate,
        articulation_rate=articulation_rate,
        pause_rate=pause_rate,
        mean_pause_duration=mean_pause_duration,
        n_pauses_per_min=n_pauses_per_min,
        asr_logprob_mean=asr_logprob_mean,
    )


# ─── 主接口 ───────────────────────────────────────────────────────────

def extract_vocal_features(
    wav_path: Path,
    transcript_json_path: Optional[Path] = None,
    target_sr: int = 16000,
    fmin: float = 50.0,
    fmax: float = 600.0,
    hop_length: int = 512,
) -> dict:
    """
    从 wav 文件（及可选的转录 JSON）提取全部声学特征。

    参数
    ----
    wav_path              : 输入 .wav 文件路径
    transcript_json_path  : WhisperX 转录 JSON（用于转录层特征）；
                            若为 None 则转录相关字段均为 nan
    target_sr             : 目标采样率（默认 16000 Hz）
    fmin / fmax           : 基频范围（Hz）
    hop_length            : STFT hop length

    返回
    ----
    dict，键见模块文档顶部"特征一览"。
    """
    nan = float("nan")
    result = {
        # ── 基本信息 ──
        "file_stem":          wav_path.stem,
        "duration_s":         nan,
        # ── F0 ──
        "f0_mean":            nan,
        "f0_std":             nan,
        "f0_range":           nan,
        "f0_slope":           nan,
        "voiced_fraction":    nan,
        # ── RMS ──
        "rms_mean":           nan,
        "rms_std":            nan,
        "rms_cv":             nan,
        "rms_dynamic_range":  nan,
        "rms_snr_proxy":      nan,
        # ── 转录 ──
        "speech_rate":        nan,
        "articulation_rate":  nan,
        "pause_rate":         nan,
        "mean_pause_duration": nan,
        "n_pauses_per_min":   nan,
        "asr_logprob_mean":   nan,
        # ── 状态 ──
        "error":              "",
    }

    try:
        y, sr = load_audio(wav_path, target_sr=target_sr)
        duration_s = float(len(y)) / sr
        result["duration_s"] = duration_s

        # F0 特征
        result.update(extract_f0_features(y, sr, fmin=fmin, fmax=fmax,
                                          hop_length=hop_length))

        # RMS 特征
        result.update(extract_rms_features(y, hop_length=hop_length))

        # 转录特征
        if transcript_json_path is not None:
            result.update(
                extract_transcript_features(transcript_json_path, duration_s)
            )

    except Exception as e:
        result["error"] = str(e)

    return result


# ─── 命令行快速测试 ───────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    wav_path = Path("路演音频/2009INDEX1848_002920_2017-12-13.wav")
    trans_path = Path("路演转录/2009INDEX1848_002920_2017-12-13.json")

    start = time.time()
    features = extract_vocal_features(wav_path, trans_path)
    elapsed = time.time() - start

    for k, v in features.items():
        print(f"  {k:<22} = {v}")
    print(f"\n提取耗时: {elapsed:.2f} 秒")
