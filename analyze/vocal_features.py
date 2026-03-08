"""
analyze/vocal_features.py
==========================
从 .wav 音频文件中提取声学特征（Vocal Component of Pitch Factor）。

特征：
  - f0_mean / f0_std     : 基本频率（基频）均值和标准差（使用 librosa.yin）
  - rms_mean / rms_std   : 均方根能量（音量代理）均值和标准差
  - speech_rate          : 语速（字/词 per second），需要对应转录 JSON

依赖：
  pip install librosa soundfile numpy
"""

import json
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


# ─── 工具函数 ─────────────────────────────────────────────────────────

def load_audio(wav_path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """加载 wav 文件，返回 (waveform, sample_rate)。"""
    if not _LIBROSA_OK:
        raise ImportError("librosa 未安装")
    y, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
    return y, sr


def extract_f0(y: np.ndarray, sr: int,
                fmin: float = 50.0, fmax: float = 600.0,
                hop_length: int = 512) -> tuple[float, float]:
    """
    使用 YIN 算法提取基频序列，返回 (均值, 标准差)。
    静音/未发声帧的 F0 为 0，计算时排除这些帧。
    """
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax,
                     sr=sr, hop_length=hop_length)
    voiced = f0[f0 > fmin]   # 排除静音帧（F0≈0 或极低）
    if len(voiced) == 0:
        return 0.0, 0.0
    return float(np.mean(voiced)), float(np.std(voiced))


def extract_rms(y: np.ndarray, sr: int,
                 hop_length: int = 512) -> tuple[float, float]:
    """
    提取帧级 RMS 能量，返回 (均值, 标准差)。
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return float(np.mean(rms)), float(np.std(rms))


def calc_speech_rate(transcript_json_path: Path) -> float:
    """
    从 WhisperX 转录 JSON 计算语速（字符/秒）。

    WhisperX JSON 结构：
    {
        "segments": [
            {"start": 0.0, "end": 3.5, "text": "..."},
            ...
        ]
    }
    语速 = 总字符数 / 总有效时长（end - start 之和）
    """
    if not transcript_json_path.exists():
        return float("nan")

    with open(transcript_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        return float("nan")

    total_chars = sum(len(seg.get("text", "").strip()) for seg in segments)
    total_duration = sum(
        max(0.0, seg.get("end", 0) - seg.get("start", 0))
        for seg in segments
    )

    if total_duration <= 0:
        return float("nan")
    return total_chars / total_duration  # 字符/秒


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
    从 wav 文件提取声学特征。

    参数
    ----
    wav_path              : 输入 .wav 文件路径
    transcript_json_path  : 对应的 WhisperX 转录 JSON（用于计算语速）；
                            若为 None 则 speech_rate = nan
    target_sr             : 目标采样率（默认 16000 Hz）
    fmin / fmax           : 基频范围（Hz）
    hop_length            : STFT hop length

    返回
    ----
    dict with keys:
        file_stem    : wav 文件名（无扩展名）
        f0_mean      : 基频均值 (Hz)
        f0_std       : 基频标准差 (Hz)
        rms_mean     : RMS 能量均值
        rms_std      : RMS 能量标准差
        speech_rate  : 语速 (字符/秒)；若无转录 JSON 则为 nan
        duration_s   : 音频总时长（秒）
        error        : 错误信息（成功时为空字符串）
    """
    result = {
        "file_stem":   wav_path.stem,
        "f0_mean":     float("nan"),
        "f0_std":      float("nan"),
        "rms_mean":    float("nan"),
        "rms_std":     float("nan"),
        "speech_rate": float("nan"),
        "duration_s":  float("nan"),
        "error":       "",
    }

    try:
        y, sr = load_audio(wav_path, target_sr=target_sr)
        result["duration_s"] = float(len(y)) / sr

        # 基频
        f0_mean, f0_std = extract_f0(y, sr, fmin=fmin, fmax=fmax,
                                      hop_length=hop_length)
        result["f0_mean"] = f0_mean
        result["f0_std"]  = f0_std

        # 音量
        rms_mean, rms_std = extract_rms(y, sr, hop_length=hop_length)
        result["rms_mean"] = rms_mean
        result["rms_std"]  = rms_std

        # 语速
        if transcript_json_path is not None:
            result["speech_rate"] = calc_speech_rate(transcript_json_path)

    except Exception as e:
        result["error"] = str(e)

    return result

if __name__ == "__main__":
    # 示例：处理单个文件
    wav_path = Path("example.wav")
    transcript_json_path = Path("example.json")
    features = extract_vocal_features(wav_path, transcript_json_path)
    print(features)