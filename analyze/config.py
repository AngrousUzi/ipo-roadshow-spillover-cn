"""
analyze/config.py
=================
集中管理路径、平台列表和全局常量。

本地（Windows）运行时数据路径在 videos/archive/ 下；
HPC（Linux）运行时数据在工作目录下，由 DATA_ROOT 控制。
"""

import os
from pathlib import Path

# ─── 项目根目录（PythonProject/）────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── 数据根目录 ──────────────────────────────────────────────────────
# 视频/音频/转录都在此目录的子目录下
if os.name == "nt":  # Windows 本地
    DATA_ROOT = PROJECT_ROOT / "videos" / "archive"
else:               # HPC / Linux
    DATA_ROOT = Path("./")  # 当前工作目录，假设已切换到数据所在目录

# ─── 平台列表 ────────────────────────────────────────────────────────
PLATFORM_LIST = ["全景", "上证", "中国证券网", "中证", "IR"]

# ─── 子目录名后缀 ─────────────────────────────────────────────────────
VIDEO_SUFFIX = "路演视频"    # e.g. 全景路演视频/
AUDIO_SUFFIX = "路演音频"    # e.g. 全景路演音频/
TRANS_SUFFIX = "路演转录"    # e.g. 全景路演转录/

# ─── 输出目录 ────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 分析参数 ────────────────────────────────────────────────────────
# 视觉分析：每秒采样帧数（越高越精确，越慢）
VISUAL_SAMPLE_FPS = 1.0

# 音频分析：librosa
AUDIO_SR = 16000          # 目标采样率（与 audio_extract.py 一致）
HOP_LENGTH = 512          # STFT hop length
FMIN = 50                 # F0 最小频率 (Hz)
FMAX = 600                # F0 最大频率 (Hz)

# PCA
PCA_N_COMPONENTS = 1      # 取第一主成分作为 Pitch Factor

# ─── 情绪词典目录（可选，放在 analyze/lexicons/） ─────────────────────
LEXICON_DIR = Path(__file__).resolve().parent / "lexicons"


def get_audio_dir(platform: str) -> Path:
    """返回指定平台的音频目录路径。"""
    return DATA_ROOT / f"{platform}{AUDIO_SUFFIX}"


def get_video_dir(platform: str) -> Path:
    """返回指定平台的视频目录路径。"""
    return DATA_ROOT / f"{platform}{VIDEO_SUFFIX}"


def get_trans_dir(platform: str) -> Path:
    """返回指定平台的转录目录路径。"""
    return DATA_ROOT / f"{platform}{TRANS_SUFFIX}"


def iter_audio_files():
    """遍历所有平台的 .wav 音频文件，yield Path 对象。"""
    for platform in PLATFORM_LIST:
        d = get_audio_dir(platform)
        if d.exists():
            yield from d.glob("*.wav")


def iter_video_files():
    """遍历所有平台的 .mp4 视频文件，yield Path 对象。"""
    for platform in PLATFORM_LIST:
        d = get_video_dir(platform)
        if d.exists():
            yield from d.glob("*.mp4")


def iter_trans_files():
    """遍历所有平台的转录 .json 文件，yield Path 对象。"""
    for platform in PLATFORM_LIST:
        d = get_trans_dir(platform)
        if d.exists():
            yield from d.glob("*.json")


def find_trans_for_audio(audio_path: Path) -> Path | None:
    """
    给定一个音频文件路径，查找对应的转录 JSON 文件。
    遍历所有平台的转录目录寻找同名（.json）文件。
    """
    stem = audio_path.stem
    for platform in PLATFORM_LIST:
        candidate = get_trans_dir(platform) / f"{stem}.json"
        if candidate.exists():
            return candidate
    return None
