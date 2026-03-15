"""
analyze/visual_fer.py
=====================
从路演视频中提取面部情绪（Visual Component of Pitch Factor）。

方法：
  每隔 sample_fps 帧抽取一帧 → 调用 DeepFace (backend='opencv') 分析面部情绪。
  将 7 类情绪聚合为：
    正面（positive）: happy, surprise
    负面（negative）: angry, fear, disgust, sad
    中性（neutral）:  neutral

  输出正面/负面/中性比率及净情绪分 (positive - negative)。

备用方案：
  若 deepface 不可用，回退到 fer 库（更轻量）。
  若两者均不可用，返回 NaN。

依赖：
  pip install deepface  （主）
  pip install fer       （备用）
  pip install opencv-python
"""

from pathlib import Path
import time
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

try:
    from deepface import DeepFace
    _DEEPFACE_OK = True
    _DEEPFACE_ERR = ""
except Exception as e:
    _DEEPFACE_OK = False
    _DEEPFACE_ERR = str(e)

try:
    from fer import FER
    _FER_OK = True
except Exception:
    _FER_OK = False

try:
    import cv2 as _cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False
    print("[WARNING] opencv-python 未安装，visual 模块不可用。pip install opencv-python")

import numpy as np


# ─── 情绪类别映射 ─────────────────────────────────────────────────────

# DeepFace / FER 输出的 7 类情绪
POSITIVE_EMOTIONS = {"happy", "surprise"}
NEGATIVE_EMOTIONS = {"angry", "fear", "disgust", "sad"}
# 各库的别名统一处理
_ALIAS = {
    "anger": "angry",
    "sadness": "sad",
    "happiness": "happy",
    "disgust": "disgust",
    "surprise": "surprise",
    "fear": "fear",
    "neutral": "neutral",
}


def _normalize_emotion(emo: str) -> str:
    return _ALIAS.get(emo.lower(), emo.lower())


# ─── Frame 级分析 ──────────────────────────────────────────────────────

def _analyze_frame_deepface(frame: np.ndarray) -> Optional[str]:
    """
    用 DeepFace 分析单帧，返回主导情绪字符串，失败返回 None。
    """
    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        # result 可能是 list（多张脸）
        if isinstance(result, list):
            result = result[0]
        dominant = result.get("dominant_emotion", None)
        return _normalize_emotion(dominant) if dominant else None
    except Exception:
        return None


def _analyze_frame_fer(detector, frame: np.ndarray) -> Optional[str]:
    """
    用 FER 分析单帧，返回主导情绪字符串，失败返回 None。
    """
    try:
        emotions = detector.detect_emotions(frame)
        if not emotions:
            return None
        # 取第一张脸，top emotion
        top_emo = detector.top_emotion(frame)
        if top_emo is None:
            return None
        emo_name, _ = top_emo
        return _normalize_emotion(emo_name)
    except Exception:
        return None


# ─── 主接口 ───────────────────────────────────────────────────────────

def extract_visual_emotions(
    video_path: Path,
    sample_fps: float = 1.0,
) -> dict:
    """
    从视频提取面部情绪统计。

    参数
    ----
    video_path  : 输入 .mp4 视频文件路径
    sample_fps  : 每秒采样帧数（默认 1.0，即每秒取 1 帧）

    返回
    ----
    dict with keys:
        file_stem          : 文件名（无扩展）
        positive_ratio     : 正面情绪帧比率
        negative_ratio     : 负面情绪帧比率
        neutral_ratio      : 中性情绪帧比率
        net_positive       : positive_ratio - negative_ratio
        frames_analyzed    : 分析的总帧数
        frames_with_face   : 检测到人脸的帧数
        method             : "deepface" / "fer" / "unavailable"
        error              : 错误信息
    """
    result = {
        "file_stem":         video_path.stem,
        "positive_ratio":    float("nan"),
        "negative_ratio":    float("nan"),
        "neutral_ratio":     float("nan"),
        "net_positive":      float("nan"),
        "frames_analyzed":   0,
        "frames_with_face":  0,
        "method":            "unavailable",
        "error":             "",
    }

    if not _CV2_OK:
        result["error"] = "opencv-python 未安装"
        return result

    if not _DEEPFACE_OK and not _FER_OK:
        if _DEEPFACE_ERR:
            result["error"] = f"deepface 不可用: {_DEEPFACE_ERR}; fer 也不可用"
        else:
            result["error"] = "deepface 和 fer 均未安装"
        return result

    if not video_path.exists():
        result["error"] = f"视频文件不存在: {video_path}"
        return result

    # 初始化 FER 检测器（若使用 fer）
    fer_detector = None
    if not _DEEPFACE_OK and _FER_OK:
        fer_detector = FER(mtcnn=False)

    method = "deepface" if _DEEPFACE_OK else "fer"
    result["method"] = method

    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result["error"] = f"无法打开视频: {video_path}"
        return result

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 25.0  # 默认帧率

        # 每隔多少帧采样一次
        frame_interval = max(1, int(round(video_fps / sample_fps)))

        emotion_counts = {"positive": 0, "negative": 0, "neutral": 0}
        frames_analyzed = 0
        frames_with_face = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frames_analyzed += 1
                if _DEEPFACE_OK:
                    emo = _analyze_frame_deepface(frame)
                else:
                    emo = _analyze_frame_fer(fer_detector, frame)

                if emo is not None:
                    frames_with_face += 1
                    if emo in POSITIVE_EMOTIONS:
                        emotion_counts["positive"] += 1
                    elif emo in NEGATIVE_EMOTIONS:
                        emotion_counts["negative"] += 1
                    else:
                        emotion_counts["neutral"] += 1

            frame_idx += 1

        result["frames_analyzed"] = frames_analyzed
        result["frames_with_face"] = frames_with_face

        if frames_with_face > 0:
            total = frames_with_face
            result["positive_ratio"] = emotion_counts["positive"] / total
            result["negative_ratio"] = emotion_counts["negative"] / total
            result["neutral_ratio"]  = emotion_counts["neutral"]  / total
            result["net_positive"]   = (
                result["positive_ratio"] - result["negative_ratio"]
            )

    except Exception as e:
        result["error"] = str(e)
    finally:
        cap.release()

    return result

if __name__ == "__main__":
    import argparse
    start_time = time.time()
    video_path = Path("上证路演视频/600025_华能水电_2017-12-04_视频1_华能水电首次公开发行A股网上路演开场致辞.mp4")  # 替换为实际视频路径

    res = extract_visual_emotions(video_path, sample_fps=1)
    print(res)
    end_time = time.time()
    print(f"分析耗时: {end_time - start_time:.2f} 秒")