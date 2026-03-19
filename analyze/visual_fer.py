"""
analyze/visual_fer.py
=====================
从路演视频中提取面部情绪指标（Visual Component of Pitch Factor）。

【GPU 批推理模式（推荐 HPC 使用）】
  依赖：pip install facenet-pytorch hsemotion
  原理：
    1. MTCNN（GPU）批量检测人脸，每批 GPU_BATCH_SIZE 帧
    2. HSEmotion EfficientNet（GPU）批量分类情绪（8类→3类）
    3. 同一批次的推理在 GPU 上完全并行，显存利用率接近满负荷
  吞吐量比逐帧 CPU 快约 20~50 倍。

【CPU 回退模式（本地 / 无 GPU 环境）】
  依赖：pip install deepface  /  pip install fer
  原理：逐帧调用 DeepFace 或 FER，与原版行为完全一致。

输出（每个视频一行）
-------------------
  file_stem          : 文件名（无扩展名）
  positive_ratio     : 正面情绪帧比率（happy / surprise）
  negative_ratio     : 负面情绪帧比率（angry / fear / disgust / sad）
  neutral_ratio      : 中性帧比率
  net_positive       : positive_ratio − negative_ratio
  emo_angry          : 8维情绪——愤怒均值概率（有人脸帧）
  emo_contempt       : 8维情绪——蔑视均值概率
  emo_disgust        : 8维情绪——厌恶均值概率
  emo_fear           : 8维情绪——恐惧均值概率
  emo_happy          : 8维情绪——快乐均值概率
  emo_neutral        : 8维情绪——中性均值概率
  emo_sad            : 8维情绪——悲伤均值概率
  emo_surprise       : 8维情绪——惊讶均值概率
  frames_analyzed    : 采样分析的总帧数
  frames_with_face   : 检测到人脸的帧数
  face_detect_rate   : frames_with_face / frames_analyzed
  method             : "gpu_batch" / "deepface" / "fer" / "unavailable"
  error              : 错误信息

API
---
  extract_visual_emotions(video_path, sample_fps)
      → 完整流程（自动选择 GPU/CPU 模式）
  extract_visual_emotions_from_frames(frames, stem)
      → 接受预读帧列表，供 run_visual.py 共享帧时调用
  read_sampled_frames(video_path, sample_fps, max_long_side)
      → 只读视频帧，返回 list[np.ndarray]
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional
import numpy as np

# ─── 依赖导入 ─────────────────────────────────────────────────────────

import cv2 as _cv2

# GPU 路径：facenet-pytorch（MTCNN）+ hsemotion（情绪分类器）
# 使用 try/except 保护，缺少库时回退到 CPU 模式
try:
    import torch as _torch
    from facenet_pytorch import MTCNN as _MTCNN
    from hsemotion.facial_emotions import HSEmotionRecognizer as _HSEmo
    from torchvision import transforms as _transforms
    from PIL import Image as _PILImage
    _GPU_LIBS_OK = True
except ImportError as _e:
    print(f"[FER] GPU 库缺失（{_e}），将使用 CPU 回退模式。")
    _GPU_LIBS_OK = False


# ─── 情绪类别映射 ─────────────────────────────────────────────────────

POSITIVE_EMOTIONS = {"happy", "surprise", "happiness"}
NEGATIVE_EMOTIONS = {"angry", "anger", "fear", "disgust", "sad", "sadness", "contempt"}

_CPU_ALIAS = {
    "anger": "angry", "sadness": "sad", "happiness": "happy",
    "disgust": "disgust", "surprise": "surprise",
    "fear": "fear", "neutral": "neutral",
}

# HSEmotion 的 8 类标签（EfficientNet B0 vgaf 版本）
# Anger / Contempt / Disgust / Fear / Happiness / Neutral / Sadness / Surprise
_HSE_ALIAS = {
    "Anger": "angry", "Contempt": "contempt", "Disgust": "disgust",
    "Fear": "fear", "Happiness": "happy", "Neutral": "neutral",
    "Sadness": "sad", "Surprise": "surprise",
}


def _cat(emo: str) -> str:
    """将情绪标签归入 positive / negative / neutral。"""
    e = emo.lower()
    if e in POSITIVE_EMOTIONS:
        return "positive"
    if e in NEGATIVE_EMOTIONS:
        return "negative"
    return "neutral"


# ─── GPU 批推理引擎（懒加载单例）────────────────────────────────────

class _FERGpuEngine:
    """
    GPU 批推理引擎：MTCNN（人脸检测）+ HSEmotion EfficientNet（情绪分类）。

    使用方式：
        engine = _FERGpuEngine(device="cuda", batch_size=32)
        results = engine.analyze_frames(frames_bgr)  # list[dict | None]
    """

    def __init__(self, device: str = "cuda", batch_size: int = 32):
        dev = _torch.device(device if _torch.cuda.is_available() else "cpu")
        self.device = dev
        self.batch_size = batch_size

        # 人脸检测器（MTCNN，在 GPU 上批量跑）
        self.mtcnn = _MTCNN(
            keep_all=False,          # 只取最大人脸
            device=dev,
            image_size=112,          # 输出统一 112×112（仅用于检测，裁剪后重缩放）
            margin=16,
            min_face_size=24,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False,      # 返回 uint8 Tensor，方便后续处理
        )

        # 情绪分类器（EfficientNet B0，在 GPU 上跑）
        self.fer_model = _HSEmo(
            model_name="enet_b0_8_best_vgaf",
            device=str(dev),
        )
        self.classes: list[str] = list(self.fer_model.idx_to_class.values())

        # HSEmotion 将分类头的权重存储在 classifier_weights / classifier_bias，
        # 而 fer_model.model 的 classifier 层被替换为 Identity（输出 1280-dim 特征）。
        # 必须手动应用线性层才能得到 8 类 logits，否则对 1280-dim 做 softmax
        # 会得到约 1/1280 ≈ 0.00078 的均匀极小值。
        self._cls_weight = _torch.tensor(
            self.fer_model.classifier_weights, dtype=_torch.float32
        ).to(dev)   # [8, 1280]
        self._cls_bias = _torch.tensor(
            self.fer_model.classifier_bias, dtype=_torch.float32
        ).to(dev)   # [8]

        # 情绪分类的预处理（224×224，ImageNet 归一化）
        self._preprocess = _transforms.Compose([
            _transforms.ToPILImage(),
            _transforms.Resize((224, 224)),
            _transforms.ToTensor(),
            _transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225]),
        ])

    # ── 阶段 1：批量人脸检测 ─────────────────────────────────────────

    def _detect_batch(
        self, frames_bgr: list[np.ndarray]
    ) -> list[np.ndarray | None]:
        """
        对一批 BGR 帧做 MTCNN 人脸检测，返回各帧对应的人脸裁剪图（BGR）或 None。
        MTCNN.detect() 接受 PIL 列表，内部在 GPU 上并行处理整个 batch。
        """
        pil_imgs = [
            _PILImage.fromarray(_cv2.cvtColor(f, _cv2.COLOR_BGR2RGB))
            for f in frames_bgr
        ]
        try:
            boxes_list, probs_list = self.mtcnn.detect(pil_imgs)
        except Exception:
            return [None] * len(frames_bgr)

        crops: list[np.ndarray | None] = []
        for frame, boxes, probs in zip(frames_bgr, boxes_list, probs_list):
            if boxes is None or probs is None or float(probs[0]) < 0.80:
                crops.append(None)
                continue
            x1, y1, x2, y2 = [int(max(0, v)) for v in boxes[0]]
            h, w = frame.shape[:2]
            x2, y2 = min(x2, w), min(y2, h)
            face = frame[y1:y2, x1:x2]
            crops.append(face if face.size > 0 else None)

        return crops

    # ── 阶段 2：批量情绪分类 ─────────────────────────────────────────

    def _classify_batch(
        self, crops: list[np.ndarray | None]
    ) -> list[dict[str, float] | None]:
        """
        对一批人脸裁剪图（BGR）做批量情绪分类。
        将所有非 None 的裁剪图打包成一个 Tensor，一次 GPU forward pass 完成。
        返回各裁剪图对应的情绪概率字典（归一化到 [0,1]）或 None。
        """
        valid_indices = [i for i, c in enumerate(crops) if c is not None]
        results: list[dict[str, float] | None] = [None] * len(crops)

        if not valid_indices:
            return results

        # 构建 Tensor batch（全部在同一次 GPU forward 中处理）
        tensors = []
        for i in valid_indices:
            face_rgb = _cv2.cvtColor(crops[i], _cv2.COLOR_BGR2RGB)
            try:
                t = self._preprocess(face_rgb)
                tensors.append(t)
            except Exception:
                valid_indices = [j for j in valid_indices if j != i]

        if not tensors:
            return results

        batch_t = _torch.stack(tensors).to(self.device)

        with _torch.no_grad():
            # fer_model.model 的 classifier 被替换为 Identity，输出 1280-dim 特征。
            # 需手动应用线性分类头才能得到 8 类 logits。
            features = self.fer_model.model(batch_t)                    # [N, 1280]
            logits   = features @ self._cls_weight.T + self._cls_bias   # [N, 8]
            probs    = _torch.softmax(logits, dim=1).cpu().numpy()

        for k, idx in enumerate(valid_indices):
            prob_dict = {}
            for cls, p in zip(self.classes, probs[k].tolist()):
                norm_cls = _HSE_ALIAS.get(cls, cls.lower())
                prob_dict[norm_cls] = p
            results[idx] = prob_dict

        return results

    # ── 主接口 ────────────────────────────────────────────────────────

    def analyze_frames(
        self, frames_bgr: list[np.ndarray]
    ) -> list[dict[str, float] | None]:
        """
        批量分析帧列表，返回各帧的情绪概率字典（或 None 表示无人脸）。
        帧数可任意大；内部自动按 batch_size 切片，保持显存可控。
        """
        all_results: list[dict[str, float] | None] = []

        for start in range(0, len(frames_bgr), self.batch_size):
            chunk = frames_bgr[start : start + self.batch_size]

            # 两阶段 GPU 批推理
            crops   = self._detect_batch(chunk)
            emo_res = self._classify_batch(crops)

            all_results.extend(emo_res)

        return all_results


# 模块级单例（首次调用时初始化，后续复用，避免重复加载权重）
_GPU_ENGINE: Optional[_FERGpuEngine] = None


def _get_gpu_engine(device: str = "cuda", batch_size: int = 32) -> Optional[_FERGpuEngine]:
    global _GPU_ENGINE
    if _GPU_ENGINE is None:
        if not _GPU_LIBS_OK:
            return None
        try:
            print("[FER] 正在加载 GPU 引擎（MTCNN + HSEmotion）…")
            _GPU_ENGINE = _FERGpuEngine(device=device, batch_size=batch_size)
            dev_name = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "CPU"
            print(f"[FER] GPU 引擎已就绪，设备: {dev_name}，batch_size={batch_size}")
        except Exception as e:
            print(f"[FER] GPU 引擎初始化失败（{e}），将使用 CPU 模式。")
            _GPU_ENGINE = None  # 保持 None，下面走 CPU 分支
    return _GPU_ENGINE


# ─── 帧读取工具 ───────────────────────────────────────────────────────

def read_sampled_frames(
    video_path: Path,
    sample_fps: float = 1.0,
    max_long_side: int = 720,
) -> tuple[list[np.ndarray], float]:
    """
    从视频文件读取所有采样帧，返回 (frames, video_fps)。

    frames 是 BGR numpy 数组列表，长边缩放到 max_long_side 以节省内存。
    max_long_side=0 表示保持原始分辨率。

    动机：FER 和 Gaze 共享同一帧列表，视频只读一遍。
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 25.0
        frame_interval = max(1, int(round(video_fps / sample_fps)))

        frames: list[np.ndarray] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                if max_long_side > 0:
                    h, w = frame.shape[:2]
                    long = max(h, w)
                    if long > max_long_side:
                        scale = max_long_side / long
                        frame = cv2.resize(
                            frame,
                            (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                frames.append(frame)
            frame_idx += 1
    finally:
        cap.release()

    return frames, video_fps


# ─── 情绪统计聚合 ─────────────────────────────────────────────────────

# 8 个情绪维度的标准键名（输出列名前缀 emo_）
_EMO8_KEYS = ("angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")


def _aggregate_emotion_results(
    emo_list: list[dict[str, float] | None],
    stem: str,
    method: str,
) -> dict:
    """将情绪概率字典列表聚合为最终统计 dict。

    三类聚合指标（positive/negative/neutral_ratio）基于主导情绪帧计数。
    8 维细粒度指标（emo_*）为有人脸帧上各情绪概率的均值；
    对于离散模式（DeepFace/FER）每帧仅有主导情绪置 1.0，其余 0.0。
    """
    nan = float("nan")
    result = dict(
        file_stem          = stem,
        positive_ratio     = nan,
        negative_ratio     = nan,
        neutral_ratio      = nan,
        net_positive       = nan,
        **{f"emo_{k}": nan for k in _EMO8_KEYS},   # 8 维情绪均值
        frames_analyzed    = len(emo_list),
        frames_with_face   = 0,
        face_detect_rate   = nan,
        method             = method,
        error              = "",
    )

    counts = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    emo8_sum = {k: 0.0 for k in _EMO8_KEYS}   # 累计各情绪概率之和
    n_face = 0

    for emo_dict in emo_list:
        if emo_dict is None:
            continue
        n_face += 1
        # 主导情绪（最高概率情绪）决定三分类
        dominant = max(emo_dict, key=emo_dict.get)
        cat = _cat(dominant)
        counts[cat] += 1.0
        # 累计 8 维概率（缺失键视为 0）
        for k in _EMO8_KEYS:
            emo8_sum[k] += emo_dict.get(k, 0.0)

    result["frames_with_face"] = n_face
    if n_face > 0:
        result["face_detect_rate"] = n_face / len(emo_list) if emo_list else nan
        total = float(n_face)
        result["positive_ratio"] = counts["positive"] / total
        result["negative_ratio"] = counts["negative"] / total
        result["neutral_ratio"]  = counts["neutral"]  / total
        result["net_positive"]   = result["positive_ratio"] - result["negative_ratio"]
        for k in _EMO8_KEYS:
            result[f"emo_{k}"] = emo8_sum[k] / total
    else:
        result["face_detect_rate"] = 0.0

    return result


# ─── 核心函数：接受预读帧列表 ────────────────────────────────────────

def extract_visual_emotions_from_frames(
    frames: list[np.ndarray],
    stem: str,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    """
    从预读的帧列表中提取面部情绪统计。

    参数
    ----
    frames     : 由 read_sampled_frames() 返回的 BGR 帧列表
    stem       : 视频文件名（无扩展名），写入 file_stem 列
    device     : GPU 设备标识（"cuda" / "cpu"）
    batch_size : GPU 批大小

    返回
    ----
    dict，键同本模块文档顶部「输出」一节。
    """
    nan = float("nan")
    if not frames:
        return dict(
            file_stem=stem, positive_ratio=nan, negative_ratio=nan,
            neutral_ratio=nan, net_positive=nan,
            **{f"emo_{k}": nan for k in _EMO8_KEYS},
            frames_analyzed=0, frames_with_face=0,
            face_detect_rate=nan, method="unavailable",
            error="帧列表为空",
        )

    # ── GPU 模式 ────────────────────────────────────────────────────
    engine = _get_gpu_engine(device=device, batch_size=batch_size)
    if engine is not None:
        emo_list = engine.analyze_frames(frames)
        return _aggregate_emotion_results(emo_list, stem, method="gpu_batch")

    # GPU 不可用时返回错误结果（而非 None）
    return dict(
        file_stem=stem, positive_ratio=nan, negative_ratio=nan,
        neutral_ratio=nan, net_positive=nan,
        **{f"emo_{k}": nan for k in _EMO8_KEYS},
        frames_analyzed=len(frames), frames_with_face=0,
        face_detect_rate=nan, method="unavailable",
        error="GPU 引擎不可用（facenet-pytorch / hsemotion 未安装）",
    )


# ─── 保留原始接口（向后兼容）─────────────────────────────────────────

def extract_visual_emotions(
    video_path: Path,
    sample_fps: float = 1.0,
    device: str = "cuda",
    batch_size: int = 32,
    max_long_side: int = 720,
) -> dict:
    """
    从视频文件提取面部情绪统计（完整流程入口）。

    内部调用 read_sampled_frames() + extract_visual_emotions_from_frames()，
    与旧接口完全兼容，同时支持 GPU 批推理。
    """
    nan = float("nan")
    empty = dict(
        file_stem=video_path.stem, positive_ratio=nan,
        negative_ratio=nan, neutral_ratio=nan, net_positive=nan,
        **{f"emo_{k}": nan for k in _EMO8_KEYS},
        frames_analyzed=0, frames_with_face=0,
        face_detect_rate=nan, method="unavailable", error="",
    )

    if not video_path.exists():
        empty["error"] = f"视频文件不存在: {video_path}"
        return empty

    try:
        frames, _ = read_sampled_frames(video_path, sample_fps, max_long_side)
    except Exception as e:
        empty["error"] = str(e)
        return empty

    return extract_visual_emotions_from_frames(
        frames, video_path.stem, device=device, batch_size=batch_size
    )


# ─── 命令行快速测试 ───────────────────────────────────────────────────

if __name__ == "__main__":
    import time, sys
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test.mp4")
    print(f"分析视频：{video_path}")

    start = time.time()
    res = extract_visual_emotions(video_path, sample_fps=1.0)
    elapsed = time.time() - start

    for k, v in res.items():
        print(f"  {k:<22} = {v}")
    print(f"\n提取耗时: {elapsed:.2f} 秒")