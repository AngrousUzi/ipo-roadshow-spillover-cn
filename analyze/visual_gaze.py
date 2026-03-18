"""
analyze/visual_gaze.py
======================
从路演视频中提取「看镜头」相关指标（Gaze & Head-Pose Component）。

原理
----
使用 MediaPipe Face Mesh（refine_landmarks=True）同时获取：
  1. 虹膜关键点（Iris landmarks）→ 计算两眼的凝视偏移量（gaze ratio）
  2. 3D 面部关键点 → 通过 solvePnP 估计头部姿态（yaw / pitch / roll）

「看镜头」判定（两个条件需同时满足）：
  - 水平凝视偏移 |gaze_x| < GAZE_THRESHOLD_H  （眼球对准中心）
  - 头部水平旋转 |head_yaw| < HEAD_YAW_THRESH  （脸正对镜头）

API
---
  extract_gaze_features(video_path, sample_fps)
      → 完整流程（自动读取视频帧），向后兼容原始接口
  extract_gaze_from_frames(frames, stem)
      → 接受预读帧列表，供 run_visual.py 共享帧时调用（避免重复读视频）
  GazeEngine
      → 可复用的 MediaPipe FaceMesh 封装，避免每帧重建模型

输出指标（每个视频一行）
-----------------------
  file_stem              : 文件名（无扩展名）
  gaze_at_camera_ratio   : 看镜头帧占比，核心指标
  gaze_x_mean            : 水平凝视偏移均值（0=正中）
  gaze_x_std             : 水平凝视偏移标准差（越大=视线越游移）
  gaze_y_mean            : 垂直凝视偏移均值
  gaze_y_std             : 垂直凝视偏移标准差
  head_yaw_mean          : 头部水平偏转均值 (°)
  head_yaw_std           : 头部水平偏转标准差 (°)
  head_pitch_mean        : 头部俯仰均值 (°)
  head_pitch_std         : 头部俯仰标准差 (°)
  head_frontal_ratio     : 头部正面朝向帧占比
  combined_attn_ratio    : 「真正看镜头」占比（眼球+头部均对准）
  frames_analyzed        : 采样分析的帧数
  frames_with_face       : 检测到人脸的帧数
  method                 : "mediapipe" / "unavailable"
  error                  : 错误信息

阈值说明
--------
  GAZE_THRESHOLD_H = 0.15   凝视偏移绝对值阈值（眼宽归一化）
  HEAD_YAW_THRESH  = 15.0   头部水平偏转阈值（度）

依赖
----
  pip install mediapipe opencv-python numpy
"""

from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np

try:
    import cv2 as _cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False
    print("[WARNING] opencv-python 未安装。pip install opencv-python")

try:
    import mediapipe as _mp
    _MP_OK = True
except ImportError:
    _MP_OK = False
    print("[WARNING] mediapipe 未安装，gaze 模块不可用。pip install mediapipe")


# ─── 阈值常量 ──────────────────────────────────────────────────────────

GAZE_THRESHOLD_H  = 0.15   # 水平凝视偏移上限（眼宽归一化，0=正中）
GAZE_THRESHOLD_V  = 0.15   # 垂直凝视偏移上限
HEAD_YAW_THRESH   = 15.0   # 头部水平偏转上限（度）
HEAD_PITCH_THRESH = 15.0   # 头部俯仰偏转上限（度）


# ─── MediaPipe 关键点索引 ───────────────────────────────────────────────

_LEFT_IRIS  = [474, 475, 476, 477]
_RIGHT_IRIS = [469, 470, 471, 472]

_LEFT_EYE_CORNERS   = [33,  133]
_LEFT_EYE_VERTICAL  = [159, 145]
_RIGHT_EYE_CORNERS  = [362, 263]
_RIGHT_EYE_VERTICAL = [386, 374]

_POSE_LM_IDS = [1, 152, 226, 446, 57, 287]

_MODEL_POINTS = np.array([
    [0.0,    0.0,    0.0  ],
    [0.0,   -330.0, -65.0],
    [-225.0,  170.0, -135.0],
    [225.0,   170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0,  -150.0, -125.0],
], dtype=np.float64)


# ─── 工具函数 ──────────────────────────────────────────────────────────

def _lm_xy(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def _iris_center(landmarks, iris_ids: list, w: int, h: int) -> np.ndarray:
    pts = [_lm_xy(landmarks, i, w, h) for i in iris_ids]
    return np.mean(pts, axis=0)


def _eye_gaze_ratio(
    landmarks,
    iris_ids: list,
    corner_ids: list,
    vertical_ids: list,
    w: int, h: int,
) -> tuple:
    iris_c   = _iris_center(landmarks, iris_ids, w, h)
    left_pt  = _lm_xy(landmarks, corner_ids[0], w, h)
    right_pt = _lm_xy(landmarks, corner_ids[1], w, h)
    top_pt   = _lm_xy(landmarks, vertical_ids[0], w, h)
    bot_pt   = _lm_xy(landmarks, vertical_ids[1], w, h)

    eye_width  = np.linalg.norm(right_pt - left_pt) + 1e-6
    eye_height = np.linalg.norm(bot_pt   - top_pt)  + 1e-6

    gaze_x = (iris_c[0] - (left_pt[0] + right_pt[0]) / 2) / eye_width
    gaze_y = (iris_c[1] - (top_pt[1]  + bot_pt[1])   / 2) / eye_height
    return float(gaze_x), float(gaze_y)


def _estimate_head_pose(
    landmarks, w: int, h: int, cam_matrix: np.ndarray,
) -> tuple:
    import cv2
    image_pts = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in _POSE_LM_IDS],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))
    try:
        ok, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS, image_pts, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return float("nan"), float("nan"), float("nan")
    except Exception:
        return float("nan"), float("nan"), float("nan")

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw   = np.arctan2(rmat[2, 1], rmat[2, 2])
        roll  = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw   = 0.0
        roll  = np.arctan2(-rmat[1, 2], rmat[1, 1])

    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


# ─── GazeEngine：可复用 FaceMesh 封装 ────────────────────────────────

class GazeEngine:
    """
    将 MediaPipe FaceMesh 封装为可复用对象，避免逐帧重建模型。

    每个视频创建一次，处理完毕后调用 close()（或用 with 语句）。
    线程安全性：MediaPipe 不保证跨线程共享，每个线程应创建独立实例。
    """

    def __init__(
        self,
        gaze_threshold_h: float = GAZE_THRESHOLD_H,
        gaze_threshold_v: float = GAZE_THRESHOLD_V,
        head_yaw_thresh:  float = HEAD_YAW_THRESH,
    ):
        self.gaze_threshold_h = gaze_threshold_h
        self.gaze_threshold_v = gaze_threshold_v
        self.head_yaw_thresh  = head_yaw_thresh
        self._face_mesh       = None

    def _ensure_mesh(self):
        if self._face_mesh is None:
            if not _MP_OK:
                raise ImportError("mediapipe 未安装")
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return self._face_mesh

    def analyze_frame(
        self,
        frame_bgr: np.ndarray,
        cam_matrix: np.ndarray,
    ) -> Optional[dict]:
        """
        分析单帧（BGR）。
        返回 dict(gaze_x, gaze_y, yaw, pitch, at_camera) 或 None（无人脸）。
        """
        import cv2
        face_mesh = self._ensure_mesh()
        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_result = face_mesh.process(rgb)
        if not mp_result.multi_face_landmarks:
            return None

        lms = mp_result.multi_face_landmarks[0].landmark

        gx_l, gy_l = _eye_gaze_ratio(lms, _LEFT_IRIS,  _LEFT_EYE_CORNERS,  _LEFT_EYE_VERTICAL,  w, h)
        gx_r, gy_r = _eye_gaze_ratio(lms, _RIGHT_IRIS, _RIGHT_EYE_CORNERS, _RIGHT_EYE_VERTICAL, w, h)
        gaze_x = (gx_l + gx_r) / 2
        gaze_y = (gy_l + gy_r) / 2

        yaw, pitch, _ = _estimate_head_pose(lms, w, h, cam_matrix)

        gaze_ok = abs(gaze_x) < self.gaze_threshold_h and abs(gaze_y) < self.gaze_threshold_v
        head_ok = not np.isnan(yaw) and abs(yaw) < self.head_yaw_thresh

        return dict(gaze_x=gaze_x, gaze_y=gaze_y, yaw=yaw, pitch=pitch,
                    at_camera=(gaze_ok and head_ok))

    def close(self):
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass
            self._face_mesh = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ─── 核心函数：接受预读帧列表 ────────────────────────────────────────

def extract_gaze_from_frames(
    frames: list,
    stem: str,
    gaze_threshold_h: float = GAZE_THRESHOLD_H,
    gaze_threshold_v: float = GAZE_THRESHOLD_V,
    head_yaw_thresh:  float = HEAD_YAW_THRESH,
) -> dict:
    """
    从预读的帧列表提取凝视 & 头部姿态指标。

    参数
    ----
    frames : read_sampled_frames() 返回的 BGR 帧列表
    stem   : 视频文件名（无扩展名），写入 file_stem 列

    这是 run_visual.py 共享帧时优先调用的接口：
    视频只读一遍，FER（GPU）和 Gaze（CPU）在同一帧缓冲上并发运行。
    """
    nan = float("nan")
    result = dict(
        file_stem            = stem,
        gaze_at_camera_ratio = nan,
        gaze_x_mean          = nan,
        gaze_x_std           = nan,
        gaze_y_mean          = nan,
        gaze_y_std           = nan,
        head_yaw_mean        = nan,
        head_yaw_std         = nan,
        head_pitch_mean      = nan,
        head_pitch_std       = nan,
        head_frontal_ratio   = nan,
        combined_attn_ratio  = nan,
        frames_analyzed      = len(frames),
        frames_with_face     = 0,
        method               = "unavailable",
        error                = "",
    )

    if not _CV2_OK or not _MP_OK:
        result["error"] = "opencv 或 mediapipe 未安装"
        return result

    if not frames:
        result["error"] = "帧列表为空"
        return result

    # 从第一帧推断宽高
    frame_h, frame_w = frames[0].shape[:2]
    focal_len  = frame_w
    cam_matrix = np.array([
        [focal_len, 0,         frame_w / 2],
        [0,         focal_len, frame_h / 2],
        [0,         0,         1          ],
    ], dtype=np.float64)

    gaze_x_list = []
    gaze_y_list = []
    yaw_list    = []
    pitch_list  = []
    at_cam_list = []

    try:
        with GazeEngine(gaze_threshold_h, gaze_threshold_v, head_yaw_thresh) as engine:
            for frame in frames:
                fr = engine.analyze_frame(frame, cam_matrix)
                if fr is None:
                    continue
                result["frames_with_face"] += 1
                gaze_x_list.append(fr["gaze_x"])
                gaze_y_list.append(fr["gaze_y"])
                if not np.isnan(fr["yaw"]):
                    yaw_list.append(fr["yaw"])
                    pitch_list.append(fr["pitch"])
                at_cam_list.append(1 if fr["at_camera"] else 0)
    except Exception as e:
        result["error"] = str(e)
        return result

    result["method"] = "mediapipe"

    if gaze_x_list:
        result["gaze_x_mean"] = float(np.mean(gaze_x_list))
        result["gaze_x_std"]  = float(np.std(gaze_x_list))
        result["gaze_y_mean"] = float(np.mean(gaze_y_list))
        result["gaze_y_std"]  = float(np.std(gaze_y_list))

    if at_cam_list:
        result["gaze_at_camera_ratio"] = float(np.mean(at_cam_list))
        result["combined_attn_ratio"]  = result["gaze_at_camera_ratio"]

    if yaw_list:
        result["head_yaw_mean"]    = float(np.mean(yaw_list))
        result["head_yaw_std"]     = float(np.std(yaw_list))
        result["head_pitch_mean"]  = float(np.mean(pitch_list))
        result["head_pitch_std"]   = float(np.std(pitch_list))
        frontal = [1 if abs(y) < head_yaw_thresh else 0 for y in yaw_list]
        result["head_frontal_ratio"] = float(np.mean(frontal))

    return result


# ─── 向后兼容接口（完整流程）─────────────────────────────────────────

def extract_gaze_features(
    video_path: Path,
    sample_fps: float = 1.0,
    gaze_threshold_h: float = GAZE_THRESHOLD_H,
    gaze_threshold_v: float = GAZE_THRESHOLD_V,
    head_yaw_thresh:  float = HEAD_YAW_THRESH,
    max_long_side: int = 720,
) -> dict:
    """
    从视频文件提取凝视与头部姿态指标（向后兼容入口）。
    内部调用 visual_fer.read_sampled_frames() + extract_gaze_from_frames()。
    """
    nan = float("nan")
    empty = dict(
        file_stem=video_path.stem,
        gaze_at_camera_ratio=nan, gaze_x_mean=nan, gaze_x_std=nan,
        gaze_y_mean=nan, gaze_y_std=nan,
        head_yaw_mean=nan, head_yaw_std=nan,
        head_pitch_mean=nan, head_pitch_std=nan,
        head_frontal_ratio=nan, combined_attn_ratio=nan,
        frames_analyzed=0, frames_with_face=0,
        method="unavailable", error="",
    )

    if not video_path.exists():
        empty["error"] = f"视频文件不存在: {video_path}"
        return empty

    try:
        from visual_fer import read_sampled_frames
        frames, _ = read_sampled_frames(video_path, sample_fps, max_long_side)
    except Exception as e:
        empty["error"] = str(e)
        return empty

    return extract_gaze_from_frames(
        frames, video_path.stem,
        gaze_threshold_h=gaze_threshold_h,
        gaze_threshold_v=gaze_threshold_v,
        head_yaw_thresh=head_yaw_thresh,
    )


# ─── 命令行快速测试 ───────────────────────────────────────────────────

if __name__ == "__main__":
    import time, sys
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test.mp4")
    print(f"分析视频：{video_path}")

    start = time.time()
    res = extract_gaze_features(video_path, sample_fps=1.0)
    elapsed = time.time() - start

    print("\n[ 结果 ]")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k:<26} = {v:.4f}")
        else:
            print(f"  {k:<26} = {v}")
    print(f"\n提取耗时: {elapsed:.2f} 秒")
