"""
analyze/run_visual.py
=====================
批量处理路演视频，同时提取：
  1. 面部情绪（FER）    → output/visual_fer.csv      [GPU 批推理]
  2. 凝视 & 头部姿态   → output/visual_gaze.csv     [CPU MediaPipe]

GPU 利用优化策略
----------------
  A. 帧共享：每个视频只读一遍，FER 和 Gaze 共用同一帧缓冲
  B. 批推理：GPU 每次接收 GPU_BATCH_SIZE 帧，一次 forward pass 处理全部
  C. 并发执行：FER（GPU 线程）和 Gaze（CPU 线程）对同一帧缓冲并发运行

运行时序（单个视频）:
  [主线程] 读帧 → frames（列表，长边≤720px）
         ├─ [GPU Thread]  FER 批推理（MTCNN + EfficientNet）
         └─ [CPU Thread]  Gaze 逐帧（MediaPipe FaceMesh）
             ↓ 两者均完成后
  [主线程] 写 CSV，处理下一个视频

断点续算：FER / Gaze 各自独立，互不影响。
串行视频处理（GPU 已被批推理占满，视频级并行反而引发争抢）。

依赖
----
  GPU 模式 : pip install facenet-pytorch hsemotion torchvision
  Gaze     : pip install mediapipe
  基础     : pip install opencv-python pandas numpy
"""

import os
import sys
import concurrent.futures
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PLATFORM_LIST, OUTPUT_DIR, PROJECT_ROOT,
    VISUAL_SAMPLE_FPS, GPU_DEVICE, GPU_BATCH_SIZE, FRAME_MAX_LONG_SIDE,
    get_video_dir,
)
from visual_fer  import read_sampled_frames, extract_visual_emotions_from_frames
from visual_gaze import extract_gaze_from_frames

# ─── 输出文件 ─────────────────────────────────────────────────────────
FER_OUTPUT_FILE   = OUTPUT_DIR / "visual_fer.csv"
GAZE_OUTPUT_FILE  = OUTPUT_DIR / "visual_gaze.csv"
INDEXED_VIDEO_DIR = PROJECT_ROOT / "路演视频"

# ─── 运行开关 ─────────────────────────────────────────────────────────
RUN_FER  = True    # 是否运行表情识别（GPU）
RUN_GAZE = True    # 是否运行凝视分析（CPU）


# ─── 任务收集 ─────────────────────────────────────────────────────────

def _collect_index_tasks(done_fer: set, done_gaze: set) -> list:
    """收集 路演视频/ 下所有待处理视频（两个输出任意一个缺失即列入任务）。"""
    tasks = []
    video_dir = INDEXED_VIDEO_DIR
    if not video_dir.exists():
        print(f"[SKIP] 目录不存在: {video_dir}")
        return tasks
    for mp4 in sorted(video_dir.glob("*.mp4")):
        need_fer  = RUN_FER  and mp4.stem not in done_fer
        need_gaze = RUN_GAZE and mp4.stem not in done_gaze
        if need_fer or need_gaze:
            tasks.append(mp4)
    return tasks


def _collect_tasks(done_fer: set, done_gaze: set) -> list:
    """收集所有平台目录下的待处理视频。"""
    tasks = []
    for platform in PLATFORM_LIST:
        video_dir = get_video_dir(platform)
        if not video_dir.exists():
            continue
        for mp4 in sorted(video_dir.glob("*.mp4")):
            need_fer  = RUN_FER  and mp4.stem not in done_fer
            need_gaze = RUN_GAZE and mp4.stem not in done_gaze
            if need_fer or need_gaze:
                tasks.append(mp4)
    return tasks


# ─── 单视频处理 ───────────────────────────────────────────────────────

def _process_video(
    video_path: Path,
    stem_in_fer:  bool,
    stem_in_gaze: bool,
) -> tuple:
    """
    处理单个视频，返回 (fer_result | None, gaze_result | None)。

    步骤：
      1. 读取所有采样帧（一次 I/O）
      2. 用 ThreadPoolExecutor 同时提交 FER（GPU）和 Gaze（CPU）
      3. 等待两者完成后返回
    """
    # 步骤 1：读帧（共享）
    try:
        frames, _ = read_sampled_frames(
            video_path,
            sample_fps=VISUAL_SAMPLE_FPS,
            max_long_side=FRAME_MAX_LONG_SIDE,
        )
        n_frames = len(frames)
    except Exception as e:
        err = str(e)
        fer_fail  = dict(file_stem=video_path.stem, error=err) if not stem_in_fer  else None
        gaze_fail = dict(file_stem=video_path.stem, error=err) if not stem_in_gaze else None
        return fer_fail, gaze_fail

    if n_frames == 0:
        err = "视频无法读取或帧数为 0"
        return (dict(file_stem=video_path.stem, error=err) if not stem_in_fer  else None,
                dict(file_stem=video_path.stem, error=err) if not stem_in_gaze else None)

    # 步骤 2：并发推理
    fer_future  = None
    gaze_future = None

    # 最多 2 个并发：GPU 线程（FER）+ CPU 线程（Gaze）
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        if RUN_FER and not stem_in_fer:
            fer_future = pool.submit(
                extract_visual_emotions_from_frames,
                frames,
                video_path.stem,
                GPU_DEVICE,
                GPU_BATCH_SIZE,
            )
        if RUN_GAZE and not stem_in_gaze:
            gaze_future = pool.submit(
                extract_gaze_from_frames,
                frames,
                video_path.stem,
            )

    fer_result  = fer_future.result()  if fer_future  else None
    gaze_result = gaze_future.result() if gaze_future else None

    return fer_result, gaze_result


# ─── 主流程 ───────────────────────────────────────────────────────────

def main():
    print("═" * 62)
    print("  Visual Feature Extraction  (FER-GPU + Gaze-CPU 并发)")
    print("═" * 62)
    print(f"  采样率        : {VISUAL_SAMPLE_FPS} fps")
    print(f"  帧缓冲长边限  : {FRAME_MAX_LONG_SIDE} px")
    print(f"  GPU 批大小    : {GPU_BATCH_SIZE}")
    print(f"  GPU 设备      : {GPU_DEVICE}")
    print(f"  FER  分析     : {'ON' if RUN_FER  else 'OFF'}")
    print(f"  Gaze 分析     : {'ON' if RUN_GAZE else 'OFF'}")

    # ── 断点续算 ─────────────────────────────────────────────────────
    done_fer: set = set()
    if RUN_FER and FER_OUTPUT_FILE.exists():
        df = pd.read_csv(FER_OUTPUT_FILE, usecols=["file_stem"])
        done_fer = set(df["file_stem"].tolist())
        print(f"\n[FER]  已有 {len(done_fer)} 条，跳过已处理文件。")

    done_gaze: set = set()
    if RUN_GAZE and GAZE_OUTPUT_FILE.exists():
        df = pd.read_csv(GAZE_OUTPUT_FILE, usecols=["file_stem"])
        done_gaze = set(df["file_stem"].tolist())
        print(f"[Gaze] 已有 {len(done_gaze)} 条，跳过已处理文件。")

    # tasks = _collect_tasks(done_fer, done_gaze)
    tasks = _collect_index_tasks(done_fer, done_gaze)
    n_total = len(tasks)
    print(f"\n待处理文件数：{n_total}\n")

    if not tasks:
        print("无待处理文件，退出。")
        return

    fer_header  = not FER_OUTPUT_FILE.exists()
    gaze_header = not GAZE_OUTPUT_FILE.exists()
    fer_buf:  list = []
    gaze_buf: list = []
    BATCH_SIZE = 50

    for i, video_path in enumerate(tasks, 1):
        stem = video_path.stem
        print(f"[{i:>4}/{n_total}] {video_path.name}")

        stem_in_fer  = stem in done_fer
        stem_in_gaze = stem in done_gaze

        fer_result, gaze_result = _process_video(
            video_path, stem_in_fer, stem_in_gaze
        )

        # ── 打印摘要 ───────────────────────────────────────────────
        if fer_result is not None:
            if fer_result.get("error"):
                print(f"  [FER  ✗] {fer_result['error']}")
            else:
                print(
                    f"  [FER  ✓] method={fer_result.get('method','?'):10s} "
                    f"faces={fer_result.get('frames_with_face',0)}/{fer_result.get('frames_analyzed',0)} "
                    f"net_pos={fer_result.get('net_positive', float('nan')):.4f}"
                )
            fer_buf.append(fer_result)

        if gaze_result is not None:
            if gaze_result.get("error"):
                print(f"  [Gaze ✗] {gaze_result['error']}")
            else:
                print(
                    f"  [Gaze ✓] camera={gaze_result.get('gaze_at_camera_ratio', float('nan')):.3f}  "
                    f"frontal={gaze_result.get('head_frontal_ratio', float('nan')):.3f}  "
                    f"yaw_mean={gaze_result.get('head_yaw_mean', float('nan')):.1f}°"
                )
            gaze_buf.append(gaze_result)

        # ── 批量写入 CSV ───────────────────────────────────────────
        if len(fer_buf) >= BATCH_SIZE or (i == n_total and fer_buf):
            pd.DataFrame(fer_buf).to_csv(
                FER_OUTPUT_FILE, mode="a", header=fer_header,
                index=False, encoding="utf-8-sig",
            )
            fer_header = False
            fer_buf = []

        if len(gaze_buf) >= BATCH_SIZE or (i == n_total and gaze_buf):
            pd.DataFrame(gaze_buf).to_csv(
                GAZE_OUTPUT_FILE, mode="a", header=gaze_header,
                index=False, encoding="utf-8-sig",
            )
            gaze_header = False
            gaze_buf = []

        if i % BATCH_SIZE == 0 or i == n_total:
            pct = i / n_total * 100
            print(f"  ──> 进度 {pct:.1f}%  ({i}/{n_total})\n")

    print("完成！")
    if RUN_FER:
        print(f"  FER  → {FER_OUTPUT_FILE}")
    if RUN_GAZE:
        print(f"  Gaze → {GAZE_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
