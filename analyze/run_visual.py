"""
analyze/run_visual.py
=====================
批量处理路演视频，同时提取：
  1. 面部情绪（FER）    → output/visual_fer.csv      [GPU 批推理]
  2. 凝视 & 头部姿态   → output/visual_gaze.csv     [CPU MediaPipe]

并行策略（A800 优化）
--------------------
  A. 帧预读：独立线程提前将下一批视频帧读入有界队列，掩盖磁盘 I/O 等待
  B. GPU 串行：FER 在主线程顺序执行，保证显存批推理不争抢
  C. Gaze 并发：N_GAZE_WORKERS 个 CPU 线程同时处理多个视频的 MediaPipe
  D. 顺序写出：OrderedDict 保持提交顺序，队头完成即刷 CSV

典型加速比（Gaze 为瓶颈时）:
  串行: read(15s) + FER(3s) + Gaze(60s) ≈ 78s/视频
  并行: max(read+FER, Gaze/N) ≈ 18s/视频（N_GAZE_WORKERS=4）→ ~4× 加速

内存峰值（N 个 Gaze worker 同时持有帧）:
  约 (1 + N_GAZE_WORKERS + PREFETCH_Q_SIZE) 个视频的帧缓冲同时存在。
  如需降低内存压力，调小 N_GAZE_WORKERS 或 PREFETCH_Q_SIZE。

断点续算：FER / Gaze 各自独立，互不影响。
"""

import os
import sys
import queue
import threading
import collections
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

# ─── 并行配置 ─────────────────────────────────────────────────────────
N_GAZE_WORKERS  = max(1, (os.cpu_count() or 4) // 2)  # MediaPipe CPU 并发线程数
PREFETCH_Q_SIZE = 2                                     # 预读帧队列深度（控制内存）


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


# ─── 帧预读线程 ───────────────────────────────────────────────────────

def _frame_reader(tasks: list, out_q: "queue.Queue"):
    """后台线程：顺序读取视频帧，放入有界队列。队列满时阻塞，天然限速防 OOM。"""
    for video_path in tasks:
        try:
            frames, _ = read_sampled_frames(
                video_path,
                sample_fps=VISUAL_SAMPLE_FPS,
                max_long_side=FRAME_MAX_LONG_SIDE,
            )
            out_q.put((video_path, frames, None))
        except Exception as e:
            out_q.put((video_path, None, str(e)))
    out_q.put(None)  # sentinel


# ─── 主流程 ───────────────────────────────────────────────────────────

def main():
    print("═" * 62)
    print("  Visual Feature Extraction  (FER-GPU + Gaze-CPU 并行)")
    print("═" * 62)
    print(f"  采样率        : {VISUAL_SAMPLE_FPS} fps")
    print(f"  帧缓冲长边限  : {FRAME_MAX_LONG_SIDE} px")
    print(f"  GPU 批大小    : {GPU_BATCH_SIZE}")
    print(f"  GPU 设备      : {GPU_DEVICE}")
    print(f"  FER  分析     : {'ON' if RUN_FER  else 'OFF'}")
    print(f"  Gaze 分析     : {'ON' if RUN_GAZE else 'OFF'}")
    print(f"  Gaze 并发数   : {N_GAZE_WORKERS}")

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

    # ── 启动帧预读后台线程 ────────────────────────────────────────────
    frame_q: queue.Queue = queue.Queue(maxsize=PREFETCH_Q_SIZE)
    reader_thread = threading.Thread(
        target=_frame_reader, args=(tasks, frame_q), daemon=True
    )
    reader_thread.start()

    # ── Gaze CPU 线程池 ───────────────────────────────────────────────
    # 每个线程在 extract_gaze_from_frames 内部创建独立的 GazeEngine 实例，
    # 满足 MediaPipe 不跨线程共享的要求。
    gaze_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=N_GAZE_WORKERS, thread_name_prefix="gaze"
    )

    # pending: OrderedDict[stem -> (fer_result | None, gaze_future | None)]
    # 保持提交顺序，队头完成即顺序写 CSV
    pending: collections.OrderedDict = collections.OrderedDict()

    fer_header  = not FER_OUTPUT_FILE.exists()
    gaze_header = not GAZE_OUTPUT_FILE.exists()
    fer_buf:  list = []
    gaze_buf: list = []
    BATCH_SIZE = 50
    n_written = 0

    def _write_buffers(force: bool = False):
        nonlocal fer_header, gaze_header
        if fer_buf and (force or len(fer_buf) >= BATCH_SIZE):
            pd.DataFrame(fer_buf).to_csv(
                FER_OUTPUT_FILE, mode="a", header=fer_header,
                index=False, encoding="utf-8-sig",
            )
            fer_header = False
            fer_buf.clear()
        if gaze_buf and (force or len(gaze_buf) >= BATCH_SIZE):
            pd.DataFrame(gaze_buf).to_csv(
                GAZE_OUTPUT_FILE, mode="a", header=gaze_header,
                index=False, encoding="utf-8-sig",
            )
            gaze_header = False
            gaze_buf.clear()

    def _record(stem: str, fer_r, gaze_r):
        """打印摘要，放入写出缓冲区。"""
        nonlocal n_written
        n_written += 1
        if fer_r is not None:
            if fer_r.get("error"):
                print(f"  [FER  ✗] {fer_r['error']}")
            else:
                print(
                    f"  [FER  ✓] method={fer_r.get('method','?'):10s} "
                    f"faces={fer_r.get('frames_with_face',0)}/{fer_r.get('frames_analyzed',0)} "
                    f"net_pos={fer_r.get('net_positive', float('nan')):.4f}"
                )
            fer_buf.append(fer_r)
        if gaze_r is not None:
            if gaze_r.get("error"):
                print(f"  [Gaze ✗] {gaze_r['error']}")
            else:
                print(
                    f"  [Gaze ✓] camera={gaze_r.get('gaze_at_camera_ratio', float('nan')):.3f}  "
                    f"frontal={gaze_r.get('head_frontal_ratio', float('nan')):.3f}  "
                    f"yaw_mean={gaze_r.get('head_yaw_mean', float('nan')):.1f}°"
                )
            gaze_buf.append(gaze_r)
        if n_written % BATCH_SIZE == 0 or n_written == n_total:
            pct = n_written / n_total * 100
            print(f"  ──> 进度 {pct:.1f}%  ({n_written}/{n_total})\n")

    def _flush(block: bool = False):
        """
        从 pending 头部取已完成的结果写出（保持 CSV 顺序）。
        block=False：非阻塞，队头未完成则停止；
        block=True ：阻塞等待，直到清空所有 pending。
        """
        while pending:
            stem, (fer_r, gaze_f) = next(iter(pending.items()))
            if gaze_f is not None:
                if not block and not gaze_f.done():
                    break           # 队头 Gaze 未完成，暂停（保持顺序）
                gaze_r = gaze_f.result()   # 已完成或阻塞等待
            else:
                gaze_r = None
            del pending[stem]
            _record(stem, fer_r, gaze_r)
            _write_buffers()

    n_submitted = 0

    # ── 主循环：从预读队列取帧 → FER 串行（GPU）→ Gaze 并发提交（CPU）
    while True:
        item = frame_q.get()
        if item is None:
            break   # 预读线程已推送所有视频

        video_path, frames, err = item
        stem = video_path.stem
        n_submitted += 1
        print(f"[{n_submitted:>4}/{n_total}] {video_path.name}")

        stem_in_fer  = stem in done_fer
        stem_in_gaze = stem in done_gaze

        # ── 读帧失败 / 空帧：立即记录错误，不入 pending ───────────────
        if err is not None or not frames:
            err_msg = err or "视频无法读取或帧数为 0"
            fer_r   = dict(file_stem=stem, error=err_msg) if RUN_FER  and not stem_in_fer  else None
            gaze_r  = dict(file_stem=stem, error=err_msg) if RUN_GAZE and not stem_in_gaze else None
            _record(stem, fer_r, gaze_r)
            _write_buffers()
            continue

        # ── 正常处理 ──────────────────────────────────────────────────
        fer_result  = None
        gaze_future = None

        # FER：主线程顺序执行，保证单 GPU 批推理不争抢
        if RUN_FER and not stem_in_fer:
            fer_result = extract_visual_emotions_from_frames(
                frames, stem, GPU_DEVICE, GPU_BATCH_SIZE
            )

        # Gaze：提交到 CPU 线程池，与后续视频的 FER 并发运行
        # frames 此时 FER 已完成读取，Gaze worker 只读不写，无竞争
        if RUN_GAZE and not stem_in_gaze:
            gaze_future = gaze_pool.submit(extract_gaze_from_frames, frames, stem)

        pending[stem] = (fer_result, gaze_future)

        # 非阻塞冲洗：把已完成的队头写出，不阻塞主循环
        _flush(block=False)

    # ── 等待所有 Gaze 任务完成，按顺序写出剩余结果 ───────────────────
    _flush(block=True)
    _write_buffers(force=True)

    gaze_pool.shutdown(wait=False)

    print("完成！")
    if RUN_FER:
        print(f"  FER  → {FER_OUTPUT_FILE}")
    if RUN_GAZE:
        print(f"  Gaze → {GAZE_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
