"""
analyze/run_gaze.py
===================
批量提取凝视 & 头部姿态（Gaze）→ output/visual_gaze.csv

专为 SJTU HPC CPU 节点优化（64 核）：
  - N_GAZE_WORKERS 自动占满所有可用核
  - N_READERS 个线程并行解码视频
  - 任一视频完成即写出，无队头阻塞
  - 无 GPU 依赖，纯 CPU 运行
  - 断点续算

运行（SJTU HPC）：
  python run_gaze.py

预期耗时（64 核，2474 个视频，~67s/视频）：
  2474 × 67s / 60 workers ≈ 46 分钟
"""

import os
import sys
import queue
import threading
import concurrent.futures
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, VISUAL_SAMPLE_FPS, FRAME_MAX_LONG_SIDE, INDEX_VIDEO_DIR
from visual_fer  import read_sampled_frames
from visual_gaze import extract_gaze_from_frames, YAW_THRESHOLDS

# ─── 路径 & 参数 ──────────────────────────────────────────────────────
INDEXED_VIDEO_DIR = INDEX_VIDEO_DIR
GAZE_OUTPUT_FILE  = OUTPUT_DIR / "visual_gaze.csv"

N_READERS      = 4                                       # 并行视频解码线程
N_GAZE_WORKERS = max(1, (os.cpu_count() or 8) - N_READERS - 1)  # 占满剩余核
PREFETCH_Q_SIZE = N_READERS * 4


# ─── 多路帧预读 ───────────────────────────────────────────────────────

def _reader_worker(task_q: queue.Queue, frame_q: queue.Queue,
                   remaining: list, lock: threading.Lock):
    while True:
        try:
            video_path = task_q.get(timeout=1)
        except queue.Empty:
            continue
        if video_path is None:
            break
        try:
            frames, _ = read_sampled_frames(
                video_path, sample_fps=VISUAL_SAMPLE_FPS,
                max_long_side=FRAME_MAX_LONG_SIDE,
            )
            frame_q.put((video_path, frames, None))
        except Exception as e:
            frame_q.put((video_path, None, str(e)))
    with lock:
        remaining[0] -= 1
        if remaining[0] == 0:
            frame_q.put(None)


# ─── 主流程 ───────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("  Gaze  (CPU HPC)  →  visual_gaze.csv")
    print("═" * 60)
    print(f"  视频目录    : {INDEXED_VIDEO_DIR}")
    print(f"  CPU 核数    : {os.cpu_count()}")
    print(f"  Gaze worker : {N_GAZE_WORKERS}")
    print(f"  Reader 线程 : {N_READERS}")
    print(f"  Yaw 阈值集  : {YAW_THRESHOLDS}°")

    # ── 断点续算 ─────────────────────────────────────────────────────
    done: set = set()
    if GAZE_OUTPUT_FILE.exists():
        df = pd.read_csv(GAZE_OUTPUT_FILE, usecols=["file_stem"])
        done = set(df["file_stem"].tolist())
        print(f"  已完成      : {len(done)} 条，跳过。")

    if not INDEXED_VIDEO_DIR.exists():
        print(f"[ERROR] 目录不存在: {INDEXED_VIDEO_DIR}")
        return

    tasks = [mp4 for mp4 in sorted(INDEXED_VIDEO_DIR.glob("*.mp4"))
             if mp4.stem not in done]
    n_total = len(tasks)
    print(f"  待处理      : {n_total} 个\n")
    if not tasks:
        print("无待处理文件，退出。")
        return

    # ── 启动多路预读 ──────────────────────────────────────────────────
    task_q:  queue.Queue = queue.Queue()
    frame_q: queue.Queue = queue.Queue(maxsize=PREFETCH_Q_SIZE)
    for t in tasks:
        task_q.put(t)
    for _ in range(N_READERS):
        task_q.put(None)

    remaining = [N_READERS]
    lock = threading.Lock()
    for _ in range(N_READERS):
        threading.Thread(
            target=_reader_worker,
            args=(task_q, frame_q, remaining, lock),
            daemon=True,
        ).start()

    # ── Gaze 线程池 ───────────────────────────────────────────────────
    gaze_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=N_GAZE_WORKERS, thread_name_prefix="gaze"
    )

    pending: dict = {}   # stem -> Future
    header = not GAZE_OUTPUT_FILE.exists()
    buf: list = []
    WRITE_BATCH = 20
    n_done = 0

    def _flush(block: bool = False):
        """冲洗任意已完成的 Gaze future，不要求顺序。"""
        nonlocal n_done, header
        while pending:
            ready_stem = next(
                (s for s, f in pending.items() if f.done()), None
            )
            if ready_stem is None:
                if not block:
                    break
                concurrent.futures.wait(
                    list(pending.values()),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                continue

            row = pending.pop(ready_stem).result()
            n_done += 1

            if row.get("error"):
                print(f"  [✗] {ready_stem}: {row['error'][:80]}")
            else:
                t_ref = YAW_THRESHOLDS[1]   # 10°
                print(
                    f"  [✓] {ready_stem}  "
                    f"camera@{t_ref}°={row.get(f'gaze_at_camera_ratio_{t_ref}', float('nan')):.3f}  "
                    f"frontal@{t_ref}°={row.get(f'head_frontal_ratio_{t_ref}', float('nan')):.3f}  "
                    f"yaw={row.get('head_yaw_mean', float('nan')):.1f}°"
                )
            buf.append(row)

            if len(buf) >= WRITE_BATCH or n_done == n_total:
                pd.DataFrame(buf).to_csv(
                    GAZE_OUTPUT_FILE, mode="a", header=header,
                    index=False, encoding="utf-8-sig",
                )
                header = False
                buf.clear()
                print(f"  ──> {n_done}/{n_total}  ({n_done/n_total*100:.1f}%)\n")

    # ── 主循环：取帧 → 提交 Gaze ──────────────────────────────────────
    n_submitted = 0
    while True:
        item = frame_q.get()
        if item is None:
            break

        video_path, frames, err = item
        stem = video_path.stem
        n_submitted += 1
        print(f"[{n_submitted:>4}/{n_total}] {video_path.name}")

        if err is not None or not frames:
            msg = err or "无法读取或帧数为 0"
            fut: concurrent.futures.Future = gaze_pool.submit(
                lambda s, m: dict(file_stem=s, error=m), stem, msg
            )
        else:
            fut = gaze_pool.submit(extract_gaze_from_frames, frames, stem)

        pending[stem] = fut
        _flush(block=False)

    _flush(block=True)
    gaze_pool.shutdown(wait=False)

    print(f"\n完成！→ {GAZE_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
