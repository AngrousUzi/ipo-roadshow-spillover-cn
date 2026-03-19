"""
analyze/run_fer.py
==================
批量提取面部情绪（FER）→ output/visual_fer.csv

专为 A800 单卡优化：
  - GPU_BATCH_SIZE=256，充分利用 80GB 显存
  - N_READERS 个线程并行解码视频，掩盖 I/O 延迟
  - 主线程 FER 串行，无显存争抢
  - 断点续算

运行：
  cd ~/IPO/ipo-roadshow-spillover-cn/analyze
  ~/miniconda3/envs/cv/bin/python run_fer.py
"""

import os
import sys
import queue
import threading
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, VISUAL_SAMPLE_FPS, GPU_DEVICE, FRAME_MAX_LONG_SIDE, INDEX_VIDEO_DIR
from visual_fer import read_sampled_frames, extract_visual_emotions_from_frames

# ─── 路径 & 参数 ──────────────────────────────────────────────────────
INDEXED_VIDEO_DIR = INDEX_VIDEO_DIR
FER_OUTPUT_FILE   = OUTPUT_DIR / "visual_fer.csv"

GPU_BATCH_SIZE  = 256   # A800 80GB
N_READERS       = 2     # 并行视频解码线程
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
    print("  FER  (GPU A800)  →  visual_fer.csv")
    print("═" * 60)
    print(f"  视频目录  : {INDEXED_VIDEO_DIR}")
    print(f"  GPU 批大小: {GPU_BATCH_SIZE}")
    print(f"  Reader 数 : {N_READERS}")

    # ── 断点续算 ─────────────────────────────────────────────────────
    done: set = set()
    if FER_OUTPUT_FILE.exists():
        df = pd.read_csv(FER_OUTPUT_FILE, usecols=["file_stem"])
        done = set(df["file_stem"].tolist())
        print(f"  已完成    : {len(done)} 条，跳过。")

    if not INDEXED_VIDEO_DIR.exists():
        print(f"[ERROR] 目录不存在: {INDEXED_VIDEO_DIR}")
        return

    tasks = [mp4 for mp4 in sorted(INDEXED_VIDEO_DIR.glob("*.mp4"))
             if mp4.stem not in done]
    n_total = len(tasks)
    print(f"  待处理    : {n_total} 个\n")
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

    # ── 主循环 ────────────────────────────────────────────────────────
    header = not FER_OUTPUT_FILE.exists()
    buf: list = []
    WRITE_BATCH = 20
    n_done = 0

    while True:
        item = frame_q.get()
        if item is None:
            break

        video_path, frames, err = item
        stem = video_path.stem
        n_done += 1
        print(f"[{n_done:>4}/{n_total}] {video_path.name}")

        if err is not None or not frames:
            row = dict(file_stem=stem, error=err or "无法读取或帧数为 0")
        else:
            row = extract_visual_emotions_from_frames(
                frames, stem, GPU_DEVICE, GPU_BATCH_SIZE
            )
            if row.get("error"):
                print(f"  [✗] {row['error'][:80]}")
            else:
                print(
                    f"  [✓] faces={row.get('frames_with_face',0)}/"
                    f"{row.get('frames_analyzed',0)}  "
                    f"net_pos={row.get('net_positive', float('nan')):.3f}  "
                    f"method={row.get('method','?')}"
                )

        buf.append(row)
        if len(buf) >= WRITE_BATCH or n_done == n_total:
            pd.DataFrame(buf).to_csv(
                FER_OUTPUT_FILE, mode="a", header=header,
                index=False, encoding="utf-8-sig",
            )
            header = False
            buf.clear()
            print(f"  ──> {n_done}/{n_total}  ({n_done/n_total*100:.1f}%)\n")

    print(f"完成！→ {FER_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
