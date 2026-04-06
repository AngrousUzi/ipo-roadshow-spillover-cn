"""
analyze/run_fer.py
==================
批量提取面部情绪（FER）→ output/visual_fer.csv

专为 A800 单卡优化：
    - 流式分块（chunked streaming）：每读完 GPU_BATCH_SIZE 帧立即送 GPU，
      GPU 推理与磁盘读取真正并行，消除"等全视频读完"的等待
    - cap.grab() 跳帧：不采样的帧只移动指针，不解码，CPU 解码量减半
    - N_READERS 个线程并行分块读取，GPU 保持持续喂满
    - 断点续算

运行：
  cd ~/IPO/ipo-roadshow-spillover-cn/analyze
  ~/miniconda3/envs/cv/bin/python run_fer.py
"""

import argparse
import multiprocessing as _mp
import os
import queue
import sys
import threading
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    OUTPUT_DIR, VISUAL_SAMPLE_FPS, GPU_DEVICE,
    FRAME_MAX_LONG_SIDE, INDEX_VIDEO_DIR, GPU_BATCH_SIZE,
)
from visual_fer import (
    iter_frame_chunks,
    aggregate_emotion_results,
    _get_gpu_engine,
)

# ─── 路径 & 参数 ──────────────────────────────────────────────────────
INDEXED_VIDEO_DIR = INDEX_VIDEO_DIR
FER_OUTPUT_FILE   = OUTPUT_DIR / "visual_fer.csv"

N_READERS       = 6      # 并行视频解码线程（threading 模式）
# 每个 queue item 现在是一个 chunk（GPU_BATCH_SIZE 帧），而非整个视频
# 24 chunks × GPU_BATCH_SIZE 帧 × ~875KB/帧 ≈ 21GB 上限，限制内存消耗
PREFETCH_Q_SIZE = N_READERS * 4

# ── 多进程（spawn）模式参数 ──────────────────────────────────────────
# NVDEC 被禁用时 ffmpeg 软解是 CPU 瓶颈；spawn 子进程绕过 GIL 实现真并行
N_WORKERS       = 8      # spawn 子进程数（建议 = 物理核数，调参用 --workers）
# IPC chunk 须远小于 GPU_BATCH_SIZE：64帧 × ~875KB ≈ 56MB/消息，可控
MP_CHUNK_SIZE   = 64


# ─── 分块预读 Worker ──────────────────────────────────────────────────
# Queue item: (video_path, chunk, is_last, error_str | None)
#   chunk   : list[np.ndarray]，可能为空列表（仅 is_last=True 时）
#   is_last : 该视频的最后一块（含读取失败时的哨兵）

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
            for chunk, is_last in iter_frame_chunks(
                video_path,
                sample_fps=VISUAL_SAMPLE_FPS,
                max_long_side=FRAME_MAX_LONG_SIDE,
                chunk_size=GPU_BATCH_SIZE,
            ):
                frame_q.put((video_path, chunk, is_last, None))
        except Exception as e:
            # 读取失败：发送哨兵，通知主线程结束该视频
            frame_q.put((video_path, [], True, str(e)))

    with lock:
        remaining[0] -= 1
        if remaining[0] == 0:
            frame_q.put(None)  # 所有 reader 结束后发送全局哨兵


# ─── 多进程预读 Worker（spawn 模式，Linux）───────────────────────────
# 须为顶层函数才可被 pickle 传入 spawn 子进程

def _mp_reader_worker(task_q, frame_q, remaining_val, mp_lock, cfg):
    """
    spawn 子进程中运行的视频解码 worker。
    NVDEC 被禁用时 ffmpeg 使用 CPU 软解，多进程可绕过 GIL 实现真并行。

    cfg: dict
        sample_fps, max_long_side, chunk_size（IPC 小块，减少 pickle 开销）
    """
    while True:
        try:
            video_path_str = task_q.get(timeout=1)
        except Exception:
            continue
        if video_path_str is None:
            break
        video_path = Path(video_path_str)
        try:
            for chunk, is_last in iter_frame_chunks(
                video_path,
                sample_fps=cfg["sample_fps"],
                max_long_side=cfg["max_long_side"],
                chunk_size=cfg["chunk_size"],
            ):
                frame_q.put((video_path_str, chunk, is_last, None))
        except Exception as e:
            frame_q.put((video_path_str, [], True, str(e)))

    with mp_lock:
        remaining_val.value -= 1
        if remaining_val.value == 0:
            frame_q.put(None)   # 所有 worker 结束后发送全局哨兵


# ─── 主流程 ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FER batch inference")
    parser.add_argument("--mp", action="store_true",
                        help="使用 spawn 多进程解码（Linux/A800，NVDEC 禁用时推荐）")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"spawn 进程数（默认 {N_WORKERS}，--mp 有效时生效）")
    args = parser.parse_args()

    use_mp    = args.mp
    n_workers = args.workers

    print("═" * 60)
    print("  FER  (GPU A800)  →  visual_fer.csv")
    print("═" * 60)
    print(f"  视频目录  : {INDEXED_VIDEO_DIR}")
    print(f"  GPU 批大小: {GPU_BATCH_SIZE}")
    if use_mp:
        print(f"  解码模式  : spawn 多进程  workers={n_workers}  IPC chunk={MP_CHUNK_SIZE}")
    else:
        print(f"  解码模式  : threading    readers={N_READERS}")

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

    # ── 提前初始化 GPU 引擎（避免第一个视频等待模型加载）────────────
    engine = _get_gpu_engine(device=GPU_DEVICE, batch_size=GPU_BATCH_SIZE)
    if engine is None:
        print("[ERROR] GPU 引擎不可用（facenet-pytorch / hsemotion 未安装）")
        return

    # ── 启动分块预读（threading 或 spawn 多进程）─────────────────────
    if use_mp:
        # spawn 模式：子进程真并行 CPU 软解，绕过 GIL
        ctx     = _mp.get_context("spawn")
        task_q  = ctx.Queue()
        # MP_CHUNK_SIZE 远小于 GPU_BATCH_SIZE，每消息更小；队列上限相应放大
        mp_prefetch = n_workers * 8
        frame_q = ctx.Queue(maxsize=mp_prefetch)
        for t in tasks:
            task_q.put(str(t))          # Path → str，可安全 pickle
        for _ in range(n_workers):
            task_q.put(None)            # 终止哨兵

        remaining_val = ctx.Value("i", n_workers)
        mp_lock       = ctx.Lock()
        cfg = dict(
            sample_fps  = VISUAL_SAMPLE_FPS,
            max_long_side = FRAME_MAX_LONG_SIDE,
            chunk_size  = MP_CHUNK_SIZE,
        )
        for _ in range(n_workers):
            ctx.Process(
                target=_mp_reader_worker,
                args=(task_q, frame_q, remaining_val, mp_lock, cfg),
                daemon=True,
            ).start()
    else:
        # threading 模式（原逻辑，兼容 Windows / NVDEC 可用时）
        task_q  = queue.Queue()
        frame_q = queue.Queue(maxsize=PREFETCH_Q_SIZE)
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

    # ── 主循环：流式 GPU 推理 + 按视频累积结果 ────────────────────────
    # video_accum: path → list[dict | None]（各帧情绪结果，跨 chunk 累积）
    video_accum: dict[Path, list] = {}

    header = not FER_OUTPUT_FILE.exists()
    buf: list = []
    WRITE_BATCH = 20
    n_done = 0
    
    import time
    last_log_time = time.time()

    def _speech_type(stem: str) -> str:
        if stem.endswith("_推介"):
            return "推介"
        if stem.endswith("_答谢"):
            return "答谢"
        return ""

    while True:
        item = frame_q.get()
        if item is None:
            break

        video_path, chunk, is_last, err = item
        video_path = Path(video_path)   # MP 模式传 str，统一转 Path
        stem = video_path.stem
        
        # 定期输出队列水位
        now = time.time()
        if now - last_log_time > 30:
            q_size  = frame_q.qsize()
            q_limit = (n_workers * 8) if use_mp else PREFETCH_Q_SIZE
            print(f"  [队列水位] {q_size}/{q_limit}  处理进度: {n_done}/{n_total}")
            last_log_time = now

        if err is not None:
            # 读取失败：丢弃已累积的部分结果，写入错误行
            video_accum.pop(video_path, None)
            n_done += 1
            print(f"[{n_done:>4}/{n_total}] {video_path.name}  [ERROR] {err[:80]}")
            buf.append(dict(file_stem=stem, speech_type=_speech_type(stem), error=err))
        else:
            if video_path not in video_accum:
                video_accum[video_path] = []

            if chunk:
                # GPU 推理当前 chunk，结果追加到该视频的累积列表
                emo_chunk = engine.analyze_frames(chunk)
                video_accum[video_path].extend(emo_chunk)

            if is_last:
                # 该视频所有 chunk 处理完毕，聚合并输出
                all_emo = video_accum.pop(video_path)
                n_done += 1
                print(f"[{n_done:>4}/{n_total}] {video_path.name}")
                row = aggregate_emotion_results(all_emo, stem, method="gpu_batch")
                row["speech_type"] = _speech_type(stem)
                if row.get("error"):
                    print(f"  [✗] {row['error'][:80]}")
                else:
                    print(
                        f"  [✓] faces={row.get('frames_with_face', 0)}/"
                        f"{row.get('frames_analyzed', 0)}  "
                        f"net_pos={row.get('net_positive', float('nan')):.3f}"
                    )
                buf.append(row)

        # 批量写入
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
