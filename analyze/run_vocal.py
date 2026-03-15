"""
analyze/run_vocal.py
====================
批量提取所有路演音频的声学特征（Vocal Component）。

输出：analyze/output/vocal_features.csv
      列：file_stem, f0_mean, f0_std, rms_mean, rms_std, speech_rate, duration_s, error

断点续算：若输出文件已存在，跳过已处理的 file_stem。

并行：使用 SLURM_CPUS_PER_TASK 环境变量（默认 4 进程）。
"""

import os
import multiprocessing
from pathlib import Path

import pandas as pd

# ─── 配置 ─────────────────────────────────────────────────────────────
# 将 analyze/ 加入路径以便直接运行
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PLATFORM_LIST, OUTPUT_DIR,
    get_audio_dir, find_trans_for_audio,
    AUDIO_SR, HOP_LENGTH, FMIN, FMAX,
)
from vocal_features import extract_vocal_features

OUTPUT_FILE      = OUTPUT_DIR / "vocal_features.csv"
PARALLEL_PROCS   = int(os.getenv("SLURM_CPUS_PER_TASK", "4"))
INDEXED_AUDIO_DIR = OUTPUT_DIR / ".." / "路演音频"

# ─── Worker ────────────────────────────────────────────────────────────

def _process_one(wav_path: Path) -> dict:
    """单个文件的处理任务（在子进程中运行）。"""
    trans_path = find_trans_for_audio(wav_path)
    result = extract_vocal_features(
        wav_path=wav_path,
        transcript_json_path=trans_path,
        target_sr=AUDIO_SR,
        fmin=FMIN,
        fmax=FMAX,
        hop_length=HOP_LENGTH,
    )
    if result["error"]:
        print(f"[WARN] {wav_path.name}: {result['error']}")
    else:
        print(f"[OK]   {wav_path.name}")
    return result


def _worker(args):
    wav_path_str = args
    return _process_one(Path(wav_path_str))


# ─── 主流程 ────────────────────────────────────────────────────────────

def collect_tasks(done_stems: set) -> list[str]:
    """收集所有待处理的 wav 文件路径（字符串），跳过已完成的。"""
    tasks = []
    for platform in PLATFORM_LIST:
        audio_dir = get_audio_dir(platform)
        if not audio_dir.exists():
            print(f"[SKIP] 目录不存在: {audio_dir}")
            continue
        for wav in audio_dir.glob("*.wav"):
            if wav.stem not in done_stems:
                tasks.append(str(wav))
    return tasks

def collect_index_tasks() -> list[str]:
    """收集所有待处理的 index 文件路径（字符串），跳过已完成的。"""
    tasks = []
    audio_dir = INDEXED_AUDIO_DIR
    if not audio_dir.exists():
        print(f"[SKIP] 目录不存在: {audio_dir}")
    if audio_dir.exists():
        for wav in audio_dir.glob("*.wav"):
            tasks.append(str(wav))
    return tasks

def main():
    print("═" * 60)
    print("  Vocal Feature Extraction")
    print("═" * 60)

    # 断点续算
    done_stems: set = set()
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE, usecols=["file_stem"])
        done_stems = set(existing["file_stem"].tolist())
        print(f"已有 {len(done_stems)} 条结果，将跳过已处理文件。")

    # tasks = collect_tasks(done_stems)
    tasks = collect_index_tasks()
    print(f"待处理文件数：{len(tasks)}")

    if not tasks:
        print("无待处理文件，退出。")
        return

    num_workers = max(1, PARALLEL_PROCS)
    print(f"使用 {num_workers} 进程并行处理...\n")

    write_header = not OUTPUT_FILE.exists()

    # 批次写入：每 200 个写一次
    BATCH_SIZE = 200
    results_buf = []

    with multiprocessing.Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, tasks), 1):
            results_buf.append(result)
            if len(results_buf) >= BATCH_SIZE or i == len(tasks):
                df = pd.DataFrame(results_buf)
                df.to_csv(
                    OUTPUT_FILE,
                    mode="a",
                    header=write_header,
                    index=False,
                    encoding="utf-8-sig",
                )
                write_header = False
                results_buf = []
                pct = i / len(tasks) * 100
                print(f"  进度：{i}/{len(tasks)} ({pct:.1f}%)")

    print(f"\n完成！结果已保存至：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
