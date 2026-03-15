"""
analyze/run_visual.py
=====================
批量处理所有路演视频，提取面部情绪指标（Visual Component）。

注意：视频分析非常耗时。强烈建议在 GPU 服务器上运行。
      本地 Windows 测试时可将 PLATFORM_LIST 限制为少量视频。

输出：analyze/output/visual_fer.csv
      列：file_stem, positive_ratio, negative_ratio, neutral_ratio,
          net_positive, frames_analyzed, frames_with_face, method, error

断点续算：跳过已处理的 file_stem。
串行执行（视频分析本身已占满 CPU/GPU，多进程反而会抢资源）。
"""

import os
from pathlib import Path

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PLATFORM_LIST, OUTPUT_DIR, VISUAL_SAMPLE_FPS,
    get_video_dir,
)
from visual_fer import extract_visual_emotions

OUTPUT_FILE = OUTPUT_DIR / "visual_fer.csv"
INDEXED_VIDEO_DIR = OUTPUT_DIR / ".." / "路演视频"

def collect_tasks(done_stems: set) -> list[Path]:
    tasks = []
    for platform in PLATFORM_LIST:
        video_dir = get_video_dir(platform)
        if not video_dir.exists():
            print(f"[SKIP] 目录不存在: {video_dir}")
            continue
        for mp4 in video_dir.glob("*.mp4"):
            if mp4.stem not in done_stems:
                tasks.append(mp4)
    return tasks

def collect_index_tasks() -> list[Path]:
    """收集所有待处理的 index 视频文件路径，跳过已完成的。"""
    tasks = []
    video_dir = INDEXED_VIDEO_DIR
    if not video_dir.exists():
        print(f"[SKIP] 目录不存在: {video_dir}")
    if video_dir.exists():
        for mp4 in video_dir.glob("*.mp4"):
            tasks.append(mp4)
    return tasks

def main():
    print("═" * 60)
    print("  Visual Emotion Recognition (FER)")
    print("═" * 60)
    print(f"  采样率: {VISUAL_SAMPLE_FPS} fps")

    done_stems: set = set()
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE, usecols=["file_stem"])
        done_stems = set(existing["file_stem"].tolist())
        print(f"已有 {len(done_stems)} 条结果，将跳过已处理文件。")

    # tasks = collect_tasks(done_stems)
    tasks = collect_index_tasks()
    print(f"待处理文件数：{len(tasks)}\n")

    if not tasks:
        print("无待处理文件，退出。")
        return

    write_header = not OUTPUT_FILE.exists()
    BATCH_SIZE = 50

    results_buf = []
    for i, video_path in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {video_path.name}")
        result = extract_visual_emotions(video_path, sample_fps=VISUAL_SAMPLE_FPS)

        if result["error"]:
            print(f"  [WARN] {result['error']}")
        else:
            print(f"  method={result['method']}  "
                  f"faces={result['frames_with_face']}/{result['frames_analyzed']}  "
                  f"net_pos={result['net_positive']:.4f}")

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
            print(f"  ===> 已保存至 {OUTPUT_FILE}  ({pct:.1f}%)\n")

    print(f"完成！结果已保存至：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
