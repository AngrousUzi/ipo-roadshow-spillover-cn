"""
analyze/run_verbal.py
=====================
批量处理所有路演转录 JSON，计算文本情绪指标（Verbal Component）。

输出：analyze/output/verbal_sentiment.csv
      列：file_stem, positive_ratio, negative_ratio, uncertain_ratio,
          tone_score, total_words, total_chars, method, error

断点续算：跳过已处理的 file_stem。
并行：多进程（SLURM_CPUS_PER_TASK，默认 4）。
"""

import os
import multiprocessing
from pathlib import Path

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PLATFORM_LIST, OUTPUT_DIR, LEXICON_DIR,
    get_trans_dir,
)
from verbal_sentiment import analyze_verbal_sentiment

OUTPUT_FILE    = OUTPUT_DIR / "verbal_sentiment.csv"
PARALLEL_PROCS = int(os.getenv("SLURM_CPUS_PER_TASK", "4"))


def _worker(json_path_str: str) -> dict:
    json_path = Path(json_path_str)
    result = analyze_verbal_sentiment(json_path, lexicon_dir=LEXICON_DIR)
    if result["error"]:
        print(f"[WARN] {json_path.name}: {result['error']}")
    else:
        print(f"[OK]   {json_path.name}  method={result['method']}  "
              f"tone={result['tone_score']:.4f}")
    return result


def collect_tasks(done_stems: set) -> list[str]:
    tasks = []
    for platform in PLATFORM_LIST:
        trans_dir = get_trans_dir(platform)
        if not trans_dir.exists():
            print(f"[SKIP] 目录不存在: {trans_dir}")
            continue
        for jp in trans_dir.glob("*.json"):
            if jp.stem not in done_stems:
                tasks.append(str(jp))
    return tasks


def main():
    print("═" * 60)
    print("  Verbal Sentiment Analysis")
    print("═" * 60)

    done_stems: set = set()
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE, usecols=["file_stem"])
        done_stems = set(existing["file_stem"].tolist())
        print(f"已有 {len(done_stems)} 条结果，将跳过已处理文件。")

    tasks = collect_tasks(done_stems)
    print(f"待处理文件数：{len(tasks)}")

    if not tasks:
        print("无待处理文件，退出。")
        return

    num_workers = max(1, PARALLEL_PROCS)
    print(f"使用 {num_workers} 进程并行处理...\n")

    write_header = not OUTPUT_FILE.exists()
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
