"""
analyze/run_qa.py
=================
Batch-process all unified Q&A CSVs → analyze/output/qa_analysis.csv

One row per roadshow. Columns:
    file_stem, index2009,
    qa_pairs, speech_count,
    avg_q_len, avg_a_len, a_q_len_ratio, num_ratio_in_answer,
    n_unique_questioners,
    q_{verbal metrics}, a_{verbal metrics},
    error

断点续算: skips already-processed file_stems.
并行: multiprocessing (SLURM_CPUS_PER_TASK, default 4).

Note: unified QA CSVs must be regenerated (delete 路演问答/ and re-run
initialize_qa.py) to include the questioner_id column. Until then, the
analysis falls back to 提问人 for n_unique_questioners.
"""

import os
import multiprocessing
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, LEXICON_DIR, INDEX_QA_DIR
from verbal_sentiment import load_lexicons
from qa_analysis import analyze_one_qa

OUTPUT_FILE    = OUTPUT_DIR / "qa_analysis.csv"
PARALLEL_PROCS = int(os.getenv("SLURM_CPUS_PER_TASK", "4"))

_LEXICONS = None


def _init_worker():
    global _LEXICONS
    _LEXICONS = load_lexicons(LEXICON_DIR)


def _worker(csv_path_str: str) -> dict:
    result = analyze_one_qa(Path(csv_path_str), _LEXICONS)
    name = Path(csv_path_str).name
    if result["error"]:
        print(f"[WARN] {name}: {result['error']}")
    else:
        q_tone = result.get("q_ann_tone_score", float("nan"))
        a_tone = result.get("a_ann_tone_score", float("nan"))
        print(f"[OK]   {name}  qa={result['qa_pairs']}  "
              f"questioners={result['n_unique_questioners']}  "
              f"q_ann_tone={q_tone:.4f}  a_ann_tone={a_tone:.4f}")
    return result


def main():
    print("═" * 60)
    print("  QA Analysis")
    print("═" * 60)
    print(f"输入目录: {INDEX_QA_DIR}")
    print(f"输出文件: {OUTPUT_FILE}\n")

    done_stems: set = set()
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE, usecols=["file_stem"])
        done_stems = set(existing["file_stem"].tolist())
        print(f"已有 {len(done_stems)} 条结果，将跳过已处理文件。")

    tasks = [
        str(p) for p in sorted(INDEX_QA_DIR.glob("*.csv"))
        if p.stem not in done_stems
    ]
    print(f"待处理文件数: {len(tasks)}\n")

    if not tasks:
        print("无待处理文件，退出。")
        return

    num_workers = max(1, PARALLEL_PROCS)
    print(f"使用 {num_workers} 进程并行处理...\n")

    write_header = not OUTPUT_FILE.exists()
    BATCH_SIZE = 100
    results_buf = []

    with multiprocessing.Pool(processes=num_workers, initializer=_init_worker) as pool:
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
                print(f"  进度: {i}/{len(tasks)} ({pct:.1f}%)")

    print(f"\n完成！结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
