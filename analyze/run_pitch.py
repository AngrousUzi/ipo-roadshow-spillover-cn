"""
analyze/run_pitch.py
====================
主入口：加载三模态结果，计算 Pitch Factor。

用法：
  cd d:\科研\IPO\PythonProject\analyze
  python run_pitch.py

  或指定不含视觉特征（若 visual_fer.csv 尚未生成）：
  python run_pitch.py --no-visual

输出：analyze/output/pitch_factor.csv
       列：file_stem, pitch_factor, pc1 [, pc2, ...]
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, PCA_N_COMPONENTS
from pitch_factor import run_pca_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="计算 Pitch Factor（三模态 PCA）")
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="跳过视觉特征（visual_fer.csv），仅使用 vocal + verbal",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=PCA_N_COMPONENTS,
        help=f"PCA 主成分数（默认 {PCA_N_COMPONENTS}）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    vocal_csv  = OUTPUT_DIR / "vocal_features.csv"
    verbal_csv = OUTPUT_DIR / "verbal_sentiment.csv"
    visual_csv = OUTPUT_DIR / "visual_fer.csv" if not args.no_visual else None

    print("═" * 60)
    print("  Pitch Factor Construction (PCA)")
    print("═" * 60)
    print(f"  Vocal    : {vocal_csv}")
    print(f"  Verbal   : {verbal_csv}")
    print(f"  Visual   : {visual_csv if visual_csv else '(跳过)'}")
    print(f"  PCA 成分数: {args.n_components}")
    print()

    # 检查必需文件
    for p in [vocal_csv, verbal_csv]:
        if not p.exists():
            print(f"[ERROR] 必需文件不存在：{p}")
            print("        请先运行 run_vocal.py 和 run_verbal.py")
            sys.exit(1)

    result = run_pca_pipeline(
        vocal_csv=vocal_csv,
        verbal_csv=verbal_csv,
        visual_csv=visual_csv,
        n_components=args.n_components,
    )

    print(f"\n样本统计:")
    print(f"  总样本数  : {len(result)}")
    print(f"  pitch_factor 均值: {result['pitch_factor'].mean():.6f}")
    print(f"  pitch_factor 标准差: {result['pitch_factor'].std():.6f}")
    print(f"  pitch_factor 最小值: {result['pitch_factor'].min():.6f}")
    print(f"  pitch_factor 最大值: {result['pitch_factor'].max():.6f}")


if __name__ == "__main__":
    main()
