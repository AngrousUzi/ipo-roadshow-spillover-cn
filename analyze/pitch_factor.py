"""
analyze/pitch_factor.py
========================
合并三模态特征，通过 PCA 构建综合「Pitch Factor（吸引力指数）」。

流程：
  1. 加载 vocal_features.csv, verbal_sentiment.csv, visual_fer.csv
  2. 选择分析指标（feature_cols），对缺失值进行插补
  3. z-score 标准化（StandardScaler）
  4. PCA，取第一主成分（PC1）作为 Pitch Factor
  5. 输出 pitch_factor.csv：
       file_stem, pitch_factor, [各 PC 得分], explained_variance_ratio

依赖：
  pip install scikit-learn pandas numpy
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, PCA_N_COMPONENTS

# ─── 各模态特征列定义 ──────────────────────────────────────────────────

# Vocal：基频均值/标准差、音量均值/标准差、语速
VOCAL_COLS = ["f0_mean", "f0_std", "rms_mean", "rms_std", "speech_rate"]

# Verbal：正面词比率、负面词比率、不确定词比率、情绪综合分
VERBAL_COLS = ["positive_ratio", "negative_ratio", "uncertain_ratio", "tone_score"]

# Visual：面部正面比率、负面比率、净正面情绪分
VISUAL_COLS = ["positive_ratio", "negative_ratio", "net_positive"]


def load_modality(csv_path: Path, rename_prefix: str, use_cols: list[str]) -> pd.DataFrame:
    """
    加载模态 CSV，仅保留 file_stem + use_cols，并给特征列加前缀以防命名冲突。

    参数
    ----
    csv_path       : 模态输出 CSV 路径
    rename_prefix  : 列名前缀（如 "vocal_", "verbal_", "visual_"）
    use_cols       : 要使用的特征列名（不含 file_stem）

    返回
    ----
    DataFrame with columns: file_stem, {prefix}{col} for col in use_cols
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"模态文件不存在: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 仅保留存在的特征列
    available = [c for c in use_cols if c in df.columns]
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        print(f"[WARN] {csv_path.name} 缺少列：{missing}")

    df = df[["file_stem"] + available].copy()

    # 重命名特征列
    rename_map = {c: f"{rename_prefix}{c}" for c in available}
    df.rename(columns=rename_map, inplace=True)

    return df


def build_feature_matrix(
    vocal_csv: Path,
    verbal_csv: Path,
    visual_csv: Optional[Path] = None,
    min_vocal: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    合并三模态特征为宽表。

    参数
    ----
    vocal_csv  : vocal_features.csv
    verbal_csv : verbal_sentiment.csv
    visual_csv : visual_fer.csv（可选；若文件不存在则跳过）
    min_vocal  : 若 True，则要求 vocal 特征必须存在（以 vocal 为基准 inner join）

    返回
    ----
    (merged_df, feature_cols): 合并后的宽表和特征列名列表
    """
    vocal_df  = load_modality(vocal_csv,  "vocal_",  VOCAL_COLS)
    verbal_df = load_modality(verbal_csv, "verbal_", VERBAL_COLS)

    how = "inner" if min_vocal else "outer"
    merged = vocal_df.merge(verbal_df, on="file_stem", how=how)

    if visual_csv is not None and visual_csv.exists():
        visual_df = load_modality(visual_csv, "visual_", VISUAL_COLS)
        merged = merged.merge(visual_df, on="file_stem", how="left")
    else:
        print("[INFO] visual_fer.csv 不存在或未提供，跳过视觉特征。")

    feature_cols = [c for c in merged.columns if c != "file_stem"]
    return merged, feature_cols


def compute_pitch_factor(
    merged: pd.DataFrame,
    feature_cols: list[str],
    n_components: int = 1,
) -> tuple[pd.DataFrame, PCA, StandardScaler]:
    """
    对特征矩阵进行标准化 + PCA，返回含 Pitch Factor 的 DataFrame。

    缺失值策略：用列均值填充（mean imputation）。
    若某列全为 NaN（如视觉模态全部缺失），则丢弃该列。

    返回
    ----
    (result_df, pca_model, scaler)
    """
    # 丢弃全为 NaN 的列
    df = merged[["file_stem"] + feature_cols].copy()
    all_nan_cols = [c for c in feature_cols if df[c].isna().all()]
    if all_nan_cols:
        print(f"[INFO] 丢弃全 NaN 列：{all_nan_cols}")
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]

    X = df[feature_cols].copy()

    # 均值插补缺失值
    for col in feature_cols:
        if X[col].isna().any():
            col_mean = X[col].mean()
            X[col] = X[col].fillna(col_mean)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca_scores = pca.fit_transform(X_scaled)

    # 构建输出 DataFrame
    result = df[["file_stem"]].copy()
    result["pitch_factor"] = pca_scores[:, 0]
    for i in range(n_components):
        result[f"pc{i+1}"] = pca_scores[:, i]

    # 主成分载荷（供解释用）
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"pc{i+1}" for i in range(n_components)],
    )

    # explained_variance_ratio
    evr = pca.explained_variance_ratio_
    print("\n  PCA 解释方差比率：")
    for i, v in enumerate(evr):
        print(f"    PC{i+1}: {v:.4f} ({v*100:.2f}%)")
    print(f"  累计解释比率：{evr.sum():.4f} ({evr.sum()*100:.2f}%)")

    print("\n  PC1 载荷（特征重要性）：")
    print(loadings["pc1"].sort_values(ascending=False).to_string())

    return result, pca, scaler


def run_pca_pipeline(
    vocal_csv: Path,
    verbal_csv: Path,
    visual_csv: Optional[Path] = None,
    output_path: Optional[Path] = None,
    n_components: int = PCA_N_COMPONENTS,
) -> pd.DataFrame:
    """
    完整 PCA 流水线的入口函数。

    参数
    ----
    vocal_csv  : vocal_features.csv
    verbal_csv : verbal_sentiment.csv
    visual_csv : visual_fer.csv（可选）
    output_path: 输出 CSV 路径（默认 analyze/output/pitch_factor.csv）
    n_components: PCA 保留主成分数

    返回
    ----
    pitch_factor DataFrame
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "pitch_factor.csv"

    print("加载各模态特征...")
    merged, feature_cols = build_feature_matrix(vocal_csv, verbal_csv, visual_csv)
    print(f"  合并后样本数：{len(merged)}，特征数：{len(feature_cols)}")
    print(f"  特征列：{feature_cols}")

    print("\n计算 Pitch Factor (PCA)...")
    result, pca, scaler = compute_pitch_factor(merged, feature_cols, n_components)

    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Pitch Factor 已保存至：{output_path}")

    return result
