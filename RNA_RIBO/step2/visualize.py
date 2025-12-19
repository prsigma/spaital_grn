#!/usr/bin/env python
"""用融合 embedding 做聚类并按空间坐标着色散点图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scanpy as sc  # noqa: E402
import torch  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

from .data import load_spatial_multiome  # noqa: E402


def plot_spatial_clusters(
    h5ad_path: Path,
    embeddings_path: Path,
    out_path: Path,
    n_clusters: int | None = None,
    figsize=(8.4, 8.14),
    alpha: float = 0.8,
    s: float = 2.0,
    embedding_key: str = "h_final",
):
    # 读取空间信息
    data = load_spatial_multiome(str(h5ad_path))
    coords = data.coords
    labels = data.labels

    # 加载融合 embedding
    ckpt = torch.load(embeddings_path, map_location="cpu")
    if embedding_key not in ckpt:
        # 回退到 fused
        embedding_key = "fused"
    emb = ckpt[embedding_key].numpy()

    # 聚类（KMeans，簇数默认等于真实域数）
    if n_clusters is None:
        if labels is not None:
            n_clusters = len(np.unique(labels))
        else:
            n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    pred = kmeans.fit_predict(emb)

    # 颜色映射
    palette = sc.pl.palettes.godsnot_102 if n_clusters > 20 else sc.pl.palettes.default_20
    colors = [palette[i % len(palette)] for i in pred]

    # 坐标
    x = coords[:, 1] if coords.shape[1] > 1 else coords[:, 0]  # column
    y = -coords[:, 0]  # row，额外做一次垂直翻转

    plt.figure(figsize=figsize)
    plt.scatter(x, y, c=colors, s=s, alpha=alpha, linewidths=0)
    # 同时做水平、垂直翻转以匹配视觉习惯
    ax = plt.gca()
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()

    # 同时保存聚类结果
    np.save(out_path.with_suffix(".npy"), pred)
    return pred


def main():
    parser = argparse.ArgumentParser(description="Plot spatial clusters from fused embeddings.")
    parser.add_argument("--h5ad", required=True, type=str, help="原始 h5ad（提供坐标/标签）")
    parser.add_argument("--embeddings", required=True, type=str, help="embeddings.pt 路径")
    parser.add_argument("--out", type=str, default="spatial.png", help="输出图路径")
    parser.add_argument("--clusters", type=int, default=None, help="簇数（默认同域标签数）")
    parser.add_argument("--embedding_key", type=str, default="h_final", help="使用 embeddings.pt 中的键，默认 h_final，若不存在则回退 fused")
    parser.add_argument("--alpha", type=float, default=0.8, help="散点透明度")
    parser.add_argument("--size", type=float, default=2.0, help="散点大小")
    args = parser.parse_args()

    plot_spatial_clusters(
        h5ad_path=Path(args.h5ad),
        embeddings_path=Path(args.embeddings),
        out_path=Path(args.out),
        n_clusters=args.clusters,
        alpha=args.alpha,
        s=args.size,
        embedding_key=args.embedding_key,
    )


if __name__ == "__main__":
    main()
