#!/usr/bin/env python
"""
使用 embeddings.pt 中的 h_final 和 best checkpoint 中的分类头进行预测，并绘制 UMAP。
不重新前向传播/重建图，只取已有 h_final 输入分类头。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import anndata as ad  # noqa: E402

# print("正在启动调试服务器，端口：9999")
# debugpy.listen(9999)
# print("等待调试客户端连接...")
# debugpy.wait_for_client()
# print("调试客户端已连接，继续执行...")


def classify_with_classifier(
    run_dir: Path,
    h5ad_path: Path,
    device: str = "cpu",
    out_name: str = "spatial_pred.png",
    out_compare: str = "spatial_domain_vs_pred.png",
    cluster_method: str = "kmeans",
    n_clusters: Optional[int] = None,
    coord_x: str = "column",
    coord_y: str = "row",
):
    run_dir = Path(run_dir)
    emb = torch.load(run_dir / "embeddings.pt", map_location=device)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    labels = adata.obs["domain"].to_numpy() if "domain" in adata.obs else None
    uniq = {v: i for i, v in enumerate(sorted(set(labels)))} if labels is not None else None
    y_true = np.array([uniq[v] for v in labels], dtype=int) if labels is not None else None
    pred = emb.get("pred_classes")
    if pred is None:
        h_final = emb.get("h_final")
        if h_final is None:
            raise KeyError("embeddings.pt 中不存在 pred_classes 或 h_final，请确认训练输出")
        if n_clusters is None:
            if y_true is None:
                raise ValueError("未提供真实标签，需显式指定 --n_clusters")
            n_clusters = len(np.unique(y_true))
        if cluster_method == "kmeans":
            from sklearn.cluster import KMeans

            pred = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(h_final.cpu().numpy())
        else:
            raise ValueError(f"Unsupported cluster_method: {cluster_method}")
    else:
        pred = pred.cpu().numpy()

    # 评估
    metrics = {}
    if y_true is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure

        ari = adjusted_rand_score(y_true, pred)
        nmi = normalized_mutual_info_score(y_true, pred)
        h, c, v = homogeneity_completeness_v_measure(y_true, pred)
        metrics = {"ARI": ari, "NMI": nmi, "homogeneity": h, "completeness": c, "v": v}
        print(metrics)
        (run_dir / "pred_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def _resolve_coord_key(key: str):
        obs = adata.obs
        if key in obs:
            return key
        if key == "columns" and "column" in obs:
            return "column"
        if key == "column" and "columns" in obs:
            return "columns"
        if key == "rows" and "row" in obs:
            return "row"
        if key == "row" and "rows" in obs:
            return "rows"
        raise KeyError(f"Coordinate column '{key}' not found in obs")

    key_x = _resolve_coord_key(coord_x)
    key_y = _resolve_coord_key(coord_y)
    x = adata.obs[key_x].to_numpy()
    y = adata.obs[key_y].to_numpy()
    y = -y

    def _label_colors(values, cmap_name="tab20"):
        uniq_vals = sorted(set(values))
        cmap = plt.get_cmap(cmap_name, len(uniq_vals))
        color_map = {v: cmap(i) for i, v in enumerate(uniq_vals)}
        return np.array([color_map[v] for v in values])

    # 生成 domain vs pred 的空间对比图
    if labels is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].scatter(x, y, c=_label_colors(labels), s=6, alpha=1, linewidths=0, marker="o")
        axes[0].invert_yaxis()
        axes[0].invert_xaxis()
        axes[0].axis("off")
        axes[0].set_title("domain")

        axes[1].scatter(x, y, c=_label_colors(pred), s=6, alpha=1, linewidths=0, marker="o")
        axes[1].invert_yaxis()
        axes[1].invert_xaxis()
        axes[1].axis("off")
        axes[1].set_title("pred")

        plt.tight_layout()
        plt.savefig(run_dir / out_compare, dpi=150)
        plt.close()

    # 空间坐标着色（按 protocol-replicate 拆分左右子图）
    # 使用固定的索引->颜色映射
    domain_colors = {
        0: "#ff909f",
        1: "#98d6f9",
        2: "#cccccc",
        3: "#7ed04b",
        4: "#1f9d5a",
        5: "#ffcf00",
    }
    colors = np.array([domain_colors.get(int(p), "#000000") for p in pred])

    prot = adata.obs.get("protocol-replicate", None)
    unique_prot = prot.unique().tolist() if prot is not None else [None]

    n_cols = len(unique_prot)
    fig, axes = plt.subplots(1, n_cols, figsize=(8.4 * n_cols, 8.14))
    if n_cols == 1:
        axes = [axes]
    handles = []
    labels_legend = []
    for ax, val in zip(axes, unique_prot):
        mask = prot == val if prot is not None else np.ones_like(pred, dtype=bool)
        ax.scatter(x[mask], y[mask], c=colors[mask], s=6, alpha=1, linewidths=0,marker='o')
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.axis("off")
        ax.set_title(str(val))
    # Legend: 使用固定索引标签
    used_preds = sorted(set(pred))
    for cls in used_preds:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=str(cls),
                                  markerfacecolor=domain_colors.get(int(cls), "#000000"), markersize=6))
        labels_legend.append(str(cls))
    fig.legend(handles, labels_legend, loc="upper right", bbox_to_anchor=(1.05, 1.05))
    plt.tight_layout()
    plt.savefig(run_dir / out_name, dpi=120)
    plt.close()

    np.save(run_dir / "pred_labels.npy", pred)


def main():
    parser = argparse.ArgumentParser(description="Classify using h_final and classifier head from best checkpoint.")
    parser.add_argument("--run_dir", required=True, type=str, help="包含 model_best.pt / embeddings.pt 的目录")
    parser.add_argument("--h5ad", required=True, type=str, help="原始 h5ad（需含 domain 和 UMAP/可重算）")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_name", type=str, default="umap_pred.png")
    parser.add_argument("--out_compare", type=str, default="spatial_domain_vs_pred.png", help="domain vs pred 的空间对比图")
    parser.add_argument("--cluster_method", type=str, default="kmeans", choices=["kmeans"], help="pred_classes 缺失时的聚类方法")
    parser.add_argument("--n_clusters", type=int, default=None, help="聚类簇数（缺省则用 domain 的类数）")
    parser.add_argument("--coord_x", type=str, default="column", help="空间坐标 X 列名")
    parser.add_argument("--coord_y", type=str, default="row", help="空间坐标 Y 列名")
    args = parser.parse_args()
    classify_with_classifier(
        Path(args.run_dir),
        Path(args.h5ad),
        device=args.device,
        out_name=args.out_name,
        out_compare=args.out_compare,
        cluster_method=args.cluster_method,
        n_clusters=args.n_clusters,
        coord_x=args.coord_x,
        coord_y=args.coord_y,
    )


if __name__ == "__main__":
    main()
