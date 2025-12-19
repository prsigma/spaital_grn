#!/usr/bin/env python
"""
使用 embeddings.pt 中的 h_final 和 best checkpoint 中的分类头进行预测，并绘制 UMAP。
不重新前向传播/重建图，只取已有 h_final 输入分类头。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from .data import load_spatial_multiome  # noqa: E402
import scanpy as sc  # noqa: E402


def classify_with_classifier(run_dir: Path, h5ad_path: Path, device: str = "cpu", out_name: str = "spatial_pred.png"):
    run_dir = Path(run_dir)
    emb = torch.load(run_dir / "embeddings.pt", map_location=device)
    h_final = emb["h_final"].to(device)

    # 加载 checkpoint 中的分类头参数
    state = torch.load(run_dir / "model_best.pt", map_location=device)
    if "classifier.weight" not in state:
        raise KeyError("model_best.pt 中没有分类头参数 (classifier.weight)")
    num_classes, dim = state["classifier.weight"].shape
    classifier = torch.nn.Linear(dim, num_classes)
    classifier.load_state_dict(
        {"weight": state["classifier.weight"].to(device), "bias": state["classifier.bias"].to(device)}
    )
    classifier.to(device)
    classifier.eval()

    # 真实标签 + 坐标
    data = load_spatial_multiome(str(h5ad_path))
    coords = data.coords
    labels = data.labels
    uniq = {v: i for i, v in enumerate(sorted(set(labels)))} if labels is not None else None
    y_true = np.array([uniq[v] for v in labels], dtype=int) if labels is not None else None

    with torch.no_grad():
        logits = classifier(h_final)
        pred = torch.argmax(logits, dim=1).cpu().numpy()

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

    # 空间坐标着色（参考 visualize），并按 protocol-replicate 拆分左右子图
    x = coords[:, 1] if coords.shape[1] > 1 else coords[:, 0]
    y = -coords[:, 0]
    # 固定 6 色高对比调色板（按类别索引映射）
    palette = ["#ff909f", "#98d6f9", "#cccccc", "#7ed04b", "#1f9d5a", "#ffcf00"]
    colors = np.array([palette[i % len(palette)] for i in pred])

    prot = data.adata.obs.get("protocol-replicate", None)
    unique_prot = prot.unique().tolist() if prot is not None else [None]

    n_cols = len(unique_prot)
    fig, axes = plt.subplots(1, n_cols, figsize=(8.4 * n_cols, 8.14))
    if n_cols == 1:
        axes = [axes]
    handles = []
    labels_legend = []
    for ax, val in zip(axes, unique_prot):
        mask = prot == val if prot is not None else np.ones_like(pred, dtype=bool)
        ax.scatter(x[mask], y[mask], c=colors[mask], s=2.0, alpha=0.8, linewidths=0)
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.axis("off")
        ax.set_title(str(val))
    # Legend
    for cls in range(num_classes if num_classes else len(np.unique(pred))):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=str(cls),
                                  markerfacecolor=palette[cls % len(palette)], markersize=6))
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
    args = parser.parse_args()
    classify_with_classifier(Path(args.run_dir), Path(args.h5ad), device=args.device, out_name=args.out_name)


if __name__ == "__main__":
    main()
