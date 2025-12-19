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
import scanpy as sc  # noqa: E402
import torch  # noqa: E402


def load_args(run_dir: Path) -> dict:
    args_path = run_dir / "args.json"
    return json.loads(args_path.read_text()) if args_path.exists() else {}


def classify_with_classifier(run_dir: Path, h5ad_path: Path, device: str = "cpu", out_name: str = "umap_pred.png"):
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

    # 真实标签
    adata = sc.read_h5ad(h5ad_path)
    labels = adata.obs["domain"].to_numpy()
    uniq = {v: i for i, v in enumerate(sorted(set(labels)))}
    y_true = np.array([uniq[v] for v in labels], dtype=int)

    with torch.no_grad():
        logits = classifier(h_final)
        pred = torch.argmax(logits, dim=1).cpu().numpy()

    # 评估
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure

    ari = adjusted_rand_score(y_true, pred)
    nmi = normalized_mutual_info_score(y_true, pred)
    h, c, v = homogeneity_completeness_v_measure(y_true, pred)
    metrics = {"ARI": ari, "NMI": nmi, "homogeneity": h, "completeness": c, "v": v}
    print(metrics)
    (run_dir / "pred_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # UMAP 绘制
    if "X_umap" in adata.obsm_keys():
        umap = adata.obsm["X_umap"]
    else:
        sc.pp.neighbors(adata, use_rep="X_pca" if "X_pca" in adata.obsm_keys() else None)
        sc.tl.umap(adata)
        umap = adata.obsm["X_umap"]

    plt.figure(figsize=(8, 8))
    plt.scatter(umap[:, 0], umap[:, 1], c=pred, cmap="tab20", s=2, alpha=0.8, linewidths=0)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(run_dir / out_name, dpi=120)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(umap[:, 0], umap[:, 1], c=y_true, cmap="tab20", s=2, alpha=0.8, linewidths=0)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(run_dir / "umap_true.png", dpi=120)
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
