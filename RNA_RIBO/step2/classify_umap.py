#!/usr/bin/env python
"""
用训练好的 best checkpoint（分类头）对所有 cells 进行预测，并绘制 UMAP。
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

from .data import load_spatial_multiome  # noqa: E402
from .graph import build_spatial_knn_graph  # noqa: E402
from .model import SpatialFusionModel  # noqa: E402
from .starnet_weights import build_starnet_weights  # noqa: E402


def load_args(run_dir: Path) -> dict:
    args_path = run_dir / "args.json"
    return json.loads(args_path.read_text()) if args_path.exists() else {}


def classify_with_trained_model(run_dir: Path, device: str = "cuda", k_spatial: int | None = None, out_name: str = "umap_pred.png"):
    run_dir = Path(run_dir)
    args_cfg = load_args(run_dir)
    h5ad_path = args_cfg.get("h5ad")
    if h5ad_path is None:
        raise FileNotFoundError("args.json 缺少 h5ad 路径")
    h5ad_path = Path(h5ad_path)

    # 数据与标签
    data = load_spatial_multiome(str(h5ad_path))
    adata = data.adata
    labels = None
    num_classes = None
    if data.labels is not None:
        uniq = {v: i for i, v in enumerate(sorted(set(data.labels)))}
        labels = torch.tensor([uniq[v] for v in data.labels], dtype=torch.long)
        num_classes = len(uniq)

    # 加载 embeddings
    emb = torch.load(run_dir / "embeddings.pt", map_location="cpu")
    z_rna = emb["z_rna"]
    z_ribo = emb["z_ribo"]

    # 参数
    adapter_dim = args_cfg.get("adapter_dim")
    adapter_dropout = args_cfg.get("adapter_dropout", 0.0)
    gene_dim = args_cfg.get("gene_dim", 128)
    gamma = args_cfg.get("gamma", 3.0)
    k_top = args_cfg.get("k_top", 30)
    w_resample = args_cfg.get("w_resample", 0.8)
    k_spatial = k_spatial if k_spatial is not None else args_cfg.get("k_spatial", 15)

    # STARNet 权重
    w_tx = build_starnet_weights(data.rna, gamma=gamma, k_top=k_top, w_resample=w_resample)
    w_ribo = build_starnet_weights(data.ribo, gamma=gamma, k_top=k_top, w_resample=w_resample)

    # 空间邻接
    adj_spatial = build_spatial_knn_graph(data.coords, k=k_spatial)

    # 模型
    model = SpatialFusionModel(
        dim=z_rna.size(1),
        gcn_hidden=z_rna.size(1),
        gcn_layers=2,
        adapter_dim=adapter_dim,
        adapter_dropout=adapter_dropout,
        gene_dim=gene_dim,
        w_tx=w_tx.to(device),
        w_ribo=w_ribo.to(device),
        num_classes=num_classes,
    ).to(device)
    state = torch.load(run_dir / "model_best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    z_rna = z_rna.to(device)
    z_ribo = z_ribo.to(device)
    adj_spatial = adj_spatial.to(device)

    with torch.no_grad():
        outputs = model(z_rna, z_ribo, adj_spatial)
        logits = outputs["logits"]
        pred = torch.argmax(logits, dim=1).cpu().numpy()

    # 评估
    metrics = {}
    if labels is not None:
        labels_np = labels.numpy()
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure

        ari = adjusted_rand_score(labels_np, pred)
        nmi = normalized_mutual_info_score(labels_np, pred)
        h, c, v = homogeneity_completeness_v_measure(labels_np, pred)
        metrics = {"ARI": ari, "NMI": nmi, "homogeneity": h, "completeness": c, "v": v}
        print(metrics)

    # UMAP
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

    if labels is not None:
        plt.figure(figsize=(8, 8))
        plt.scatter(umap[:, 0], umap[:, 1], c=labels_np, cmap="tab20", s=2, alpha=0.8, linewidths=0)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(run_dir / "umap_true.png", dpi=120)
        plt.close()

    np.save(run_dir / "pred_labels.npy", pred)
    (run_dir / "pred_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Use trained best checkpoint to classify cells and plot UMAP.")
    parser.add_argument("--run_dir", required=True, type=str, help="包含 model_best.pt、embeddings.pt、args.json 的目录")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--k_spatial", type=int, default=None, help="可覆盖 args.json 中的 k_spatial")
    parser.add_argument("--out_name", type=str, default="umap_pred.png")
    args = parser.parse_args()
    classify_with_trained_model(Path(args.run_dir), device=args.device, k_spatial=args.k_spatial, out_name=args.out_name)


if __name__ == "__main__":
    main()
