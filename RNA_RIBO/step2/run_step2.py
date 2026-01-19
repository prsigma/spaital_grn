#!/usr/bin/env python
"""
Step2 训练脚本：端到端训练空间融合模型（无需预训练）。

用法示例：
NUMBA_DISABLE_JIT=1 python RNA_RIBO/step2/run_step2.py \
  --h5ad RNA_RIBO/smoothing_umap/smoothk15.h5ad \
  --out_dir RNA_RIBO/step2/runs/no_pretrain \
  --device cuda --epochs 100 --k_spatial 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import random
import numpy as np
import torch

from RNA_RIBO.step2.data import load_spatial_multiome  # noqa: E402
from RNA_RIBO.step2.graph import build_spatial_knn_graph  # noqa: E402
from RNA_RIBO.step2.starnet_weights import build_starnet_weights  # noqa: E402
from RNA_RIBO.step2.model import SpatialFusionModel  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_step2(
    h5ad_path: Path,
    out_dir: Path,
    device: str = "cuda",
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler: str = "cosine",
    eta_min: Optional[float] = None,
    patience: int = 300,
    min_delta: float = 1e-4,
    # Model architecture
    dim: int = 128,
    encoder_hidden: int = 512,
    encoder_layers: int = 2,
    encoder_dropout: float = 0.1,
    gcn_hidden: int = 128,
    gcn_layers: int = 2,
    gcn_dropout: float = 0.0,
    # STARNet gene embedding
    gene_dim: Optional[int] = 128,
    gamma: float = 3.0,
    k_top: int = 30,
    w_resample: float = 0.8,
    # Spatial graph
    k_spatial: int = 15,
    # Data layers
    rna_layer: str = "rna_log1p",
    ribo_layer: str = "ribo_log1p",
    # Loss weights
    lambda_recon: float = 1.0,
    lambda_contrast: float = 0.5,
    lambda_link: float = 0.2,
    # Eval
    eval_every: int = 10,
    seed: int = 42,
):
    """
    端到端训练空间融合模型（修复后版本，无需预训练）。

    参数
    ----
    h5ad_path: 输入h5ad文件路径
    out_dir: 输出目录
    device: 训练设备
    epochs: 训练轮数
    lr: 学习率
    weight_decay: 权重衰减
    scheduler: 学习率调度器 (cosine/plateau/none)
    eta_min: 最小学习率
    patience: early stopping耐心轮数
    min_delta: early stopping改善阈值
    dim: embedding维度
    encoder_hidden: encoder隐藏层维度
    encoder_layers: encoder层数
    encoder_dropout: encoder dropout率
    gcn_hidden: GCN隐藏层维度
    gcn_layers: GCN层数
    gcn_dropout: GCN dropout率
    gene_dim: 基因embedding维度（None则不使用）
    gamma: STARNet gamma指数
    k_top: 每个spot选取的基因数
    w_resample: STARNet混合系数
    k_spatial: 空间KNN的k值
    rna_layer: h5ad中RNA layer名称
    ribo_layer: h5ad中RIBO layer名称
    lambda_recon: 重构损失权重
    lambda_contrast: 对比损失权重
    lambda_link: 链接预测损失权重
    eval_every: 每隔多少epoch做一次评估（<=0关闭）
    seed: 随机种子
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    # 保存参数
    (out_dir / "args.json").write_text(
        json.dumps(
            dict(
                h5ad=str(h5ad_path),
                device=device,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                scheduler=scheduler,
                eta_min=eta_min,
                patience=patience,
                min_delta=min_delta,
                dim=dim,
                encoder_hidden=encoder_hidden,
                encoder_layers=encoder_layers,
                encoder_dropout=encoder_dropout,
                gcn_hidden=gcn_hidden,
                gcn_layers=gcn_layers,
                gcn_dropout=gcn_dropout,
                gene_dim=gene_dim,
                gamma=gamma,
                k_top=k_top,
                w_resample=w_resample,
                k_spatial=k_spatial,
                rna_layer=rna_layer,
                ribo_layer=ribo_layer,
                lambda_recon=lambda_recon,
                lambda_contrast=lambda_contrast,
                lambda_link=lambda_link,
                eval_every=eval_every,
                seed=seed,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # 1) 加载数据
    print("Loading data...")
    data = load_spatial_multiome(str(h5ad_path), rna_layer=rna_layer, ribo_layer=ribo_layer)
    print(f"Data loaded: {data.rna.shape[0]} cells × {data.rna.shape[1]} genes")

    # 2) 构建空间图
    print(f"Building spatial KNN graph (k={k_spatial})...")
    adj_spatial = build_spatial_knn_graph(data.coords, k=k_spatial)

    # 3) 构建STARNet权重（如果使用基因embedding）
    w_tx = None
    w_ribo = None
    if gene_dim is not None:
        print(f"Building STARNet weights (gamma={gamma}, k_top={k_top})...")
        w_tx = build_starnet_weights(data.rna, gamma=gamma, k_top=k_top, w_resample=w_resample)
        w_ribo = build_starnet_weights(data.ribo, gamma=gamma, k_top=k_top, w_resample=w_resample)

    # 4) 准备训练数据
    rna_expr = torch.tensor(data.rna, dtype=torch.float32).to(device)
    ribo_expr = torch.tensor(data.ribo, dtype=torch.float32).to(device)
    adj_spatial = adj_spatial.to(device)
    if w_tx is not None:
        w_tx = w_tx.to(device)
        w_ribo = w_ribo.to(device)

    n_genes = data.rna.shape[1]
    print(f"Input: RNA {rna_expr.shape}, RIBO {ribo_expr.shape}")

    labels = data.labels
    y_true = None
    n_clusters = None
    if labels is not None:
        uniq = {v: i for i, v in enumerate(sorted(set(labels)))}
        y_true = np.array([uniq[v] for v in labels], dtype=int)
        n_clusters = len(np.unique(y_true))

    # 5) 初始化模型
    print("Initializing model...")
    model = SpatialFusionModel(
        n_genes=n_genes,
        dim=dim,
        encoder_hidden=encoder_hidden,
        encoder_layers=encoder_layers,
        encoder_dropout=encoder_dropout,
        gcn_hidden=gcn_hidden,
        gcn_layers=gcn_layers,
        gcn_dropout=gcn_dropout,
        use_decoder=True,
        temperature=0.07,
        gene_dim=gene_dim,
        w_tx=w_tx,
        w_ribo=w_ribo,
    ).to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # 7) 优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if eta_min is None:
        eta_min = lr * 0.1

    if scheduler == "cosine":
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    elif scheduler == "plateau":
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=50, min_lr=eta_min
        )
    else:
        lr_sched = None

    # 8) 训练循环
    print(f"\nStarting training for {epochs} epochs...")
    log_lines = []
    best_total = float("inf")
    best_epoch = -1
    no_improve = 0
    last_best = float("inf")
    best_ari = -1.0
    best_ari_epoch = -1
    eval_lines = []

    for epoch in range(1, epochs + 1):
        model.train()

        # Forward pass
        outputs = model(rna_expr, ribo_expr, adj_spatial)

        # Compute losses
        total_loss, loss_dict = model.compute_losses(
            outputs,
            rna_expr,
            ribo_expr,
            adj_spatial,
            lambda_recon=lambda_recon,
            lambda_contrast=lambda_contrast,
            lambda_link=lambda_link,
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Learning rate scheduling
        if lr_sched is not None:
            if scheduler == "plateau":
                lr_sched.step(total_loss)
            else:
                lr_sched.step()

        # Logging
        log = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
        log["total"] = float(total_loss.detach().cpu())
        log["epoch"] = epoch
        log["lr"] = optimizer.param_groups[0]["lr"]
        log_lines.append(log)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[epoch {epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in log.items() if k not in ["epoch", "lr"]))
            print(f"  lr={log['lr']:.6f}")

        # 评估（沿用 classify_umap 的聚类与指标计算逻辑，不改聚类方法）
        if eval_every > 0 and y_true is not None and (epoch % eval_every == 0 or epoch == 1):
            model.eval()
            with torch.no_grad():
                eval_outputs = model(rna_expr, ribo_expr, adj_spatial)
                h_final_eval = eval_outputs["h_final"]

            from sklearn.cluster import KMeans
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure

            pred = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(
                h_final_eval.cpu().numpy()
            )
            ari = adjusted_rand_score(y_true, pred)
            nmi = normalized_mutual_info_score(y_true, pred)
            h, c, v = homogeneity_completeness_v_measure(y_true, pred)
            eval_metrics = {
                "epoch": epoch,
                "ARI": ari,
                "NMI": nmi,
                "homogeneity": h,
                "completeness": c,
                "v": v,
            }
            eval_lines.append(eval_metrics)
            print(f"[eval {epoch}] ARI={ari:.4f} NMI={nmi:.4f}")

            if ari > best_ari:
                best_ari = ari
                best_ari_epoch = epoch
                torch.save(model.state_dict(), out_dir / "model_best_ari.pt")
                torch.save(
                    {
                        "h_final": h_final_eval.cpu(),
                        "pred_classes": torch.tensor(pred),
                        "epoch": epoch,
                        "metrics": eval_metrics,
                    },
                    out_dir / "embeddings_best_ari.pt",
                )

                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                adata = data.adata
                labels = adata.obs["rna_nn_alg1_label"].to_numpy() if "rna_nn_alg1_label" in adata.obs else None

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

                def _label_colors(values, color_map):
                    return np.array([color_map.get(v, "#000000") for v in values])

                key_x = _resolve_coord_key("column")
                key_y = _resolve_coord_key("row")
                x = adata.obs[key_x].to_numpy()
                y = -adata.obs[key_y].to_numpy()

                domain_label_colors = {}
                if labels is not None:
                    uniq_labels = sorted(set(labels))
                    cmap = plt.get_cmap("tab20", len(uniq_labels))
                    domain_label_colors = {lab: cmap(i) for i, lab in enumerate(uniq_labels)}

                uniq_pred = sorted(set(pred))
                cmap_pred = plt.get_cmap("tab20", len(uniq_pred))
                pred_color_map = {cls: cmap_pred(i) for i, cls in enumerate(uniq_pred)}

                if labels is not None:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    axes[0].scatter(
                        x,
                        y,
                        c=_label_colors(labels, domain_label_colors),
                        s=6,
                        alpha=1,
                        linewidths=0,
                        marker="o",
                    )
                    axes[0].invert_yaxis()
                    axes[0].invert_xaxis()
                    axes[0].axis("off")
                    axes[0].set_title("rna_nn_alg1_label")

                    axes[1].scatter(
                        x,
                        y,
                        c=_label_colors(pred, pred_color_map),
                        s=6,
                        alpha=1,
                        linewidths=0,
                        marker="o",
                    )
                    axes[1].invert_yaxis()
                    axes[1].invert_xaxis()
                    axes[1].axis("off")
                    axes[1].set_title("pred")

                    plt.tight_layout()
                    plt.savefig(out_dir / "spatial_domain_vs_pred_best_ari.png", dpi=150)
                    plt.close()

                colors = _label_colors(pred, pred_color_map)

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
                    ax.scatter(x[mask], y[mask], c=colors[mask], s=6, alpha=1, linewidths=0, marker="o")
                    ax.invert_yaxis()
                    ax.invert_xaxis()
                    ax.axis("off")
                    ax.set_title(str(val))
                if labels is not None:
                    uniq_labels = sorted(set(labels))
                    for lab in uniq_labels:
                        handles.append(
                            plt.Line2D(
                                [0],
                                [0],
                                marker="o",
                                color="w",
                                label=str(lab),
                                markerfacecolor=domain_label_colors.get(lab, "#000000"),
                                markersize=6,
                            )
                        )
                        labels_legend.append(str(lab))
                else:
                    for cls in uniq_pred:
                        handles.append(
                            plt.Line2D(
                                [0],
                                [0],
                                marker="o",
                                color="w",
                                label=str(cls),
                                markerfacecolor=pred_color_map.get(cls, "#000000"),
                                markersize=6,
                            )
                        )
                        labels_legend.append(str(cls))
                fig.legend(handles, labels_legend, loc="upper right", bbox_to_anchor=(1.05, 1.05))
                plt.tight_layout()
                plt.savefig(out_dir / "spatial_pred_best_ari.png", dpi=120)
                plt.close()

        # 保存最新模型
        torch.save(model.state_dict(), out_dir / "model_last.pt")

        # 保存最佳模型
        if total_loss.item() < best_total:
            best_total = total_loss.item()
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "model_best.pt")

        # Early stopping
        if total_loss.item() + min_delta < last_best:
            last_best = total_loss.item()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best epoch: {best_epoch}, Best loss: {best_total:.6f}")
            break

    # 9) 保存训练日志为TSV格式
    if log_lines:
        # 提取所有列名（保持一致的顺序）
        columns = ["epoch", "lr", "total"]
        # 添加各项loss列（按字母顺序排列，方便查看）
        loss_keys = sorted(set(k for log in log_lines for k in log.keys() if k not in ["epoch", "lr", "total"]))
        columns.extend(loss_keys)

        # 写入TSV文件
        with open(out_dir / "losses.tsv", "w", encoding="utf-8") as f:
            # 写入表头
            f.write("\t".join(columns) + "\n")
            # 写入每行数据
            for log in log_lines:
                row = [str(log.get(col, "")) for col in columns]
                f.write("\t".join(row) + "\n")

    if eval_lines:
        eval_columns = ["epoch", "ARI", "NMI", "homogeneity", "completeness", "v"]
        with open(out_dir / "eval_metrics.tsv", "w", encoding="utf-8") as f:
            f.write("\t".join(eval_columns) + "\n")
            for log in eval_lines:
                row = [str(log.get(col, "")) for col in eval_columns]
                f.write("\t".join(row) + "\n")

    # 10) 生成最终embeddings（使用best模型）
    print("\nGenerating final embeddings...")
    best_path = out_dir / "model_best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        model.load_state_dict(torch.load(out_dir / "model_last.pt", map_location=device))

    model.eval()
    with torch.no_grad():
        outputs = model(rna_expr, ribo_expr, adj_spatial)
        h_final = outputs["h_final"]
        weights = outputs["weights"]

    torch.save(
        {
            "h_final": h_final.cpu(),
            "weights": weights.cpu(),
            "best_epoch": best_epoch,
            "best_total": best_total,
        },
        out_dir / "embeddings.pt",
    )

    print(f"\nTraining complete!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best loss: {best_total:.6f}")
    if best_ari_epoch >= 0:
        print(f"Best ARI epoch: {best_ari_epoch}")
        print(f"Best ARI: {best_ari:.6f}")
    print(f"Output directory: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Step2: 端到端空间融合训练（无需预训练）")

    # Data
    parser.add_argument("--h5ad", type=str, required=True, help="输入h5ad文件路径")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录（默认 runs/timestamp）")
    parser.add_argument("--rna_layer", type=str, default="rna_log1p", help="h5ad中RNA layer，或'X'")
    parser.add_argument("--ribo_layer", type=str, default="ribo_log1p", help="h5ad中RIBO layer，或'X'")

    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    parser.add_argument("--eta_min", type=float, default=None, help="最小lr（默认lr*0.1）")
    parser.add_argument("--patience", type=int, default=300, help="early stopping耐心")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="early stopping阈值")

    # Model architecture
    parser.add_argument("--dim", type=int, default=128, help="Embedding维度")
    parser.add_argument("--encoder_hidden", type=int, default=512, help="Encoder隐藏层维度")
    parser.add_argument("--encoder_layers", type=int, default=2, help="Encoder层数")
    parser.add_argument("--encoder_dropout", type=float, default=0.1, help="Encoder dropout")
    parser.add_argument("--gcn_hidden", type=int, default=128, help="GCN隐藏层维度")
    parser.add_argument("--gcn_layers", type=int, default=2, help="GCN层数")
    parser.add_argument("--gcn_dropout", type=float, default=0.0, help="GCN dropout")

    # STARNet gene embedding
    parser.add_argument("--gene_dim", type=int, default=128, help="基因embedding维度（None关闭）")
    parser.add_argument("--gamma", type=float, default=3.0, help="STARNet gamma")
    parser.add_argument("--k_top", type=int, default=30, help="STARNet top-k基因数")
    parser.add_argument("--w_resample", type=float, default=0.8, help="STARNet混合系数")

    # Spatial graph
    parser.add_argument("--k_spatial", type=int, default=15, help="空间KNN的k值")

    # Loss weights
    parser.add_argument("--lambda_recon", type=float, default=1.0, help="重构损失权重")
    parser.add_argument("--lambda_contrast", type=float, default=0.5, help="对比损失权重")
    parser.add_argument("--lambda_link", type=float, default=0.2, help="链接预测损失权重")
    parser.add_argument("--eval_every", type=int, default=10, help="每隔多少epoch做一次评估（<=0关闭）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = (
        Path(args.out_dir) if args.out_dir else Path("RNA_RIBO/step2/runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    train_step2(
        h5ad_path=Path(args.h5ad),
        out_dir=out_dir,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        eta_min=args.eta_min,
        patience=args.patience,
        min_delta=args.min_delta,
        dim=args.dim,
        encoder_hidden=args.encoder_hidden,
        encoder_layers=args.encoder_layers,
        encoder_dropout=args.encoder_dropout,
        gcn_hidden=args.gcn_hidden,
        gcn_layers=args.gcn_layers,
        gcn_dropout=args.gcn_dropout,
        gene_dim=args.gene_dim,
        gamma=args.gamma,
        k_top=args.k_top,
        w_resample=args.w_resample,
        k_spatial=args.k_spatial,
        rna_layer=args.rna_layer,
        ribo_layer=args.ribo_layer,
        lambda_recon=args.lambda_recon,
        lambda_contrast=args.lambda_contrast,
        lambda_link=args.lambda_link,
        eval_every=args.eval_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
