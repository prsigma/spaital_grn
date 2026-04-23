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
from RNA_RIBO.step2.checkpoint_gene_rank import (  # noqa: E402
    evaluate_checkpoint_gene_rank,
    extract_layer_aligned_to_reference,
    load_gene_rank_reference,
)


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
    cell_id_key: str = "cell_id",
    # Loss weights
    lambda_recon: float = 1.0,
    lambda_contrast: float = 0.5,
    lambda_link: float = 0.2,
    # Eval
    eval_every: int = 10,
    ckpt_ref_csv: str = "gene_integrated_scores_weighted.csv",
    ckpt_ref_gene_col: str = "gene",
    ckpt_ref_rank_col: str = "rank",
    ckpt_ribo_norm_layer: str = "rbRNA_norm",
    ckpt_topk: int = 100,
    ckpt_save_best_cos: bool = True,
    ckpt_cell_chunk_size: int = 256,
    ckpt_gene_chunk_size: int = 256,
    ckpt_corr_block_size: int = 256,
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
    cell_id_key: h5ad中显式 cell id 列名（缺失时报错）
    lambda_recon: 重构损失权重
    lambda_contrast: 对比损失权重
    lambda_link: 链接预测损失权重
    eval_every: 每隔多少epoch做一次评估（<=0关闭）
    ckpt_ref_csv: 基因参考排序CSV路径
    ckpt_ref_gene_col: 基因列名
    ckpt_ref_rank_col: 排序列名（升序）
    ckpt_ribo_norm_layer: h5ad中的RIBO归一化层名
    ckpt_topk: checkpoint比较使用的Top-K
    ckpt_save_best_cos: 是否保存最佳epoch的S(c,g)矩阵
    ckpt_cell_chunk_size: (c,g)计算时的cell分块
    ckpt_gene_chunk_size: (c,g)计算时的gene分块
    ckpt_corr_block_size: 相关矩阵计算时的gene分块
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
                cell_id_key=cell_id_key,
                lambda_recon=lambda_recon,
                lambda_contrast=lambda_contrast,
                lambda_link=lambda_link,
                eval_every=eval_every,
                ckpt_ref_csv=ckpt_ref_csv,
                ckpt_ref_gene_col=ckpt_ref_gene_col,
                ckpt_ref_rank_col=ckpt_ref_rank_col,
                ckpt_ribo_norm_layer=ckpt_ribo_norm_layer,
                ckpt_topk=ckpt_topk,
                ckpt_save_best_cos=ckpt_save_best_cos,
                ckpt_cell_chunk_size=ckpt_cell_chunk_size,
                ckpt_gene_chunk_size=ckpt_gene_chunk_size,
                ckpt_corr_block_size=ckpt_corr_block_size,
                seed=seed,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # 1) 加载数据
    print("Loading data...")
    data = load_spatial_multiome(
        str(h5ad_path),
        rna_layer=rna_layer,
        ribo_layer=ribo_layer,
        cell_id_key=cell_id_key,
    )
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
    cell_idx = torch.tensor(data.cell_idx, dtype=torch.long).to(device)
    adj_spatial = adj_spatial.to(device)
    if w_tx is not None:
        w_tx = w_tx.to(device)
        w_ribo = w_ribo.to(device)

    n_genes = data.rna.shape[1]
    print(f"Input: RNA {rna_expr.shape}, RIBO {ribo_expr.shape}")

    if gene_dim is None:
        raise ValueError("gene_dim must be set for gene-rank checkpoint evaluation")

    # 4.1) 准备基于gene ranking的checkpoint参考
    ref_csv_path = Path(ckpt_ref_csv)
    if not ref_csv_path.exists():
        candidate = Path(__file__).resolve().parents[2] / ref_csv_path
        if candidate.exists():
            ref_csv_path = candidate
    print(f"Loading checkpoint reference from {ref_csv_path} ...")
    reference = load_gene_rank_reference(
        csv_path=ref_csv_path,
        var_names=data.adata.var_names,
        gene_col=ckpt_ref_gene_col,
        rank_col=ckpt_ref_rank_col,
        topk=ckpt_topk,
    )
    ribo_norm_np = extract_layer_aligned_to_reference(
        adata=data.adata,
        layer=ckpt_ribo_norm_layer,
        var_indices=reference.var_indices,
    )
    ribo_norm = torch.tensor(ribo_norm_np, dtype=torch.float32)
    print(
        f"Checkpoint ranking reference: {len(reference.gene_names)} genes, "
        f"Top-{len(reference.reference_topk)} target"
    )
    ref_rank_lookup = {g: i + 1 for i, g in enumerate(reference.gene_names.tolist())}

    # 5) 初始化模型
    print("Initializing model...")
    model = SpatialFusionModel(
        n_genes=n_genes,
        num_cells=data.num_cells,
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
    best_rank_metric = -1.0
    best_rank_epoch = -1
    eval_lines = []

    for epoch in range(1, epochs + 1):
        model.train()

        # Forward pass
        outputs = model(rna_expr, ribo_expr, adj_spatial, cell_idx=cell_idx)

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

        # 评估：基于(c,g)余弦与外部gene ranking的一致性
        if eval_every > 0 and (epoch % eval_every == 0 or epoch == 1):
            model.eval()
            eval_metrics = evaluate_checkpoint_gene_rank(
                model=model,
                cell_idx=cell_idx,
                ribo_norm=ribo_norm,
                reference=reference,
                topk=ckpt_topk,
                cell_chunk_size=ckpt_cell_chunk_size,
                gene_chunk_size=ckpt_gene_chunk_size,
                corr_block_size=ckpt_corr_block_size,
                save_cosine=ckpt_save_best_cos,
            )
            rank_metric = float(eval_metrics["metric_topk_overlap"])
            eval_lines.append(
                {
                    "epoch": epoch,
                    "topk_overlap": rank_metric,
                    "overlap_count": int(eval_metrics["overlap_count"]),
                    "topk": int(eval_metrics["topk"]),
                }
            )
            print(
                f"[eval {epoch}] Top-{int(eval_metrics['topk'])} overlap="
                f"{rank_metric:.4f} ({int(eval_metrics['overlap_count'])})"
            )

            if rank_metric > best_rank_metric:
                best_rank_metric = rank_metric
                best_rank_epoch = epoch
                torch.save(model.state_dict(), out_dir / "model_best.pt")

                gene_scores = np.asarray(eval_metrics["gene_scores"], dtype=np.float32)
                gene_names = np.asarray(eval_metrics["gene_names"], dtype=object)
                order = np.argsort(-gene_scores)
                with open(out_dir / "best_gene_scores.tsv", "w", encoding="utf-8") as f:
                    f.write("gene\tmodel_score\tmodel_rank\treference_rank\n")
                    for ridx, gidx in enumerate(order, start=1):
                        g = str(gene_names[gidx])
                        ref_rank = ref_rank_lookup.get(g, "")
                        f.write(f"{g}\t{float(gene_scores[gidx]):.8f}\t{ridx}\t{ref_rank}\n")

                if ckpt_save_best_cos and eval_metrics["cosine_cg"] is not None:
                    torch.save(
                        {
                            "cosine_cg": eval_metrics["cosine_cg"],
                            "gene_names": gene_names.tolist(),
                            "epoch": epoch,
                            "metric_topk_overlap": rank_metric,
                            "topk": int(eval_metrics["topk"]),
                        },
                        out_dir / "cosine_cg_best.pt",
                    )

        # 保存最新模型
        torch.save(model.state_dict(), out_dir / "model_last.pt")

        # 记录loss最优模型（不作为主checkpoint）
        if total_loss.item() < best_total:
            best_total = total_loss.item()
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "model_best_loss.pt")

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
        eval_columns = ["epoch", "topk_overlap", "overlap_count", "topk"]
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
        outputs = model(rna_expr, ribo_expr, adj_spatial, cell_idx=cell_idx)
        h_final = outputs["h_final"]
        weights = outputs["weights"]

    torch.save(
        {
            "h_final": h_final.cpu(),
            "weights": weights.cpu(),
            "best_epoch": best_rank_epoch,
            "best_total": best_total,
            "best_topk_overlap": best_rank_metric,
        },
        out_dir / "embeddings.pt",
    )

    print(f"\nTraining complete!")
    print(f"Best checkpoint epoch: {best_rank_epoch}")
    print(f"Best Top-{ckpt_topk} overlap: {best_rank_metric:.6f}")
    print(f"Best loss epoch: {best_epoch}")
    print(f"Best loss: {best_total:.6f}")
    print(f"Output directory: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Step2: 端到端空间融合训练（无需预训练）")

    # Data
    parser.add_argument("--h5ad", type=str, required=True, help="输入h5ad文件路径")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录（默认 runs/timestamp）")
    parser.add_argument("--rna_layer", type=str, default="rna_log1p", help="h5ad中RNA layer，或'X'")
    parser.add_argument("--ribo_layer", type=str, default="ribo_log1p", help="h5ad中RIBO layer，或'X'")
    parser.add_argument("--cell_id_key", type=str, default="cell_id", help="h5ad中显式 cell id 列名")

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
    parser.add_argument("--eval_every", type=int, default=10, help="每隔多少epoch做一次checkpoint评估（<=0关闭）")
    parser.add_argument("--ckpt_ref_csv", type=str, default="gene_integrated_scores_weighted.csv", help="checkpoint参考排序CSV")
    parser.add_argument("--ckpt_ref_gene_col", type=str, default="gene", help="参考CSV中的gene列名")
    parser.add_argument("--ckpt_ref_rank_col", type=str, default="rank", help="参考CSV中的rank列名（升序）")
    parser.add_argument("--ckpt_ribo_norm_layer", type=str, default="rbRNA_norm", help="h5ad中用于相关分析的RIBO归一化层")
    parser.add_argument("--ckpt_topk", type=int, default=100, help="checkpoint比较使用Top-K")
    parser.add_argument("--ckpt_save_best_cos", type=int, choices=[0, 1], default=1, help="是否保存最佳epoch的S(c,g)矩阵")
    parser.add_argument("--ckpt_cell_chunk_size", type=int, default=256, help="(c,g)计算cell分块")
    parser.add_argument("--ckpt_gene_chunk_size", type=int, default=256, help="(c,g)计算gene分块")
    parser.add_argument("--ckpt_corr_block_size", type=int, default=256, help="相关矩阵计算gene分块")
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
        cell_id_key=args.cell_id_key,
        lambda_recon=args.lambda_recon,
        lambda_contrast=args.lambda_contrast,
        lambda_link=args.lambda_link,
        eval_every=args.eval_every,
        ckpt_ref_csv=args.ckpt_ref_csv,
        ckpt_ref_gene_col=args.ckpt_ref_gene_col,
        ckpt_ref_rank_col=args.ckpt_ref_rank_col,
        ckpt_ribo_norm_layer=args.ckpt_ribo_norm_layer,
        ckpt_topk=args.ckpt_topk,
        ckpt_save_best_cos=bool(args.ckpt_save_best_cos),
        ckpt_cell_chunk_size=args.ckpt_cell_chunk_size,
        ckpt_gene_chunk_size=args.ckpt_gene_chunk_size,
        ckpt_corr_block_size=args.ckpt_corr_block_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
