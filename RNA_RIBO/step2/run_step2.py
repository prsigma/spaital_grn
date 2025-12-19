#!/usr/bin/env python
"""
Step2 训练脚本：载入预训练 CLIP 模型，提取 RNA/Ribo embedding，基于空间图训练融合+GCN 模型。

用法示例：
NUMBA_DISABLE_JIT=1 python RNA_RIBO/step2/run_step2.py \\
  --h5ad RNA_RIBO/smoothing_umap/smoothk15.h5ad \\
  --ckpt scCLIP_ribo/results/rna_ribo_layers_withlabel/rna_ribo_3.0_True_5000_0.0002_/csv_logs/version_1/checkpoints/best-epoch=243.ckpt \\
  --out_dir RNA_RIBO/step2/runs/smoothk15 \\
  --device cuda --epochs 50 --k_spatial 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset

# 把 scCLIP_ribo 包加入路径
REPO_ROOT = Path(__file__).resolve().parents[2]
SC_CLIP_ROOT = REPO_ROOT / "scCLIP_ribo"
if str(SC_CLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_CLIP_ROOT))

from scclip.clip import CLIPModel  # noqa: E402
from scclip.data import RnaRiboDataModule  # noqa: E402
from scclip.vit import ViTConfig  # noqa: E402
from torch.serialization import add_safe_globals  # noqa: E402

from .data import load_spatial_multiome  # noqa: E402
from .graph import build_spatial_knn_graph  # noqa: E402
from .starnet_weights import build_starnet_weights  # noqa: E402
from .model import SpatialFusionModel  # noqa: E402


def extract_embeddings(
    ckpt_path: Path,
    h5ad_path: Path,
    batch_size: int = 512,
    num_workers: int = 0,
    device: str = "cuda",
    rna_layer: str = "rna_log1p",
    ribo_layer: str = "ribo_log1p",
):
    """
    载入预训练 CLIP，提取全量 RNA/Ribo embedding（对应 rna/ribo 模态）。
    """
    add_safe_globals([argparse.Namespace, ViTConfig])  # 兼容 ckpt 中保存的 config
    # 直接读取 h5ad，使用指定 layer
    import scanpy as sc

    adata = sc.read_h5ad(h5ad_path)
    if rna_layer == "X":
        rna = torch.tensor(adata.X, dtype=torch.float32)
    else:
        rna = torch.tensor(adata.layers[rna_layer], dtype=torch.float32)
    if ribo_layer == "X":
        ribo = torch.tensor(adata.X, dtype=torch.float32)
    else:
        ribo = torch.tensor(adata.layers[ribo_layer], dtype=torch.float32)
    dataset = TensorDataset(ribo, rna)  # 顺序：ribo -> atac 分支, rna -> rna 分支
    full_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    model = CLIPModel.load_from_checkpoint(str(ckpt_path))
    model.config.normalize = getattr(model.config, "normalize", True)
    model = model.to(device)
    model.eval()

    rna_list, ribo_list = [], []
    with torch.no_grad():
        for batch in full_loader:
            ribo_batch, rna_batch = [x.to(device) for x in batch]
            ribo_emb, rna_emb = model(ribo_batch, rna_batch)
            rna_list.append(rna_emb.cpu())
            ribo_list.append(ribo_emb.cpu())

    z_rna = torch.cat(rna_list, dim=0)
    z_ribo = torch.cat(ribo_list, dim=0)
    return z_rna, z_ribo


def train_step2(
    h5ad_path: Path,
    ckpt_path: Path,
    out_dir: Path,
    device: str = "cuda",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler: str = "cosine",
    eta_min: Optional[float] = None,
    patience: int = 300,
    min_delta: float = 1e-4,
    adapter_dim: Optional[int] = None,
    adapter_dropout: float = 0.0,
    gene_dim: Optional[int] = 128,
    gamma: float = 3.0,
    k_top: int = 30,
    w_resample: float = 0.8,
    k_spatial: int = 15,
    batch_size: int = 512,
    num_workers: int = 0,
    rna_layer: str = "rna_log1p",
    ribo_layer: str = "ribo_log1p",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(
        json.dumps(
            dict(
                h5ad=str(h5ad_path),
                ckpt=str(ckpt_path),
                device=device,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                scheduler=scheduler,
                eta_min=eta_min,
                patience=patience,
                min_delta=min_delta,
                adapter_dim=adapter_dim,
                adapter_dropout=adapter_dropout,
                gene_dim=gene_dim,
                gamma=gamma,
                k_top=k_top,
                w_resample=w_resample,
                k_spatial=k_spatial,
                batch_size=batch_size,
                num_workers=num_workers,
                rna_layer=rna_layer,
                ribo_layer=ribo_layer,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # 1) 读取 coords/labels
    data = load_spatial_multiome(str(h5ad_path))
    labels = None
    if data.labels is not None:
        # 将标签转为连续 id
        uniq = {v: i for i, v in enumerate(sorted(set(data.labels)))}
        labels = torch.tensor([uniq[v] for v in data.labels], dtype=torch.long)

    # 2) 构建空间邻接
    adj_spatial = build_spatial_knn_graph(data.coords, k=k_spatial)

    # 3) 提取预训练 embedding
    z_rna, z_ribo = extract_embeddings(
        ckpt_path,
        h5ad_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        rna_layer=rna_layer,
        ribo_layer=ribo_layer,
    )

    # 4) 构造 STARNet 权重并初始化基因 embedding
    w_tx = build_starnet_weights(data.rna, gamma=gamma, k_top=k_top, w_resample=w_resample)
    w_ribo = build_starnet_weights(data.ribo, gamma=gamma, k_top=k_top, w_resample=w_resample)
    # 移到 device
    w_tx = w_tx.to(device)
    w_ribo = w_ribo.to(device)

    # 4) 训练融合模型（全图批次）
    model = SpatialFusionModel(
        dim=z_rna.size(1),
        gcn_hidden=z_rna.size(1),
        gcn_layers=2,
        adapter_dim=adapter_dim,
        adapter_dropout=adapter_dropout,
        gene_dim=gene_dim,
        w_tx=w_tx,
        w_ribo=w_ribo,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if eta_min is None:
        eta_min = lr * 0.1
    if scheduler == "cosine":
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    elif scheduler == "plateau":
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50, min_lr=eta_min)
    else:
        lr_sched = None

    z_rna = z_rna.to(device)
    z_ribo = z_ribo.to(device)
    adj_spatial = adj_spatial.to(device)
    labels = labels.to(device) if labels is not None else None

    log_lines = []
    best_total = float("inf")
    best_epoch = -1
    no_improve = 0
    last_best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        outputs = model(z_rna, z_ribo, adj_spatial)
        total_loss, loss_dict = model.compute_losses(
            outputs,
            z_rna,
            z_ribo,
            adj_spatial,
            labels=labels,
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if lr_sched is not None:
            if scheduler == "plateau":
                lr_sched.step(total_loss)
            else:
                lr_sched.step()

        log = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
        log["total"] = float(total_loss.detach().cpu())
        log["epoch"] = epoch
        log_lines.append(log)
        print(f"[epoch {epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in log.items() if k != "epoch"))

        # 保存最新模型
        torch.save(model.state_dict(), out_dir / "model_last.pt")
        # 保存 best（以总损失为准）
        if total_loss.item() < best_total:
            best_total = total_loss.item()
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "model_best.pt")
        # early stopping based on total (domain)
        if total_loss.item() + min_delta < last_best:
            last_best = total_loss.item()
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}, best_epoch={best_epoch}, best_total={best_total:.6f}")
            break

    # 保存日志
    (out_dir / "losses.jsonl").write_text("\n".join(json.dumps(x) for x in log_lines), encoding="utf-8")
    # 保存融合 embedding（使用 best 权重，如无则 last）
    best_path = out_dir / "model_best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        model.load_state_dict(torch.load(out_dir / "model_last.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        outputs = model(z_rna, z_ribo, adj_spatial)
        fused = outputs["fused"]
        h_final = outputs["h_final"]
        weights = outputs["weights"]
    torch.save(
        {
            "fused": fused.cpu(),
            "h_final": h_final.cpu(),
            "weights": weights.cpu(),
            "z_rna": z_rna.cpu(),
            "z_ribo": z_ribo.cpu(),
            "labels": labels.cpu() if labels is not None else None,
            "best_epoch": best_epoch,
            "best_total": best_total,
        },
        out_dir / "embeddings.pt",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Step2: spatial fusion training")
    parser.add_argument("--h5ad", type=str, required=True, help="路径：平滑后的 h5ad，例如 RNA_RIBO/smoothing_umap/smoothk15.h5ad")
    parser.add_argument("--ckpt", type=str, required=True, help="预训练 CLIP ckpt 路径")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录（默认 runs/timestamp）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    parser.add_argument("--eta_min", type=float, default=None, help="cosine/plateau 最小 lr，默认 lr*0.1")
    parser.add_argument("--patience", type=int, default=300, help="early stopping 的耐心轮数")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="early stopping 改善阈值")
    parser.add_argument("--adapter_dim", type=int, default=None, help="Adapter 瓶颈维度，None 表示不启用")
    parser.add_argument("--adapter_dropout", type=float, default=0.0, help="Adapter dropout")
    parser.add_argument("--gene_dim", type=int, default=128, help="基因 embedding 维度，None 关闭基因部分")
    parser.add_argument("--gamma", type=float, default=3.0, help="STARNet gamma 指数")
    parser.add_argument("--k_top", type=int, default=30, help="每个 spot 选取的基因数（top-k）")
    parser.add_argument("--w_resample", type=float, default=0.8, help="STARNet 混合系数")
    parser.add_argument("--k_spatial", type=int, default=15)
    parser.add_argument("--lambda_domain", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=512, help="提取预训练 embedding 的批大小")
    parser.add_argument("--num_workers", type=int, default=0, help="提取预训练 embedding 的数据加载线程数")
    parser.add_argument("--rna_layer", type=str, default="rna_log1p", help="h5ad 中 RNA 使用的 layer，或 'X'")
    parser.add_argument("--ribo_layer", type=str, default="ribo_log1p", help="h5ad 中 RIBO 使用的 layer，或 'X'")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path("RNA_RIBO/step2/runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    train_step2(
        h5ad_path=Path(args.h5ad),
        ckpt_path=Path(args.ckpt),
        out_dir=out_dir,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        eta_min=args.eta_min,
        patience=args.patience,
        min_delta=args.min_delta,
        adapter_dim=args.adapter_dim,
        adapter_dropout=args.adapter_dropout,
        k_spatial=args.k_spatial,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rna_layer=args.rna_layer,
        ribo_layer=args.ribo_layer,
    )


if __name__ == "__main__":
    main()
