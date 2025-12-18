#!/usr/bin/env python
"""
Validate an RNA–RIBO checkpoint on the held-out split.

Usage (example):
python validate_checkpoint.py \
  --checkpoint results/.../checkpoints/best-epoch=266.ckpt \
  --data_path /home/pengrui/yly_spatial_2/rna_ribo_layers.h5ad \
  --batch_size 512 --num_workers 4 --split 0.9
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.serialization import add_safe_globals
import pandas as pd
import scanpy as sc
from anndata import AnnData, concat

from scclip.clip import CLIPModel
from scclip.data import RnaRiboDataModule
from scclip.metrics import matching_metrics
from scclip.vit import ViTConfig


@torch.no_grad()
def compute_metrics_on_loader(model: CLIPModel, dataloader, device: str = "cuda") -> Dict[str, Any]:
    """Compute embeddings on a dataloader (e.g., all samples) and return matching metrics."""
    model = model.to(device)
    model.eval()
    atac_list, rna_list = [], []
    for batch in dataloader:
        atac = batch["atac"].to(device)
        rna = batch["rna"].to(device)
        atac_emb, rna_emb = model(atac, rna)
        atac_list.append(atac_emb.cpu())
        rna_list.append(rna_emb.cpu())
    atac_all = torch.cat(atac_list, dim=0)
    rna_all = torch.cat(rna_list, dim=0)

    # cosine similarity with learned scale (same as training)
    logit_scale = model.criterion.logit_scale.exp().item()
    similarity = torch.matmul(atac_all, rna_all.T) * logit_scale
    acc, matchscore, foscttm = matching_metrics(similarity)

    # diagnostics
    row_max = similarity.max(dim=1).values
    col_max = similarity.max(dim=0).values
    diag = similarity.diag()
    acc_x = (torch.argmax(similarity, dim=1) == torch.arange(similarity.shape[0])).float().mean().item()
    acc_y = (torch.argmax(similarity, dim=0) == torch.arange(similarity.shape[0])).float().mean().item()
    row_diag_eq_max = (diag == row_max).float().mean().item()
    col_diag_eq_max = (diag == col_max).float().mean().item()
    row_rank = torch.argsort(torch.argsort(similarity, dim=1, descending=True), dim=1)
    col_rank = torch.argsort(torch.argsort(similarity, dim=0, descending=True), dim=0)
    row_diag_rank = row_rank[torch.arange(similarity.shape[0]), torch.arange(similarity.shape[0])] + 1
    col_diag_rank = col_rank[torch.arange(similarity.shape[0]), torch.arange(similarity.shape[0])] + 1

    return {
        "acc_full": float(acc),
        "matchscore_full": float(matchscore),
        "foscttm_full": float(foscttm),
        "logit_scale": float(logit_scale),
        "n_samples": atac_all.shape[0],
        "acc_x": acc_x,
        "acc_y": acc_y,
        "row_diag_eq_max": row_diag_eq_max,
        "col_diag_eq_max": col_diag_eq_max,
        "row_diag_rank_median": float(row_diag_rank.float().median().item()),
        "col_diag_rank_median": float(col_diag_rank.float().median().item()),
        "sim_min": float(similarity.min().item()),
        "sim_max": float(similarity.max().item()),
        "sim_mean": float(similarity.mean().item()),
        "sim_std": float(similarity.std().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate an RNA–RIBO checkpoint on ALL samples (no train/val split).")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to .ckpt file")
    parser.add_argument("--data_path", required=True, type=str, help="Path to rna_ribo_layers.h5ad")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--out_dir", type=str, default=None, help="Where to save eval_summary.json (defaults to ckpt dir)")
    parser.add_argument("--save_umap", action="store_true", help="Generate UMAP on val embeddings (colored by modality)")
    parser.add_argument("--diag_report", action="store_true", help="Save detailed similarity diagnostics to diagnostics.log")
    parser.add_argument("--neighbors", type=int, default=30, help="UMAP neighbor count")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist")
    args = parser.parse_args()

    # Allow classes stored in checkpoints (PyTorch 2.6 weights_only=True default)
    add_safe_globals([argparse.Namespace, ViTConfig])

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "validate.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(f"checkpoint={ckpt_path}")
    logging.info(f"data_path={args.data_path}, batch_size={args.batch_size}, device={args.device}")
    logging.info(f"umap neighbors={args.neighbors}, min_dist={args.min_dist}")

    dm = RnaRiboDataModule(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=0.9,  # split is unused below; we evaluate on all samples
    )
    full_loader = dm.dataloader(dataset=dm.dataset, shuffle=False)
    model = CLIPModel.load_from_checkpoint(ckpt_path)
    # respect normalize flag from checkpoint config
    model.config.normalize = getattr(model.config, "normalize", True)
    if hasattr(dm.dataset, "num_labels"):
        model.config.num_labels = dm.dataset.num_labels

    full_metrics = compute_metrics_on_loader(model, full_loader, device=args.device)
    logging.info(f"metrics: {full_metrics}")

    umap_paths: List[str] = []
    if args.save_umap:
        # compute embeddings on all samples and plot UMAP colored by modality/label(如果存在)
        val_atac, val_rna = [], []
        model.eval().to(args.device)
        with torch.no_grad():
            for batch in full_loader:
                atac = batch["atac"].to(args.device)
                rna = batch["rna"].to(args.device)
                ae, re = model(atac, rna)
                val_atac.append(ae.cpu())
                val_rna.append(re.cpu())
        atac_all = torch.cat(val_atac, dim=0).numpy()
        rna_all = torch.cat(val_rna, dim=0).numpy()
        # 继承原始 obs（含 label）
        obs_df = dm.dataset.obs.reset_index(drop=True)
        atac_obs = obs_df.copy()
        atac_obs["modality"] = "ribo"
        rna_obs = obs_df.copy()
        rna_obs["modality"] = "rna"
        atac_adata = AnnData(atac_all, obs=atac_obs)
        rna_adata = AnnData(rna_all, obs=rna_obs)
        concat_adata = concat([atac_adata, rna_adata], label="modality", keys=["ribo", "rna"], index_unique="#")
        sc.settings.figdir = str(out_dir)
        sc.pp.neighbors(concat_adata, metric="cosine", use_rep="X", n_neighbors=args.neighbors)
        sc.tl.umap(concat_adata, min_dist=args.min_dist)
        sc.pl.umap(concat_adata, color="modality", show=False, save="_rna_ribo_val.png")
        umap_paths.append(str(out_dir / "umap_rna_ribo_val.png"))
        if "label" in concat_adata.obs.columns:
            sc.pl.umap(concat_adata, color="label", show=False, save="_rna_ribo_label.png")
            umap_paths.append(str(out_dir / "umap_rna_ribo_label.png"))

    summary = {"checkpoint": str(ckpt_path), "all_metrics": full_metrics}
    if umap_paths:
        summary["umap_pngs"] = umap_paths
    out_path = out_dir / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("All-sample matching metrics:")
    for k, v in full_metrics.items():
        print(f"  {k}: {v}")
    print(f"Saved summary to {out_path}")
    if umap_paths:
        for p in umap_paths:
            print(f"Saved UMAP to {p}")


if __name__ == "__main__":
    main()
