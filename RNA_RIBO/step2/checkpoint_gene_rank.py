"""Gene-rank based checkpoint evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy import sparse


@dataclass
class GeneRankReference:
    """Reference ranking aligned to current AnnData var names."""

    gene_names: np.ndarray
    var_indices: np.ndarray
    reference_topk: np.ndarray


def _as_dense_float32(arr) -> np.ndarray:
    if sparse.issparse(arr):
        arr = arr.A
    return np.asarray(arr, dtype=np.float32)


def load_gene_rank_reference(
    csv_path: Path | str,
    var_names: Sequence[str],
    gene_col: str = "gene",
    rank_col: str = "rank",
    topk: int = 100,
) -> GeneRankReference:
    """Load reference rank CSV and align with current var names."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference rank csv not found: {path}")

    df = pd.read_csv(path)
    if gene_col not in df.columns or rank_col not in df.columns:
        raise KeyError(
            f"Expected columns '{gene_col}' and '{rank_col}' in {path}. "
            f"Got: {list(df.columns)}"
        )
    df = df[[gene_col, rank_col]].copy()
    df[gene_col] = df[gene_col].astype(str)
    df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce")
    df = df.dropna(subset=[gene_col, rank_col])
    df = df.sort_values(rank_col, ascending=True).drop_duplicates(subset=[gene_col], keep="first")

    var_names = np.asarray(var_names, dtype=str)
    var_to_idx = {g: i for i, g in enumerate(var_names)}
    common_genes = [g for g in df[gene_col].tolist() if g in var_to_idx]
    if len(common_genes) < 2:
        raise ValueError(
            f"Too few common genes between reference ({path}) and var_names: {len(common_genes)}"
        )
    common_indices = np.asarray([var_to_idx[g] for g in common_genes], dtype=np.int64)
    effective_topk = min(max(topk, 1), len(common_genes))
    reference_topk = np.asarray(common_genes[:effective_topk], dtype=object)

    return GeneRankReference(
        gene_names=np.asarray(common_genes, dtype=object),
        var_indices=common_indices,
        reference_topk=reference_topk,
    )


def extract_layer_aligned_to_reference(adata, layer: str, var_indices: np.ndarray) -> np.ndarray:
    """Read dense (N, G_ref) matrix from adata layer aligned by var_indices."""
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found. Available: {list(adata.layers.keys())}")
    full = _as_dense_float32(adata.layers[layer])
    return np.asarray(full[:, var_indices], dtype=np.float32)


def topk_overlap_ratio(
    model_ranked_genes: Sequence[str],
    reference_ranked_genes: Sequence[str],
    topk: int,
) -> tuple[int, float]:
    """Compute overlap count and ratio for Top-K genes."""
    k = min(max(topk, 1), len(model_ranked_genes), len(reference_ranked_genes))
    model_top = set(model_ranked_genes[:k])
    ref_top = set(reference_ranked_genes[:k])
    overlap = len(model_top & ref_top)
    return overlap, overlap / float(k)


def _zscore_columns(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def evaluate_checkpoint_gene_rank(
    model,
    cell_idx: torch.Tensor,
    ribo_norm: torch.Tensor,
    reference: GeneRankReference,
    topk: int = 100,
    cell_chunk_size: int = 256,
    gene_chunk_size: int = 256,
    corr_block_size: int = 256,
    save_cosine: bool = False,
) -> dict:
    """
    Evaluate checkpoint by gene-ranking agreement.

    Returns a dict including top-k overlap metric and per-gene scores.
    """
    if not getattr(model, "use_gene_emb", False):
        raise ValueError("Gene-rank checkpoint evaluation requires model.use_gene_emb=True")
    if cell_idx.dim() != 1:
        raise ValueError(f"cell_idx must be 1D, got shape={tuple(cell_idx.shape)}")

    device = next(model.parameters()).device
    cell_idx = cell_idx.to(device=device, dtype=torch.long)
    ribo_norm = ribo_norm.to(dtype=torch.float32, device="cpu")

    n_cells = int(cell_idx.shape[0])
    n_ref_genes = int(reference.gene_names.shape[0])
    if ribo_norm.shape[0] != n_cells:
        raise ValueError(f"ribo_norm rows {ribo_norm.shape[0]} != n_cells {n_cells}")
    if ribo_norm.shape[1] != n_ref_genes:
        raise ValueError(f"ribo_norm cols {ribo_norm.shape[1]} != n_ref_genes {n_ref_genes}")

    with torch.no_grad():
        e_rna_all, e_ribo_all = model.get_projected_gene_embeddings()
        gene_idx = torch.tensor(reference.var_indices, dtype=torch.long, device=device)
        e_rna = e_rna_all.index_select(0, gene_idx)  # (G_ref, D)
        e_ribo = e_ribo_all.index_select(0, gene_idx)  # (G_ref, D)

        c_all = model.cell_embedding(cell_idx)  # (N, D)
        k_rna_all = model.rna_cell_gate.k_proj(c_all)
        v_rna_all = model.rna_cell_gate.v_proj(c_all)
        k_ribo_all = model.ribo_cell_gate.k_proj(c_all)
        v_ribo_all = model.ribo_cell_gate.v_proj(c_all)

        q_rna_all = model.rna_cell_gate.q_proj(e_rna)
        q_ribo_all = model.ribo_cell_gate.q_proj(e_ribo)

        e_dot = torch.sum(e_rna * e_ribo, dim=-1)  # (G_ref,)
        e_rna_norm2 = torch.sum(e_rna * e_rna, dim=-1)  # (G_ref,)
        e_ribo_norm2 = torch.sum(e_ribo * e_ribo, dim=-1)  # (G_ref,)
        scale_rna = float(model.rna_cell_gate.scale)
        scale_ribo = float(model.ribo_cell_gate.scale)

        cosine_cg = torch.empty((n_cells, n_ref_genes), dtype=torch.float32, device="cpu")

        for c0 in range(0, n_cells, cell_chunk_size):
            c1 = min(c0 + cell_chunk_size, n_cells)
            k_rna = k_rna_all[c0:c1]
            v_rna = v_rna_all[c0:c1]
            k_ribo = k_ribo_all[c0:c1]
            v_ribo = v_ribo_all[c0:c1]

            vdot = torch.sum(v_rna * v_ribo, dim=-1, keepdim=True)  # (Nc, 1)
            v_rna_norm2 = torch.sum(v_rna * v_rna, dim=-1, keepdim=True)  # (Nc, 1)
            v_ribo_norm2 = torch.sum(v_ribo * v_ribo, dim=-1, keepdim=True)  # (Nc, 1)

            for g0 in range(0, n_ref_genes, gene_chunk_size):
                g1 = min(g0 + gene_chunk_size, n_ref_genes)
                q_rna = q_rna_all[g0:g1]
                q_ribo = q_ribo_all[g0:g1]
                er = e_rna[g0:g1]
                eb = e_ribo[g0:g1]

                alpha_rna = torch.sigmoid((k_rna @ q_rna.t()) / scale_rna)  # (Nc, Gc)
                alpha_ribo = torch.sigmoid((k_ribo @ q_ribo.t()) / scale_ribo)  # (Nc, Gc)

                v_rna_eb = v_rna @ eb.t()  # (Nc, Gc)
                v_ribo_er = v_ribo @ er.t()  # (Nc, Gc)
                v_rna_er = v_rna @ er.t()  # (Nc, Gc)
                v_ribo_eb = v_ribo @ eb.t()  # (Nc, Gc)

                dot = (
                    e_dot[g0:g1].unsqueeze(0)
                    + alpha_rna * v_rna_eb
                    + alpha_ribo * v_ribo_er
                    + (alpha_rna * alpha_ribo) * vdot
                )
                norm_rna = (
                    e_rna_norm2[g0:g1].unsqueeze(0)
                    + 2.0 * alpha_rna * v_rna_er
                    + (alpha_rna * alpha_rna) * v_rna_norm2
                )
                norm_ribo = (
                    e_ribo_norm2[g0:g1].unsqueeze(0)
                    + 2.0 * alpha_ribo * v_ribo_eb
                    + (alpha_ribo * alpha_ribo) * v_ribo_norm2
                )
                denom = torch.sqrt(norm_rna.clamp_min(1e-8) * norm_ribo.clamp_min(1e-8))
                cos = (dot / denom).to(dtype=torch.float32)
                cosine_cg[c0:c1, g0:g1] = cos.cpu()

    s_std = _zscore_columns(cosine_cg)
    r_std = _zscore_columns(ribo_norm)
    n = float(n_cells)

    row_abs_sum = torch.zeros(n_ref_genes, dtype=torch.float32)
    row_index = torch.arange(n_ref_genes, dtype=torch.long).unsqueeze(1)
    for g0 in range(0, n_ref_genes, corr_block_size):
        g1 = min(g0 + corr_block_size, n_ref_genes)
        corr_block = (s_std.t() @ r_std[:, g0:g1]) / n  # (G_ref, Gc)
        abs_block = corr_block.abs()
        col_index = torch.arange(g0, g1, dtype=torch.long).unsqueeze(0)
        abs_block = abs_block.masked_fill(row_index == col_index, 0.0)
        row_abs_sum += abs_block.sum(dim=1)

    denom = max(n_ref_genes - 1, 1)
    gene_scores = row_abs_sum / float(denom)
    order = torch.argsort(gene_scores, descending=True)
    model_ranked_genes = reference.gene_names[order.cpu().numpy()]
    overlap_count, overlap_ratio = topk_overlap_ratio(
        model_ranked_genes=model_ranked_genes,
        reference_ranked_genes=reference.gene_names,
        topk=topk,
    )
    effective_topk = min(max(topk, 1), len(model_ranked_genes), len(reference.gene_names))

    return {
        "metric_topk_overlap": float(overlap_ratio),
        "overlap_count": int(overlap_count),
        "topk": int(effective_topk),
        "gene_scores": gene_scores.cpu().numpy(),
        "gene_names": reference.gene_names.copy(),
        "model_ranked_genes": np.asarray(model_ranked_genes, dtype=object),
        "reference_topk_genes": np.asarray(reference.gene_names[:effective_topk], dtype=object),
        "model_topk_genes": np.asarray(model_ranked_genes[:effective_topk], dtype=object),
        "cosine_cg": cosine_cg if save_cosine else None,
    }
