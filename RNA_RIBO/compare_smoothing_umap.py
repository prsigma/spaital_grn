#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import List

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def to_float32(mat):
    if sp.issparse(mat):
        return mat.astype(np.float32)
    return np.asarray(mat, dtype=np.float32)


def align_genes(adata: ad.AnnData, target_genes: List[str]) -> ad.AnnData:
    gene_set = set(adata.var_names)
    keep_genes = [g for g in target_genes if g in gene_set]
    missing = [g for g in target_genes if g not in gene_set]
    print(f"[info] genes matched: {len(keep_genes)} / {len(target_genes)}; missing={len(missing)}")
    if missing:
        print(f"[warn] first 10 missing genes: {missing[:10]}")
    if len(keep_genes) == 0:
        raise ValueError("No genes matched between spatial data and reference genes.")
    return adata[:, keep_genes].copy()


def smooth_counts(coords: np.ndarray, mat: np.ndarray, k: int) -> np.ndarray:
    raise RuntimeError("smooth_counts should not be used without domain info")


def build_adj_with_domain(coords: np.ndarray, domains: np.ndarray, k: int) -> csr_matrix:
    n = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    knn_indices = nbrs.kneighbors(return_distance=False)
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for i in range(n):
        neigh = knn_indices[i]
        same = neigh[domains[neigh] == domains[i]]
        if same.size == 0:
            same = np.array([i])
        rows.extend([i] * len(same))
        cols.extend(same.tolist())
        weights.extend([1.0 / len(same)] * len(same))
    adj = csr_matrix((weights, (rows, cols)), shape=(n, n))
    return adj


def main():
    parser = argparse.ArgumentParser(description="Compare smoothing K on UMAP.")
    parser.add_argument("--input", type=str, default="mousebrain_C1_B4_with_domain.h5ad")
    parser.add_argument("--ref_genes", type=str, default="../rna_ribo_layers_withlabel.h5ad")
    parser.add_argument("--out_dir", type=str, default="smoothing_umap")
    parser.add_argument("--rna_layer", type=str, default="totalRNA_norm")
    parser.add_argument("--ribo_layer", type=str, default="rbRNA_norm")
    parser.add_argument("--smooth_k", type=int, nargs="+", default=[0, 4, 8, 12, 15])
    parser.add_argument("--sample", type=int, default=20000, help="Optional subsample for faster UMAP; 0=all")
    parser.add_argument("--umap_neighbors", type=int, default=50)
    parser.add_argument("--umap_min_dist", type=float, default=0.001)
    parser.add_argument("--umap_spread", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = str(out_dir)

    ref = ad.read_h5ad(Path(args.ref_genes), backed="r")
    target_genes = list(ref.var_names)
    ref.file.close()

    print(f"[info] load spatial data {args.input}")
    adata = ad.read_h5ad(args.input)
    if args.rna_layer not in adata.layers or args.ribo_layer not in adata.layers:
        raise ValueError(
            f"layers must include '{args.rna_layer}' and '{args.ribo_layer}', found {list(adata.layers.keys())}"
        )
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
    elif all(c in adata.obs.columns for c in ["column", "row", "z"]):
        coords = adata.obs[["column", "row", "z"]].to_numpy()
    else:
        raise ValueError("No spatial coordinates found (need obsm['spatial'] or obs columns column/row/z)")
    if "domain" not in adata.obs.columns:
        raise ValueError("obs must contain 'domain' for domain-restricted smoothing")
    domains = adata.obs["domain"].to_numpy()

    adata = align_genes(adata, target_genes)
    coords = coords[: adata.n_obs]  # align obs if needed

    rna_norm = to_float32(adata.layers[args.rna_layer])
    ribo_norm = to_float32(adata.layers[args.ribo_layer])

    metrics_rows = []
    for k in args.smooth_k:
        print(f"[info] smoothing with k={k}")
        adj = build_adj_with_domain(coords, domains, k) if k > 0 else csr_matrix(
            (np.ones(coords.shape[0], dtype=np.float32), (np.arange(coords.shape[0]), np.arange(coords.shape[0]))),
            shape=(coords.shape[0], coords.shape[0]),
        )
        rna_smooth = adj @ rna_norm
        ribo_smooth = adj @ ribo_norm
        rna_log1p = np.log1p(rna_smooth).astype(np.float32)
        
        tmp = ad.AnnData(X=rna_log1p, obs=adata.obs.copy())
        tmp.layers["rna_log1p"] = rna_log1p
        tmp.layers["ribo_log1p"] = np.log1p(ribo_smooth).astype(np.float32)
        tmp.layers["rna_raw_smooth"] = rna_smooth.astype(np.float32)
        tmp.layers["ribo_raw_smooth"] = ribo_smooth.astype(np.float32)
        # Seurat-like pipeline: HVG 1500, scale, PCA 100, neighbors k=20 (PCs 1:25) for clustering,
        # UMAP with neighbors=50, min_dist=0.001, spread=3
        sc.pp.highly_variable_genes(
            tmp,
            n_top_genes=1500,
            subset=False,
            layer=None,
        )
        sc.pp.scale(tmp, max_value=None)
        sc.tl.pca(tmp, n_comps=100, use_highly_variable=True, random_state=args.seed)
        sc.pp.neighbors(
            tmp,
            n_neighbors=20,
            n_pcs=25,
            random_state=args.seed,
        )
        # 固定簇数（已知为 6），使用 KMeans 在前 25 个 PC 上聚类
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=6, random_state=args.seed, n_init="auto")
        tmp.obs["states_nn_alg1"] = km.fit_predict(tmp.obsm["X_pca"][:, :25])
        sc.tl.umap(tmp, min_dist=args.umap_min_dist, spread=args.umap_spread, random_state=args.seed)
        
        colors = []
        for c in ["domain"]:
            if c in tmp.obs.columns:
                colors.append(c)
        if not colors:
            colors = None
        sc.pl.umap(
            tmp,
            color=colors,
            show=False,
            wspace=0.4,
            save=f"_smoothk{k}.png",
            edges=False,
        )
       
        fig_path = Path(sc.settings.figdir) / f"umap_smoothk{k}.png"
        if not fig_path.exists():
            fig_path = Path(f"figures/umap_smoothk{k}.png")
        if fig_path.exists():
            fig_path.rename(out_dir / fig_path.name)
            print(f"[done] saved {out_dir / fig_path.name}")
        else:
            print("[warn] figure not found after plotting")
        
        if "domain" in tmp.obs.columns and tmp.obs["domain"].nunique() > 1:
            dom = tmp.obs["domain"]
            row = {"k": k, "clusters": tmp.obs["states_nn_alg1"].nunique()}
            row["ari_domain"] = adjusted_rand_score(dom, tmp.obs["states_nn_alg1"])
            row["nmi_domain"] = normalized_mutual_info_score(dom, tmp.obs["states_nn_alg1"])
            try:
                row["silhouette_domain"] = silhouette_score(tmp.obsm["X_umap"], dom)
            except Exception:
                row["silhouette_domain"] = np.nan
            metrics_rows.append(row)

        
        h5_path = out_dir / f"smoothk{k}.h5ad"
        tmp.write(h5_path)
        print(f"[done] saved smoothed h5ad to {h5_path}")

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics_smoothing.csv", index=False)
        print(f"[done] metrics saved to {out_dir/'metrics_smoothing.csv'}")


if __name__ == "__main__":
    sc.settings.figdir = "."
    main()
