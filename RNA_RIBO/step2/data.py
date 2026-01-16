"""Data loading utilities for spatial RNA/Ribo multi-omics."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy import sparse


@dataclass
class SpatialMultiOmics:
    """Container for spatial multi-omics inputs."""

    adata: AnnData
    rna: np.ndarray
    ribo: np.ndarray
    coords: np.ndarray
    labels: Optional[np.ndarray]


def _to_dense_float(arr) -> np.ndarray:
    """Convert sparse/arraylike to dense float32 numpy."""
    if sparse.issparse(arr):
        arr = arr.A
    return np.asarray(arr, dtype=np.float32)


def load_spatial_multiome(
    h5ad_path: str,
    rna_layer: str = "rna_log1p",
    ribo_layer: str = "ribo_log1p",
    coord_keys: Sequence[str] = ("row", "column"),
    label_key: Optional[str] = "domain",
) -> SpatialMultiOmics:
    """
    Load RNA/Ribo modalities plus coordinates/labels from an h5ad file.

    Parameters
    ----------
    h5ad_path:
        Path to the AnnData file (e.g., RNA_RIBO/smoothing_umap/smoothk15.h5ad).
    rna_layer:
        Layer name for RNA counts/log-expression. Use "X" to read from adata.X.
    ribo_layer:
        Layer name for Ribo counts/log-expression.
    coord_keys:
        obs columns holding spatial coordinates (e.g., ("row", "column")).
    label_key:
        obs column for domain labels. If None or missing, labels will be None.

    Returns
    -------
    SpatialMultiOmics
        Contains dense float32 arrays for RNA, Ribo, coords, labels and the AnnData object.
    """
    adata = sc.read_h5ad(h5ad_path)

    def pick_layer(name: str):
        if name == "X":
            return adata.X
        if name in adata.layers:
            return adata.layers[name]
        raise KeyError(f"Layer '{name}' not found. Available: {list(adata.layers.keys())}")

    rna = _to_dense_float(pick_layer(rna_layer))
    ribo = _to_dense_float(pick_layer(ribo_layer))

    coords = adata.obs.loc[:, coord_keys].to_numpy(dtype=np.float32)
    labels = None
    if label_key is not None and label_key in adata.obs:
        labels = adata.obs[label_key].to_numpy()

    return SpatialMultiOmics(
        adata=adata,
        rna=rna,
        ribo=ribo,
        coords=coords,
        labels=labels,
    )
