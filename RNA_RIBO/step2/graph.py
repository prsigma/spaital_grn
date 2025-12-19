"""图构建：仅基于空间坐标的邻接，归一化并转为 torch 稀疏矩阵."""

from typing import Optional

import numpy as np
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph


def _symmetrize(adj: sparse.spmatrix) -> sparse.spmatrix:
    """强制邻接矩阵对称且去重."""
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def _normalize(adj: sparse.spmatrix, add_self_loops: bool = True) -> sparse.spmatrix:
    """D^-1/2 (A + I) D^-1/2 归一化，避免度为 0 的行产生 inf."""
    if add_self_loops:
        adj = adj + sparse.eye(adj.shape[0], dtype=np.float32, format="csr")
    adj = adj.tocsr()
    rowsum = np.asarray(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sparse.diags(d_inv_sqrt)
    return d_mat @ adj @ d_mat


def _to_torch_sparse(adj: sparse.spmatrix) -> torch.Tensor:
    """scipy 稀疏 -> torch 稀疏张量."""
    adj = adj.tocoo().astype(np.float32)
    indices = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
    values = torch.tensor(adj.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, adj.shape)


def build_spatial_knn_graph(
    coords: np.ndarray,
    k: int = 10,
    include_self: bool = False,
    normalize: bool = True,
    n_jobs: Optional[int] = -1,
    labels: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    基于空间坐标的 KNN 邻接，返回 torch 稀疏张量（归一化后）。

    参数
    ----
    coords: (n_cells, 2/3) 空间坐标
    k: 近邻数
    include_self: 是否保留自环（归一化前后仍会加 self-loop）
    normalize: 是否做 D^-1/2 A D^-1/2 归一化
    n_jobs: 并行线程数，默认 -1
    """
    adj = kneighbors_graph(
        coords, n_neighbors=k, mode="connectivity", include_self=include_self, n_jobs=n_jobs
    )
    adj = _symmetrize(adj)
    if labels is not None:
        # 去掉跨域边（labels 长度与 coords 匹配）
        adj = adj.tocoo()
        rows, cols, vals = [], [], []
        lab = np.asarray(labels)
        for i, j, v in zip(adj.row, adj.col, adj.data):
            if lab[i] == lab[j]:
                rows.append(i)
                cols.append(j)
                vals.append(v)
        adj = sparse.coo_matrix((vals, (rows, cols)), shape=adj.shape)
    if normalize:
        adj = _normalize(adj)
    return _to_torch_sparse(adj)


def build_spatial_radius_graph(
    coords: np.ndarray,
    radius: float,
    max_neighbors: Optional[int] = None,
    include_self: bool = False,
    normalize: bool = True,
    n_jobs: Optional[int] = -1,
) -> torch.Tensor:
    """
    基于空间半径的邻接，返回 torch 稀疏张量。

    参数
    ----
    coords: (n_cells, 2/3) 坐标
    radius: 半径阈值
    max_neighbors: 限制每个节点最大邻居数（可避免过密）
    include_self: 是否包含自环
    normalize: 是否归一化
    """
    adj = radius_neighbors_graph(
        coords, radius=radius, mode="connectivity", include_self=include_self, n_jobs=n_jobs
    )
    if max_neighbors is not None:
        # 裁剪过多邻居：逐行保留最近的 max_neighbors（需距离信息）
        nbrs = NearestNeighbors(radius=radius, n_jobs=n_jobs).fit(coords)
        dists, indices = nbrs.radius_neighbors(coords, return_distance=True)
        rows, cols = [], []
        for i, (dist_row, idx_row) in enumerate(zip(dists, indices)):
            if len(idx_row) == 0:
                continue
            order = np.argsort(dist_row)[:max_neighbors]
            rows.extend([i] * len(order))
            cols.extend(idx_row[order])
        adj = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=adj.shape)
    adj = _symmetrize(adj)
    if normalize:
        adj = _normalize(adj)
    return _to_torch_sparse(adj)


def build_graphs(
    coords: np.ndarray,
    k_spatial: int = 10,
    radius: Optional[float] = None,
    max_neighbors_radius: Optional[int] = None,
    labels: Optional[np.ndarray] = None,
) -> dict:
    """
    构建空间邻接图（KNN 或半径）。

    返回
    ----
    dict 包含：
        - adj_spatial: torch 稀疏邻接（KNN 或 radius）
    """
    adj_spatial = (
        build_spatial_radius_graph(
            coords,
            radius=radius,
            max_neighbors=max_neighbors_radius,
            include_self=False,
            normalize=True,
        )
        if radius is not None
        else build_spatial_knn_graph(coords[:, :2], k=k_spatial, include_self=False, normalize=True)
    )

    # 如果提供标签，去掉跨域边
    if labels is not None:
        adj = adj_spatial.coalesce()
        idx = adj.indices()
        val = adj.values()
        keep = labels[idx[0]] == labels[idx[1]]
        adj = torch.sparse_coo_tensor(idx[:, keep], val[keep], size=adj.shape)
        adj_spatial = adj.coalesce()

    return {"adj_spatial": adj_spatial}
