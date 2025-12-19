"""STARNet 风格的权重构造：列归一 -> gamma 放大 -> top-k mask -> 行归一混合."""

from typing import Tuple

import numpy as np
import torch


def build_starnet_weights(
    x: np.ndarray,
    gamma: float = 3.0,
    k_top: int = 30,
    w_resample: float = 0.8,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    输入
    ----
    x: (N_s, N_g) numpy array，原始表达（如 rna_log1p/ribo_log1p）
    gamma: 概率强化指数
    k_top: 每个 spot 选取的基因数（使用 top-k 代替随机采样）
    w_resample: 混合系数，W = (1-w)*G + w*S
    epsilon: 数值稳定项

    输出
    ----
    W: torch.FloatTensor (N_s, N_g)，行为权重和为 1
    """
    n_spots, n_genes = x.shape
    # 列归一
    col_sum = np.sum(x, axis=0, keepdims=True) + epsilon
    tilde = x / col_sum  # (N_s, N_g)

    # gamma 放大
    y = np.power(tilde, gamma)
    # P: 行归一概率
    p = y / (np.sum(y, axis=1, keepdims=True) + epsilon)

    # Global part: 行归一 tilde
    g = tilde / (np.sum(tilde, axis=1, keepdims=True) + epsilon)

    # Top-k mask (per row)
    k = min(k_top, n_genes)
    # argsort 取 top-k
    idx = np.argpartition(-p, kth=k - 1, axis=1)[:, :k]
    mask = np.zeros_like(p)
    row_indices = np.arange(n_spots)[:, None]
    mask[row_indices, idx] = 1.0

    # Salient part: mask * (tilde^gamma) 行归一
    s_num = mask * y
    s = s_num / (np.sum(s_num, axis=1, keepdims=True) + epsilon)

    w = (1 - w_resample) * g + w_resample * s

    return torch.tensor(w, dtype=torch.float32)
