"""跨模态动态注意力融合层."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFusionAttention(nn.Module):
    """
    按 cell 自适应学习 RNA/Ribo 权重，输出融合 embedding 与权重。

    输入形状
    -------
    rna, ribo: (B, D)

    输出
    ----
    fused: (B, D)
    weights: (B, 2)  # [β_RNA, β_Ribo]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w_omega = nn.Parameter(torch.empty(dim, dim))
        self.u_omega = nn.Parameter(torch.empty(dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_omega)
        nn.init.xavier_uniform_(self.u_omega)

    def forward(self, rna: torch.Tensor, ribo: torch.Tensor):
        # stack -> (B, 2, D)
        emb = torch.stack([rna, ribo], dim=1)
        # (B, 2, D) @ (D, D) -> (B, 2, D)
        v = torch.tanh(torch.matmul(emb, self.w_omega))
        # (B, 2, D) @ (D, 1) -> (B, 2, 1) -> (B, 2)
        scores = torch.matmul(v, self.u_omega).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        # 加权求和 -> (B, D)
        fused = torch.sum(emb * weights.unsqueeze(-1), dim=1)
        return fused, weights
