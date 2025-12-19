"""空间感知的跨模态融合模型（GCN 精炼 + 解码 + 多种损失）。"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import DynamicFusionAttention


class GraphConvolution(nn.Module):
    """简化版 GCN 层（源自 SpaGCN 实现）。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.weight.size(1) ** 0.5)
        nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        out = torch.spmm(adj, support)
        if self.bias is not None:
            out = out + self.bias
        return out


class GCNBlock(nn.Module):
    """GCN + 激活 + Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, activation=F.relu):
        super().__init__()
        self.gcn = GraphConvolution(in_dim, out_dim)
        self.dropout = dropout
        self.activation = activation

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.gcn(x, adj)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class Decoder(nn.Module):
    """线性+图乘重构（参考 SpatialGlue Decoder）。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.mm(x, self.weight)
        x = torch.spmm(adj, x)
        return x


class Adapter(nn.Module):
    """简单瓶颈 Adapter，用于将预训练 embedding 适配到单细胞."""

    def __init__(self, dim: int, bottleneck: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, bottleneck)
        self.fc2 = nn.Linear(bottleneck, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fc2(h)
        return x + h  # 残差


def contrastive_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    对齐 Z_RNA vs Z_Ribo（对称 InfoNCE）。
    假设同索引为正样本，余弦相似度作为 logits。
    """
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    logits = torch.matmul(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss12 = F.cross_entropy(logits, labels)
    loss21 = F.cross_entropy(logits.t(), labels)
    return (loss12 + loss21) * 0.5


def link_prediction_loss(
    h: torch.Tensor,
    adj: torch.Tensor,
    max_pos_edges: int = 20000,
    neg_ratio: int = 1,
) -> torch.Tensor:
    """
    STARNet 风格的邻边内积二分类损失。

    参数
    ----
    h: 最终 embedding (N, D)
    adj: 归一化或未归一化的稀疏邻接 (torch.sparse_coo_tensor)
    max_pos_edges: 正样本上限（随机截断，防止过大图）
    neg_ratio: 负样本/正样本比
    """
    edge_index = adj.coalesce().indices().t()  # (E, 2)
    # 只取 i < j 去重
    mask = edge_index[:, 0] < edge_index[:, 1]
    edge_index = edge_index[mask]
    if edge_index.size(0) > max_pos_edges:
        perm = torch.randperm(edge_index.size(0), device=h.device)[:max_pos_edges]
        edge_index = edge_index[perm]

    pos_i, pos_j = edge_index[:, 0], edge_index[:, 1]
    pos_score = torch.sum(h[pos_i] * h[pos_j], dim=1)
    pos_label = torch.ones_like(pos_score)

    # 负样本随机采样
    num_neg = edge_index.size(0) * neg_ratio
    n = h.size(0)
    neg_i = torch.randint(0, n, (num_neg,), device=h.device)
    neg_j = torch.randint(0, n, (num_neg,), device=h.device)
    neg_score = torch.sum(h[neg_i] * h[neg_j], dim=1)
    neg_label = torch.zeros_like(neg_score)

    scores = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat([pos_label, neg_label], dim=0)
    return F.binary_cross_entropy_with_logits(scores, labels)


class SpatialFusionModel(nn.Module):
    """
    融合 + GCN 精炼 + 解码 + 多头损失的总体模型。

    输入：预训练得到的 Z_RNA, Z_Ribo 以及空间邻接矩阵 adj_spatial。
    """

    def __init__(
        self,
        dim: int,
        gcn_hidden: int = 128,
        gcn_layers: int = 1,
        dropout: float = 0.0,
        use_decoder: bool = True,
        temperature: float = 0.07,
        adapter_dim: Optional[int] = None,
        adapter_dropout: float = 0.0,
        gene_dim: Optional[int] = None,
        w_tx: Optional[torch.Tensor] = None,
        w_ribo: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.adapter_rna = Adapter(dim, adapter_dim, adapter_dropout) if adapter_dim else None
        self.adapter_ribo = Adapter(dim, adapter_dim, adapter_dropout) if adapter_dim else None
        self.use_gene_emb = gene_dim is not None and w_tx is not None and w_ribo is not None
        if self.use_gene_emb:
            self.E_tx = nn.Parameter(torch.randn(w_tx.shape[1], gene_dim))
            self.E_ribo = nn.Parameter(torch.randn(w_ribo.shape[1], gene_dim))
            self.register_buffer("W_tx", w_tx)
            self.register_buffer("W_ribo", w_ribo)
            if gene_dim != dim:
                self.tx_proj = nn.Linear(gene_dim, dim, bias=False)
                self.ribo_proj = nn.Linear(gene_dim, dim, bias=False)
            else:
                self.tx_proj = None
                self.ribo_proj = None
        self.fusion = DynamicFusionAttention(dim)
        gcn_dims = [dim] + [gcn_hidden] * (gcn_layers - 1) + [dim]
        self.gcn_blocks = nn.ModuleList(
            [
                GCNBlock(gcn_dims[i], gcn_dims[i + 1], dropout=dropout, activation=F.relu if i < gcn_layers - 1 else None)
                for i in range(gcn_layers)
            ]
        )
        self.use_decoder = use_decoder
        if use_decoder:
            self.decoder_rna = Decoder(dim, dim)
            self.decoder_ribo = Decoder(dim, dim)
        self.temperature = temperature

    def forward(
        self,
        z_rna: torch.Tensor,
        z_ribo: torch.Tensor,
        adj_spatial: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.use_gene_emb:
            z_tx_gene = torch.sparse.mm(self.W_tx, self.E_tx) if self.W_tx.is_sparse else self.W_tx @ self.E_tx
            z_ribo_gene = torch.sparse.mm(self.W_ribo, self.E_ribo) if self.W_ribo.is_sparse else self.W_ribo @ self.E_ribo
            if self.tx_proj is not None:
                z_tx_gene = self.tx_proj(z_tx_gene)
                z_ribo_gene = self.ribo_proj(z_ribo_gene)
            z_rna = z_rna + z_tx_gene
            z_ribo = z_ribo + z_ribo_gene

        if self.adapter_rna is not None:
            z_rna = self.adapter_rna(z_rna)
        if self.adapter_ribo is not None:
            z_ribo = self.adapter_ribo(z_ribo)
        fused, weights = self.fusion(z_rna, z_ribo)
        h = fused
        for block in self.gcn_blocks:
            h = block(h, adj_spatial)

        out = {"fused": fused, "h_final": h, "weights": weights}
        if self.use_decoder:
            out["recon_rna"] = self.decoder_rna(h, adj_spatial)
            out["recon_ribo"] = self.decoder_ribo(h, adj_spatial)
        return out

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        z_rna: torch.Tensor,
        z_ribo: torch.Tensor,
        adj_spatial: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        仅保留域质心约束损失。
        """
        loss_dict: Dict[str, torch.Tensor] = {}

        loss_dict["domain"] = (
            domain_center_loss(outputs["h_final"], labels)
            if labels is not None
            else torch.tensor(0.0, device=z_rna.device)
        )
        total = loss_dict["domain"]
        return total, loss_dict


def domain_center_loss(
    h: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1,
) -> torch.Tensor:
    """
    质心约束：同域拉近、不同域推远。

    - pull: 样本到各自域质心的均方距离
    - push: 域质心两两之间保持至少 margin 的距离（小于 margin 则惩罚）
    """
    device = h.device
    uniq = labels.unique()
    centers = []
    for c in uniq:
        mask = labels == c
        centers.append(h[mask].mean(dim=0))
    centers = torch.stack(centers, dim=0)  # (C, D)

    # pull term
    pull = 0.0
    for center, c in zip(centers, uniq):
        diff = h[labels == c] - center
        pull = pull + (diff.pow(2).sum(dim=1)).mean()
    pull = pull / len(uniq)

    # push term
    if centers.size(0) > 1:
        dist_mat = torch.cdist(centers, centers, p=2)
        # 只取上三角
        mask = torch.triu(torch.ones_like(dist_mat), diagonal=1).bool()
        dists = dist_mat[mask]
        push = F.relu(margin - dists).pow(2).mean()
    else:
        push = torch.tensor(0.0, device=device)

    return pull + push
