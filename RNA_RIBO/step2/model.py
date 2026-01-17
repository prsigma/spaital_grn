"""空间感知的跨模态融合模型（无需预训练，端到端训练）。"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import DynamicFusionAttention


class Encoder(nn.Module):
    """将原始基因表达编码到低维embedding空间。"""

    def __init__(self, n_genes: int, dim: int, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.1):
        """
        参数
        ----
        n_genes: 基因数量（输入维度）
        dim: embedding维度（输出维度）
        hidden_dim: 隐藏层维度
        num_layers: MLP层数
        dropout: dropout率
        """
        super().__init__()
        self.n_genes = n_genes
        self.dim = dim

        # 构建MLP
        layers = []
        in_dim = n_genes
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x: (N, n_genes) 原始基因表达

        返回
        ----
        z: (N, dim) embedding
        """
        return self.encoder(x)


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
    """解码器：从embedding重构回基因表达空间（修复后版本）。"""

    def __init__(self, in_dim: int, out_dim: int):
        """
        参数
        ----
        in_dim: embedding维度（例如128）
        out_dim: 基因数量（例如1867）- 修复：输出到基因空间
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x: (N, in_dim) embedding
        adj: (N, N) 空间邻接矩阵

        返回
        ----
        recon: (N, out_dim) 重构的基因表达
        """
        x = torch.mm(x, self.weight)  # (N, in_dim) @ (in_dim, out_dim) = (N, out_dim)
        x = torch.spmm(adj, x)  # (N, N) @ (N, out_dim) = (N, out_dim)
        return x


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
    完整的端到端空间融合模型（无需预训练）。

    架构流程：
    1. 原始表达 → Encoder → embedding
    2. 可选：+基因embedding（STARNet风格）
    3. 动态注意力融合
    4. GCN空间精炼
    5. Decoder重构 + 分类器
    """

    def __init__(
        self,
        n_genes: int,
        dim: int = 128,
        encoder_hidden: int = 512,
        encoder_layers: int = 2,
        encoder_dropout: float = 0.1,
        gcn_hidden: int = 128,
        gcn_layers: int = 2,
        gcn_dropout: float = 0.0,
        use_decoder: bool = True,
        temperature: float = 0.07,
        gene_dim: Optional[int] = None,
        w_tx: Optional[torch.Tensor] = None,
        w_ribo: Optional[torch.Tensor] = None,
    ):
        """
        参数
        ----
        n_genes: 基因数量（输入/输出维度）
        dim: embedding维度
        encoder_hidden: encoder隐藏层维度
        encoder_layers: encoder层数
        encoder_dropout: encoder dropout率
        gcn_hidden: GCN隐藏层维度
        gcn_layers: GCN层数
        gcn_dropout: GCN dropout率
        use_decoder: 是否使用decoder重构
        temperature: 对比损失温度
        gene_dim: 基因embedding维度（可选，STARNet风格）
        w_tx: RNA的STARNet权重矩阵 (N_cells, N_genes)
        w_ribo: RIBO的STARNet权重矩阵
        """
        super().__init__()
        self.n_genes = n_genes
        self.dim = dim

        # 编码器：原始表达 → embedding
        self.encoder_rna = Encoder(n_genes, dim, encoder_hidden, encoder_layers, encoder_dropout)
        self.encoder_ribo = Encoder(n_genes, dim, encoder_hidden, encoder_layers, encoder_dropout)

        # 可选：基因embedding（STARNet风格）
        self.use_gene_emb = gene_dim is not None and w_tx is not None and w_ribo is not None
        if self.use_gene_emb:
            self.E_tx = nn.Parameter(torch.randn(w_tx.shape[1], gene_dim) * 0.01)
            self.E_ribo = nn.Parameter(torch.randn(w_ribo.shape[1], gene_dim) * 0.01)
            self.register_buffer("W_tx", w_tx)
            self.register_buffer("W_ribo", w_ribo)
            if gene_dim != dim:
                self.tx_proj = nn.Linear(gene_dim, dim, bias=False)
                self.ribo_proj = nn.Linear(gene_dim, dim, bias=False)
            else:
                self.tx_proj = None
                self.ribo_proj = None

        # 动态注意力融合
        self.fusion = DynamicFusionAttention(dim)

        # GCN空间精炼
        gcn_dims = [dim] + [gcn_hidden] * (gcn_layers - 1) + [dim]
        self.gcn_blocks = nn.ModuleList(
            [
                GCNBlock(
                    gcn_dims[i],
                    gcn_dims[i + 1],
                    dropout=gcn_dropout,
                    activation=F.relu if i < gcn_layers - 1 else None,
                )
                for i in range(gcn_layers)
            ]
        )

        # 解码器：重构回基因表达空间（修复后）
        self.use_decoder = use_decoder
        if use_decoder:
            self.decoder_rna = Decoder(dim, n_genes)  # 输出维度改为n_genes
            self.decoder_ribo = Decoder(dim, n_genes)

        self.temperature = temperature

    def forward(
        self,
        rna_expr: torch.Tensor,
        ribo_expr: torch.Tensor,
        adj_spatial: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        参数
        ----
        rna_expr: (N, n_genes) 原始RNA表达
        ribo_expr: (N, n_genes) 原始RIBO表达
        adj_spatial: (N, N) 空间邻接矩阵（稀疏）

        返回
        ----
        outputs: 包含所有中间结果的字典
        """
        # 1. 编码：原始表达 → embedding
        z_rna = self.encoder_rna(rna_expr)  # (N, n_genes) → (N, dim)
        z_ribo = self.encoder_ribo(ribo_expr)  # (N, n_genes) → (N, dim)

        # 2. 可选：加入基因embedding（STARNet风格）
        if self.use_gene_emb:
            z_tx_gene = torch.sparse.mm(self.W_tx, self.E_tx) if self.W_tx.is_sparse else self.W_tx @ self.E_tx
            z_ribo_gene = torch.sparse.mm(self.W_ribo, self.E_ribo) if self.W_ribo.is_sparse else self.W_ribo @ self.E_ribo
            if self.tx_proj is not None:
                z_tx_gene = self.tx_proj(z_tx_gene)
                z_ribo_gene = self.ribo_proj(z_ribo_gene)
            z_rna = z_rna + z_tx_gene
            z_ribo = z_ribo + z_ribo_gene

        # 3. 动态注意力融合
        fused, weights = self.fusion(z_rna, z_ribo)  # (N, dim), (N, 2)

        # 4. GCN空间精炼
        h = fused
        for block in self.gcn_blocks:
            h = block(h, adj_spatial)

        # 5. 构建输出字典
        out = {
            "fused": fused,  # 融合后的特征
            "h_final": h,  # GCN精炼后的最终embedding
            "weights": weights,  # 融合权重 [β_RNA, β_Ribo]
            "z_rna": z_rna,  # RNA embedding（用于对比损失）
            "z_ribo": z_ribo,  # RIBO embedding（用于对比损失）
        }

        # 6. 解码器：重构回基因表达空间
        if self.use_decoder:
            out["recon_rna"] = self.decoder_rna(h, adj_spatial)  # (N, n_genes)
            out["recon_ribo"] = self.decoder_ribo(h, adj_spatial)  # (N, n_genes)

        return out

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        rna_expr: torch.Tensor,
        ribo_expr: torch.Tensor,
        adj_spatial: torch.Tensor,
        lambda_recon: float = 1.0,
        lambda_contrast: float = 0.5,
        lambda_link: float = 0.2,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算多任务损失（无监督版本）。

        参数
        ----
        outputs: forward的输出字典
        rna_expr: (N, n_genes) 原始RNA表达 - 用于重构损失
        ribo_expr: (N, n_genes) 原始RIBO表达 - 用于重构损失
        adj_spatial: (N, N) 空间邻接矩阵
        lambda_recon: 重构损失权重
        lambda_contrast: 对比损失权重
        lambda_link: 链接预测损失权重

        返回
        ----
        total_loss: 总损失
        loss_dict: 各项损失的字典
        """
        loss_dict: Dict[str, torch.Tensor] = {}

        # 1. 重构损失（修复：目标改为原始表达）
        if "recon_rna" in outputs and "recon_ribo" in outputs:
            loss_dict["recon_rna"] = F.mse_loss(outputs["recon_rna"], rna_expr)
            loss_dict["recon_ribo"] = F.mse_loss(outputs["recon_ribo"], ribo_expr)
            loss_dict["recon"] = loss_dict["recon_rna"] + loss_dict["recon_ribo"]
        else:
            loss_dict["recon"] = torch.tensor(0.0, device=rna_expr.device)

        # 2. 对比损失（修复：现在可以正常训练encoder）
        if "z_rna" in outputs and "z_ribo" in outputs:
            loss_dict["contrast"] = contrastive_nce_loss(
                outputs["z_rna"], outputs["z_ribo"], temperature=self.temperature
            )
        else:
            loss_dict["contrast"] = torch.tensor(0.0, device=rna_expr.device)

        # 3. 链接预测损失（图结构保持）
        loss_dict["link"] = link_prediction_loss(outputs["h_final"], adj_spatial)

        # 4. 加权总损失
        total = (
            lambda_recon * loss_dict["recon"]
            + lambda_contrast * loss_dict["contrast"]
            + lambda_link * loss_dict["link"]
        )
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
