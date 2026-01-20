import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# === 基础组件：LayerNorm (与您原代码保持一致) ===
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# === CoLA-Former 核心组件 ===

class CoLAttention(nn.Module):
    """
    Communal Linear Attention (CoLAttention) as described in the paper.
    Uses shared C_K and C_V matrices instead of input-dependent K/V.
    Complexity: O(N * alpha * d)
    """

    def __init__(self, in_dim, alpha=64, dropout=0.1):
        super(CoLAttention, self).__init__()
        self.in_dim = in_dim
        self.alpha = alpha  # Hyper-parameter controlling dimension of communal units [cite: 2073]
        self.dropout = nn.Dropout(dropout)

        # Communal Units C_K and C_V
        # These are shared across batches (Parameter)
        self.C_K = nn.Parameter(torch.Tensor(in_dim, alpha))
        self.C_V = nn.Parameter(torch.Tensor(in_dim, alpha))

        # Query projection (Standard linear transformation)
        self.W_Q = nn.Linear(in_dim, in_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.C_K)
        nn.init.xavier_uniform_(self.C_V)
        nn.init.xavier_uniform_(self.W_Q.weight)
        if self.W_Q.bias is not None:
            nn.init.constant_(self.W_Q.bias, 0)

    def forward(self, x, mask=None):
        # x: [Batch, Length, Dim]
        B, L, D = x.size()

        # 1. Generate Query: Q = x * W_Q
        Q = self.W_Q(x)  # [B, L, D]

        # 2. Compute CoLAttention Map: a_{i,j} = Norm(Q * C_K / sqrt(d)) [cite: 2070]
        # Q: [B, L, D], C_K: [D, alpha] -> A: [B, L, alpha]
        A = torch.matmul(Q, self.C_K) / math.sqrt(D)

        # Dual Normalization (Softmax usually acts on the last dimension,
        # but paper mentions normalizing columns and rows separately).
        # Here we apply Softmax to mimic the "Norm" operation which makes it attention-like.
        A = F.softmax(A, dim=-1)  # Normalize over alpha dimension

        if mask is not None:
            # Apply mask if necessary (though CoLAttention is global, mask handles padding)
            mask = mask.unsqueeze(-1)  # [B, L, 1]
            A = A * mask.float()

        # 3. Compute Output: e_hat = Norm(a_{i,j} * C_V^T) [cite: 2070]
        # A: [B, L, alpha], C_V.T: [alpha, D] -> Out: [B, L, D]
        # Note: Paper says C_V^T. The shape match is (alpha, D).
        output = torch.matmul(A, self.C_V.t())

        return self.dropout(output)


class PointWiseConv1D(nn.Module):
    """
    Point-wise Conv1D to capture local features
    Replaces the FFN in standard Transformer.
    """

    def __init__(self, in_dim, dropout=0.1):
        super(PointWiseConv1D, self).__init__()
        # Kernel size = 1, Stride = 1 [cite: 2058]
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(in_dim)
        self.activation = nn.LeakyReLU()  # Typically Conv layers use ReLU/LeakyReLU

    def forward(self, x):
        # x: [Batch, Length, Dim]
        # Conv1d expects [Batch, Dim, Length]
        residual = x
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(self.activation(x))
        return self.norm(residual + x)


class CoLAEncoderLayer(nn.Module):
    """
    Single Layer of CoLA-Former Encoder
    Structure: CoLAttention -> Add&Norm -> Point-Wise Conv1D -> Add&Norm
    """

    def __init__(self, in_dim, alpha=64, dropout=0.1):
        super(CoLAEncoderLayer, self).__init__()
        self.col_attention = CoLAttention(in_dim, alpha, dropout)
        self.norm1 = LayerNorm(in_dim)
        self.point_conv = PointWiseConv1D(in_dim, dropout)

    def forward(self, x, mask=None):
        # 1. CoLAttention Block
        residual = x
        x = self.col_attention(x, mask)
        x = self.norm1(residual + x)  # Add & Norm

        # 2. Point-Wise Conv1D Block (Local Aggregation)
        # The PointWiseConv1D class handles Add & Norm internally or we do it here.
        # Based on Fig 2, it's: Input -> Conv -> Add&Norm.
        # My PointWiseConv1D class above does: Conv -> Add(residual) -> Norm.
        x = self.point_conv(x)

        return x


class CoLAEncoder(nn.Module):
    def __init__(self, in_dim, alpha=64, dropout=0.1, N=2):
        super(CoLAEncoder, self).__init__()
        self.layers = nn.ModuleList([
            CoLAEncoderLayer(in_dim, alpha, dropout) for _ in range(N)
        ])
        self.norm = LayerNorm(in_dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# === 完整的 CoLA-Former 模型 (适配 Link Prediction) ===

class CoLA_Former_Model(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1, N=2):
        super(CoLA_Former_Model, self).__init__()
        # 注意: n_heads 在 CoLAttention 中不是主要参数(由 alpha 控制)，但为了接口兼容保留
        # alpha 取值参考论文 RQ4 分析，64 是一个稳健的值 [cite: 2254]
        self.alpha = 64

        # 使用 CoLAEncoder 替换原有的 Spatial/Temporal Encoder
        self.spatial = CoLAEncoder(in_dim, alpha=self.alpha, dropout=dropout, N=N)
        self.temporal = CoLAEncoder(in_dim, alpha=self.alpha, dropout=dropout, N=N)

        # 下游预测层 (保持不变以公平对比)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 4, in_dim),
            nn.LeakyReLU()
        )
        self.affinity_score = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(64, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )
        self.norm = LayerNorm(in_dim, eps=1e-6)
        self.linear = nn.Linear(in_dim * 2, in_dim)
        self.embed_norm = LayerNorm(in_dim, eps=1e-6)
        self.norm2 = LayerNorm(in_dim, eps=1e-6)

    def forward(self, spa_seq, tem_seq, spa_mask, tem_mask):
        # Spatial View
        spatial_output = self.spatial(spa_seq, spa_mask)
        spatial_output_1 = spatial_output[:, 0, :]  # Target Node
        spatial_output_2 = torch.mean(spatial_output[:, 1:, :], dim=1)  # Context Mean

        # Temporal View
        temporal_output = self.temporal(tem_seq, tem_mask)
        temporal_output_1 = temporal_output[:, 0, :]
        temporal_output_2 = torch.mean(temporal_output[:, 1:, :], dim=1)

        # Concat
        output = torch.cat([spatial_output_1, spatial_output_2, temporal_output_1, temporal_output_2], dim=1)
        output = self.mlp(output)
        return output.squeeze(dim=1)

    def getEmbed(self, spa_seq, tem_seq, spa_mask, tem_mask):
        spatial_output = self.spatial(spa_seq, spa_mask)
        spatial_output_1 = spatial_output[:, 0, :]
        spatial_output_2 = torch.mean(spatial_output[:, 1:, :], dim=1)

        temporal_output = self.temporal(tem_seq, tem_mask)
        temporal_output_1 = temporal_output[:, 0, :]
        temporal_output_2 = torch.mean(temporal_output[:, 1:, :], dim=1)

        output = torch.cat([spatial_output_1, spatial_output_2, temporal_output_1, temporal_output_2], dim=1)
        output = self.mlp(output)
        output = self.embed_norm(output)
        return output

    def linkPredict(self, src_spa, src_mask1, src_tmp, src_mask2, tgt_spa, tgt_mask1, tgt_tmp, tgt_mask2, fake_spa,
                    fake_mask1, fake_tmp, fake_mask2):
        # 这一部分逻辑完全复用，确保评估指标(AUC/AP)的一致性
        src_embed = self.getEmbed(src_spa, src_tmp, src_mask1, src_mask2)
        tgt_embed = self.getEmbed(tgt_spa, tgt_tmp, tgt_mask1, tgt_mask2)
        fake_embed = self.getEmbed(fake_spa, fake_tmp, fake_mask1, fake_mask2)

        pos_embed = torch.cat([src_embed, tgt_embed], dim=1)
        neg_embed = torch.cat([src_embed, fake_embed], dim=1)

        pos_embed = self.norm(self.linear(pos_embed))
        neg_embed = self.norm(self.linear(neg_embed))

        pos_score = self.affinity_score(pos_embed).squeeze(dim=1)
        neg_score = self.affinity_score(neg_embed).squeeze(dim=1)

        return pos_score.sigmoid(), neg_score.sigmoid()


# === 保留原模型以供 main.py 调用 ===
# (为了节省篇幅，这里假设 SpatialTemporal 类代码也存在于此文件中，或者您从原文件复制过来)
# 请务必将您原有的 SpatialTemporal 类代码粘贴在 model.py 中，以便对比。
from graph_transformer import *  # 确保您的原始 Transformer 定义可用


class SpatialTemporal(nn.Module):
    # ... (粘贴您原来的 SpatialTemporal 代码) ...
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1, N=2):
        super(SpatialTemporal, self).__init__()
        self.spatial = Encoder(EncoderLayer(in_dim, out_dim, n_heads, dropout), N)
        self.temporal = Encoder(EncoderLayer(in_dim, out_dim, n_heads, dropout), N)
        # ... (其余部分保持不变)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 4, in_dim),
            nn.LeakyReLU()
        )
        self.affinity_score = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(64, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.linear = nn.Linear(in_dim * 2, in_dim)
        self.embed_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(in_dim, eps=1e-6)

    def forward(self, spa_seq, tem_seq, spa_mask, tem_mask):
        spatial_output = self.spatial(spa_seq, spa_mask)
        spatial_output_1 = spatial_output[:, 0, :]
        spatial_output_2 = torch.mean(spatial_output[:, 1:, :], dim=1)
        temporal_output = self.temporal(tem_seq, tem_mask)
        temporal_output_1 = temporal_output[:, 0, :]
        temporal_output_2 = torch.mean(temporal_output[:, 1:, :], dim=1)
        output = torch.cat([spatial_output_1, spatial_output_2, temporal_output_1, temporal_output_2], dim=1)
        output = self.mlp(output)

        return output.squeeze(dim=1)

    def getEmbed(self, spa_seq, tem_seq, spa_mask, tem_mask):
        spatial_output = self.spatial(spa_seq, spa_mask)
        spatial_output_1 = spatial_output[:, 0, :]
        spatial_output_2 = torch.mean(spatial_output[:, 1:, :], dim=1)
        temporal_output = self.temporal(tem_seq, tem_mask)
        temporal_output_1 = temporal_output[:, 0, :]
        temporal_output_2 = torch.mean(temporal_output[:, 1:, :], dim=1)
        output = torch.cat([spatial_output_1, spatial_output_2, temporal_output_1, temporal_output_2], dim=1)
        output = self.mlp(output)

        output = self.embed_norm(output)
        return output

    def get_seq_embed(self, seq, mask, view):
        if view == 'spatial':
            output = self.spatial(seq, mask);
        elif view == 'temporal':
            output = self.temporal(seq, mask)
        output = self.norm2(output)
        return output

    def linkPredict(self, src_spa, src_mask1, src_tmp, src_mask2, tgt_spa, tgt_mask1, tgt_tmp, tgt_mask2, fake_spa,
                    fake_mask1, fake_tmp, fake_mask2):
        src_embed = self.getEmbed(src_spa, src_tmp, src_mask1, src_mask2)
        tgt_embed = self.getEmbed(tgt_spa, tgt_tmp, tgt_mask1, tgt_mask2)
        fake_embed = self.getEmbed(fake_spa, fake_tmp, fake_mask1, fake_mask2)

        pos_embed = torch.cat([src_embed, tgt_embed], dim=1)
        neg_embed = torch.cat([src_embed, fake_embed], dim=1)

        pos_embed = self.norm(self.linear(pos_embed))
        neg_embed = self.norm(self.linear(neg_embed))

        pos_score = self.affinity_score(pos_embed).squeeze(dim=1)
        neg_score = self.affinity_score(neg_embed).squeeze(dim=1)

        return pos_score.sigmoid(), neg_score.sigmoid()