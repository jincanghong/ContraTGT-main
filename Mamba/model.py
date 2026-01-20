import torch
import torch.nn as nn
from graph_transformer import *
import math
import torch.nn.functional as F


# === 纯 PyTorch 实现的 Mamba 核心算子 (S6 机制) - 增强稳定性版 ===
class PSSM(nn.Module):
    """
    Pure PyTorch Selective State Space Model with Stability Fixes.
    无需安装 mamba_ssm，直接使用 PyTorch 原生算子。
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, bias=True,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
        )
        self.activation = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.weight.data.zero_()

    def parallel_scan(self, u, delta, A, B, C):
        batch, seq_len, d_in = u.shape
        # [Fix] 限制 delta 的范围，防止 exp 后溢出
        delta = delta.clamp(max=20.0)

        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)

        x = torch.zeros(batch, d_in, self.d_state, device=u.device)
        ys = []
        for t in range(seq_len):
            x = deltaA[:, t] * x + deltaB_u[:, t]
            y = torch.einsum('bdn,bn->bd', x, C[:, t])
            ys.append(y)
        return torch.stack(ys, dim=1)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = self.activation(x)

        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # [Fix] 增加 eps 防止除零或无效值
        dt = F.softplus(self.dt_proj(dt)) + 1e-4
        # A_log 可能会变得非常小或非常大，限制范围
        A = -torch.exp(self.A_log.clamp(max=10.0))

        y = self.parallel_scan(x, dt, A, B, C)
        y = y + x * self.D
        y = y * self.activation(z)
        return self.dropout(self.out_proj(y))


class MambaEncoderLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(MambaEncoderLayer, self).__init__()
        self.mamba = PSSM(d_model=d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return residual + x


class MambaEncoder(nn.Module):
    def __init__(self, d_model, n_layers=2, dropout=0.1):
        super(MambaEncoder, self).__init__()
        self.layers = nn.ModuleList([
            MambaEncoderLayer(d_model=d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class GraphMamba_Model(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=2, dropout=0.1):
        super(GraphMamba_Model, self).__init__()
        self.spatial = MambaEncoder(d_model=in_dim, n_layers=n_layers, dropout=dropout)
        self.temporal = MambaEncoder(d_model=in_dim, n_layers=n_layers, dropout=dropout)

        self.mlp = nn.Sequential(nn.Linear(in_dim * 4, in_dim), nn.LeakyReLU())
        self.affinity_score = nn.Sequential(
            nn.Linear(in_dim, 64), nn.LeakyReLU(), nn.Dropout(p=dropout),
            nn.Linear(64, 10), nn.LeakyReLU(), nn.Linear(10, 1)
        )
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.linear = nn.Linear(in_dim * 2, in_dim)
        self.embed_norm = nn.LayerNorm(in_dim, eps=1e-6)

    def getEmbed(self, spa_seq, tem_seq, spa_mask, tem_mask):
        spatial_output = self.spatial(spa_seq)
        spatial_output_1 = spatial_output[:, 0, :]
        spatial_output_2 = torch.mean(spatial_output[:, 1:, :], dim=1)

        temporal_output = self.temporal(tem_seq)
        temporal_output_1 = temporal_output[:, 0, :]
        temporal_output_2 = torch.mean(temporal_output[:, 1:, :], dim=1)

        output = torch.cat([spatial_output_1, spatial_output_2, temporal_output_1, temporal_output_2], dim=1)
        output = self.mlp(output)
        output = self.embed_norm(output)
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


# === 保留原模型 SpatialTemporal 以兼容引用 ===
class SpatialTemporal(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1, N=2):
        super(SpatialTemporal, self).__init__()
        self.spatial = Encoder(EncoderLayer(in_dim, out_dim, n_heads, dropout), N)
        self.temporal = Encoder(EncoderLayer(in_dim, out_dim, n_heads, dropout), N)
        self.mlp = nn.Sequential(nn.Linear(in_dim * 4, in_dim), nn.LeakyReLU())
        self.affinity_score = nn.Sequential(nn.Linear(in_dim, 64), nn.LeakyReLU(), nn.Dropout(p=dropout),
                                            nn.Linear(64, 10), nn.LeakyReLU(), nn.Linear(10, 1))
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.linear = nn.Linear(in_dim * 2, in_dim)
        self.embed_norm = nn.LayerNorm(in_dim, eps=1e-6)

    def forward(self, spa_seq, tem_seq, spa_mask, tem_mask):
        return self.mlp(torch.cat(
            [self.spatial(spa_seq, spa_mask)[:, 0, :], torch.mean(self.spatial(spa_seq, spa_mask)[:, 1:, :], 1),
             self.temporal(tem_seq, tem_mask)[:, 0, :], torch.mean(self.temporal(tem_seq, tem_mask)[:, 1:, :], 1)],
            dim=1)).squeeze(1)

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
