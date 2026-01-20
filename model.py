import torch
import torch.nn as nn
from graph_transformer import *



class SpatialTemporal(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1, N=2):
        super(SpatialTemporal, self).__init__()
        self.spatial = Encoder(EncoderLayer(in_dim, out_dim, n_heads, dropout), N)
        self.temporal = Encoder(EncoderLayer(in_dim, out_dim, n_heads, dropout), N)
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


class Top_k(nn.Module):
    def __init__(self, in_dim):
        super(Top_k, self).__init__()
        self.W = nn.Parameter(torch.empty(size=(in_dim, in_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1 / math.sqrt(2))
        self.ones = torch.nn.Parameter(torch.ones(1, in_dim).float(), requires_grad=False)

    def forward(self, feature, m, tk, embed):
        attn = self.get_attention(embed, m)  
        mask = attn
        y = torch.zeros_like(mask)
        k = tk

        zero = torch.zeros_like(attn)

        for ep in range(k):
            tmp_y = y.detach()

            tmp_y[tmp_y > 0.9] = 1
            tmp_y[tmp_y <= 0.9] = 0
            w = (1. - tmp_y)
            # softmax_w
            logw = torch.log(w + 1e-12)
            y1 = (mask + logw)

            y1 = y1 - torch.amax(y1, dim=1, keepdim=True)  
            y1 = y1 / (w * 1e-06 + (1 - w) * 1e-02)

            y1 = torch.exp(y1) / (torch.sum(torch.exp(y1), dim=1, keepdim=True) + 1e-12)

            y = y + y1 * w

        mask = y.unsqueeze(dim=-1)
        mask = torch.matmul(mask, self.ones)  

        feature = feature * mask

        y = torch.where(y < 0.1, zero, y)

        return feature, y  

    def get_attention(self, feature, mask):
        #         mask = torch.tensor(mask)
        feature = torch.matmul(feature, self.W)

        attention = torch.matmul(feature, feature.transpose(-2, -1)) / math.sqrt(
            feature.shape[2])

        zero_vec = -9e15 * torch.ones_like(mask)
        attn = torch.where(mask > 0, attention[:, 0, :], zero_vec)
        attn = F.softmax(attn, dim=1)
        return attn
