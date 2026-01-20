import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MultHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, bias=True, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim  ##################################################################################
        self.n_heads = n_heads
        self.head_dim = self.out_dim // n_heads
        assert (
                self.head_dim * n_heads == self.out_dim
        )
        self.scaling = self.head_dim ** -0.5

        self.Q_linear = nn.Linear(self.in_dim, self.out_dim, bias=bias)
        self.K_linear = nn.Linear(self.in_dim, self.out_dim, bias=bias)
        self.V_linear = nn.Linear(self.in_dim, self.out_dim, bias=bias)
        self.out_linear = nn.Linear(self.out_dim, self.out_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.K_linear.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.V_linear.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.Q_linear.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.out_linear.weight)
        if self.out_linear.bias is not None:
            nn.init.constant_(self.out_linear.bias, 0.0)

    def forward(self, query, key, value, mask=None):
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len

        Q = self.Q_linear(query)
        K = self.K_linear(key)
        V = self.V_linear(value)
        Q *= self.scaling

        Q = (Q.contiguous().view(tgt_len, -1, self.n_heads, self.head_dim).transpose(1, 2))
        K = (K.contiguous().view(tgt_len, -1, self.n_heads, self.head_dim).transpose(1, 2))
        V = (V.contiguous().view(tgt_len, -1, self.n_heads, self.head_dim).transpose(1, 2))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(tgt_len, -1, self.n_heads * self.head_dim)
        #         print('out1:',out.shape)

        out = self.out_linear(out)
        return out


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


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.4):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


import copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # self.embed = embed
        # self.attn = attn
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.FFN = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim * 2),
            nn.Linear(self.in_dim * 2, self.in_dim),
            nn.Dropout(dropout)
        )
        self.MHA = MultHeadAttention(in_dim, in_dim, n_heads, bias=True)
        self.sublayer = clones(SublayerConnection(in_dim, dropout), 2)
        self.linear = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.MHA(x, x, x, mask))
        #         print('x',x.shape)
        return self.sublayer[1](x, self.FFN)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.in_dim)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)