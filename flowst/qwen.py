"""Qwen-style gated transformer backbone (capacity and transfer experiments, Appendix E)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import TimestepEmbedder


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, size, inner, dropout):
        super().__init__()
        self.gate_proj = nn.Linear(size, inner, bias=False)
        self.up_proj = nn.Linear(size, inner, bias=False)
        self.down_proj = nn.Linear(inner, size, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class QwenAttention(nn.Module):
    """Scaled dot-product attention with a query-dependent sigmoid gate on each head's output."""

    def __init__(self, size, heads, dropout):
        super().__init__()
        self.heads, self.head_dim, self.dropout = heads, size // heads, dropout
        self.q_proj = nn.Linear(size, heads * (self.head_dim + 1), bias=False)
        self.k_proj = nn.Linear(size, size, bias=False)
        self.v_proj = nn.Linear(size, size, bias=False)
        self.o_proj = nn.Linear(size, size, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.heads, self.head_dim + 1)
        q, gate = q[..., :-1], q[..., -1:]
        k = self.k_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(self.q_norm(q.transpose(1, 2)), self.k_norm(k), v, attn_mask=mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj((out.transpose(1, 2) * torch.sigmoid(gate)).reshape(B, N, -1))


class QwenBlock(nn.Module):
    def __init__(self, size, heads, dropout, attn_dropout):
        super().__init__()
        self.input_layernorm = RMSNorm(size)
        self.self_attn = QwenAttention(size, heads, attn_dropout)
        self.post_attention_layernorm = RMSNorm(size)
        self.mlp = SwiGLU(size, 4 * size, dropout)

    def forward(self, x, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class QwenFlowModel(nn.Module):
    """Same token sequence and interface as GatedFlowModel, with Qwen-style blocks."""

    def __init__(self, feature_dim, dim, hidden_dim=256, n_layers=4, n_heads=8, dropout=0.2,
                 attn_dropout=0.2, img_dropout=0.5, **_):
        super().__init__()
        self.dim, self.img_dropout, self.route = dim, img_dropout, "none"
        self.image_transform = nn.Linear(feature_dim, feature_dim)
        self.img_proj = nn.Linear(feature_dim, hidden_dim)
        self.gene_embed = nn.Embedding(dim, hidden_dim)
        self.noisy_gene_proj = nn.Linear(dim, hidden_dim)
        self.token_type_embed = nn.Embedding(3, hidden_dim)
        self.time_embedder = TimestepEmbedder(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(QwenBlock(hidden_dim, n_heads, dropout, attn_dropout) for _ in range(n_layers))
        self.norm = RMSNorm(hidden_dim)
        self.output_proj_spot = nn.Linear(hidden_dim, dim)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def set_route(self, route):
        self.route = route

    def _mask(self, M, D, N, z):
        if self.route == "none":
            return None
        blocked = torch.zeros(N, N, dtype=torch.bool, device=z.device)
        if self.route in ("no_ig", "no_cross"):
            blocked[:M, M + D:] = True
        if self.route in ("no_gi", "no_cross"):
            blocked[M + D:, :M] = True
        return torch.zeros(N, N, device=z.device, dtype=z.dtype).masked_fill(blocked, torch.finfo(z.dtype).min)[None, None]

    def forward(self, img, xt, t, from_st_tokens=False):
        B, M, _ = img.shape
        h = self.image_transform(img)
        if self.training and self.img_dropout > 0:
            keep = torch.rand(B, device=h.device) >= self.img_dropout
            h = torch.where(keep[:, None, None], h, torch.zeros_like(h))
        type_emb = self.token_type_embed.weight
        z = torch.cat([self.img_proj(h) + type_emb[0],
                       self.gene_embed.weight[None].expand(B, -1, -1) + type_emb[1],
                       self.noisy_gene_proj(xt) + type_emb[2]], dim=1)
        z = z + self.time_proj(self.time_embedder(t))[:, None, :]
        mask = self._mask(M, self.dim, z.shape[1], z)
        for layer in self.layers:
            z = layer(z, mask)
        z = self.norm(z)
        return self.output_proj_spot(z[:, -M:] if from_st_tokens else z[:, :M])
