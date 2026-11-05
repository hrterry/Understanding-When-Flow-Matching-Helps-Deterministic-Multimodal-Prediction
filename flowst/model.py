"""Quadrant-gated self-attention over histology tokens and noisy ST-state tokens."""
import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t, max_period=10000):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.view(-1).float()[..., None] * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class GatedAttention(nn.Module):
    """Self-attention with one learned gate per quadrant, applied to the logits before softmax.

    The first M tokens are histology tokens (I); all remaining tokens form the ST side (G).
    `route` masks a quadrant at inference: no_ig (I cannot attend to G), no_gi, no_gg or no_cross.
    """

    def __init__(self, d_model, d_time, num_heads, attn_drop, proj_drop, gated=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.scale = (d_model // num_heads) ** -0.5
        self.gated = gated
        self.route = "none"
        self.qkv = nn.Linear(d_model + d_time, d_model * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        self.g_img2img = nn.Parameter(torch.tensor(1.0))
        self.g_img2gene = nn.Parameter(torch.tensor(1.0))
        self.g_gene2img = nn.Parameter(torch.tensor(1.0))
        self.g_gene2gene = nn.Parameter(torch.tensor(1.0))

    def forward(self, z, t_emb, M):
        B, N, C = z.shape
        h = torch.cat([z, t_emb[:, None, :].expand(B, N, -1)], dim=-1)
        q, k, v = self.qkv(h).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-2, -1)) * self.scale
        if self.gated:
            a = torch.cat([
                torch.cat([a[..., :M, :M] * torch.sigmoid(self.g_img2img),
                           a[..., :M, M:] * torch.sigmoid(self.g_img2gene)], dim=-1),
                torch.cat([a[..., M:, :M] * torch.sigmoid(self.g_gene2img),
                           a[..., M:, M:] * torch.sigmoid(self.g_gene2gene)], dim=-1),
            ], dim=-2)
        if self.route != "none":
            blocked = torch.finfo(a.dtype).min
            if self.route in ("no_ig", "no_cross"):
                a[..., :M, M:] = blocked
            if self.route in ("no_gi", "no_cross"):
                a[..., M:, :M] = blocked
            if self.route == "no_gg":
                a[..., M:, M:] = blocked
        a = self.attn_drop(a.softmax(dim=-1))
        return self.proj_drop(self.proj((a @ v).transpose(1, 2).reshape(B, N, C)))


class Block(nn.Module):
    def __init__(self, d_model, d_time, num_heads, attn_drop, drop, gated):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = GatedAttention(d_model, d_time, num_heads, attn_drop, drop, gated)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(approximate="tanh"),
            nn.Dropout(drop),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(drop),
        )
        # ReZero residual scaling
        self.alpha_attn = nn.Parameter(torch.tensor(1e-3))
        self.alpha_mlp = nn.Parameter(torch.tensor(1e-3))

    def forward(self, z, t_emb, M):
        z = z + self.alpha_attn * self.attn(self.norm1(z), t_emb, M)
        return z + self.alpha_mlp * self.mlp(self.norm2(z))


class GatedFlowModel(nn.Module):
    """Token sequence [M histology ; D gene/latent-slot embeddings ; M noisy ST states].

    `dim` is the flow dimension: the gene count, or the VAE latent size. Noisy tokens carry no
    spot index, so each histology token reads them as a set: the shared expression context.
    With `xt_tokens=False` the sequence holds histology tokens only (Context-Matched Regression
    when `gene_tokens=False` as well).
    """

    def __init__(self, feature_dim, dim, hidden_dim=128, n_layers=4, n_heads=4, dropout=0.2,
                 attn_dropout=0.2, gated=True, xt_tokens=True, gene_tokens=True, img_dropout=0.5):
        super().__init__()
        self.dim, self.xt_tokens, self.gene_tokens, self.img_dropout = dim, xt_tokens, gene_tokens, img_dropout
        self.image_transform = nn.Linear(feature_dim, feature_dim)
        self.img_proj = nn.Linear(feature_dim, hidden_dim)
        self.gene_embed = nn.Embedding(dim, hidden_dim)
        self.noisy_gene_proj = nn.Linear(dim, hidden_dim)
        self.time_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList(
            Block(hidden_dim, hidden_dim, n_heads, attn_dropout, dropout, gated) for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.output_proj_spot = nn.Linear(hidden_dim, dim)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def set_route(self, route):
        for blk in self.blocks:
            blk.attn.route = route

    def forward(self, img, xt, t, from_st_tokens=False):
        """img [B,M,F], xt [B,M,dim], t [B] -> prediction [B,M,dim]."""
        B, M, _ = img.shape
        h = self.image_transform(img)
        if self.training and self.img_dropout > 0:
            keep = torch.rand(B, device=h.device) >= self.img_dropout
            h = torch.where(keep[:, None, None], h, torch.zeros_like(h))
        tokens = [self.img_proj(h)]
        if self.gene_tokens:
            tokens.append(self.gene_embed.weight[None].expand(B, -1, -1))
        if self.xt_tokens:
            tokens.append(self.noisy_gene_proj(xt))
        z = torch.cat(tokens, dim=1)
        t_emb = self.time_embedder(t)
        for blk in self.blocks:
            z = blk(z, t_emb, M)
        z = self.final_norm(z)
        return self.output_proj_spot(z[:, -M:] if from_st_tokens else z[:, :M])


class GeneVAE(nn.Module):
    """Gene-expression VAE trained jointly with the flow (the flow runs in its latent space)."""

    def __init__(self, n_genes, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(n_genes, hidden_dim), nn.GELU())
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_genes))

    def encode(self, x):
        h = self.backbone(x)
        mu, logvar = self.mu_head(h), self.logvar_head(h)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar), mu, logvar

    def decode(self, z):
        return self.decoder(z)

    @staticmethod
    def kl(mu, logvar):
        return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)).mean()
