"""translated <-> latent translated：MLP AE translated GAT encoder translated。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneLatentAE(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def encode(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [B, M, n_genes] -> z: [B, M, latent_dim]"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, M, latent_dim] -> x_hat: [B, M, n_genes]"""
        return self.decoder(z)


class GeneLatentVAE(nn.Module):
    """translated GeneLatentAE translated VAE：MLP encoder + translated + MLP decoder。"""

    def __init__(self, n_genes: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_with_stats(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [B, M, n_genes] -> z/mu/logvar: [B, M, latent_dim]"""
        h = self.backbone(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z, _, _ = self.encode_with_stats(x, coords=coords, keep_mask=keep_mask)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, M, latent_dim] -> x_hat: [B, M, n_genes]"""
        return self.decoder(z)

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        translated KL(q(z|x)||N(0,1))，translated。
        mu/logvar: [B, M, D]
        keep_mask: [B, M]，True translated token。
        """
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)  # [B, M]
        if keep_mask is None:
            return kl.mean()
        keep = keep_mask.to(dtype=kl.dtype)
        denom = keep.sum().clamp_min(1.0)
        return (kl * keep).sum() / denom


class _DenseKnnGATLayer(nn.Module):
    """translated kNN translated GAT translated（translated token translated）。"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        n_neighbors: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.heads = max(1, int(heads))
        self.n_neighbors = max(1, int(n_neighbors))
        self.dropout = float(dropout)

        self.proj = nn.Linear(self.in_dim, self.out_dim * self.heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(self.heads, self.out_dim))
        self.attn_dst = nn.Parameter(torch.empty(self.heads, self.out_dim))
        self.out_proj = nn.Linear(self.out_dim * self.heads, self.out_dim)
        self.norm = nn.LayerNorm(self.out_dim)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _build_knn_idx(self, coords: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, _ = coords.shape
        k_eff = min(self.n_neighbors + 1, n_tokens)

        dist = torch.cdist(coords.float(), coords.float(), p=2)
        inf = torch.full_like(dist, float("inf"))
        valid_pair = keep_mask.unsqueeze(1) & keep_mask.unsqueeze(2)
        dist = torch.where(valid_pair, dist, inf)
        eye = torch.eye(n_tokens, device=coords.device, dtype=torch.bool).unsqueeze(0)
        dist = torch.where(eye & keep_mask.unsqueeze(1), torch.zeros_like(dist), dist)
        knn_idx = torch.topk(dist, k=k_eff, dim=-1, largest=False).indices
        return knn_idx

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, n_tokens, _ = x.shape
        if keep_mask is None:
            keep_mask = torch.ones(bsz, n_tokens, device=x.device, dtype=torch.bool)
        if coords is None:
            idx = torch.arange(n_tokens, device=x.device)
            knn_idx = idx.view(1, 1, n_tokens).expand(bsz, n_tokens, n_tokens)
        else:
            knn_idx = self._build_knn_idx(coords, keep_mask)

        h = self.proj(x).view(bsz, n_tokens, self.heads, self.out_dim)
        src_score = (h * self.attn_src.view(1, 1, self.heads, self.out_dim)).sum(dim=-1)
        dst_score = (h * self.attn_dst.view(1, 1, self.heads, self.out_dim)).sum(dim=-1)

        batch_offset = torch.arange(bsz, device=x.device).view(bsz, 1, 1) * n_tokens
        flat_idx = knn_idx + batch_offset
        h_flat = h.reshape(bsz * n_tokens, self.heads, self.out_dim)
        dst_flat = dst_score.reshape(bsz * n_tokens, self.heads)
        keep_flat = keep_mask.reshape(bsz * n_tokens)

        h_nei = h_flat[flat_idx]
        dst_nei = dst_flat[flat_idx]
        nei_keep = keep_flat[flat_idx]

        logits = F.leaky_relu(src_score.unsqueeze(2) + dst_nei, negative_slope=0.2)
        logits = logits.masked_fill(~nei_keep.unsqueeze(-1), -1e9)
        query_keep = keep_mask.unsqueeze(-1).unsqueeze(-1)
        logits = torch.where(query_keep, logits, torch.zeros_like(logits))
        attn = F.softmax(logits, dim=2)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        attn = torch.where(query_keep, attn, torch.zeros_like(attn))

        out = (attn.unsqueeze(-1) * h_nei).sum(dim=2)
        out = out.reshape(bsz, n_tokens, self.heads * self.out_dim)
        out = self.out_proj(out)
        out = self.norm(out)
        out = self.act(out)
        out = out * keep_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


class GeneLatentGAT(nn.Module):
    """GAT encoder + MLP decoder：translated FM translated latent/gene translated。"""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dim: int = 256,
        gat_heads: int = 4,
        gat_neighbors: int = 8,
        gat_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.input_proj = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.GELU(),
        )
        self.gat1 = _DenseKnnGATLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            heads=gat_heads,
            n_neighbors=gat_neighbors,
            dropout=gat_dropout,
        )
        self.gat2 = _DenseKnnGATLayer(
            in_dim=hidden_dim,
            out_dim=latent_dim,
            heads=max(1, gat_heads // 2),
            n_neighbors=gat_neighbors,
            dropout=gat_dropout,
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def encode(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.gat1(h, coords=coords, keep_mask=keep_mask)
        z = self.gat2(h, coords=coords, keep_mask=keep_mask)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
