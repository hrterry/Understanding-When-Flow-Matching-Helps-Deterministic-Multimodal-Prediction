# mmdit_velocity.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def _to_device(module, *tensors):
    device = next(module.parameters()).device
    return [t.to(device) for t in tensors]


# -----------------------------
# Timestep Embedder
# -----------------------------
class TimestepEmbedder(nn.Module):
    """
    Sinusoidal γ(t) + 2-layer MLP -> hidden_dim
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim: int, max_period: int = 10000):
        """
        t: [B] float/long
        returns: [B, dim]
        """
        if t.dtype != torch.float32 and t.dtype != torch.float64:
            t = t.float()
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        t = t.view(-1)  # [B]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb  # [B, hidden]


# -----------------------------
# MLP
# -----------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = approx_gelu()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


# -----------------------------
# Gated Decomposed Attention (GDA) with t in QKV
# -----------------------------
class DecomposedAttention(nn.Module):
    """
    translated；translated：
    img↔img / img↔gene / gene↔img / gene↔gene.

    translated：
    1) concat: translated γ(t) translated token translated QKV；
    2) adaln: translated block translated AdaLN translated/MLPtranslated。
    """
    def __init__(
        self,
        d_model,
        d_time,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.,
        time_modulation: str = "concat",
        use_gated_attention: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.time_modulation = str(time_modulation).lower().strip()
        if self.time_modulation not in ("concat", "adaln"):
            raise ValueError(f"Unsupported time_modulation: {time_modulation}")
        self.use_gated_attention = bool(use_gated_attention)

        in_dim = d_model + d_time if self.time_modulation == "concat" else d_model
        self.qkv = nn.Linear(in_dim, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        # translated（translated，translated；translated per-head translated）
        self.g_img2img   = nn.Parameter(torch.tensor(1.0))
        self.g_img2gene  = nn.Parameter(torch.tensor(1.0))
        self.g_gene2img  = nn.Parameter(torch.tensor(1.0))
        self.g_gene2gene = nn.Parameter(torch.tensor(1.0))

    def forward(self, z, t_emb, num_img_tokens: int):
        """
        z: [B, N, d_model],  N = M + C
        t_emb: [B, d_time]
        num_img_tokens: M
        returns: [B, N, d_model]
        """
        B, N, C = z.shape
        M = num_img_tokens
        if self.time_modulation == "concat":
            d_time = t_emb.shape[-1]
            t_expanded = t_emb[:, None, :].expand(B, N, d_time)
            h = torch.cat([z, t_expanded], dim=-1)  # [B, N, d_model + d_time]
        else:
            h = z

        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, H, N, D]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        if self.use_gated_attention:
            # translated（translated sigmoid translated）
            g_ii = torch.sigmoid(self.g_img2img)
            g_ig = torch.sigmoid(self.g_img2gene)
            g_gi = torch.sigmoid(self.g_gene2img)
            g_gg = torch.sigmoid(self.g_gene2gene)

            # translated
            # translated，translated mask（translated）
            # attn[:, :, :M, :M] *= g_ii
            # translated PyTorch translated in-place translated autograd translated，translated out-of-place：
            top_left     = attn[:, :, :M, :M]      * g_ii
            top_right    = attn[:, :, :M, M:]      * g_ig
            bottom_left  = attn[:, :, M:, :M]      * g_gi
            bottom_right = attn[:, :, M:, M:]      * g_gg
            attn = torch.cat([
                torch.cat([top_left, top_right], dim=-1),
                torch.cat([bottom_left, bottom_right], dim=-1)
            ], dim=-2)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -----------------------------
# MMDiT Block (with ReZero residual scaling)
# -----------------------------
class MMDiTBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_time,
        num_heads,
        mlp_ratio=4.0,
        attn_drop=0.,
        drop=0.,
        time_modulation: str = "concat",
        use_gated_attention: bool = True,
    ):
        super().__init__()
        self.time_modulation = str(time_modulation).lower().strip()
        if self.time_modulation not in ("concat", "adaln"):
            raise ValueError(f"Unsupported time_modulation: {time_modulation}")
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = DecomposedAttention(
            d_model=d_model,
            d_time=d_time,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop,
            time_modulation=self.time_modulation,
            use_gated_attention=use_gated_attention,
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp = Mlp(d_model, int(d_model * mlp_ratio), drop=drop)

        # ReZero residual scaling
        self.alpha_attn = nn.Parameter(torch.tensor(1e-3))
        self.alpha_mlp  = nn.Parameter(torch.tensor(1e-3))
        if self.time_modulation == "adaln":
            self.adaln = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_time, d_model * 6),
            )
            nn.init.zeros_(self.adaln[1].weight)
            nn.init.zeros_(self.adaln[1].bias)
            with torch.no_grad():
                self.adaln[1].bias[:d_model] = 1.0
                self.adaln[1].bias[d_model * 2:d_model * 3] = 1.0
                self.adaln[1].bias[d_model * 3:d_model * 4] = 1.0
                self.adaln[1].bias[d_model * 5:d_model * 6] = 1.0

    def forward(self, z, t_emb, num_img_tokens):
        if self.time_modulation == "adaln":
            g1, b1, g2, g3, b3, g4 = self.adaln(t_emb).chunk(6, dim=-1)
            h = self.norm1(z)
            h = h * g1.unsqueeze(1) + b1.unsqueeze(1)
            attn_out = self.attn(h, None, num_img_tokens)
            attn_out = attn_out * g2.unsqueeze(1)
            z = z + self.alpha_attn * attn_out

            h2 = self.norm2(z)
            h2 = h2 * g3.unsqueeze(1) + b3.unsqueeze(1)
            mlp_out = self.mlp(h2)
            mlp_out = mlp_out * g4.unsqueeze(1)
            z = z + self.alpha_mlp * mlp_out
            return z

        z = z + self.alpha_attn * self.attn(self.norm1(z), t_emb, num_img_tokens)
        z = z + self.alpha_mlp * self.mlp(self.norm2(z))
        return z


# -----------------------------
# MMDiT Transformer (unified sequence, spot->C outputs)
# -----------------------------
class MMDiTTransformer(nn.Module):
    """
    translated:
      img_tokens:  [B, M, d_img_in]
      gene_tokens: [B, C, d_gene_in] translated gene_indices: [B, C]
      t:           [B]
    translated:
      v_pred:      [B, M, out_dim]  (translated out_dim = n_genes = C)
    """
    def __init__(
        self,
        d_img_in: int,
        d_gene_in: int = None,
        n_genes: int = None,
        gene_input_is_indices: bool = False,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        time_hidden_dim: int = None,
        out_dim: int = None,   # translated n_genes
        use_gene_rank_embedding: bool = False,
        gene_rank_bins: int = 32,
        use_noisy_gene_tokens: bool = False,
        patch_out_dim: int = None,
        time_modulation: str = "concat",
        use_gated_attention: bool = True,
    ):
        super().__init__()
        assert (gene_input_is_indices and n_genes is not None) or (not gene_input_is_indices and d_gene_in is not None), \
            "gene_input_is_indices=True translated n_genes；translated d_gene_in"

        self.n_genes = n_genes
        self.gene_input_is_indices = gene_input_is_indices
        self.d_model = d_model
        self.out_dim = out_dim if out_dim is not None else n_genes
        assert self.out_dim is not None, "out_dim/n_genes translated"
        self.use_gene_rank_embedding = bool(use_gene_rank_embedding)
        self.gene_rank_bins = int(gene_rank_bins)
        self.use_noisy_gene_tokens = bool(use_noisy_gene_tokens)
        self.patch_out_dim = int(patch_out_dim) if patch_out_dim is not None else int(d_img_in)
        self.time_modulation = str(time_modulation).lower().strip()
        if self.time_modulation not in ("concat", "adaln"):
            raise ValueError(f"Unsupported time_modulation: {time_modulation}")
        self.use_gated_attention = bool(use_gated_attention)

        # Project to d_model
        self.img_proj = nn.Linear(d_img_in, d_model)
        if gene_input_is_indices:
            embed_dim = d_gene_in if d_gene_in is not None else d_model
            self.gene_embed = nn.Embedding(n_genes, embed_dim)
            self.gene_rank_embed = (
                nn.Embedding(self.gene_rank_bins, embed_dim)
                if self.use_gene_rank_embedding
                else None
            )
            self.gene_proj = nn.Linear(embed_dim, d_model) if embed_dim != d_model else nn.Identity()
        else:
            self.gene_proj = nn.Linear(d_gene_in, d_model)
        # noisy gene/latent token translated：
        # 1) translated use_noisy_gene_tokens translated；
        # 2) vanilla loss translated gene exp token translated。
        self.noisy_gene_proj = nn.Linear(self.out_dim, d_model)

        # time embedding
        self.time_hidden_dim = time_hidden_dim if time_hidden_dim is not None else d_model
        self.time_embedder = TimestepEmbedder(self.time_hidden_dim)

        # transformer blocks
        self.blocks = nn.ModuleList([
            MMDiTBlock(
                d_model=d_model,
                d_time=self.time_hidden_dim,
                num_heads=n_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_dropout,
                drop=dropout,
                time_modulation=self.time_modulation,
                use_gated_attention=self.use_gated_attention,
            ) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)

        # translated spot/image tokens translated: [B, M, d_model] -> [B, M, C]
        self.output_proj_spot = nn.Linear(d_model, self.out_dim)
        # translated：translated noisy gene token translated patch/image translated
        self.output_proj_patch = nn.Linear(d_model, self.patch_out_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        img_tokens,
        gene_tokens=None,
        gene_indices=None,
        gene_rank_bins=None,
        noisy_gene_tokens=None,
        t=None,
        return_aux: bool = False,
        output_from: str = "spot",
    ):
        """
        returns: v_pred [B, M, C]
        """
        assert t is not None, "translated t"
        if self.gene_input_is_indices:
            assert gene_indices is not None
        else:
            assert gene_tokens is not None

        B, M, d_img_in = img_tokens.shape

        img_z = self.img_proj(img_tokens)  # [B, M, d_model]
        if self.gene_input_is_indices:
            gene_emb = self.gene_embed(gene_indices)  # [B, C, emb]
            if self.gene_rank_embed is not None:
                if gene_rank_bins is None:
                    gene_rank_bins = torch.full_like(gene_indices, self.gene_rank_bins // 2)
                gene_emb = gene_emb + self.gene_rank_embed(gene_rank_bins)
            gene_z = self.gene_proj(gene_emb)         # [B, C, d_model]
            C = gene_indices.shape[1]
        else:
            gene_z = self.gene_proj(gene_tokens)      # [B, C, d_model]
            C = gene_tokens.shape[1]

        noisy_len = 0
        if noisy_gene_tokens is not None:
            noisy_z = self.noisy_gene_proj(noisy_gene_tokens)  # [B, M, d_model]
            noisy_len = noisy_z.shape[1]
            z = torch.cat([img_z, gene_z, noisy_z], dim=1)     # [B, M+C+M, d_model]
        else:
            z = torch.cat([img_z, gene_z], dim=1)               # [B, M+C, d_model]
        t_emb = self.time_embedder(t)                 # [B, d_time]

        for blk in self.blocks:
            z = blk(z, t_emb, num_img_tokens=M)

        z = self.final_norm(z)
        spot_hidden = z[:, :M, :]                     # [B, M, d_model]
        noisy_hidden = None
        if noisy_len > 0:
            noisy_hidden = z[:, M + C : M + C + noisy_len, :]

        # translated spot token translated；translated gene exp token（noisy token）translated
        if output_from == "gene_exp":
            if noisy_hidden is None:
                raise ValueError(
                    "output_from='gene_exp' requires noisy_gene_tokens to be provided."
                )
            v_pred = self.output_proj_spot(noisy_hidden)
        else:
            v_pred = self.output_proj_spot(spot_hidden)   # [B, M, C]
        if not return_aux:
            return v_pred

        patch_pred = None
        if noisy_hidden is not None:
            patch_pred = self.output_proj_patch(noisy_hidden)
        return {
            "wsi_to_st": v_pred,
            "st_to_wsi_patch": patch_pred,
        }


# -----------------------------
# High-level Denoiser Wrapper (interface-aligned)
# -----------------------------
class MMDiTDenoiser(nn.Module):
    """
    translated + translated + translated，translated：
      forward(exp, img_features, coords, t_steps) -> [B, M, C]

    - exp:            [B, M, C]   (translated/translated；FM translated)
    - img_features:   [B, M, feature_dim]
    - coords:         [B, M, 2]   (translated，translated)
    - t_steps:        [B]
    """
    def __init__(self, config, device=None):
        super().__init__()
        self.feature_dim = config.feature_dim
        self.hidden_dim  = config.hidden_dim
        self.n_genes     = config.n_genes
        self.use_latent_flow = getattr(config, 'use_latent_flow', False)
        self.flow_dim    = int(getattr(config, 'latent_dim', config.n_genes)) if self.use_latent_flow else config.n_genes
        self.n_layers    = config.n_layers
        self.n_heads     = config.n_heads
        self.dropout     = getattr(config, 'dropout', 0.0)
        self.attn_dropout= getattr(config, 'attn_dropout', 0.0)
        self.use_gene_rank_embedding = bool(
            getattr(config, "use_gene_rank_embedding", False)
        )
        self.gene_rank_bins = int(getattr(config, "gene_rank_bins", 32))
        self.gene_rank_dropout_prob = float(
            getattr(config, "gene_rank_dropout_prob", 0.3)
        )
        self.gene_rank_two_pass_uncond = bool(
            getattr(config, "gene_rank_two_pass_uncond", True)
        )
        self.img_cond_dropout_prob = float(
            getattr(config, "img_cond_dropout_prob", 0.0)
        )
        self.use_noisy_gene_latent_tokens = bool(
            getattr(config, "use_noisy_gene_latent_tokens", False)
        )
        self.sequence_gene_token_source = str(
            getattr(config, "sequence_gene_token_source", "noisy")
        ).lower().strip()
        if self.sequence_gene_token_source not in ("noisy", "clean"):
            raise ValueError(
                f"Unsupported sequence_gene_token_source: {self.sequence_gene_token_source}"
            )
        self.token_training_mode = str(
            getattr(config, "token_training_mode", "wsi_to_st")
        ).lower().strip()
        self.st_to_wsi_loss_weight = float(
            getattr(config, "st_to_wsi_loss_weight", 1.0)
        )
        self.time_modulation = str(
            getattr(config, "time_modulation", "concat")
        ).lower().strip()
        self.use_gated_attention = bool(
            getattr(config, "use_gated_attention", True)
        )
        
        # translated（translated next(self.parameters()).device，translated）
        self.device = device if device is not None else torch.device('cpu')

        # translated：translated backbone translated，translated fourier_proj
        # self.fourier_proj translated（translated）

        # translated
        # translated：translated image_transform translated feature_dim -> feature_dim（translated）
        # translated backbone translated img_proj (feature_dim -> hidden_dim) translated
        # translated feature_dim == hidden_dim，translated，translated
        self.image_transform = nn.Linear(self.feature_dim, self.feature_dim)

        # backbone
        # img_proj translated feature_dim translated hidden_dim
        self.backbone = MMDiTTransformer(
            d_img_in=self.feature_dim,
            d_gene_in=None,                # use indices
            n_genes=self.flow_dim,         # gene translated token translated = flow translated（translated latent slot）
            gene_input_is_indices=True,
            d_model=self.hidden_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            mlp_ratio=4.0,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            time_hidden_dim=self.hidden_dim,
            out_dim=self.flow_dim,         # translated spot translated flow translated velocity
            use_gene_rank_embedding=self.use_gene_rank_embedding,
            gene_rank_bins=self.gene_rank_bins,
            use_noisy_gene_tokens=self.use_noisy_gene_latent_tokens,
            patch_out_dim=self.feature_dim,
            time_modulation=self.time_modulation,
            use_gated_attention=self.use_gated_attention,
        )
        
        # translated
        self.loss_func = nn.MSELoss()
        
        # translated
        self.to(self.device)

    def get_device(self):
        """translated"""
        return self.device

    @torch.no_grad()
    def _make_gene_indices(self, B, C, device=None):
        if device is None:
            device = next(self.parameters()).device  # translated
        # translated
        idx = torch.arange(C, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)  # [B, C]
        return idx

    def _compute_rank_bins_from_exp(self, exp_like: torch.Tensor) -> torch.Tensor:
        """
        exp_like: [B, M, C]，translated spot translated [B, C]，translated rank bin。
        translated：translated gene token translated [B, C]，translated spot translated rank，
        translated batch translated spot translated rank translated。
        """
        B, _, C = exp_like.shape
        score = exp_like.mean(dim=1)  # [B, C]
        ranks = torch.argsort(torch.argsort(score, dim=-1, descending=True), dim=-1)
        bins = (ranks * self.gene_rank_bins) // max(C, 1)
        bins = torch.clamp(bins, 0, self.gene_rank_bins - 1).long()
        return bins

    def inference(
        self,
        noisy_exp,
        img_features,
        coords,
        t_steps,
        predict=False,
        return_aux: bool = False,
        output_from: str = "spot",
        force_noisy_gene_tokens: bool = False,
        clean_sequence_tokens=None,
    ):
        """
        translated velocity translated: [B, M, C]
        
        Args:
            noisy_exp: [B, M, C] - translated（translated）
            img_features: [B, M, feature_dim] - translated
            coords: [B, M, 2] - translated（translated，translated）
            t_steps: [B] - translated
        """
        # translated - translated（translated self.device，translated）
        img_features, t_steps = _to_device(self, img_features, t_steps)
        B, M, C = noisy_exp.shape

        # translated（translated feature_dim != hidden_dim，translated）
        img_tokens = self.image_transform(img_features)  # [B, M, hidden_dim] translated [B, M, feature_dim]
        if self.training and self.img_cond_dropout_prob > 0:
            # translated（translated），translated/translated。
            img_drop_mask = (
                torch.rand(B, device=img_tokens.device) < self.img_cond_dropout_prob
            )[:, None, None]
            img_tokens = torch.where(img_drop_mask, torch.zeros_like(img_tokens), img_tokens)

        # gene translated - translated
        device = next(self.parameters()).device  # translated
        gene_indices = self._make_gene_indices(B, C, device)  # [B, C]
        gene_rank_bins = None
        if self.use_gene_rank_embedding and self.flow_dim == C:
            gene_rank_bins = self._compute_rank_bins_from_exp(noisy_exp)
            if self.training and self.gene_rank_dropout_prob > 0:
                drop_mask = (
                    torch.rand(B, device=device) < self.gene_rank_dropout_prob
                )[:, None]
                default_bin = torch.full_like(gene_rank_bins, self.gene_rank_bins // 2)
                gene_rank_bins = torch.where(drop_mask, default_bin, gene_rank_bins)

        # t embedding translated backbone translated，translated t_steps translated。
        if (
            predict
            and self.use_gene_rank_embedding
            and self.gene_rank_two_pass_uncond
            and self.flow_dim == C
        ):
            # translated：
            # 1) translated rank（translated bin）translated
            # 2) translated noisy_exp + draft_pred translated rank，translated refine translated
            if self.use_noisy_gene_latent_tokens or force_noisy_gene_tokens:
                if (
                    self.sequence_gene_token_source == "clean"
                    and clean_sequence_tokens is not None
                ):
                    noisy_tokens = clean_sequence_tokens
                else:
                    noisy_tokens = noisy_exp
            else:
                noisy_tokens = None
            draft_pred = self.backbone(
                img_tokens=img_tokens,
                gene_indices=gene_indices,
                gene_rank_bins=None,
                noisy_gene_tokens=noisy_tokens,
                t=t_steps,
                output_from=output_from,
            )
            boot_bins = self._compute_rank_bins_from_exp(noisy_exp + draft_pred)
            pred = self.backbone(
                img_tokens=img_tokens,
                gene_indices=gene_indices,
                gene_rank_bins=boot_bins,
                noisy_gene_tokens=noisy_tokens,
                t=t_steps,
                return_aux=return_aux,
                output_from=output_from,
            )
        else:
            if self.use_noisy_gene_latent_tokens or force_noisy_gene_tokens:
                if (
                    self.sequence_gene_token_source == "clean"
                    and clean_sequence_tokens is not None
                ):
                    noisy_tokens = clean_sequence_tokens
                else:
                    noisy_tokens = noisy_exp
            else:
                noisy_tokens = None
            pred = self.backbone(
                img_tokens=img_tokens,       # [B, M, feature_dim]
                gene_indices=gene_indices,   # [B, C]
                gene_rank_bins=gene_rank_bins,
                noisy_gene_tokens=noisy_tokens,
                return_aux=return_aux,
                output_from=output_from,
                t=t_steps                    # [B]
            )                                # [B, M, C]
        if return_aux:
            if isinstance(pred, dict):
                pred["mode"] = self.token_training_mode
                pred["st_to_wsi_weight"] = self.st_to_wsi_loss_weight
            return pred
        return pred  # [B, M, C]

    def forward(self, exp, img_features, coords, t_steps):
        """
        translated Denoiser translated。
        exp: [B, M, C]  (translated B/M/C translated loss translated；FM translated trainer translated)
        translated v_pred: [B, M, C]
        """
        return self.inference(
            noisy_exp=exp,
            img_features=img_features,
            coords=coords,
            t_steps=t_steps,
        )
