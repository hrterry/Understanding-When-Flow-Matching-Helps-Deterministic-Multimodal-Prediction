"""
Flow translated：noisy_exp + translated embedding translated + translated t，translated mmDiT（mmdit/mmdit_pytorch.py）translated。
"""
import os
import sys

import torch
import torch.nn as nn

# translated，translated import mmdit
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from mmdit.mmdit_pytorch import MMDiT

from model.vpredictor import TimestepEmbedder


def _to_device(module, *tensors):
    device = next(module.parameters()).device
    return [t.to(device) for t in tensors]


class DitFlowDenoiser(nn.Module):
    """
    translated：forward(exp, img_features, coords, t_steps) -> v_pred [B, M, C]
    - noisy_exp: [B, M, C]
    - img_features: [B, M, feature_dim]，translated patch embedding translated
    - t_steps: [B]，translated，translated mmDiT
    """

    def __init__(self, config, device=None):
        super().__init__()
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.n_genes = config.n_genes
        self.use_latent_flow = getattr(config, "use_latent_flow", False)
        self.flow_dim = int(getattr(config, "latent_dim", config.n_genes)) if self.use_latent_flow else config.n_genes
        self.n_layers = getattr(config, "n_layers", 1)
        self.n_heads = config.n_heads
        self.dropout = getattr(config, "dropout", 0.0)
        self.attn_dropout = getattr(config, "attn_dropout", 0.0)
        self.device = device if device is not None else torch.device("cpu")

        self.time_hidden_dim = self.hidden_dim
        self.time_embedder = TimestepEmbedder(self.time_hidden_dim)

        self.proj_exp = nn.Linear(self.flow_dim, self.hidden_dim)
        self.proj_cond = nn.Linear(self.feature_dim, self.hidden_dim)

        self.backbone = MMDiT(
            depth=max(1, int(self.n_layers)),
            dim_text=self.hidden_dim,
            dim_image=self.hidden_dim,
            dim_cond=self.time_hidden_dim,
            dim_head=max(1, self.hidden_dim // self.n_heads),
            heads=self.n_heads,
            ff_mult=4.0,
            attn_drop=self.attn_dropout,
            drop=self.dropout,
            final_norm=True,
        )

        self.out_proj = nn.Linear(self.hidden_dim, self.flow_dim)
        self.loss_func = nn.MSELoss()
        self.apply(self._init_weights)
        self.to(self.device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def inference(self, noisy_exp, img_features, coords, t_steps, predict=False):
        img_features, t_steps = _to_device(self, img_features, t_steps)
        noisy_exp = noisy_exp.to(img_features.device)
        coords = coords.to(img_features.device)

        B, M, C = noisy_exp.shape
        t_emb = self.time_embedder(t_steps)

        text_tokens = self.proj_exp(noisy_exp)
        image_tokens = self.proj_cond(img_features)

        pad_mask = img_features.sum(dim=-1) != 0

        text_tokens, _ = self.backbone(
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            time_cond=t_emb,
            text_mask=pad_mask,
        )

        v_pred = self.out_proj(text_tokens)
        v_pred = v_pred.masked_fill(~pad_mask.unsqueeze(-1), 0.0)
        return v_pred

    def forward(self, exp, img_features, coords, t_steps):
        return self.inference(
            noisy_exp=exp,
            img_features=img_features,
            coords=coords,
            t_steps=t_steps,
        )
